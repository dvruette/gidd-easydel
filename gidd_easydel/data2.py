from abc import ABC, abstractmethod
import os
import dataclasses
import typing as tp
import pickle
from concurrent.futures import ProcessPoolExecutor, Future

import numpy as np
import jax.numpy as jnp
import dask.dataframe as dd
from eformer.paths import ePath, ePathLike
from eformer.loggings import get_logger


logger = get_logger(__name__)



def _child_init():
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"


def _load_partition(path, seed=0, storage_options=None):
    ddf = dd.read_parquet(
        path,
        columns=["tokens"],
        engine="pyarrow",
        split_row_groups=False,
        storage_options=storage_options,
    )
    vals = ddf["tokens"].persist().compute().values
    rng = np.random.default_rng(seed)
    rng.shuffle(vals)
    return vals


S = tp.TypeVar("S")

class StatefulSampler(ABC, tp.Generic[S]):
    state: tp.Optional[S] = None
    state_save_dir: tp.Optional[ePathLike] = None
    rng: np.random.Generator
    seed: int

    @abstractmethod
    def init_state(self):
        pass

    @abstractmethod
    def load_state(self, state: S):
        pass

    def load_closest_state(self, target_i: int):
        if self.state_save_dir is None:
            raise ValueError("No state_save_dir provided to load state from.")
        state_path = ePath(self.state_save_dir)

        closest_state_file = None
        closest_step = -1
        if state_path.exists():
            state_files: list[ePathLike] = sorted(state_path.glob("step_*.pkl"), key=lambda p: p.stem())
            for sf in state_files:
                step_str = sf.stem().split("_")[1]
                step = int(step_str)
                if step <= target_i and step > closest_step:
                    closest_step = step
                    closest_state_file = sf
        if closest_state_file is None:
            logger.warning(f"No suitable state found in {self.state_save_dir} to load from for target step {target_i}. Initializing new state.")
            self.init_state()
        else:
            logger.info(f"Loading state from {closest_state_file} for target iteration {target_i}.")
            state = self.load_state_from_disk(closest_state_file)
            self.load_state(state)

    def save_state(self, path: str | ePathLike | None = None):
        if self.state is None:
            raise ValueError("No state to save.")

        if path is None:
            if self.state_save_dir is None:
                return
            path = self.state_save_dir / f"step_{self.state.i}.pkl"
        path = ePath(path)
        self.state.rng_state = self.rng.bit_generator.state
        path.write_bytes(pickle.dumps(self.state))
        logger.info(f"State saved to {path}")

    @classmethod
    def load_state_from_disk(cls, path: str | ePathLike) -> S:
        path = ePath(path)
        data = path.read_bytes()
        state = pickle.loads(data)
        return state

    def __getitem__(self, i):
        if self.state is None:
            self.load_closest_state(i)

        if i < self.state.i:
            raise ValueError(f"Cannot go back to previous items in the iterator (requested {i}, current {self.state.i})")

        if self.state.i < i:
            logger.info(f"Advancing from iteration {self.state.i} to {i}")

        while self.state.i < i:
            self.next()
        return self.next()
    
    def iterate(self, start: int = 0, end: tp.Optional[int] = None):
        i = start
        while end is None or i < end:
            yield self[i]
            i += 1

    @abstractmethod
    def next(self):
        pass


@dataclasses.dataclass
class BucketRowSamplerState:
    rng_state: dict
    partitions: dict
    buffers: dict
    i: int = 0
    seed: int = 0


class BucketRowSampler(StatefulSampler[BucketRowSamplerState]):
    def __init__(
        self,
        bucket_files: list[list[str]],
        storage_options=None,
        max_workers: tp.Literal["auto"] | int = "auto",
        preload_factor: int = 1,
        state: BucketRowSamplerState | None = None,
        state_save_interval: int = -1,
        state_save_dir: str | ePathLike | None = None,
        seed: int = 0,
    ):
        if preload_factor < 1:
            raise ValueError("preload_factor must be at least 1")
        if len(bucket_files) == 0:
            raise ValueError("bucket_files must contain at least one bucket")
        if any(len(b) == 0 for b in bucket_files):
            raise ValueError("each bucket in bucket_files must contain at least one file")

        self.bucket_files = bucket_files
        self.storage_options = storage_options
        self.num_buckets = len(bucket_files)
        self.state_save_interval = state_save_interval
        self.state_save_dir = state_save_dir
        self.preload_factor = preload_factor
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.futures: dict[str, Future] = {}

        if max_workers == "auto":
            max_workers = min(self.preload_factor * self.num_buckets, (os.cpu_count() or 1))
        self.executor = ProcessPoolExecutor(max_workers=max_workers, initializer=_child_init)

        if state is None and self.state_save_dir is not None:
            self.state = None
        elif state is None:
            self.init_state()
        else:
            self.load_state(state)

    def init_state(self):
        self.partitions = {i: 0 for i in range(self.num_buckets)}
        self.preload_partitions = self.partitions.copy()
        self.buffers = {i: {"file": self._next_file(i), "idx": 0} for i in range(self.num_buckets)}
        self.state = BucketRowSamplerState(
            i=0,
            rng_state=self.rng.bit_generator.state,
            partitions=self.partitions.copy(),
            buffers={k: v.copy() for k, v in self.buffers.items()},
            seed=self.seed,
        )
        self._preload_partitions()

    def load_state(self, state: BucketRowSamplerState):
        self.state = state
        self.rng.bit_generator.state = self.state.rng_state
        self.partitions = self.state.partitions.copy()
        self.preload_partitions = self.state.partitions.copy()
        self.buffers = {k: v.copy() for k, v in self.state.buffers.items()}
        self._preload_partitions()

    def _preload_partitions(self):
        assert self.state is not None, "State must be initialized."
        self.futures = {}
        for _ in range(self.preload_factor):
            for bucket_idx in self.preload_partitions.keys():
                self._preload_next_partition(bucket_idx)

    def __del__(self):
        if hasattr(self, "executor") and isinstance(self.executor, ProcessPoolExecutor):
            self.executor.shutdown(wait=True)

    def _random_bucket(self):
        assert self.state is not None, "State must be initialized."
        assert len(self.buffers) > 0, "No buckets to sample from."
        x = self.rng.choice(list(self.buffers.keys()))
        self.state.rng_state = self.rng.bit_generator.state
        return x

    def _get_seed(self, bucket_idx, partition_idx):
        assert self.state is not None, "State must be initialized."
        seed = (2**16 * bucket_idx + partition_idx) % (2**32)
        return seed + self.state.seed

    def _next_file(self, bucket_idx):
        assert self.partitions is not None, "Partitions must be initialized."
        partition_idx = self.partitions[bucket_idx]
        bucket_len = len(self.bucket_files[bucket_idx])
        file = self.bucket_files[bucket_idx][partition_idx % bucket_len]
        return file

    def _preload_next_partition(self, bucket_idx):
        assert self.state is not None, "State must be initialized."
        partition_idx = self.preload_partitions[bucket_idx]
        bucket_len = len(self.bucket_files[bucket_idx])
        file = self.bucket_files[bucket_idx][partition_idx % bucket_len]
        seed = self._get_seed(bucket_idx, partition_idx)
        self.preload_partitions[bucket_idx] += 1
        self.futures[file] = self.executor.submit(_load_partition, file, seed, self.storage_options)
        return file

    def next(self):
        bucket_idx = self._random_bucket()
        buffer = self.buffers.get(bucket_idx)
        assert buffer is not None, "Buffer should not be None"

        values = buffer.get("values")
        if values is None:
            buffer["values"] = self.futures.pop(buffer["file"]).result()
            self._preload_next_partition(bucket_idx)

        if buffer["idx"] >= len(buffer["values"]):
            self.partitions[bucket_idx] += 1

            next_file = self._next_file(bucket_idx)
            self.buffers[bucket_idx] = {
                "file": next_file,
                "idx": 0,
            }
            self.state.buffers[bucket_idx] = self.buffers[bucket_idx].copy()
            self.state.partitions = self.partitions.copy()
            return self.next()
        buffer["idx"] += 1
        self.state.buffers[bucket_idx]["idx"] = buffer["idx"]
        assert "values" not in self.state.buffers[bucket_idx], f"State buffer should not contain values ({bucket_idx=}, {self.state.i=})"
        self.state.i += 1

        if (
            self.state_save_interval > 0
            and self.state_save_dir is not None
            and (self.state.i % self.state_save_interval == 0)
        ):
            self.save_state()
            logger.info(f"State saved after {self.state.i} iterations")
        return {"tokens": buffer["values"][buffer["idx"] - 1]}


@dataclasses.dataclass
class PackedRowSamplerState:
    rng_state: dict
    buffer: np.ndarray
    shuffle_buffer: list[jnp.ndarray]
    i: int = 0
    inner_i: int = 0
    seed: int = 0


class PackedRowSampler(StatefulSampler[PackedRowSamplerState]):
    def __init__(
        self,
        dataset,
        tokens_field: str = "tokens",
        seq_length: int = 1024,
        eos_token_id: int = 0,
        shuffle: bool = True,
        batch_size: int = 512,
        shuffle_buffer_batch_factor: int = 16,
        append_eos_token: bool = True,
        state: PackedRowSamplerState | None = None,
        state_save_interval: int = 512 * 1000,
        state_save_dir: str | ePathLike | None = None,
        seed: int = 0,
    ):
        """
        Creates a sampler that yields constant length chunks of tokens from a stream of text files.

        Args:
            dataset: Dataset with text files.
            tokens_field: Name of the field in the dataset that contains the text.
            seq_length: Length of token sequences to return.
            eos_token_id: Id of the end of sequence token if the passed processing_class does not have an EOS token.
            shuffle: Shuffle the examples before they are returned.
            batch_size: Batch size for the dataset. Used to compute the shuffle buffer size.
            shuffle_buffer_batch_factor: Factor to compute the shuffle buffer size. The shuffle buffer size is
                `batch_size * shuffle_buffer_batch_factor`.
            append_eos_token: If true, appends eos_token_id at the end of each sample being packed.

        """

        self.dataset = dataset
        self.tokens_field = tokens_field
        self.seq_length = seq_length
        self.shuffle = shuffle
        self.append_eos_token = append_eos_token
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        self.shuffle_buffer_size = batch_size * shuffle_buffer_batch_factor
        self.eos_token = np.array([eos_token_id], dtype=np.int32)

        self.state_save_interval = state_save_interval
        self.state_save_dir = state_save_dir

        if state is None and self.state_save_dir is not None:
            self.state = None
        elif state is None:
            self.init_state()
        else:
            self.load_state(state)

    def init_state(self):
        self.buffer = np.array([], dtype=np.int32)
        self.shuffle_buffer = []
        self.state = PackedRowSamplerState(
            i=0,
            inner_i=0,
            rng_state=self.rng.bit_generator.state,
            seed=self.seed,
            buffer=self.buffer,
            shuffle_buffer=self.shuffle_buffer,
        )

    def load_state(self, state: PackedRowSamplerState):
        self.state = state
        self.rng.bit_generator.state = self.state.rng_state
        self.buffer = self.state.buffer
        self.shuffle_buffer = self.state.shuffle_buffer

    def next(self):
        while True:
            while len(self.buffer) < self.seq_length:
                tokens = self.dataset[self.state.inner_i][self.tokens_field]
                self.state.inner_i += 1

                if not isinstance(tokens, np.ndarray):
                    assert isinstance(tokens, (jnp.ndarray, list)), (
                        f"Expected tokens to be a list or np.ndarray or jnp.ndarray, got {type(tokens)}"
                    )
                    tokens = np.array(tokens, dtype=np.int32)
                else:
                    tokens = tokens.astype(np.int32)

                # append EOS token
                self.buffer = np.concatenate([self.buffer, tokens], axis=0)
                if self.append_eos_token and len(self.buffer) % self.seq_length != 0:
                    self.buffer = np.concatenate([self.buffer, self.eos_token], axis=0)

            # Pop the first seq_length tokens to form a complete example
            example = {"input_ids": jnp.array(self.buffer[:self.seq_length])}
            self.buffer = self.buffer[self.seq_length:]
            if self.shuffle:
                if len(self.shuffle_buffer) < self.shuffle_buffer_size:
                    self.shuffle_buffer.append(example)
                    continue
                else:
                    idx = self.rng.integers(self.shuffle_buffer_size)
                    output = self.shuffle_buffer[idx]
                    self.shuffle_buffer[idx] = example
                    break
            else:
                output = example
                break

        self.state.buffer = self.buffer
        self.state.i += 1

        if (
            self.state_save_interval > 0
            and self.state_save_dir is not None
            and (self.state.i % self.state_save_interval == 0)
        ):
            self.dataset.save_state()
            self.save_state()
            logger.info(f"State saved after {self.state.i} iterations")
        return output


def generate_packed_samples(
    bucket_files: list[list[str]],
    seed: int = 0,
    storage_options=None,
    preload_factor: int = 1,
    max_workers: tp.Literal["auto"] | int = "auto",
    tokens_field: str = "tokens",
    max_sequence_length: int = 1024,
    batch_size: int = 512,
    eos_token_id: int = 0,
    append_eos_token: bool = True,
    skip_first_n_batches: int = 0,
    state_save_interval: int = 60,
    state_save_dir: str | ePathLike | None = None,
):
    if state_save_dir is not None:
        save_path = ePath(state_save_dir)
        bucket_row_state_path = save_path / "bucket_row_sampler"
        packed_row_state_path = save_path / "packed_row_sampler"

    sampler = BucketRowSampler(
        bucket_files=bucket_files,
        storage_options=storage_options,
        max_workers=max_workers,
        preload_factor=preload_factor,
        state_save_interval=-1,
        state_save_dir=bucket_row_state_path,
        seed=seed,
    )

    packed_sampler = PackedRowSampler(
        dataset=sampler,
        tokens_field=tokens_field,
        seq_length=max_sequence_length,
        eos_token_id=eos_token_id,
        shuffle=True,
        batch_size=batch_size,
        append_eos_token=append_eos_token,
        state_save_interval=state_save_interval,
        state_save_dir=packed_row_state_path,
        seed=seed,
    )

    for x in packed_sampler.iterate(start=skip_first_n_batches * batch_size):
        yield x
