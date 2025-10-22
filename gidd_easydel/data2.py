import os
import dataclasses
import time
import typing as tp
import warnings
import pickle
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import jax.numpy as jnp
import dask.dataframe as dd
from eformer.paths import ePath, ePathLike
from eformer.loggings import get_logger


logger = get_logger(__name__)


@dataclasses.dataclass
class BucketRowSamplerState:
    rng_state: dict
    partitions: dict
    buffers: dict
    i: int = 0
    seed: int = 0


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

class BucketRowSampler:
    def __init__(
        self,
        bucket_files: list[list[str]],
        storage_options=None,
        max_workers: tp.Literal["auto"] | int = "auto",
        preload_factor: int = 1,
        state: BucketRowSamplerState | None = None,
        state_save_interval: int = 60,  # save interval in seconds
        state_save_path: str | ePathLike | None = None,
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
        self.state_save_path = state_save_path
        self.last_saved_time = None

        if max_workers == "auto":
            max_workers = min(preload_factor * self.num_buckets, (os.cpu_count() or 1))

        if state is None and state_save_path is not None:
            state_path = ePath(state_save_path)
            if state_path.exists():
                state = self.load_state(state_path)
                logger.info(f"Loaded state from {state_save_path} at iteration {state.i}")

        if state is None:
            self.rng = np.random.default_rng(seed)
            self.partitions = {i: 0 for i in range(self.num_buckets)}
            self.preload_partitions = self.partitions.copy()
            self.buffers = {i: {"file": self._next_file(i), "idx": 0} for i in range(self.num_buckets)}
            self.state = BucketRowSamplerState(
                i=0,
                rng_state=self.rng.bit_generator.state,
                partitions=self.partitions.copy(),
                buffers={k: v.copy() for k, v in self.buffers.items()},
                seed=seed,
            )
        else:
            self.state = state
            self.rng = np.random.default_rng()
            self.rng.bit_generator.state = self.state.rng_state
            self.partitions = self.state.partitions.copy()
            self.preload_partitions = self.state.partitions.copy()
            self.buffers = {k: v.copy() for k, v in self.state.buffers.items()}

        self.executor = ProcessPoolExecutor(max_workers=max_workers, initializer=_child_init)
        self.futures = {}
        for _ in range(preload_factor):
            for bucket_idx in self.preload_partitions.keys():
                self._preload_next_partition(bucket_idx)

    def save_state(self, path: str | ePathLike):
        path = ePath(path)
        self.state.rng_state = self.rng.bit_generator.state
        path.write_bytes(pickle.dumps(self.state))

    @classmethod
    def load_state(cls, path: str | ePathLike) -> BucketRowSamplerState:
        path = ePath(path)
        data = path.read_bytes()
        state = pickle.loads(data)
        return state

    def __del__(self):
        if hasattr(self, "executor") and isinstance(self.executor, ProcessPoolExecutor):
            self.executor.shutdown(wait=True)

    def _random_bucket(self):
        assert len(self.buffers) > 0, "No buckets to sample from."
        x = self.rng.choice(list(self.buffers.keys()))
        self.state.rng_state = self.rng.bit_generator.state
        return x

    def _get_seed(self, bucket_idx, partition_idx):
        seed = (2**16 * bucket_idx + partition_idx) % (2**32)
        return seed + self.state.seed

    def _next_file(self, bucket_idx):
        partition_idx = self.partitions[bucket_idx]
        bucket_len = len(self.bucket_files[bucket_idx])
        file = self.bucket_files[bucket_idx][partition_idx % bucket_len]
        return file

    def _preload_next_partition(self, bucket_idx):
        partition_idx = self.preload_partitions[bucket_idx]
        bucket_len = len(self.bucket_files[bucket_idx])
        file = self.bucket_files[bucket_idx][partition_idx % bucket_len]
        seed = self._get_seed(bucket_idx, partition_idx)
        self.preload_partitions[bucket_idx] += 1
        self.futures[file] = self.executor.submit(_load_partition, file, seed, self.storage_options)
        return file

    def __getitem__(self, i):
        if i < self.state.i:
            raise ValueError("Cannot go back to previous items in the iterator")

        if self.state.i < i:
            logger.info(f"Advancing from iteration {self.state.i} to {i}")

        while self.state.i < i:
            self.next()
        return self.next()
    
    def __iter__(self):
        i = 0
        while True:
            yield self[i]
            i += 1

    def next(self):
        bucket_idx = self._random_bucket()
        # next_file = self._next_file(bucket_idx)
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

        if self.last_saved_time is None:
            self.last_saved_time = time.time()

        if (
            self.state_save_path is not None
            and (time.time() - self.last_saved_time) > self.state_save_interval
        ):
            self.save_state(self.state_save_path)
            self.last_saved_time = time.time()
            logger.info(f"State saved after {self.state.i} iterations")
        return {"tokens": buffer["values"][buffer["idx"] - 1]}

