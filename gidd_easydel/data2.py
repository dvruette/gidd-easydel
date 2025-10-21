import os
import dataclasses
import typing as tp
import warnings
import random
import pickle
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import jax.numpy as jnp
import dask.dataframe as dd
from eformer.paths import ePath, ePathLike


@dataclasses.dataclass
class BucketRowGeneratorState:
    i: int = 0
    rng_state: dict
    partitions: dict
    buffers: dict


def _child_init():
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"


class BucketRowGenerator:
    def __init__(
        self,
        bucket_files: list[list[str]],
        storage_options=None,
        state: BucketRowGeneratorState | None = None,
        max_workers: tp.Literal["auto"] | int = "auto",
        preload_factor: int = 1,
        state_save_interval: int = 1000,
        state_save_path: str | ePathLike | None = None,
        seed: int = 0,
    ):
        if preload_factor < 1:
            raise ValueError("preload_factor must be at least 1")

        self.bucket_files = bucket_files
        self.storage_options = storage_options
        self.num_buckets = len(bucket_files)
        self.state_save_interval = state_save_interval
        self.state_save_path = state_save_path

        if max_workers == "auto":
            max_workers = min(preload_factor * self.num_buckets, (os.cpu_count() or 1))

        if state is None:
            self.rng = np.random.default_rng(seed)
            self.preload_partitions = {i: 0 for i in range(self.num_buckets)}
            self.partitions = {i: 0 for i in range(self.num_buckets)}
            self.buffers = {i: {"file": self._next_file(i), "idx": 0} for i in range(self.num_buckets)}
            self.state = BucketRowGeneratorState(
                i=0,
                rng_state=self.rng.get_state(),
                partitions=self.partitions.copy(),
                buffers=self.buffers.copy(),
            )
        else:
            self.state = state
            self.rng = np.random.default_rng()
            self.rng.set_state(self.state.rng_state)
            self.preload_partitions = self.state.partitions.copy()
            self.partitions = self.state.partitions.copy()
            self.buffers = {k: v.copy() for k, v in self.state.buffers.items()}

        self.executor = ProcessPoolExecutor(max_workers=max_workers, initializer=_child_init)
        self.futures = {}
        for _ in range(preload_factor):
            for bucket_idx in self.preload_partitions.keys():
                self._preload_next_partition(bucket_idx)

    def save_state(self, path: str | ePathLike):
        path = ePath(path)
        self.state.rng_state = self.rng.get_state()
        path.write_bytes(pickle.dumps(self.state))

    @classmethod
    def load_state(cls, path: str | ePathLike) -> BucketRowGeneratorState:
        path = ePath(path)
        data = path.read_bytes()
        state = pickle.loads(data)
        return state

    def __del__(self):
        self.executor.shutdown(wait=True)

    def _random_bucket(self):
        assert len(self.buffers) > 0, "No buckets to sample from."
        x = self.rng.choice(list(self.buffers.keys()))
        self.state.rng_state = self.rng.get_state()
        return x

    def _random_seed(self):
        seed = self.rng.integers(1e6)
        self.state.rng_state = self.rng.get_state()
        return seed

    def _load_partition(self, path, seed=0):
        ddf = dd.read_parquet(
            path,
            columns=["tokens"],
            engine="pyarrow",
            split_row_groups=False,
            storage_options=self.storage_options,
        )
        vals = ddf["tokens"].persist().compute().values
        rng = np.random.default_rng(seed)
        rng.shuffle(vals)
        return vals

    def _next_file(self, bucket_idx):
        partition_idx = self.partitions[bucket_idx]
        bucket_len = len(self.bucket_files[bucket_idx])
        file = self.bucket_files[bucket_idx][partition_idx % bucket_len]
        self.partitions[bucket_idx] += 1
        return file

    def _preload_next_partition(self, bucket_idx):
        partition_idx = self.preload_partitions[bucket_idx]
        bucket_len = len(self.bucket_files[bucket_idx])
        file = self.bucket_files[bucket_idx][partition_idx % bucket_len]
        self.preload_partitions[bucket_idx] += 1
        self.futures[file] = self.executor.submit(self._load_partition, file, self._random_seed())
        return file

    def __getitem__(self, i):
        if i < self.state.i:
            raise ValueError("Cannot go back to previous items in the iterator")

        while self.state.i < i:
            self.next()
        return self.next()

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
            next_file = self._next_file(bucket_idx)
            self.buffers[bucket_idx] = {
                "file": next_file,
                "idx": 0,
            }
            self.state.buffers[bucket_idx] = self.buffers[bucket_idx].copy()
            return self.next()
        buffer["idx"] += 1
        self.state.buffers[bucket_idx]["idx"] = buffer["idx"]
        self.state.i += 1
        if (
            self.state_save_path is not None
            and (self.state.i + 1) % self.state_save_interval == 0
        ):
            self.save_state(self.state_save_path)
        return {"tokens": buffer["values"][buffer["idx"] - 1]}


