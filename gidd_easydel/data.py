import os
import random
import warnings
import typing as tp
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import jax.numpy as jnp
import dask.dataframe as dd


def _child_init():
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"


def _load_partition(path, storage_options=None, seed=0):
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


def _loop_forever(it):
    while True:
        for x in it:
            yield x


def generate_rows_from_buckets(bucket_files, storage_options=None, seed=0, max_workers="auto", preload_factor=1):
    partition_iters = {
        i: _loop_forever(ps)
        for i, ps in enumerate(bucket_files)
    }
    num_buckets = len(bucket_files)

    if max_workers == "auto":
        max_workers = min(preload_factor * num_buckets, (os.cpu_count() or 1))

    rng = np.random.default_rng(seed)

    def _submit_next_partition(i):
        futures[i].append(executor.submit(_load_partition, next(partition_iters[i]), storage_options, rng.integers(1e6)))

    with ProcessPoolExecutor(max_workers=max_workers, initializer=_child_init) as executor:
        futures = {i: [] for i in range(num_buckets)}
        buffers = {}
        for _ in range(preload_factor):
            for i in partition_iters.keys():
                _submit_next_partition(i)

        next_bucket = rng.choice(list(futures.keys()))
        while futures:
            buffer = buffers.get(next_bucket)
            if buffer is None:
                values = futures[next_bucket].pop(0).result()
                buffer = {
                    "values": values,
                    "idx": 0,
                }
                buffers[next_bucket] = buffer
                _submit_next_partition(next_bucket)
            if buffer["idx"] >= len(buffer["values"]):
                del buffers[next_bucket]
                continue
            yield {"tokens": buffer["values"][buffer["idx"]]}
            buffer["idx"] += 1
            next_bucket = rng.choice(list(futures.keys()))



def create_constant_length_dataset(
    dataset,
    tokens_field: str | None = None,
    infinite: bool = False,
    seq_length: int = 1024,
    eos_token_id: int = 0,
    shuffle: bool = True,
    batch_size: int = 512,
    shuffle_buffer_batch_factor: int = 16,
    append_concat_token: bool = True,
) -> tp.Callable[[], tp.Iterator[dict[str, jnp.ndarray]]]:
    """
    Creates a generator function that yields constant length chunks of tokens from a stream of text files.

    Args:
        dataset: Dataset with text files.
        tokens_field: Name of the field in the dataset that contains the text.
        infinite: If True the iterator is reset after dataset reaches end else stops.
        seq_length: Length of token sequences to return.
        eos_token_id: Id of the end of sequence token if the passed processing_class does not have an EOS token.
        shuffle: Shuffle the examples before they are returned.
        batch_size: Batch size for the dataset. Used to compute the shuffle buffer size.
        shuffle_buffer_batch_factor: Factor to compute the shuffle buffer size. The shuffle buffer size is
            `batch_size * shuffle_buffer_batch_factor`.
        append_concat_token: If true, appends eos_token_id at the end of each sample being packed.

    Returns:
        A generator function that yields fixed-length token arrays as jnp.ndarray
    """

    def constant_length_generator() -> tp.Iterator[jnp.ndarray]:
        iterator = iter(dataset)

        buffer: np.ndarray = np.array([], dtype=np.int32)
        shuffle_buffer: list[jnp.ndarray] = []
        shuffle_buffer_size = batch_size * shuffle_buffer_batch_factor
        eos_token = np.array([eos_token_id], dtype=np.int32)

        while True:
            try:
                tokens = next(iterator)[tokens_field]
                if not isinstance(tokens, np.ndarray):
                    assert isinstance(tokens, (jnp.ndarray, list)), (
                        f"Expected tokens to be a list or np.ndarray or jnp.ndarray, got {type(tokens)}"
                    )
                    tokens = np.array(tokens, dtype=np.int32)
                else:
                    tokens = tokens.astype(np.int32)

                # append EOS token
                buffer = np.concatenate([buffer, tokens], axis=0)
                if append_concat_token and len(buffer) % seq_length != 0:
                    buffer = np.concatenate([buffer, eos_token], axis=0)

                while len(buffer) >= seq_length:
                    # Pop the first seq_length tokens to form a complete example
                    example = {"input_ids": jnp.array(buffer[:seq_length])}
                    buffer = buffer[seq_length:]
                    if shuffle:
                        if len(shuffle_buffer) < shuffle_buffer_size:
                            shuffle_buffer.append(example)
                        else:
                            idx = random.randrange(0, shuffle_buffer_size)
                            yield shuffle_buffer[idx]
                            shuffle_buffer[idx] = example
                    else:
                        yield example
            except StopIteration:
                if infinite:
                    iterator = iter(dataset)
                    warnings.warn(
                        "The dataset reached end and the iterator is reset to the start.",
                        stacklevel=1,
                    )
                else:
                    break

        if len(shuffle_buffer) == 0:
            raise ValueError(
                "The dataset is empty or does not contain enough samples to yield a single packed sequence."
            )

        for idx in random.sample(range(len(shuffle_buffer)), len(shuffle_buffer)):
            yield shuffle_buffer[idx]


    return constant_length_generator



def packed_dataset(
    dataset,
    dataset_tokens_field,
    max_sequence_length,
    num_of_sequences,
    append_eos_token=True,
    eos_token_id: int | None = None,
):
    """
    Prepares a packed dataloader from the given dataset.

    This method is designed for efficient training of language models by packing multiple
    sequences from the dataset into a single sample. This can be particularly beneficial
    for handling long sequences and optimizing GPU/TPU utilization.

    Args:
        processing_class: The processing_class used for text encoding.
        dataset (Dataset): The dataset to prepare.
        dataset_text_field (str): The name of the text field in the dataset.
        max_sequence_length (int): The maximum length of each packed sequence.
        num_of_sequences (int): The number of sequences to pack into a single sample.
        chars_per_token (float): The average number of characters per token, used for estimating
            the number of tokens in a text sequence.
        formatting_func (tp.Callable, optional): A function to format each sample from the dataset
            before packing. It should take a sample as input and return a dictionary with a "text"
            key containing the processed text. Defaults to None.
        append_eos_token (bool, optional): Whether to append a special concatenation token
            between packed sequences. Defaults to True.

    Returns:
        Dataset: The processed dataset with packed sequences.

    Raises:
        ValueError: If both `dataset_text_field` and `formatting_func` are None, or if there's
            an error during dataset packing.
    """
    if dataset_tokens_field is None:
        raise ValueError(
            "You need to pass a `dataset_text_field` argument."
        )

    constant_length_iterator = create_constant_length_dataset(
        dataset,
        tokens_field=dataset_tokens_field,
        seq_length=max_sequence_length,
        eos_token_id=eos_token_id,
        batch_size=num_of_sequences,
        append_concat_token=append_eos_token,
    )

    def data_generator(inner_constant_length_iterator):
        yield from inner_constant_length_iterator()

    # Import Only and Only when needed, don't dst the runtime.
    try:
        from datasets import IterableDataset
        from datasets.arrow_writer import SchemaInferenceError
        from datasets.builder import DatasetGenerationError
    except ImportError as exc:
        raise ImportError(
            "Could not import `datasets` from Hugging Face. Make sure to install the "
            "library using `pip install datasets`."
        ) from exc
    try:
        packed_dataset = IterableDataset.from_generator(
            data_generator,
            gen_kwargs={"inner_constant_length_iterator": constant_length_iterator},
        )
    except (DatasetGenerationError, SchemaInferenceError) as exc:
        raise ValueError(
            "Error occurred while packing the dataset. "
            "Make sure that your dataset has enough samples to at least yield one packed sequence.\n"
            f"External Information : {exc}"
        ) from exc
    return packed_dataset
