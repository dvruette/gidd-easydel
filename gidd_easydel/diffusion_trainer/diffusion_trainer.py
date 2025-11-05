import random
import typing as tp

import jax
import jax.numpy as jnp
import flax.nnx as nn
from jax.sharding import NamedSharding, PartitionSpec
from transformers import PreTrainedTokenizerBase

from easydel.infra.base_module import EasyDeLBaseModule
from easydel.infra.base_state import EasyDeLState
from easydel.utils.helpers import get_logger
from easydel.trainers.trainer import Trainer
from easydel.trainers.trainer_protocol import TrainerConfigureFunctionOutput
from easydel.utils.compiling_utils import ejit

from ._fn import training_step
from .loss import GiddLoss
from .schedule import MixingSchedule, create_mixing_schedule
from .diffusion_config import DiffusionConfig

if tp.TYPE_CHECKING:
    from datasets import Dataset
else:
    Dataset = tp.Any

logger = get_logger(__name__)


class DiffusionTrainer(Trainer):
    arguments: DiffusionConfig
    tokenizer: PreTrainedTokenizerBase
    mixing_schedule: MixingSchedule
    loss_fn: GiddLoss

    def __init__(
        self,
        arguments: DiffusionConfig,
        tokenizer: PreTrainedTokenizerBase,
        model_state: EasyDeLState | None = None,
        model: EasyDeLBaseModule | None = None,
        train_dataset: Dataset | None = None,
        eval_dataset: Dataset | dict[str, Dataset] | None = None,
        append_eos_token: bool = True,
        seed: int | None = None,
        dtype: jnp.dtype = None,
    ):
        assert isinstance(arguments, DiffusionConfig), "passed argument must be a `DiffusionConfig`."
        assert model is not None or model_state is not None, "You must pass a `model` to the DiffusionTrainer."
        _model = model
        if _model is None:
            _model = model_state.model

        if seed is None:
            seed = random.randint(0, 2**31 - 1)
        self.key = jax.random.PRNGKey(seed)

        self.arguments = arguments
        self.tokenizer = tokenizer
        self.mixing_schedule = create_mixing_schedule(
            rate=arguments.mixing_rate,
            min_log_snr=arguments.min_log_snr,
            max_log_snr=arguments.max_log_snr,
            distribution="hybrid",
            vocab_size=len(tokenizer),
            prior_distribution="masked",
            mask_token_id=tokenizer.mask_token_id,
            hybrid_scale=arguments.hybrid_mixing_scale,
            hybrid_shift=arguments.hybrid_mixing_shift,
            dtype=dtype,
        )
        self.loss_fn = GiddLoss(
            mixing_schedule=self.mixing_schedule,
            vocab_size=len(tokenizer),
            beta_is_div=arguments.beta_is_div,
            mask_token_id=tokenizer.mask_token_id,
            partition_axis=_model.config.partition_axis,
        )

        super().__init__(
            arguments=arguments,
            dataset_train=train_dataset,
            dataset_eval=eval_dataset,
            model_state=model_state,
            model=model,
            data_collator=self.prepare_batch,
        )
        logger.info("Initialized DiffusionTrainer")

    def prepare_batch(self, batch: dict[str, jax.Array]) -> dict[str, jax.Array]:
        self.key, curr_key = jax.random.split(self.key, 2)
        batch["rng_key"] = curr_key
        return batch
    
    def _sample_log_snr(self, key: jax.Array, shape: tuple[int, ...]) -> jax.Array:
        """
        Sample log-SNR values for the given shape based on the configured noise parameters.
        Args:
            key: Random key for sampling
            shape: Shape of the sequence (batch_size, sequence_length)
        Returns:
            jax.Array: Sampled log-SNR values for each token in the sequence
        """
        if len(shape) != 2:
            raise ValueError(f"Expected shape to be 2D (batch_size, sequence_length), got {len(shape)}D shape: {shape}")

        batch_size, seq_len = shape

        key, snr_key, ind_key, lin_key = jax.random.split(key, 4)
        log_snr = self.mixing_schedule.sample_log_snr(snr_key, shape)

        r = jax.random.uniform(key, (batch_size,))

        if self.arguments.noise_p_independent > 0:
            # Sample independent SNR for each token
            is_independent = r < self.arguments.noise_p_independent
            log_snr = jnp.where(is_independent[:, None], log_snr, log_snr[:, 0, None])
        else:
            is_independent = jnp.zeros(batch_size, dtype=bool)
            log_snr = jnp.broadcast_to(log_snr[:, 0, None], (batch_size, seq_len))

        if self.arguments.noise_p_linear > 0:
            # Sample linear SNR based on token position
            is_linear = ~is_independent & (r < self.arguments.noise_p_linear + self.arguments.noise_p_independent)
            linear_t = jnp.linspace(0, 1, seq_len + 2)[1:-1]
            linear_log_snr = self.mixing_schedule.log_snr_from_time(linear_t)
            log_snr = jnp.where(is_linear[:, None], linear_log_snr[None, :], log_snr)

        # Ensure log_snr is broadcasted correctly
        assert log_snr.shape == (batch_size, seq_len), "log_snr shape mismatch"
        return log_snr

    def _sample_noise_mask(self, key: jax.Array, shape: tuple[int, ...]) -> jax.Array:
        """
        Sample a noise mask for the given shape based on the configured noise parameters.
        
        Args:
            key: Random key for sampling
            shape: Shape of the sequence (batch_size, sequence_length)
            
        Returns:
            Boolean mask where True indicates tokens that should have noise applied
        """
        batch_size, seq_len = shape

        # Start with all tokens receiving noise
        noise_mask = jnp.ones(shape, dtype=bool)
        attn_mask = jnp.ones((batch_size, seq_len, seq_len), dtype=bool)
        causal_mask = jnp.tril(jnp.ones((1, seq_len, seq_len), dtype=bool))

        # Split key for different sampling operations
        key, prompt_key, infill_key = jax.random.split(key, 3)

        # Sample which sequences get prompt conditioning (noise-free prefix)
        if self.arguments.noise_mask_p_prompt > 0:
            prompt_key, k = jax.random.split(prompt_key, 2)
            has_prompt_mask = jax.random.bernoulli(
                k, 
                self.arguments.noise_mask_p_prompt, 
                shape=(batch_size,)
            )

            # Sample fraction of prompt tokens for all sequences
            prompt_key, k = jax.random.split(prompt_key, 2)
            r = jax.random.uniform(k, (batch_size,))
            prompt_frac = jnp.arccos(1 - 2*r) / jnp.pi * self.arguments.noise_mask_max_cond_frac

            # Create position masks for prompts
            positions = jnp.arange(seq_len)[None, :]  # (1, seq_len)
            promp_mask = positions <= (prompt_frac[:, None] * (seq_len - 1))  # (batch_size, seq_len)

            # Apply prompt conditioning where has_prompt_mask is True
            prompt_mask = has_prompt_mask[:, None] & promp_mask
            noise_mask = noise_mask & ~prompt_mask

            if self.arguments.causal_prompt_attention:
                attn_mask = attn_mask & jnp.where(noise_mask[..., None], True, causal_mask)
            else:
                attn_mask = attn_mask & (noise_mask[..., None] >= noise_mask[..., None, :])


        # Sample which sequences get infilling conditioning (random noise-free tokens)
        if self.arguments.noise_mask_p_infilling > 0:
            has_infill_mask = jax.random.bernoulli(
                infill_key, 
                self.arguments.noise_mask_p_infilling, 
                shape=(batch_size,)
            )

            # Sample fraction of infill tokens for all sequences
            infill_key, k1, k2 = jax.random.split(infill_key, 3)
            r1 = jax.random.uniform(k1, (batch_size,))
            infill_frac = jnp.arccos(1 - 2*r1) / jnp.pi * self.arguments.noise_mask_max_cond_frac

            # Sample positions for infill tokens
            infill_p = jax.random.uniform(k2, (batch_size, seq_len))
            infill_mask = infill_p < infill_frac[:, None]  # (batch_size, seq_len)

            # Apply infill conditioning where has_infill_mask is True
            infill_mask = has_infill_mask[:, None] & infill_mask
            noise_mask = noise_mask & ~infill_mask

        # if self.arguments.attn_mask_pattern == "lognormal":
        #     rand = 5 * jax.random.lognormal(key, 1.0, (batch_size, seq_len))
        #     order = jnp.arange(seq_len)[None, :] + rand
        #     attn_mask = order[..., None] >= order[..., None, :]

        return noise_mask, attn_mask
    
    def _insert_empty_tokens(self, key, input_ids, empty_token_id: int = None, max_empty_token_frac: float | None = None):
        max_empty_token_frac = max_empty_token_frac or self.arguments.max_empty_token_frac
        empty_token_id = empty_token_id or self.tokenizer.pad_token_id
        assert 0.0 <= max_empty_token_frac < 1.0

        if max_empty_token_frac == 0.0:
            return input_ids

        batch_size, seq_len = input_ids.shape
        total_len = int(round(seq_len / (1.0 - max_empty_token_frac)))

        frac_key, key = jax.random.split(key)
        empty_token_fracs = jax.random.uniform(frac_key, shape=(batch_size,)) * max_empty_token_frac

        keys = jax.random.split(key, batch_size)

        def per_example(seq, k, empty_frac):
            empty_count = jnp.ceil(seq_len * (empty_frac / (1.0 - empty_frac))).astype(jnp.int32)
            perm = jax.random.permutation(k, seq_len)
            ranks = jnp.full((total_len,), total_len, dtype=jnp.int32).at[perm].set(jnp.arange(seq_len, dtype=jnp.int32))
            empty_mask = ranks < empty_count
            dest = jnp.nonzero(~empty_mask, size=seq_len, fill_value=0)[0]
            out = jnp.full((total_len,), empty_token_id, dtype=seq.dtype).at[dest].set(seq)
            return out

        return jax.vmap(per_example)(input_ids, keys, empty_token_fracs)[:, :seq_len]

    def configure_functions(self) -> TrainerConfigureFunctionOutput:
        """
        Configures and JIT-compiles the training and evaluation step functions.

        This method sets up the necessary functions for training and evaluation, including:
            - Initialization of the model state.
            - Sharding of the model parameters and optimizer state.
            - JIT-compilation of the training and evaluation step functions.

        Returns:
            TrainerConfigureFunctionOutput: An object containing the configured functions and other relevant information.
        """
        logger.info("Configuring functions for DiffusionTrainer...")
        mesh = self.model.mesh

        empty_sharding = NamedSharding(spec=PartitionSpec(), mesh=mesh)

        self._train_shared_fn_static_args = (
            self.loss_fn,
            self._sample_log_snr,
            self._sample_noise_mask,
            self.mixing_schedule.sample_marginals,
            self._insert_empty_tokens,
            self.arguments.loss_config,
            self.scheduler,
            self.arguments.step_partition_spec,
            self.arguments.gradient_accumulation_steps,
            self.arguments.loss_aggregation,
            self.arguments.loss_scale,
            True,  # is_train
        )
        static_argnames = (2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13)

        sharded_training_step_function = ejit(
            training_step,
            in_shardings=(self.state_shardings, empty_sharding),
            out_shardings=(self.state_shardings, empty_sharding),
            donate_argnums=(0,),
            static_argnums=static_argnames,
        )

        self._eval_shared_fn_static_args = self._train_shared_fn_static_args[:-1] + (False,)  # is_train=False

        sharded_evaluation_step_function = ejit(
            training_step,
            in_shardings=(self.state_shardings, empty_sharding),
            out_shardings=empty_sharding,
            static_argnums=static_argnames,
        )

        self.arguments.ensure_checkpoint_path()

        logger.info("Functions configured successfully.")
        return TrainerConfigureFunctionOutput(
            sharded_training_step_function=sharded_training_step_function,
            sharded_evaluation_step_function=sharded_evaluation_step_function,
            mesh=mesh,
            checkpoint_manager=self.arguments.get_streaming_checkpointer(),
        )

