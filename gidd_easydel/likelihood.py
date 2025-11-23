import re
from typing import Literal

import flax.nnx as nn
import jax
import jax.numpy as jnp
import tqdm.auto as tqdm
from jax.sharding import PartitionSpec
from eformer.escale import PartitionAxis
from eformer.escale.partition.constraints import with_sharding_constraint

from gidd_easydel.diffusion_trainer.schedule import create_mixing_schedule


def loss_fn(
    mixing_schedule,
    log_snr,
    logits,
    input_ids,
    labels,
    token_partition_spec,
    logit_partition_spec,
    vocab_size,
):
    logits = with_sharding_constraint(logits, logit_partition_spec)
    x_hat = nn.softmax(logits.astype(jnp.float32), axis=-1)
    log_p_t = mixing_schedule.marginal_log_probs(log_snr, x_hat).astype(jnp.float32)

    labels_one_hot = with_sharding_constraint(jax.nn.one_hot(labels, vocab_size, dtype=logits.dtype), logit_partition_spec)

    elbo_weights, _ = mixing_schedule.get_elbo_weights(log_snr, input_ids, labels, return_aux=True)
    elbo_weights = elbo_weights.clip(0, 1e6)
    elbo_weights = with_sharding_constraint(elbo_weights, token_partition_spec)

    log_q_t = mixing_schedule.marginal_log_probs(log_snr, labels_one_hot)
    log_q_t = with_sharding_constraint(log_q_t, logit_partition_spec)

    log_p_zt = with_sharding_constraint(
        jnp.take_along_axis(log_p_t, input_ids[..., None], axis=-1).squeeze(-1).astype(jnp.float32),
        token_partition_spec,
    )
    log_q_zt = with_sharding_constraint(
        jnp.take_along_axis(log_q_t, input_ids[..., None], axis=-1).squeeze(-1).astype(jnp.float32),
        token_partition_spec,
    )

    ratio = jnp.exp(log_q_zt) / (jnp.exp(log_p_zt) + 1e-12)
    log_ratio = log_q_zt - log_p_zt
    is_div = ratio - log_ratio - 1

    kl_div = with_sharding_constraint(
        (jnp.exp(log_q_t) * (log_q_t - log_p_t)).sum(-1).astype(jnp.float32),
        token_partition_spec,
    )

    elbo = elbo_weights * kl_div + elbo_weights * is_div
    return elbo


def likelihood_step(
    module,
    mixing_schedule,
    labels,
    attn_mask,
    noise_mask,
    log_snr,
    key,
    vocab_size=131072,
    mask_token_id=3,
    logit_partition_spec=None,
    token_partition_spec=None,
):
    input_ids = mixing_schedule.sample_marginals(key, log_snr, labels)
    input_ids = jnp.where(noise_mask, input_ids, labels)
    actual_alpha = ((input_ids == labels) * noise_mask).sum(-1) / noise_mask.sum(-1)
    actual_log_snr = jnp.log(actual_alpha / (1 - actual_alpha + 1e-12) + 1e-12)

    log_snrs = actual_log_snr
    # log_snrs = (log_snr * jnp.ones((labels.shape[0],), dtype=jnp.float32))

    outputs = module(
        input_ids=input_ids,
        attention_mask=attn_mask,
    )
    logits = outputs.logits.at[..., mask_token_id].set(-1e6)
    
    elbos = loss_fn(
        mixing_schedule,
        log_snrs[:, None],
        logits,
        input_ids,
        labels,
        token_partition_spec,
        logit_partition_spec,
        vocab_size,
    )

    elbos = elbos * noise_mask
    return elbos, logits, input_ids


likelihood_step_jit = jax.jit(likelihood_step, static_argnames=("vocab_size", "mask_token_id", "logit_partition_spec", "token_partition_spec"))


def likelihood(
    mesh,
    module,
    tokenizer,
    prompts: list[str],
    completions: list[str],
    hybrid_mixing_shift: float,
    prior: Literal["uniform", "mask"],
    num_denoising_steps=128,
    max_sequence_length=2048,
    noise_schedule: Literal["linear", "cosine"] = "linear",
    min_log_snr: float = -9.0,
    max_log_snr: float = 9.0,
    seed=0,
    show_progress=True,
    step_callback=None,
):
    assert prompts is not None and len(prompts) > 0, "Prompts must be provided for generation (use empty prompts for unconditional generation)."
    vocab_size = len(tokenizer)
    batch_size = len(prompts)
    assert len(completions) == batch_size, "Number of completions must match number of prompts."

    partition_axis = PartitionAxis()
    logit_partition_spec = PartitionSpec(
        partition_axis.batch_axis,
        partition_axis.sequence_axis,
        None,
    )
    token_partition_spec = PartitionSpec(
        partition_axis.batch_axis,
        partition_axis.sequence_axis,
    )

    step_fn = likelihood_step_jit

    mixing_schedule = create_mixing_schedule(
        rate=noise_schedule,
        min_log_snr=min_log_snr,
        max_log_snr=max_log_snr,
        prior_distribution=prior,
        hybrid_shift=hybrid_mixing_shift,
        vocab_size=vocab_size,
        mask_token_id=tokenizer.mask_token_id,
    )

    key = jax.random.PRNGKey(seed)

    if noise_schedule == "cosine":
        ts = jnp.linspace(0.999, 1e-3, num_denoising_steps + 1)
        alpha_ts = 0.5 + 0.5 * jnp.cos(ts * jnp.pi)
    elif noise_schedule == "linear":
        alpha_ts = jnp.linspace(1e-4, 0.9999, num_denoising_steps + 1)
    else:
        raise ValueError(f"Unknown noise schedule: {noise_schedule}")

    log_snrs = jnp.log(alpha_ts / (1 - alpha_ts)).clip(min_log_snr, max_log_snr)

    if prior == "mask":
        input_ids = tokenizer.mask_token_id * jnp.ones((batch_size, max_sequence_length), dtype=jnp.int32)
    elif prior == "uniform":
        input_ids = jax.random.randint(key, (batch_size, max_sequence_length), 0, vocab_size, dtype=jnp.int32)
    else:
        raise ValueError(f"Unknown prior: {prior}")
    input_ids = with_sharding_constraint(input_ids, token_partition_spec)
    
    noise_mask = jnp.ones_like(input_ids, dtype=jnp.bool_)
    completion_mask = jnp.zeros_like(input_ids, dtype=jnp.bool_)

    prompt_lens, completion_lens = [], []
    prompt_ids = tokenizer(prompts, add_special_tokens=True).input_ids
    completion_ids = tokenizer(completions, add_special_tokens=True).input_ids
    for i in range(len(prompt_ids)):
        y = jnp.asarray(completion_ids[i][1:])  # remove bos token
        completion_len = y.shape[0]

        x = jnp.asarray(prompt_ids[i][:-1])  # remove eos token
        x = x[: max_sequence_length - completion_len]
        prompt_len = x.shape[0]

        xy = jnp.concatenate([x, y], axis=0)
        input_ids = input_ids.at[i, : prompt_len + completion_len].set(xy)
        noise_mask = noise_mask.at[i, :prompt_len].set(False)
        completion_mask = completion_mask.at[i, prompt_len : prompt_len + completion_len].set(True)

        prompt_lens.append(prompt_len)
        completion_lens.append(completion_len)

    attn_mask = (noise_mask[..., None] >= noise_mask[..., None, :])

    total_nlls = 0
    num_samples = 0
    with mesh:
        for i in tqdm.trange(num_denoising_steps, disable=not (jax.process_index() == 0 and show_progress)):
            key, key_i = jax.random.split(key)
            nlls, logits, noisy_ids = step_fn(
                module,
                mixing_schedule,
                input_ids,
                attn_mask,
                noise_mask,
                log_snrs[i],
                key_i,
                vocab_size,
                tokenizer.mask_token_id,
                logit_partition_spec=logit_partition_spec,
                token_partition_spec=token_partition_spec,
            )
            if step_callback is not None:
                step_callback(
                    step=i,
                    log_snr=log_snrs[i],
                    input_ids=noisy_ids,
                    labels=input_ids,
                    nlls=nlls,
                    logits=logits,
                    noise_mask=noise_mask,
                    completion_mask=completion_mask,
                )
            total_nlls = total_nlls + nlls
            num_samples = num_samples + 1

    nlls = total_nlls / num_samples

    total_nll = (nlls * completion_mask).sum(axis=1)
    avg_nll = total_nll / jnp.array(completion_lens)
    ppl = jnp.exp(avg_nll)
    return {
        "nlls": nlls,
        "total_nll": total_nll,
        "avg_nll": avg_nll,
        "ppl": ppl,
    }
