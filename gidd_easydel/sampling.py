import re
from typing import Literal

import flax.nnx as nn
import jax
import jax.numpy as jnp
import tqdm.auto as tqdm
from jax.sharding import PartitionSpec
from eformer.escale import PartitionAxis
from eformer.escale.partition.constraints import with_sharding_constraint


def safe_sigmoid(x, precision=jnp.float32):
    return nn.sigmoid(x.astype(precision)).astype(x.dtype)

def pi_lambda(log_snr, shift=0.0, mask_token_id=3, vocab_size=131072):
    uniform_vec = jnp.ones((vocab_size,)) / (vocab_size - 1)
    uniform_vec = uniform_vec.at[mask_token_id].set(0.0)

    alpha = safe_sigmoid(log_snr + shift)
    pi = alpha * uniform_vec
    pi = pi.at[mask_token_id].set(1.0 - alpha)
    return pi

def prior_prob(input_ids, shift=0.0, mask_token_id=3, vocab_size=131072, min_log_snr=-9.0):
    pi = pi_lambda(jnp.asarray(min_log_snr), shift=shift, mask_token_id=mask_token_id, vocab_size=vocab_size)
    prs = jnp.take_along_axis(pi[None, None, :], input_ids[..., None], axis=-1).squeeze(-1)
    return prs

def sample_categorical(key, probs):
    vocab_size = probs.shape[-1]
    cumprobs = jnp.cumsum(probs, axis=-1)
    cumprobs = cumprobs.at[..., -1].set(1.0)
    u = jax.random.uniform(
        key,
        shape=cumprobs.shape[:-1] + (1,),
        dtype=cumprobs.dtype,
    )
    output = jnp.sum(
        (cumprobs < u).astype(jnp.int32),
        axis=-1,
    ).clip(0, vocab_size - 1)
    return output


def ancestral_sampling_step(
    module,
    input_ids,
    attn_mask,
    noise_mask,
    log_snr_t,
    log_snr_s,
    key,
    hybrid_mixing_shift,
    vocab_size,
    mask_token_id,
    top_k=1,
    temp=0.0,
    logits=None,
    partition_spec=None,
):
    # attn_mask = (noise_mask[..., None] >= noise_mask[..., None, :])
    if logits is None:
        outputs = module(
            input_ids=input_ids,
            attention_mask=attn_mask,
        )
        logits = with_sharding_constraint(outputs.logits, partition_spec)
        logits = logits.at[..., mask_token_id].set(-1e6)
    x_hat = nn.softmax(logits.astype(jnp.float32))

    alpha_t = safe_sigmoid(log_snr_t)
    alpha_s = safe_sigmoid(log_snr_s)
    beta_t, beta_s = 1 - alpha_t, 1 - alpha_s
    alpha_t_s = alpha_t / alpha_s

    pi_t = pi_lambda(log_snr_t, shift=hybrid_mixing_shift, mask_token_id=mask_token_id, vocab_size=vocab_size)
    pi_s = pi_lambda(log_snr_s, shift=hybrid_mixing_shift, mask_token_id=mask_token_id, vocab_size=vocab_size)
    beta_pi_t_s = beta_t * pi_t - alpha_t_s * beta_s * pi_s
    beta_pi_t_s_at_z = jnp.take_along_axis(
        beta_pi_t_s[None, None, :],
        input_ids[..., None],
        axis=-1,
    )

    q_t = alpha_t * x_hat + beta_t * pi_t[None, None, :]
    q_t_at_zt = jnp.take_along_axis(
        q_t,
        input_ids[..., None],
        axis=-1,
    )
    q_s = alpha_s * x_hat + beta_s * pi_s[None, None, :]

    z_vec = nn.one_hot(input_ids, vocab_size, dtype=x_hat.dtype)
    q_t_s_at_zt = alpha_t_s * z_vec + beta_pi_t_s_at_z

    p_s_t = q_s / q_t_at_zt * q_t_s_at_zt

    next_input_ids = sample_categorical(key, p_s_t)

    return next_input_ids * noise_mask + input_ids * (1 - noise_mask), logits, None, None, None


def adaptive_sampling_step(
    module,
    input_ids,
    attn_mask,
    noise_mask,
    log_snr_t,
    log_snr_s,
    key,
    hybrid_mixing_shift,
    vocab_size,
    mask_token_id,
    top_k=1,
    temp=0.0,
    logits=None,
    partition_spec=None,
):
    # attn_mask = (noise_mask[..., None] >= noise_mask[..., None, :])
    if logits is None:
        outputs = module(
            input_ids=input_ids,
            attention_mask=attn_mask,
        )
        logits = with_sharding_constraint(outputs.logits, partition_spec)
        logits = logits.at[..., mask_token_id].set(-1e6)
    probs = nn.softmax(logits.astype(jnp.float32))
    p_prior = prior_prob(
        input_ids,
        shift=hybrid_mixing_shift,
        mask_token_id=mask_token_id,
        vocab_size=vocab_size,
    )
    # entropy = -jnp.sum(probs * jnp.log(probs + 1e-10), axis=-1)

    # new method
    curr_prob = jnp.take_along_axis(probs, input_ids[..., None], axis=-1).squeeze(-1)
    best_prob = probs.max(axis=-1)
    prob_delta = (best_prob - curr_prob) * p_prior
    # prob_delta = jnp.exp(-entropy) - curr_prob
    # prob_delta = 1 - curr_prob
    # prob_delta = best_prob
    next_pos = jnp.argsort(prob_delta * noise_mask, axis=-1)[:, -top_k:]

    if temp > 0.0:
        probs = nn.softmax(logits.astype(jnp.float32) / temp)
        pred_token = sample_categorical(key, probs)
    else:
        pred_token = probs.argmax(axis=-1)

    batch_indices = jnp.arange(input_ids.shape[0])[:, None]
    next_input_ids = input_ids.at[batch_indices, next_pos].set(pred_token[batch_indices, next_pos])

    # # old method (masked diffusion only)
    # best_prob = prob.max(axis=-1) * (input_ids == mask_token_id) * noise_mask
    # pred_token = prob.argmax(axis=-1)
    # next_pos = best_prob.argmax(axis=-1)
    # next_input_ids = input_ids.at[jnp.arange(input_ids.shape[0]), next_pos].set(pred_token[jnp.arange(input_ids.shape[0]), next_pos])

    return next_input_ids * noise_mask + input_ids * (1 - noise_mask), logits, prob_delta, curr_prob, best_prob


ancestral_sampling_step_jit = jax.jit(ancestral_sampling_step, static_argnames=("vocab_size", "mask_token_id", "top_k", "temp", "partition_spec"))
adaptive_sampling_step_jit = jax.jit(adaptive_sampling_step, static_argnames=("vocab_size", "mask_token_id", "top_k", "temp", "partition_spec"))

    
def clean_up_spaces(text):
    text = re.sub(r" (?=[.,:'’?])", "", text)
    text = re.sub(r" (?=-[\w])", "", text)
    text = re.sub(r"(?<=[0-9]) (?=[0-9])", "", text)
    text = re.sub(r"(?<=\$) (?=[0-9])", "", text)
    text = re.sub(r"(?<=[0-9][.,]) (?=[0-9])", "", text)
    text = re.sub(r"  (?=[0-9.,|/'])", " ", text)
    return text

def generate(
    mesh,
    module,
    tokenizer,
    prompts: list[str],
    hybrid_mixing_shift: float,
    prior: Literal["uniform", "mask"],
    num_denoising_steps=128,
    max_sequence_length=2048,
    min_completion_length=128,
    max_completion_length=128,
    noise_schedule="cosine",
    seed=0,
    sampler: Literal["ancestral", "adaptive"] = "adaptive",
    top_k=1,
    temperature=0.0,
    completion_only=True,
    clean_up_tokenization_spaces=True,
    return_token_ids=False,
    show_progress=True,
    step_callback=None,
):
    assert prompts is not None and len(prompts) > 0, "Prompts must be provided for generation (use empty prompts for unconditional generation)."
    vocab_size = len(tokenizer)
    batch_size = len(prompts)

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

    if sampler == "ancestral":
        sampling_step = ancestral_sampling_step_jit
    elif sampler == "adaptive":
        sampling_step = adaptive_sampling_step_jit
        # sampling_step = adaptive_sampling_step
    else:
        raise ValueError(f"Unknown sampler: {sampler}")

    key = jax.random.PRNGKey(seed)

    if noise_schedule == "cosine":
        ts = jnp.linspace(0.999, 1e-3, num_denoising_steps + 1)
        alpha_ts = 0.5 + 0.5 * jnp.cos(ts * jnp.pi)
    elif noise_schedule == "linear":
        alpha_ts = jnp.linspace(1e-4, 0.9999, num_denoising_steps + 1)
    else:
        raise ValueError(f"Unknown noise schedule: {noise_schedule}")

    log_snrs = jnp.log(alpha_ts / (1 - alpha_ts)).clip(-9.0, 9.0)

    if prior == "mask":
        input_ids = tokenizer.mask_token_id * jnp.ones((batch_size, max_sequence_length), dtype=jnp.int32)
    elif prior == "uniform":
        input_ids = jax.random.randint(key, (batch_size, max_sequence_length), 0, vocab_size, dtype=jnp.int32)
    else:
        raise ValueError(f"Unknown prior: {prior}")
    input_ids = with_sharding_constraint(input_ids, token_partition_spec)
    
    noise_mask = jnp.ones_like(input_ids, dtype=jnp.bool_)

    if prompts is not None:
        prompt_lens = []
        prompt_ids = tokenizer(prompts, add_special_tokens=True).input_ids
        for i in range(len(prompt_ids)):
            x = jnp.asarray(prompt_ids[i][-(max_sequence_length - min_completion_length):])
            x = x[:-1]  # remove eos token
            prompt_len = x.shape[0]
            input_ids = input_ids.at[i, :prompt_len].set(x)
            noise_mask = noise_mask.at[i, :prompt_len].set(False)
            prompt_lens.append(prompt_len)
    else:
        prompt_lens = [0] * batch_size

    attn_mask = (noise_mask[..., None] >= noise_mask[..., None, :])

    for i in range(batch_size):
        noise_mask = noise_mask.at[i, prompt_lens[i] + max_completion_length :].set(False)

    use_cached_logits, logits = False, None
    with mesh:
        for i in tqdm.trange(num_denoising_steps, disable=not (jax.process_index() == 0 and show_progress)):
            key, key_i = jax.random.split(key)
            output_ids, logits, pos_scores, curr_conf, best_conf = sampling_step(
                module,
                input_ids,
                attn_mask,
                noise_mask,
                log_snrs[i],
                log_snrs[i+1],
                key_i,
                hybrid_mixing_shift,
                vocab_size,
                tokenizer.mask_token_id,
                partition_spec=logit_partition_spec,
                top_k=top_k,
                temp=temperature,
                logits=logits if use_cached_logits else None,
            )
            if step_callback is not None:
                step_callback(
                    step=i,
                    input_ids=input_ids,
                    output_ids=output_ids,
                    logits=logits,
                    noise_mask=noise_mask,
                    position_scores=pos_scores,
                    curr_conf=curr_conf,
                    best_conf=best_conf,
                )
            use_cached_logits = (input_ids == output_ids).all()
            input_ids = output_ids

    if return_token_ids:
        if completion_only:
            completion_ids = []
            for i in range(batch_size):
                prompt_len = prompt_lens[i]
                completion_id = input_ids[i, prompt_len : prompt_len + max_completion_length]
                completion_ids.append(completion_id)
            return completion_ids
        return input_ids

    texts = []
    if completion_only:
        for i in range(batch_size):
            prompt_len = prompt_lens[i]
            completion = tokenizer.batch_decode(
                input_ids[None, i, prompt_len : prompt_len + max_completion_length],
                clean_up_tokenization_spaces=False,
            )[0].split(tokenizer.eos_token)[0]
            texts.append(completion)
    else:
        completions = tokenizer.batch_decode(
            input_ids,
            clean_up_tokenization_spaces=False,
        )
        for text in completions:
            texts.append(text.split(tokenizer.eos_token)[0])

    if clean_up_tokenization_spaces:
        texts = [clean_up_spaces(text) for text in texts]

    return texts
