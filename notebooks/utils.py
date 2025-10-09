from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import functools

import wandb
import numpy as np
import pandas as pd
import tqdm.auto as tqdm
from scipy.optimize import curve_fit, fsolve


WANDB_PROJECT = os.getenv("WANDB_PROJECT", "dvruette/EasyDeL-diffusiontrainer-Gidd")


def loss_curve(steps, C, B, b):
    return C + B * steps**(-b)

def fit_loss_curve(history, x_key="_step", y_key="train/elbo", min_x=2000, max_x=None):
    h = history.copy()
    if min_x is not None:
        h = h.loc[h[x_key] >= min_x]
    if max_x is not None:
        h = h.loc[h[x_key] <= max_x]
    if len(h) < 10:
        raise ValueError("Not enough data points to fit loss curve")
    h = h[~h[y_key].isnull() & ~(h[y_key] == 'NaN')]
    xs, ys = h[x_key].values, h[y_key].values
    popt, pcov = curve_fit(loss_curve, xs, ys, p0=(0.0, 1.0, 1.0), maxfev=10000)

    r2 = 1 - (np.sum((ys - loss_curve(xs, *popt)) ** 2) / np.sum((ys - np.mean(ys)) ** 2))

    y_hat = loss_curve(history[x_key], *popt)

    return y_hat, popt, r2


def loss_at_t(steps, config_row):
    return loss_curve(steps, *config_row[["fit_C", "fit_B", "fit_b"]])

def loss_at_flops(flops, config_row, include_emb_flops=True):
    if include_emb_flops:
        steps = flops / config_row["flops_per_step"]
    else:
        steps = flops / config_row["non_emb_flops_per_step"]
    if steps > config_row["total_steps"] or steps < 2000:
        return np.nan
    return loss_curve(steps, *config_row[["fit_C", "fit_B", "fit_b"]])

def steps_at_loss(target_loss, config_row):
    return fsolve(
        lambda x: loss_curve(x, *config_row[["fit_C", "fit_B", "fit_b"]]) - target_loss,
        1.0
    )[0]

def tokens_at_loss(target_loss, config_row):
    steps = steps_at_loss(target_loss, config_row)
    return steps * config_row["batch_size"] * config_row["max_seq_len"]

def flops_at_loss(target_loss, config_row, include_emb_flops=True):
    steps = steps_at_loss(target_loss, config_row)
    if include_emb_flops:
        return steps * config_row["non_emb_flops_per_step"]
    else:
        return steps * config_row["flops_per_step"]


def loss_curve_ema(history: pd.DataFrame, y_key="train/elbo", gamma=0.96):
    ema = history[y_key].ewm(alpha=1 - gamma).mean()
    return ema


def flops_per_token(num_layers, hidden_size, num_attn_heads, seq_length, vocab_size=131072, include_embed_flops=True):
    attn_qkv = 2 * 3 * hidden_size * hidden_size
    attn_logits = 2 * hidden_size * seq_length
    attn_softmax = 3 * num_attn_heads * seq_length
    attn_reduce = 2 * seq_length * hidden_size
    attn_proj = 2 * hidden_size * hidden_size

    attn_flops = attn_qkv + attn_logits + attn_softmax + attn_reduce + attn_proj
    mlp_flops = 2 * 2 * hidden_size * (4 * hidden_size)

    embed_flops = 2 * vocab_size * hidden_size
    unembed_flops = 2 * hidden_size * vocab_size

    flops_per_token = num_layers * (attn_flops + mlp_flops)

    if include_embed_flops:
        flops_per_token += embed_flops + unembed_flops

    return 3.0 * flops_per_token  # fwd + bwd


def flops_per_step(num_layers, hidden_size, num_attn_heads, seq_length, batch_size, vocab_size=131072, include_embed_flops=True):
    fpt = flops_per_token(num_layers, hidden_size, num_attn_heads, seq_length, vocab_size, include_embed_flops)
    flops_per_batch = fpt * seq_length * batch_size
    return flops_per_batch


def params_from_config(config: dict, include_embed_params=True):
    num_layers = config["num_layers"]
    hidden_size = config["hidden_size"]
    vocab_size = config.get("vocab_size", 131072)

    attn_ln = hidden_size
    attn_qkv = 3 * hidden_size * hidden_size
    attn_proj = hidden_size * hidden_size
    qk_norm = 2 * hidden_size
    attn_params = attn_ln + attn_qkv + attn_proj + qk_norm

    mlp_ln = hidden_size
    mlp_projs = 2 * (hidden_size * (4 * hidden_size) + hidden_size + 4*hidden_size)
    mlp_params = mlp_ln + mlp_projs

    final_ln = hidden_size

    embed_params = vocab_size * hidden_size
    unembed_params = hidden_size * vocab_size

    params = num_layers * (attn_params + mlp_params) + final_ln

    if include_embed_params:
        params += embed_params + unembed_params

    return params

def flops_per_step_from_config(config: dict, include_embed_flops=True):
    return flops_per_step(
        num_layers=config["num_layers"],
        hidden_size=config["hidden_size"],
        num_attn_heads=config.get("num_attn_heads", config["hidden_size"] // config.get("head_dim", 64)),
        seq_length=config["max_seq_len"],
        batch_size=config["batch_size"],
        vocab_size=config.get("vocab_size", 131072),
        include_embed_flops=include_embed_flops
    )


def load_runs(
    filters: dict,
    order=None,
    history_keys=["train/loss", "train/elbo", "train/visited_tokens"],
    add_flops=True,
    add_params=True,
    fit_loss_curves=True,
    drop_duplicates=None,
    min_history_steps=2000,
    remove_outliers=False,
    remove_nans=True,
):
    api = wandb.Api()
    runs = api.runs(
        path=WANDB_PROJECT,
        filters=filters,
        order=order,
        per_page=1000,
    )

    def _load_single_run(run):
        # filter out runs that don't have (long enough) histories
        if "train/train_step" not in run.summary or run.summary["train/train_step"] < min_history_steps:
            print(f"Skipping run {run.name} with insufficient steps: {run.summary.get('_step')}")
            return None
        config = run.config
        config["name"] = run.name
        config["id"] = run.id
        config["state"] = run.state
        config["total_steps"] = run.summary.get("train/train_step", 0)
        h: pd.DataFrame = run.history(
            keys=history_keys,
            samples=10_000_000,
        )

        assert isinstance(h, pd.DataFrame)
        loss = h["train/elbo"].infer_objects(copy=False).astype(np.float32)
        if remove_outliers and (loss.diff() > 2.0).any():
            print(f"Removing outlier run {run.name}")
            return None
        if remove_nans and loss.isnull().any():
            print(f"Removing NaN run {run.name}")
            return None

        h["train/elbo_ema"] = loss_curve_ema(h)
        config["final_elbo"] = h["train/elbo_ema"].iloc[-1]
        if add_flops:
            config["flops_per_step"] = flops_per_step_from_config(config)
            config["non_emb_flops_per_step"] = flops_per_step_from_config(config, include_embed_flops=False)
            config["total_flops"] = config["flops_per_step"] * config["total_steps"]
            config["total_non_emb_flops"] = config["non_emb_flops_per_step"] * config["total_steps"]
            config["flops_per_token"] = config["flops_per_step"] / (config["batch_size"] * config["max_seq_len"])
            config["non_emb_flops_per_token"] = config["non_emb_flops_per_step"] / (config["batch_size"] * config["max_seq_len"])
        if add_params:
            config["params"] = params_from_config(config)
            config["non_emb_params"] = params_from_config(config, include_embed_params=False)
        if fit_loss_curves:
            try:
                y_hat, popt, r2 = fit_loss_curve(h)
                h["train/elbo_fit"] = y_hat
                config["fit_C"] = popt[0]
                config["fit_B"] = popt[1]
                config["fit_b"] = popt[2]
                config["fit_r2"] = r2
            except Exception as e:
                print(f"Failed to fit loss curve for run {run.name}: {e}")
        return run, h, config
    
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(_load_single_run, run) for run in runs]
        results = []
        for future in tqdm.tqdm(as_completed(futures), total=len(futures), desc="Processing runs"):
            result = future.result()
            if result is not None:
                results.append(result)
    runs, histories, config_dicts = zip(*results) if results else ([], [], [])

    config_df = pd.DataFrame(config_dicts)
    config_df = config_df.sort_values(["final_elbo", "total_steps"], ascending=[True, False])

    # deduplicate runs
    if drop_duplicates:
        config_df = config_df.drop_duplicates(subset=drop_duplicates, keep="first")
        histories = [histories[i] for i in config_df.index]
        runs = [runs[i] for i in config_df.index]
        config_df = config_df.reset_index(drop=True)

    return runs, histories, config_df


def deduplicate_runs(
    filters,
    deduplicate_on="name",
    deduplicate_running=False,
    deduplicate_crashed=True,
    dry_run=True,
):
    api = wandb.Api()
    runs = api.runs(
        path=WANDB_PROJECT,
        filters=filters,
    )

    if not deduplicate_running:
        runs = [r for r in runs if r.state != "running"]
    if not deduplicate_crashed:
        runs = [r for r in runs if r.state != "crashed"]

    if not all(hasattr(r, deduplicate_on) for r in runs):
        raise ValueError(f"Not all runs have attribute '{deduplicate_on}'")

    # sort runs by finished -> running -> crashed
    runs = sorted(runs, key=lambda r: (r.state == "finished", r.state == "running", r.state == "crashed"), reverse=True)

    preserved_runs = {}
    duplicates = []
    for run in runs:
        run_id = getattr(run, deduplicate_on)
        if run_id not in preserved_runs:
            preserved_runs[run_id] = run
        else:
            duplicates.append(run)

    if dry_run:
        num_finished_keep = len([r for r in preserved_runs.values() if r.state == "finished"])
        num_running_keep = len([r for r in preserved_runs.values() if r.state == "running"])
        num_crashed_keep = len([r for r in preserved_runs.values() if r.state == "crashed"])
        num_finished_del = len([r for r in duplicates if r.state == "finished"])
        num_running_del = len([r for r in duplicates if r.state == "running"])
        num_crashed_del = len([r for r in duplicates if r.state == "crashed"])
        print("Dry run: not deleting any runs")
        print(f"- Will keep {len(preserved_runs)} runs ({num_finished_keep} finished, {num_running_keep} running, {num_crashed_keep} crashed)")
        print(f"- Will delete {len(duplicates)} runs ({num_finished_del} finished, {num_running_del} running, {num_crashed_del} crashed)")
        for run in duplicates:
            print(f"will delete {run} (duplicate group: {deduplicate_on}={getattr(run, deduplicate_on)})")
    else:
        for run in tqdm.tqdm(duplicates, desc="Deleting duplicates"):
            run.delete()
