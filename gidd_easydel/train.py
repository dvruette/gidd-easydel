import easydel as ed

import os
import random
import typing as tp
from copy import deepcopy
from datetime import datetime

import dask.dataframe as dd
import jax
import optax
import orbax.checkpoint as ocp
import chex
import tqdm
import wandb
from eformer.loggings import get_logger
from eformer.paths import ePath
from jax import numpy as jnp
from transformers import AutoTokenizer
from datasets import IterableDataset

from .data2 import generate_packed_samples
from .diffusion_trainer import DiffusionTrainer, DiffusionConfig
from .model import GiddForDiffusionLM, GiddConfig
from .optimizer import lapropw


logger = get_logger(__name__)


def get_sharding_axis(strategy: str, batch_size: int, num_procs: int, num_devices: int):
    STRATS = {
        "fsdp": (1, -1, 1, 1, 1),
        "fsdp+tp": (1, num_procs, 1, -1, 1),  # FSDP across processes + TP across devices
        "dp": (-1, 1, 1, 1, 1),
        "tp": (1, 1, 1, -1, 1),
    }

    if "," in strategy:
        strategy = strategy.split(",")
        strategy = tuple(int(s) for s in strategy)
        assert len(strategy) == 5, f"Invalid sharding strategy: {strategy}"
        logger.info("Using sharding strategy: %s", strategy)
        return strategy

    if strategy == "auto":
        if batch_size % num_devices == 0:
            strategy = "fsdp"
        elif batch_size % num_procs == 0:
            strategy = "fsdp+tp"
        else:
            strategy = "tp"

    logger.info("Using sharding strategy: %s", strategy)
    if strategy not in STRATS:
        raise ValueError(f"Unknown sharding strategy: {strategy} (available strategies: {list(STRATS.keys())})")
    return STRATS[strategy]


def wsd_lr_schedule(total_steps: int, base_lr: float, warmup_steps: int = 0, cooldown_steps: int = 0) -> tp.Callable[[chex.Numeric], chex.Numeric]:
    """
    Implements a warmup-stable-decay learning rate schedule.
    
    Args:
        base_lr (float): Base learning rate.
        warmup_steps (int): Number of steps for warmup.
        cooldown_steps (int): Number of steps for decay.
        total_steps (int): Total number of training steps.
        curr_step (int): Current training step.
    
    Returns:
        float: Adjusted learning rate for the current step.
    """

    def sqrt1m_schedule(init_value: float, decay_steps: int) -> tp.Callable[[chex.Numeric], chex.Numeric]:
        def schedule(count: chex.Numeric) -> chex.Numeric:
            count = jnp.minimum(count, decay_steps)
            return init_value * (1 - (count / max(1, decay_steps))**0.5)
        return schedule

    return optax.schedules.join_schedules([
        optax.schedules.linear_schedule(0, base_lr, warmup_steps),
        optax.schedules.constant_schedule(base_lr),
        sqrt1m_schedule(base_lr, cooldown_steps),
    ], [warmup_steps, total_steps - cooldown_steps])



def get_latest_checkpoint(checkpoint_dir):
    save_path = ePath(checkpoint_dir)
    checkpoint_files = list(save_path.glob("*"))
    assert len(checkpoint_files) > 0, f"No checkpoints found in {checkpoint_dir}"

    def get_mtime(path):
        try:
            return path.stat().get("mtime", 0)
        except Exception:
            return 0
        
    checkpoint_files.sort(key=get_mtime)

    latest_ckpt = checkpoint_files[-1]
    step = int(latest_ckpt.stem().split("-")[-1])

    return latest_ckpt, step


def train(args):
    num_procs = jax.process_count()
    num_local_devices = jax.local_device_count()
    num_devices = jax.device_count()

    logger.info("Process count: %d, local device count: %d, process index: %d, global device count: %d",
                num_procs, num_local_devices, jax.process_index(), num_devices)

    dtype = {
        "fp32": jnp.float32,
        "bf16": jnp.bfloat16,
    }[args.dtype]

    # --- Basic Training Parameters ---
    seed = args.seed
    random.seed(seed)

    max_length = args.max_seq_len
    total_batch_size = args.batch_size
    micro_batch_size = args.micro_batch_size
    if micro_batch_size is None or micro_batch_size <= 0:
        micro_batch_size = total_batch_size
    assert total_batch_size % micro_batch_size == 0, "Total batch size must be divisible by micro batch size."
    grad_accum_steps = total_batch_size // micro_batch_size

    num_layers = args.num_layers
    hidden_size = args.hidden_size
    head_dim = args.head_dim
    if args.num_attn_heads and args.num_attn_heads > 0:
        assert hidden_size % args.num_attn_heads == 0, "Hidden size must be divisible by number of attention heads."
        head_dim = hidden_size // args.num_attn_heads
    else:
        assert hidden_size % head_dim == 0, "Hidden size must be divisible by head dimension."
        args.num_attn_heads = hidden_size // head_dim

    lr = args.lr  # 0.5
    aux_lr = lr * args.aux_lr_factor  # 1e-3
    init_scale = args.init_scale  # 0.4
    resid_scale = args.resid_scale  # 4
    aux_init_scale = args.aux_init_scale  # 0.02
    weight_decay = args.weight_decay  # 1e-4
    ln_wd = weight_decay if args.ln_wd is None else args.ln_wd
    bias_wd = args.bias_wd  # 0.0
    head_scale = args.head_scale  # 512
    adam_eps = args.adam_eps  # 1e-8

    total_steps = args.max_training_steps  # 100000
    warmup_steps = args.warmup_steps  # 2000
    if args.cooldown_steps < 1.0:
        cooldown_steps = int(args.max_training_steps * args.cooldown_steps)
    else:
        cooldown_steps = int(args.cooldown_steps)

    optimizer_fn = {
        "laprop": lapropw,
        "adam": optax.adamw,
    }[args.optimizer]

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_id)

    # --- Model Definition ---

    args.sharding_axis_dims = get_sharding_axis(args.sharding, total_batch_size, num_procs, num_devices)
    args.sharding_dcn_axis_dims = None
    if args.dcn_sharding is not None:
        args.sharding_dcn_axis_dims = get_sharding_axis(args.dcn_sharding, total_batch_size, num_procs, num_devices)

    if args.resume_wandb_id:
        run = wandb.Api().run(f"EasyDeL-diffusiontrainer-Gidd/{args.resume_wandb_id}")
        args.save_directory = run.config["save_directory"]
        logger.info(f"Resuming W&B run {args.resume_wandb_id} with save directory {args.save_directory}")
        # try:
        #     checkpoint_path, start_step = get_latest_checkpoint(os.path.join(args.save_directory, "gidd", "orbax"))
        #     logger.info(f"Resuming from checkpoint: {checkpoint_path}")
        #     has_checkpoint = True
        # except Exception as e:
        #     logger.warning(f"Failed to resume from checkpoint: {e}")
        #     # args.resume_wandb_id = None
    elif args.resume_from_checkpoint:
        args.save_directory = args.resume_from_checkpoint
    else:
        args.save_directory = os.path.join(
            args.save_directory,
            datetime.now().strftime("%Y-%m-%d"),
            args.wandb_tags,
            args.wandb_name,
            datetime.now().strftime("%H-%M-%S"),
        )

    logger.info(f"Saving models to {args.save_directory}")

    if os.getenv("SKIP_GS_LOCATION_CHECK", "0") != "1" and any("gs://" in p for p in [args.data_files, args.save_directory]):
        zone = os.getenv("TPU_ZONE", None)
        assert zone is not None, "TPU_ZONE environment variable must be set"
        location = zone[:-2]
        if args.data_files.startswith("gs://"):
            assert location in args.data_files, f"Data files {args.data_files} must be in the same location as the TPU zone {zone}"
        if args.save_directory.startswith("gs://"):
            assert location in args.save_directory, f"Save directory {args.save_directory} must be in the same location as the TPU zone {zone}"

    # if has_checkpoint:
    #     resume_if_possible = True
    #     model_state = ed.EasyDeLState.load_state(
    #         checkpoint_path,
    #         dtype=dtype,
    #         param_dtype=dtype,
    #         precision=jax.lax.Precision.HIGH,
    #         sharding_axis_dims=args.sharding_axis_dims,
    #         sharding_dcn_axis_dims=args.sharding_dcn_axis_dims,
    #         partition_axis=ed.PartitionAxis(),
    #     )
    #     # model_state = model_state.replace(step=jnp.asarray(0))
    # else:
    resume_if_possible = False
    model = GiddForDiffusionLM(
        config=GiddConfig(
            vocab_size=len(tokenizer),
            hidden_size=hidden_size,
            intermediate_size=4*hidden_size,
            num_hidden_layers=num_layers,
            num_attention_heads=hidden_size // head_dim,
            head_dim=head_dim,
            is_causal=False,
            max_position_embeddings=max_length,
            resid_scale=resid_scale,
            init_scale=init_scale / hidden_size**0.5,
            emb_init_scale=aux_init_scale,
            head_init_scale=0.0 if args.zero_head_init else aux_init_scale,
            weight_scaling=1.0,
            head_scaling=head_scale / hidden_size,
            use_qk_norm=True,
            sharding_axis_dims=args.sharding_axis_dims,
            sharding_dcn_axis_dims=args.sharding_dcn_axis_dims,
            partition_axis=ed.PartitionAxis(),
            gradient_checkpointing=args.gradient_checkpointing,
            attn_mechanism=args.attn_mechanism,
            attn_dtype=dtype,
            attention_bias=args.attn_bias,
            mlp_bias=True,
            # scan_layers=True,
        ),
        dtype=dtype,
        param_dtype=dtype,
        precision=jax.lax.Precision.HIGH,
        rngs=ed.Rngs(0),
    ).shard_model()  # Shard the newly created model across devices.
    model_state = model.to_state()


    class CustomDiffusionConfig(DiffusionConfig):
        # Hacky: override the `get_optimizer_and_scheduler` method to implement per-layer learning rates
        def get_optimizer_and_scheduler(self, steps):
            optimizer_kwargs = deepcopy(self.optimizer_kwargs)
            clip_grad = optimizer_kwargs.pop("clip_grad", None)

            bulk_schedule = wsd_lr_schedule(
                total_steps=total_steps,
                base_lr=lr / hidden_size,
                warmup_steps=warmup_steps,
                cooldown_steps=cooldown_steps,
            )
            aux_schedule = wsd_lr_schedule(
                total_steps=total_steps,
                base_lr=aux_lr,
                warmup_steps=warmup_steps,
                cooldown_steps=cooldown_steps,
            )

            def param_label_fn(params: tp.Any) -> str:
                def label_leaf(path: str, param: chex.Array) -> str:
                    path = ''.join(str(k) for k in path)

                    if "norm" in path:
                        return "ln_params"
                    elif "embed_tokens" in path:
                        return "emb_unemb_params"
                    elif "bias" in path:
                        return "bias_params"
                    elif "lm_head" in path:
                        return "emb_unemb_params"
                    elif param.ndim > 1:
                        return "bulk_params"
                    else:
                        raise ValueError(f"Unknown parameter type: {path}")

                labels = jax.tree.map_with_path(label_leaf, params)
                return labels


            opt_kwargs = dict(b1=args.beta1, b2=args.beta2, eps=adam_eps / hidden_size / num_layers)
            optimizer = optax.multi_transform({
                "bulk_params": optimizer_fn(learning_rate=bulk_schedule, weight_decay=weight_decay * hidden_size, **opt_kwargs),
                "ln_params": optimizer_fn(learning_rate=aux_schedule, weight_decay=ln_wd, **opt_kwargs),
                "bias_params": optimizer_fn(learning_rate=aux_schedule, weight_decay=bias_wd, **opt_kwargs),
                "emb_unemb_params": optimizer_fn(learning_rate=aux_schedule, weight_decay=0.0, **opt_kwargs),
            }, param_label_fn)

            if clip_grad:
                tx = optax.chain(
                    optax.clip_by_global_norm(clip_grad),
                    optimizer,
                )
            else:
                tx = optimizer

            if optimizer_kwargs.get("gradient_accumulation_steps", 0) > 1:
                tx = optax.MultiSteps(tx, optimizer_kwargs["gradient_accumulation_steps"])
            
            # the LR schedule returned here is only used for logging purposes
            return optimizer, bulk_schedule

    # --- Configuration ---
    arguments = CustomDiffusionConfig(
        ## Diffusion arguments
        loss_aggregation=args.loss_aggregation,
        loss_scale=args.loss_scale,
        beta_is_div=args.beta_is_div,
        noise_p_independent=args.max_empty_token_frac,  # 0.0,
        noise_p_linear=args.noise_p_linear,
        noise_mask_p_prompt=args.noise_mask_p_prompt,
        noise_mask_p_infilling=args.noise_mask_p_infilling,
        noise_mask_max_cond_frac=1.0,
        mixing_rate="linear",
        min_log_snr=args.min_log_snr,
        max_log_snr=args.max_log_snr,
        hybrid_mixing_scale=args.hybrid_mixing_scale,
        hybrid_mixing_shift=args.hybrid_mixing_shift,
        causal_prompt_attention=args.causal_prompt_attn,
        ## Trainer arguments
        model_name="gidd",  # for wandb project name
        use_wandb=True,
        wandb_entity=args.wandb_entity,
        wandb_kwargs={
            "name": args.wandb_name,
            "tags": args.wandb_tags.split(",") if args.wandb_tags else None,
            "id": args.resume_wandb_id or None,
            "resume": "must" if args.resume_wandb_id else None,
            # "resume_from": f"{args.resume_wandb_id}?_step={start_step}" if args.resume_wandb_id else None
        },
        num_train_epochs=1,
        total_batch_size=micro_batch_size,  # easydel is weird like that
        gradient_accumulation_steps=grad_accum_steps,
        do_last_save=True,
        max_sequence_length=max_length,
        # This is MANDATORY for streaming datasets. It tells the trainer how many
        # steps constitute one "epoch". Should be ~ (total_dataset_size // total_batch_size).
        per_epoch_training_steps=98_000_000,
        max_training_steps=total_steps * grad_accum_steps,
        # step_start_point=start_step,
        learning_rate=lr / hidden_size,
        optimizer=ed.EasyDeLOptimizers.ADAMW,
        scheduler=ed.EasyDeLSchedulers.COSINE,
        warmup_steps=args.warmup_steps,
        weight_decay=0.0,
        loss_config=ed.LossConfig(break_on_nan=args.break_on_nan),
        save_directory=args.save_directory,
        save_steps=args.save_steps,
        save_total_limit=(
            args.save_total_limit
            if args.save_total_limit is not None and args.save_total_limit >= 0
            else None
        ),
        save_optimizer_state=True,
        resume_if_possible=False,
        clip_grad=1.0,
        report_steps=50,
        log_steps=100,
        metrics_aggregation="mean",
        # progress_bar_type="json",
        track_memory=args.track_memory,
        use_grain=False,
        weight_distribution_log_steps=args.weight_distribution_log_steps,
        log_grad_norms=args.log_grad_norms,
    )

    # jax.experimental.multihost_utils.sync_global_devices("gidd_easydel:before_load_dataset")

    bucket_paths = sorted(str(p) for p in ePath(args.data_files).glob("bucket=*"))
    bucket_files = [
        sorted(str(p) for p in ePath(bp).glob("*.parquet"))
        for bp in bucket_paths
    ]

    storage_options = None
    if args.data_files.startswith("gs://"):
        bucket_paths = [["simplecache::" + p for p in ps] for ps in bucket_files]
        storage_options = {
            "gcs": {"token": "cloud"},
            "simplecache": {
                "cache_storage": "/dev/shm/fsspec-cache",
            },
        }

    print(f"Found {len(bucket_paths)} buckets and {len(bucket_files)} files in {args.data_files}")


    checkpoint_dir = str(arguments.get_path().resolve() / "orbax")
    options = ocp.CheckpointManagerOptions(
        enable_hns=True if "_hns" in checkpoint_dir else False,
        create=True,
    )
    with ocp.CheckpointManager(
        checkpoint_dir,
        options=options,
    ) as checkpoint_manager:
        latest_step = checkpoint_manager.latest_step() or 0

    train_dataset = IterableDataset.from_generator(generate_packed_samples, gen_kwargs=dict(
        bucket_files=bucket_files,
        seed=random.randint(0, 2**32 - 1),
        storage_options=storage_options,
        preload_factor=1,
        tokens_field="tokens",
        max_sequence_length=max_length,
        batch_size=micro_batch_size,
        append_eos_token=True,
        eos_token_id=tokenizer.eos_token_id,
        state_save_interval=args.save_steps * micro_batch_size * grad_accum_steps,
        state_save_dir=os.path.join(args.save_directory, "dataset_states"),
        skip_first_n_batches=latest_step * grad_accum_steps,
    ))

    jax.experimental.multihost_utils.sync_global_devices("gidd_easydel:after_load_dataset")
    logger.info("Loaded dataset on all hosts")

    # train_dataset = load_dataset(
    #     "parquet",
    #     data_files=args.data_files,
    #     split="train",
    #     streaming=True,
    # )

    # train_dataset = train_dataset.shuffle(seed=random.randint(0, 2**32 - 1))


    # --- Trainer Setup and Execution ---
    trainer = DiffusionTrainer(
        arguments=arguments,
        model_state=model_state,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=None,
        seed=seed,
        dtype=dtype,
    )

    if trainer.arguments.can_log_metrics:
        wandb.config.update(vars(args), allow_val_change=True)
        wandb.config.update({
            "tpu_name": os.getenv("TPU_NAME"),
            "tpu_version": os.getenv("TPU_VERSION"),
            "tpu_zone": os.getenv("TPU_ZONE"),
            "tpu_pod_count": os.getenv("TPU_POD_COUNT"),
            "process_count": num_procs,
            "num_local_devices": num_local_devices,
            "num_devices": num_devices,
        }, allow_val_change=True)

    if args.compile_aot:
        logger.info("Compiling ahead of time...")
        trainer.compile_aot()

    if trainer.memory_monitor is not None:
        trainer.memory_monitor.start_monitoring()

    jax.experimental.multihost_utils.sync_global_devices("gidd_easydel:before_training")
    logger.info("Starting training...")
    trainer.train()
