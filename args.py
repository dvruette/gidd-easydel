import sys
import argparse


def parse_args(
    default_save_directory="outputs/diffusion_trainer",
    default_wandb_entity="dvruette",
    default_data_files="/local/home/dvruette/nemotron_tokenized/",
):
    command = sys.argv
    if "--resume_wandb_id" in command:
        # remove the wandb_id and subsequent argument from the command
        idx = command.index("--resume_wandb_id")
        command = command[:idx] + command[idx+2:]
    command = " ".join(command)

    parser = argparse.ArgumentParser(description="Run the diffusion training process.")
    # architecture
    parser.add_argument("--max_seq_len", type=int, default=512, help="Maximum sequence length for the model.")
    parser.add_argument("--batch_size", type=int, default=8, help="Total batch size for training.")
    parser.add_argument("--micro_batch_size", type=int, default=None, help="Micro batch size for gradient accumulation. Default: Same as batch size.")
    parser.add_argument("--num_layers", type=int, default=8, help="Number of layers in the model.")
    parser.add_argument("--hidden_size", type=int, default=512, help="Hidden size of the model.")
    parser.add_argument("--head_dim", type=int, default=64, help="Dimension of each attention head.")
    parser.add_argument("--num_attn_heads", type=int, default=None, help="Number of attention heads. `--head_dim` is ignored if this is provided.")
    parser.add_argument("--attn_mechanism", type=str, default="vanilla", choices=["auto", "vanilla", "sdpa", "flash_attn2", "ring", "splash", "cudnn", "blockwise", "cuda_flash_attn2", "paged_attention"], help="Attention mechanism to use.")
    parser.add_argument("--causal_prompt_attn", action=argparse.BooleanOptionalAction, default=False, help="Use causal prompt attention.")
    parser.add_argument("--attn_bias", action=argparse.BooleanOptionalAction, default=True, help="Use attention bias in the model.")
    # training schedule
    parser.add_argument("--max_training_steps", type=int, default=100_000, help="Maximum number of training steps.")
    parser.add_argument("--warmup_steps", type=int, default=2000, help="Number of warmup steps for learning rate scheduling.")
    parser.add_argument("--cooldown_steps", type=float, default=0.1, help="Number or fraction of steps for learning rate cooldown.")
    # init
    parser.add_argument("--init_scale", type=float, default=0.4, help="Initial scale for model parameters.")
    parser.add_argument("--aux_init_scale", type=float, default=0.02, help="Initial scale for embedding parameters.")
    parser.add_argument("--zero_head_init", action=argparse.BooleanOptionalAction, default=True, help="Use zero initialization for the output head.")
    parser.add_argument("--resid_scale", type=float, default=4.0, help="Scale for residual connections.")
    parser.add_argument("--head_scale", type=float, default=512, help="Scale for output layer.")
    # optimizer
    parser.add_argument("--loss_aggregation", type=str, default="sum", choices=["mean", "sum"], help="Loss aggregation method.")
    parser.add_argument("--loss_scale", type=float, default=1.0, help="Loss scaling factor.")
    parser.add_argument("--optimizer", type=str, default="laprop", choices=["laprop", "adam"], help="Optimizer to use.")
    parser.add_argument("--lr", type=float, default=0.5, help="Learning rate for the optimizer.")
    parser.add_argument("--aux_lr_factor", type=float, default=0.02, help="Auxiliary learning rate for the optimizer (as a fraction of the main learning rate).")
    parser.add_argument("--beta1", type=float, default=0.9, help="Beta1 parameter for the optimizer.")
    parser.add_argument("--beta2", type=float, default=0.99, help="Beta2 parameter for the optimizer.")
    parser.add_argument("--adam_eps", type=float, default=1e-8, help="Epsilon for Adam/Laprop optimizer.")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay for the optimizer.")
    parser.add_argument("--ln_wd", type=float, default=0.0, help="Weight decay for layer normalization parameters.")
    parser.add_argument("--bias_wd", type=float, default=0.0, help="Weight decay for bias parameters.")
    # mixing schedule & diffusion loss
    parser.add_argument("--beta_is_div", type=float, default=1.0, help="Weight of the IS-divergence loss.")
    parser.add_argument("--max_empty_token_frac", type=float, default=0.2, help="Maximum fraction of empty tokens.")
    parser.add_argument("--noise_p_independent", type=float, default=0.5, help="Probability that SNR is sampled independently for each token.")
    parser.add_argument("--noise_p_linear", type=float, default=0.0, help="Probability that SNR is sampled linearly for each token.")
    parser.add_argument("--noise_mask_p_prompt", type=float, default=0.2, help="Probability that a sample has a noise-free prompt.")
    parser.add_argument("--noise_mask_p_infilling", type=float, default=0.0, help="Probability that a sample has a noise-free infilling.")
    parser.add_argument("--min_log_snr", type=float, default=-9.0, help="Minimum log SNR.")
    parser.add_argument("--max_log_snr", type=float, default=9.0, help="Maximum log SNR.")
    parser.add_argument("--hybrid_mixing_scale", type=float, default=1.0, help="Scale for hybrid mixing schedule.")
    parser.add_argument("--hybrid_mixing_shift", type=float, default=0.0, help="Shift for hybrid mixing schedule.")
    # checkpointing
    parser.add_argument("--save_directory", type=str, default=default_save_directory, help="Directory to save model checkpoints.")
    parser.add_argument("--save_steps", type=int, default=1000, help="Number of steps between saving model checkpoints.")
    parser.add_argument("--save_total_limit", type=int, default=3, help="Maximum number of checkpoints to keep. (default: keep only most recent)")
    # wandb
    parser.add_argument("--wandb_entity", type=str, default=default_wandb_entity, help="Weights & Biases entity for logging.")
    parser.add_argument("--wandb_name", type=str, default=None, help="Weights & Biases run name.")
    parser.add_argument("--wandb_tags", type=str, default=None, help="Weights & Biases tags for the run.")
    parser.add_argument("--resume_wandb_id", type=str, default=None, help="Weights & Biases run ID to resume from.")
    # data
    parser.add_argument("--tokenizer_id", type=str, default="dvruette/nemotron-cc-bpe", help="Tokenizer ID for the model.")
    parser.add_argument("--data_files", type=str, default=default_data_files, help="Path to training data files.")
    parser.add_argument("--sampler", type=str, default="buckets", choices=["buckets", "buffered", "simple"], help="Sampler to use.")
    # others
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility.")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["fp32", "bf16"], help="Activation data type.")
    parser.add_argument("--sharding", type=str, default="fsdp", help="Sharding strategy to use.")
    parser.add_argument("--dcn_sharding", type=str, default=None, help="Sharding strategy for the DCN layers. If None, EasyDeL default.")
    parser.add_argument("--compile_aot", action=argparse.BooleanOptionalAction, default=False, help="Compile model ahead of time for faster training.")
    parser.add_argument("--break_on_nan", action=argparse.BooleanOptionalAction, default=True, help="Break training on NaN loss.")
    parser.add_argument("--gradient_checkpointing", type=str, default="", choices=["", "everything_saveable", "nothing_saveable", "checkpoint_dots", "checkpoint_dots_with_no_batch_dims"], help="Gradient checkpointing strategy.")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Path to a checkpoint to resume training from.")
    # logging
    parser.add_argument("--track_memory", type=float, default=None, help="Track memory usage during training.")
    parser.add_argument("--weight_distribution_log_steps", type=int, default=0, help="Log weight distribution every N steps. (set to 0 to disable)")
    parser.add_argument("--log_grad_norms", action=argparse.BooleanOptionalAction, default=False, help="Log gradient norms during training.")
    args = parser.parse_args()
    args.command = command
    return args
