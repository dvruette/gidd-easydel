
import easydel as ed

import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp
import flax.nnx as nn
from transformers import AutoTokenizer

from gidd_easydel.model import GiddForDiffusionLM, GiddConfig


# checkpoint_dir = "/pub/hofmann-scratch/dvruette/gidd-easydel/gidd-unif-3b/gidd/orbax"

def load_checkpoint(
    checkpoint_dir: str,
    num_layers: int,
    hidden_size: int,
    num_attn_heads: int,
    max_seq_len: int = 2048,
    resid_scale: float = 4.0,
    head_scale: float = 512.0,
    tokenizer_id: str = "dvruette/nemotron-cc-bpe",
    sharding_axis_dims: tuple[int, int, int, int, int] = (1, -1, 1, 1, 1),
    dtype: str = "bf16",
    attn_mechanism: str = "vanilla",
):
    dtype = {
        "fp32": jnp.float32,
        "bf16": jnp.bfloat16,
    }[dtype]

    assert hidden_size % num_attn_heads == 0, "Hidden size must be divisible by number of attention heads."
    head_dim = hidden_size // num_attn_heads

    init_scale = 0.4
    aux_init_scale = 0.02

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)

    model = GiddForDiffusionLM(
        config=GiddConfig(
            vocab_size=len(tokenizer),
            hidden_size=hidden_size,
            intermediate_size=4*hidden_size,
            num_hidden_layers=num_layers,
            num_attention_heads=hidden_size // head_dim,
            head_dim=head_dim,
            is_causal=False,
            max_position_embeddings=max_seq_len,
            resid_scale=resid_scale,
            init_scale=init_scale / hidden_size**0.5,
            emb_init_scale=aux_init_scale,
            head_init_scale=aux_init_scale,
            weight_scaling=1.0,
            head_scaling=head_scale / hidden_size,
            use_qk_norm=True,
            sharding_axis_dims=sharding_axis_dims,
            partition_axis=ed.PartitionAxis(),
            # gradient_checkpointing=args.gradient_checkpointing,
            attn_mechanism=attn_mechanism,
            attn_dtype=dtype,
            attention_bias=True,
            mlp_bias=True,
        ),
        dtype=dtype,
        param_dtype=dtype,
        precision=jax.lax.Precision.HIGH,
        rngs=ed.Rngs(0),
    ).shard_model()
    model_state = model.to_state()

    def restore_args_from_target(target):
        def leaf_to_restore_args(x):
            if isinstance(x, jax.Array):
                return ocp.ArrayRestoreArgs(dtype=x.dtype, sharding=x.sharding)
            return ocp.RestoreArgs()
        return jax.tree_util.tree_map(leaf_to_restore_args, target)
    

    print(checkpoint_dir)

    options = ocp.CheckpointManagerOptions(
        enable_hns=True if "_hns" in checkpoint_dir else False,
    )
    with ocp.CheckpointManager(
        checkpoint_dir,
        options=options,
    ) as checkpoint_manager:
        latest_step = checkpoint_manager.latest_step()

        array_restore_args = restore_args_from_target({"graphstate": model_state.graphstate})

        restore_args = ocp.args.PyTreeRestore(
            item={"graphstate": model_state.graphstate},
            restore_args=array_restore_args,
            partial_restore=True,
        )

        # Restore the state. Orbax will read the saved data and apply the sharding
        # from your target_state_template.
        restored_state = checkpoint_manager.restore(
            latest_step, 
            args=restore_args
        )
    
    model_state = model_state.replace(graphstate=restored_state["graphstate"])
    module = nn.merge(model_state.graphdef, model_state.graphstate, model_state.graphother)

    return model.mesh, module, tokenizer, dtype
