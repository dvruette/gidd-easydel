# Copyright 2023 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import typing as tp

from eformer.common_types import ColumnWise, Replicated, RowWise
from jax.sharding import PartitionSpec

from easydel.infra.base_module import EasyDeLBaseConfig
from easydel.infra.etils import EasyDeLGradientCheckPointers
from easydel.infra.factory import register_config


@register_config("gidd")
class GiddConfig(EasyDeLBaseConfig):
    """
    Configuration objects inherit from [`EasyDeLBaseConfig`] and can be used to control the model outputs. Read
    the documentation from [`EasyDeLBaseConfig`] for more information.

    Args:
        vocab_size (`int`, *optional*, defaults to 131072):
            Vocabulary size of the `Gidd` model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed to the forward method.
        hidden_size (`int`, *optional*, defaults to 4096):
            Dimensionality of the encoder layers and the pooler layer.
        intermediate_size (`int`, *optional*, defaults to 11008):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        num_hidden_layers (`int`, *optional*, defaults to 32):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 32):
            Number of attention heads for each attention layer in the Transformer encoder.
        number_rep_kv (`int`, *optional*, defaults to 1):
            Number of repetitions for the key and value vectors.
        num_key_value_heads (`int`, *optional*):
            Number of key and value heads for each attention layer in the Transformer encoder. Will default to
            `number_rep_kv * num_attention_heads` if not set.
        max_position_embeddings (`int`, *optional*, defaults to 2048):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 2048 or 4096).
            head_dim (`int`, *optional*):
                    head_dim for attention qkv.
        rms_norm_eps (`float`, *optional*, defaults to 1e-6):
            The epsilon used by the rms normalization layers.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        bos_token_id (`int`, *optional*, defaults to 0):
            The id of the *beginning-of-sequence* token.
        eos_token_id (`int`, *optional*, defaults to 1):
            The id of the *end-of-sequence* token.
        resid_pdrop (`float`, *optional*, defaults to 0.0):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        embd_pdrop (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the embeddings.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        rope_theta (`float`, *optional*, defaults to 10000.0):
            The theta value to use for rotary position embeddings.
        attention_bias (`bool`, *optional*, defaults to `False`):
            Whether to use attention bias.
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether to tie the weights of the input embeddings and the output embeddings.
        gradient_checkpointing (`str`, *optional*, defaults to `"nothing_saveable"`):
            The gradient checkpointing configuration.
        fcm_min_ratio (`float`, *optional*, defaults to -1):
            The minimum ratio for Flash Attention.
        fcm_max_ratio (`float`, *optional*, defaults to -1):
            The maximum ratio for Flash Attention.
        rope_scaling (`tp.Dict[str, tp.Union[str, float]]`, *optional*):
            The configuration for rope scaling.
        scan_mlp_chunk_size (`int`, *optional*, defaults to 1024):
            The chunk size to use when scanning the MLP.
        bits (`int`, *optional*):
            The number of bits to quantize the model to.
        pretraining_tp (`int`, *optional*, defaults to 1):
            The tensor parallelism degree used during pretraining.
        mlp_bias (`bool`, *optional*, defaults to `False`):
            Whether to use bias in the MLP.
        scan_layers (`bool`, *optional*, defaults to `False`):
            Whether to use the scan implementation for the layers.
    """

    model_type: str = "gidd"

    def __init__(
        self,
        vocab_size: int = 131072,
        hidden_size: int = 768,
        intermediate_size: int = 3072,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        head_dim: tp.Optional[int] = None,
        is_causal: bool = False,
        attn_soft_cap: float = 30.0,
        max_position_embeddings: int = 1024,
        resid_scale: float = 4.0,
        rms_norm_eps: float = 1e-6,
        use_qk_norm: bool = True,
        init_scale: float = 0.4,
        emb_init_scale: float = 0.1,
        head_init_scale: float = 0.0,
        weight_scaling: str = "fan_in",
        head_scaling: float = 1.0,
        bos_token_id: int = 0,
        eos_token_id: int = 1,
        rope_theta: float = 10000.0,
        tie_word_embeddings: bool = False,
        gradient_checkpointing: EasyDeLGradientCheckPointers = EasyDeLGradientCheckPointers.NONE,
        rope_scaling: tp.Dict[str, tp.Union[str, float]] = None,
        scan_mlp_chunk_size: int = 1024,
        bits: tp.Optional[int] = None,
        pretraining_tp: int = 1,
        attention_bias: bool = False,
        mlp_bias: bool = False,
        scan_layers: bool = False,
        **kwargs,
    ):
        super().__init__(
            tie_word_embeddings=tie_word_embeddings,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            scan_mlp_chunk_size=scan_mlp_chunk_size,
            bits=bits,
            **kwargs,
        )
        self.vocab_size = vocab_size

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.rope_theta = rope_theta
        self.num_attention_heads = num_attention_heads
        self.attn_soft_cap = attn_soft_cap
        self.is_causal = is_causal
        self.max_position_embeddings = max_position_embeddings
        self.resid_scale = resid_scale
        self.init_scale = init_scale
        self.emb_init_scale = emb_init_scale
        self.head_init_scale = head_init_scale
        self.weight_scaling = weight_scaling
        self.head_scaling = head_scaling
        self.rms_norm_eps = rms_norm_eps
        self.use_qk_norm = use_qk_norm
        self.pretraining_tp = pretraining_tp
        self.tie_word_embeddings = tie_word_embeddings
        self.gradient_checkpointing = gradient_checkpointing
        self.attention_bias = attention_bias
        self.mlp_bias = mlp_bias
        self.rope_scaling = rope_scaling
        self.bits = bits
        self.scan_layers = scan_layers
        self.head_dim = (
            head_dim if head_dim is not None else hidden_size // num_attention_heads
        )

    def get_partition_rules(self, *args, **kwargs):
        """
        Get the partition rules for the model.
        Returns:
            `tp.Tuple[tp.Tuple[str, PartitionSpec]]`: The partition rules.
        """
        pmag = self.partition_manager
        return (
            (r"embed_tokens/embedding", pmag.resolve(ColumnWise)),
            (r"self_attn/(q_proj|k_proj|v_proj)/kernel", pmag.resolve(ColumnWise)),
            (f"(k|v)_bias", pmag.resolve(Replicated)),
            (r"self_attn/o_proj/kernel", pmag.resolve(RowWise)),
            (r"self_attn/.*proj/bias", pmag.resolve(Replicated)),
            (r"mlp/(gate_proj|up_proj)/kernel", pmag.resolve(ColumnWise)),
            (r"mlp/down_proj/kernel", pmag.resolve(RowWise)),
            (r"mlp/.*proj/bias", pmag.resolve(Replicated)),
            (r".*(attn_layernorm|mlp_layernorm|norm)/kernel", pmag.resolve(Replicated)),
            (r"lm_head/kernel", pmag.resolve(ColumnWise)),
            (r"score/kernel", pmag.resolve(RowWise)),
            (r".*bias", pmag.resolve(Replicated)),
            (r".*", pmag.resolve(Replicated)),
        )

    @staticmethod
    def get_weight_decay_exclusions():
        return tuple()

    @staticmethod
    def rng_keys():
        return "params"

    @property
    def granted_freq_max_position_embedding(self) -> int:
        return getattr(
            self,
            "freq_max_position_embeddings",
            self.max_position_embeddings,
        )

    @property
    def granted_mask_max_position_embedding(self) -> int:
        return getattr(
            self,
            "mask_max_position_embeddings",
            self.max_position_embeddings,
        )
