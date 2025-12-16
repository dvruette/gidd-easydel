import typing as tp

from transformers import PretrainedConfig


class GiddConfig(PretrainedConfig):
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
        rope_scaling: tp.Dict[str, tp.Union[str, float]] = None,
        attention_bias: bool = False,
        mlp_bias: bool = False,
        tie_word_embeddings: bool = False,
        attn_performer: str = "eager",
        noise_type: float = 0.0,
        min_log_snr: float = -9.0,
        max_log_snr: float = 9.0,
        **kwargs,
    ):
        super().__init__(
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
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
        self.attention_bias = attention_bias
        self.mlp_bias = mlp_bias
        self.rope_scaling = rope_scaling
        self.head_dim = (
            head_dim if head_dim is not None else hidden_size // num_attention_heads
        )
        self.tie_word_embeddings = tie_word_embeddings
        self.attn_performer = attn_performer
        self.noise_type = noise_type
        self.min_log_snr = min_log_snr
        self.max_log_snr = max_log_snr
