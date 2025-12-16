import typing as tp
import warnings
from functools import partial
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn.attention.flex_attention import flex_attention
from transformers import PreTrainedModel
from transformers.cache_utils import Cache, DynamicCache
from transformers.generation.utils import GenerationMixin
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast

from .configuration_gidd import GiddConfig


@dataclass
class AttentionLayerOutput:
    hidden_states: torch.Tensor
    attentions: tp.Optional[torch.Tensor] = None
    past_key_values: tp.Optional[tp.List[tp.Tuple[torch.Tensor, torch.Tensor]]] = None

@dataclass
class DecoderLayerOutput:
    hidden_states: torch.Tensor
    attentions: tp.Optional[torch.Tensor] = None
    past_key_values: tp.Optional[tp.List[tp.Tuple[torch.Tensor, torch.Tensor]]] = None


def promote_dtype(args: tuple, *, dtype: torch.dtype | None = None) -> tuple:
    return tuple(
        torch.as_tensor(x, dtype=dtype) if x is not None else None
        for x in args
    )


class ScaledLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        scale: float | tp.Literal["fan_in", "fan_out"] = 1.0,
        use_bias: bool = True,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()

        if scale == "fan_in":
            scale = in_features**-0.5
        elif scale == "fan_out":
            scale = out_features**-0.5

        if scale != 1.0:
            def _scale_operator(x):
                return x * scale
        else:
            def _scale_operator(x):
                return x

        self._scale_operator = _scale_operator
        self.in_features = in_features
        self.out_features = out_features

        self.use_bias = use_bias

        weight_shape = (out_features, in_features)
        weight = torch.zeros(weight_shape, dtype=dtype)
        self.weight = nn.Parameter(weight)

        if use_bias:
            bias = torch.zeros((out_features,), dtype=dtype)
            self.bias = nn.Parameter(bias)
        else:
            self.bias = None

    def forward(
        self,
        inputs: torch.Tensor,
        w: torch.Tensor | None = None,
    ) -> torch.Tensor:
        dtype = inputs.dtype
        weight = self.weight if w is None else w
        bias = self.bias if self.use_bias else None

        if bias is not None:
            inputs, weight, bias = promote_dtype((inputs, weight, bias), dtype=dtype)
        else:
            inputs, weight = promote_dtype((inputs, weight), dtype=dtype)

        y = torch.matmul(
            inputs,
            weight.T,
        )

        y = self._scale_operator(y)

        if bias is not None:
            y = y + bias.reshape((1,) * (y.ndim - 1) + (-1,))

        return y


def _apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    is_neox_style: bool,
) -> torch.Tensor:
    cos = cos.unsqueeze(2).to(dtype=x.dtype)
    sin = sin.unsqueeze(2).to(dtype=x.dtype)
    assert sin.ndim == x.ndim
    if is_neox_style:
        x1, x2 = torch.chunk(x, 2, dim=-1)
    else:
        x1 = x[..., ::2]
        x2 = x[..., 1::2]

    o1 = x1 * cos - x2 * sin
    o2 = x2 * cos + x1 * sin

    if is_neox_style:
        return torch.cat((o1, o2), dim=-1)
    else:
        return torch.stack((o1, o2), dim=-1).reshape(x.shape)

def apply_basic_rope(
    query: torch.Tensor,
    key: torch.Tensor,
    positions: torch.Tensor,
    frequencies: torch.Tensor,
    rotary_dim: int,
    is_neox_style: bool,
    offsets: torch.Tensor | None = None,
    dtype: torch.dtype = torch.float32,
):
    if offsets is not None:
        positions = positions + offsets
    cos, sin = torch.chunk(frequencies[positions], 2, dim=-1)
    if rotary_dim != query.shape[-1]:
        query_rot = _apply_rotary_emb(query[..., :rotary_dim], cos, sin, is_neox_style)
        query = torch.cat((query_rot, query[..., rotary_dim:]), dim=-1)
        key_rot = _apply_rotary_emb(key[..., :rotary_dim], cos, sin, is_neox_style)
        key = torch.cat((key_rot, key[..., rotary_dim:]), dim=-1)
        return query.to(dtype), key.to(dtype), cos, sin
    else:
        query = _apply_rotary_emb(query, cos, sin, is_neox_style)
        key = _apply_rotary_emb(key, cos, sin, is_neox_style)
        return query.to(dtype), key.to(dtype), cos, sin

def compute_basic_frequencies(
    base: int,
    rotary_dim: int,
    max_position_embeddings: int,
):
    inv = 1.0 / torch.pow(
        base,
        torch.arange(0, rotary_dim, 2, dtype=torch.float32) / rotary_dim,
    )
    freqs = torch.einsum(
        "i,j->ij",
        torch.arange(max_position_embeddings, dtype=torch.float32),
        inv,
    )
    freqs = torch.cat([freqs.cos(), freqs.sin()], dim=-1)
    return freqs

class RotaryEmbedding(nn.Module):
    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: int,
        is_neox_style: bool,
        dtype: torch.dtype,
    ):
        super().__init__()
        self.head_size = head_size
        self.rotary_dim = rotary_dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.is_neox_style = is_neox_style
        self.dtype = dtype

    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        offsets: torch.Tensor | None = None,
        frequencies: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if frequencies is None:
            frequencies = compute_basic_frequencies(
                base=self.base,
                rotary_dim=self.rotary_dim,
                max_position_embeddings=self.max_position_embeddings,
            )
        if hasattr(frequencies, "value"):
            frequencies = frequencies.value
        return apply_basic_rope(
            query=query,
            key=key,
            positions=positions,
            frequencies=frequencies,
            rotary_dim=self.rotary_dim,
            is_neox_style=self.is_neox_style,
            offsets=offsets,
            dtype=self.dtype,
        )


class GiddRMSNorm(nn.Module):
    def __init__(
        self,
        config: GiddConfig,
        dtype=torch.float32,
    ):
        super().__init__()
        self.config = config
        self.epsilon = self.config.rms_norm_eps
        self.weight = nn.Parameter(torch.zeros(self.config.hidden_size, dtype=dtype))
        # self.bias = nn.Parameter(torch.zeros(self.config.hidden_size, dtype=dtype))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        dtype = hidden_states.dtype
        variance = hidden_states.to(torch.float32)
        variance = variance.pow(2.0)
        variance = variance.mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.epsilon)

        hidden_states = ((1 + self.weight) * hidden_states)
        return hidden_states.to(dtype)

ALL_LAYERNORM_LAYERS.append(GiddRMSNorm)


class GiddMLP(nn.Module):
    def __init__(
        self,
        config: GiddConfig,
        dtype=torch.float32,
    ):
        super().__init__()
        self.config = config
        self.dtype = dtype

        linear_class = partial(
            ScaledLinear,
            scale=config.weight_scaling,
            dtype=dtype,
            use_bias=self.config.mlp_bias,
        )
        self.up_proj = linear_class(config.hidden_size, config.intermediate_size)
        self.down_proj = linear_class(config.intermediate_size, config.hidden_size)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        h = self.up_proj(h)
        h = torch.relu(h) ** 2
        h = self.down_proj(h)
        return h


class FlexSoftcapAttention(nn.Module):
    def __init__(self, head_dim, n_heads, softmax_scale, soft_cap):
        super().__init__()
        self.d_model = head_dim * n_heads
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.scale = float(softmax_scale)
        self.soft_cap = float(soft_cap)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ):
        B, _, L = q.shape[:3]

        def score_mod(score, b, h, q_idx, kv_idx):
            soft_cap = self.soft_cap
            score = soft_cap * torch.tanh(score / soft_cap)
            keep = attention_mask[b, q_idx, kv_idx]
            return torch.where(keep, score, torch.finfo(score.dtype).min)

        out = flex_attention(
            q,
            k,
            v,
            score_mod=score_mod,
            scale=self.scale,
        )
        out = out.transpose(1, 2).contiguous().view(B, L, self.d_model)
        return out, None


class VanillaSoftcapAttention(nn.Module):
    def __init__(self, head_dim, n_heads, softmax_scale, soft_cap):
        super().__init__()
        self.d_model = head_dim * n_heads
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.scale = float(softmax_scale)
        self.soft_cap = float(soft_cap)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ):
        B, _, L = q.shape[:3]
        scores = torch.einsum(
            "bhqd,bhkd->bhqk",
            q * self.scale,
            k,
        )
        scores = self.soft_cap * torch.tanh(scores / self.soft_cap)
        if attention_mask is not None:
            scores = scores.masked_fill(~attention_mask.unsqueeze(1), torch.finfo(scores.dtype).min)
        probs = torch.softmax(scores.to(torch.float32), dim=-1).to(scores.dtype)
        out = torch.einsum(
            "bhqk,bhkd->bhqd",
            probs,
            v,
        )
        out = out.transpose(1, 2).contiguous().view(B, L, self.d_model)
        return out, probs


class GiddAttention(nn.Module):
    def __init__(
        self,
        config: GiddConfig,
        layer_idx: int,
        dtype=torch.float32,
    ):
        super().__init__()

        self.hidden_size = config.hidden_size
        head_dim = config.hidden_size // config.num_attention_heads
        self.head_dim = getattr(config, "head_dim", head_dim)
        self.num_attention_heads = self.hidden_size // self.head_dim
        self.is_causal = config.is_causal
        self.layer_idx = layer_idx

        self.use_qk_norm = config.use_qk_norm
        if self.use_qk_norm:
            self.q_norm = GiddRMSNorm(config, dtype=torch.float32)
            self.k_norm = GiddRMSNorm(config, dtype=torch.float32)
        else:
            self.q_norm = None
            self.k_norm = None

        self.attention_bias = config.attention_bias
        if self.attention_bias:
            self.k_bias = nn.Parameter(
                torch.zeros((self.num_attention_heads, self.head_dim), dtype=dtype),
            )
            self.v_bias = nn.Parameter(
                torch.zeros((self.num_attention_heads, self.head_dim), dtype=dtype),
            )
        else:
            self.k_bias = None
            self.v_bias = None

        linear_class = partial(
            ScaledLinear,
            scale=config.weight_scaling,
            dtype=dtype,
            use_bias=False,
        )
        self.q_proj = linear_class(
            self.hidden_size,
            self.num_attention_heads * self.head_dim,
        )
        self.k_proj = linear_class(
            self.hidden_size,
            self.num_attention_heads * self.head_dim,
        )
        self.v_proj = linear_class(
            self.hidden_size,
            self.num_attention_heads * self.head_dim,
        )
        self.o_proj = linear_class(
            self.num_attention_heads * self.head_dim,
            self.hidden_size,
        )

        self.rotary = RotaryEmbedding(
            head_size=self.head_dim,
            rotary_dim=self.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
            is_neox_style=True,
            dtype=dtype,
        )

        if config.attn_performer == "flex":
            self.attention_performer = FlexSoftcapAttention(
                head_dim=self.head_dim,
                n_heads=self.num_attention_heads,
                softmax_scale=self.head_dim**-0.5,
                soft_cap=config.attn_soft_cap,
            )
        elif config.attn_performer == "eager":
            self.attention_performer = VanillaSoftcapAttention(
                head_dim=self.head_dim,
                n_heads=self.num_attention_heads,
                softmax_scale=self.head_dim**-0.5,
                soft_cap=config.attn_soft_cap,
            )
        else:
            raise ValueError(f"Unknown attn_performer: {config.attn_performer}")

    def concatenate(
        self,
        *,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor,
        past_key_values: tp.Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ):
        assert query.shape[1] == key.shape[1], "Query and Key lengths must match for GIDD attention."
        if attention_mask is not None:
            if attention_mask.dtype != torch.bool:
                warnings.warn("attention_mask should be a boolean array", stacklevel=1)
                attention_mask = (attention_mask == 1)

        batch_size = query.shape[0]

        # shape of attention_mask: (batch_size, seq_len)
        # or (batch_size, query_len, kv_len)

        if attention_mask.ndim == 2:
            attention_mask = attention_mask.unsqueeze(1)
            attention_mask = attention_mask.expand(-1, query.shape[1], -1)
        elif attention_mask.ndim == 3:
            # already in correct shape
            pass

        if self.attention_bias:
            ones = torch.ones(
                attention_mask.shape[:2] + (1,),
                dtype=attention_mask.dtype,
                device=attention_mask.device,
            )
            attention_mask = torch.cat(
                [
                    ones,
                    attention_mask,
                ],
                dim=-1,
            )

        if past_key_values is not None:
            past_keys, past_values = past_key_values
            key = torch.cat([past_keys, key], dim=1)
            value = torch.cat([past_values, value], dim=1)
        elif self.attention_bias:
            n_heads = self.num_attention_heads
            bias_shape = (batch_size, 1, n_heads, self.head_dim)
            k_bias = self.k_bias.view(1, 1, n_heads, self.head_dim).expand(bias_shape)
            v_bias = self.v_bias.view(1, 1, n_heads, self.head_dim).expand(bias_shape)
            key = torch.cat([k_bias, key], dim=1)
            value = torch.cat([v_bias, value], dim=1)

        # shape of attention_mask: (batch_size, 1, query_len, kv_len + 1)
        return query, key, value, attention_mask, (key, value)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        past_key_values: tp.Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        frequencies: tp.Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> AttentionLayerOutput:
        batch_size, sequence_length = hidden_states.shape[:2]
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        if self.use_qk_norm:
            query_states = self.q_norm(query_states)
            key_states = self.k_norm(key_states)

        qshape = (
            batch_size,
            sequence_length,
            self.num_attention_heads,
            self.head_dim,
        )
        kv_shape = (
            batch_size,
            sequence_length,
            self.num_attention_heads,
            self.head_dim,
        )
        query_states = query_states.view(qshape)
        key_states = key_states.view(kv_shape)
        value_states = value_states.view(kv_shape)

        query_states, key_states, cos, sin = self.rotary(
            positions=position_ids,
            query=query_states,
            key=key_states,
            frequencies=frequencies,
        )

        (
            query_states,
            key_states,
            value_states,
            attention_mask,
            past_key_values,
        ) = self.concatenate(
            query=query_states,
            key=key_states,
            value=value_states,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
        )

        attention_out, attentions = self.attention_performer.forward(
            q=query_states.transpose(1, 2),
            k=key_states.transpose(1, 2),
            v=value_states.transpose(1, 2),
            attention_mask=attention_mask,
        )

        attn_output = self.o_proj(attention_out)

        return AttentionLayerOutput(
            hidden_states=attn_output,
            attentions=attentions if output_attentions else None,
            past_key_values=past_key_values,
        )


class GiddLayer(nn.Module):
    def __init__(
        self,
        config: GiddConfig,
        layer_idx: int,
        dtype=torch.float32,
        resid_scale: float = 1.0,
    ):
        super().__init__()
        self.config = config
        self.resid_scale = resid_scale
        self.layer_idx = layer_idx

        self.self_attn = GiddAttention(
            layer_idx=layer_idx,
            config=config,
            dtype=dtype,
        )

        self.mlp = GiddMLP(
            config=config,
            dtype=dtype,
        )
        self.attn_layernorm = GiddRMSNorm(
            config=config,
            dtype=torch.float32,
        )
        self.mlp_layernorm = GiddRMSNorm(
            config=config,
            dtype=torch.float32,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        past_key_values: tp.Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        frequencies: tp.Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> DecoderLayerOutput:
        attn_inputs = self.attn_layernorm(hidden_states)
        attn_outputs = self.self_attn(
            attn_inputs,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            frequencies=frequencies,
            output_attentions=output_attentions,
        )
        hidden_states = hidden_states + self.resid_scale * attn_outputs.hidden_states

        mlp_inputs = self.mlp_layernorm(hidden_states)
        mlp_output = self.mlp(mlp_inputs)
        hidden_states = hidden_states + self.resid_scale * mlp_output

        return DecoderLayerOutput(
            hidden_states=hidden_states,
            attentions=attn_outputs.attentions,
            past_key_values=attn_outputs.past_key_values,
        )
    

class GiddPreTrainedModel(PreTrainedModel):
    config_class = GiddConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = False
    _no_split_modules = ["GiddLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = False
    _supports_sdpa = False
    _supports_flex_attn = False
    _can_compile_fullgraph = False
    _supports_attention_backend = False
    _can_record_outputs = {
        "hidden_states": GiddLayer,
        "attentions": GiddAttention,
    }

    def _init_weights(self, module):
        super()._init_weights(module)
        nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)


class GiddModel(GiddPreTrainedModel):
    def __init__(
        self,
        config: GiddConfig,
    ):
        super().__init__(config=config)

        self.resid_scale = config.resid_scale / config.num_hidden_layers
        dtype = config.torch_dtype

        self.embed_tokens = nn.Embedding(
            num_embeddings=self.config.vocab_size,
            embedding_dim=self.config.hidden_size,
        )
        self.embed_tokens.weight.data = self.embed_tokens.weight.data.to(dtype)
        nn.init.normal_(self.embed_tokens.weight, mean=0.0, std=self.config.emb_init_scale)

        freqs = compute_basic_frequencies(
            base=config.rope_theta,
            rotary_dim=config.hidden_size // config.num_attention_heads,
            max_position_embeddings=config.max_position_embeddings,
        )
        self.frequencies = nn.Buffer(freqs, persistent=False)

        self.layers = nn.ModuleList(
            [
                GiddLayer(
                    config=config,
                    layer_idx=i,
                    resid_scale=self.resid_scale,
                    dtype=dtype,
                )
                for i in range(self.config.num_hidden_layers)
            ]
        )
        self.norm = GiddRMSNorm(
            config=config,
            dtype=torch.float32,
        )

    def forward(
        self,
        input_ids: tp.Optional[torch.Tensor] = None,
        inputs_embeds: tp.Optional[torch.Tensor] = None,
        attention_mask: tp.Optional[torch.Tensor] = None,
        position_ids: tp.Optional[torch.Tensor] = None,
        past_key_values: tp.Optional[list[tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
        cache_position: tp.Optional[torch.LongTensor] = None,
        output_attentions: tp.Optional[bool] = None,
        output_hidden_states: tp.Optional[bool] = None,
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids.to(torch.long))

        if use_cache and past_key_values is None:
            past_key_values = [None] * self.config.num_hidden_layers
        elif past_key_values is not None:
            past_key_values = list(past_key_values)

        if position_ids is None:
            past_seen_tokens = 0
            if past_key_values is not None and any(past_key_values):
                past_seen_tokens = [kv[0].shape[1] for kv in past_key_values if kv is not None][0]
            cache_position = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
            position_ids = cache_position.unsqueeze(0)

        batch_size, sequence_length, _ = inputs_embeds.shape

        assert sequence_length <= self.config.max_position_embeddings, (
            f"Maximum Position Embedding Reached ! (expected <= {self.config.max_position_embeddings} got {sequence_length})"
        )
        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, sequence_length),
                dtype=torch.bool,
                device=inputs_embeds.device,
            )
        else:
            if attention_mask.dtype != torch.bool:
                attention_mask = (attention_mask == 1)

        if position_ids is None:
            position_ids = torch.arange(
                inputs_embeds.shape[-2],
                dtype=torch.int32,
                device=inputs_embeds.device,
            )
            position_ids = position_ids.unsqueeze(0).expand(inputs_embeds.shape[:-1])

        hidden_states = inputs_embeds

        all_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None
        for idx, block in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = block(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                output_attentions=output_attentions,
                frequencies=self.frequencies,
                past_key_values=past_key_values[idx] if past_key_values is not None else None,
            )
            hidden_states = layer_outputs.hidden_states

            if output_attentions:
                all_attentions += (layer_outputs.attentions,)

            if use_cache:
                past_key_values[idx] = layer_outputs.past_key_values

        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            past_key_values=past_key_values,
        )


class GiddForDiffusionLM(GiddPreTrainedModel, GenerationMixin):
    def __init__(
        self,
        config: GiddConfig,
    ):
        super().__init__(config=config)

        self.model = GiddModel(config=config)

        self.lm_head = ScaledLinear(
            config.hidden_size,
            config.vocab_size,
            scale=config.head_scaling,
            dtype=config.torch_dtype,
            use_bias=False,
        )

    def forward(
        self,
        input_ids: tp.Optional[torch.Tensor] = None,
        inputs_embeds: tp.Optional[torch.Tensor] = None,
        attention_mask: tp.Optional[torch.Tensor] = None,
        position_ids: tp.Optional[torch.Tensor] = None,
        past_key_values: tp.Optional[list[tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
        output_attentions: tp.Optional[bool] = None,
        output_hidden_states: tp.Optional[bool] = None,
    ) -> CausalLMOutputWithPast:
        outputs = self.model(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            use_cache=use_cache,
        )

        hidden_states = outputs.last_hidden_state

        if self.config.tie_word_embeddings:
            logits = hidden_states @ self.model.embed_tokens.weight.t()
        else:
            logits = self.lm_head(hidden_states)

        return CausalLMOutputWithPast(
            loss=None,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            past_key_values=outputs.past_key_values,
        )
    
    def _sample_prior(self, shape: tuple[int, ...], device: torch.device, mask_token_id: int = 3) -> torch.Tensor:
        p_unif = torch.sigmoid(
            torch.ones(shape, device=device) * self.config.min_log_snr + self.config.noise_type
        )
        r = torch.rand(shape, device=device)
        unif = torch.randint(0, self.config.vocab_size, shape, device=device)
        samples = torch.where(r < p_unif, unif, mask_token_id)
        return samples
    
    def _probs_with_topk_topp(self, logits, temperature: float, top_p: float | None, top_k: int | None):
        if temperature == 0.0:
            probs = torch.zeros_like(logits)
            indices = torch.argmax(logits, dim=-1, keepdim=True)
            probs.scatter_(-1, indices, 1.0)
            return probs
        
        x = logits / temperature

        if top_k is not None and 0 < top_k < x.size(-1):
            kth = torch.topk(x, top_k, dim=-1).values[..., -1, None]
            x = torch.where(x < kth, torch.full_like(x, float("-inf")), x)

        if top_p is not None and 0.0 < top_p < 1.0:
            sorted_logits, sorted_idx = torch.sort(x, descending=True, dim=-1)
            sorted_probs = torch.softmax(sorted_logits, dim=-1)
            cumprobs = sorted_probs.cumsum(dim=-1)

            remove = cumprobs > top_p
            remove[..., 1:] = remove[..., :-1].clone()
            remove[..., 0] = False

            sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
            x = x.scatter(-1, sorted_idx, sorted_logits)

        probs = torch.softmax(x, dim=-1)

        return probs
    
    def _pi_lambda(self, log_snr, mask_token_id=3):
        unif_vec = torch.ones((self.config.vocab_size,), device=log_snr.device) / (self.config.vocab_size - 1)
        unif_vec[mask_token_id] = 0.0
        alpha = torch.sigmoid(log_snr + self.config.noise_type)
        pi = alpha * unif_vec
        pi[..., mask_token_id] = 1.0 - alpha
        return pi
    
    def _sample_ancestral(
        self,
        z: torch.Tensor,
        x_hat: torch.Tensor,
        log_snr_t: torch.Tensor,
        log_snr_s: torch.Tensor,
        mask_token_id: int = 3,
    ):
        alpha_s = log_snr_s.sigmoid()
        alpha_t = log_snr_t.sigmoid()
        beta_s, beta_t = 1.0 - alpha_s, 1.0 - alpha_t
        alpha_t_s = alpha_t / alpha_s

        pi_s = self._pi_lambda(log_snr_s, mask_token_id=mask_token_id)
        pi_t = self._pi_lambda(log_snr_t, mask_token_id=mask_token_id)
        beta_pi_t_s = beta_t * pi_t - alpha_t_s * beta_s * pi_s
        # beta_pi_t_s_at_z = beta_pi_t_s[z]

        q_t = alpha_t * x_hat + beta_t * pi_t[None, None, :]
        q_s = alpha_s * x_hat + beta_s * pi_s[None, None, :]
        q_t_at_z = q_t.gather(-1, z.unsqueeze(-1)).squeeze(-1)

        z_vec = torch.nn.functional.one_hot(z, num_classes=self.config.vocab_size).to(q_t.dtype)
        q_t_s_at_z = alpha_t_s * z_vec + beta_pi_t_s[z, None]

        p_s_t = q_s * q_t_s_at_z / q_t_at_z[..., None]

        z_next = torch.multinomial(p_s_t.flatten(0, 1), num_samples=1).view_as(z)
        return z_next

    def _sample_adaptive(
        self,
        z: torch.Tensor,
        logits: torch.Tensor,
        log_snr: torch.Tensor,
        n_tokens: int = 1,
        mask_token_id: int = 3,
        temperature: float = 0.0,
        top_p: float | None = None,
        top_k: int | None = None,
    ):
        pi_vec = self._pi_lambda(log_snr, mask_token_id=mask_token_id)
        p_noise = pi_vec[z]
        p_noise = p_noise / p_noise.sum(dim=-1, keepdim=True)

        x_hat = logits.softmax(dim=-1)
        p_max = x_hat.max(dim=-1).values
        p_curr = x_hat.gather(-1, z.unsqueeze(-1)).squeeze(-1)
        p_delta = (p_max - p_curr) * p_noise

        next_poss = torch.topk(p_delta, n_tokens, dim=-1).indices
        probs = self._probs_with_topk_topp(
            logits=logits,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )
        next_tokens = torch.multinomial(probs.flatten(0, 1), num_samples=1).view_as(z)

        z_next = z.clone()
        batch_indices = torch.arange(z.shape[0], device=z.device).unsqueeze(-1)
        z_next[batch_indices, next_poss] = next_tokens[batch_indices, next_poss]
        return z_next
    
    @torch.no_grad()
    def generate(
        self,
        inputs: tp.Optional[torch.Tensor] = None,
        max_length: int = 2048,
        min_length: int = 0,
        temperature: float = 1.0,
        block_length: int = 128,
        steps: int = 128,
        top_p: tp.Optional[float] = None,
        top_k: tp.Optional[int] = None,
        bos_token_id: int = 0,
        eos_token_id: int = 1,
        pad_token_id: int = 2,
        mask_token_id: int = 3,
        sampling_method: tp.Literal["ancestral", "adaptive"] = "ancestral",
        noise_schedule: tp.Literal["linear", "cosine"] | tp.Callable[[torch.Tensor], torch.Tensor] = "cosine",
        tokens_per_step: int = 1,
        show_progress: bool = False,
    ):
        r"""
        Generates tokens with block-wise denoising diffusion.

        Parameters:
            inputs (`torch.Tensor`):
                The token sequence used as a prompt for the generation.
            temperature (`float`, *optional*, defaults to 0.0):
                The value used to module the next token probabilities. A value of 0.0 corresponds to greedy decoding.
            block_length (`int`, *optional*, defaults to 32):
                The size of each generation block. The model generates text in parallel within these blocks. This is a
                key parameter for controlling the granularity of the generation process.
            steps (`int`, *optional*, defaults to 32):
                The number of denoising steps to perform for each block.
            max_length (`int`, *optional*, defaults to 2048):
                The maximum length of the sequence to be generated.
            min_length (`int`, *optional*, defaults to 0):
                The minimum length of the sequence to be generated.
            top_p (`float`, *optional*):
                If set to a float value between 0 and 1, only the most probable tokens with probabilities that add up to
                `top_p` or higher are kept for generation (nucleus sampling).
            top_k (`int`, *optional*):
                The number of highest probability vocabulary tokens to keep for top-k-filtering.
            bos_token_id (`int`, *optional*, defaults to 0):
                The token ID for the beginning-of-sequence token.
            eos_token_id (`int`, *optional*, defaults to 1):
                The token ID for the end-of-sequence token.
            pad_token_id (`int`, *optional*, defaults to 2):
                The token ID for the padding token.
            mask_token_id (`int`, *optional*, defaults to 3):
                The token ID used as a placeholder for tokens that are yet to be generated.
        Return:
            `torch.Tensor`: A string containing the generated token IDs, starting
            after the prompt and stopping at the first `eos_id` or `gen_length`.
        """
        if sampling_method not in ["ancestral", "adaptive"]:
            raise ValueError(f"Unsupported sampling method: {sampling_method}")
        if noise_schedule not in ["linear", "cosine"] and not callable(noise_schedule):
            raise ValueError("noise_schedule must be 'linear', 'cosine', or a callable function.")

        if inputs is None:
            inputs = torch.tensor([[bos_token_id]], device=self.device, dtype=torch.long)
            batch_size = 1
            prompt_length = 0
        else:
            batch_size = inputs.shape[0]
            prompt_length = inputs.shape[1]
            if eos_token_id in inputs:
                warnings.warn("Input prompt contains eos_token_id. Generation may stop earlier than expected.", stacklevel=1)
            input_ids = inputs.to(self.device)

        total_length = self.config.max_position_embeddings

        if noise_schedule == "linear":
            noise_schedule_fn = lambda t: 1.0 - t
        elif noise_schedule == "cosine":
            noise_schedule_fn = lambda t: 0.5 + 0.5 * torch.cos(t * torch.pi)
        else:
            noise_schedule_fn = noise_schedule

        x_prior = self._sample_prior(
            shape=(batch_size, total_length),
            device=self.device,
            mask_token_id=mask_token_id,
        )
        x = x_prior.clone()
        if prompt_length > 0:
            x[:, :prompt_length] = input_ids.clone()

        position_ids = torch.arange(total_length, device=self.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)

        noise_mask = torch.ones_like(x, dtype=torch.bool)
        noise_mask[:, :prompt_length] = False

        min_log_snr = torch.tensor(self.config.min_log_snr, device=self.device)
        max_log_snr = torch.tensor(self.config.max_log_snr, device=self.device)
        alpha_min = torch.sigmoid(min_log_snr)
        alpha_max = torch.sigmoid(max_log_snr)
        ts = torch.linspace(0.0, 1.0, steps=steps + 1, device=self.device)
        alpha_t = (alpha_max - alpha_min) * noise_schedule_fn(ts) + alpha_min
        log_snrs = torch.log(alpha_t / (1.0 - alpha_t)).clip(min_log_snr, max_log_snr)

        if show_progress:
            import tqdm.auto as tqdm
            est_num_blocks = (max_length + block_length - 1) // block_length
            est_num_steps = est_num_blocks * steps
            pbar = tqdm.tqdm(total=est_num_steps)
            update_pbar = lambda n: pbar.update(n)
            def stop_pbar():
                pbar.total = pbar.n
                pbar.refresh()
            close_pbar = lambda: pbar.close()
        else:
            update_pbar = lambda n: None
            stop_pbar = lambda: None
            close_pbar = lambda: None

        try:
            num_blocks = 0
            while True:
                current_window_start = prompt_length + num_blocks * block_length
                current_window_end = current_window_start + block_length
                attn_mask = (noise_mask[..., :, None] >= noise_mask[..., None, :])

                keep_logits = False
                past_key_values = None
                for step in range(steps, 0, -1):
                    if past_key_values is None:
                        output = self.forward(
                            input_ids=x[:, :current_window_start],
                            attention_mask=attn_mask[:, :current_window_start, :current_window_start],
                            position_ids=position_ids[:, :current_window_start],
                            use_cache=True,
                        )
                        past_key_values = output.past_key_values

                    if not keep_logits:
                        logits = self.forward(
                            input_ids=x[:, current_window_start:],
                            attention_mask=attn_mask[:, current_window_start:],
                            position_ids=position_ids[:, current_window_start:],
                            past_key_values=past_key_values,
                        ).logits
                        active_logits = logits[:, :block_length, :]
                        # logits = self.forward(
                        #     input_ids=x,
                        #     attention_mask=attn_mask,
                        #     position_ids=position_ids,
                        #     past_key_values=None
                        # ).logits
                        # active_logits = logits[:, current_window_start:current_window_end, :]

                        active_logits[..., mask_token_id] = float("-inf")
                        min_eos_idx = max(0, min_length + prompt_length - current_window_start)
                        active_logits[:, :min_eos_idx, eos_token_id] = float("-inf")
                    
                    z_t = x[:, current_window_start:current_window_end]
                    if sampling_method == "ancestral":
                        x_hat = self._probs_with_topk_topp(
                            active_logits.to(torch.float32),
                            temperature=temperature,
                            top_k=top_k,
                            top_p=top_p,
                        )

                        z_s = self._sample_ancestral(
                            z=z_t,
                            x_hat=x_hat,
                            log_snr_t=log_snrs[step],
                            log_snr_s=log_snrs[step - 1],
                            mask_token_id=mask_token_id,
                        )
                    elif sampling_method == "adaptive":
                        z_s = self._sample_adaptive(
                            z=z_t,
                            logits=active_logits.to(torch.float32),
                            log_snr=log_snrs[step],
                            n_tokens=tokens_per_step,
                            mask_token_id=mask_token_id,
                            temperature=temperature,
                            top_p=top_p,
                            top_k=top_k,
                        )
                    keep_logits = (z_s == z_t).all().item()

                    x[:, current_window_start:current_window_end] = z_s.clone()

                    update_pbar(1)

                num_blocks += 1
                noise_mask[:, :current_window_end] = False

                has_eos = (x == eos_token_id).any(-1).all().item()
                all_done = current_window_end >= max_length + prompt_length or has_eos
                if all_done:
                    stop_pbar()
                    break
        finally:
            close_pbar()

        generated_answer = x[:, :max_length + prompt_length]

        eos_idx = (generated_answer == eos_token_id).int().argmax(dim=-1)
        for i, idx in enumerate(eos_idx):
            if idx > 0:
                generated_answer[i, idx:] = pad_token_id

        return generated_answer
