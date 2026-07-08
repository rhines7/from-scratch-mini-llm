"""
Qwen3-small model architecture.

A compact decoder-only transformer trained from scratch. The design follows
modern Qwen3 choices so the same code path serves pre-training, generation, and
feature extraction for the downstream classifier:

- RMSNorm for pre-normalization (no mean centering, no bias)
- Rotary position embeddings (RoPE) applied to queries and keys
- SwiGLU feed-forward blocks
- Multi-head attention with optional grouped-query attention

Weights are always initialized from scratch; no pre-trained checkpoints are
loaded. Defaults are sized for CPU training within 32 GB RAM.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class Qwen3Config:
    """Configuration for the Qwen3-small model.

    head_dim is derived in __post_init__ from d_model / num_heads. Set
    num_kv_heads < num_heads to enable grouped-query attention; keep them equal
    for standard multi-head attention.
    """
    # Model architecture. Defaults match the 19.52M-parameter model used for the
    # reported run (see pretrain.default_model_config), so instantiating
    # Qwen3Config() reproduces that architecture rather than a larger placeholder.
    vocab_size: int = 7392  # Vocabulary size from the 10K TinyStories subset
    max_seq_len: int = 256
    d_model: int = 512  # Hidden dimension
    num_layers: int = 6  # Number of transformer blocks
    num_heads: int = 8  # Number of attention heads (must divide d_model)
    num_kv_heads: int = 8  # For GQA: set < num_heads; for MHA: set = num_heads
    intermediate_size: int = 1024  # FFN intermediate dimension (SwiGLU)

    # RoPE settings
    rope_base: float = 10000.0  # Base frequency for RoPE (10000 for short sequences)
    rope_scaling: Optional[float] = None  # For length extrapolation if needed

    # Normalization
    rms_norm_eps: float = 1e-6  # Epsilon for RMSNorm stability

    # Regularization
    dropout: float = 0.0  # Dropout probability (0.0-0.1 typical for pre-training)
    attention_dropout: float = 0.0  # Attention-specific dropout

    # Optimization settings
    use_bias: bool = False  # Modern practice: no bias in attention projections
    tie_embeddings: bool = True  # Tie input and output embeddings to save parameters

    # Weight initialization
    initializer_range: float = 0.02  # Standard deviation for weight initialization

    def __post_init__(self):
        """Validate configuration after initialization."""
        # Ensure num_heads divides d_model evenly
        assert self.d_model % self.num_heads == 0, \
            f"d_model ({self.d_model}) must be divisible by num_heads ({self.num_heads})"

        # Ensure num_kv_heads divides num_heads for GQA
        assert self.num_heads % self.num_kv_heads == 0, \
            f"num_heads ({self.num_heads}) must be divisible by num_kv_heads ({self.num_kv_heads})"

        # Validate vocabulary size
        assert self.vocab_size > 0, "vocab_size must be positive"

        # Calculate head dimensions
        self.head_dim = self.d_model // self.num_heads

    def get_parameter_count(self) -> dict:
        """Return an analytical parameter-count breakdown for this config.

        Useful for sizing before instantiation; the model's own
        count_parameters() reports the realized count after construction.
        """
        # Token embeddings
        embedding_params = self.vocab_size * self.d_model

        # Per transformer block
        # Attention (Q, K, V, O projections)
        q_params = self.d_model * self.d_model
        kv_params = 2 * self.d_model * (self.d_model * self.num_kv_heads // self.num_heads)
        o_params = self.d_model * self.d_model
        attention_params = q_params + kv_params + o_params

        # RMSNorm (2 per block: input and post-attention)
        norm_params_per_block = 2 * self.d_model

        # SwiGLU FFN (3 projections: gate, value, output)
        ffn_params = 3 * self.d_model * self.intermediate_size
        if self.use_bias:
            ffn_params += 2 * self.intermediate_size + self.d_model

        block_params = attention_params + norm_params_per_block + ffn_params

        # Total transformer blocks
        transformer_params = self.num_layers * block_params

        # Final RMSNorm
        final_norm_params = self.d_model

        # Output layer (tied head reuses the embedding matrix)
        if self.tie_embeddings:
            output_params = 0
        else:
            output_params = self.vocab_size * self.d_model

        # RoPE has no parameters (computed on-the-fly)
        rope_params = 0

        total_params = (
            embedding_params +
            transformer_params +
            final_norm_params +
            output_params
        )

        return {
            'embedding': embedding_params,
            'transformer_blocks': transformer_params,
            'final_norm': final_norm_params,
            'output_layer': output_params,
            'rope': rope_params,
            'total': total_params,
            'total_millions': total_params / 1e6
        }


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    RMSNorm(x) = (x / sqrt(mean(x^2) + eps)) * weight. No mean centering and no
    bias, which is cheaper than LayerNorm and matches the Qwen3 reference.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # RMS over the last dimension, then scale
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        x_normalized = x / rms
        return self.weight * x_normalized


class RotaryPositionEmbedding(nn.Module):
    """Rotary position embeddings (RoPE).

    Encodes position by rotating query and key vectors, so there are no learned
    position parameters and relative position falls out of the dot product.
    cos/sin tables are precomputed and extended on demand for longer sequences.
    """

    def __init__(self, head_dim: int, max_seq_len: int = 2048, base: float = 10000.0):
        super().__init__()
        self.head_dim = head_dim
        self.base = base
        self.max_seq_len = max_seq_len

        # Inverse frequencies: theta_i = base^(-2i/d) for i in [0, d/2)
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer('inv_freq', inv_freq, persistent=False)

        self._precompute_freqs_cis(max_seq_len)

    def _precompute_freqs_cis(self, seq_len: int):
        """Cache cos/sin for all positions up to seq_len."""
        t = torch.arange(seq_len, device=self.inv_freq.device).type_as(self.inv_freq)

        # Outer product of positions and inverse frequencies: (seq_len, head_dim/2)
        freqs = torch.outer(t, self.inv_freq)

        self.register_buffer('cos_cached', freqs.cos(), persistent=False)
        self.register_buffer('sin_cached', freqs.sin(), persistent=False)

    def forward(self, x: torch.Tensor, seq_len: int, offset: int = 0) -> torch.Tensor:
        """Apply RoPE to x of shape (batch, num_heads, seq_len, head_dim).

        offset is the absolute position of the first token in x. During cached
        decoding only the new token(s) are passed, so offset equals the number of
        tokens already in the KV cache; without it the new token would always be
        rotated as position 0 and RoPE would carry no positional signal.
        """
        # Extend cache when the absolute span exceeds what has been precomputed
        total_len = offset + seq_len
        if total_len > self.max_seq_len:
            self._precompute_freqs_cis(total_len)
            self.max_seq_len = total_len

        cos = self.cos_cached[offset:offset + seq_len].unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, head_dim/2)
        sin = self.sin_cached[offset:offset + seq_len].unsqueeze(0).unsqueeze(0)

        # Split into halves and apply the 2D rotation
        x1, x2 = x.chunk(2, dim=-1)
        rotated = torch.cat([
            x1 * cos - x2 * sin,
            x1 * sin + x2 * cos
        ], dim=-1)

        return rotated


class SwiGLU(nn.Module):
    """SwiGLU feed-forward block: Swish(xW_gate) * (xW_value), then down-project.

    Three projections (gate, value, output) cost more than a standard FFN but
    train better at this scale.
    """

    def __init__(self, d_model: int, intermediate_size: int, use_bias: bool = False):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, intermediate_size, bias=use_bias)
        self.value_proj = nn.Linear(d_model, intermediate_size, bias=use_bias)
        self.output_proj = nn.Linear(intermediate_size, d_model, bias=use_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.gate_proj(x)
        value = self.value_proj(x)

        # Swish(gate) = gate * sigmoid(gate)
        gate_swished = gate * torch.sigmoid(gate)
        hidden = gate_swished * value

        return self.output_proj(hidden)


class MultiHeadAttention(nn.Module):
    """Causal multi-head attention with RoPE and optional grouped-query attention.

    When num_kv_heads < num_heads, key/value heads are repeated to match the
    query heads, which shrinks the KV cache during generation.
    """

    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.d_model = config.d_model
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.d_model // config.num_heads
        self.attention_dropout = config.attention_dropout

        # Query projection (all heads)
        self.q_proj = nn.Linear(self.d_model, self.d_model, bias=config.use_bias)

        # Key/value projections (fewer heads under GQA)
        kv_dim = self.head_dim * self.num_kv_heads
        self.k_proj = nn.Linear(self.d_model, kv_dim, bias=config.use_bias)
        self.v_proj = nn.Linear(self.d_model, kv_dim, bias=config.use_bias)

        # Output projection
        self.o_proj = nn.Linear(self.d_model, self.d_model, bias=config.use_bias)

        self.rope = RotaryPositionEmbedding(
            head_dim=self.head_dim,
            max_seq_len=config.max_seq_len,
            base=config.rope_base
        )

        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.dropout = nn.Dropout(self.attention_dropout) if self.attention_dropout > 0 else None

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """Attention over (batch, seq_len, d_model); returns output and optional KV cache."""
        batch_size, seq_len, _ = hidden_states.size()

        # Project to Q, K, V
        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)

        # Reshape to (batch, heads, seq_len, head_dim)
        query = query.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # RoPE on queries and keys, offset by the cached length so the new
        # token(s) get their true absolute positions rather than restarting at 0
        past_len = past_key_value[0].size(2) if past_key_value is not None else 0
        query = self.rope(query, seq_len, offset=past_len)
        key = self.rope(key, seq_len, offset=past_len)

        # Append cached K/V during incremental generation
        if past_key_value is not None:
            past_key, past_value = past_key_value
            key = torch.cat([past_key, key], dim=2)
            value = torch.cat([past_value, value], dim=2)

        if use_cache:
            past_key_value = (key, value)
        else:
            past_key_value = None

        # Expand K/V heads to match query heads under GQA
        if self.num_kv_heads < self.num_heads:
            repeat_factor = self.num_heads // self.num_kv_heads
            key = key.repeat_interleave(repeat_factor, dim=1)
            value = value.repeat_interleave(repeat_factor, dim=1)

        # Scaled dot-product scores: (batch, heads, seq_len, kv_seq_len)
        attn_scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale

        # Causal mask; diagonal offset handles cached (longer) key length
        kv_seq_len = key.size(2)
        causal_mask = torch.triu(
            torch.ones(seq_len, kv_seq_len, device=hidden_states.device, dtype=torch.bool),
            diagonal=kv_seq_len - seq_len + 1
        )
        attn_scores = attn_scores.masked_fill(causal_mask, float('-inf'))

        # Padding mask (0 marks padding to be ignored)
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attn_scores = attn_scores.masked_fill(attention_mask == 0, float('-inf'))

        # Softmax in float32 for stability, then cast back
        attn_probs = F.softmax(attn_scores, dim=-1, dtype=torch.float32).type_as(attn_scores)

        if self.dropout is not None:
            attn_probs = self.dropout(attn_probs)

        attn_output = torch.matmul(attn_probs, value)

        # Merge heads back to (batch, seq_len, d_model)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

        output = self.o_proj(attn_output)

        return output, past_key_value


class Qwen3TransformerBlock(nn.Module):
    """Pre-norm transformer block: RMSNorm -> attention -> residual, then
    RMSNorm -> SwiGLU -> residual."""

    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.input_norm = RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.attention = MultiHeadAttention(config)
        self.post_attention_norm = RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.ffn = SwiGLU(
            d_model=config.d_model,
            intermediate_size=config.intermediate_size,
            use_bias=config.use_bias
        )
        self.dropout = nn.Dropout(config.dropout) if config.dropout > 0 else None

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        # Attention sublayer
        residual = hidden_states
        hidden_states = self.input_norm(hidden_states)
        hidden_states, past_key_value = self.attention(
            hidden_states,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            use_cache=use_cache
        )
        if self.dropout is not None:
            hidden_states = self.dropout(hidden_states)
        hidden_states = residual + hidden_states

        # Feed-forward sublayer
        residual = hidden_states
        hidden_states = self.post_attention_norm(hidden_states)
        hidden_states = self.ffn(hidden_states)
        if self.dropout is not None:
            hidden_states = self.dropout(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states, past_key_value


class Qwen3Model(nn.Module):
    """Decoder-only Qwen3-small for causal language modeling.

    Token embeddings -> N transformer blocks -> final RMSNorm -> LM head. The
    head reuses the embedding matrix when tie_embeddings is set. output_hidden_states
    exposes per-layer states used by the downstream sentiment classifier.
    """

    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.config = config

        self.token_embeddings = nn.Embedding(config.vocab_size, config.d_model)

        self.layers = nn.ModuleList([
            Qwen3TransformerBlock(config) for _ in range(config.num_layers)
        ])

        self.final_norm = RMSNorm(config.d_model, eps=config.rms_norm_eps)

        # Tied head reuses token_embeddings.weight; otherwise a dedicated matrix
        if config.tie_embeddings:
            self.lm_head = None
        else:
            self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        self.apply(self._init_weights)

        param_info = config.get_parameter_count()
        print(f"Initialized Qwen3-small with {param_info['total_millions']:.2f}M parameters")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, RMSNorm):
            torch.nn.init.ones_(module.weight)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[list] = None,
        use_cache: bool = False,
        return_dict: bool = True,
        output_hidden_states: bool = False
    ) -> dict:
        """Return logits and last_hidden_state; optionally KV cache and per-layer states."""
        hidden_states = self.token_embeddings(input_ids)

        if past_key_values is None and use_cache:
            past_key_values = [None] * len(self.layers)

        all_hidden_states = [] if output_hidden_states else None

        new_past_key_values = [] if use_cache else None
        for i, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states.append(hidden_states)

            past_key_value = past_key_values[i] if past_key_values is not None else None

            hidden_states, new_past_key_value = layer(
                hidden_states,
                attention_mask=attention_mask,
                past_key_value=past_key_value,
                use_cache=use_cache
            )

            if use_cache:
                new_past_key_values.append(new_past_key_value)

        hidden_states = self.final_norm(hidden_states)

        # Tied head: matmul against the embedding matrix
        if self.config.tie_embeddings:
            logits = F.linear(hidden_states, self.token_embeddings.weight)
        else:
            logits = self.lm_head(hidden_states)

        if return_dict:
            output = {
                'logits': logits,
                'last_hidden_state': hidden_states,
            }
            if use_cache:
                output['past_key_values'] = new_past_key_values
            if output_hidden_states:
                output['hidden_states'] = all_hidden_states
            return output
        else:
            return (logits, hidden_states, new_past_key_values if use_cache else None)

    def count_parameters(self) -> dict:
        """Realized parameter counts, compared against the analytical estimate."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        expected = self.config.get_parameter_count()

        return {
            'actual_total': total_params,
            'actual_trainable': trainable_params,
            'expected_total': expected['total'],
            'difference': total_params - expected['total'],
            'actual_millions': total_params / 1e6,
            'breakdown': expected
        }


def create_qwen3_small(
    vocab_size: int = 7392,
    d_model: int = 512,
    num_layers: int = 6,
    num_heads: int = 8,
    intermediate_size: int = 1024,
    **kwargs
) -> Qwen3Model:
    """Build a Qwen3Model from the common knobs, forwarding extras to Qwen3Config.

    Defaults reproduce the 19.52M-parameter reported model; override any knob for
    a different size.
    """
    config = Qwen3Config(
        vocab_size=vocab_size,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        intermediate_size=intermediate_size,
        **kwargs
    )

    return Qwen3Model(config)
