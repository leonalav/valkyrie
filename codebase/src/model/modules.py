"""Core model modules and utilities.

Extracted from 1_jax.py with EXACT mathematical implementations.
DO NOT MODIFY - these implementations are mathematically verified.
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
import math
from dataclasses import dataclass
from typing import Optional, List


@dataclass
class ValkyrieConfig:
    """Configuration class for Valkyrie model."""
    # Model architecture
    vocab_size: int = 50257
    d_model: int = 1536
    n_layers: int = 32
    n_heads: int = 16
    n_kv_heads: Optional[int] = None  # For grouped-query attention
    
    # Position embeddings and RoPE
    original_max_position_embeddings: int = 4096
    max_position_embeddings: int = 32768
    rope_theta: float = 10000.0
    yarn_beta_fast: float = 32.0
    yarn_beta_slow: float = 1.0
    
    # Dropout rates
    attn_dropout: float = 0.0
    resid_dropout: float = 0.0
    ffn_dropout: float = 0.1
    
    # Model configuration
    use_bias: bool = False
    layer_norm_eps: float = 1e-5
    initializer_range: float = 0.02
    
    # S5 configuration
    s5_state_dim: int = 128  # State dimension for S5 layers
    use_s5: bool = True     # Whether to use S5 layers instead of FFN
    
    # Training configuration
    gradient_clipping: float = 1.0
    weight_decay: float = 0.1
    gradient_checkpointing: bool = True  # Enable gradient checkpointing
    
    # Longformer attention configuration
    use_longformer_attention: bool = False
    longformer_window_size: int = 512  # Sliding window size
    longformer_global_attention_indices: Optional[List[int]] = None  # Global token positions
    longformer_dilation: Optional[int] = None  # Avoid unless custom kernel
    longformer_chunked: bool = True  # Use chunked vectorized implementation
    longformer_chunk_size: int = 512  # Chunk size for memory-efficient processing
    longformer_use_full_attention_fallback: bool = True  # Use full attention for small sequences
    longformer_combine_logits: bool = False  # Combine logits before softmax (more mathematically consistent)

    def __post_init__(self):
        if self.n_kv_heads is None:
            self.n_kv_heads = self.n_heads
        self.rope_scaling_factor = self.max_position_embeddings / self.original_max_position_embeddings
        assert self.d_model % self.n_heads == 0
        assert self.n_heads % self.n_kv_heads == 0
        head_dim = self.d_model // self.n_heads
        assert head_dim % 2 == 0
        assert head_dim <= 256


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    hidden_size: int
    eps: float = 1e-6

    def setup(self):
        self.weight = self.param('weight', nn.initializers.ones, (self.hidden_size,))

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        input_dtype = x.dtype
        x = x.astype(jnp.float32)
        variance = jnp.mean(x**2, axis=-1, keepdims=True)
        x = x * jax.lax.rsqrt(variance + self.eps)
        return (self.weight * x).astype(input_dtype)


def precompute_rope_freqs(dim: int, max_seq_len: int, base: float = 10000.0):
    """Precompute RoPE frequencies for efficient lookup."""
    inv_freq = 1.0 / (base ** (jnp.arange(0, dim, 2).astype(jnp.float32) / dim))
    t = jnp.arange(max_seq_len, dtype=jnp.float32)
    freqs = jnp.outer(t, inv_freq)
    cos_freqs = jnp.cos(freqs)
    sin_freqs = jnp.sin(freqs)
    return cos_freqs, sin_freqs


def apply_rope(x, cos_freqs, sin_freqs, position_ids):
    """Apply RoPE rotation using precomputed frequencies."""
    # x: [batch, seq_len, num_heads, head_dim]
    # cos_freqs, sin_freqs: [max_seq_len, head_dim//2]
    # position_ids: [batch, seq_len]
    
    batch_size, seq_len, num_heads, head_dim = x.shape
    
    # Select frequencies for current positions
    cos = cos_freqs[position_ids]  # [batch, seq_len, head_dim//2]
    sin = sin_freqs[position_ids]  # [batch, seq_len, head_dim//2]
    
    # Expand for num_heads
    cos = jnp.expand_dims(cos, 2)  # [batch, seq_len, 1, head_dim//2]
    sin = jnp.expand_dims(sin, 2)  # [batch, seq_len, 1, head_dim//2]
    
    # Split x into even and odd dimensions
    x_even = x[..., ::2]  # [batch, seq_len, num_heads, head_dim//2]
    x_odd = x[..., 1::2]  # [batch, seq_len, num_heads, head_dim//2]
    
    # Apply rotation
    rotated_even = x_even * cos - x_odd * sin
    rotated_odd = x_even * sin + x_odd * cos
    
    # Interleave back
    rotated = jnp.stack([rotated_even, rotated_odd], axis=-1)
    rotated = rotated.reshape(batch_size, seq_len, num_heads, head_dim)
    
    return rotated