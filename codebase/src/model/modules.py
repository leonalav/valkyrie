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
    
    # BigBird attention configuration
    use_bigbird_attention: bool = True
    bigbird_block_size: int = 64  # Block size for sparse attention
    bigbird_num_global_tokens: int = 64  # Number of global tokens (for HRM)
    bigbird_num_window_blocks: int = 3  # Number of window blocks on each side
    bigbird_num_random_blocks: int = 2  # Number of random blocks per block
    bigbird_use_blockified_gemm: bool = True  # Use blockified GEMM for TPU efficiency

    # Longformer attention configuration
    use_longformer_attention: bool = False
    longformer_window_size: int = 64
    longformer_global_attention_indices: Optional[List[int]] = None
    longformer_chunked: bool = True
    longformer_chunk_size: int = 32
    longformer_use_full_attention_fallback: bool = False

    # HRM configuration
    use_hrm: bool = True
    hrm_plan_length: int = 32  # Length of HRM planning sequence
    hrm_H_cycles: int = 3  # High-level reasoning cycles
    hrm_L_cycles: int = 3  # Low-level reasoning cycles
    hrm_H_layers: int = 6  # High-level transformer layers
    hrm_L_layers: int = 6  # Low-level transformer layers
    hrm_intermediate_size: Optional[int] = None  # FFN intermediate size for HRM
    hrm_use_act: bool = True  # Use Adaptive Computation Time
    hrm_act_threshold: float = 0.9  # ACT halting threshold
    
    # Additional HRM parameters from config
    hrm_planner_layers: int = 2  # Number of planner layers
    hrm_executor_steps: int = 4  # Number of executor steps
    hrm_planner_update_frequency: int = 4  # Planner update frequency
    hrm_use_act_halting: bool = True  # Use ACT halting
    hrm_one_step_gradient: bool = True  # Use one-step gradient
    hrm_deep_supervision: bool = True  # Use deep supervision
    def __post_init__(self):
        if self.n_kv_heads is None:
            self.n_kv_heads = self.n_heads
        self.rope_scaling_factor = self.max_position_embeddings / self.original_max_position_embeddings
        assert self.d_model % self.n_heads == 0
        assert self.n_heads % self.n_kv_heads == 0
        head_dim = self.d_model // self.n_heads
        assert head_dim % 2 == 0
        assert head_dim <= 256
        
        # Set HRM intermediate size if not specified
        if self.hrm_intermediate_size is None:
            self.hrm_intermediate_size = 4 * self.d_model


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


class TiedEmbedding(nn.Module):
    """Embedding layer with tied weights for output projection.
    
    This layer supports both embedding lookup (for input tokens) and 
    tied weight output projection (for computing logits).
    """
    vocab_size: int
    embed_dim: int
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        """Embedding lookup for input tokens."""
        if inputs.dtype not in [jnp.int32, jnp.int64]:
            raise ValueError(f"Input dtype must be int32 or int64, got {inputs.dtype}")
        
        # Define the embedding parameter
        embedding = self.param(
            'embedding',
            nn.initializers.normal(stddev=0.02),
            (self.vocab_size, self.embed_dim),
            self.param_dtype
        )
        
        # Ensure inputs are within vocabulary range
        inputs = jnp.clip(inputs, 0, self.vocab_size - 1)
        
        # Perform embedding lookup
        embeddings = embedding[inputs]
        return embeddings.astype(self.dtype)

    @nn.compact
    def attend(self, inputs: jnp.ndarray) -> jnp.ndarray:
        """Output projection using tied embedding weights.
        
        Args:
            inputs: Hidden states of shape [..., embed_dim]
            
        Returns:
            Logits of shape [..., vocab_size]
        """
        # inputs: [..., embed_dim]
        # embedding: [vocab_size, embed_dim]
        # output: [..., vocab_size]
        
        # Get the same embedding parameter (Flax will reuse the same parameter)
        embedding = self.param(
            'embedding',
            nn.initializers.normal(stddev=0.02),
            (self.vocab_size, self.embed_dim),
            self.param_dtype
        )
        
        # Ensure inputs are the right dtype
        inputs = inputs.astype(self.param_dtype)
        
        # Matrix multiplication: inputs @ embedding.T
        logits = jnp.dot(inputs, embedding.T)
        
        return logits.astype(self.dtype)


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