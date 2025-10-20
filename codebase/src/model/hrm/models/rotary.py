"""
Rotary Positional Embeddings (RoPE) for HRM JAX implementation.

TPU-friendly implementation that avoids complex numbers and uses efficient
dot_general operations. Based on the PyTorch implementation but optimized for JAX/TPU.
"""

import jax
import jax.numpy as jnp
from jax import lax
from typing import Tuple, Optional
import math


def make_rope(
    head_dim: int,
    max_seq_len: int = 2048,
    base: float = 10000.0,
    dtype: jnp.dtype = jnp.float32
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Create rotary position embedding cos/sin tables.
    
    TPU-friendly implementation that precomputes cos/sin tables for all positions.
    Avoids complex numbers and uses real-valued operations only.
    
    Args:
        head_dim: Dimension of each attention head
        max_seq_len: Maximum sequence length to support
        base: Base for the geometric progression (default: 10000.0)
        dtype: Data type for the embeddings
        
    Returns:
        Tuple of (cos_table, sin_table) with shape [max_seq_len, head_dim]
    """
    # Create frequency inverse scaling
    # inv_freq = 1.0 / (base ** (arange(0, head_dim, 2) / head_dim))
    inv_freq = 1.0 / (base ** (jnp.arange(0, head_dim, 2, dtype=dtype) / head_dim))
    
    # Create position indices
    positions = jnp.arange(max_seq_len, dtype=dtype)
    
    # Compute angles: outer product of positions and inv_freq
    # Shape: [max_seq_len, head_dim // 2]
    angles = jnp.outer(positions, inv_freq)
    
    # Duplicate angles to match head_dim
    # Shape: [max_seq_len, head_dim]
    angles = jnp.repeat(angles, 2, axis=-1)
    
    # Compute cos and sin tables
    cos_table = jnp.cos(angles)
    sin_table = jnp.sin(angles)
    
    return cos_table, sin_table


def apply_rotary(
    x: jnp.ndarray,
    cos: jnp.ndarray,
    sin: jnp.ndarray,
    seq_len: Optional[int] = None
) -> jnp.ndarray:
    """
    Apply rotary position embedding to input tensor.
    
    TPU-optimized implementation using real-valued operations.
    Equivalent to complex rotation but avoids complex dtypes.
    
    Args:
        x: Input tensor with shape [..., seq_len, head_dim]
        cos: Cosine table with shape [max_seq_len, head_dim]
        sin: Sine table with shape [max_seq_len, head_dim]
        seq_len: Actual sequence length (if None, use x.shape[-2])
        
    Returns:
        Rotated tensor with same shape as input
    """
    if seq_len is None:
        seq_len = x.shape[-2]
    
    # Extract cos/sin for current sequence length
    cos = cos[:seq_len]  # [seq_len, head_dim]
    sin = sin[:seq_len]  # [seq_len, head_dim]
    
    # Split x into even and odd indices
    # x_even: [..., seq_len, head_dim // 2]
    # x_odd: [..., seq_len, head_dim // 2]
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    
    # Split cos/sin similarly
    cos_even = cos[..., 0::2]  # [seq_len, head_dim // 2]
    cos_odd = cos[..., 1::2]   # [seq_len, head_dim // 2]
    sin_even = sin[..., 0::2]  # [seq_len, head_dim // 2]
    sin_odd = sin[..., 1::2]   # [seq_len, head_dim // 2]
    
    # Apply rotation: 
    # Real part: x_even * cos - x_odd * sin
    # Imaginary part: x_even * sin + x_odd * cos
    rotated_even = x_even * cos_even - x_odd * sin_even
    rotated_odd = x_even * sin_odd + x_odd * cos_odd
    
    # Interleave back to original format
    # Stack along last dimension and reshape
    rotated = jnp.stack([rotated_even, rotated_odd], axis=-1)
    rotated = rotated.reshape(x.shape)
    
    return rotated


class RotaryEmbedding:
    """
    Rotary Position Embedding module for JAX/Flax.
    
    Precomputes and caches cos/sin tables for efficient application.
    Compatible with the PyTorch RotaryEmbedding interface.
    """
    
    def __init__(
        self,
        head_dim: int,
        max_seq_len: int = 2048,
        base: float = 10000.0,
        dtype: jnp.dtype = jnp.float32
    ):
        """
        Initialize RotaryEmbedding.
        
        Args:
            head_dim: Dimension of each attention head
            max_seq_len: Maximum sequence length to support
            base: Base for the geometric progression
            dtype: Data type for the embeddings
        """
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.base = base
        self.dtype = dtype
        
        # Precompute cos/sin tables
        self.cos_cached, self.sin_cached = make_rope(
            head_dim=head_dim,
            max_seq_len=max_seq_len,
            base=base,
            dtype=dtype
        )
    
    def __call__(self, seq_len: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Get cos/sin tables for given sequence length.
        
        Args:
            seq_len: Sequence length
            
        Returns:
            Tuple of (cos, sin) tensors with shape [seq_len, head_dim]
        """
        if seq_len > self.max_seq_len:
            # Extend tables if needed
            self.cos_cached, self.sin_cached = make_rope(
                head_dim=self.head_dim,
                max_seq_len=seq_len,
                base=self.base,
                dtype=self.dtype
            )
            self.max_seq_len = seq_len
        
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]


# Utility functions for backward compatibility
def rotate_half(x: jnp.ndarray) -> jnp.ndarray:
    """
    Rotate half the hidden dims of the input.
    
    This is an alternative implementation that matches the PyTorch version
    but is less TPU-efficient than apply_rotary.
    
    Args:
        x: Input tensor with shape [..., head_dim]
        
    Returns:
        Rotated tensor with same shape
    """
    x1, x2 = jnp.split(x, 2, axis=-1)
    return jnp.concatenate([-x2, x1], axis=-1)


def apply_rotary_pos_emb_legacy(
    q: jnp.ndarray,
    k: jnp.ndarray,
    cos: jnp.ndarray,
    sin: jnp.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Legacy implementation matching PyTorch exactly.
    
    Less efficient than apply_rotary but provided for compatibility.
    
    Args:
        q: Query tensor [..., seq_len, head_dim]
        k: Key tensor [..., seq_len, head_dim]
        cos: Cosine embeddings [seq_len, head_dim]
        sin: Sine embeddings [seq_len, head_dim]
        
    Returns:
        Tuple of rotated (q, k) tensors
    """
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed