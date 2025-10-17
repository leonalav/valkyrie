"""
Utility functions for KV cache management in attention mechanisms.
"""

import jax.numpy as jnp


def detect_cache_length_from_ks(past_ks: jnp.ndarray) -> jnp.ndarray:
    """
    JIT-friendly cache length detection.
    
    Args:
        past_ks: Cached keys [B, n_kv_heads, max_seq_len, head_dim]
        
    Returns:
        cache_seq_len: Number of filled positions in cache (scalar)
    """
    # past_ks: [B, n_kv_heads, max_seq_len, head_dim]
    # Check if each position is all zeros across batch, heads, and head_dim
    is_zero_pos = jnp.all(past_ks == 0, axis=(0, 1, 3))  # [max_seq_len]
    
    # Find if there's any zero position
    any_zero = jnp.any(is_zero_pos)
    
    # Find first zero position (argmax returns first True index)
    first_zero = jnp.argmax(is_zero_pos.astype(jnp.int32))
    
    # If no zeros found, cache is full
    max_seq_len = past_ks.shape[2]
    cache_seq_len = jnp.where(any_zero, first_zero, max_seq_len)
    
    return cache_seq_len