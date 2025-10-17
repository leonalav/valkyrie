"""Gryphon Utilities

Utility functions for sparse attention patterns, block operations, and other
helper functions needed for the BigBird-S5 hybrid architecture.
"""

import jax
import jax.numpy as jnp
from typing import Tuple, List, Optional
import math


def pad_to_block_size(x: jnp.ndarray, block_size: int, axis: int = 1) -> Tuple[jnp.ndarray, int]:
    """Pad sequence to be divisible by block_size.
    
    Args:
        x: Input tensor [..., seq_len, ...]
        block_size: Size of each block
        axis: Axis to pad (default: 1 for sequence dimension)
        
    Returns:
        Tuple of (padded_tensor, original_length)
    """
    seq_len = x.shape[axis]
    
    if seq_len % block_size == 0:
        return x, seq_len
    
    # Calculate padding needed
    pad_len = block_size - (seq_len % block_size)
    
    # Create padding specification
    pad_spec = [(0, 0)] * x.ndim
    pad_spec[axis] = (0, pad_len)
    
    # Pad with zeros
    padded_x = jnp.pad(x, pad_spec, mode='constant', constant_values=0)
    
    return padded_x, seq_len


def reshape_to_blocks(x: jnp.ndarray, block_size: int) -> jnp.ndarray:
    """Reshape sequence tensor to block format.
    
    Args:
        x: Input tensor [batch, seq_len, hidden_size]
        block_size: Size of each block
        
    Returns:
        Reshaped tensor [batch, num_blocks, block_size, hidden_size]
    """
    batch_size, seq_len, hidden_size = x.shape
    
    if seq_len % block_size != 0:
        raise ValueError(f"seq_len ({seq_len}) must be divisible by block_size ({block_size})")
    
    num_blocks = seq_len // block_size
    
    return x.reshape(batch_size, num_blocks, block_size, hidden_size)


def reshape_from_blocks(x: jnp.ndarray) -> jnp.ndarray:
    """Reshape block tensor back to sequence format.
    
    Args:
        x: Block tensor [batch, num_blocks, block_size, hidden_size]
        
    Returns:
        Sequence tensor [batch, seq_len, hidden_size]
    """
    batch_size, num_blocks, block_size, hidden_size = x.shape
    seq_len = num_blocks * block_size
    
    return x.reshape(batch_size, seq_len, hidden_size)


def get_window_indices(block_id: int, num_blocks: int, window_size: int) -> jnp.ndarray:
    """Get window attention indices for a given block.
    
    Args:
        block_id: Current block index
        num_blocks: Total number of blocks
        window_size: Window size in blocks (includes current block)
        
    Returns:
        Array of block indices within the window
    """
    # Window extends (window_size - 1) // 2 blocks in each direction
    half_window = (window_size - 1) // 2
    
    start_idx = max(0, block_id - half_window)
    end_idx = min(num_blocks, block_id + half_window + 1)
    
    return jnp.arange(start_idx, end_idx)


def get_random_block_indices(
    block_id: int, 
    num_blocks: int, 
    num_random: int,
    global_indices: jnp.ndarray,
    window_indices: jnp.ndarray,
    seed: int = 42
) -> jnp.ndarray:
    """Get random attention indices for a given block.
    
    Uses deterministic pseudo-random generation for JIT compatibility.
    
    Args:
        block_id: Current block index
        num_blocks: Total number of blocks
        num_random: Number of random blocks to select
        global_indices: Global block indices to exclude
        window_indices: Window block indices to exclude
        seed: Random seed for deterministic generation
        
    Returns:
        Array of random block indices
    """
    if num_random <= 0:
        return jnp.array([], dtype=jnp.int32)
    
    # Create exclusion mask
    exclude_mask = jnp.zeros(num_blocks, dtype=bool)
    exclude_mask = exclude_mask.at[global_indices].set(True)
    exclude_mask = exclude_mask.at[window_indices].set(True)
    
    # Get available indices using JAX-compatible operations
    available_indices = jnp.where(~exclude_mask, jnp.arange(num_blocks), -1)
    # Filter out -1 values using JAX operations
    valid_mask = available_indices >= 0
    num_valid = jnp.sum(valid_mask)
    
    # Create a compact array of valid indices
    available_indices = jnp.where(
        jnp.arange(num_blocks) < num_valid,
        jnp.sort(jnp.where(valid_mask, available_indices, num_blocks))[:num_valid],
        -1
    )
    
    # Handle empty case
    available_indices = jnp.where(num_valid > 0, available_indices, jnp.array([], dtype=jnp.int32))
    
    # Use block_id as part of the seed for deterministic but varied selection
    rng_key = jax.random.PRNGKey(seed + block_id)
    
    # Select random indices without replacement
    num_to_select = min(num_random, len(available_indices))
    selected_indices = jax.random.choice(
        rng_key, 
        available_indices, 
        shape=(num_to_select,), 
        replace=False
    )
    
    return jnp.sort(selected_indices)


def create_sparse_attention_mask(
    num_blocks: int,
    block_size: int,
    num_global_blocks: int,
    window_size: int,
    num_random_blocks: int,
    seed: int = 42
) -> jnp.ndarray:
    """Create sparse attention mask for BigBird pattern.
    
    Args:
        num_blocks: Total number of blocks
        block_size: Size of each block
        num_global_blocks: Number of global blocks
        window_size: Window size in blocks
        num_random_blocks: Number of random blocks per query
        seed: Random seed for deterministic patterns
        
    Returns:
        Attention mask [num_blocks, num_blocks] where 1 indicates attention
    """
    # Input validation
    assert isinstance(num_blocks, int) and num_blocks > 0, \
        f"num_blocks must be positive integer, got {num_blocks}"
    assert isinstance(block_size, int) and block_size > 0, \
        f"block_size must be positive integer, got {block_size}"
    assert isinstance(num_global_blocks, int) and num_global_blocks >= 0, \
        f"num_global_blocks must be non-negative integer, got {num_global_blocks}"
    assert isinstance(window_size, int) and window_size > 0, \
        f"window_size must be positive integer, got {window_size}"
    assert isinstance(num_random_blocks, int) and num_random_blocks >= 0, \
        f"num_random_blocks must be non-negative integer, got {num_random_blocks}"
    assert num_global_blocks <= num_blocks, \
        f"num_global_blocks ({num_global_blocks}) cannot exceed num_blocks ({num_blocks})"
    
    mask = jnp.zeros((num_blocks, num_blocks), dtype=bool)
    
    # Global block indices (first num_global_blocks)
    global_indices = jnp.arange(num_global_blocks)
    
    for block_id in range(num_blocks):
        # 1. Global attention: all blocks attend to/from global blocks
        mask = mask.at[block_id, global_indices].set(True)
        if block_id < num_global_blocks:
            mask = mask.at[block_id, :].set(True)  # Global blocks attend to all
        
        # 2. Window attention: local neighborhood
        window_indices = get_window_indices(block_id, num_blocks, window_size)
        mask = mask.at[block_id, window_indices].set(True)
        
        # 3. Random attention: random distant blocks
        if num_random_blocks > 0:
            random_indices = get_random_block_indices(
                block_id, num_blocks, num_random_blocks,
                global_indices, window_indices, seed
            )
            if len(random_indices) > 0:
                mask = mask.at[block_id, random_indices].set(True)
    
    # Final shape validation
    expected_shape = (num_blocks, num_blocks)
    assert mask.shape == expected_shape, \
        f"Output shape mismatch: expected {expected_shape}, got {mask.shape}"
    
    return mask


def create_block_attention_indices(
    num_blocks: int,
    num_global_blocks: int,
    window_size: int,
    num_random_blocks: int,
    seed: int = 42
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Create efficient attention indices for block-sparse attention.
    
    Returns indices in a format optimized for gather operations.
    
    Args:
        num_blocks: Total number of blocks
        num_global_blocks: Number of global blocks
        window_size: Window size in blocks
        num_random_blocks: Number of random blocks per query
        seed: Random seed for deterministic patterns
        
    Returns:
        Tuple of (query_indices, key_indices) for efficient attention computation
    """
    max_attention_blocks = num_global_blocks + window_size + num_random_blocks
    
    # Initialize with -1 (invalid indices)
    query_indices = jnp.full((num_blocks, max_attention_blocks), -1, dtype=jnp.int32)
    key_indices = jnp.full((num_blocks, max_attention_blocks), -1, dtype=jnp.int32)
    
    global_indices = jnp.arange(num_global_blocks)
    
    for block_id in range(num_blocks):
        attention_blocks = []
        
        # Add global blocks
        attention_blocks.extend(global_indices.tolist())
        
        # Add window blocks
        window_indices = get_window_indices(block_id, num_blocks, window_size)
        attention_blocks.extend(window_indices.tolist())
        
        # Add random blocks
        if num_random_blocks > 0:
            random_indices = get_random_block_indices(
                block_id, num_blocks, num_random_blocks,
                global_indices, window_indices, seed
            )
            attention_blocks.extend(random_indices.tolist())
        
        # Remove duplicates and sort
        attention_blocks = sorted(list(set(attention_blocks)))
        
        # Fill indices arrays
        num_attention = min(len(attention_blocks), max_attention_blocks)
        query_indices = query_indices.at[block_id, :num_attention].set(
            jnp.array(attention_blocks[:num_attention])
        )
        key_indices = key_indices.at[block_id, :num_attention].set(
            jnp.array(attention_blocks[:num_attention])
        )
    
    return query_indices, key_indices


def compute_attention_sparsity(
    num_blocks: int,
    num_global_blocks: int,
    window_size: int,
    num_random_blocks: int
) -> dict:
    """Compute sparsity statistics for the attention pattern.
    
    Args:
        num_blocks: Total number of blocks
        num_global_blocks: Number of global blocks
        window_size: Window size in blocks
        num_random_blocks: Number of random blocks per query
        
    Returns:
        Dictionary with sparsity statistics
    """
    # Full attention operations
    full_ops = num_blocks ** 2
    
    # Sparse attention operations
    avg_attention_per_block = num_global_blocks + window_size + num_random_blocks
    
    # Global blocks attend to all blocks
    global_ops = num_global_blocks * num_blocks
    
    # Regular blocks attend to limited set
    regular_ops = (num_blocks - num_global_blocks) * avg_attention_per_block
    
    sparse_ops = global_ops + regular_ops
    
    sparsity_ratio = 1.0 - (sparse_ops / full_ops)
    
    return {
        'full_attention_ops': full_ops,
        'sparse_attention_ops': sparse_ops,
        'sparsity_ratio': sparsity_ratio,
        'memory_reduction': sparsity_ratio,
        'avg_attention_per_block': avg_attention_per_block,
        'max_attention_per_block': max(avg_attention_per_block, num_blocks)  # Global blocks
    }


def validate_attention_pattern(
    mask: jnp.ndarray,
    num_global_blocks: int,
    window_size: int
) -> bool:
    """Validate that attention mask follows BigBird pattern correctly.
    
    Args:
        mask: Attention mask [num_blocks, num_blocks]
        num_global_blocks: Expected number of global blocks
        window_size: Expected window size
        
    Returns:
        True if pattern is valid, False otherwise
    """
    num_blocks = mask.shape[0]
    
    # Check global blocks attend to all
    for i in range(num_global_blocks):
        if not jnp.all(mask[i, :]):
            return False
    
    # Check all blocks attend to global blocks
    for i in range(num_blocks):
        if not jnp.all(mask[i, :num_global_blocks]):
            return False
    
    # Check window attention (at least self-attention)
    for i in range(num_blocks):
        if not mask[i, i]:
            return False
    
    return True


# Gradient checkpointing utilities
def checkpoint_attention_block(attention_fn, *args, **kwargs):
    """Apply gradient checkpointing to attention computation.
    
    Args:
        attention_fn: Attention function to checkpoint
        *args: Positional arguments for attention function
        **kwargs: Keyword arguments for attention function
        
    Returns:
        Checkpointed attention output
    """
    return jax.checkpoint(attention_fn)(*args, **kwargs)


def create_causal_mask(seq_len: int, block_size: int) -> jnp.ndarray:
    """Create causal mask for autoregressive generation.
    
    Args:
        seq_len: Sequence length
        block_size: Block size for block-wise operations
        
    Returns:
        Causal mask [seq_len, seq_len]
    """
    # Input validation
    assert isinstance(seq_len, int) and seq_len > 0, \
        f"seq_len must be positive integer, got {seq_len}"
    assert isinstance(block_size, int) and block_size > 0, \
        f"block_size must be positive integer, got {block_size}"
    
    # Create standard causal mask
    mask = jnp.tril(jnp.ones((seq_len, seq_len), dtype=bool))
    
    # Final shape validation
    expected_shape = (seq_len, seq_len)
    assert mask.shape == expected_shape, \
        f"Output shape mismatch: expected {expected_shape}, got {mask.shape}"
    
    return mask