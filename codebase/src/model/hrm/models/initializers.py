"""
Initialization utilities for HRM JAX implementation.

Provides truncated LeCun normal initialization matching the PyTorch implementation.
This is critical for proper model convergence and matches the original HRM paper.
"""

import jax
import jax.numpy as jnp
from jax import random
from typing import Tuple, Optional
import math


def truncated_lecun_normal(
    key: jax.Array,
    shape: Tuple[int, ...],
    dtype: jnp.dtype = jnp.float32,
    truncation: float = 2.0
) -> jax.Array:
    """
    Truncated LeCun normal initialization.
    
    This matches the PyTorch implementation in common.py:trunc_normal_init_
    Uses LeCun normal scaling (fan_in) with truncation at ±2 standard deviations.
    
    Args:
        key: JAX random key
        shape: Shape of the tensor to initialize
        dtype: Data type (default: float32)
        truncation: Truncation threshold in standard deviations (default: 2.0)
        
    Returns:
        Initialized tensor with truncated LeCun normal distribution
    """
    if len(shape) < 2:
        # For 1D tensors (biases), use standard normal
        fan_in = 1
    else:
        # For weight matrices, fan_in is the input dimension
        fan_in = shape[-2] if len(shape) >= 2 else shape[0]
    
    # LeCun normal scaling: std = sqrt(1 / fan_in)
    std = jnp.sqrt(1.0 / fan_in)
    
    # Generate truncated normal samples directly without nested function
    samples = random.normal(key, shape, dtype=dtype)
    # Truncate to [-truncation, truncation] standard deviations
    samples = jnp.clip(samples, -truncation, truncation)
    return samples * std


def lecun_normal(
    key: jax.Array,
    shape: Tuple[int, ...],
    dtype: jnp.dtype = jnp.float32
) -> jax.Array:
    """
    Standard LeCun normal initialization (without truncation).
    
    Args:
        key: JAX random key
        shape: Shape of the tensor to initialize
        dtype: Data type (default: float32)
        
    Returns:
        Initialized tensor with LeCun normal distribution
    """
    if len(shape) < 2:
        fan_in = 1
    else:
        fan_in = shape[-2] if len(shape) >= 2 else shape[0]
    
    std = math.sqrt(1.0 / fan_in)
    return random.normal(key, shape, dtype=dtype) * std


def zeros_init(
    key: jax.Array,
    shape: Tuple[int, ...],
    dtype: jnp.dtype = jnp.float32
) -> jax.Array:
    """
    Zero initialization.
    
    Args:
        key: JAX random key (unused but kept for consistency)
        shape: Shape of the tensor to initialize
        dtype: Data type (default: float32)
        
    Returns:
        Zero-initialized tensor
    """
    return jnp.zeros(shape, dtype=dtype)


def constant_init(
    key: jax.Array,
    shape: Tuple[int, ...],
    value: float,
    dtype: jnp.dtype = jnp.float32
) -> jax.Array:
    """
    Constant initialization.
    
    Args:
        key: JAX random key (unused but kept for consistency)
        shape: Shape of the tensor to initialize
        value: Constant value to initialize with
        dtype: Data type (default: float32)
        
    Returns:
        Constant-initialized tensor
    """
    return jnp.full(shape, value, dtype=dtype)


# Convenience function for Q-head bias initialization (bias to -5 as per notes)
def q_head_bias_init(
    key: jax.Array,
    shape: Tuple[int, ...],
    dtype: jnp.dtype = jnp.float32
) -> jax.Array:
    """
    Q-head bias initialization to -5.0 as specified in the notes.
    
    Args:
        key: JAX random key (unused but kept for consistency)
        shape: Shape of the tensor to initialize
        dtype: Data type (default: float32)
        
    Returns:
        Bias tensor initialized to -5.0
    """
    return constant_init(key, shape, -5.0, dtype)