"""
HRM Configuration Classes

This module contains configuration classes for the Hierarchical Reasoning Model (HRM).
"""

import jax.numpy as jnp
from dataclasses import dataclass
from typing import Optional


@dataclass
class HRMConfig:
    """Configuration for HRM models."""
    
    # Core model configuration
    vocab_size: int = 50257
    hidden_size: int = 768
    seq_len: int = 1024
    puzzle_emb_ndim: int = 0
    num_puzzle_identifiers: int = 0
    
    # Hierarchical reasoning config
    H_cycles: int = 3
    L_cycles: int = 3
    H_layers: int = 6
    L_layers: int = 6
    
    # Transformer config
    num_heads: int = 8
    num_key_value_heads: Optional[int] = None
    intermediate_size: Optional[int] = None
    eps: float = 1e-5
    
    # Positional encoding
    pos_encodings: str = "rope"
    rope_theta: float = 10000.0
    
    # ACT-specific configuration
    max_steps: int = 10
    exploration_prob: float = 0.1
    q_target_discount: float = 0.95
    min_steps: int = 1  # Minimum steps before allowing halt
    
    # Data types
    dtype: jnp.dtype = jnp.bfloat16
    param_dtype: jnp.dtype = jnp.float32


def get_hrm_small_config() -> HRMConfig:
    """Get a small HRM configuration for testing."""
    return HRMConfig(
        vocab_size=1000,
        hidden_size=256,
        seq_len=512,
        H_cycles=2,
        L_cycles=2,
        H_layers=4,
        L_layers=4,
        num_heads=4,
        max_steps=5
    )


def get_hrm_base_config() -> HRMConfig:
    """Get a base HRM configuration."""
    return HRMConfig()


def get_hrm_large_config() -> HRMConfig:
    """Get a large HRM configuration."""
    return HRMConfig(
        vocab_size=50257,
        hidden_size=1024,
        seq_len=2048,
        H_cycles=4,
        L_cycles=4,
        H_layers=8,
        L_layers=8,
        num_heads=16,
        max_steps=15
    )