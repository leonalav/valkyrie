"""Gryphon Configuration

Extended configuration class for the hybrid BigBird-S5 architecture.
Includes all parameters needed for sparse attention patterns and S5 integration.
"""

import jax.numpy as jnp
from dataclasses import dataclass
from typing import Optional, List, Tuple
from ..modules import ValkyrieConfig


@dataclass
class GryphonConfig(ValkyrieConfig):
    """Configuration for Gryphon hybrid BigBird-S5 model.
    
    Extends ValkyrieConfig with BigBird-specific sparse attention parameters.
    Follows the architectural guide's Blueprint A: alternating S5 and BigBird blocks.
    """
    
    # === BigBird Sparse Attention Configuration ===
    
    # Block configuration for sparse attention
    block_size: int = 64  # Tokens per block (TPU-optimized)
    num_global_blocks: int = 2  # Number of global attention blocks
    window_size: int = 3  # Window size in blocks (current + left + right)
    num_random_blocks: int = 2  # Random blocks per query block
    
    # Attention pattern configuration
    attention_dropout: float = 0.1  # Dropout for attention weights
    use_random_attention: bool = True  # Enable random attention patterns
    random_seed: int = 42  # Seed for deterministic random patterns
    
    # Memory and efficiency settings
    use_gradient_checkpointing: bool = True  # Checkpoint attention blocks
    use_mixed_precision: bool = True  # bfloat16 forward, float32 gradients
    max_blocks_per_query: int = 8  # Maximum blocks a query can attend to
    
    # === Hybrid Architecture Configuration ===
    
    # Block alternation pattern
    s5_blocks_per_layer: int = 1  # S5 blocks before each BigBird block
    bigbird_blocks_per_layer: int = 1  # BigBird blocks after S5 blocks
    
    # S5-specific overrides for hybrid use
    s5_state_dim: int = 128  # S5 state dimension (should be ~H for comparable complexity)
    s5_init_mode: str = "hippo"  # Use HiPPO initialization for long-range dependencies
    
    # === Global Token Coupling Configuration ===
    
    # Global token coupling for S5 state management
    use_global_coupling: bool = True  # Enable global token -> S5 state coupling
    coupling_frequency: int = 4  # Apply coupling every N layers (0 = every layer)
    coupling_strength: float = 0.1  # Default coupling gate strength
    coupling_hidden_dim: int = 64  # Hidden dimension for coupling MLP
    
    # === Training Configuration ===
    
    # Parameter-specific learning rates (following guide recommendations)
    s5_learning_rate_multiplier: float = 0.1  # S5 params need smaller LR
    attention_learning_rate_multiplier: float = 1.0  # Standard LR for attention
    
    # Numerical stability
    attention_temperature: float = 1.0  # Temperature scaling for attention
    epsilon: float = 1e-8  # Numerical stability epsilon
    
    # === Sequence Length Configuration ===
    
    # Maximum sequence length must be divisible by block_size
    max_sequence_length: int = 4096  # Must be divisible by block_size
    
    def __post_init__(self):
        """Validate configuration and compute derived parameters."""
        super().__post_init__()
        
        # Validate block size compatibility
        if self.max_sequence_length % self.block_size != 0:
            raise ValueError(
                f"max_sequence_length ({self.max_sequence_length}) must be "
                f"divisible by block_size ({self.block_size})"
            )
        
        # Compute derived parameters
        self.num_blocks = self.max_sequence_length // self.block_size
        
        # Validate global blocks
        if self.num_global_blocks >= self.num_blocks:
            raise ValueError(
                f"num_global_blocks ({self.num_global_blocks}) must be less than "
                f"num_blocks ({self.num_blocks})"
            )
        
        # Validate window size
        if self.window_size > self.num_blocks:
            self.window_size = self.num_blocks
            
        # Validate random blocks
        max_possible_random = self.num_blocks - self.num_global_blocks - self.window_size
        if self.num_random_blocks > max_possible_random:
            self.num_random_blocks = max(0, max_possible_random)
        
        # Compute attention complexity reduction
        full_attention_ops = self.num_blocks ** 2
        sparse_attention_ops = self.num_blocks * (
            self.num_global_blocks + self.window_size + self.num_random_blocks
        )
        self.sparsity_ratio = 1.0 - (sparse_attention_ops / full_attention_ops)
        
        # Validate S5 state dimension for comparable complexity
        if self.s5_state_dim > 2 * self.d_model:
            import warnings
            warnings.warn(
                f"s5_state_dim ({self.s5_state_dim}) is much larger than d_model "
                f"({self.d_model}). This may lead to excessive memory usage."
            )
    
    def get_attention_pattern_info(self) -> dict:
        """Get information about the sparse attention patterns."""
        return {
            'num_blocks': self.num_blocks,
            'block_size': self.block_size,
            'global_blocks': self.num_global_blocks,
            'window_size': self.window_size,
            'random_blocks': self.num_random_blocks,
            'sparsity_ratio': self.sparsity_ratio,
            'max_attention_per_block': (
                self.num_global_blocks + self.window_size + self.num_random_blocks
            )
        }
    
    def get_memory_estimates(self, batch_size: int = 1) -> dict:
        """Estimate memory usage for different components."""
        # S5 memory: complex64 hidden states + real parameters
        s5_hidden_memory = batch_size * self.max_sequence_length * self.s5_state_dim * 8  # complex64
        s5_param_memory = self.s5_state_dim * self.d_model * 4 * 4  # B, C matrices (real + imag)
        
        # BigBird memory: sparse attention matrices
        attention_memory = (
            batch_size * self.num_blocks * self.max_blocks_per_query * 
            self.block_size * self.d_model * 2  # bfloat16
        )
        
        # Total model parameters
        total_params = (
            self.vocab_size * self.d_model +  # Embeddings
            self.n_layers * (
                self.s5_state_dim * self.d_model * 4 +  # S5 parameters
                self.d_model * self.d_model * 4 +  # Attention Q,K,V,O
                self.d_model * 4 * self.d_model * 2  # FFN (if used)
            )
        )
        
        return {
            's5_hidden_memory_gb': s5_hidden_memory / (1024**3),
            's5_param_memory_gb': s5_param_memory / (1024**3),
            'attention_memory_gb': attention_memory / (1024**3),
            'total_params_millions': total_params / 1e6,
            'estimated_total_memory_gb': (
                s5_hidden_memory + s5_param_memory + attention_memory
            ) / (1024**3)
        }


# Predefined configurations for common use cases
def get_gryphon_small_config() -> GryphonConfig:
    """Small Gryphon model for experimentation."""
    return GryphonConfig(
        d_model=768,
        n_layers=12,
        n_heads=12,
        s5_state_dim=768,
        max_sequence_length=2048,
        block_size=64,
        num_global_blocks=2,
        window_size=3,
        num_random_blocks=2
    )


def get_gryphon_base_config() -> GryphonConfig:
    """Base Gryphon model for general use."""
    return GryphonConfig(
        d_model=1024,
        n_layers=24,
        n_heads=16,
        s5_state_dim=1024,
        max_sequence_length=4096,
        block_size=64,
        num_global_blocks=2,
        window_size=3,
        num_random_blocks=2
    )


def get_gryphon_large_config() -> GryphonConfig:
    """Large Gryphon model for high-performance applications."""
    return GryphonConfig(
        d_model=1536,
        n_layers=32,
        n_heads=24,
        s5_state_dim=1536,
        max_sequence_length=8192,
        block_size=64,
        num_global_blocks=4,
        window_size=5,
        num_random_blocks=3
    )