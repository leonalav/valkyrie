"""Gryphon Configuration

Extended configuration class for the hybrid BigBird-S5 architecture.
Includes all parameters needed for sparse attention patterns and S5 integration.
"""

import jax.numpy as jnp
from dataclasses import dataclass
from typing import Optional, List, Tuple

# Robust import of ValkyrieConfig whether this file is imported as a package or standalone
try:
    from ..modules import ValkyrieConfig
except Exception:
    # Fallback to dynamic import via file path when not imported as a package
    import importlib.util as _ilu
    import os as _os
    _model_dir = _os.path.dirname(_os.path.dirname(__file__))  # .../src/model
    _modules_py = _os.path.join(_model_dir, 'modules.py')
    _spec = _ilu.spec_from_file_location("valkyrie_modules", _modules_py)
    _mod = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    ValkyrieConfig = _mod.ValkyrieConfig


@dataclass
class GryphonConfig(ValkyrieConfig):
    """Configuration for Gryphon hybrid BigBird-S5-HRM model.
    
    Extends ValkyrieConfig with BigBird-specific sparse attention parameters
    and HRM hierarchical reasoning configuration. Implements the PLAN blueprint
    for 1.2B parameter model with 64k context length.
    """
    
    # === BigBird Sparse Attention Configuration ===
    
    # Block configuration for sparse attention (updated for 64k context)
    block_size: int = 128  # Tokens per block (doubled for 64k efficiency)
    num_global_blocks: int = 16  # g=16 global tokens (HRM planner tokens)
    window_size: int = 3  # w=3 window blocks
    num_random_blocks: int = 3  # r=3 random blocks
    
    # Legacy parameter aliases for backward compatibility
    max_seq_len: Optional[int] = None  # Alias for max_position_embeddings
    num_rand_blocks: Optional[int] = None  # Alias for num_random_blocks
    
    # Attention pattern configuration
    attention_dropout: float = 0.1  # Dropout for attention weights
    use_random_attention: bool = True  # Enable random attention patterns
    random_seed: int = 42  # Seed for deterministic random patterns
    
    # Memory and efficiency settings
    use_gradient_checkpointing: bool = True  # Checkpoint attention blocks
    use_mixed_precision: bool = True  # bfloat16 forward, float32 gradients
    max_blocks_per_query: int = 8  # Maximum blocks a query can attend to
    param_dtype: jnp.dtype = jnp.bfloat16  # Parameter dtype for mixed precision
    
    # === Hybrid Architecture Configuration ===
    
    # Block alternation pattern
    s5_blocks_per_layer: int = 1  # S5 blocks before each BigBird block
    bigbird_blocks_per_layer: int = 1  # BigBird blocks after S5 blocks
    
    # S5-specific overrides for hybrid use (updated for 1.2B model)
    s5_state_dim: int = 768  # S5 state dimension P=768 for Config B
    s5_init_mode: str = "hippo_n"  # Use HiPPO-N initialization for long-range dependencies
    s5_use_diagonal: bool = True  # Diagonal SSM for efficiency
    s5_use_conjugate_symmetry: bool = True  # Conjugate pairs for stability
    s5_learnable_delta: bool = True  # Learnable timescales (ZOH discretization)
    s5_continuous_time: bool = True  # Continuous-time parameterization
    
    # === HRM (Hierarchical Reasoning Model) Configuration ===
    
    # HRM core settings
    hrm_enabled: bool = True  # Enable HRM integration
    hrm_planner_layers: int = 2  # Tiny 2-layer Transformer for planner
    hrm_planner_hidden_dim: int = 512  # Hidden dimension for planner MLP
    hrm_executor_cycles: int = 2  # N=2 cycles (start conservative)
    hrm_executor_steps: int = 2  # T=2 steps per cycle (start conservative)
    
    # HRM training configuration
    hrm_use_one_step_gradient: bool = True  # Use HRM's one-step gradient approximation
    hrm_detach_states: bool = True  # Detach states between segments
    hrm_deep_supervision_weight: float = 1.0  # Weight for segment-level losses
    
    # HRM Adaptive Computation Time (ACT)
    hrm_use_act: bool = False  # Disable ACT initially, enable in Phase 3
    hrm_act_penalty: float = 0.01  # Ponder cost when ACT is enabled
    hrm_max_steps: int = 10  # Maximum ACT steps
    hrm_min_steps: int = 1  # Minimum steps before allowing halt
    hrm_exploration_prob: float = 0.1  # Exploration probability for ACT
    hrm_q_target_discount: float = 0.95  # Q-learning discount factor
    
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
    
    # Maximum sequence length must be divisible by block_size (updated for 64k)
    max_sequence_length: int = 65536  # 64k context as specified in PLAN
    
    def __post_init__(self):
        """Validate configuration and compute derived parameters."""
        
        # Handle backward compatibility aliases
        if self.max_seq_len is not None:
            if not hasattr(self, 'max_position_embeddings') or self.max_position_embeddings is None:
                self.max_position_embeddings = self.max_seq_len
            if not hasattr(self, 'max_sequence_length') or self.max_sequence_length == 65536:  # Default value
                self.max_sequence_length = self.max_seq_len
        
        if self.num_rand_blocks is not None:
            self.num_random_blocks = self.num_rand_blocks
        
        super().__post_init__()
        
        # Validate block size compatibility
        if self.max_sequence_length % self.block_size != 0:
            raise ValueError(
                f"max_sequence_length ({self.max_sequence_length}) must be "
                f"divisible by block_size ({self.block_size})"
            )
        
        # Compute derived parameters
        self.num_blocks = self.max_sequence_length // self.block_size
        
        # Adjust num_global_blocks if it's too large for the sequence length
        if self.num_global_blocks >= self.num_blocks:
            self.num_global_blocks = max(1, self.num_blocks // 4)  # Use 1/4 of blocks as global
        
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


def get_gryphon_1_2b_config() -> GryphonConfig:
    """1.2B Gryphon model following PLAN Config B specifications."""
    return GryphonConfig(
        # Core architecture (Config B: deeper, same width)
        d_model=1536,
        n_layers=36,
        n_heads=24,
        vocab_size=32000,  # SentencePiece as specified in PLAN
        
        # S5 configuration for 1.2B model
        s5_state_dim=768,  # P=768 for Config B
        s5_init_mode="hippo_n",
        s5_use_diagonal=True,
        s5_use_conjugate_symmetry=True,
        s5_learnable_delta=True,
        s5_continuous_time=True,
        
        # BigBird configuration for 64k context
        max_sequence_length=65536,  # 64k context
        block_size=128,  # Doubled for 64k efficiency
        num_global_blocks=16,  # g=16 planner tokens
        window_size=3,  # w=3
        num_random_blocks=3,  # r=3
        
        # HRM configuration
        hrm_enabled=True,
        hrm_planner_layers=2,
        hrm_planner_hidden_dim=512,
        hrm_executor_cycles=2,  # Start conservative
        hrm_executor_steps=2,
        hrm_use_one_step_gradient=True,
        hrm_detach_states=True,
        hrm_use_act=False,  # Enable in Phase 3
        
        # Training optimizations
        use_gradient_checkpointing=True,
        use_mixed_precision=True,
        use_global_coupling=True,
        coupling_frequency=4
    )


def get_gryphon_large_config() -> GryphonConfig:
    """Large Gryphon model for high-performance applications."""
    return GryphonConfig(
        d_model=2048,
        n_layers=48,
        n_heads=32,
        s5_state_dim=2048,
        max_sequence_length=8192,
        block_size=128,
        num_global_blocks=4,
        window_size=3,
        num_random_blocks=3,
        
        # Training optimizations for large model
        use_gradient_checkpointing=True,
        use_mixed_precision=True,
        use_global_coupling=True,
        coupling_frequency=8
    )