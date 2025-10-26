"""PartitionSpec patterns for TPU sharding.

Implements canonical sharding patterns from precautionfortpu.md:
- 2D tensor parallelism across x,y axes
- Data parallelism across z axis  
- Optimized patterns for different layer types
- Consistent naming and reusable specs
"""

from jax.sharding import PartitionSpec as P
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)

# Logical axis names matching mesh_setup.py
MP1 = 'x'  # Model parallel dimension 1 (tensor parallel width)
MP2 = 'y'  # Model parallel dimension 2 (tensor parallel height)  
DP = 'z'   # Data parallel dimension (batch sharding)

# ============================================================================
# Core PartitionSpec Patterns
# ============================================================================

# Replication (no sharding)
REPLICATED = P()

# 1D sharding patterns
W_ROW = P(MP1, None)      # Row-wise sharding (split rows across MP1)
W_COL = P(None, MP2)      # Column-wise sharding (split columns across MP2)
EMBED_ROW = P(MP1)        # Embedding row sharding (vocab dimension)
BATCH_SHARD = P(DP)       # Batch dimension sharding

# 2D sharding patterns  
W_2D = P(MP1, MP2)        # 2D weight sharding (both dimensions)
ATTN_2D = P(MP1, MP2)     # 2D attention weight sharding

# 1D-on-x helpers (use these when use_2d_sharding=False)
W_ROW_X = P(MP1, None)    # row-shard on x
W_COL_X = P(None, MP1)    # col-shard on x

# Activation patterns
ACT_BATCH = P(DP, None)                    # Batch sharded activations [batch, ...]
ACT_BATCH_SEQ = P(DP, None, None)          # [batch, seq_len, hidden]
ACT_BATCH_SEQ_HEAD = P(DP, None, MP1, None)  # [batch, seq_len, n_heads, head_dim]

# ============================================================================
# Layer-Specific Patterns
# ============================================================================

def get_embedding_specs() -> Dict[str, P]:
    """PartitionSpecs for embedding layers."""
    return REPLICATED


def get_attention_specs(use_2d_sharding: bool = True) -> Dict[str, P]:
    """
    PartitionSpecs for attention layers.
    
    Args:
        use_2d_sharding: Whether to use 2D sharding for weight matrices
        
    Returns:
        Dictionary of PartitionSpecs for attention components
    """
    if use_2d_sharding:
        weight_spec = W_2D            # P('x','y') – only valid if mesh has 'y'
    else:
        weight_spec = W_COL_X         # 1D TP on 'x'
    
    return {
        # Query, Key, Value projections (only the ones that exist in actual model)
        'q_proj': weight_spec,
        'k_proj': weight_spec, 
        'v_proj': weight_spec,
        
        # Output projection (use row sharding to reduce communication)
        'o_proj': W_ROW_X,            # row-shard on 'x'
    }


def get_ffn_specs(use_2d_sharding: bool = True) -> Dict[str, P]:
    """
    PartitionSpecs for FFN layers.
    
    Args:
        use_2d_sharding: Whether to use 2D sharding for weight matrices
        
    Returns:
        Dictionary of PartitionSpecs for FFN components
    """
    if use_2d_sharding:
        input_spec  = W_2D
        output_spec = W_ROW
    else:
        input_spec  = W_COL_X         # gate/up on 'x'
        output_spec = W_ROW_X         # down on 'x'
    
    return {
        # SwiGLU projections
        'gate_proj': input_spec,
        'up_proj': input_spec,
        'down_proj': output_spec,
        
        # Activations
        'hidden_states': ACT_BATCH_SEQ,
        'gate_output': ACT_BATCH_SEQ,
        'up_output': ACT_BATCH_SEQ,
    }


def get_s5_specs() -> Dict[str, P]:
    """
    PartitionSpecs for S5 state space layers.
    
    S5 parameters are relatively small but numerically sensitive.
    Use conservative sharding to maintain stability.
    """
    return {
        # S5 layer (actual structure has single s5_layer key)
        's5_layer': REPLICATED,  # Keep replicated for numerical stability
    }


def get_layernorm_specs() -> Dict[str, P]:
    """PartitionSpecs for layer normalization."""
    return {
        'weight': REPLICATED,  # Small parameters, keep replicated
        # Note: bias removed as it doesn't exist in actual model
    }


def get_optimizer_specs(param_spec: P) -> Dict[str, P]:
    """
    Get optimizer state PartitionSpecs matching parameter sharding.
    
    Args:
        param_spec: PartitionSpec of the parameter
        
    Returns:
        Dictionary of optimizer state PartitionSpecs
    """
    return {
        'params': param_spec,
        'momentum': param_spec,  # First moment (Adam m)
        'variance': param_spec,  # Second moment (Adam v)
        'step': REPLICATED,      # Step counter (scalar)
    }


# ============================================================================
# Model-Level Patterns
# ============================================================================

def get_hrm_specs() -> Dict[str, P]:
    """PartitionSpecs for HRM (Hierarchical Reasoning Model) components."""
    return {
        'executor': REPLICATED,  # Keep HRM executor replicated for now
        'planner': REPLICATED,   # Keep HRM planner replicated for now
    }


def get_model_specs(config: Any, use_2d_sharding: bool = True) -> Dict[str, Any]:
    """
    Get complete PartitionSpecs for Valkyrie model.
    
    Args:
        config: Model configuration
        use_2d_sharding: Whether to use 2D tensor parallelism
        
    Returns:
        Nested dictionary of PartitionSpecs for all model components
    """
    specs = {
        # Embedding layer
        'embedding': {
            'embedding': get_embedding_specs()
            
        },
        # Layer normalization
        'norm': get_layernorm_specs(),
    }
    
    # Add specs for each transformer block (individual keys, not nested under 'blocks')
    for i in range(config.n_layers):
        block_specs = {
            'norm1': get_layernorm_specs(),
            'norm2': get_layernorm_specs(),
        }
        
        # Attention layer
        if config.use_bigbird_attention:
            block_specs['attn'] = get_attention_specs(use_2d_sharding)
        else:
            block_specs['attn'] = get_attention_specs(use_2d_sharding)
        
        # FFN or S5 layer
        if config.use_s5:
            block_specs['s5'] = get_s5_specs()
        else:
            block_specs['ffn'] = get_ffn_specs(use_2d_sharding)
        
        specs[f'block_{i}'] = block_specs
    
    # Add HRM specs if enabled
    if hasattr(config, 'use_hrm') and config.use_hrm:
        specs['hrm'] = get_hrm_specs()
    
    return specs


# ============================================================================
# Utility Functions
# ============================================================================

def print_partition_specs(specs: Dict[str, Any], prefix: str = ""):
    """Print PartitionSpecs in a readable format."""
    for key, value in specs.items():
        current_path = f"{prefix}.{key}" if prefix else key
        
        if isinstance(value, P):
            print(f"{current_path}: {value}")
        elif isinstance(value, dict):
            print(f"{current_path}:")
            print_partition_specs(value, current_path)
        else:
            print(f"{current_path}: {value}")


def validate_partition_specs(specs: Dict[str, Any], mesh_axes: tuple):
    """
    Validate that PartitionSpecs use only valid mesh axes.
    
    Args:
        specs: PartitionSpecs to validate
        mesh_axes: Valid mesh axis names
    """
    def check_spec(spec, path):
        if isinstance(spec, P):
            for axis in spec:
                if axis is not None and axis not in mesh_axes:
                    raise ValueError(f"Invalid axis '{axis}' in PartitionSpec at {path}")
        elif isinstance(spec, dict):
            for key, value in spec.items():
                check_spec(value, f"{path}.{key}")
    
    check_spec(specs, "root")
    logger.info("✓ PartitionSpec validation passed")


# ============================================================================
# Presets for Common Configurations
# ============================================================================

# Standard 2D tensor parallel configuration
STANDARD_2D_CONFIG = {
    'use_2d_sharding': True,
    'shard_embeddings': True,
    'shard_s5': False,  # Keep S5 replicated for stability
}

# Memory-optimized configuration (more aggressive sharding)
MEMORY_OPTIMIZED_CONFIG = {
    'use_2d_sharding': True,
    'shard_embeddings': True,
    'shard_s5': True,  # Shard S5 if memory is tight
}

# Simple 1D configuration (easier debugging)
SIMPLE_1D_CONFIG = {
    'use_2d_sharding': False,
    'shard_embeddings': True,
    'shard_s5': False,
}


def get_training_specs(model_specs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get PartitionSpecs for training state (TrainingState).
    
    Note: We intentionally do NOT specify opt_state sharding here.
    The optimizer state structure depends on the specific optimizer chain
    (e.g., optax.chain with clip_by_global_norm + adamw) and should be
    inferred by JAX to avoid pytree structure mismatches.
    
    Args:
        model_specs: Model parameter PartitionSpecs from get_model_specs()
        
    Returns:
        PartitionSpecs for TrainingState fields
    """
    return {
        'params': model_specs,
        # opt_state: Intentionally omitted - let JAX infer sharding
        'step': REPLICATED,
        'rng': REPLICATED,
        's5_states': REPLICATED,  # Keep S5 states replicated for now
        'chunk_position': REPLICATED,
        'phase_index': REPLICATED,
        'hrm_enabled': REPLICATED,
        'hrm_training_state': REPLICATED,
    }