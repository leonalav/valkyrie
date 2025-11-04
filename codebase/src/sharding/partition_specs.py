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

# Logical axis names for 2D mesh (current configuration)
# Support both 2D (v4-16) and 3D (v4-32) mesh configurations
MP1 = 'model'  # Model parallel dimension 1 (updated for current mesh)
MP2 = 'y'      # Model parallel dimension 2 (legacy 2D mesh)  
DP = 'data'    # Data parallel dimension (updated for current mesh)

# 3D mesh axis names for v4-32 (semantic naming)
DATA = 'data'  # Data parallel dimension (batch sharding)
MODEL = 'model'  # Model parallel dimension (tensor parallel)
FSDP = 'fsdp'  # FSDP dimension (parameter sharding)

# ============================================================================
# Core PartitionSpec Patterns
# ============================================================================

# Replication (no sharding)
REPLICATED = P()

# Legacy 2D patterns (v4-8 compatibility)
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

# 3D mesh patterns for v4-32 with FSDP
# Data parallel patterns
DATA_SHARD = P(DATA)                    # Batch sharding
DATA_REPLICATED = P()                   # Replicated across data axis

# Model parallel patterns (1D tensor parallel)
MODEL_SHARD_ROW = P(MODEL, None)        # Row-wise model parallel
MODEL_SHARD_COL = P(None, MODEL)        # Column-wise model parallel

# FSDP patterns (parameter sharding)
FSDP_SHARD = P(FSDP)                    # Parameter sharding
FSDP_REPLICATED = P()                   # Replicated across FSDP axis

# 2D sharding patterns (model × fsdp)
W_2D_MODEL_FSDP = P(MODEL, FSDP)        # 2D sharding across model and fsdp
W_ROW_MODEL_FSDP = P(MODEL, FSDP, None) # Row-wise 2D sharding
W_COL_MODEL_FSDP = P(None, MODEL, FSDP) # Column-wise 2D sharding

# 3D activation patterns
ACT_BATCH = P(DATA, None)                           # Batch sharded activations [batch, ...]
ACT_BATCH_SEQ = P(DATA, None, None)                 # [batch, seq_len, hidden]
ACT_BATCH_SEQ_HEAD = P(DATA, None, MODEL, None)     # [batch, seq_len, n_heads, head_dim]
ACT_BATCH_SEQ_MODEL = P(DATA, None, MODEL)          # [batch, seq_len, model_dim]

# ============================================================================
# Layer-Specific Patterns
# ============================================================================

def get_embedding_specs() -> Dict[str, P]:
    """PartitionSpecs for embedding layers."""
    return REPLICATED


def get_attention_specs(use_2d_sharding: bool = True, use_3d_mesh: bool = False) -> Dict[str, P]:
    """
    PartitionSpecs for attention layers.
    
    Args:
        use_2d_sharding: Whether to use 2D sharding for weight matrices
        use_3d_mesh: Whether to use 3D mesh (v4-32) or 2D mesh (v4-8)
        
    Returns:
        Dictionary of PartitionSpecs for attention components
    """
    if use_3d_mesh:
        # 3D mesh patterns for v4-32
        if use_2d_sharding:
            # 2D sharding across model and fsdp axes
            weight_spec = W_COL_MODEL_FSDP    # P(None, 'model', 'fsdp')
            output_spec = W_ROW_MODEL_FSDP    # P('model', 'fsdp', None)
        else:
            # 1D sharding on model axis only
            weight_spec = MODEL_SHARD_COL     # P(None, 'model')
            output_spec = MODEL_SHARD_ROW     # P('model', None)
    else:
        # Legacy 2D mesh patterns for v4-8
        if use_2d_sharding:
            weight_spec = W_2D            # P('x','y') – only valid if mesh has 'y'
        else:
            weight_spec = W_COL_X         # 1D TP on 'x'
        output_spec = W_ROW_X             # row-shard on 'x'
    
    return {
        # Query, Key, Value projections
        'q_proj': weight_spec,
        'k_proj': weight_spec, 
        'v_proj': weight_spec,
        
        # Output projection (use row sharding to reduce communication)
        'o_proj': output_spec,
    }


def get_ffn_specs(use_2d_sharding: bool = True, use_3d_mesh: bool = False) -> Dict[str, P]:
    """
    PartitionSpecs for FFN layers.
    
    Args:
        use_2d_sharding: Whether to use 2D sharding for weight matrices
        use_3d_mesh: Whether to use 3D mesh (v4-32) or 2D mesh (v4-8)
        
    Returns:
        Dictionary of PartitionSpecs for FFN components
    """
    if use_3d_mesh:
        # 3D mesh patterns for v4-32
        if use_2d_sharding:
            # 2D sharding across model and fsdp axes
            input_spec = W_COL_MODEL_FSDP     # P(None, 'model', 'fsdp')
            output_spec = W_ROW_MODEL_FSDP    # P('model', 'fsdp', None)
        else:
            # 1D sharding on model axis only
            input_spec = MODEL_SHARD_COL      # P(None, 'model')
            output_spec = MODEL_SHARD_ROW     # P('model', None)
    else:
        # Legacy 2D mesh patterns for v4-8
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


def get_model_specs(config: Any, use_2d_sharding: bool = False, use_3d_mesh: bool = False) -> Dict[str, Any]:
    """
    Get complete PartitionSpecs for Valkyrie model.
    
    Args:
        config: Model configuration
        use_2d_sharding: Whether to use 2D tensor parallelism
        use_3d_mesh: Whether to use 3D mesh (v4-32) or 2D mesh (v4-8)
        
    Returns:
        Nested dictionary of PartitionSpecs for all model components
    """
    specs = {
        # Embedding layer - shard vocab dimension for large models
        'embedding': {
            'embedding': MODEL_SHARD_ROW if use_3d_mesh else get_embedding_specs()
        },
        # Layer normalization - keep replicated (small parameters)
        'norm': get_layernorm_specs(),
    }
    
    # Add specs for each transformer block
    for i in range(config.n_layers):
        block_specs = {
            'norm1': get_layernorm_specs(),
            'norm2': get_layernorm_specs(),
        }
        
        # Attention layer
        if config.use_bigbird_attention:
            block_specs['attn'] = get_attention_specs(use_2d_sharding, use_3d_mesh)
        else:
            block_specs['attn'] = get_attention_specs(use_2d_sharding, use_3d_mesh)
        
        # FFN or S5 layer
        if config.use_s5:
            block_specs['s5'] = get_s5_specs()
        else:
            block_specs['ffn'] = get_ffn_specs(use_2d_sharding, use_3d_mesh)
        
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


def get_training_specs(use_2d_sharding: bool = False, use_3d_mesh: bool = False) -> Dict[str, P]:
    """
    Get PartitionSpecs for training state (params, optimizer state, etc.).
    
    Args:
        use_2d_sharding: Whether to use 2D tensor parallelism
        use_3d_mesh: Whether to use 3D mesh (v4-32) or 2D mesh (v4-8)
        
    Returns:
        Dictionary of PartitionSpecs for training components
    """
    if use_3d_mesh:
        # 3D mesh patterns - match parameter sharding
        return {
            'params': None,  # Will be filled by model specs
            'opt_state': None,  # Mirror parameter sharding
            'step': REPLICATED,  # Step counter replicated
            'rng': REPLICATED,  # RNG state replicated
            'batch': {
                'input_ids': DATA_SHARD,      # P('data', None, None)
                'attention_mask': DATA_SHARD,  # P('data', None, None)
                'labels': DATA_SHARD,         # P('data', None, None)
            }
        }
    else:
        # Legacy 2D mesh patterns for v4-8
        return {
            'params': None,  # Will be filled by model specs
            'opt_state': None,  # Mirror parameter sharding
            'step': REPLICATED,
            'rng': REPLICATED,
            'batch': {
                'input_ids': BATCH_SHARD,    # P('z')
                'attention_mask': BATCH_SHARD,
                'labels': BATCH_SHARD,
            }
        }