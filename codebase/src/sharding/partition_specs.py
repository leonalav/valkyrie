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

# Activation patterns
ACT_BATCH = P(DP, None)                    # Batch sharded activations [batch, ...]
ACT_BATCH_SEQ = P(DP, None, None)          # [batch, seq_len, hidden]
ACT_BATCH_SEQ_HEAD = P(DP, None, MP1, None)  # [batch, seq_len, n_heads, head_dim]

# ============================================================================
# Layer-Specific Patterns
# ============================================================================

def get_embedding_specs() -> Dict[str, P]:
    """PartitionSpecs for embedding layers."""
    return {
        'embedding': EMBED_ROW,  # Shard vocab dimension across MP1
    }


def get_attention_specs(use_2d_sharding: bool = True) -> Dict[str, P]:
    """
    PartitionSpecs for attention layers.
    
    Args:
        use_2d_sharding: Whether to use 2D sharding for weight matrices
        
    Returns:
        Dictionary of PartitionSpecs for attention components
    """
    if use_2d_sharding:
        # 2D sharding for better memory distribution and scaling
        weight_spec = W_2D
    else:
        # 1D sharding for simpler setup
        weight_spec = W_COL
    
    return {
        # Query, Key, Value projections
        'q_proj': weight_spec,
        'k_proj': weight_spec, 
        'v_proj': weight_spec,
        
        # Longformer global projections
        'qg_proj': weight_spec,
        'kg_proj': weight_spec,
        'vg_proj': weight_spec,
        
        # Output projection (use row sharding to reduce communication)
        'o_proj': W_ROW,
        
        # Activations
        'queries': ACT_BATCH_SEQ_HEAD,
        'keys': ACT_BATCH_SEQ_HEAD,
        'values': ACT_BATCH_SEQ_HEAD,
        'attention_output': ACT_BATCH_SEQ,
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
        # 2D sharding for input projections
        input_spec = W_2D
        # Row sharding for output to reduce all-gather
        output_spec = W_ROW
    else:
        # 1D column sharding
        input_spec = W_COL
        output_spec = W_ROW
    
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
        # S5 parameters (keep replicated for numerical stability)
        'Lambda_re': REPLICATED,
        'Lambda_im': REPLICATED, 
        'B_real': REPLICATED,
        'B_imag': REPLICATED,
        'C_real': REPLICATED,
        'C_imag': REPLICATED,
        'D': REPLICATED,
        'log_Delta': REPLICATED,
        
        # S5 states (shard batch dimension only)
        's5_state': P(DP, None),  # [batch, state_dim]
        's5_output': ACT_BATCH_SEQ,  # [batch, seq_len, d_model]
    }


def get_layernorm_specs() -> Dict[str, P]:
    """PartitionSpecs for layer normalization."""
    return {
        'weight': REPLICATED,  # Small parameters, keep replicated
        'bias': REPLICATED,    # Small parameters, keep replicated
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
        'embedding': get_embedding_specs(),
        
        # Layer normalization
        'norm': get_layernorm_specs(),
        
        # Transformer blocks
        'blocks': {},
    }
    
    # Add specs for each transformer block
    for i in range(config.n_layers):
        block_specs = {
            'norm1': get_layernorm_specs(),
            'norm2': get_layernorm_specs(),
        }
        
        # Attention layer
        if config.use_longformer_attention:
            block_specs['attn'] = get_attention_specs(use_2d_sharding)
        else:
            block_specs['attn'] = get_attention_specs(use_2d_sharding)
        
        # FFN or S5 layer
        if config.use_s5:
            block_specs['s5'] = get_s5_specs()
        else:
            block_specs['ffn'] = get_ffn_specs(use_2d_sharding)
        
        specs['blocks'][f'block_{i}'] = block_specs
    
    return specs


def get_training_specs(model_specs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get PartitionSpecs for training state (params + optimizer).
    
    Args:
        model_specs: Model PartitionSpecs from get_model_specs()
        
    Returns:
        Training state PartitionSpecs
    """
    def add_optimizer_specs(param_specs):
        """Recursively add optimizer specs for parameters."""
        if isinstance(param_specs, P):
            return get_optimizer_specs(param_specs)
        elif isinstance(param_specs, dict):
            return {k: add_optimizer_specs(v) for k, v in param_specs.items()}
        else:
            return param_specs
    
    return {
        'params': model_specs,
        'opt_state': add_optimizer_specs(model_specs),
        'step': REPLICATED,
        'rng': REPLICATED,
    }


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