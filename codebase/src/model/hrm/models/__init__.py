"""HRM Models package for JAX/Flax implementation.

This package contains the core components of the Hierarchical Reasoning Model:
- Initializers: Weight initialization functions
- Rotary: Rotary positional embeddings (RoPE)
- Blocks: Transformer blocks, RMSNorm, SwiGLU
- Attention: Multi-head attention with RoPE support
- HRMInner: Core reasoning module with one-step gradient
- HRMWithACT: Adaptive Computation Time wrapper
"""

from .initializers import (
    truncated_lecun_normal,
    lecun_normal,
    zeros_init,
    constant_init,
    q_head_bias_init
)

from .rotary import (
    RotaryEmbedding,
    make_rope,
    apply_rotary,
    rotate_half,
    apply_rotary_pos_emb_legacy
)

from .blocks import (
    RMSNorm,
    rms_norm,
    SwiGLU,
    TransformerBlock,
    make_causal_mask,
    find_multiple
)

from .attention import (
    Attention,
    FlashAttention
)

from .hrm_inner import (
    HRMInner,
    HRMInnerCarry,
    HRMReasoningModule
)

from .hrm_act import (
    HRMWithACT,
    ACTState,
    ACTOutput,
    compute_act_loss,
    compute_efficiency_metrics
)

from ..config import (
    HRMConfig,
    get_hrm_small_config,
    get_hrm_base_config,
    get_hrm_large_config
)

__all__ = [
    # Initializers
    "truncated_lecun_normal",
    "lecun_normal", 
    "zeros_init",
    "constant_init",
    "q_head_bias_init",
    
    # Rotary embeddings
    "RotaryEmbedding",
    "make_rope",
    "apply_rotary",
    "rotate_half",
    "apply_rotary_pos_emb_legacy",
    
    # Blocks
    "RMSNorm",
    "rms_norm",
    "SwiGLU", 
    "TransformerBlock",
    "make_causal_mask",
    "find_multiple",
    
    # Attention
    "Attention",
    "FlashAttention",
    
    # HRM Inner
    "HRMInner",
    "HRMInnerCarry",
    "HRMReasoningModule",
    
    # HRM with ACT
    "HRMWithACT",
    "ACTState",
    "ACTOutput",
    "compute_act_loss",
    "compute_efficiency_metrics",
    
    # Configuration
    "HRMConfig",
    "get_hrm_small_config",
    "get_hrm_base_config",
    "get_hrm_large_config"
]