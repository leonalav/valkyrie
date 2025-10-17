"""Gryphon: Hybrid BigBird-S5 Architecture

A surgical fusion of BigBird's sparse attention with S5's state space modeling.
Implements Blueprint A from the architectural guide: alternating S5 and BigBird blocks.

Key Components:
- GryphonConfig: Extended configuration for hybrid architecture
- BigBirdSparseAttention: JAX-native sparse attention patterns
- GryphonBlock: Hybrid S5 + BigBird processing block
- GryphonModel: Complete model with embedding and output layers

Mathematical Foundation:
- S5: O(L) sequential processing with HiPPO initialization
- BigBird: O(L) sparse attention with window + global + random patterns
- Synergy: S5 enriches local context → BigBird routes globally

Optimizations:
- Block-wise operations for TPU efficiency
- Mixed precision training (bfloat16/float32)
- Gradient checkpointing for memory efficiency
- Parameter-specific learning rates
"""

from .gryphon_config import GryphonConfig
from .bigbird_attention import BigBirdSparseAttention
from .gryphon_blocks import S5Block, BigBirdBlock, GryphonBlock
from .gryphon_model import GryphonModel
from .gryphon_utils import (
    create_sparse_attention_mask,
    get_random_block_indices,
    pad_to_block_size
)

__all__ = [
    'GryphonConfig',
    'BigBirdSparseAttention', 
    'S5Block',
    'BigBirdBlock',
    'GryphonBlock',
    'GryphonModel',
    'create_sparse_attention_mask',
    'get_random_block_indices',
    'pad_to_block_size'
]