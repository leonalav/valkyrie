"""TPU sharding utilities for distributed training."""

from .mesh_setup import make_mesh, get_mesh_context
from .partition_specs import *

__all__ = [
    "make_mesh",
    "get_mesh_context", 
    "W_2D",
    "W_ROW", 
    "W_COL",
    "EMBED_ROW",
    "REPLICATED",
    "MP1",
    "MP2", 
    "DP",
]