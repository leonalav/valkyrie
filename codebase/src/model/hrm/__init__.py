"""
Hierarchical Reasoning Model (HRM) - JAX/Flax Implementation for TPU

This package contains the JAX/Flax implementation of the Hierarchical Reasoning Model,
optimized for TPU training with proper gradient handling and ACT (Adaptive Computation Time).

Key components:
- models/: Core model implementations (HRMInner, HRM with ACT wrapper)
- data/: Data loading and preprocessing utilities
- training/: Training step implementations with deep supervision
- eval/: Evaluation metrics and utilities
- tests/: Unit tests for gradient handling and shape validation
"""

from .models import *
from .data import *  # This imports from data.py
from .training import *
from .eval import *

# Import the data.py module directly to avoid conflicts
from . import data as data_module

__version__ = "1.0.0"
__author__ = "HRM JAX Implementation"