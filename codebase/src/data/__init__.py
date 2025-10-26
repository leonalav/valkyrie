"""Data Pipeline Module

Multi-source data loading and processing for BigBird+S5+HRM training.
Implements the PLAN data pipeline with proper deduplication, packing, and phase-wise sampling.
"""

# Legacy FineWeb components
from .fineweb_reader import FineWebDataset, create_data_loader
from .tokenizer import create_tokenizer, TokenizerConfig

# New PLAN-based components
from .multi_source_loader import (
    MultiSourceDataLoader, 
    MultiSourceConfig, 
    DataSourceConfig,
    PackedSequence,
    create_plan_data_loader
)
from .deduplication import (
    ExactDeduplicator,
    MinHashDeduplicator, 
    RepoLevelDeduplicator,
    CombinedDeduplicator,
    create_plan_deduplicator
)
from .algorithmic_tasks import (
    AlgorithmicTaskGenerator,
    AlgorithmicExample,
    MazeGenerator,
    GraphTaskGenerator,
    ArithmeticGenerator,
    ParsingGenerator,
    create_plan_task_generator
)
from .phase_sampler import (
    PhaseBasedSampler,
    PhaseConfig,
    create_plan_phases,
    create_plan_sampler
)

__all__ = [
    # Legacy components
    "FineWebDataset",
    "create_data_loader",
    "create_tokenizer", 
    "TokenizerConfig",
    
    # Multi-source loading
    "MultiSourceDataLoader",
    "MultiSourceConfig",
    "DataSourceConfig", 
    "PackedSequence",
    "create_plan_data_loader",
    
    # Deduplication
    "ExactDeduplicator",
    "MinHashDeduplicator",
    "RepoLevelDeduplicator", 
    "CombinedDeduplicator",
    "create_plan_deduplicator",
    
    # Algorithmic tasks
    "AlgorithmicTaskGenerator",
    "AlgorithmicExample",
    "MazeGenerator",
    "GraphTaskGenerator", 
    "ArithmeticGenerator",
    "ParsingGenerator",
    "create_plan_task_generator",
    
    # Phase-based sampling
    "PhaseBasedSampler",
    "PhaseConfig",
    "create_plan_phases",
    "create_plan_sampler"
]