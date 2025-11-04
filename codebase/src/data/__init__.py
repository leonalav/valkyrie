"""Data Pipeline Module

Multi-source data loading and processing for BigBird+S5+HRM training.
Implements the PLAN data pipeline with proper deduplication, packing, and phase-wise sampling.
"""

# Legacy FineWeb components (available)
from .fineweb_reader import FineWebDataset, create_data_loader
from .tokenizer import create_tokenizer, TokenizerConfig

# New PLAN-based components (commented out until implemented)
# from .multi_source_loader import (
#     MultiSourceDataLoader, 
#     MultiSourceConfig, 
#     DataSourceConfig,
#     PackedSequence,
#     create_plan_data_loader
# )
# from .deduplication import (
#     ExactDeduplicator,
#     MinHashDeduplicator, 
#     RepoLevelDeduplicator,
#     CombinedDeduplicator,
#     create_plan_deduplicator
# )
# from .algorithmic_tasks import (
#     AlgorithmicTaskGenerator,
#     AlgorithmicExample,
#     MazeGenerator,
#     GraphTaskGenerator,
#     ArithmeticGenerator,
#     ParsingGenerator,
#     create_plan_task_generator
# )
# from .phase_sampler import (
#     PhaseBasedSampler,
#     PhaseConfig,
#     create_plan_phases,
#     create_plan_sampler
# )

__all__ = [
    # Available components
    "FineWebDataset",
    "create_data_loader",
    "create_tokenizer", 
    "TokenizerConfig",
    
    # Commented out until implemented
    # "MultiSourceDataLoader",
    # "MultiSourceConfig",
    # "DataSourceConfig", 
    # "PackedSequence",
    # "create_plan_data_loader",
    # "ExactDeduplicator",
    # "MinHashDeduplicator",
    # "RepoLevelDeduplicator", 
    # "CombinedDeduplicator",
    # "create_plan_deduplicator",
    # "AlgorithmicTaskGenerator",
    # "AlgorithmicExample",
    # "MazeGenerator",
    # "GraphTaskGenerator", 
    # "ArithmeticGenerator",
    # "ParsingGenerator",
    # "create_plan_task_generator",
    # "PhaseBasedSampler",
    # "PhaseConfig",
    # "create_plan_phases",
    # "create_plan_sampler"
]