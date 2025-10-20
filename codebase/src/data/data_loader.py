"""
Data loading utilities for HRM training.
Loads real data from .npy files in the data directory.
"""

import json
import numpy as np
import jax.numpy as jnp
from pathlib import Path
from typing import Dict, Tuple, Optional, NamedTuple
import jax


class DatasetInfo(NamedTuple):
    """Information about a dataset."""
    pad_id: int
    ignore_label_id: int
    blank_identifier_id: int
    vocab_size: int
    seq_len: int
    num_puzzle_identifiers: int
    total_groups: int
    mean_puzzle_examples: float


class DataBatch(NamedTuple):
    """A batch of training data."""
    inputs: jnp.ndarray  # [batch_size, seq_len]
    labels: jnp.ndarray  # [batch_size, seq_len]
    group_indices: jnp.ndarray  # [batch_size]
    puzzle_indices: jnp.ndarray  # [batch_size]
    puzzle_identifiers: jnp.ndarray  # [batch_size]


def load_dataset_info(data_dir: str, split: str = "train") -> DatasetInfo:
    """Load dataset information from JSON file."""
    dataset_path = Path(data_dir) / split / "dataset.json"
    
    with open(dataset_path, 'r') as f:
        info = json.load(f)
    
    return DatasetInfo(
        pad_id=info["pad_id"],
        ignore_label_id=info["ignore_label_id"],
        blank_identifier_id=info["blank_identifier_id"],
        vocab_size=info["vocab_size"],
        seq_len=info["seq_len"],
        num_puzzle_identifiers=info["num_puzzle_identifiers"],
        total_groups=info["total_groups"],
        mean_puzzle_examples=info["mean_puzzle_examples"]
    )


def load_dataset_arrays(data_dir: str, split: str = "train") -> Dict[str, np.ndarray]:
    """Load all numpy arrays for a dataset split using memory mapping to avoid RAM exhaustion."""
    split_path = Path(data_dir) / split
    
    arrays = {}
    for npy_file in split_path.iterdir():
        if npy_file.suffix == '.npy':
            array_name = npy_file.stem.replace("all__", "")
            # Use memory mapping to avoid loading entire files into RAM
            arrays[array_name] = np.load(npy_file, mmap_mode='r')
    
    return arrays


def create_data_batch(
    arrays: Dict[str, np.ndarray], 
    batch_indices: np.ndarray,
    target_seq_len: Optional[int] = None
) -> DataBatch:
    """Create a data batch from arrays and indices with optional sequence length optimization."""
    # For arrays that have the same length as inputs, use batch_indices directly
    # For smaller arrays, we need to handle them differently
    inputs_array = arrays["inputs"]
    labels_array = arrays["labels"]
    
    # Slice the memory-mapped arrays (cheap operation, no full copy)
    inputs_batch = inputs_array[batch_indices]
    labels_batch = labels_array[batch_indices]
    
    # Optimize sequence length if target_seq_len is provided
    if target_seq_len is not None and inputs_batch.shape[1] > target_seq_len:
        # Truncate to target length before converting to JAX arrays
        inputs_batch = inputs_batch[:, :target_seq_len]
        labels_batch = labels_batch[:, :target_seq_len]
    
    # Check if other arrays have the same length as inputs
    if "group_indices" in arrays and len(arrays["group_indices"]) == len(inputs_array):
        group_indices = jnp.array(arrays["group_indices"][batch_indices])
    else:
        # If group_indices is smaller, create dummy values or sample appropriately
        group_indices = jnp.zeros(len(batch_indices), dtype=jnp.int32)
    
    if "puzzle_indices" in arrays and len(arrays["puzzle_indices"]) == len(inputs_array):
        puzzle_indices = jnp.array(arrays["puzzle_indices"][batch_indices])
    else:
        puzzle_indices = jnp.zeros(len(batch_indices), dtype=jnp.int32)
        
    if "puzzle_identifiers" in arrays and len(arrays["puzzle_identifiers"]) == len(inputs_array):
        puzzle_identifiers = jnp.array(arrays["puzzle_identifiers"][batch_indices])
    else:
        puzzle_identifiers = jnp.zeros(len(batch_indices), dtype=jnp.int32)
    
    return DataBatch(
        inputs=jnp.array(inputs_batch),
        labels=jnp.array(labels_batch),
        group_indices=group_indices,
        puzzle_indices=puzzle_indices,
        puzzle_identifiers=puzzle_identifiers
    )


def get_random_batch(
    data_dir: str, 
    batch_size: int, 
    split: str = "train",
    rng_key: Optional[jax.Array] = None,
    target_seq_len: Optional[int] = None
) -> Tuple[DataBatch, DatasetInfo]:
    """Get a random batch from the dataset with memory-efficient loading."""
    if rng_key is None:
        rng_key = jax.random.key(42)
    
    # Load dataset info and arrays (now memory-mapped)
    dataset_info = load_dataset_info(data_dir, split)
    arrays = load_dataset_arrays(data_dir, split)
    
    # Get total number of examples
    total_examples = arrays["inputs"].shape[0]
    
    # Sample random indices using numpy for proper indexing
    if rng_key is not None:
        # Convert JAX key to numpy seed
        seed = int(jax.random.bits(rng_key, shape=(), dtype=jnp.uint32))
        np.random.seed(seed)
    
    batch_indices = np.random.choice(
        total_examples, 
        size=batch_size, 
        replace=False
    )
    
    # Create batch with optional sequence length optimization
    batch = create_data_batch(arrays, batch_indices, target_seq_len)
    
    return batch, dataset_info


def get_sequential_batches(
    data_dir: str,
    batch_size: int,
    num_batches: int,
    split: str = "train",
    start_idx: int = 0
) -> Tuple[list[DataBatch], DatasetInfo]:
    """Get sequential batches from the dataset."""
    # Load dataset info and arrays
    dataset_info = load_dataset_info(data_dir, split)
    arrays = load_dataset_arrays(data_dir, split)
    
    total_examples = arrays["inputs"].shape[0]
    batches = []
    
    for i in range(num_batches):
        batch_start = (start_idx + i * batch_size) % total_examples
        batch_end = min(batch_start + batch_size, total_examples)
        
        # Handle wrap-around if needed
        if batch_end - batch_start < batch_size:
            indices1 = np.arange(batch_start, total_examples)
            indices2 = np.arange(0, batch_size - len(indices1))
            batch_indices = np.concatenate([indices1, indices2])
        else:
            batch_indices = np.arange(batch_start, batch_end)
        
        batch = create_data_batch(arrays, batch_indices)
        batches.append(batch)
    
    return batches, dataset_info


# Available datasets
AVAILABLE_DATASETS = [
    "arc-aug-1000",
    "maze-30x30-hard-1k", 
    "sudoku-extreme-full"
]


def list_available_datasets(data_root: str = "/home/ravkeave/v1/data") -> list[str]:
    """List all available datasets in the data directory."""
    data_path = Path(data_root)
    datasets = []
    
    for dataset_dir in data_path.iterdir():
        if dataset_dir.is_dir() and (dataset_dir / "train" / "dataset.json").exists():
            datasets.append(dataset_dir.name)
    
    return datasets