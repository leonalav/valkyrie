"""
HRM Data Pipeline for segment-wise training.

Handles loading and preprocessing of puzzle datasets with support for:
- Multiple datasets (ARC, Maze, Sudoku)
- Segment-wise batching for HRM training
- Efficient data loading with JAX/NumPy
- Puzzle grouping and sequence management
"""

import jax
import jax.numpy as jnp
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Iterator, NamedTuple, Union
import functools
from dataclasses import dataclass, field


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
    sets: Optional[List[str]] = None  # Optional field for dataset sets


class DataBatch(NamedTuple):
    """A batch of training data."""
    
    inputs: jnp.ndarray      # [batch_size, seq_len]
    targets: jnp.ndarray     # [batch_size, seq_len]
    puzzle_ids: jnp.ndarray  # [batch_size]
    group_ids: jnp.ndarray   # [batch_size]
    mask: jnp.ndarray        # [batch_size, seq_len] - valid token mask


class SegmentBatch(NamedTuple):
    """A segment batch for segment-wise training."""
    
    batches: List[DataBatch]
    segment_id: int
    total_segments: int


@dataclass
class DataConfig:
    """Configuration for data loading and processing."""
    
    # Data paths
    data_root: str = "/home/ravkeave/v1/data"
    datasets: List[str] = field(default_factory=lambda: ["arc-aug-1000", "maze-30x30-hard-1k", "sudoku-extreme-full"]) 
    
    # Batching
    batch_size: int = 32
    segment_size: int = 8  # Number of batches per segment
    shuffle: bool = True
    
    # Processing
    max_seq_len: Optional[int] = None  # Truncate sequences if needed
    pad_to_multiple: Optional[int] = None  # Pad sequences to multiple of this
    
    # Filtering
    min_seq_len: int = 10  # Filter out very short sequences
    max_examples_per_puzzle: Optional[int] = None  # Limit examples per puzzle
    
    def __post_init__(self):
        if self.datasets is None:
            self.datasets = ["arc-aug-1000", "maze-30x30-hard-1k", "sudoku-extreme-full"]


class HRMDataLoader:
    """
    Data loader for HRM training with segment-wise batching.
    
    Supports multiple datasets and efficient loading for the segment-wise
    training approach described in the HRM paper.
    """
    
    def __init__(self, config: DataConfig, split: str = "train"):
        self.config = config
        self.split = split
        self.data_root = Path(config.data_root)
        
        # Load dataset information
        self.dataset_infos = {}
        self.datasets = {}
        
        for dataset_name in config.datasets:
            dataset_path = self.data_root / dataset_name
            if dataset_path.exists():
                self._load_dataset(dataset_name)
            else:
                print(f"Warning: Dataset {dataset_name} not found at {dataset_path}")
        
        if not self.datasets:
            raise ValueError("No valid datasets found!")
        
        # Combine datasets
        self._combine_datasets()
        
        # Create batching indices
        self._create_batching_indices()
    
    def _load_dataset(self, dataset_name: str):
        """Load a single dataset."""
        dataset_path = self.data_root / dataset_name
        split_path = dataset_path / self.split
        
        # Load dataset info
        with open(dataset_path / "identifiers.json", "r") as f:
            identifiers = json.load(f)
        
        with open(split_path / "dataset.json", "r") as f:
            info_dict = json.load(f)
            info = DatasetInfo(**info_dict)
        
        # Load data arrays
        inputs = np.load(split_path / "all__inputs.npy")
        labels = np.load(split_path / "all__labels.npy")
        group_indices = np.load(split_path / "all__group_indices.npy")
        puzzle_indices = np.load(split_path / "all__puzzle_indices.npy")
        puzzle_identifiers = np.load(split_path / "all__puzzle_identifiers.npy")
        
        self.dataset_infos[dataset_name] = info
        self.datasets[dataset_name] = {
            "inputs": inputs,
            "labels": labels,
            "group_indices": group_indices,
            "puzzle_indices": puzzle_indices,
            "puzzle_identifiers": puzzle_identifiers,
            "identifiers": identifiers
        }
        
        print(f"Loaded {dataset_name}: {inputs.shape[0]} examples, "
              f"vocab_size={info.vocab_size}, seq_len={info.seq_len}")
    
    def _combine_datasets(self):
        """Combine multiple datasets into unified arrays."""
        all_inputs = []
        all_labels = []
        all_puzzle_ids = []
        all_group_ids = []
        all_dataset_ids = []
        
        dataset_id = 0
        for dataset_name, data in self.datasets.items():
            inputs = data["inputs"]
            labels = data["labels"]
            puzzle_ids = data["puzzle_identifiers"]
            
            # Create group IDs (simplified - use puzzle IDs as group IDs)
            group_ids = puzzle_ids
            
            # Filter sequences if needed
            if self.config.min_seq_len > 0:
                # Count non-padding tokens
                valid_lens = np.sum(inputs != 0, axis=1)
                valid_mask = valid_lens >= self.config.min_seq_len
                
                inputs = inputs[valid_mask]
                labels = labels[valid_mask]
                puzzle_ids = puzzle_ids[valid_mask] if len(puzzle_ids) == len(inputs) else puzzle_ids
                group_ids = group_ids[valid_mask] if len(group_ids) == len(inputs) else group_ids
            
            # Truncate sequences if needed
            if self.config.max_seq_len is not None:
                seq_len = min(self.config.max_seq_len, inputs.shape[1])
                inputs = inputs[:, :seq_len]
                labels = labels[:, :seq_len]
            
            all_inputs.append(inputs)
            all_labels.append(labels)
            
            # Handle puzzle IDs (may be shorter than inputs)
            if len(puzzle_ids) < len(inputs):
                # Repeat puzzle IDs to match input length
                puzzle_ids = np.repeat(puzzle_ids, len(inputs) // len(puzzle_ids) + 1)[:len(inputs)]
            
            all_puzzle_ids.append(puzzle_ids[:len(inputs)])
            all_group_ids.append(group_ids[:len(inputs)] if len(group_ids) == len(inputs) else puzzle_ids[:len(inputs)])
            all_dataset_ids.append(np.full(len(inputs), dataset_id, dtype=np.int32))
            
            dataset_id += 1
        
        # Concatenate all datasets
        self.inputs = np.concatenate(all_inputs, axis=0)
        self.labels = np.concatenate(all_labels, axis=0)
        self.puzzle_ids = np.concatenate(all_puzzle_ids, axis=0)
        self.group_ids = np.concatenate(all_group_ids, axis=0)
        self.dataset_ids = np.concatenate(all_dataset_ids, axis=0)
        
        # Create masks for valid tokens (non-padding)
        self.masks = (self.inputs != 0).astype(np.float32)
        
        print(f"Combined dataset: {len(self.inputs)} examples, "
              f"seq_len={self.inputs.shape[1]}")
    
    def _create_batching_indices(self):
        """Create indices for batching."""
        num_examples = len(self.inputs)
        self.indices = np.arange(num_examples)
        
        if self.config.shuffle:
            np.random.shuffle(self.indices)
        
        # Calculate number of complete batches
        self.num_batches = num_examples // self.config.batch_size
        self.num_segments = max(1, self.num_batches // self.config.segment_size)
        
        print(f"Created {self.num_batches} batches, {self.num_segments} segments")
    
    def get_batch(self, batch_idx: int) -> DataBatch:
        """Get a single batch by index."""
        start_idx = batch_idx * self.config.batch_size
        end_idx = start_idx + self.config.batch_size
        
        batch_indices = self.indices[start_idx:end_idx]
        
        # Extract batch data
        inputs = self.inputs[batch_indices]
        targets = self.labels[batch_indices]
        puzzle_ids = self.puzzle_ids[batch_indices]
        group_ids = self.group_ids[batch_indices]
        masks = self.masks[batch_indices]
        
        # Convert to JAX arrays
        return DataBatch(
            inputs=jnp.array(inputs),
            targets=jnp.array(targets),
            puzzle_ids=jnp.array(puzzle_ids),
            group_ids=jnp.array(group_ids),
            mask=jnp.array(masks)
        )
    
    def get_segment(self, segment_idx: int) -> SegmentBatch:
        """Get a segment (multiple batches) for segment-wise training."""
        start_batch = segment_idx * self.config.segment_size
        end_batch = min(start_batch + self.config.segment_size, self.num_batches)
        
        batches = []
        for batch_idx in range(start_batch, end_batch):
            batches.append(self.get_batch(batch_idx))
        
        return SegmentBatch(
            batches=batches,
            segment_id=segment_idx,
            total_segments=self.num_segments
        )
    
    def segment_iterator(self) -> Iterator[SegmentBatch]:
        """Iterate over segments for training."""
        for segment_idx in range(self.num_segments):
            yield self.get_segment(segment_idx)
    
    def batch_iterator(self) -> Iterator[DataBatch]:
        """Iterate over individual batches."""
        for batch_idx in range(self.num_batches):
            yield self.get_batch(batch_idx)
    
    def shuffle_data(self):
        """Shuffle the data indices."""
        if self.config.shuffle:
            np.random.shuffle(self.indices)
    
    def get_dataset_stats(self) -> Dict[str, float]:
        """Get statistics about the dataset."""
        stats = {
            "num_examples": len(self.inputs),
            "num_batches": self.num_batches,
            "num_segments": self.num_segments,
            "mean_seq_len": float(np.mean(np.sum(self.masks, axis=1))),
            "vocab_size": int(np.max(self.inputs)) + 1,
            "num_puzzles": len(np.unique(self.puzzle_ids)),
            "num_groups": len(np.unique(self.group_ids))
        }
        
        return stats


# Utility functions for data processing

def create_data_loaders(
    config: DataConfig,
    splits: List[str] = ["train", "test"]
) -> Dict[str, HRMDataLoader]:
    """Create data loaders for multiple splits."""
    loaders = {}
    
    for split in splits:
        try:
            loader = HRMDataLoader(config, split=split)
            loaders[split] = loader
        except Exception as e:
            print(f"Warning: Could not create {split} loader: {e}")
    
    return loaders


def collate_batches(batches: List[DataBatch]) -> DataBatch:
    """Collate multiple batches into a single batch."""
    inputs = jnp.concatenate([b.inputs for b in batches], axis=0)
    targets = jnp.concatenate([b.targets for b in batches], axis=0)
    puzzle_ids = jnp.concatenate([b.puzzle_ids for b in batches], axis=0)
    group_ids = jnp.concatenate([b.group_ids for b in batches], axis=0)
    masks = jnp.concatenate([b.mask for b in batches], axis=0)
    
    return DataBatch(
        inputs=inputs,
        targets=targets,
        puzzle_ids=puzzle_ids,
        group_ids=group_ids,
        mask=masks
    )


def pad_sequences(
    sequences: jnp.ndarray,
    max_len: Optional[int] = None,
    pad_value: int = 0
) -> jnp.ndarray:
    """Pad sequences to a fixed length."""
    if max_len is None:
        max_len = sequences.shape[1]
    
    if sequences.shape[1] >= max_len:
        return sequences[:, :max_len]
    
    pad_width = ((0, 0), (0, max_len - sequences.shape[1]))
    return jnp.pad(sequences, pad_width, constant_values=pad_value)


def create_segment_schedule(
    num_segments: int,
    warmup_segments: int = 0,
    schedule_type: str = "linear"
) -> List[float]:
    """Create a learning rate schedule for segment-wise training."""
    if schedule_type == "linear":
        # Linear warmup then constant
        schedule = []
        for i in range(num_segments):
            if i < warmup_segments:
                lr_mult = (i + 1) / warmup_segments
            else:
                lr_mult = 1.0
            schedule.append(lr_mult)
        return schedule
    
    elif schedule_type == "cosine":
        # Cosine annealing
        import math
        schedule = []
        for i in range(num_segments):
            if i < warmup_segments:
                lr_mult = (i + 1) / warmup_segments
            else:
                progress = (i - warmup_segments) / (num_segments - warmup_segments)
                lr_mult = 0.5 * (1 + math.cos(math.pi * progress))
            schedule.append(lr_mult)
        return schedule
    
    else:
        return [1.0] * num_segments


# Example usage and testing functions

def test_data_loader():
    """Test the data loader functionality."""
    config = DataConfig(
        batch_size=4,
        segment_size=2,
        datasets=["arc-aug-1000"]  # Test with one dataset
    )
    
    try:
        loader = HRMDataLoader(config, split="train")
        
        print("Dataset stats:", loader.get_dataset_stats())
        
        # Test batch loading
        batch = loader.get_batch(0)
        print(f"Batch shapes: inputs={batch.inputs.shape}, targets={batch.targets.shape}")
        print(f"Sample input: {batch.inputs[0][:10]}")
        print(f"Sample target: {batch.targets[0][:10]}")
        
        # Test segment loading
        segment = loader.get_segment(0)
        print(f"Segment has {len(segment.batches)} batches")
        
        # Test iteration
        print("Testing segment iteration...")
        for i, segment in enumerate(loader.segment_iterator()):
            print(f"Segment {i}: {len(segment.batches)} batches")
            if i >= 2:  # Only test first few segments
                break
        
        print("Data loader test passed!")
        
    except Exception as e:
        print(f"Data loader test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_data_loader()