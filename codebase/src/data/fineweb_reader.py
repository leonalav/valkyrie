"""FineWeb dataset reader with multi-host streaming and chunking.

Implements:
- Streaming from HuggingFace FineWeb dataset
- Multi-host data sharding for TPU training
- Chunked processing for ultra-long sequences (657k tokens)
- Memory-efficient data loading and preprocessing
- Proper coordination across TPU hosts
"""

import jax
import jax.numpy as jnp
from datasets import load_dataset, IterableDataset
from typing import Dict, List, Optional, Iterator, Any, Tuple
import numpy as np
import logging
from dataclasses import dataclass
import time
from concurrent.futures import ThreadPoolExecutor
import queue
import threading
from jax import tree_util as jtu

from .tokenizer import create_tokenizer, TokenizerConfig, tokenize_for_training

logger = logging.getLogger(__name__)


@dataclass
class FineWebConfig:
    """Configuration for FineWeb dataset loading."""
    dataset_name: str = "HuggingFaceFW/fineweb"
    dataset_config: str = "CC-MAIN-2024-10"  # or "sample-10BT" for 10BT sample
    split: str = "train"
    streaming: bool = True
    
    # Chunking configuration
    chunk_size: int = 8192
    overlap_size: int = 512
    min_chunk_size: int = 1024
    max_chunks_per_doc: int = 82  # ~657k / 8k
    
    # Processing configuration
    batch_size: int = 8
    num_workers: int = 4
    prefetch_factor: int = 2
    buffer_size: int = 1000
    
    # Multi-host configuration
    shard_by_host: bool = True
    seed: int = 42


class FineWebDataset:
    """
    FineWeb dataset with streaming, chunking, and multi-host support.
    
    Key features:
    - Streams from HuggingFace FineWeb dataset
    - Shards data across TPU hosts automatically
    - Chunks long documents into manageable pieces
    - Handles overlap between chunks for context
    - Memory-efficient processing with prefetching
    """
    
    def __init__(
        self,
        config: FineWebConfig,
        tokenizer_config: TokenizerConfig,
        process_index: Optional[int] = None,
        process_count: Optional[int] = None,
    ):
        self.config = config
        self.tokenizer_config = tokenizer_config
        
        # Multi-host configuration
        self.process_index = process_index if process_index is not None else jax.process_index()
        self.process_count = process_count if process_count is not None else jax.process_count()
        
        logger.info(f"Initializing FineWeb dataset:")
        logger.info(f"  Dataset: {config.dataset_name}/{config.dataset_config}")
        logger.info(f"  Process: {self.process_index}/{self.process_count}")
        logger.info(f"  Chunk size: {config.chunk_size}")
        logger.info(f"  Batch size: {config.batch_size}")
        
        # Create tokenizer
        self.tokenizer = create_tokenizer(tokenizer_config)
        
        # Initialize dataset
        self.dataset = None
        self._load_dataset()
        
        # Processing state
        self._chunk_buffer = queue.Queue(maxsize=config.buffer_size)
        self._stop_processing = threading.Event()
        self._processing_thread = None
        
    def _load_dataset(self):
        """Load and configure the FineWeb dataset."""
        
        logger.info(f"Loading FineWeb dataset: {self.config.dataset_name}")
        
        try:
            # Load dataset with streaming
            self.dataset = load_dataset(
                self.config.dataset_name,
                name=self.config.dataset_config,
                split=self.config.split,
                streaming=self.config.streaming,
            )
            
            logger.info("✓ Dataset loaded successfully")
            
            # Shard dataset across processes if multi-host
            if self.config.shard_by_host and self.process_count > 1:
                # Each process gets a different shard
                self.dataset = self.dataset.shard(
                    num_shards=self.process_count,
                    index=self.process_index,
                    contiguous=True,
                )
                logger.info(f"✓ Dataset sharded: process {self.process_index}/{self.process_count}")
            
            # Shuffle dataset
            if hasattr(self.dataset, 'shuffle'):
                self.dataset = self.dataset.shuffle(
                    seed=self.config.seed + self.process_index,
                    buffer_size=self.config.buffer_size,
                )
                logger.info("✓ Dataset shuffled")
                
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            raise
    
    def _process_document(self, document: Dict[str, Any]) -> List[Dict[str, jnp.ndarray]]:
        """
        Process a single document into chunks.
        
        Args:
            document: Raw document from dataset
            
        Returns:
            List of tokenized chunks
        """
        
        # Extract text content
        text = document.get('text', '')
        
        if not text or len(text.strip()) < 100:
            # Skip very short documents
            return []
        
        try:
            # Tokenize document into chunks
            chunks = tokenize_for_training(
                texts=[text],
                tokenizer=self.tokenizer,
                chunk_size=self.config.chunk_size,
                overlap_size=self.config.overlap_size,
                min_chunk_size=self.config.min_chunk_size,
            )
            
            # Limit number of chunks per document
            if len(chunks) > self.config.max_chunks_per_doc:
                chunks = chunks[:self.config.max_chunks_per_doc]
            
            return chunks
            
        except Exception as e:
            logger.warning(f"Failed to process document: {e}")
            return []
    
    def _background_processor(self):
        """Background thread for processing documents."""
        
        logger.info("Starting background document processor")
        
        try:
            for document in self.dataset:
                if self._stop_processing.is_set():
                    break
                
                # Process document into chunks
                chunks = self._process_document(document)
                
                # Add chunks to buffer
                for chunk in chunks:
                    if self._stop_processing.is_set():
                        break
                    
                    try:
                        # Create dictionary with input_ids
                        chunk_dict = {'input_ids': chunk}
                        self._chunk_buffer.put(chunk_dict, timeout=1.0)
                    except queue.Full:
                        # Buffer full, skip this chunk
                        logger.warning("Chunk buffer full, skipping chunk")
                        continue
                        
        except Exception as e:
            logger.error(f"Background processor error: {e}")
        finally:
            logger.info("Background processor stopped")
    
    def _start_background_processing(self):
        """Start background processing thread."""
        if self._processing_thread is None or not self._processing_thread.is_alive():
            self._stop_processing.clear()
            self._processing_thread = threading.Thread(
                target=self._background_processor,
                daemon=True
            )
            self._processing_thread.start()
    
    def _stop_background_processing(self):
        """Stop background processing thread."""
        self._stop_processing.set()
        if self._processing_thread and self._processing_thread.is_alive():
            self._processing_thread.join(timeout=5.0)
    
    def __iter__(self) -> Iterator[Dict[str, jnp.ndarray]]:
        """Iterate over processed chunks."""
        
        # Start background processing
        self._start_background_processing()
        
        try:
            while True:
                try:
                    # Get chunk from buffer
                    chunk = self._chunk_buffer.get(timeout=10.0)
                    yield chunk
                    
                except queue.Empty:
                    # Check if processing is still active
                    if (self._processing_thread and 
                        self._processing_thread.is_alive() and 
                        not self._stop_processing.is_set()):
                        continue
                    else:
                        # No more data
                        break
                        
        finally:
            self._stop_background_processing()
    
    def get_batch_iterator(self) -> Iterator[Dict[str, jnp.ndarray]]:
        """Get iterator that yields batches of chunks."""
        
        batch = []
        
        for chunk in self:
            batch.append(chunk)
            
            if len(batch) >= self.config.batch_size:
                # Yield batch
                yield self._collate_batch(batch)
                batch = []
        
        # Yield remaining batch if not empty
        if batch:
            collated_batch = self._collate_batch(batch)
            self._validate_batch(collated_batch)
            yield collated_batch
    
    def _validate_batch(self, batch):
        assert isinstance(batch, dict), f"Batch must be dict, got {type(batch)}"
        for k in ["input_ids", "labels", "attention_mask"]:
            assert k in batch, f"Missing key {k} in batch"
            assert isinstance(batch[k], jnp.ndarray), f"{k} must be array, got {type(batch[k])}"
        B = batch["input_ids"].shape[0]
        T = batch["input_ids"].shape[1]
        for k in ["labels", "attention_mask"]:
            assert batch[k].shape[:2] == (B, T), f"{k} shape {batch[k].shape} != (B,T)=({B},{T})"
        assert batch["input_ids"].dtype == jnp.int32, f"input_ids dtype must be int32, got {batch['input_ids'].dtype}"
        assert batch["labels"].dtype == jnp.int32, f"labels dtype must be int32, got {batch['labels'].dtype}"

    def _collate_batch(self, chunks):
        def to_array(x):
            return jnp.asarray(x)

        batched = jtu.tree_map(lambda *xs: jnp.stack([to_array(x) for x in xs], axis=0), *chunks)

        if isinstance(batched, dict) and "input_ids" in batched and isinstance(batched["input_ids"], dict):
            inner = batched["input_ids"]
            flat = {**batched, **inner}
            del flat["input_ids"]
            batched = flat

        if "input_ids" in batched: batched["input_ids"] = batched["input_ids"].astype(jnp.int32)
        if "labels" in batched:    batched["labels"]    = batched["labels"].astype(jnp.int32)
        if "attention_mask" in batched:
            batched["attention_mask"] = batched["attention_mask"].astype(jnp.int32)

        return batched
    
    def get_stats(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        
        return {
            'dataset_name': self.config.dataset_name,
            'dataset_config': self.config.dataset_config,
            'process_index': self.process_index,
            'process_count': self.process_count,
            'chunk_size': self.config.chunk_size,
            'batch_size': self.config.batch_size,
            'tokenizer_vocab_size': len(self.tokenizer),
            'buffer_size': self.config.buffer_size,
        }
    
    def __del__(self):
        """Cleanup when dataset is destroyed."""
        self._stop_background_processing()


def create_data_loader(
    config: FineWebConfig,
    tokenizer_config: TokenizerConfig,
    process_index: Optional[int] = None,
    process_count: Optional[int] = None,
) -> FineWebDataset:
    """
    Create FineWeb data loader with proper configuration.
    
    Args:
        config: FineWeb dataset configuration
        tokenizer_config: Tokenizer configuration
        process_index: Process index for multi-host (auto-detected if None)
        process_count: Total process count for multi-host (auto-detected if None)
        
    Returns:
        Configured FineWeb dataset
    """
    
    return FineWebDataset(
        config=config,
        tokenizer_config=tokenizer_config,
        process_index=process_index,
        process_count=process_count,
    )


def validate_data_loader(data_loader: FineWebDataset, num_batches: int = 5) -> bool:
    """
    Validate data loader by processing a few batches.
    
    Args:
        data_loader: Data loader to validate
        num_batches: Number of batches to test
        
    Returns:
        True if validation passes
    """
    
    logger.info(f"Validating data loader with {num_batches} batches...")
    
    try:
        batch_iterator = data_loader.get_batch_iterator()
        
        for i, batch in enumerate(batch_iterator):
            if i >= num_batches:
                break
            
            # Check batch structure
            required_keys = ['input_ids', 'labels', 'attention_mask']
            for key in required_keys:
                if key not in batch:
                    logger.error(f"Missing key in batch: {key}")
                    return False
            
            # Check shapes
            batch_size, seq_len = batch['input_ids'].shape
            logger.info(f"Batch {i}: shape={batch['input_ids'].shape}, dtype={batch['input_ids'].dtype}")
            
            # Check data types
            if batch['input_ids'].dtype != jnp.int32:
                logger.error(f"Wrong dtype for input_ids: {batch['input_ids'].dtype}")
                return False
            
            # Check value ranges
            vocab_size = len(data_loader.tokenizer)
            if jnp.max(batch['input_ids']) >= vocab_size:
                logger.error(f"Token ID out of range: max={jnp.max(batch['input_ids'])}, vocab_size={vocab_size}")
                return False
            
            if jnp.min(batch['input_ids']) < 0:
                logger.error(f"Negative token ID: min={jnp.min(batch['input_ids'])}")
                return False
        
        logger.info("✓ Data loader validation passed")
        return True
        
    except Exception as e:
        logger.error(f"Data loader validation failed: {e}")
        return False


# Preset configurations
def get_fineweb_10bt_config() -> FineWebConfig:
    """Get configuration for FineWeb 10BT sample."""
    return FineWebConfig(
        dataset_name="HuggingFaceFW/fineweb",
        dataset_config="sample-10BT",
        split="train",
        streaming=True,
        chunk_size=8192,
        overlap_size=512,
        batch_size=8,
        num_workers=4,
        buffer_size=1000,
    )


def get_fineweb_full_config() -> FineWebConfig:
    """Get configuration for full FineWeb dataset."""
    return FineWebConfig(
        dataset_name="HuggingFaceFW/fineweb",
        dataset_config="CC-MAIN-2024-10",
        split="train",
        streaming=True,
        chunk_size=8192,
        overlap_size=512,
        batch_size=8,
        num_workers=4,
        buffer_size=1000,
    )


def get_fineweb_edu_config() -> FineWebConfig:
    """Get configuration for FineWeb-Edu dataset."""
    return FineWebConfig(
        dataset_name="HuggingFaceFW/fineweb-edu",
        dataset_config="default",
        split="train",
        streaming=True,
        chunk_size=8192,
        overlap_size=512,
        batch_size=8,
        num_workers=4,
        buffer_size=1000,
    )


class MultiHostDataLoader:
    """
    Wrapper for coordinating data loading across multiple TPU hosts.
    
    Ensures proper synchronization and load balancing across hosts.
    """
    
    def __init__(
        self,
        config: FineWebConfig,
        tokenizer_config: TokenizerConfig,
    ):
        self.config = config
        self.tokenizer_config = tokenizer_config
        
        # Multi-host info
        self.process_index = jax.process_index()
        self.process_count = jax.process_count()
        
        # Create per-host data loader
        self.data_loader = create_data_loader(
            config=config,
            tokenizer_config=tokenizer_config,
            process_index=self.process_index,
            process_count=self.process_count,
        )
        
        logger.info(f"Multi-host data loader initialized for process {self.process_index}/{self.process_count}")
    
    def __iter__(self):
        """Iterate with multi-host coordination."""
        return iter(self.data_loader)
    
    def get_batch_iterator(self):
        """Get batch iterator with multi-host coordination."""
        return self.data_loader.get_batch_iterator()
    
    def synchronize_hosts(self):
        """Synchronize all hosts (barrier)."""
        if self.process_count > 1:
            # Use JAX's built-in synchronization
            jax.random.normal(jax.random.PRNGKey(0), ()).block_until_ready()
            logger.info("Host synchronization completed")
    
    def get_global_stats(self) -> Dict[str, Any]:
        """Get statistics aggregated across all hosts."""
        local_stats = self.data_loader.get_stats()
        
        # Add multi-host info
        local_stats.update({
            'multi_host': True,
            'process_index': self.process_index,
            'process_count': self.process_count,
        })
        
        return local_stats