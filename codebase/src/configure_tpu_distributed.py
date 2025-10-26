#!/usr/bin/env python3
"""
Distributed TPU Configuration Module

Provides utilities for configuring JAX for distributed TPU training,
including device mesh creation, sharding strategies, and memory management.
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from jax.sharding import Mesh, NamedSharding, PartitionSpec
import logging

logger = logging.getLogger(__name__)


def configure_jax_for_tpu() -> Tuple[List[jax.Device], List[jax.Device]]:
    """Configure JAX for TPU usage and return available devices.
    
    Returns:
        Tuple of (all_devices, local_devices)
    """
    # Get all available devices
    devices = jax.devices()
    local_devices = jax.local_devices()
    
    logger.info(f"Found {len(devices)} total devices, {len(local_devices)} local devices")
    logger.info(f"JAX backend: {jax.default_backend()}")
    
    return devices, local_devices


def create_device_mesh(devices: List[jax.Device]) -> Mesh:
    """Create a device mesh for distributed training.
    
    Args:
        devices: List of JAX devices
        
    Returns:
        JAX Mesh object for sharding
    """
    num_devices = len(devices)
    
    # Simple mesh configuration based on device count
    if num_devices == 1:
        mesh_shape = (1,)
        axis_names = ('batch',)
    elif num_devices == 2:
        mesh_shape = (2,)
        axis_names = ('batch',)
    elif num_devices == 4:
        mesh_shape = (2, 2)
        axis_names = ('batch', 'model')
    elif num_devices == 8:
        mesh_shape = (4, 2)
        axis_names = ('batch', 'model')
    elif num_devices == 16:
        mesh_shape = (4, 4)
        axis_names = ('batch', 'model')
    elif num_devices == 32:
        mesh_shape = (4, 4, 2)
        axis_names = ('batch', 'model', 'pipeline')
    else:
        # Default to single axis for other configurations
        mesh_shape = (num_devices,)
        axis_names = ('batch',)
    
    # Reshape devices array to match mesh shape
    devices_array = np.array(devices[:np.prod(mesh_shape)]).reshape(mesh_shape)
    
    mesh = Mesh(devices_array, axis_names)
    logger.info(f"Created mesh with shape {mesh_shape} and axes {axis_names}")
    
    return mesh


def create_sharding_strategy(
    mesh: Mesh, 
    batch_size: int, 
    seq_len: int, 
    hidden_size: int,
    vocab_size: int = 32000
) -> Tuple[NamedSharding, Dict[str, NamedSharding]]:
    """Create sharding strategies for data and parameters.
    
    Args:
        mesh: JAX Mesh for sharding
        batch_size: Batch size for training
        seq_len: Sequence length
        hidden_size: Hidden dimension size
        vocab_size: Vocabulary size
        
    Returns:
        Tuple of (data_sharding, param_shardings)
    """
    # Data sharding - shard batch dimension across 'batch' axis
    data_sharding = NamedSharding(mesh, PartitionSpec('batch', None))
    
    # Parameter sharding strategies
    param_shardings = {
        # Embedding layers - shard vocab dimension
        'embeddings': NamedSharding(mesh, PartitionSpec('model', None)),
        
        # Attention weights - shard hidden dimension
        'attention_weights': NamedSharding(mesh, PartitionSpec(None, 'model')),
        
        # Feed-forward weights - shard hidden dimension
        'ffn_weights': NamedSharding(mesh, PartitionSpec('model', None)),
        
        # Layer norm parameters - replicated
        'layer_norm': NamedSharding(mesh, PartitionSpec()),
        
        # Output projection - shard vocab dimension
        'output_projection': NamedSharding(mesh, PartitionSpec('model', None)),
        
        # Default sharding for other parameters
        'default': NamedSharding(mesh, PartitionSpec()),
    }
    
    logger.info(f"Created sharding strategies for batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}")
    
    return data_sharding, param_shardings


def shard_batch_to_devices(
    batch: Dict[str, jnp.ndarray], 
    data_sharding: NamedSharding
) -> Dict[str, jnp.ndarray]:
    """Shard a batch across devices using the specified sharding strategy.
    
    Args:
        batch: Dictionary containing batch data
        data_sharding: Sharding strategy for data
        
    Returns:
        Sharded batch dictionary
    """
    sharded_batch = {}
    for key, value in batch.items():
        sharded_batch[key] = jax.device_put(value, data_sharding)
    
    return sharded_batch


def estimate_memory_usage(
    batch_size: int,
    seq_len: int, 
    hidden_size: int,
    num_layers: int,
    vocab_size: int = 32000,
    num_devices: int = 1
) -> Tuple[bool, int]:
    """Estimate memory usage for model training and return safety check.
    
    Args:
        batch_size: Batch size
        seq_len: Sequence length
        hidden_size: Hidden dimension size
        num_layers: Number of transformer layers
        vocab_size: Vocabulary size
        num_devices: Number of devices for sharding
        
    Returns:
        Tuple of (memory_ok, recommended_batch_size)
    """
    # Activation memory (forward pass)
    activation_memory = batch_size * seq_len * hidden_size * num_layers * 4  # 4 bytes per float32
    
    # Parameter memory
    param_memory = (
        vocab_size * hidden_size +  # Embeddings
        num_layers * (
            4 * hidden_size * hidden_size +  # Attention weights (Q, K, V, O)
            2 * hidden_size * 4 * hidden_size  # FFN weights
        ) +
        num_layers * 2 * hidden_size  # Layer norms
    ) * 4  # 4 bytes per float32
    
    # Gradient memory (same as parameters)
    gradient_memory = param_memory
    
    # Optimizer state memory (Adam: 2x parameters)
    optimizer_memory = param_memory * 2
    
    total_memory = activation_memory + param_memory + gradient_memory + optimizer_memory
    total_memory_gb = total_memory / (1024**3)
    
    # Estimate available memory per device (conservative estimate)
    memory_per_device_gb = 16.0  # Conservative estimate for TPU v4
    total_available_memory_gb = memory_per_device_gb * num_devices
    
    # Check if memory usage is safe (use 80% of available memory)
    memory_ok = total_memory_gb < (total_available_memory_gb * 0.8)
    
    # Calculate recommended batch size if current is too large
    if not memory_ok:
        recommended_batch_size = max(1, int(batch_size * 0.8 * total_available_memory_gb / total_memory_gb))
    else:
        recommended_batch_size = batch_size
    
    return memory_ok, recommended_batch_size


def replicate_params_to_devices(
    params: Dict[str, Any], 
    param_shardings: Dict[str, NamedSharding]
) -> Dict[str, Any]:
    """Replicate parameters to devices using sharding strategies.
    
    Args:
        params: Model parameters
        param_shardings: Parameter sharding strategies
        
    Returns:
        Sharded parameters
    """
    sharded_params = {}
    
    for key, value in params.items():
        # Use specific sharding if available, otherwise use default
        sharding = param_shardings.get(key, param_shardings['default'])
        sharded_params[key] = jax.device_put(value, sharding)
    
    return sharded_params