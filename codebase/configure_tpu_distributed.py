#!/usr/bin/env python3
"""
Comprehensive TPU multi-device configuration for distributed training.
Based on JAX best practices for TPU v4-8 with 4 devices.
"""

import jax
import jax.numpy as jnp
from jax import sharding
from jax.sharding import Mesh, PartitionSpec as P
import os
from typing import Any, Dict, Tuple
import functools


def configure_jax_for_tpu():
    """Configure JAX for optimal TPU usage with distributed training.
    
    Returns:
        devices: List of TPU devices available to JAX
    """
    # Set JAX configuration for TPU
    jax.config.update('jax_platform_name', 'tpu')

    # Enable memory optimization
    jax.config.update('jax_enable_x64', False)  # Use bfloat16/float32 for TPU
    jax.config.update('jax_default_matmul_precision', 'tensorfloat32')

    # Initialize distributed if not already done
    try:
        if jax.process_count() == 1:
            print("Initializing JAX distributed mode...")
            jax.distributed.initialize()
    except Exception as e:
        print(f"Distributed initialization not needed or failed: {e}")

    # Get device information
    devices = jax.devices()
    local_devices = jax.local_devices()

    print(f"JAX backend: {jax.default_backend()}")
    print(f"Total devices: {len(devices)}")
    print(f"Local devices: {len(local_devices)}")
    print(f"Process index: {jax.process_index()}")
    print(f"Process count: {jax.process_count()}")

    # Return only devices to match test expectations
    return devices


def create_device_mesh(devices):
    """Create a device mesh for sharding across TPU devices."""

    num_devices = len(devices)
    print(f"Creating mesh with {num_devices} devices")

    # For TPU v4-8, we have 4 devices arranged in a 2x2 topology
    if num_devices == 4:
        # Create a 1D mesh for data parallelism - use devices directly
        mesh = Mesh(devices, axis_names=('data',))
        print(f"Created 1D mesh: {mesh}")
        return mesh
    elif num_devices == 8:
        # For 8 devices, use 2D mesh - reshape devices list
        devices_2d = [devices[i:i+4] for i in range(0, 8, 4)]
        mesh = Mesh(devices_2d, axis_names=('data', 'model'))
        print(f"Created 2D mesh: {mesh}")
        return mesh
    else:
        # Fallback to 1D mesh
        mesh = Mesh(devices, axis_names=('data',))
        print(f"Created fallback 1D mesh: {mesh}")
        return mesh


def create_sharding_strategy(mesh, batch_size: int, seq_len: int, hidden_size: int):
    """Create sharding strategy for model parameters and data."""

    # Data sharding - shard batch dimension across devices
    data_sharding = sharding.NamedSharding(mesh, P('data', None, None))

    # Parameter sharding strategies
    param_shardings = {
        # Embedding layers - shard vocab dimension
        'embed': sharding.NamedSharding(mesh, P(None, None)),

        # Linear layers - shard output dimension for data parallelism
        'linear': sharding.NamedSharding(mesh, P(None, None)),

        # Attention weights - replicate across devices
        'attention': sharding.NamedSharding(mesh, P(None, None)),

        # Layer norm - replicate
        'layer_norm': sharding.NamedSharding(mesh, P(None)),

        # Output head - shard vocab dimension
        'output': sharding.NamedSharding(mesh, P(None, None))
    }

    print("Created sharding strategies:")
    print(f"  Data sharding: {data_sharding}")
    for name, shard in param_shardings.items():
        print(f"  {name}: {shard}")

    return data_sharding, param_shardings


def create_sharding_strategies(mesh):
    """Compat wrapper expected by tests.
    
    Returns a dict containing at least the 'data' sharding strategy used to shard batches.
    """
    data_sharding = sharding.NamedSharding(mesh, P('data', None, None))
    return {'data': data_sharding}


def estimate_memory_usage(batch_size: int, seq_len: int, hidden_size: int, 
                         vocab_size: int, num_layers: int, num_devices: int):
    """Estimate memory usage for the model across devices."""

    # Model parameters (in elements)
    embed_params = vocab_size * hidden_size
    layer_params = num_layers * (
        4 * hidden_size * hidden_size +  # Attention weights
        2 * hidden_size +                # Layer norm
        8 * hidden_size * hidden_size    # MLP weights (assuming 4x expansion)
    )
    output_params = vocab_size * hidden_size
    total_params = embed_params + layer_params + output_params

    # Activations (in elements, per sample)
    activations_per_sample = seq_len * hidden_size * num_layers * 4  # Rough estimate
    total_activations = batch_size * activations_per_sample

    # Memory in bytes (assuming bfloat16 = 2 bytes)
    param_memory_gb = (total_params * 2) / (1024**3)
    activation_memory_gb = (total_activations * 2) / (1024**3)

    # Add gradients (same size as parameters)
    gradient_memory_gb = param_memory_gb

    # Add optimizer states (Adam: 2x parameters)
    optimizer_memory_gb = param_memory_gb * 2

    total_memory_gb = param_memory_gb + activation_memory_gb + gradient_memory_gb + optimizer_memory_gb
    memory_per_device_gb = total_memory_gb / num_devices

    print(f"\n=== Memory Estimation ===")
    print(f"Total parameters: {total_params:,} ({param_memory_gb:.2f} GB)")
    print(f"Activations: {activation_memory_gb:.2f} GB")
    print(f"Gradients: {gradient_memory_gb:.2f} GB")
    print(f"Optimizer states: {optimizer_memory_gb:.2f} GB")
    print(f"Total memory: {total_memory_gb:.2f} GB")
    print(f"Memory per device: {memory_per_device_gb:.2f} GB")
    print(f"Available per device: 30.75 GB")

    if memory_per_device_gb > 30.75:
        print("⚠️  WARNING: Estimated memory exceeds available memory per device!")
        recommended_batch_size = int(batch_size * 30.75 / memory_per_device_gb)
        print(f"Recommended batch size: {recommended_batch_size}")
        return False, recommended_batch_size
    else:
        print("✅ Memory usage looks good!")
        return True, batch_size


def shard_batch_to_devices(batch, data_sharding):
    """Shard a batch across devices."""
    return jax.device_put(batch, data_sharding)


def shard_batch_data(batch, data_sharding):
    """Compat alias for tests expecting shard_batch_data name."""
    return shard_batch_to_devices(batch, data_sharding)


def replicate_params_to_devices(params, mesh):
    """Replicate parameters across all devices."""
    # For data parallelism, we replicate all parameters
    param_sharding = sharding.NamedSharding(mesh, P())
    return jax.device_put(params, param_sharding)


@functools.partial(jax.jit, static_argnames=['num_devices'])
def all_reduce_gradients(gradients, num_devices):
    """All-reduce gradients across devices for data parallelism."""
    return jax.lax.pmean(gradients, axis_name='data')


def setup_distributed_training(batch_size: int = 32, seq_len: int = 512, 
                              hidden_size: int = 768, vocab_size: int = 32000, 
                              num_layers: int = 12):
    """Set up distributed training configuration."""

    print("=== Setting up Distributed TPU Training ===")

    # Configure JAX
    devices = configure_jax_for_tpu()
    local_devices = jax.local_devices()

    # Create device mesh
    mesh = create_device_mesh(devices)

    # Estimate memory usage
    memory_ok, recommended_batch_size = estimate_memory_usage(
        batch_size, seq_len, hidden_size, vocab_size, num_layers, len(devices)
    )

    if not memory_ok:
        print(f"Adjusting batch size from {batch_size} to {recommended_batch_size}")
        batch_size = recommended_batch_size

    # Create sharding strategy
    data_sharding, param_shardings = create_sharding_strategy(
        mesh, batch_size, seq_len, hidden_size
    )

    config = {
        'mesh': mesh,
        'devices': devices,
        'local_devices': local_devices,
        'data_sharding': data_sharding,
        'param_shardings': param_shardings,
        'batch_size': batch_size,
        'num_devices': len(devices)
    }

    print(f"\n✅ Distributed training setup complete!")
    print(f"Using {len(devices)} TPU devices with {batch_size} batch size")

    return config


if __name__ == "__main__":
    # Test the configuration
    config = setup_distributed_training(
        batch_size=64,  # Start with larger batch size
        seq_len=512,
        hidden_size=768,
        vocab_size=32000,
        num_layers=12
    )

    # Test data sharding
    print("\n=== Testing Data Sharding ===")
    test_batch = jnp.ones((config['batch_size'], 512, 768))
    sharded_batch = shard_batch_to_devices(test_batch, config['data_sharding'])
    print(f"Original batch shape: {test_batch.shape}")
    print(f"Sharded batch shape per device: {sharded_batch.shape}")
    print(f"Sharded batch sharding: {sharded_batch.sharding}")