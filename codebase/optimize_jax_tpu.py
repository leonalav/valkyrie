#!/usr/bin/env python3
"""
JAX TPU Optimization Configuration

This script configures JAX for optimal TPU usage, including:
- Memory allocation settings
- Device placement
- TPU-specific optimizations
- Memory management
"""

import jax
import jax.numpy as jnp
import os


def configure_jax_for_tpu():
    """Configure JAX for optimal TPU performance."""
    
    print("Configuring JAX for TPU optimization...")
    
    # Set JAX configuration for TPU
    # Enable memory preallocation to avoid fragmentation
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    
    # Configure TPU memory allocation
    os.environ['TPU_ML_PLATFORM'] = 'PyTorch/XLA'
    
    # Enable memory profiling
    os.environ['XLA_FLAGS'] = '--xla_dump_to=/tmp/xla_dump'
    
    # Configure JAX for better memory management
    jax.config.update('jax_enable_x64', False)  # Use float32 for memory efficiency
    jax.config.update('jax_platform_name', 'tpu')
    
    print("JAX TPU configuration completed.")
    
    # Print device information
    devices = jax.devices()
    print(f"Available devices: {len(devices)}")
    for i, device in enumerate(devices):
        print(f"  Device {i}: {device}")
    
    # Test memory allocation
    print("\nTesting TPU memory allocation...")
    try:
        # Test small allocation
        test_array = jnp.ones((1000, 1000), dtype=jnp.float32)
        print(f"✓ Small allocation successful: {test_array.shape}")
        
        # Test medium allocation
        test_array = jnp.ones((5000, 5000), dtype=jnp.float32)
        print(f"✓ Medium allocation successful: {test_array.shape}")
        
        # Test larger allocation with explicit device placement
        device = jax.devices()[0]
        test_array = jax.device_put(jnp.ones((10000, 1000), dtype=jnp.float32), device)
        print(f"✓ Large allocation with device placement successful: {test_array.shape}")
        
    except Exception as e:
        print(f"✗ Memory allocation test failed: {e}")
    
    return devices


def optimize_batch_size_for_tpu(model_params_count: int, seq_len: int = 512) -> int:
    """
    Calculate optimal batch size for TPU based on model size and available memory.
    
    Args:
        model_params_count: Number of model parameters
        seq_len: Sequence length
        
    Returns:
        Recommended batch size
    """
    # Estimate memory usage per sample (rough approximation)
    # Each parameter: 4 bytes (float32)
    # Activations: roughly 2x parameters for forward + backward
    # Add sequence length factor
    
    memory_per_param = 4  # bytes
    activation_multiplier = 2
    seq_len_factor = seq_len / 512  # normalize to 512 tokens
    
    memory_per_sample = model_params_count * memory_per_param * activation_multiplier * seq_len_factor
    
    # TPU memory per device (33GB from our config check)
    tpu_memory_bytes = 33 * 1024 * 1024 * 1024  # 33GB
    
    # Use 80% of available memory for safety
    usable_memory = tpu_memory_bytes * 0.8
    
    # Calculate batch size
    recommended_batch_size = int(usable_memory / memory_per_sample)
    
    # Ensure minimum batch size of 1 and maximum of 64 for practical reasons
    recommended_batch_size = max(1, min(64, recommended_batch_size))
    
    print(f"Memory estimation:")
    print(f"  Model parameters: {model_params_count:,}")
    print(f"  Memory per sample: {memory_per_sample / (1024*1024):.2f} MB")
    print(f"  Available TPU memory: {tpu_memory_bytes / (1024*1024*1024):.1f} GB")
    print(f"  Recommended batch size: {recommended_batch_size}")
    
    return recommended_batch_size


def create_tpu_optimized_batch(batch_data, target_device=None):
    """
    Place batch data on TPU with optimal memory layout.
    
    Args:
        batch_data: Dictionary containing batch data
        target_device: Target TPU device (uses first device if None)
        
    Returns:
        Batch data placed on TPU
    """
    if target_device is None:
        target_device = jax.devices()[0]
    
    # Place all batch data on the target TPU device
    optimized_batch = {}
    for key, value in batch_data.items():
        if isinstance(value, jnp.ndarray):
            optimized_batch[key] = jax.device_put(value, target_device)
        else:
            optimized_batch[key] = value
    
    return optimized_batch


if __name__ == "__main__":
    # Configure JAX for TPU
    devices = configure_jax_for_tpu()
    
    # Test batch size optimization
    example_model_size = 100_000_000  # 100M parameters
    recommended_batch_size = optimize_batch_size_for_tpu(example_model_size)
    
    print(f"\nFor a {example_model_size:,} parameter model:")
    print(f"Recommended batch size: {recommended_batch_size}")