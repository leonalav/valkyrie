#!/usr/bin/env python3
"""Check JAX configuration and available devices."""

import jax
import jax.numpy as jnp
import os

def check_jax_config():
    print("=" * 60)
    print("JAX CONFIGURATION CHECK")
    print("=" * 60)
    
    # Basic JAX info
    print(f"JAX version: {jax.__version__}")
    print(f"Default backend: {jax.default_backend()}")
    
    # Available devices
    devices = jax.devices()
    print(f"Available devices ({len(devices)}):")
    for i, device in enumerate(devices):
        print(f"  [{i}] {device}")
        
    # Memory information
    print("\nMemory Information:")
    for device in devices:
        if hasattr(device, 'memory_stats'):
            try:
                stats = device.memory_stats()
                print(f"  {device}: {stats}")
            except Exception as e:
                print(f"  {device}: Error getting memory stats - {e}")
        else:
            print(f"  {device}: No memory stats available")
    
    # Environment variables
    print("\nRelevant Environment Variables:")
    jax_vars = [
        'JAX_PLATFORM_NAME', 'JAX_PLATFORMS', 'CUDA_VISIBLE_DEVICES',
        'JAX_ENABLE_X64', 'XLA_PYTHON_CLIENT_MEM_FRACTION',
        'XLA_PYTHON_CLIENT_PREALLOCATE', 'JAX_TRACEBACK_FILTERING'
    ]
    
    for var in jax_vars:
        value = os.environ.get(var, 'Not set')
        print(f"  {var}: {value}")
    
    # Test memory allocation
    print("\nTesting Memory Allocation:")
    try:
        # Try to allocate a small array
        test_array = jnp.ones((1000, 1000), dtype=jnp.float32)
        print(f"  Successfully allocated 1000x1000 float32 array: {test_array.shape}")
        
        # Try a larger array
        test_array_large = jnp.ones((5000, 5000), dtype=jnp.float32)
        print(f"  Successfully allocated 5000x5000 float32 array: {test_array_large.shape}")
        
    except Exception as e:
        print(f"  Memory allocation failed: {e}")
    
    # Check if we're using GPU/TPU
    print(f"\nUsing accelerator: {any('gpu' in str(d).lower() or 'tpu' in str(d).lower() for d in devices)}")
    
    return devices

if __name__ == "__main__":
    check_jax_config()