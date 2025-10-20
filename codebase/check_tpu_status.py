#!/usr/bin/env python3
"""
Check TPU device configuration and memory usage.
"""

import jax
import jax.numpy as jnp
import os
import psutil

def check_tpu_configuration():
    """Check current TPU configuration and memory usage."""
    print("=== JAX TPU Configuration Check ===")
    
    # Check JAX backend
    print(f"JAX backend: {jax.default_backend()}")
    
    # Check available devices
    devices = jax.devices()
    print(f"Total devices: {len(devices)}")
    for i, device in enumerate(devices):
        print(f"  Device {i}: {device}")
    
    # Check local devices
    local_devices = jax.local_devices()
    print(f"Local devices: {len(local_devices)}")
    for i, device in enumerate(local_devices):
        print(f"  Local Device {i}: {device}")
    
    # Check device count by type
    tpu_devices = [d for d in devices if d.platform == 'tpu']
    print(f"TPU devices: {len(tpu_devices)}")
    
    # Check memory info for each device
    print("\n=== Memory Information ===")
    for i, device in enumerate(devices):
        try:
            # Get device memory info
            memory_info = device.memory_stats()
            print(f"Device {i} ({device.platform}):")
            for key, value in memory_info.items():
                if 'bytes' in key.lower():
                    print(f"  {key}: {value / (1024**3):.2f} GB")
                else:
                    print(f"  {key}: {value}")
        except Exception as e:
            print(f"Device {i}: Could not get memory info - {e}")
    
    # Check system memory
    print(f"\n=== System Memory ===")
    memory = psutil.virtual_memory()
    print(f"Total system memory: {memory.total / (1024**3):.2f} GB")
    print(f"Available system memory: {memory.available / (1024**3):.2f} GB")
    print(f"Used system memory: {memory.used / (1024**3):.2f} GB")
    
    # Check environment variables
    print(f"\n=== Environment Variables ===")
    env_vars = [
        'XLA_PYTHON_CLIENT_MEM_FRACTION',
        'XLA_PYTHON_CLIENT_PREALLOCATE',
        'XLA_PYTHON_CLIENT_ALLOCATOR',
        'TF_GPU_ALLOCATOR',
        'JAX_PLATFORM_NAME',
        'TPU_NAME'
    ]
    
    for var in env_vars:
        value = os.environ.get(var, 'Not set')
        print(f"  {var}: {value}")
    
    # Test simple computation on each device
    print(f"\n=== Device Test ===")
    test_array = jnp.ones((1000, 1000))
    
    for i, device in enumerate(devices):
        try:
            with jax.default_device(device):
                result = jnp.sum(test_array)
                print(f"Device {i}: Test computation successful - sum = {result}")
        except Exception as e:
            print(f"Device {i}: Test computation failed - {e}")
    
    # Check if distributed is initialized
    print(f"\n=== Distributed Configuration ===")
    try:
        print(f"Process index: {jax.process_index()}")
        print(f"Process count: {jax.process_count()}")
        print(f"Local device count: {jax.local_device_count()}")
        print(f"Device count: {jax.device_count()}")
    except Exception as e:
        print(f"Distributed not initialized: {e}")

if __name__ == "__main__":
    check_tpu_configuration()