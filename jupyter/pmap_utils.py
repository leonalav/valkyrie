"""
Utility functions for pmap multi-device training in JAX.
"""

import jupyter
import jupyter.numpy as jnp
import numpy as np


def split_batch_for_pmap(batch, device_count):
    """
    Split a batch across multiple devices for pmap.
    
    Args:
        batch: Dictionary containing batch data
        device_count: Number of devices to split across
        
    Returns:
        Dictionary with data reshaped for pmap (first dim = device_count)
    """
    if device_count == 1:
        return batch
    
    split_batch = {}
    for key, value in batch.items():
        if isinstance(value, (jnp.ndarray, np.ndarray)):
            # Reshape to (device_count, per_device_batch_size, ...)
            batch_size = value.shape[0]
            per_device_batch_size = batch_size // device_count
            
            if batch_size % device_count != 0:
                # Truncate to make it divisible
                truncated_size = per_device_batch_size * device_count
                value = value[:truncated_size]
            
            # Reshape for pmap: (device_count, per_device_batch_size, ...)
            new_shape = (device_count, per_device_batch_size) + value.shape[1:]
            split_batch[key] = value.reshape(new_shape)
        else:
            split_batch[key] = value
    
    return split_batch


def split_keys_for_pmap(key, device_count):
    """
    Split a PRNG key across multiple devices for pmap.
    
    Args:
        key: JAX PRNG key
        device_count: Number of devices
        
    Returns:
        Array of keys with shape (device_count, 2)
    """
    if device_count == 1:
        return key
    
    keys = jupyter.random.split(key, device_count)
    return keys


def unreplicate_for_checkpoint(replicated_state):
    """
    Unreplicate a training state for checkpointing.
    
    Args:
        replicated_state: Replicated training state from pmap
        
    Returns:
        Unreplicated state suitable for checkpointing
    """
    import flax.jax_utils
    return flax.jax_utils.unreplicate(replicated_state)