"""TPU v4-32 mesh setup and management.

Implements the exact mesh topology and axis mapping recommended in precautionfortpu.md:
- 4×4×2 device array for 32 TPU v4 chips
- x,y axes for model parallelism (2D tensor sharding)  
- z axis for data parallelism
- Optimized for 3D torus topology and bisection bandwidth
"""

"""
Enhanced TPU Mesh Setup for Valkyrie Model Training

This module provides comprehensive mesh creation and management for TPU v4 configurations,
with support for dynamic configuration via environment variables, optimal device ordering,
topology assertions, and global mesh storage for reuse across modules.

Key improvements:
- Uses jax.devices() for correct physical device ordering on multi-host pods
- Environment variable parameterization for dynamic TPU configuration selection
- Topology assertions with experimental mesh utils when available
- Optional global mesh storage with thread-safe access patterns
- Comprehensive validation and error handling
"""

import os
import logging
import threading
from typing import Optional, Tuple, Dict, Any, List
from contextlib import contextmanager

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec

# Global mesh storage with thread-safe access
_global_mesh_lock = threading.RLock()
_global_mesh: Optional[Mesh] = None
_global_mesh_config: Optional[Dict[str, Any]] = None

logger = logging.getLogger(__name__)


def get_tpu_config_from_env() -> Dict[str, Any]:
    """
    Extract TPU configuration from environment variables.
    
    Environment variables:
    - TPU_MESH_CONFIG: Comma-separated topology (e.g., "4,4,2" for 32 devices)
    - TPU_AXIS_NAMES: Comma-separated axis names (e.g., "x,y,z")
    - TPU_DEVICE_COUNT: Override device count (optional)
    - TPU_FORCE_TOPOLOGY: Force specific topology regardless of device count
    - JAX_PLATFORMS: Should be set to "tpu" for TPU usage
    
    Returns:
        Dict containing TPU configuration parameters
    """
    config = {
        'device_count': None,
        'topology': None,
        'axis_names': None,
        'force_topology': False,
        'use_global_mesh': os.getenv('TPU_USE_GLOBAL_MESH', 'true').lower() == 'true'
    }
    
    # Parse mesh topology from environment
    mesh_config = os.getenv('TPU_MESH_CONFIG')
    if mesh_config:
        try:
            topology = tuple(map(int, mesh_config.split(',')))
            config['topology'] = topology
            config['device_count'] = np.prod(topology)
        except ValueError as e:
            logger.warning(f"Invalid TPU_MESH_CONFIG '{mesh_config}': {e}")
    
    # Parse axis names
    axis_names = os.getenv('TPU_AXIS_NAMES')
    if axis_names:
        config['axis_names'] = tuple(axis_names.split(','))
    
    # Override device count if specified
    device_count_env = os.getenv('TPU_DEVICE_COUNT')
    if device_count_env:
        try:
            config['device_count'] = int(device_count_env)
        except ValueError as e:
            logger.warning(f"Invalid TPU_DEVICE_COUNT '{device_count_env}': {e}")
    
    # Force topology flag
    config['force_topology'] = os.getenv('TPU_FORCE_TOPOLOGY', 'false').lower() == 'true'
    
    return config


def get_optimal_topology(device_count: int, force_2d_sharding: bool = True) -> Tuple[Tuple[int, ...], Tuple[str, ...]]:
    """
    Get optimal topology and axis names for given device count.
    
    Args:
        device_count: Number of TPU devices
        force_2d_sharding: If True, ensure at least 2D mesh with 'x' and 'z' axes for sharding compatibility
        
    Returns:
        Tuple of (topology_shape, axis_names)
        
    Raises:
        ValueError: If device count is not supported
    """
    # Optimal topologies for TPU v4 configurations
    # Updated to ensure 2D sharding compatibility with 'x' and 'z' axes
    topology_map = {
        1: ((1, 1), ('x', 'z')) if force_2d_sharding else ((1,), ('x',)),
        4: ((4, 1), ('x', 'z')) if force_2d_sharding else ((4,), ('x',)),
        8: ((2, 4), ('x', 'z')) if force_2d_sharding else ((2, 4), ('x', 'y')),
        16: ((4, 4), ('x', 'z')) if force_2d_sharding else ((4, 4), ('x', 'y')),
        32: ((4, 4, 2), ('x', 'y', 'z')),  # 4x4x2 for TPU v4-32
        64: ((4, 4, 4), ('x', 'y', 'z')),
        128: ((8, 4, 4), ('x', 'y', 'z')),
        256: ((8, 8, 4), ('x', 'y', 'z')),
    }
    
    if device_count not in topology_map:
        # Try to find a reasonable factorization
        factors = []
        n = device_count
        for i in [2, 4, 8]:
            while n % i == 0:
                factors.append(i)
                n //= i
        
        if n == 1 and len(factors) <= 3:
            # Create topology from factors
            topology = tuple(factors + [1] * (3 - len(factors)))[:3]
            axis_names = ('x', 'y', 'z')[:len([f for f in topology if f > 1])]
            logger.info(f"Using computed topology {topology} for {device_count} devices")
            return topology, axis_names
        else:
            raise ValueError(f"Unsupported device count: {device_count}. "
                           f"Supported counts: {list(topology_map.keys())}")
    
    return topology_map[device_count]


def create_device_mesh_with_assertions(devices: List[Any], topology: Tuple[int, ...], 
                                     axis_names: Tuple[str, ...]) -> Mesh:
    """
    Create device mesh with topology assertions and experimental utils when available.
    
    Args:
        devices: List of JAX devices in physical order
        topology: Mesh topology shape
        axis_names: Axis names for the mesh
        
    Returns:
        JAX Mesh object
        
    Raises:
        ValueError: If topology doesn't match device count or other validation fails
    """
    device_count = len(devices)
    expected_count = np.prod(topology)
    
    if device_count != expected_count:
        raise ValueError(f"Device count mismatch: got {device_count} devices, "
                        f"topology {topology} expects {expected_count}")
    
    if len(axis_names) != len(topology):
        raise ValueError(f"Axis names count {len(axis_names)} doesn't match "
                        f"topology dimensions {len(topology)}")
    
    # Try to use experimental mesh utils if available
    try:
        from jax.experimental import mesh_utils
        if hasattr(mesh_utils, 'create_device_mesh'):
            logger.info("Using jax.experimental.mesh_utils.create_device_mesh for optimal layout")
            device_mesh = mesh_utils.create_device_mesh(topology, devices)
            mesh = Mesh(device_mesh, axis_names)
            logger.info(f"Created mesh using experimental utils: {mesh}")
            return mesh
    except (ImportError, AttributeError) as e:
        logger.debug(f"Experimental mesh utils not available: {e}")
    
    # Fallback to standard mesh creation
    try:
        # Reshape devices to match topology
        device_array = np.array(devices).reshape(topology)
        mesh = Mesh(device_array, axis_names)
        logger.info(f"Created mesh using standard approach: {mesh}")
        return mesh
    except Exception as e:
        logger.error(f"Failed to create mesh with topology {topology}: {e}")
        # Final fallback: create 1D mesh
        logger.warning("Falling back to 1D mesh configuration")
        return Mesh(np.array(devices), ('devices',))


def make_mesh(device_count: Optional[int] = None, 
              topology: Optional[Tuple[int, ...]] = None,
              axis_names: Optional[Tuple[str, ...]] = None,
              use_env_config: bool = True) -> Mesh:
    """
    Create a JAX Mesh for TPU training with enhanced configuration options.
    
    This function creates an optimal mesh topology for various TPU v4 configurations,
    using jax.devices() for correct physical device ordering and supporting
    environment variable configuration.
    
    Args:
        device_count: Number of devices (auto-detected if None)
        topology: Explicit topology shape (overrides auto-detection)
        axis_names: Explicit axis names (overrides auto-detection)
        use_env_config: Whether to use environment variable configuration
        
    Returns:
        JAX Mesh object configured for optimal TPU performance
        
    Raises:
        ValueError: If configuration is invalid or unsupported
        RuntimeError: If no devices are available
    """
    # Get configuration from environment if requested
    env_config = get_tpu_config_from_env() if use_env_config else {}
    
    # Use jax.devices() for correct physical ordering
    devices = jax.devices()
    if not devices:
        raise RuntimeError("No JAX devices available")
    
    actual_device_count = len(devices)
    logger.info(f"Found {actual_device_count} JAX devices: {[d.platform for d in devices]}")
    
    # Determine device count to use
    if device_count is None:
        device_count = env_config.get('device_count', actual_device_count)
    
    # Ensure device_count is not None
    if device_count is None:
        device_count = actual_device_count
        logger.info(f"Using all available devices: {device_count}")
    
    if device_count > actual_device_count:
        raise ValueError(f"Requested {device_count} devices, but only {actual_device_count} available")
    
    # Use subset of devices if needed
    if device_count < actual_device_count:
        devices = devices[:device_count]
        logger.info(f"Using {device_count} out of {actual_device_count} available devices")
    
    # Determine topology and axis names
    if topology is None:
        topology = env_config.get('topology')
    if axis_names is None:
        axis_names = env_config.get('axis_names')
    
    if topology is None or axis_names is None:
            auto_topology, auto_axis_names = get_optimal_topology(device_count, force_2d_sharding=True)
            if topology is None:
                topology = auto_topology
            if axis_names is None:
                axis_names = auto_axis_names
    
    # Force topology if requested
    if env_config.get('force_topology') and env_config.get('topology'):
        topology = env_config['topology']
        logger.info(f"Forcing topology {topology} from environment")
    
    logger.info(f"Creating mesh with topology {topology}, axis names {axis_names}")
    
    # Create mesh with assertions and experimental utils
    mesh = create_device_mesh_with_assertions(devices, topology, axis_names)
    
    # Log mesh information
    logger.info(f"Created mesh: {mesh}")
    logger.info(f"Mesh shape: {mesh.shape}")
    logger.info(f"Mesh axis names: {mesh.axis_names}")
    logger.info(f"Device array shape: {mesh.devices.shape}")
    
    return mesh


def get_global_mesh() -> Optional[Mesh]:
    """
    Get the globally stored mesh if available.
    
    Returns:
        Global mesh or None if not set
    """
    with _global_mesh_lock:
        return _global_mesh


def set_global_mesh(mesh: Mesh, config: Optional[Dict[str, Any]] = None) -> None:
    """
    Set the global mesh for reuse across modules.
    
    Args:
        mesh: JAX Mesh to store globally
        config: Optional configuration dict for reference
    """
    global _global_mesh, _global_mesh_config
    with _global_mesh_lock:
        _global_mesh = mesh
        _global_mesh_config = config or {}
        logger.info(f"Set global mesh: {mesh}")


def clear_global_mesh() -> None:
    """Clear the global mesh storage."""
    global _global_mesh, _global_mesh_config
    with _global_mesh_lock:
        _global_mesh = None
        _global_mesh_config = None
        logger.info("Cleared global mesh")


def make_or_get_global_mesh(**kwargs) -> Mesh:
    """
    Create a new mesh or return the global mesh if available and compatible.
    
    Args:
        **kwargs: Arguments passed to make_mesh()
        
    Returns:
        JAX Mesh object
    """
    env_config = get_tpu_config_from_env()
    
    if env_config.get('use_global_mesh', True):
        with _global_mesh_lock:
            if _global_mesh is not None:
                logger.info("Reusing global mesh")
                return _global_mesh
    
    # Create new mesh
    mesh = make_mesh(**kwargs)
    
    # Store globally if requested
    if env_config.get('use_global_mesh', True):
        set_global_mesh(mesh, env_config)
    
    return mesh


def get_mesh_context(mesh: Optional[Mesh] = None, **mesh_kwargs):
    """
    Context manager for mesh operations with automatic cleanup.
    
    Args:
        mesh: Existing mesh to use, or None to create one
        **mesh_kwargs: Arguments for mesh creation if mesh is None
        
    Yields:
        JAX Mesh object
    """
    if mesh is None:
        mesh = make_or_get_global_mesh(**mesh_kwargs)
    
    try:
        yield mesh
    finally:
        # Cleanup if needed
        pass


def validate_mesh_setup(mesh: Mesh) -> bool:
    """
    Validate mesh configuration for training compatibility.
    
    Args:
        mesh: JAX Mesh to validate
        
    Returns:
        True if mesh is valid for training
        
    Raises:
        ValueError: If mesh configuration is invalid
    """
    if not isinstance(mesh, Mesh):
        raise ValueError(f"Expected Mesh object, got {type(mesh)}")
    
    # Check device availability
    devices = mesh.devices.flatten()
    if len(devices) == 0:
        raise ValueError("Mesh has no devices")
    
    # Validate device platforms
    platforms = {d.platform for d in devices}
    if len(platforms) > 1:
        logger.warning(f"Mixed device platforms in mesh: {platforms}")
    
    # Check for TPU-specific requirements
    if 'tpu' in platforms:
        device_count = len(devices)
        if device_count not in [1, 4, 8, 16, 32, 64, 128, 256]:
            logger.warning(f"Unusual TPU device count: {device_count}")
    
    # Validate axis names
    if not mesh.axis_names:
        raise ValueError("Mesh has no axis names")
    
    if len(set(mesh.axis_names)) != len(mesh.axis_names):
        raise ValueError(f"Duplicate axis names: {mesh.axis_names}")
    
    logger.info(f"Mesh validation passed: {mesh}")
    return True


def get_mesh_info(mesh: Mesh) -> Dict[str, Any]:
    """
    Get comprehensive information about a mesh.
    
    Args:
        mesh: JAX Mesh to analyze
        
    Returns:
        Dictionary with mesh information
    """
    devices = mesh.devices.flatten()
    
    info = {
        'shape': mesh.shape,
        'axis_names': mesh.axis_names,
        'device_count': len(devices),
        'device_platforms': list({d.platform for d in devices}),
        'device_array_shape': mesh.devices.shape,
        'total_memory_gb': None,  # Could be computed if needed
        'is_multi_host': len(set(d.process_index for d in devices)) > 1,
        'process_indices': sorted(set(d.process_index for d in devices)),
    }
    
    # Add TPU-specific information
    if 'tpu' in info['device_platforms']:
        info['tpu_topology'] = mesh.shape
        info['is_tpu_pod'] = len(devices) > 8
    
    return info


def print_mesh_info(mesh: Mesh) -> None:
    """
    Print detailed mesh information for debugging.
    
    Args:
        mesh: JAX Mesh to analyze
    """
    info = get_mesh_info(mesh)
    
    print(f"\n=== Mesh Information ===")
    print(f"Shape: {info['shape']}")
    print(f"Axis names: {info['axis_names']}")
    print(f"Device count: {info['device_count']}")
    print(f"Device platforms: {info['device_platforms']}")
    print(f"Device array shape: {info['device_array_shape']}")
    print(f"Multi-host: {info['is_multi_host']}")
    
    if info['is_multi_host']:
        print(f"Process indices: {info['process_indices']}")
    
    if 'tpu_topology' in info:
        print(f"TPU topology: {info['tpu_topology']}")
        print(f"TPU pod: {info['is_tpu_pod']}")
    
    print(f"========================\n")


def setup_tpu_mesh(device_count: Optional[int] = None, 
                   use_global: bool = True,
                   validate: bool = True,
                   print_info: bool = True) -> Mesh:
    """
    High-level function to set up TPU mesh with all enhancements.
    
    Args:
        device_count: Number of devices (auto-detected if None)
        use_global: Whether to use/store global mesh
        validate: Whether to validate mesh configuration
        print_info: Whether to print mesh information
        
    Returns:
        Configured JAX Mesh
        
    Raises:
        ValueError: If mesh configuration is invalid
        RuntimeError: If mesh setup fails
    """
    try:
        logger.info("Setting up TPU mesh with enhanced configuration")
        
        # Create or get mesh
        if use_global:
            mesh = make_or_get_global_mesh(device_count=device_count)
        else:
            mesh = make_mesh(device_count=device_count)
        
        # Validate if requested
        if validate:
            validate_mesh_setup(mesh)
        
        # Print information if requested
        if print_info:
            print_mesh_info(mesh)
        
        logger.info("TPU mesh setup completed successfully")
        return mesh
        
    except Exception as e:
        logger.error(f"Failed to setup TPU mesh: {e}")
        raise RuntimeError(f"TPU mesh setup failed: {e}") from e


# Environment variable usage examples and documentation
def print_env_usage() -> None:
    """Print usage examples for environment variable configuration."""
    print("""
=== TPU Mesh Environment Variable Configuration ===

Environment Variables:
  TPU_MESH_CONFIG     - Comma-separated topology (e.g., "4,4,2" for 32 devices)
  TPU_AXIS_NAMES      - Comma-separated axis names (e.g., "x,y,z")
  TPU_DEVICE_COUNT    - Override device count (optional)
  TPU_FORCE_TOPOLOGY  - Force specific topology regardless of device count
  TPU_USE_GLOBAL_MESH - Enable global mesh storage (default: true)
  JAX_PLATFORMS       - Should be set to "tpu" for TPU usage

Examples:
  # TPU v4-8 configuration
  export TPU_MESH_CONFIG="2,4"
  export TPU_AXIS_NAMES="x,y"
  
  # TPU v4-32 configuration  
  export TPU_MESH_CONFIG="4,4,2"
  export TPU_AXIS_NAMES="x,y,z"
  
  # Force 1D mesh for data parallelism only
  export TPU_MESH_CONFIG="32"
  export TPU_AXIS_NAMES="data"
  export TPU_FORCE_TOPOLOGY="true"
  
  # Disable global mesh storage
  export TPU_USE_GLOBAL_MESH="false"

Usage in code:
  # Use environment configuration
  mesh = setup_tpu_mesh()
  
  # Override environment
  mesh = make_mesh(device_count=8, topology=(2,4), axis_names=('x','y'))
  
  # Use global mesh
  mesh = make_or_get_global_mesh()

====================================================
""")


if __name__ == "__main__":
    # Example usage and testing
    print_env_usage()
    
    # Test mesh creation
    try:
        mesh = setup_tpu_mesh()
        print("Mesh setup successful!")
    except Exception as e:
        print(f"Mesh setup failed: {e}")