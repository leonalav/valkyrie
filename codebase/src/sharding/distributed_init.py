"""Multi-host TPU initialization and distributed runtime setup.

This module provides initialization functions for multi-host TPU v4-32 training,
handling distributed runtime setup, coordinator address detection, and process
rank management for 4-host × 8-chip configurations.

Key features:
- Automatic coordinator address detection from environment
- Process rank determination for multi-host setups  
- Idempotent initialization (safe to call multiple times)
- Compatible with 3D mesh configurations (4×4×2)
- Proper error handling and logging
"""

import os
import logging
import socket
from typing import Optional, Dict, Any, Tuple
from contextlib import contextmanager

import jax
import jax.distributed

logger = logging.getLogger(__name__)

# Global state to track initialization
_distributed_initialized = False
_initialization_config: Optional[Dict[str, Any]] = None


def is_multi_host_environment() -> bool:
    """
    Detect if we're running in a multi-host TPU environment.
    
    Returns:
        True if multi-host environment detected
    """
    # Check for TPU pod environment variables
    tpu_name = os.getenv('TPU_NAME')
    if tpu_name and ('v4-32' in tpu_name or 'v4-64' in tpu_name):
        return True
    
    # Check for explicit multi-host configuration
    if os.getenv('JAX_COORDINATOR_ADDRESS'):
        return True
    
    # Check for process index environment variable
    process_index = os.getenv('JAX_PROCESS_INDEX')
    if process_index is not None:
        return True
    
    # Check device count - v4-32 has 32 devices across 4 hosts
    try:
        device_count = len(jax.devices())
        if device_count >= 32:
            logger.info(f"Detected {device_count} devices, likely multi-host environment")
            return True
    except Exception as e:
        logger.debug(f"Could not check device count: {e}")
    
    return False


def get_coordinator_address() -> str:
    """
    Get the coordinator address for distributed training.
    
    Returns:
        Coordinator address string
        
    Raises:
        ValueError: If coordinator address cannot be determined
    """
    # Check environment variable first
    coordinator_addr = os.getenv('JAX_COORDINATOR_ADDRESS')
    if coordinator_addr:
        logger.info(f"Using coordinator address from environment: {coordinator_addr}")
        return coordinator_addr
    
    # For TPU pods, try to detect from metadata
    try:
        # This is a common pattern for TPU pods
        hostname = socket.gethostname()
        if 'tpu-vm' in hostname:
            # Extract base hostname and use first host as coordinator
            base_name = hostname.rsplit('-', 1)[0]
            coordinator_host = f"{base_name}-0"
            coordinator_addr = f"{coordinator_host}:8476"
            logger.info(f"Detected TPU pod coordinator: {coordinator_addr}")
            return coordinator_addr
    except Exception as e:
        logger.debug(f"Could not auto-detect coordinator from hostname: {e}")
    
    # Default fallback - assume first host is coordinator
    coordinator_addr = "10.0.0.2:8476"  # Common TPU pod internal IP
    logger.warning(f"Using default coordinator address: {coordinator_addr}")
    return coordinator_addr


def get_process_index() -> int:
    """
    Get the process index (rank) for the current host.
    
    Returns:
        Process index (0-based)
        
    Raises:
        ValueError: If process index cannot be determined
    """
    # Check environment variable first
    process_index_env = os.getenv('JAX_PROCESS_INDEX')
    if process_index_env is not None:
        try:
            process_index = int(process_index_env)
            logger.info(f"Using process index from environment: {process_index}")
            return process_index
        except ValueError as e:
            logger.error(f"Invalid JAX_PROCESS_INDEX '{process_index_env}': {e}")
            raise
    
    # Try to detect from hostname for TPU pods
    try:
        hostname = socket.gethostname()
        if 'tpu-vm' in hostname and hostname.endswith(('-0', '-1', '-2', '-3')):
            # Extract process index from hostname suffix
            process_index = int(hostname[-1])
            logger.info(f"Detected process index from hostname: {process_index}")
            return process_index
    except Exception as e:
        logger.debug(f"Could not auto-detect process index from hostname: {e}")
    
    # Default to 0 if we can't determine
    logger.warning("Could not determine process index, defaulting to 0")
    return 0


def get_process_count() -> int:
    """
    Get the total number of processes (hosts) in the distributed setup.
    
    Returns:
        Total number of processes
    """
    # Check environment variable
    process_count_env = os.getenv('JAX_PROCESS_COUNT')
    if process_count_env is not None:
        try:
            process_count = int(process_count_env)
            logger.info(f"Using process count from environment: {process_count}")
            return process_count
        except ValueError as e:
            logger.error(f"Invalid JAX_PROCESS_COUNT '{process_count_env}': {e}")
    
    # For TPU v4-32, we have 4 hosts
    tpu_name = os.getenv('TPU_NAME')
    if tpu_name and 'v4-32' in tpu_name:
        logger.info("Detected TPU v4-32, using 4 processes")
        return 4
    
    # Default to 1 for single-host
    logger.info("Defaulting to single process")
    return 1


def initialize_distributed_runtime(
    coordinator_address: Optional[str] = None,
    process_index: Optional[int] = None,
    process_count: Optional[int] = None,
    force_reinit: bool = False
) -> Dict[str, Any]:
    """
    Initialize JAX distributed runtime for multi-host training.
    
    Args:
        coordinator_address: Coordinator address (auto-detected if None)
        process_index: Process index/rank (auto-detected if None)
        process_count: Total number of processes (auto-detected if None)
        force_reinit: Force re-initialization even if already initialized
        
    Returns:
        Dictionary with initialization configuration
        
    Raises:
        RuntimeError: If initialization fails
    """
    global _distributed_initialized, _initialization_config
    
    # Check if already initialized
    if _distributed_initialized and not force_reinit:
        logger.info("Distributed runtime already initialized")
        return _initialization_config or {}
    
    try:
        logger.info("Initializing JAX distributed runtime for multi-host training")
        
        # Auto-detect parameters if not provided
        if coordinator_address is None:
            coordinator_address = get_coordinator_address()
        if process_index is None:
            process_index = get_process_index()
        if process_count is None:
            process_count = get_process_count()
        
        # Validate parameters
        if process_index < 0 or process_index >= process_count:
            raise ValueError(f"Invalid process_index {process_index} for process_count {process_count}")
        
        logger.info(f"Initializing with coordinator={coordinator_address}, "
                   f"process_index={process_index}, process_count={process_count}")
        
        # Initialize distributed runtime
        jax.distributed.initialize(
            coordinator_address=coordinator_address,
            num_processes=process_count,
            process_id=process_index
        )
        
        # Store configuration
        config = {
            'coordinator_address': coordinator_address,
            'process_index': process_index,
            'process_count': process_count,
            'local_device_count': jax.local_device_count(),
            'global_device_count': jax.device_count(),
        }
        
        _distributed_initialized = True
        _initialization_config = config
        
        logger.info("JAX distributed runtime initialized successfully")
        logger.info(f"Local devices: {config['local_device_count']}, "
                   f"Global devices: {config['global_device_count']}")
        
        return config
        
    except Exception as e:
        logger.error(f"Failed to initialize distributed runtime: {e}")
        raise RuntimeError(f"Distributed initialization failed: {e}") from e


def is_distributed_initialized() -> bool:
    """
    Check if distributed runtime has been initialized.
    
    Returns:
        True if distributed runtime is initialized
    """
    return _distributed_initialized


def get_initialization_config() -> Optional[Dict[str, Any]]:
    """
    Get the current initialization configuration.
    
    Returns:
        Configuration dictionary or None if not initialized
    """
    return _initialization_config


def setup_multi_host_environment() -> Dict[str, Any]:
    """
    High-level function to set up multi-host environment for TPU training.
    
    This function:
    1. Detects if we're in a multi-host environment
    2. Initializes distributed runtime if needed
    3. Returns configuration information
    
    Returns:
        Dictionary with environment and initialization information
        
    Raises:
        RuntimeError: If setup fails
    """
    try:
        logger.info("Setting up multi-host TPU environment")
        
        # Check if multi-host environment
        is_multi_host = is_multi_host_environment()
        
        config = {
            'is_multi_host': is_multi_host,
            'distributed_initialized': False,
        }
        
        if is_multi_host:
            logger.info("Multi-host environment detected, initializing distributed runtime")
            init_config = initialize_distributed_runtime()
            config.update(init_config)
            config['distributed_initialized'] = True
        else:
            logger.info("Single-host environment detected, skipping distributed initialization")
            config.update({
                'process_index': 0,
                'process_count': 1,
                'local_device_count': jax.local_device_count(),
                'global_device_count': jax.device_count(),
            })
        
        logger.info("Multi-host environment setup completed")
        return config
        
    except Exception as e:
        logger.error(f"Failed to setup multi-host environment: {e}")
        raise RuntimeError(f"Multi-host setup failed: {e}") from e


@contextmanager
def distributed_context(auto_setup: bool = True):
    """
    Context manager for distributed operations.
    
    Args:
        auto_setup: Whether to automatically set up multi-host environment
        
    Yields:
        Configuration dictionary
    """
    config = None
    try:
        if auto_setup:
            config = setup_multi_host_environment()
        yield config
    finally:
        # Cleanup if needed
        pass


def print_distributed_info() -> None:
    """Print information about the distributed setup."""
    if not _distributed_initialized:
        print("Distributed runtime not initialized")
        return
    
    config = _initialization_config or {}
    
    print(f"\n=== Distributed Runtime Information ===")
    print(f"Coordinator address: {config.get('coordinator_address', 'N/A')}")
    print(f"Process index: {config.get('process_index', 'N/A')}")
    print(f"Process count: {config.get('process_count', 'N/A')}")
    print(f"Local devices: {config.get('local_device_count', 'N/A')}")
    print(f"Global devices: {config.get('global_device_count', 'N/A')}")
    print(f"Multi-host: {config.get('process_count', 1) > 1}")
    print(f"=======================================\n")


if __name__ == "__main__":
    # Example usage and testing
    try:
        config = setup_multi_host_environment()
        print_distributed_info()
        print("Multi-host setup successful!")
    except Exception as e:
        print(f"Multi-host setup failed: {e}")