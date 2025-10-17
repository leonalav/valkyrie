"""Debugging and mixed precision utilities.

Implements:
- Mixed precision policy for TPU training
- Shape debugging and validation
- NaN/Inf detection and handling
- Parameter statistics and analysis
- Memory usage monitoring
- Gradient flow analysis
"""

import jax
import jax.numpy as jnp
from typing import Dict, Any, Optional, Tuple, Union, Callable
import logging
import numpy as np
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class MixedPrecisionConfig:
    """Configuration for mixed precision training."""
    # Global precision policy
    param_dtype: jnp.dtype = jnp.float32      # Parameters (always fp32 for stability)
    compute_dtype: jnp.dtype = jnp.bfloat16   # General computation (bfloat16 for TPU)
    
    # Specific operation overrides (critical for numerical stability)
    attention_softmax_dtype: jnp.dtype = jnp.float32    # Attention softmax (fp32 for stability)
    s5_complex_dtype: jnp.dtype = jnp.complex64         # S5 complex arithmetic (complex64)
    layernorm_dtype: jnp.dtype = jnp.float32            # Layer normalization (fp32)
    
    # Loss computation
    loss_dtype: jnp.dtype = jnp.float32       # Loss computation (fp32)
    
    # Gradient handling
    grad_dtype: jnp.dtype = jnp.float32       # Gradients (fp32)
    
    # Casting policy
    cast_inputs: bool = True                   # Cast inputs to compute_dtype
    cast_outputs: bool = True                  # Cast outputs back to param_dtype


class MixedPrecisionPolicy:
    """
    Mixed precision policy for Valkyrie training.
    
    Implements the exact precision requirements from output.txt and precautionfortpu.md:
    - Keep attention softmax in fp32 to avoid NaNs
    - Keep S5 complex arithmetic in complex64 for stability
    - Use bfloat16 for general computation on TPU
    - Maintain fp32 for parameters and gradients
    """
    
    def __init__(self, config: MixedPrecisionConfig):
        self.config = config
        
        logger.info("Mixed precision policy initialized:")
        logger.info(f"  Parameters: {config.param_dtype}")
        logger.info(f"  Computation: {config.compute_dtype}")
        logger.info(f"  Attention softmax: {config.attention_softmax_dtype}")
        logger.info(f"  S5 complex: {config.s5_complex_dtype}")
        logger.info(f"  Layer norm: {config.layernorm_dtype}")
    
    def cast_for_computation(self, x: jnp.ndarray, operation_type: str = "general") -> jnp.ndarray:
        """
        Cast tensor for specific computation type.
        
        Args:
            x: Input tensor
            operation_type: Type of operation ("general", "attention", "s5", "layernorm", "loss")
            
        Returns:
            Tensor cast to appropriate dtype
        """
        
        if operation_type == "attention":
            # Attention operations use fp32 for softmax stability
            return x.astype(self.config.attention_softmax_dtype)
        elif operation_type == "s5":
            # S5 operations use complex64 for numerical stability
            if jnp.iscomplexobj(x):
                return x.astype(self.config.s5_complex_dtype)
            else:
                # Real inputs for S5 stay in fp32 initially
                return x.astype(jnp.float32)
        elif operation_type == "layernorm":
            # Layer normalization uses fp32 for stability
            return x.astype(self.config.layernorm_dtype)
        elif operation_type == "loss":
            # Loss computation uses fp32
            return x.astype(self.config.loss_dtype)
        else:
            # General computation uses compute_dtype (bfloat16 on TPU)
            if self.config.cast_inputs:
                return x.astype(self.config.compute_dtype)
            else:
                return x
    
    def cast_for_output(self, x: jnp.ndarray) -> jnp.ndarray:
        """Cast tensor for output (typically back to param_dtype)."""
        if self.config.cast_outputs and not jnp.iscomplexobj(x):
            return x.astype(self.config.param_dtype)
        else:
            return x
    
    def get_param_dtype(self) -> jnp.dtype:
        """Get dtype for parameters."""
        return self.config.param_dtype
    
    def get_grad_dtype(self) -> jnp.dtype:
        """Get dtype for gradients."""
        return self.config.grad_dtype


def debug_shapes(pytree: Any, name: str = "tensor", max_depth: int = 3) -> None:
    """
    Debug shapes of a pytree structure.
    
    Args:
        pytree: JAX pytree to analyze
        name: Name for logging
        max_depth: Maximum depth to traverse
    """
    
    def _debug_shapes_recursive(tree, path: str, depth: int):
        if depth > max_depth:
            logger.debug(f"{path}: <max depth reached>")
            return
        
        if isinstance(tree, jnp.ndarray):
            logger.debug(f"{path}: shape={tree.shape}, dtype={tree.dtype}")
        elif isinstance(tree, dict):
            for key, value in tree.items():
                _debug_shapes_recursive(value, f"{path}.{key}", depth + 1)
        elif isinstance(tree, (list, tuple)):
            for i, value in enumerate(tree):
                _debug_shapes_recursive(value, f"{path}[{i}]", depth + 1)
        else:
            logger.debug(f"{path}: {type(tree)}")
    
    logger.debug(f"=== Shape Debug: {name} ===")
    _debug_shapes_recursive(pytree, name, 0)
    logger.debug("=" * (len(name) + 20))


def check_for_nans(pytree: Any, name: str = "tensor") -> bool:
    """
    Check for NaN or Inf values in a pytree.
    
    Args:
        pytree: JAX pytree to check
        name: Name for logging
        
    Returns:
        True if NaN/Inf found
    """
    
    def _check_array(arr: jnp.ndarray, path: str) -> bool:
        if jnp.iscomplexobj(arr):
            # Check both real and imaginary parts
            has_nan_real = jnp.any(jnp.isnan(arr.real))
            has_inf_real = jnp.any(jnp.isinf(arr.real))
            has_nan_imag = jnp.any(jnp.isnan(arr.imag))
            has_inf_imag = jnp.any(jnp.isinf(arr.imag))
            
            if has_nan_real or has_nan_imag:
                logger.error(f"NaN detected in {path}: real={has_nan_real}, imag={has_nan_imag}")
                return True
            if has_inf_real or has_inf_imag:
                logger.error(f"Inf detected in {path}: real={has_inf_real}, imag={has_inf_imag}")
                return True
        else:
            # Real-valued array
            has_nan = jnp.any(jnp.isnan(arr))
            has_inf = jnp.any(jnp.isinf(arr))
            
            if has_nan:
                logger.error(f"NaN detected in {path}")
                return True
            if has_inf:
                logger.error(f"Inf detected in {path}")
                return True
        
        return False
    
    def _check_recursive(tree, path: str) -> bool:
        found_issues = False
        
        if isinstance(tree, jnp.ndarray):
            found_issues |= _check_array(tree, path)
        elif isinstance(tree, dict):
            for key, value in tree.items():
                found_issues |= _check_recursive(value, f"{path}.{key}")
        elif isinstance(tree, (list, tuple)):
            for i, value in enumerate(tree):
                found_issues |= _check_recursive(value, f"{path}[{i}]")
        
        return found_issues
    
    has_issues = _check_recursive(pytree, name)
    
    if has_issues:
        logger.error(f"❌ NaN/Inf check failed for {name}")
    else:
        logger.debug(f"✓ NaN/Inf check passed for {name}")
    
    return has_issues


def print_param_stats(params: Dict[str, Any], name: str = "model") -> Dict[str, Any]:
    """
    Print statistics about model parameters.
    
    Args:
        params: Parameter pytree
        name: Name for logging
        
    Returns:
        Dictionary with parameter statistics
    """
    
    def _collect_arrays(tree, arrays: list, paths: list, path: str = ""):
        if isinstance(tree, jnp.ndarray):
            arrays.append(tree)
            paths.append(path)
        elif isinstance(tree, dict):
            for key, value in tree.items():
                _collect_arrays(value, arrays, paths, f"{path}.{key}" if path else key)
        elif isinstance(tree, (list, tuple)):
            for i, value in enumerate(tree):
                _collect_arrays(value, arrays, paths, f"{path}[{i}]")
    
    # Collect all arrays
    arrays = []
    paths = []
    _collect_arrays(params, arrays, paths)
    
    # Compute statistics
    total_params = sum(arr.size for arr in arrays)
    total_bytes = sum(arr.nbytes for arr in arrays)
    
    # Group by dtype
    dtype_stats = {}
    for arr in arrays:
        dtype = str(arr.dtype)
        if dtype not in dtype_stats:
            dtype_stats[dtype] = {'count': 0, 'params': 0, 'bytes': 0}
        
        dtype_stats[dtype]['count'] += 1
        dtype_stats[dtype]['params'] += arr.size
        dtype_stats[dtype]['bytes'] += arr.nbytes
    
    # Find largest parameters
    largest_params = sorted(
        [(path, arr.size, arr.shape, arr.dtype) for path, arr in zip(paths, arrays)],
        key=lambda x: x[1],
        reverse=True
    )[:10]
    
    # Compute value statistics
    all_values = jnp.concatenate([arr.flatten() for arr in arrays if not jnp.iscomplexobj(arr)])
    if len(all_values) > 0:
        value_stats = {
            'mean': float(jnp.mean(all_values)),
            'std': float(jnp.std(all_values)),
            'min': float(jnp.min(all_values)),
            'max': float(jnp.max(all_values)),
            'abs_mean': float(jnp.mean(jnp.abs(all_values))),
        }
    else:
        value_stats = {}
    
    # Create summary
    stats = {
        'total_parameters': total_params,
        'total_bytes': total_bytes,
        'total_mb': total_bytes / (1024 * 1024),
        'total_gb': total_bytes / (1024 * 1024 * 1024),
        'num_arrays': len(arrays),
        'dtype_breakdown': dtype_stats,
        'largest_parameters': largest_params,
        'value_statistics': value_stats,
    }
    
    # Log summary
    logger.info(f"=== Parameter Statistics: {name} ===")
    logger.info(f"Total parameters: {total_params:,} ({total_params/1e6:.2f}M, {total_params/1e9:.2f}B)")
    logger.info(f"Total memory: {total_bytes/1024/1024:.1f} MB ({total_bytes/1024/1024/1024:.2f} GB)")
    logger.info(f"Number of arrays: {len(arrays)}")
    
    logger.info("Dtype breakdown:")
    for dtype, info in dtype_stats.items():
        logger.info(f"  {dtype}: {info['count']} arrays, {info['params']:,} params, {info['bytes']/1024/1024:.1f} MB")
    
    if value_stats:
        logger.info("Value statistics:")
        logger.info(f"  Mean: {value_stats['mean']:.6f}, Std: {value_stats['std']:.6f}")
        logger.info(f"  Range: [{value_stats['min']:.6f}, {value_stats['max']:.6f}]")
        logger.info(f"  Abs mean: {value_stats['abs_mean']:.6f}")
    
    logger.info("Largest parameters:")
    for path, size, shape, dtype in largest_params[:5]:
        logger.info(f"  {path}: {size:,} params, shape={shape}, dtype={dtype}")
    
    logger.info("=" * (len(name) + 30))
    
    return stats


def validate_s5_gradients(
    s5_params: Dict[str, Any],
    loss_fn: Callable,
    inputs: jnp.ndarray,
    tolerance: float = 1e-5,
) -> bool:
    """
    Validate S5 gradients using finite differences.
    
    Args:
        s5_params: S5 parameters
        loss_fn: Loss function
        inputs: Input data
        tolerance: Tolerance for gradient check
        
    Returns:
        True if gradients are valid
    """
    
    logger.info("Validating S5 gradients with finite differences...")
    
    try:
        # Compute analytical gradients
        loss, grads = jax.value_and_grad(loss_fn)(s5_params, inputs)
        
        # Compute finite difference gradients for a subset of parameters
        def finite_diff_grad(params, param_path, h=1e-5):
            """Compute finite difference gradient for a specific parameter."""
            
            def get_param(p, path):
                for key in path:
                    p = p[key]
                return p
            
            def set_param(p, path, value):
                p = p.copy()
                current = p
                for key in path[:-1]:
                    current = current[key]
                current[path[-1]] = value
                return p
            
            # Get original parameter
            orig_param = get_param(params, param_path)
            
            if jnp.iscomplexobj(orig_param):
                # For complex parameters, check real and imaginary parts separately
                grad_real = jnp.zeros_like(orig_param.real)
                grad_imag = jnp.zeros_like(orig_param.imag)
                
                # Check a few elements (not all for efficiency)
                flat_real = orig_param.real.flatten()
                flat_imag = orig_param.imag.flatten()
                
                indices = jnp.linspace(0, len(flat_real)-1, min(10, len(flat_real)), dtype=int)
                
                for idx in indices:
                    # Real part
                    param_plus = orig_param.at[jnp.unravel_index(idx, orig_param.shape)].add(h)
                    param_minus = orig_param.at[jnp.unravel_index(idx, orig_param.shape)].add(-h)
                    
                    params_plus = set_param(params, param_path, param_plus)
                    params_minus = set_param(params, param_path, param_minus)
                    
                    loss_plus = loss_fn(params_plus, inputs)
                    loss_minus = loss_fn(params_minus, inputs)
                    
                    finite_grad = (loss_plus - loss_minus) / (2 * h)
                    grad_real = grad_real.at[jnp.unravel_index(idx, orig_param.shape)].set(finite_grad)
                
                return grad_real + 1j * grad_imag
            else:
                # Real parameters
                grad = jnp.zeros_like(orig_param)
                flat_param = orig_param.flatten()
                
                indices = jnp.linspace(0, len(flat_param)-1, min(10, len(flat_param)), dtype=int)
                
                for idx in indices:
                    param_plus = orig_param.at[jnp.unravel_index(idx, orig_param.shape)].add(h)
                    param_minus = orig_param.at[jnp.unravel_index(idx, orig_param.shape)].add(-h)
                    
                    params_plus = set_param(params, param_path, param_plus)
                    params_minus = set_param(params, param_path, param_minus)
                    
                    loss_plus = loss_fn(params_plus, inputs)
                    loss_minus = loss_fn(params_minus, inputs)
                    
                    finite_grad = (loss_plus - loss_minus) / (2 * h)
                    grad = grad.at[jnp.unravel_index(idx, orig_param.shape)].set(finite_grad)
                
                return grad
        
        # Check gradients for key S5 parameters
        s5_param_paths = [
            ['Lambda_re'],
            ['Lambda_im'], 
            ['B_real'],
            ['C_real'],
            ['D'],
        ]
        
        all_passed = True
        
        for param_path in s5_param_paths:
            if param_path[0] in s5_params:
                analytical_grad = grads[param_path[0]]
                finite_grad = finite_diff_grad(s5_params, param_path)
                
                # Compare gradients
                if jnp.iscomplexobj(analytical_grad):
                    diff_real = jnp.max(jnp.abs(analytical_grad.real - finite_grad.real))
                    diff_imag = jnp.max(jnp.abs(analytical_grad.imag - finite_grad.imag))
                    max_diff = max(diff_real, diff_imag)
                else:
                    max_diff = jnp.max(jnp.abs(analytical_grad - finite_grad))
                
                if max_diff > tolerance:
                    logger.error(f"Gradient check failed for {param_path[0]}: max_diff={max_diff}")
                    all_passed = False
                else:
                    logger.debug(f"✓ Gradient check passed for {param_path[0]}: max_diff={max_diff}")
        
        if all_passed:
            logger.info("✓ S5 gradient validation passed")
        else:
            logger.error("❌ S5 gradient validation failed")
        
        return all_passed
        
    except Exception as e:
        logger.error(f"S5 gradient validation error: {e}")
        return False


def monitor_memory_usage() -> Dict[str, float]:
    """
    Monitor JAX memory usage.
    
    Returns:
        Dictionary with memory statistics
    """
    
    try:
        # Get memory info from JAX devices
        devices = jax.devices()
        memory_stats = {}
        
        for i, device in enumerate(devices):
            try:
                # Get device memory info (if available)
                memory_stats[f'device_{i}_type'] = str(device.device_kind)
                
                # Try to get memory usage (this may not work on all backends)
                if hasattr(device, 'memory_stats'):
                    stats = device.memory_stats()
                    memory_stats[f'device_{i}_bytes_in_use'] = stats.get('bytes_in_use', 0)
                    memory_stats[f'device_{i}_peak_bytes_in_use'] = stats.get('peak_bytes_in_use', 0)
                
            except Exception as e:
                logger.debug(f"Could not get memory stats for device {i}: {e}")
        
        return memory_stats
        
    except Exception as e:
        logger.warning(f"Memory monitoring failed: {e}")
        return {}


# Preset mixed precision configurations
def get_tpu_mixed_precision_config() -> MixedPrecisionConfig:
    """Get mixed precision configuration optimized for TPU."""
    return MixedPrecisionConfig(
        param_dtype=jnp.float32,
        compute_dtype=jnp.bfloat16,  # TPU-optimized
        attention_softmax_dtype=jnp.float32,  # Critical for stability
        s5_complex_dtype=jnp.complex64,      # Critical for S5 stability
        layernorm_dtype=jnp.float32,         # Stability
        loss_dtype=jnp.float32,
        grad_dtype=jnp.float32,
    )


def get_gpu_mixed_precision_config() -> MixedPrecisionConfig:
    """Get mixed precision configuration optimized for GPU."""
    return MixedPrecisionConfig(
        param_dtype=jnp.float32,
        compute_dtype=jnp.float16,           # GPU-optimized
        attention_softmax_dtype=jnp.float32, # Critical for stability
        s5_complex_dtype=jnp.complex64,     # Critical for S5 stability
        layernorm_dtype=jnp.float32,        # Stability
        loss_dtype=jnp.float32,
        grad_dtype=jnp.float32,
    )


def get_debug_mixed_precision_config() -> MixedPrecisionConfig:
    """Get mixed precision configuration for debugging (all fp32)."""
    return MixedPrecisionConfig(
        param_dtype=jnp.float32,
        compute_dtype=jnp.float32,           # All fp32 for debugging
        attention_softmax_dtype=jnp.float32,
        s5_complex_dtype=jnp.complex64,     # Keep complex64 for S5
        layernorm_dtype=jnp.float32,
        loss_dtype=jnp.float32,
        grad_dtype=jnp.float32,
        cast_inputs=False,                   # No casting for debugging
        cast_outputs=False,
    )