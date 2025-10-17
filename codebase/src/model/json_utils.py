"""JSON serialization utilities for S5 model diagnostics and metrics.

This module provides safe JSON serialization for complex numbers, JAX scalars,
and nested data structures commonly used in S5 model analysis.
"""

import jax.numpy as jnp
import numpy as np
from typing import Any, Dict, List, Union


def make_json_safe(obj: Any) -> Any:
    """
    Convert complex numbers, JAX scalars, and nested structures to JSON-safe formats.
    
    This function recursively processes data structures to ensure they can be
    serialized to JSON without errors. Complex numbers are converted to dictionaries
    with 'real', 'imag', and 'abs' fields. JAX scalars are converted to Python
    native types using .item().
    
    Args:
        obj: Object to make JSON-safe (dict, list, tuple, complex, JAX scalar, etc.)
        
    Returns:
        JSON-safe version of the input object
        
    Examples:
        >>> make_json_safe({'eigenval': 1+2j, 'magnitude': jnp.array(3.14)})
        {'eigenval': {'real': 1.0, 'imag': 2.0, 'abs': 2.23606797749979}, 'magnitude': 3.14}
        
        >>> make_json_safe([1+1j, jnp.array(2.0), 'string'])
        [{'real': 1.0, 'imag': 1.0, 'abs': 1.4142135623730951}, 2.0, 'string']
    """
    if isinstance(obj, dict):
        return {k: make_json_safe(v) for k, v in obj.items()}
    
    if isinstance(obj, (list, tuple)):
        return [make_json_safe(v) for v in obj]
    
    # Handle JAX arrays and scalars
    if hasattr(obj, 'item'):
        try:
            v = obj.item()
            if isinstance(v, complex):
                return {'real': float(v.real), 'imag': float(v.imag), 'abs': float(abs(v))}
            return float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v
        except (ValueError, TypeError):
            # Fallback for arrays that can't be converted to scalar
            if hasattr(obj, 'tolist'):
                return make_json_safe(obj.tolist())
            return str(obj)
    
    # Handle Python complex numbers
    if isinstance(obj, complex):
        return {'real': float(obj.real), 'imag': float(obj.imag), 'abs': float(abs(obj))}
    
    # Handle NumPy arrays
    if isinstance(obj, np.ndarray):
        if obj.size == 1:
            # Single element array
            item = obj.item()
            if isinstance(item, complex):
                return {'real': float(item.real), 'imag': float(item.imag), 'abs': float(abs(item))}
            return float(item) if isinstance(item, (int, float, np.floating, np.integer)) else item
        else:
            # Multi-element array - convert to list
            return make_json_safe(obj.tolist())
    
    # Handle NumPy scalars
    if isinstance(obj, (np.floating, np.integer, np.complexfloating)):
        if isinstance(obj, np.complexfloating):
            return {'real': float(obj.real), 'imag': float(obj.imag), 'abs': float(abs(obj))}
        return float(obj)
    
    # Handle JAX DeviceArray/Array types by converting to numpy first
    if hasattr(obj, '__array__'):
        try:
            np_obj = np.asarray(obj)
            return make_json_safe(np_obj)
        except (ValueError, TypeError):
            return str(obj)
    
    # Return primitive types as-is
    if isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    
    # Fallback: convert to string representation
    return str(obj)


def safe_json_dump(obj: Any, file_path: str, indent: int = 2) -> None:
    """
    Safely dump an object to JSON file with proper complex number handling.
    
    Args:
        obj: Object to serialize
        file_path: Path to output JSON file
        indent: JSON indentation level
        
    Raises:
        IOError: If file cannot be written
        
    Example:
        >>> metrics = {'spectral_radius': jnp.array(0.99), 'eigenvals': [1+2j, 3-1j]}
        >>> safe_json_dump(metrics, 'results.json')
    """
    import json
    
    safe_obj = make_json_safe(obj)
    
    with open(file_path, 'w') as f:
        json.dump(safe_obj, f, indent=indent)


def safe_json_dumps(obj: Any, indent: int = 2) -> str:
    """
    Safely serialize an object to JSON string with proper complex number handling.
    
    Args:
        obj: Object to serialize
        indent: JSON indentation level
        
    Returns:
        JSON string representation
        
    Example:
        >>> metrics = {'spectral_radius': jnp.array(0.99), 'eigenvals': [1+2j]}
        >>> json_str = safe_json_dumps(metrics)
    """
    import json
    
    safe_obj = make_json_safe(obj)
    return json.dumps(safe_obj, indent=indent)