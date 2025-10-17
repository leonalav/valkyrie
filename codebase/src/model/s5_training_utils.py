"""
S5 Training Utilities for parameter extraction and gradient flow analysis.

This module provides utilities to extract S5 parameters from model state
and analyze gradient flow through complex S5 computations during training.
"""

import jax
import jax.numpy as jnp
from typing import Dict, Any, Optional, Tuple
import flax


def extract_s5_params_from_state(
    params: Dict[str, Any],
    config: Any
) -> Dict[str, jnp.ndarray]:
    """
    Extract S5 parameters from model parameter tree for regularization.
    
    Args:
        params: Full model parameter tree
        config: Model configuration
        
    Returns:
        Dictionary containing S5 parameters (Lambda, B_tilde, C_tilde) if available
    """
    s5_params = {}
    
    if not config.use_s5:
        return s5_params
    
    try:
        # Navigate through the parameter tree to find S5 parameters
        # Structure: params['block_i']['s5_layer']['param_name']
        for i in range(config.n_layers):
            block_key = f'block_{i}'
            if block_key in params:
                block_params = params[block_key]
                
                # Check if this block has S5 layer
                if 's5_layer' in block_params:
                    s5_layer_params = block_params['s5_layer']
                    
                    # Extract S5-specific parameters
                    if 'Lambda_unconstrained_re' in s5_layer_params:
                        # We have the raw parameters, need to reconstruct complex params
                        # This matches the _get_complex_params method in S5
                        
                        # Extract raw parameters
                        Lambda_unconstrained_re = s5_layer_params['Lambda_unconstrained_re']
                        Lambda_im = s5_layer_params.get('Lambda_im', None)
                        B_real = s5_layer_params.get('B_real', None)
                        B_imag = s5_layer_params.get('B_imag', None)
                        C_real = s5_layer_params.get('C_real', None)
                        C_imag = s5_layer_params.get('C_imag', None)
                        
                        if all(param is not None for param in [Lambda_im, B_real, B_imag, C_real, C_imag]):
                            # Reconstruct complex parameters following S5._get_complex_params logic
                            eps = 1e-4
                            Lambda_re_constrained = -jax.nn.softplus(Lambda_unconstrained_re) - eps
                            
                            # Create conjugate pairs
                            Lambda_real = jnp.concatenate([Lambda_re_constrained, Lambda_re_constrained])
                            Lambda_imag = jnp.concatenate([Lambda_im, -Lambda_im])
                            Lambda = Lambda_real + 1j * Lambda_imag
                            
                            B_tilde = jnp.concatenate([
                                B_real + 1j * B_imag,
                                B_real - 1j * B_imag
                            ], axis=0)
                            
                            C_tilde = jnp.concatenate([
                                C_real + 1j * C_imag,
                                C_real - 1j * C_imag
                            ], axis=1)
                            
                            # Store parameters for this layer
                            layer_key = f'layer_{i}'
                            s5_params[f'{layer_key}_Lambda'] = Lambda
                            s5_params[f'{layer_key}_B_tilde'] = B_tilde
                            s5_params[f'{layer_key}_C_tilde'] = C_tilde
                            
                            # Also store aggregated parameters for global regularization
                            if 'Lambda' not in s5_params:
                                s5_params['Lambda'] = Lambda
                                s5_params['B_tilde'] = B_tilde
                                s5_params['C_tilde'] = C_tilde
                            else:
                                # Concatenate with existing parameters
                                s5_params['Lambda'] = jnp.concatenate([s5_params['Lambda'], Lambda])
                                s5_params['B_tilde'] = jnp.concatenate([s5_params['B_tilde'], B_tilde], axis=0)
                                s5_params['C_tilde'] = jnp.concatenate([s5_params['C_tilde'], C_tilde], axis=1)
                            
    except (KeyError, AttributeError) as e:
        # If parameter extraction fails, return empty dict
        # This allows the training to continue without S5-specific regularization
        pass
    
    return s5_params


def analyze_s5_gradient_flow(
    params: Dict[str, Any],
    grads: Dict[str, Any],
    config: Any
) -> Dict[str, float]:
    """
    Analyze gradient flow through S5 parameters to detect gradient issues.
    
    Args:
        params: Model parameters
        grads: Computed gradients
        config: Model configuration
        
    Returns:
        Dictionary with gradient flow statistics
    """
    stats = {
        'total_s5_grad_norm': 0.0,
        'real_grad_norm': 0.0,
        'imag_grad_norm': 0.0,
        'grad_ratio_real_to_imag': 0.0,
        'num_s5_layers': 0
    }
    
    if not config.use_s5:
        return stats
    
    try:
        total_real_grad_norm = 0.0
        total_imag_grad_norm = 0.0
        total_s5_grad_norm = 0.0
        num_layers = 0
        
        for i in range(config.n_layers):
            block_key = f'block_{i}'
            if block_key in grads and 's5_layer' in grads[block_key]:
                s5_grads = grads[block_key]['s5_layer']
                num_layers += 1
                
                # Analyze gradients for real and imaginary parts
                for param_name, grad_value in s5_grads.items():
                    if 'real' in param_name.lower() or 'Lambda_unconstrained_re' in param_name:
                        total_real_grad_norm += jnp.sum(grad_value ** 2)
                    elif 'imag' in param_name.lower() or 'Lambda_im' in param_name:
                        total_imag_grad_norm += jnp.sum(grad_value ** 2)
                    
                    total_s5_grad_norm += jnp.sum(grad_value ** 2)
        
        # Compute final statistics
        stats['total_s5_grad_norm'] = float(jnp.sqrt(total_s5_grad_norm))
        stats['real_grad_norm'] = float(jnp.sqrt(total_real_grad_norm))
        stats['imag_grad_norm'] = float(jnp.sqrt(total_imag_grad_norm))
        stats['num_s5_layers'] = num_layers
        
        # Compute ratio (avoid division by zero)
        if total_imag_grad_norm > 1e-10:
            stats['grad_ratio_real_to_imag'] = float(total_real_grad_norm / total_imag_grad_norm)
        else:
            stats['grad_ratio_real_to_imag'] = float('inf') if total_real_grad_norm > 1e-10 else 1.0
            
    except Exception as e:
        # If analysis fails, return default stats
        pass
    
    return stats


def check_conjugate_symmetry_violation(
    s5_params: Dict[str, jnp.ndarray],
    tolerance: float = 1e-6
) -> Dict[str, float]:
    """
    Check how well S5 parameters maintain conjugate symmetry.
    
    Args:
        s5_params: S5 parameters extracted from model
        tolerance: Tolerance for symmetry violation
        
    Returns:
        Dictionary with symmetry violation statistics
    """
    violations = {
        'lambda_violation': 0.0,
        'b_violation': 0.0,
        'c_violation': 0.0,
        'max_violation': 0.0,
        'is_symmetric': True
    }
    
    try:
        if 'Lambda' in s5_params:
            Lambda = s5_params['Lambda']
            state_dim = Lambda.shape[0]
            
            if state_dim % 2 == 0:
                # Check Lambda conjugate symmetry
                Lambda_pairs = Lambda.reshape(-1, 2)
                lambda_violation = jnp.max(jnp.abs(Lambda_pairs[:, 0] - jnp.conj(Lambda_pairs[:, 1])))
                violations['lambda_violation'] = float(lambda_violation)
                
        if 'B_tilde' in s5_params:
            B_tilde = s5_params['B_tilde']
            state_dim = B_tilde.shape[0]
            
            if state_dim % 2 == 0:
                # Check B_tilde conjugate symmetry
                B_pairs = B_tilde.reshape(-1, 2, B_tilde.shape[-1])
                b_violation = jnp.max(jnp.abs(B_pairs[:, 0, :] - jnp.conj(B_pairs[:, 1, :])))
                violations['b_violation'] = float(b_violation)
                
        if 'C_tilde' in s5_params:
            C_tilde = s5_params['C_tilde']
            state_dim = C_tilde.shape[1]
            
            if state_dim % 2 == 0:
                # Check C_tilde conjugate symmetry
                C_pairs = C_tilde.reshape(C_tilde.shape[0], -1, 2)
                c_violation = jnp.max(jnp.abs(C_pairs[:, :, 0] - jnp.conj(C_pairs[:, :, 1])))
                violations['c_violation'] = float(c_violation)
        
        # Compute overall statistics
        max_violation = max(violations['lambda_violation'], violations['b_violation'], violations['c_violation'])
        violations['max_violation'] = max_violation
        violations['is_symmetric'] = max_violation < tolerance
        
    except Exception as e:
        # If check fails, assume not symmetric
        violations['is_symmetric'] = False
        
    return violations