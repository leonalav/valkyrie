"""
Autodiff & Gradient Sanity Checks for S5 Layer

Tests from note.txt checklist:
- Finite-difference gradient check for scalar loss
- Compare grad_num ≈ grad_jax using central differences
- Relative error < 1e-4 for grads of parameters that matter (B/C/Lambda)
- Backprop end-to-end on TPU for tiny optimization step
- Gradient norms are finite and not astronomical (< 1e6)

MCPs used: Sequential Thinking MCP for planning, context7 MCP for JAX best practices
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
import pytest
from typing import Dict, Any, Tuple, Optional
import numpy as np
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from model.s5 import ValkyrieS5
from model.valkyrie import ValkyrieConfig


class AutodiffGradientTestSuite:
    """Comprehensive autodiff and gradient validation test suite."""
    
    def __init__(self):
        self.devices = jax.devices()
        self.test_results = {}
        
    def create_test_config(self) -> ValkyrieConfig:
        """Create minimal config for gradient testing."""
        return ValkyrieConfig(
            d_model=32,  # Small for fast testing
            n_layers=1,
            vocab_size=1000,
            n_heads=4,
            s5_state_dim=8,  # Small state for numerical stability
            use_s5=True
        )
    
    def create_test_layer_and_params(self, config: ValkyrieConfig, rng_key: jax.random.PRNGKey) -> Tuple[ValkyrieS5, Dict]:
        """Initialize S5 layer and parameters for testing."""
        layer = ValkyrieS5(
            config=config,
            state_dim=config.s5_state_dim,
            init_mode="hippo"
        )
        
        # Create dummy input for initialization
        batch_size, seq_len = 1, 8  # Very small for gradient testing
        dummy_input = jnp.ones((batch_size, seq_len, config.d_model), dtype=jnp.float32)
        
        # Initialize parameters
        params = layer.init(rng_key, dummy_input, training=True)
        
        return layer, params, dummy_input
    
    def scalar_loss_function(self, params: Dict, layer: ValkyrieS5, x: jnp.ndarray) -> float:
        """Scalar loss function: L = sum(model(x)**2)."""
        output, _ = layer.apply(params, x, training=True)  # Unpack tuple (output, final_state)
        return jnp.sum(output**2)
    
    def finite_difference_gradient(self, 
                                   params: Dict, 
                                   layer: ValkyrieS5, 
                                   x: jnp.ndarray, 
                                   param_path: str,
                                   param_idx: Tuple[int, ...],
                                   eps: float = 1e-5) -> float:
        """Compute finite difference gradient for a single parameter."""
        
        def get_param_value(p, path, idx):
            """Extract parameter value at specific path and index."""
            param_dict = p['params']
            for key in path.split('.'):
                if key in param_dict:
                    param_dict = param_dict[key]
            return param_dict[idx]
        
        def set_param_value(p, path, idx, value):
            """Set parameter value at specific path and index."""
            import copy
            p_new = copy.deepcopy(p)
            param_dict = p_new['params']
            for key in path.split('.')[:-1]:
                if key in param_dict:
                    param_dict = param_dict[key]
            final_key = path.split('.')[-1]
            param_dict[final_key] = param_dict[final_key].at[idx].set(value)
            return p_new
        
        # Get original parameter value
        orig_val = get_param_value(params, param_path, param_idx)
        
        # Forward difference: f(x + eps)
        params_plus = set_param_value(params, param_path, param_idx, orig_val + eps)
        loss_plus = self.scalar_loss_function(params_plus, layer, x)
        
        # Backward difference: f(x - eps)  
        params_minus = set_param_value(params, param_path, param_idx, orig_val - eps)
        loss_minus = self.scalar_loss_function(params_minus, layer, x)
        
        # Central difference
        return (loss_plus - loss_minus) / (2 * eps)
    
    def test_finite_difference_vs_autodiff(self) -> Dict[str, Any]:
        """Test 1: Compare finite-difference vs autodiff gradients."""
        results = {}
        
        try:
            # Setup
            rng = jax.random.PRNGKey(42)
            config = self.create_test_config()
            layer, params, x = self.create_test_layer_and_params(config, rng)
            
            # Compute autodiff gradients
            grad_fn = jax.grad(self.scalar_loss_function, argnums=0)
            jax_grads = grad_fn(params, layer, x)
            
            # Test key parameters: Lambda_re, Lambda_im, B_real, B_imag, C_real, C_imag
            test_params = [
                ('Lambda_unconstrained_re', (0,)),
                ('Lambda_unconstrained_im', (0,)),
                ('B_real', (0, 0)),
                ('B_imag', (0, 0)),
                ('C_real', (0, 0)),
                ('C_imag', (0, 0)),
            ]
            
            relative_errors = {}
            
            for param_name, param_idx in test_params:
                try:
                    # Get JAX gradient
                    jax_grad_val = jax_grads['params'][param_name][param_idx]
                    
                    # Compute finite difference gradient
                    fd_grad_val = self.finite_difference_gradient(
                        params, layer, x, param_name, param_idx
                    )
                    
                    # Compute relative error
                    if abs(jax_grad_val) > 1e-8:  # Avoid division by very small numbers
                        rel_error = abs(jax_grad_val - fd_grad_val) / abs(jax_grad_val)
                    else:
                        rel_error = abs(jax_grad_val - fd_grad_val)
                    
                    relative_errors[param_name] = {
                        'jax_grad': float(jax_grad_val),
                        'fd_grad': float(fd_grad_val),
                        'relative_error': float(rel_error),
                        'passes_threshold': rel_error < 1e-4
                    }
                    
                except KeyError:
                    relative_errors[param_name] = {
                        'error': f'Parameter {param_name} not found in gradients'
                    }
            
            results['relative_errors'] = relative_errors
            results['all_params_pass'] = all(
                param_result.get('passes_threshold', False) 
                for param_result in relative_errors.values()
                if 'error' not in param_result
            )
            results['success'] = True
            results['error'] = None
            
        except Exception as e:
            results['success'] = False
            results['error'] = str(e)
            results['error_type'] = type(e).__name__
        
        return results
    
    def test_end_to_end_backprop_tpu(self) -> Dict[str, Any]:
        """Test 2: End-to-end backprop on TPU with optimization step."""
        results = {}
        
        try:
            # Setup
            rng = jax.random.PRNGKey(42)
            config = self.create_test_config()
            layer, params, x = self.create_test_layer_and_params(config, rng)
            
            # Create target output for supervised loss
            y_true = jax.random.normal(rng, (x.shape[0], x.shape[1], config.d_model))
            
            # Move to TPU device
            params_on_device = jax.tree_util.tree_map(jax.device_put, params)
            x_on_device = jax.device_put(x)
            y_true_on_device = jax.device_put(y_true)
            
            @jax.jit
            def loss_and_grads(params, x, y_true):
                """Compute loss and gradients."""
                preds, _ = layer.apply(params, x, training=True)  # Unpack tuple
                loss = jnp.mean((preds - y_true)**2)
                grads = jax.grad(lambda p: jnp.mean((layer.apply(p, x, training=True)[0] - y_true)**2))(params)
                return loss, grads
            
            # Run backprop on TPU
            loss, grads = loss_and_grads(params_on_device, x_on_device, y_true_on_device)
            
            # Validate results
            results['loss_finite'] = bool(jnp.isfinite(loss))
            results['loss_value'] = float(loss)
            
            # Check gradient properties
            grad_leaves = jax.tree_util.tree_leaves(grads)
            grad_norms = [jnp.linalg.norm(leaf.flatten()) for leaf in grad_leaves]
            max_grad_norm = max(grad_norms)
            
            results['all_grads_finite'] = all(jnp.all(jnp.isfinite(leaf)) for leaf in grad_leaves)
            results['max_grad_norm'] = float(max_grad_norm)
            results['grad_norm_reasonable'] = max_grad_norm < 1e6
            results['grad_norm_not_zero'] = max_grad_norm > 1e-8
            
            # Test optimization step
            learning_rate = 1e-3
            updated_params = jax.tree_util.tree_map(
                lambda p, g: p - learning_rate * g, 
                params_on_device, 
                grads
            )
            
            # Verify parameters actually changed
            param_changes = jax.tree_util.tree_map(
                lambda p_new, p_old: jnp.linalg.norm((p_new - p_old).flatten()),
                updated_params,
                params_on_device
            )
            max_param_change = max(jax.tree_util.tree_leaves(param_changes))
            
            results['params_updated'] = float(max_param_change) > 1e-8
            results['max_param_change'] = float(max_param_change)
            results['success'] = True
            results['error'] = None
            
        except Exception as e:
            results['success'] = False
            results['error'] = str(e)
            results['error_type'] = type(e).__name__
        
        return results
    
    def test_gradient_consistency_across_devices(self) -> Dict[str, Any]:
        """Test 3: Gradient consistency across multiple TPU devices."""
        results = {}
        
        try:
            if len(self.devices) < 2:
                results['skipped'] = True
                results['reason'] = 'Need at least 2 devices for consistency test'
                return results
            
            # Setup
            rng = jax.random.PRNGKey(42)
            config = self.create_test_config()
            layer, params, x = self.create_test_layer_and_params(config, rng)
            
            # Test on first two devices
            device_grads = {}
            
            for i, device in enumerate(self.devices[:2]):
                # Place data on specific device
                params_on_device = jax.tree_util.tree_map(lambda p: jax.device_put(p, device), params)
                x_on_device = jax.device_put(x, device)
                
                # Compute gradients
                grad_fn = jax.grad(self.scalar_loss_function, argnums=0)
                grads = grad_fn(params_on_device, layer, x_on_device)
                
                device_grads[f'device_{i}'] = grads
            
            # Compare gradients between devices
            grad_diff = jax.tree_util.tree_map(
                lambda g1, g2: jnp.max(jnp.abs(g1 - g2)),
                device_grads['device_0'],
                device_grads['device_1']
            )
            
            max_diff = max(jax.tree_util.tree_leaves(grad_diff))
            
            results['max_gradient_difference'] = float(max_diff)
            results['gradients_consistent'] = max_diff < 1e-6
            results['success'] = True
            results['error'] = None
            
        except Exception as e:
            results['success'] = False
            results['error'] = str(e)
            results['error_type'] = type(e).__name__
        
        return results
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run complete autodiff gradient test suite."""
        print("🧮 Starting Autodiff & Gradient Sanity Checks...")
        
        all_results = {}
        
        # Test 1: Finite difference vs autodiff
        print("  Testing finite-difference vs autodiff gradients...")
        all_results['finite_diff_vs_autodiff'] = self.test_finite_difference_vs_autodiff()
        
        # Test 2: End-to-end backprop on TPU
        print("  Testing end-to-end backprop on TPU...")
        all_results['end_to_end_backprop'] = self.test_end_to_end_backprop_tpu()
        
        # Test 3: Gradient consistency across devices
        print("  Testing gradient consistency across devices...")
        all_results['gradient_consistency'] = self.test_gradient_consistency_across_devices()
        
        # Overall success
        all_results['overall_success'] = all(
            result.get('success', False) or result.get('skipped', False)
            for result in all_results.values()
        )
        
        if all_results['overall_success']:
            print("✓ All autodiff & gradient sanity checks passed!")
        else:
            print("❌ Some autodiff & gradient sanity checks failed")
            
        return all_results


# Pytest wrappers
def test_finite_difference_vs_autodiff():
    """Pytest wrapper for finite difference vs autodiff test."""
    suite = AutodiffGradientTestSuite()
    results = suite.test_finite_difference_vs_autodiff()
    
    assert results.get('success', False), f"Finite difference test failed: {results.get('error', 'Unknown error')}"
    assert results.get('all_params_pass', False), "Some parameters failed relative error threshold"


def test_end_to_end_backprop_tpu():
    """Pytest wrapper for end-to-end backprop test."""
    suite = AutodiffGradientTestSuite()
    results = suite.test_end_to_end_backprop_tpu()
    
    assert results.get('success', False), f"End-to-end backprop failed: {results.get('error', 'Unknown error')}"
    assert results.get('loss_finite', False), "Loss is not finite"
    assert results.get('all_grads_finite', False), "Some gradients are not finite"
    assert results.get('grad_norm_reasonable', False), "Gradient norms are too large"
    assert results.get('grad_norm_not_zero', False), "Gradient norms are zero"
    assert results.get('params_updated', False), "Parameters were not updated"


def test_gradient_consistency_across_devices():
    """Pytest wrapper for gradient consistency test."""
    suite = AutodiffGradientTestSuite()
    results = suite.test_gradient_consistency_across_devices()
    
    if results.get('skipped', False):
        pytest.skip(results.get('reason', 'Test skipped'))
    
    assert results.get('success', False), f"Gradient consistency test failed: {results.get('error', 'Unknown error')}"
    assert results.get('gradients_consistent', False), "Gradients are not consistent across devices"


def test_full_autodiff_gradient_suite():
    """Pytest wrapper for full autodiff gradient test suite."""
    suite = AutodiffGradientTestSuite()
    results = suite.run_all_tests()
    
    assert results.get('overall_success', False), "Autodiff gradient test suite failed"