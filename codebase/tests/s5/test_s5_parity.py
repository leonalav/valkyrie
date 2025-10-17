"""
S5 Parity Tests - Critical Mathematical Correctness Validation

This module contains tests to validate that the S5 implementation is mathematically correct
by comparing parallel scan vs recurrent computation results.

These tests MUST pass for production use.
"""

import pytest
import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from typing import Tuple

# Import the S5 implementation
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from model.s5 import ValkyrieS5
from model.modules import ValkyrieConfig


class TestS5Parity:
    """Test suite for S5 mathematical correctness."""
    
    @pytest.fixture
    def config(self):
        """Create a test configuration."""
        return ValkyrieConfig(
            d_model=64,
            n_layers=1,
            vocab_size=1000,
            layer_norm_eps=1e-6
        )
    
    @pytest.fixture
    def s5_model(self, config):
        """Create S5 model for testing."""
        return ValkyrieS5(config=config, state_dim=8, init_mode="hippo")
    
    @pytest.fixture
    def test_data(self):
        """Generate deterministic test data."""
        key = jax.random.PRNGKey(42)
        batch_size, seq_len, d_model = 2, 16, 64
        
        # Generate random input sequence
        u = jax.random.normal(key, (batch_size, seq_len, d_model), dtype=jnp.float32)
        return u, batch_size, seq_len, d_model
    
    def test_state_dim_parity_assertion(self, config):
        """Test that odd state_dim raises assertion error."""
        s5_model = ValkyrieS5(config=config, state_dim=7)  # Odd state_dim
        
        # The assertion should be raised during initialization
        key = jax.random.PRNGKey(42)
        u = jnp.ones((1, 4, 64), dtype=jnp.float32)
        
        with pytest.raises(AssertionError, match="state_dim must be even"):
            params = s5_model.init(key, u, training=False)
    
    def test_recurrent_vs_parallel_parity(self, s5_model, test_data, config):
        """
        Critical test: Verify parallel scan produces same results as recurrent stepping.
        
        This is the most important test - if this fails, the implementation is wrong.
        """
        u, batch_size, seq_len, d_model = test_data
        state_dim = s5_model.state_dim
        
        # Initialize model parameters deterministically
        key = jax.random.PRNGKey(123)
        params = s5_model.init(key, u, training=False)
        
        # Get the S5 parameters
        s5_params = params['params']
        
        # Create a bound model for easier parameter access
        bound_model = s5_model.bind(params)
        
        # Get complex parameters
        Lambda, B_tilde, C_tilde = bound_model._get_complex_params()
        Delta = jnp.exp(bound_model.log_Delta)
        
        # Discretize parameters
        Lambda_bar, B_bar = bound_model.discretize(Lambda, B_tilde, Delta)
        
        # Method 1: Parallel scan (full sequence)
        parallel_result, parallel_final_state = bound_model(u, training=False)
        
        # Method 2: Recurrent stepping (step by step)
        x_recurrent = jnp.zeros((batch_size, state_dim), dtype=jnp.complex64)
        recurrent_outputs = []
        
        for t in range(seq_len):
            u_t = u[:, t, :]  # [batch, d_model]
            y_t, x_recurrent = bound_model.step(u_t, x_recurrent, Lambda_bar, B_bar, C_tilde)
            recurrent_outputs.append(y_t)
        
        # Stack recurrent outputs
        recurrent_result = jnp.stack(recurrent_outputs, axis=1)  # [batch, seq_len, d_model]
        
        # Compare results with appropriate tolerance
        tolerance = 1e-5
        
        # Check output sequences match
        max_diff_output = jnp.max(jnp.abs(parallel_result - recurrent_result))
        print(f"Max output difference: {max_diff_output}")
        assert max_diff_output < tolerance, f"Output mismatch: {max_diff_output} >= {tolerance}"
        
        # Check final states match
        max_diff_state = jnp.max(jnp.abs(parallel_final_state - x_recurrent))
        print(f"Max state difference: {max_diff_state}")
        assert max_diff_state < tolerance, f"Final state mismatch: {max_diff_state} >= {tolerance}"
        
        print("✅ Recurrent vs Parallel parity test PASSED")
    
    def test_epsilon_handling_edge_cases(self, s5_model, config):
        """Test epsilon handling with tiny eigenvalues."""
        key = jax.random.PRNGKey(456)
        batch_size, seq_len, d_model = 1, 4, 64
        u = jax.random.normal(key, (batch_size, seq_len, d_model), dtype=jnp.float32)
        
        # Initialize model
        params = s5_model.init(key, u, training=False)
        bound_model = s5_model.bind(params)
        
        # Create a Lambda with very small values to test epsilon handling
        Lambda_tiny = jnp.array([1e-10 + 1e-10j, 1e-9 + 1e-9j, -0.1 + 0.1j, -1.0 + 0.5j], dtype=jnp.complex64)
        B_tilde = jnp.ones((4, d_model), dtype=jnp.complex64) * 0.1
        Delta = jnp.ones(4, dtype=jnp.float32) * 0.1
        
        # This should not raise any errors or produce NaN/Inf
        Lambda_bar, B_bar = bound_model.discretize(Lambda_tiny, B_tilde, Delta)
        
        # Check for NaN or Inf
        assert not jnp.any(jnp.isnan(Lambda_bar)), "Lambda_bar contains NaN"
        assert not jnp.any(jnp.isinf(Lambda_bar)), "Lambda_bar contains Inf"
        assert not jnp.any(jnp.isnan(B_bar)), "B_bar contains NaN"
        assert not jnp.any(jnp.isinf(B_bar)), "B_bar contains Inf"
        
        print("✅ Epsilon handling edge case test PASSED")
    
    def test_hippo_vs_random_initialization(self, config):
        """Compare HiPPO vs random initialization patterns."""
        key = jax.random.PRNGKey(789)
        batch_size, seq_len, d_model = 1, 8, 64
        u = jax.random.normal(key, (batch_size, seq_len, d_model), dtype=jnp.float32)
        
        # Create models with different initialization
        s5_hippo = ValkyrieS5(config=config, state_dim=8, init_mode="hippo")
        s5_random = ValkyrieS5(config=config, state_dim=8, init_mode="random")
        
        # Initialize both models
        params_hippo = s5_hippo.init(key, u, training=False)
        params_random = s5_random.init(jax.random.PRNGKey(790), u, training=False)
        
        # Check that HiPPO initialization follows the expected pattern
        hippo_lambda_re = params_hippo['params']['Lambda_re']
        expected_pattern = -(2 * jnp.arange(4) + 1)  # For state_dim=8, half_state=4
        
        # HiPPO should match the expected pattern exactly
        assert jnp.allclose(hippo_lambda_re, expected_pattern), f"HiPPO pattern mismatch: {hippo_lambda_re} vs {expected_pattern}"
        
        # Random initialization should be different
        random_lambda_re = params_random['params']['Lambda_re']
        assert not jnp.allclose(random_lambda_re, expected_pattern), "Random init should not match HiPPO pattern"
        
        print("✅ HiPPO vs Random initialization test PASSED")
    
    def test_complex_conjugate_symmetry(self, s5_model, test_data):
        """Test that conjugate symmetry is properly maintained."""
        u, batch_size, seq_len, d_model = test_data
        key = jax.random.PRNGKey(999)
        
        # Initialize model
        params = s5_model.init(key, u, training=False)
        bound_model = s5_model.bind(params)
        
        # Get complex parameters
        Lambda, B_tilde, C_tilde = bound_model._get_complex_params()
        
        half_state = s5_model.state_dim // 2
        
        # Check Lambda conjugate symmetry
        Lambda_first_half = Lambda[:half_state]
        Lambda_second_half = Lambda[half_state:]
        expected_second_half = jnp.conj(Lambda_first_half)
        
        assert jnp.allclose(Lambda_second_half, expected_second_half, atol=1e-6), "Lambda conjugate symmetry violated"
        
        # Check B_tilde conjugate symmetry
        B_first_half = B_tilde[:half_state, :]
        B_second_half = B_tilde[half_state:, :]
        expected_B_second_half = jnp.conj(B_first_half)
        
        assert jnp.allclose(B_second_half, expected_B_second_half, atol=1e-6), "B_tilde conjugate symmetry violated"
        
        # Check C_tilde conjugate symmetry
        C_first_half = C_tilde[:, :half_state]
        C_second_half = C_tilde[:, half_state:]
        expected_C_second_half = jnp.conj(C_first_half)
        
        assert jnp.allclose(C_second_half, expected_C_second_half, atol=1e-6), "C_tilde conjugate symmetry violated"
        
        print("✅ Complex conjugate symmetry test PASSED")
    
    def test_output_is_real(self, s5_model, test_data):
        """Test that outputs are real-valued due to conjugate symmetry."""
        u, batch_size, seq_len, d_model = test_data
        key = jax.random.PRNGKey(111)
        
        # Initialize and run model
        params = s5_model.init(key, u, training=False)
        bound_model = s5_model.bind(params)
        
        output, final_state = bound_model(u, training=False)
        
        # Output should be real (float32)
        assert output.dtype == jnp.float32, f"Output should be float32, got {output.dtype}"
        
        # Final state should be complex64
        assert final_state.dtype == jnp.complex64, f"Final state should be complex64, got {final_state.dtype}"
        
        print("✅ Output dtype test PASSED")


def run_all_tests():
    """Run all S5 parity tests."""
    print("Running S5 Parity Tests...")
    print("=" * 50)
    
    # Create test instance
    test_instance = TestS5Parity()
    
    # Create fixtures
    config = ValkyrieConfig(
        d_model=64,
        n_layers=1,
        vocab_size=1000,
        layer_norm_eps=1e-6
    )
    
    s5_model = ValkyrieS5(config=config, state_dim=8, init_mode="hippo")
    
    key = jax.random.PRNGKey(42)
    batch_size, seq_len, d_model = 2, 16, 64
    u = jax.random.normal(key, (batch_size, seq_len, d_model), dtype=jnp.float32)
    test_data = (u, batch_size, seq_len, d_model)
    
    try:
        # Run tests
        test_instance.test_state_dim_parity_assertion(config)
        test_instance.test_recurrent_vs_parallel_parity(s5_model, test_data, config)
        test_instance.test_epsilon_handling_edge_cases(s5_model, config)
        test_instance.test_hippo_vs_random_initialization(config)
        test_instance.test_complex_conjugate_symmetry(s5_model, test_data)
        test_instance.test_output_is_real(s5_model, test_data)
        
        print("=" * 50)
        print("🎉 ALL S5 PARITY TESTS PASSED!")
        print("The S5 implementation is mathematically correct.")
        
    except Exception as e:
        print("=" * 50)
        print(f"❌ TEST FAILED: {e}")
        raise


if __name__ == "__main__":
    run_all_tests()