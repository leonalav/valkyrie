"""S5 unit tests for parallel vs sequential equality.

Critical validation tests from output.txt:
- S5 parallel scan vs sequential recurrence equality
- Complex arithmetic and dtype handling
- Gradient flow validation
- Numerical stability checks
"""

import jax
import jax.numpy as jnp
import pytest
import numpy as np
from typing import Tuple, Dict, Any
import logging

from ...model import ValkyrieS5, ValkyrieConfig
from ...utils.debug import check_for_nans, debug_shapes, validate_s5_gradients

logger = logging.getLogger(__name__)


class TestS5Implementation:
    """Test suite for S5 state space model implementation."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return ValkyrieConfig(
            d_model=256,
            s5_state_dim=64,
            use_s5=True,
            n_layers=2,
        )
    
    @pytest.fixture
    def s5_layer(self, config):
        """Create S5 layer for testing."""
        return ValkyrieS5(config=config, state_dim=config.s5_state_dim)
    
    @pytest.fixture
    def test_data(self):
        """Create test data."""
        batch_size = 2
        seq_len = 8  # Small for testing
        d_model = 256
        
        key = jax.random.PRNGKey(42)
        u = jax.random.normal(key, (batch_size, seq_len, d_model))
        
        return u
    
    def test_s5_initialization(self, s5_layer, config):
        """Test S5 layer initialization."""
        
        # Initialize parameters
        key = jax.random.PRNGKey(0)
        batch_size, seq_len, d_model = 1, 4, config.d_model
        dummy_input = jax.random.normal(key, (batch_size, seq_len, d_model))
        
        params = s5_layer.init(key, dummy_input, training=True)
        
        # Check parameter structure
        assert 'params' in params
        s5_params = params['params']
        
        # Check required parameters exist
        required_params = ['Lambda_re', 'Lambda_im', 'B_real', 'B_imag', 'C_real', 'C_imag', 'D', 'log_Delta']
        for param_name in required_params:
            assert param_name in s5_params, f"Missing parameter: {param_name}"
        
        # Check parameter shapes
        half_state = config.s5_state_dim // 2
        
        assert s5_params['Lambda_re'].shape == (half_state,)
        assert s5_params['Lambda_im'].shape == (half_state,)
        assert s5_params['B_real'].shape == (half_state, d_model)
        assert s5_params['B_imag'].shape == (half_state, d_model)
        assert s5_params['C_real'].shape == (d_model, half_state)
        assert s5_params['C_imag'].shape == (d_model, half_state)
        assert s5_params['D'].shape == (d_model,)
        assert s5_params['log_Delta'].shape == (config.s5_state_dim,)
        
        logger.info("✓ S5 initialization test passed")
    
    def test_complex_parameter_construction(self, s5_layer, config):
        """Test complex parameter construction and conjugate symmetry."""
        
        # Initialize layer
        key = jax.random.PRNGKey(0)
        batch_size, seq_len, d_model = 1, 4, config.d_model
        dummy_input = jax.random.normal(key, (batch_size, seq_len, d_model))
        
        # Get initialized parameters
        variables = s5_layer.init(key, dummy_input, training=True)
        
        # Apply layer to get complex parameters
        def get_complex_params(variables):
            # Create a bound method to access _get_complex_params
            bound_layer = s5_layer.bind(variables)
            return bound_layer._get_complex_params()
        
        Lambda, B_tilde, C_tilde = get_complex_params(variables)
        
        # Check shapes
        assert Lambda.shape == (config.s5_state_dim,)
        assert B_tilde.shape == (config.s5_state_dim, d_model)
        assert C_tilde.shape == (d_model, config.s5_state_dim)
        
        # Check dtypes
        assert Lambda.dtype == jnp.complex64
        assert B_tilde.dtype == jnp.complex64
        assert C_tilde.dtype == jnp.complex64
        
        # Check conjugate symmetry
        half_state = config.s5_state_dim // 2
        
        # Lambda should have conjugate pairs
        Lambda_first_half = Lambda[:half_state]
        Lambda_second_half = Lambda[half_state:]
        
        # Check that second half is conjugate of first half
        conjugate_diff = jnp.max(jnp.abs(Lambda_second_half - jnp.conj(Lambda_first_half)))
        assert conjugate_diff < 1e-6, f"Lambda conjugate symmetry violated: {conjugate_diff}"
        
        # Similar checks for B_tilde and C_tilde
        B_first_half = B_tilde[:half_state, :]
        B_second_half = B_tilde[half_state:, :]
        B_conjugate_diff = jnp.max(jnp.abs(B_second_half - jnp.conj(B_first_half)))
        assert B_conjugate_diff < 1e-6, f"B_tilde conjugate symmetry violated: {B_conjugate_diff}"
        
        logger.info("✓ Complex parameter construction test passed")
    
    def test_parallel_vs_sequential_equality(self, s5_layer, config, test_data):
        """
        Critical test: S5 parallel scan vs sequential recurrence equality.
        
        This is the most important test from output.txt - must pass for correctness.
        """
        
        logger.info("Running S5 parallel vs sequential equality test...")
        
        # Initialize layer
        key = jax.random.PRNGKey(42)
        variables = s5_layer.init(key, test_data, training=True)
        
        # Get complex parameters
        def get_complex_params(variables):
            bound_layer = s5_layer.bind(variables)
            return bound_layer._get_complex_params()
        
        Lambda, B_tilde, C_tilde = get_complex_params(variables)
        
        # Get other parameters
        params = variables['params']
        Delta = jnp.exp(params['log_Delta'])
        D = params['D']
        
        # Discretize parameters
        bound_layer = s5_layer.bind(variables)
        Lambda_bar, B_bar = bound_layer.discretize(Lambda, B_tilde, Delta)
        
        batch_size, seq_len, d_model = test_data.shape
        
        # 1. Parallel scan implementation (from model)
        parallel_output = bound_layer.parallel_scan(Lambda_bar, B_bar, test_data)
        
        # 2. Sequential recurrence implementation
        def sequential_scan(Lambda_bar, B_bar, u):
            """Sequential implementation for comparison."""
            batch_size, seq_len, d_model = u.shape
            state_dim = Lambda_bar.shape[0]
            
            # Initialize state
            x = jnp.zeros((batch_size, state_dim), dtype=jnp.complex64)
            xs = []
            
            for t in range(seq_len):
                # u_t: [batch, d_model]
                u_t = u[:, t, :]
                
                # Bu_t = B_bar @ u_t: [batch, state_dim]
                Bu_t = jnp.einsum('sd,bd->bs', B_bar, u_t)
                
                # x_t = Lambda_bar * x_{t-1} + Bu_t
                x = Lambda_bar[None, :] * x + Bu_t
                
                xs.append(x)
            
            # Stack results
            xs = jnp.stack(xs, axis=1)  # [batch, seq_len, state_dim]
            return xs
        
        sequential_output = sequential_scan(Lambda_bar, B_bar, test_data)
        
        # 3. Compare outputs
        max_diff = jnp.max(jnp.abs(parallel_output - sequential_output))
        relative_error = max_diff / (jnp.max(jnp.abs(sequential_output)) + 1e-8)
        
        logger.info(f"Parallel vs Sequential comparison:")
        logger.info(f"  Max absolute difference: {max_diff}")
        logger.info(f"  Relative error: {relative_error}")
        logger.info(f"  Parallel output shape: {parallel_output.shape}")
        logger.info(f"  Sequential output shape: {sequential_output.shape}")
        
        # Tolerance check
        tolerance = 1e-5
        assert max_diff < tolerance, f"Parallel vs sequential mismatch: {max_diff} > {tolerance}"
        assert relative_error < tolerance, f"Relative error too high: {relative_error} > {tolerance}"
        
        logger.info("✓ S5 parallel vs sequential equality test PASSED")
        
        return True
    
    def test_s5_forward_pass(self, s5_layer, config, test_data):
        """Test S5 forward pass and output shapes."""
        
        # Initialize layer
        key = jax.random.PRNGKey(0)
        variables = s5_layer.init(key, test_data, training=True)
        
        # Forward pass
        output, final_state = s5_layer.apply(variables, test_data, training=True)
        
        # Check output shapes
        batch_size, seq_len, d_model = test_data.shape
        
        assert output.shape == (batch_size, seq_len, d_model)
        assert final_state.shape == (batch_size, config.s5_state_dim)
        
        # Check dtypes
        assert output.dtype == jnp.float32  # Should be real-valued output
        assert final_state.dtype == jnp.complex64  # S5 state is complex
        
        # Check for NaNs/Infs
        assert not check_for_nans(output, "S5 output")
        assert not check_for_nans(final_state, "S5 final_state")
        
        logger.info("✓ S5 forward pass test passed")
    
    def test_s5_recurrent_mode(self, s5_layer, config):
        """Test S5 recurrent mode for generation."""
        
        # Initialize layer
        key = jax.random.PRNGKey(0)
        batch_size, d_model = 2, config.d_model
        
        # Single token input for recurrent mode
        single_token = jax.random.normal(key, (batch_size, 1, d_model))
        variables = s5_layer.init(key, single_token, training=False)
        
        # Initialize state
        initial_state = jnp.zeros((batch_size, config.s5_state_dim), dtype=jnp.complex64)
        
        # Recurrent forward pass
        output, next_state = s5_layer.apply(
            variables, single_token, training=False, state=initial_state
        )
        
        # Check shapes
        assert output.shape == (batch_size, 1, d_model)
        assert next_state.shape == (batch_size, config.s5_state_dim)
        
        # Check dtypes
        assert output.dtype == jnp.float32
        assert next_state.dtype == jnp.complex64
        
        # Check for NaNs
        assert not check_for_nans(output, "S5 recurrent output")
        assert not check_for_nans(next_state, "S5 next_state")
        
        logger.info("✓ S5 recurrent mode test passed")
    
    def test_s5_gradient_flow(self, s5_layer, config, test_data):
        """Test gradient flow through S5 layer."""
        
        # Initialize layer
        key = jax.random.PRNGKey(0)
        variables = s5_layer.init(key, test_data, training=True)
        
        # Define simple loss function
        def loss_fn(variables, inputs):
            output, _ = s5_layer.apply(variables, inputs, training=True)
            return jnp.mean(output**2)
        
        # Compute gradients
        loss, grads = jax.value_and_grad(loss_fn)(variables, test_data)
        
        # Check that gradients exist for all parameters
        assert 'params' in grads
        s5_grads = grads['params']
        
        required_grads = ['Lambda_re', 'Lambda_im', 'B_real', 'B_imag', 'C_real', 'C_imag', 'D', 'log_Delta']
        for grad_name in required_grads:
            assert grad_name in s5_grads, f"Missing gradient: {grad_name}"
            assert not check_for_nans(s5_grads[grad_name], f"S5 gradient {grad_name}")
        
        # Check gradient magnitudes are reasonable
        for grad_name, grad_value in s5_grads.items():
            grad_norm = jnp.linalg.norm(grad_value.flatten())
            assert grad_norm > 1e-8, f"Gradient too small for {grad_name}: {grad_norm}"
            assert grad_norm < 1e3, f"Gradient too large for {grad_name}: {grad_norm}"
        
        logger.info("✓ S5 gradient flow test passed")
    
    def test_s5_discretization(self, s5_layer, config):
        """Test S5 discretization step."""
        
        # Initialize layer
        key = jax.random.PRNGKey(0)
        batch_size, seq_len, d_model = 1, 4, config.d_model
        dummy_input = jax.random.normal(key, (batch_size, seq_len, d_model))
        
        variables = s5_layer.init(key, dummy_input, training=True)
        bound_layer = s5_layer.bind(variables)
        
        # Get complex parameters
        Lambda, B_tilde, C_tilde = bound_layer._get_complex_params()
        
        # Get Delta
        params = variables['params']
        Delta = jnp.exp(params['log_Delta'])
        
        # Test discretization
        Lambda_bar, B_bar = bound_layer.discretize(Lambda, B_tilde, Delta)
        
        # Check shapes
        assert Lambda_bar.shape == Lambda.shape
        assert B_bar.shape == B_tilde.shape
        
        # Check dtypes
        assert Lambda_bar.dtype == jnp.complex64
        assert B_bar.dtype == jnp.complex64
        
        # Check for NaNs/Infs
        assert not check_for_nans(Lambda_bar, "Lambda_bar")
        assert not check_for_nans(B_bar, "B_bar")
        
        # Check discretization properties
        # |Lambda_bar| should be <= 1 for stability (approximately)
        Lambda_bar_abs = jnp.abs(Lambda_bar)
        max_eigenvalue = jnp.max(Lambda_bar_abs)
        
        # Log eigenvalue statistics
        logger.info(f"Discretized eigenvalue statistics:")
        logger.info(f"  Max |Lambda_bar|: {max_eigenvalue}")
        logger.info(f"  Mean |Lambda_bar|: {jnp.mean(Lambda_bar_abs)}")
        logger.info(f"  Min |Lambda_bar|: {jnp.min(Lambda_bar_abs)}")
        
        # Stability check (eigenvalues shouldn't be too large)
        assert max_eigenvalue < 10.0, f"Eigenvalues too large: {max_eigenvalue}"
        
        logger.info("✓ S5 discretization test passed")
    
    def test_s5_binary_operator(self, s5_layer, config):
        """Test S5 binary operator for associative scan."""
        
        state_dim = config.s5_state_dim
        batch_size = 2
        
        # Create test elements
        key = jax.random.PRNGKey(0)
        key1, key2 = jax.random.split(key)
        
        A_i = jax.random.normal(key1, (batch_size, state_dim), dtype=jnp.complex64)
        Bu_i = jax.random.normal(key1, (batch_size, state_dim), dtype=jnp.complex64)
        
        A_j = jax.random.normal(key2, (batch_size, state_dim), dtype=jnp.complex64)
        Bu_j = jax.random.normal(key2, (batch_size, state_dim), dtype=jnp.complex64)
        
        q_i = (A_i, Bu_i)
        q_j = (A_j, Bu_j)
        
        # Initialize layer to get binary operator
        dummy_input = jax.random.normal(key, (1, 4, config.d_model))
        variables = s5_layer.init(key, dummy_input, training=True)
        bound_layer = s5_layer.bind(variables)
        
        # Test binary operator
        result = bound_layer.binary_operator(q_i, q_j)
        A_combined, Bu_combined = result
        
        # Check shapes
        assert A_combined.shape == A_i.shape
        assert Bu_combined.shape == Bu_i.shape
        
        # Check dtypes
        assert A_combined.dtype == jnp.complex64
        assert Bu_combined.dtype == jnp.complex64
        
        # Check associativity property: (q_i ∙ q_j) ∙ q_k = q_i ∙ (q_j ∙ q_k)
        A_k = jax.random.normal(key, (batch_size, state_dim), dtype=jnp.complex64)
        Bu_k = jax.random.normal(key, (batch_size, state_dim), dtype=jnp.complex64)
        q_k = (A_k, Bu_k)
        
        # Left association: (q_i ∙ q_j) ∙ q_k
        q_ij = bound_layer.binary_operator(q_i, q_j)
        q_ij_k = bound_layer.binary_operator(q_ij, q_k)
        
        # Right association: q_i ∙ (q_j ∙ q_k)
        q_jk = bound_layer.binary_operator(q_j, q_k)
        q_i_jk = bound_layer.binary_operator(q_i, q_jk)
        
        # Compare results
        A_diff = jnp.max(jnp.abs(q_ij_k[0] - q_i_jk[0]))
        Bu_diff = jnp.max(jnp.abs(q_ij_k[1] - q_i_jk[1]))
        
        tolerance = 1e-5
        assert A_diff < tolerance, f"Binary operator not associative (A): {A_diff}"
        assert Bu_diff < tolerance, f"Binary operator not associative (Bu): {Bu_diff}"
        
        logger.info("✓ S5 binary operator test passed")
    
    def test_s5_numerical_stability(self, s5_layer, config):
        """Test S5 numerical stability with edge cases."""
        
        # Test with various input magnitudes
        key = jax.random.PRNGKey(0)
        batch_size, seq_len, d_model = 2, 8, config.d_model
        
        # Initialize layer
        dummy_input = jax.random.normal(key, (batch_size, seq_len, d_model))
        variables = s5_layer.init(key, dummy_input, training=True)
        
        test_cases = [
            ("normal", jax.random.normal(key, (batch_size, seq_len, d_model))),
            ("small", jax.random.normal(key, (batch_size, seq_len, d_model)) * 1e-3),
            ("large", jax.random.normal(key, (batch_size, seq_len, d_model)) * 10.0),
            ("zeros", jnp.zeros((batch_size, seq_len, d_model))),
        ]
        
        for case_name, test_input in test_cases:
            logger.info(f"Testing S5 stability with {case_name} inputs...")
            
            try:
                output, final_state = s5_layer.apply(variables, test_input, training=True)
                
                # Check for NaNs/Infs
                assert not check_for_nans(output, f"S5 output ({case_name})")
                assert not check_for_nans(final_state, f"S5 final_state ({case_name})")
                
                # Check output magnitude is reasonable
                output_norm = jnp.linalg.norm(output)
                assert output_norm < 1e6, f"Output too large ({case_name}): {output_norm}"
                
                logger.info(f"  ✓ {case_name} case passed")
                
            except Exception as e:
                logger.error(f"  ❌ {case_name} case failed: {e}")
                raise
        
        logger.info("✓ S5 numerical stability test passed")


def run_s5_tests():
    """Run all S5 tests."""
    
    logger.info("=== Running S5 Unit Tests ===")
    
    # Create test configuration
    config = ValkyrieConfig(
        d_model=256,
        s5_state_dim=64,
        use_s5=True,
        n_layers=2,
    )
    
    # Create S5 layer
    s5_layer = ValkyrieS5(config=config, state_dim=config.s5_state_dim)
    
    # Create test data
    key = jax.random.PRNGKey(42)
    test_data = jax.random.normal(key, (2, 8, config.d_model))
    
    # Create test instance
    test_instance = TestS5Implementation()
    
    try:
        # Run tests
        test_instance.test_s5_initialization(s5_layer, config)
        test_instance.test_complex_parameter_construction(s5_layer, config)
        test_instance.test_parallel_vs_sequential_equality(s5_layer, config, test_data)
        test_instance.test_s5_forward_pass(s5_layer, config, test_data)
        test_instance.test_s5_recurrent_mode(s5_layer, config)
        test_instance.test_s5_gradient_flow(s5_layer, config, test_data)
        test_instance.test_s5_discretization(s5_layer, config)
        test_instance.test_s5_binary_operator(s5_layer, config)
        test_instance.test_s5_numerical_stability(s5_layer, config)
        
        logger.info("🎉 ALL S5 TESTS PASSED!")
        return True
        
    except Exception as e:
        logger.error(f"❌ S5 tests failed: {e}")
        return False


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Run tests
    success = run_s5_tests()
    
    if success:
        print("✅ S5 unit tests completed successfully")
        exit(0)
    else:
        print("❌ S5 unit tests failed")
        exit(1)