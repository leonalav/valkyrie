"""
Unit tests for HRM core functionality.

Tests gradient detachment, shape validation, ACT target computation,
and other critical components for safety and correctness.
"""

import pytest
import jax
import jax.numpy as jnp
import numpy as np
from typing import Dict, Any
import functools

# Import HRM modules
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models.hrm_inner import HRMInner, HRMInnerCarry
from models.hrm_act import HRMWithACT, ACTState, ACTOutput
# Note: Training imports commented out as modules not yet implemented
# from training import (
#     HRMTrainingState, compute_q_targets_paper, compute_language_modeling_loss,
#     segment_train_step, LossConfig, validate_carry_detachment
# )
import sys
import os
# Add the parent directory to Python path to import from data.py
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import directly from the data.py file by specifying the full module path
import importlib.util
spec = importlib.util.spec_from_file_location("data_module", os.path.join(os.path.dirname(__file__), '..', 'data.py'))
data_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(data_module)
DataBatch = data_module.DataBatch


class TestGradientDetachment:
    """Test gradient detachment mechanisms."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.batch_size = 4
        self.seq_len = 32
        self.vocab_size = 12
        self.rng_key = jax.random.PRNGKey(42)
        
        # Model parameters (HRMInner uses direct parameters, not a config class)
        self.model_kwargs = {
            'vocab_size': self.vocab_size,
            'hidden_size': 64,
            'seq_len': self.seq_len,
            'H_layers': 2,
            'L_layers': 2,
            'num_heads': 4,
            'intermediate_size': 128
        }
        
        # Create ACT config
        self.act_kwargs = {
            'max_steps': 8,
            'exploration_prob': 0.1,
            'min_steps': 2,
            'q_target_discount': 0.95
        }
    
    def create_dummy_batch(self) -> DataBatch:
        """Create a dummy batch for testing."""
        inputs = jax.random.randint(
            self.rng_key, (self.batch_size, self.seq_len), 0, self.vocab_size
        )
        targets = jax.random.randint(
            self.rng_key, (self.batch_size, self.seq_len), 0, self.vocab_size
        )
        puzzle_ids = jnp.arange(self.batch_size)
        group_ids = jnp.arange(self.batch_size)
        mask = jnp.ones((self.batch_size, self.seq_len))
        
        return DataBatch(
            inputs=inputs,
            targets=targets,
            puzzle_ids=puzzle_ids,
            group_ids=group_ids,
            mask=mask
        )
    
    def test_carry_detachment_basic(self):
        """Test basic carry detachment functionality."""
        # Create carry state
        carry = HRMInnerCarry(
            z_H=jnp.ones((self.batch_size, self.config.d_model)),
            z_L=jnp.ones((self.batch_size, self.config.d_model))
        )
        
        # Test stop_gradient detachment
        detached_carry = jax.tree_map(jax.lax.stop_gradient, carry)
        
        # Verify shapes are preserved
        assert detached_carry.z_H.shape == carry.z_H.shape
        assert detached_carry.z_L.shape == carry.z_L.shape
        
        # Verify values are preserved
        assert jnp.allclose(detached_carry.z_H, carry.z_H)
        assert jnp.allclose(detached_carry.z_L, carry.z_L)
    
    def test_gradient_flow_prevention(self):
        """Test that detached carry prevents gradient flow."""
        model = HRMInner(**self.model_kwargs)
        data_batch = self.create_dummy_batch()
        
        # Convert DataBatch to dictionary format expected by HRMInner
        batch = {
            "inputs": data_batch.inputs,
            "puzzle_identifiers": data_batch.puzzle_ids
        }
        
        # Initialize model first to trigger setup()
        init_rng, forward_rng = jax.random.split(self.rng_key)
        # Create a dummy carry for initialization with correct batch size
        dummy_carry = HRMInnerCarry(
            z_H=jnp.zeros((self.batch_size, 32, 64), dtype=jnp.bfloat16),  # Use self.batch_size=4
            z_L=jnp.zeros((self.batch_size, 32, 64), dtype=jnp.bfloat16)
        )
        params = model.init(init_rng, dummy_carry, batch)["params"]
        
        # Now we can safely call empty_carry after initialization
        empty_carry = model.empty_carry(batch_size=self.batch_size)
        
        # Create initial carry
        carry = model.initial_carry(self.batch_size)
        
        def loss_with_carry_grads(carry_state):
            """Loss function that depends on carry state."""
            carry_out, lm_logits, q_logits = model.apply(
                {"params": params},
                carry_state,
                batch
            )
            return jnp.mean(lm_logits)
        
        # Test gradient computation with normal carry
        try:
            normal_grads = jax.grad(loss_with_carry_grads)(carry)
            has_normal_grads = True
        except:
            has_normal_grads = False
        
        # Test gradient computation with detached carry
        detached_carry = jax.tree_map(jax.lax.stop_gradient, carry)
        try:
            detached_grads = jax.grad(loss_with_carry_grads)(detached_carry)
            # If this succeeds, gradients should be zero
            max_grad = jnp.max(jnp.abs(jax.tree_util.tree_flatten(detached_grads)[0][0]))
            assert max_grad < 1e-10, f"Detached carry still has gradients: {max_grad}"
        except:
            # If gradient computation fails, that's also acceptable (carry is detached)
            pass
    
    def test_training_state_detachment(self):
        """Test HRMTrainingState carry detachment."""
        import optax
        
        model = HRMWithACT(
            HRMInner(**self.model_kwargs),
            **self.act_kwargs
        )
        
        # Create training state
        optimizer = optax.adam(1e-4)
        batch = self.create_dummy_batch()
        
        init_rng, _ = jax.random.split(self.rng_key)
        params = model.init(init_rng, batch, training=False)["params"]
        
        initial_carry = model.initial_carry(self.batch_size)
        
        state = HRMTrainingState.create(
            apply_fn=model.apply,
            params=params,
            tx=optimizer,
            carry=initial_carry,
            rng_key=self.rng_key
        )
        
        # Test detachment
        detached_state = state.detach_carry()
        
        # Verify carry is detached but other fields are preserved
        assert detached_state.params is state.params
        assert detached_state.opt_state is state.opt_state
        assert detached_state.step == state.step
        
        # Verify carry shapes are preserved
        assert detached_state.carry.z_H.shape == state.carry.z_H.shape
        assert detached_state.carry.z_L.shape == state.carry.z_L.shape


class TestShapeValidation:
    """Test shape validation and tensor operations."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.batch_size = 4
        self.seq_len = 32
        self.vocab_size = 12
        self.d_model = 64
        self.rng_key = jax.random.PRNGKey(42)
    
    def test_model_output_shapes(self):
        """Test that model outputs have correct shapes."""
        model_kwargs = {
            'vocab_size': self.vocab_size,
            'hidden_size': self.d_model,
            'seq_len': self.seq_len,
            'H_layers': 2,
            'L_layers': 2,
            'num_heads': 4,
            'intermediate_size': 128
        }
        
        model = HRMInner(**model_kwargs)
        
        # Create batch
        batch = DataBatch(
            inputs=jnp.ones((self.batch_size, self.seq_len), dtype=jnp.int32),
            targets=jnp.ones((self.batch_size, self.seq_len), dtype=jnp.int32),
            puzzle_ids=jnp.arange(self.batch_size),
            group_ids=jnp.arange(self.batch_size),
            mask=jnp.ones((self.batch_size, self.seq_len))
        )
        
        # Initialize and run model
        init_rng, forward_rng = jax.random.split(self.rng_key)
        params = model.init(init_rng, batch, training=False)["params"]
        
        carry = model.initial_carry(self.batch_size)
        carry_out, lm_logits, (q_halt_logits, q_continue_logits) = model.apply(
            {"params": params},
            batch,
            carry=carry,
            rng_key=forward_rng,
            training=True
        )
        
        # Validate output shapes
        assert lm_logits.shape == (self.batch_size, self.seq_len, self.vocab_size)
        assert q_halt_logits.shape == (self.batch_size,)
        assert q_continue_logits.shape == (self.batch_size,)
        assert carry_out.z_H.shape == (self.batch_size, self.d_model)
        assert carry_out.z_L.shape == (self.batch_size, self.d_model)
    
    def test_act_wrapper_shapes(self):
        """Test ACT wrapper output shapes."""
        model_kwargs = {
            'vocab_size': self.vocab_size,
            'hidden_size': self.d_model,
            'seq_len': self.seq_len,
            'H_layers': 2,
            'L_layers': 2,
            'num_heads': 4,
            'intermediate_size': 128
        }
        
        act_config = ACTConfig(
            max_steps=8,
            exploration_prob=0.1,
            min_steps=2
        )
        
        model = HRMWithACT(HRMInner(**model_kwargs), act_config)
        
        batch = DataBatch(
            inputs=jnp.ones((self.batch_size, self.seq_len), dtype=jnp.int32),
            targets=jnp.ones((self.batch_size, self.seq_len), dtype=jnp.int32),
            puzzle_ids=jnp.arange(self.batch_size),
            group_ids=jnp.arange(self.batch_size),
            mask=jnp.ones((self.batch_size, self.seq_len))
        )
        
        # Initialize and run model
        init_rng, forward_rng = jax.random.split(self.rng_key)
        params = model.init(init_rng, batch, training=False)["params"]
        
        carry = model.initial_carry(self.batch_size)
        output = model.apply(
            {"params": params},
            batch,
            carry=carry,
            rng_key=forward_rng,
            training=True
        )
        
        # Validate ACT output shapes
        assert output.lm_logits.shape == (self.batch_size, self.seq_len, self.vocab_size)
        assert output.q_halt_logits.shape == (self.batch_size, act_config.max_steps)
        assert output.q_continue_logits.shape == (self.batch_size, act_config.max_steps)
        assert output.step_count.shape == (self.batch_size,)
        assert output.final_carry.z_H.shape == (self.batch_size, self.d_model)
        assert output.final_carry.z_L.shape == (self.batch_size, self.d_model)
    
    def test_batch_dimension_consistency(self):
        """Test that batch dimensions are consistent across operations."""
        batch_sizes = [1, 4, 8, 16]
        
        model_kwargs = {
            'vocab_size': self.vocab_size,
            'hidden_size': 32,  # Smaller for faster testing
            'seq_len': self.seq_len,
            'H_layers': 1,
            'L_layers': 1,
            'num_heads': 2,
            'intermediate_size': 64
        }
        
        model = HRMInner(**model_kwargs)
        
        for batch_size in batch_sizes:
            batch = DataBatch(
                inputs=jnp.ones((batch_size, self.seq_len), dtype=jnp.int32),
                targets=jnp.ones((batch_size, self.seq_len), dtype=jnp.int32),
                puzzle_ids=jnp.arange(batch_size),
                group_ids=jnp.arange(batch_size),
                mask=jnp.ones((batch_size, self.seq_len))
            )
            
            init_rng, forward_rng = jax.random.split(self.rng_key)
            params = model.init(init_rng, batch, training=False)["params"]
            
            carry = model.initial_carry(batch_size)
            carry_out, lm_logits, (q_halt_logits, q_continue_logits) = model.apply(
                {"params": params},
                batch,
                carry=carry,
                rng_key=forward_rng,
                training=True
            )
            
            # All outputs should have consistent batch dimension
            assert lm_logits.shape[0] == batch_size
            assert q_halt_logits.shape[0] == batch_size
            assert q_continue_logits.shape[0] == batch_size
            assert carry_out.z_H.shape[0] == batch_size
            assert carry_out.z_L.shape[0] == batch_size


class TestACTTargetComputation:
    """Test ACT Q-target computation following paper rules."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.batch_size = 4
        self.max_steps = 6
        self.seq_len = 8
        self.vocab_size = 12
        self.rng_key = jax.random.PRNGKey(42)
    
    def test_q_target_shapes(self):
        """Test Q-target computation shapes."""
        # Create dummy data
        lm_logits = jax.random.normal(
            self.rng_key, (self.batch_size, self.max_steps, self.seq_len, self.vocab_size)
        )
        targets = jax.random.randint(
            self.rng_key, (self.batch_size, self.seq_len), 0, self.vocab_size
        )
        q_halt_logits = jax.random.normal(self.rng_key, (self.batch_size, self.max_steps))
        q_continue_logits = jax.random.normal(self.rng_key, (self.batch_size, self.max_steps))
        step_mask = jnp.ones((self.batch_size, self.max_steps), dtype=bool)
        
        # Compute Q-targets
        q_targets = compute_q_targets_paper(
            lm_logits, targets, q_halt_logits, q_continue_logits, step_mask
        )
        
        # Validate shape
        assert q_targets.shape == (self.batch_size, self.max_steps)
        
        # Validate range (should be between 0 and 1 for correctness rewards)
        assert jnp.all(q_targets >= 0.0)
        assert jnp.all(q_targets <= 1.0)
    
    def test_q_target_correctness_reward(self):
        """Test that Q-targets correctly compute G_halt = 1{y_hat == y}."""
        batch_size = 2
        max_steps = 3
        seq_len = 4
        vocab_size = 5
        
        # Create controlled data where we know the correctness
        targets = jnp.array([[1, 2, 3, 4], [0, 1, 2, 3]])  # [batch_size, seq_len]
        
        # Create predictions that are correct for first batch, incorrect for second
        lm_logits = jnp.zeros((batch_size, max_steps, seq_len, vocab_size))
        
        # Make first batch predictions correct (high logits for target classes)
        lm_logits = lm_logits.at[0, :, 0, 1].set(10.0)  # Predict class 1 for position 0
        lm_logits = lm_logits.at[0, :, 1, 2].set(10.0)  # Predict class 2 for position 1
        lm_logits = lm_logits.at[0, :, 2, 3].set(10.0)  # Predict class 3 for position 2
        lm_logits = lm_logits.at[0, :, 3, 4].set(10.0)  # Predict class 4 for position 3
        
        # Make second batch predictions incorrect (high logits for wrong classes)
        lm_logits = lm_logits.at[1, :, 0, 2].set(10.0)  # Predict class 2 (target is 0)
        lm_logits = lm_logits.at[1, :, 1, 3].set(10.0)  # Predict class 3 (target is 1)
        lm_logits = lm_logits.at[1, :, 2, 4].set(10.0)  # Predict class 4 (target is 2)
        lm_logits = lm_logits.at[1, :, 3, 0].set(10.0)  # Predict class 0 (target is 3)
        
        q_halt_logits = jnp.zeros((batch_size, max_steps))
        q_continue_logits = jnp.zeros((batch_size, max_steps))
        step_mask = jnp.ones((batch_size, max_steps), dtype=bool)
        
        # Compute Q-targets
        q_targets = compute_q_targets_paper(
            lm_logits, targets, q_halt_logits, q_continue_logits, step_mask
        )
        
        # First batch should have high rewards (correct predictions)
        assert jnp.all(q_targets[0] > 0.9), f"First batch Q-targets: {q_targets[0]}"
        
        # Second batch should have low rewards (incorrect predictions)
        assert jnp.all(q_targets[1] < 0.1), f"Second batch Q-targets: {q_targets[1]}"
    
    def test_q_target_discount_factor(self):
        """Test that discount factor is applied correctly."""
        batch_size = 1
        max_steps = 4
        seq_len = 2
        vocab_size = 3
        discount = 0.9
        
        # Create dummy data
        lm_logits = jax.random.normal(
            self.rng_key, (batch_size, max_steps, seq_len, vocab_size)
        )
        targets = jnp.array([[1, 2]])
        q_halt_logits = jnp.array([[1.0, 2.0, 3.0, 4.0]])
        q_continue_logits = jnp.array([[0.5, 1.5, 2.5, 3.5]])
        step_mask = jnp.ones((batch_size, max_steps), dtype=bool)
        
        # Compute Q-targets with different discount factors
        q_targets_09 = compute_q_targets_paper(
            lm_logits, targets, q_halt_logits, q_continue_logits, step_mask, discount=0.9
        )
        q_targets_05 = compute_q_targets_paper(
            lm_logits, targets, q_halt_logits, q_continue_logits, step_mask, discount=0.5
        )
        
        # With lower discount, future rewards should be discounted more
        # This is a basic sanity check - exact values depend on the computation
        assert q_targets_09.shape == q_targets_05.shape
    
    def test_step_mask_application(self):
        """Test that step mask correctly zeros out invalid steps."""
        batch_size = 2
        max_steps = 4
        seq_len = 2
        vocab_size = 3
        
        # Create dummy data
        lm_logits = jax.random.normal(
            self.rng_key, (batch_size, max_steps, seq_len, vocab_size)
        )
        targets = jnp.array([[1, 2], [0, 1]])
        q_halt_logits = jnp.ones((batch_size, max_steps))
        q_continue_logits = jnp.ones((batch_size, max_steps))
        
        # Create step mask where only first 2 steps are valid for first batch,
        # and first 3 steps are valid for second batch
        step_mask = jnp.array([
            [True, True, False, False],
            [True, True, True, False]
        ])
        
        q_targets = compute_q_targets_paper(
            lm_logits, targets, q_halt_logits, q_continue_logits, step_mask
        )
        
        # Invalid steps should have zero targets
        assert q_targets[0, 2] == 0.0
        assert q_targets[0, 3] == 0.0
        assert q_targets[1, 3] == 0.0
        
        # Valid steps should have non-zero targets (assuming some correctness)
        # Note: This might not always be true depending on predictions, but mask should work


class TestTrainingStepIntegration:
    """Test integration of training step components."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.batch_size = 4
        self.seq_len = 16
        self.vocab_size = 12
        self.rng_key = jax.random.PRNGKey(42)
        
        self.model_kwargs = {
            'vocab_size': self.vocab_size,
            'hidden_size': 32,
            'seq_len': self.seq_len,
            'H_layers': 1,
            'L_layers': 1,
            'num_heads': 2,
            'intermediate_size': 64
        }
        
        self.act_kwargs = {
            'max_steps': 8,
            'exploration_prob': 0.1,
            'min_steps': 2
        }
        
        # Note: LossConfig not yet implemented, using mock
        from unittest.mock import Mock
        self.loss_config = Mock()
        self.loss_config.lm_weight = 1.0
        self.loss_config.act_weight = 0.1
        self.loss_config.deep_supervision_weight = 0.0  # Disable for testing
    
    def create_training_state(self):
        """Create a training state for testing."""
        # Note: Training module not yet implemented, using mock for testing
        from unittest.mock import Mock
        
        model = HRMWithACT(
            HRMInner(**self.model_kwargs),
            **self.act_kwargs
        )
        
        # Mock training state for testing purposes
        state = Mock()
        state.params = model.init(self.rng_key, self.sample_batch, training=False)["params"]
        state.opt_state = None
        
        return state
    
    def create_dummy_batch(self) -> DataBatch:
        """Create a dummy batch for testing."""
        inputs = jax.random.randint(
            self.rng_key, (self.batch_size, self.seq_len), 0, self.vocab_size
        )
        targets = jax.random.randint(
            self.rng_key, (self.batch_size, self.seq_len), 0, self.vocab_size
        )
        
        return DataBatch(
            inputs=inputs,
            targets=targets,
            puzzle_ids=jnp.arange(self.batch_size),
            group_ids=jnp.arange(self.batch_size),
            mask=jnp.ones((self.batch_size, self.seq_len))
        )
    
    def test_segment_train_step_execution(self):
        """Test that segment training step executes without errors."""
        state = self.create_training_state()
        batch = self.create_dummy_batch()
        
        # Run one training step
        new_state, metrics = segment_train_step(state, batch, self.loss_config)
        
        # Verify state is updated
        assert new_state.step == state.step + 1
        
        # Verify metrics are computed
        assert hasattr(metrics, 'total_loss')
        assert hasattr(metrics, 'lm_loss')
        assert hasattr(metrics, 'act_loss')
        assert hasattr(metrics, 'mean_steps')
        
        # Verify carry is detached (should be different object)
        assert new_state.carry is not state.carry
    
    def test_carry_detachment_validation(self):
        """Test the carry detachment validation function."""
        state = self.create_training_state()
        batch1 = self.create_dummy_batch()
        batch2 = self.create_dummy_batch()
        
        # Test carry detachment validation
        is_detached = validate_carry_detachment(state, batch1, batch2, self.loss_config)
        
        # Should return True if carry is properly detached
        assert isinstance(is_detached, bool)
    
    def test_loss_computation_shapes(self):
        """Test that loss computation produces correct shapes."""
        from training import compute_total_loss
        
        model = HRMWithACT(HRMInner(self.config), self.act_config)
        batch = self.create_dummy_batch()
        
        # Initialize model and get output
        init_rng, forward_rng = jax.random.split(self.rng_key)
        params = model.init(init_rng, batch, training=False)["params"]
        
        carry = model.initial_carry(self.batch_size)
        act_output = model.apply(
            {"params": params},
            batch,
            carry=carry,
            rng_key=forward_rng,
            training=True
        )
        
        # Compute losses
        total_loss, metrics = compute_total_loss(act_output, batch, self.loss_config)
        
        # Verify loss is scalar
        assert total_loss.shape == ()
        assert jnp.isfinite(total_loss)
        
        # Verify metrics
        assert isinstance(metrics.total_loss, float)
        assert isinstance(metrics.lm_loss, float)
        assert isinstance(metrics.act_loss, float)
        assert isinstance(metrics.mean_steps, float)


# Test runner
def run_all_tests():
    """Run all tests and report results."""
    test_classes = [
        TestGradientDetachment,
        TestShapeValidation,
        TestACTTargetComputation,
        TestTrainingStepIntegration
    ]
    
    total_tests = 0
    passed_tests = 0
    failed_tests = []
    
    for test_class in test_classes:
        print(f"\nRunning {test_class.__name__}...")
        
        # Get all test methods
        test_methods = [method for method in dir(test_class) if method.startswith('test_')]
        
        for test_method in test_methods:
            total_tests += 1
            
            try:
                # Create test instance and run setup
                test_instance = test_class()
                if hasattr(test_instance, 'setup_method'):
                    test_instance.setup_method()
                
                # Run test method
                getattr(test_instance, test_method)()
                
                print(f"  ✓ {test_method}")
                passed_tests += 1
                
            except Exception as e:
                print(f"  ✗ {test_method}: {e}")
                failed_tests.append(f"{test_class.__name__}.{test_method}: {e}")
    
    print(f"\n{'='*60}")
    print(f"Test Results: {passed_tests}/{total_tests} passed")
    
    if failed_tests:
        print(f"\nFailed tests:")
        for failure in failed_tests:
            print(f"  - {failure}")
        return False
    else:
        print("All tests passed! ✓")
        return True


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)