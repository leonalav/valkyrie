"""Gryphon Integration Tests

Comprehensive tests for the hybrid BigBird-S5 architecture.
Tests cover model creation, forward passes, training steps, generation,
numerical stability, and performance characteristics.
"""

import jax
import jax.numpy as jnp
import pytest
import time
from typing import Dict, Any

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model.gryphon.gryphon_config import GryphonConfig, get_gryphon_small_config
from src.model.gryphon.gryphon_model import GryphonModel, create_gryphon_small
from src.model.gryphon.gryphon_utils import (
    create_sparse_attention_mask,
    pad_to_block_size,
    compute_attention_sparsity
)
from src.model.gryphon.training_utils import (
    create_gryphon_optimizer,
    compute_gryphon_loss,
    monitor_s5_stability,
    check_gradient_health
)


class TestGryphonBasics:
    """Test basic model functionality."""
    
    def test_model_creation(self):
        """Test model creation and parameter initialization."""
        model = create_gryphon_small(vocab_size=1000)
        
        # Check model info
        info = model.get_model_info()
        assert info['model_type'] == 'Gryphon (BigBird + S5 Hybrid)'
        assert 'total_parameters' in info
        assert 'sparse_attention' in info
        
        # Check configuration
        config = model.config
        assert config.d_model > 0
        assert config.n_layers > 0
        assert config.s5_state_dim > 0
        assert config.block_size > 0
    
    def test_parameter_initialization(self):
        """Test parameter initialization."""
        model = create_gryphon_small(vocab_size=100)
        rng = jax.random.PRNGKey(42)
        
        # Create dummy input
        batch_size, seq_len = 2, 128
        input_ids = jax.random.randint(rng, (batch_size, seq_len), 0, 100)
        
        # Initialize parameters
        params = model.init(rng, input_ids, training=False)
        
        # Check parameter structure
        assert 'params' in params
        assert 'embeddings' in params['params']
        assert 'layers' in params['params']
        assert 'final_norm' in params['params']
        assert 'lm_head' in params['params']
        
        # Check for NaN/Inf in parameters
        def check_params(param_dict):
            for key, value in param_dict.items():
                if isinstance(value, dict):
                    check_params(value)
                elif isinstance(value, jnp.ndarray):
                    assert not jnp.any(jnp.isnan(value)), f"NaN found in {key}"
                    assert not jnp.any(jnp.isinf(value)), f"Inf found in {key}"
        
        check_params(params)


class TestGryphonForward:
    """Test forward pass functionality."""
    
    def test_forward_pass_shapes(self):
        """Test forward pass output shapes."""
        model = create_gryphon_small(vocab_size=100)
        rng = jax.random.PRNGKey(42)
        
        batch_size, seq_len = 2, 128
        input_ids = jax.random.randint(rng, (batch_size, seq_len), 0, 100)
        attention_mask = jnp.ones((batch_size, seq_len))
        
        params = model.init(rng, input_ids, attention_mask, training=False)
        outputs = model.apply(params, input_ids, attention_mask, training=False)
        
        # Check output shapes
        assert outputs['logits'].shape == (batch_size, seq_len, 100)
        assert outputs['hidden_states'].shape == (batch_size, seq_len, model.config.d_model)
        assert len(outputs['s5_states']) == model.config.n_layers
    
    def test_forward_pass_different_lengths(self):
        """Test forward pass with different sequence lengths."""
        model = create_gryphon_small(vocab_size=100)
        rng = jax.random.PRNGKey(42)
        
        batch_size = 2
        test_lengths = [64, 128, 192, 256]
        
        for seq_len in test_lengths:
            input_ids = jax.random.randint(rng, (batch_size, seq_len), 0, 100)
            attention_mask = jnp.ones((batch_size, seq_len))
            
            params = model.init(rng, input_ids, attention_mask, training=False)
            outputs = model.apply(params, input_ids, attention_mask, training=False)
            
            assert outputs['logits'].shape == (batch_size, seq_len, 100)
    
    def test_forward_pass_with_padding(self):
        """Test forward pass with sequence padding."""
        model = create_gryphon_small(vocab_size=100)
        rng = jax.random.PRNGKey(42)
        
        batch_size, seq_len = 2, 100  # Not divisible by block_size (64)
        input_ids = jax.random.randint(rng, (batch_size, seq_len), 0, 100)
        attention_mask = jnp.ones((batch_size, seq_len))
        
        params = model.init(rng, input_ids, attention_mask, training=False)
        outputs = model.apply(params, input_ids, attention_mask, training=False)
        
        # Output should match original sequence length
        assert outputs['logits'].shape == (batch_size, seq_len, 100)
    
    def test_training_vs_inference_mode(self):
        """Test differences between training and inference modes."""
        model = create_gryphon_small(vocab_size=100)
        rng = jax.random.PRNGKey(42)
        
        batch_size, seq_len = 2, 128
        input_ids = jax.random.randint(rng, (batch_size, seq_len), 0, 100)
        attention_mask = jnp.ones((batch_size, seq_len))
        
        params = model.init(rng, input_ids, attention_mask, training=True)
        
        # Training mode
        train_outputs = model.apply(params, input_ids, attention_mask, training=True)
        
        # Inference mode
        eval_outputs = model.apply(params, input_ids, attention_mask, training=False)
        
        # Shapes should be the same
        assert train_outputs['logits'].shape == eval_outputs['logits'].shape
        
        # Values might be different due to dropout
        # But they should be close if dropout is low
        assert jnp.allclose(train_outputs['logits'], eval_outputs['logits'], atol=0.1)


class TestGryphonTraining:
    """Test training functionality."""
    
    def test_gradient_computation(self):
        """Test gradient computation."""
        model = create_gryphon_small(vocab_size=100)
        rng = jax.random.PRNGKey(42)
        
        batch_size, seq_len = 2, 128
        input_ids = jax.random.randint(rng, (batch_size, seq_len), 0, 100)
        attention_mask = jnp.ones((batch_size, seq_len))
        
        params = model.init(rng, input_ids, attention_mask, training=True)
        
        def loss_fn(params):
            outputs = model.apply(params, input_ids, attention_mask, training=True)
            loss, metrics = compute_gryphon_loss(
                outputs['logits'], input_ids, attention_mask
            )
            return loss, metrics
        
        # Compute gradients
        (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        
        # Check loss is finite
        assert jnp.isfinite(loss)
        assert loss > 0
        
        # Check gradients exist and are finite
        def check_grads(grad_dict):
            for key, value in grad_dict.items():
                if isinstance(value, dict):
                    check_grads(value)
                elif isinstance(value, jnp.ndarray):
                    assert not jnp.any(jnp.isnan(value)), f"NaN gradient in {key}"
                    assert not jnp.any(jnp.isinf(value)), f"Inf gradient in {key}"
        
        check_grads(grads)
        
        # Check metrics
        assert 'accuracy' in metrics
        assert 'perplexity' in metrics
        assert metrics['accuracy'] >= 0.0
        assert metrics['accuracy'] <= 1.0
    
    def test_optimizer_integration(self):
        """Test optimizer integration."""
        config = get_gryphon_small_config()
        config.vocab_size = 100
        
        model = GryphonModel(config=config)
        rng = jax.random.PRNGKey(42)
        
        batch_size, seq_len = 2, 128
        input_ids = jax.random.randint(rng, (batch_size, seq_len), 0, 100)
        
        params = model.init(rng, input_ids, training=True)
        optimizer = create_gryphon_optimizer(config, base_learning_rate=1e-3)
        opt_state = optimizer.init(params)
        
        def loss_fn(params):
            outputs = model.apply(params, input_ids, training=True)
            loss, metrics = compute_gryphon_loss(outputs['logits'], input_ids)
            return loss, metrics
        
        # Training step
        (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = jax.tree_map(lambda p, u: p + u, params, updates)
        
        # Check parameters were updated
        def params_changed(old_params, new_params):
            for key in old_params:
                if isinstance(old_params[key], dict):
                    if params_changed(old_params[key], new_params[key]):
                        return True
                elif isinstance(old_params[key], jnp.ndarray):
                    if not jnp.allclose(old_params[key], new_params[key]):
                        return True
            return False
        
        assert params_changed(params, new_params)


class TestGryphonGeneration:
    """Test generation functionality."""
    
    def test_s5_state_initialization(self):
        """Test S5 state initialization."""
        model = create_gryphon_small(vocab_size=100)
        batch_size = 3
        
        s5_states = model.init_s5_states(batch_size)
        
        # Check structure
        assert len(s5_states) == model.config.n_layers
        
        for layer_states in s5_states:
            assert len(layer_states) >= 1  # At least one block per layer
            
            for state in layer_states:
                assert state.shape == (batch_size, model.config.s5_state_dim)
                assert state.dtype == jnp.complex64
                assert jnp.allclose(state, 0.0)  # Should be initialized to zeros
    
    def test_generation_step(self):
        """Test single generation step."""
        model = create_gryphon_small(vocab_size=100)
        rng = jax.random.PRNGKey(42)
        
        batch_size = 1
        input_ids = jax.random.randint(rng, (batch_size, 1), 0, 100)
        
        params = model.init(rng, input_ids, training=False)
        s5_states = model.init_s5_states(batch_size)
        
        # Generation step
        next_token, updated_states = model.apply(
            params, input_ids, s5_states=s5_states, training=False,
            method=model.generate_step, temperature=1.0, top_k=50, top_p=0.9
        )
        
        # Check output
        assert next_token.shape == (batch_size, 1)
        assert 0 <= next_token[0, 0] < 100
        
        # Check updated states
        assert len(updated_states) == len(s5_states)
        
        # States should have changed
        for old_layer, new_layer in zip(s5_states, updated_states):
            for old_state, new_state in zip(old_layer, new_layer):
                assert not jnp.allclose(old_state, new_state)


class TestGryphonStability:
    """Test numerical stability."""
    
    def test_s5_stability_monitoring(self):
        """Test S5 stability monitoring."""
        model = create_gryphon_small(vocab_size=100)
        rng = jax.random.PRNGKey(42)
        
        batch_size, seq_len = 2, 128
        input_ids = jax.random.randint(rng, (batch_size, seq_len), 0, 100)
        
        params = model.init(rng, input_ids, training=False)
        
        # Monitor stability
        stability_metrics = monitor_s5_stability(params)
        
        # Check that we get meaningful metrics
        assert len(stability_metrics) > 0
        
        # Check for problematic values
        for key, value in stability_metrics.items():
            if 'nan_count' in key or 'inf_count' in key:
                assert value == 0, f"Found NaN/Inf in {key}"
    
    def test_gradient_health_monitoring(self):
        """Test gradient health monitoring."""
        model = create_gryphon_small(vocab_size=100)
        rng = jax.random.PRNGKey(42)
        
        batch_size, seq_len = 2, 128
        input_ids = jax.random.randint(rng, (batch_size, seq_len), 0, 100)
        
        params = model.init(rng, input_ids, training=True)
        
        def loss_fn(params):
            outputs = model.apply(params, input_ids, training=True)
            loss, metrics = compute_gryphon_loss(outputs['logits'], input_ids)
            return loss
        
        grads = jax.grad(loss_fn)(params)
        grad_health = check_gradient_health(grads)
        
        # Check gradient health metrics
        assert 'global_grad_norm' in grad_health
        assert grad_health['global_grad_norm'] > 0
        assert jnp.isfinite(grad_health['global_grad_norm'])
        
        # Check for problematic gradients
        for key, value in grad_health.items():
            if 'nan_count' in key or 'inf_count' in key:
                assert value == 0, f"Found NaN/Inf gradients in {key}"


class TestGryphonUtils:
    """Test utility functions."""
    
    def test_sparse_attention_mask(self):
        """Test sparse attention mask creation."""
        num_blocks = 16
        block_size = 64
        num_global_blocks = 2
        window_size = 3
        num_random_blocks = 2
        
        mask = create_sparse_attention_mask(
            num_blocks, block_size, num_global_blocks, 
            window_size, num_random_blocks
        )
        
        # Check mask shape
        assert mask.shape == (num_blocks, num_blocks)
        
        # Check global blocks attend to all
        for i in range(num_global_blocks):
            assert jnp.all(mask[i, :])
        
        # Check all blocks attend to global blocks
        for i in range(num_blocks):
            assert jnp.all(mask[i, :num_global_blocks])
        
        # Check self-attention
        for i in range(num_blocks):
            assert mask[i, i]
    
    def test_attention_sparsity_computation(self):
        """Test attention sparsity computation."""
        num_blocks = 16
        num_global_blocks = 2
        window_size = 3
        num_random_blocks = 2
        
        sparsity_info = compute_attention_sparsity(
            num_blocks, num_global_blocks, window_size, num_random_blocks
        )
        
        # Check sparsity metrics
        assert 'sparsity_ratio' in sparsity_info
        assert 'full_attention_ops' in sparsity_info
        assert 'sparse_attention_ops' in sparsity_info
        
        # Sparsity should be significant
        assert sparsity_info['sparsity_ratio'] > 0.5
        assert sparsity_info['sparse_attention_ops'] < sparsity_info['full_attention_ops']
    
    def test_padding_utilities(self):
        """Test padding utilities."""
        # Test padding to block size
        x = jnp.ones((2, 100, 64))  # Not divisible by 64
        block_size = 64
        
        padded_x, original_len = pad_to_block_size(x, block_size, axis=1)
        
        assert original_len == 100
        assert padded_x.shape[1] % block_size == 0
        assert padded_x.shape[1] >= original_len
        
        # Check padding values are zero
        if padded_x.shape[1] > original_len:
            padding_region = padded_x[:, original_len:, :]
            assert jnp.allclose(padding_region, 0.0)


class TestGryphonPerformance:
    """Test performance characteristics."""
    
    def test_memory_scaling(self):
        """Test memory scaling with sequence length."""
        vocab_size = 100
        batch_size = 1
        
        # Test different sequence lengths
        seq_lengths = [128, 256, 512]
        memory_usage = []
        
        for seq_len in seq_lengths:
            config = get_gryphon_small_config()
            config.vocab_size = vocab_size
            config.max_sequence_length = seq_len
            
            model = GryphonModel(config=config)
            rng = jax.random.PRNGKey(42)
            
            input_ids = jax.random.randint(rng, (batch_size, seq_len), 0, vocab_size)
            params = model.init(rng, input_ids, training=False)
            
            # Estimate memory usage (rough approximation)
            memory_est = config.get_memory_estimates(batch_size=batch_size)
            memory_usage.append(memory_est['estimated_total_memory_gb'])
        
        # Memory should scale roughly linearly with sequence length
        assert memory_usage[1] > memory_usage[0]
        assert memory_usage[2] > memory_usage[1]
    
    def test_forward_pass_timing(self):
        """Test forward pass timing."""
        model = create_gryphon_small(vocab_size=100)
        rng = jax.random.PRNGKey(42)
        
        batch_size, seq_len = 2, 256
        input_ids = jax.random.randint(rng, (batch_size, seq_len), 0, 100)
        
        params = model.init(rng, input_ids, training=False)
        
        # Warm up JIT compilation
        _ = model.apply(params, input_ids, training=False)
        
        # Time forward pass
        start_time = time.time()
        outputs = model.apply(params, input_ids, training=False)
        forward_time = time.time() - start_time
        
        # Should complete in reasonable time (< 1 second for small model)
        assert forward_time < 1.0
        assert outputs['logits'].shape == (batch_size, seq_len, 100)


# Test fixtures and utilities
@pytest.fixture
def small_gryphon_model():
    """Fixture for small Gryphon model."""
    return create_gryphon_small(vocab_size=100)


@pytest.fixture
def sample_batch():
    """Fixture for sample batch data."""
    rng = jax.random.PRNGKey(42)
    batch_size, seq_len = 2, 128
    
    return {
        'input_ids': jax.random.randint(rng, (batch_size, seq_len), 0, 100),
        'attention_mask': jnp.ones((batch_size, seq_len)),
        'rng': rng
    }


def test_end_to_end_workflow(small_gryphon_model, sample_batch):
    """Test complete end-to-end workflow."""
    model = small_gryphon_model
    batch = sample_batch
    
    # Initialize parameters
    params = model.init(batch['rng'], batch['input_ids'], batch['attention_mask'], training=True)
    
    # Forward pass
    outputs = model.apply(params, batch['input_ids'], batch['attention_mask'], training=True)
    
    # Compute loss
    loss, metrics = compute_gryphon_loss(outputs['logits'], batch['input_ids'], batch['attention_mask'])
    
    # Compute gradients
    grads = jax.grad(lambda p: compute_gryphon_loss(
        model.apply(p, batch['input_ids'], batch['attention_mask'], training=True)['logits'],
        batch['input_ids'], batch['attention_mask']
    )[0])(params)
    
    # Check everything is working
    assert jnp.isfinite(loss)
    assert loss > 0
    assert 'accuracy' in metrics
    assert not any(jnp.any(jnp.isnan(g)) for g in jax.tree_leaves(grads))


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])