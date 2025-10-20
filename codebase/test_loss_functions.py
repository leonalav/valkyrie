#!/usr/bin/env python3
"""
Test script to verify compute_act_loss and compute_total_loss functions work correctly.

This script tests:
1. Import of compute_act_loss from models package
2. compute_act_loss function with realistic data
3. compute_total_loss function with ACTOutput
4. JAX tracing compatibility (no ConcretizationTypeError)
"""

import sys
import os
sys.path.append('/home/ravkeave/v1/codebase/src')

import jax
import jax.numpy as jnp
import numpy as np
from typing import Dict

def test_imports():
    """Test that all required functions can be imported."""
    print("Testing imports...")
    
    try:
        from model.hrm.models import compute_act_loss, ACTOutput, HRMInnerCarry
        # Import directly from training.py in hrm directory (not the training package)
        import model.hrm.training as training_module
        compute_total_loss = training_module.compute_total_loss
        LossConfig = training_module.LossConfig
        TrainingMetrics = training_module.TrainingMetrics
        print("✓ All imports successful")
        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False

def test_compute_act_loss():
    """Test compute_act_loss function with realistic data."""
    print("\nTesting compute_act_loss...")
    
    try:
        from model.hrm.models import compute_act_loss
        
        # Create test data
        batch_size = 4
        max_steps = 6
        
        # Generate realistic Q-values
        key = jax.random.PRNGKey(42)
        key1, key2, key3 = jax.random.split(key, 3)
        
        q_halt_logits = jax.random.normal(key1, (batch_size, max_steps)) * 0.5
        q_continue_logits = jax.random.normal(key2, (batch_size, max_steps)) * 0.5
        q_targets = jax.random.uniform(key3, (batch_size, max_steps), minval=0.0, maxval=1.0)
        
        # Create step mask (some sequences halt early)
        step_counts = jnp.array([3, 5, 2, 6])  # Different halt times
        step_mask = jnp.arange(max_steps)[None, :] < step_counts[:, None]
        
        # Test the function
        loss = compute_act_loss(q_halt_logits, q_continue_logits, q_targets, step_mask)
        
        print(f"✓ compute_act_loss returned: {loss}")
        print(f"  Loss value: {float(loss):.6f}")
        print(f"  Loss shape: {loss.shape}")
        print(f"  Loss dtype: {loss.dtype}")
        
        # Verify loss is a scalar
        assert loss.shape == (), f"Expected scalar loss, got shape {loss.shape}"
        assert jnp.isfinite(loss), f"Loss is not finite: {loss}"
        
        return True
        
    except Exception as e:
        print(f"✗ compute_act_loss test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_compute_total_loss():
    """Test compute_total_loss function with ACTOutput."""
    print("\nTesting compute_total_loss...")
    
    try:
        from model.hrm.models import ACTOutput, HRMInnerCarry
        # Import directly from training.py
        import model.hrm.training as training_module
        compute_total_loss = training_module.compute_total_loss
        LossConfig = training_module.LossConfig
        
        # Create test data
        batch_size = 4
        seq_len = 32
        vocab_size = 1000
        max_steps = 6
        hidden_size = 128
        
        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, 8)
        
        # Create ACTOutput
        lm_logits = jax.random.normal(keys[0], (batch_size, seq_len, vocab_size)) * 0.1
        q_halt_logits = jax.random.normal(keys[1], (batch_size, max_steps)) * 0.5
        q_continue_logits = jax.random.normal(keys[2], (batch_size, max_steps)) * 0.5
        q_targets = jax.random.uniform(keys[3], (batch_size, max_steps), minval=0.0, maxval=1.0)
        step_count = jnp.array([3, 5, 2, 6])
        
        # Create final carry
        final_carry = HRMInnerCarry(
            z_H=jax.random.normal(keys[4], (batch_size, seq_len, hidden_size)),
            z_L=jax.random.normal(keys[5], (batch_size, seq_len, hidden_size))
        )
        
        act_output = ACTOutput(
            lm_logits=lm_logits,
            q_halt_logits=q_halt_logits,
            q_continue_logits=q_continue_logits,
            q_targets=q_targets,
            step_count=step_count,
            final_carry=final_carry
        )
        
        # Create batch with targets
        batch = {
            "inputs": jax.random.randint(keys[6], (batch_size, seq_len), 0, vocab_size),
            "targets": jax.random.randint(keys[7], (batch_size, seq_len), 0, vocab_size)
        }
        
        # Create loss config
        loss_config = LossConfig(
            lm_weight=1.0,
            act_weight=0.1,
            deep_supervision_weight=0.5,
            q_target_discount=0.95,
            label_smoothing=0.0
        )
        
        # Test the function
        total_loss, metrics = compute_total_loss(act_output, batch, loss_config)
        
        print(f"✓ compute_total_loss returned successfully")
        print(f"  Total loss: {float(total_loss):.6f}")
        print(f"  LM loss: {metrics.lm_loss:.6f}")
        print(f"  ACT loss: {metrics.act_loss:.6f}")
        print(f"  Deep supervision loss: {metrics.deep_supervision_loss:.6f}")
        print(f"  Mean steps: {metrics.mean_steps:.2f}")
        print(f"  LM accuracy: {metrics.lm_accuracy:.4f}")
        
        # Verify outputs
        assert total_loss.shape == (), f"Expected scalar loss, got shape {total_loss.shape}"
        assert jnp.isfinite(total_loss), f"Total loss is not finite: {total_loss}"
        assert isinstance(metrics.total_loss, float), f"Expected float metrics, got {type(metrics.total_loss)}"
        
        return True
        
    except Exception as e:
        print(f"✗ compute_total_loss test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_jax_tracing():
    """Test that functions work correctly under JAX transformations."""
    print("\nTesting JAX tracing compatibility...")
    
    try:
        from model.hrm.models import compute_act_loss
        
        # Create a simple function that uses compute_act_loss
        @jax.jit
        def traced_act_loss(q_halt, q_continue, q_targets, mask):
            return compute_act_loss(q_halt, q_continue, q_targets, mask)
        
        # Test data
        batch_size = 2
        max_steps = 4
        
        key = jax.random.PRNGKey(123)
        key1, key2, key3 = jax.random.split(key, 3)
        
        q_halt = jax.random.normal(key1, (batch_size, max_steps))
        q_continue = jax.random.normal(key2, (batch_size, max_steps))
        q_targets = jax.random.uniform(key3, (batch_size, max_steps))
        mask = jnp.ones((batch_size, max_steps), dtype=bool)
        
        # Test JIT compilation and execution
        loss = traced_act_loss(q_halt, q_continue, q_targets, mask)
        
        print(f"✓ JAX JIT compilation successful")
        print(f"  Traced loss: {float(loss):.6f}")
        
        # Test gradient computation
        grad_fn = jax.grad(lambda q: traced_act_loss(q, q_continue, q_targets, mask))
        grads = grad_fn(q_halt)
        
        print(f"✓ Gradient computation successful")
        print(f"  Gradient shape: {grads.shape}")
        print(f"  Gradient norm: {float(jnp.linalg.norm(grads)):.6f}")
        
        return True
        
    except Exception as e:
        print(f"✗ JAX tracing test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing HRM Loss Functions")
    print("=" * 60)
    
    tests = [
        ("Import Test", test_imports),
        ("compute_act_loss Test", test_compute_act_loss),
        ("compute_total_loss Test", test_compute_total_loss),
        ("JAX Tracing Test", test_jax_tracing)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"✗ {test_name} crashed: {e}")
            results.append((test_name, False))
    
    print("\n" + "=" * 60)
    print("Test Results Summary")
    print("=" * 60)
    
    all_passed = True
    for test_name, success in results:
        status = "PASS" if success else "FAIL"
        print(f"{test_name:25} {status}")
        if not success:
            all_passed = False
    
    print("=" * 60)
    if all_passed:
        print("🎉 All tests passed! The loss functions are working correctly.")
    else:
        print("❌ Some tests failed. Please check the errors above.")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)