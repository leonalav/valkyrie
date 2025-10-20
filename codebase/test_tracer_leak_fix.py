#!/usr/bin/env python3
"""
Test script to verify that the tracer leak in HRM model has been fixed.
"""

import jax
import jax.numpy as jnp
from src.model.hrm.models.hrm_inner import HRMInner, HRMInnerCarry

def test_hrm_tracer_leak_fix():
    """Test that HRM model can be initialized and run without tracer leaks."""
    print("Testing HRM model tracer leak fix...")
    
    # Model configuration - using correct HRMInner parameters
    config = {
        'vocab_size': 1000,
        'hidden_size': 128,
        'seq_len': 64,
        'puzzle_emb_ndim': 32,
        'num_puzzle_identifiers': 10,
        'H_cycles': 3,
        'L_cycles': 3,
        'H_layers': 2,  # Reduced for testing
        'L_layers': 2,  # Reduced for testing
        'num_heads': 8,
        'num_key_value_heads': 8,
        'intermediate_size': 256,
        'eps': 1e-5,
        'pos_encodings': "rope",
        'rope_theta': 10000.0,
        'dtype': jnp.bfloat16,
        'param_dtype': jnp.float32
    }
    
    # Create model
    model = HRMInner(**config)
    
    # Create dummy input
    batch_size = 2
    seq_len = config['seq_len']
    vocab_size = config['vocab_size']
    hidden_size = config['hidden_size']
    
    key = jax.random.PRNGKey(42)
    key1, key2, key3 = jax.random.split(key, 3)
    
    # Create batch dict as expected by HRMInner
    batch = {
        'inputs': jax.random.randint(key1, (batch_size, seq_len), 0, vocab_size),
        'puzzle_identifiers': jax.random.randint(key2, (batch_size,), 0, config['num_puzzle_identifiers'])
    }
    
    # Create initial carry state with correct total sequence length
    # Total seq len = seq_len + puzzle_emb_len
    puzzle_emb_len = 0
    if config['puzzle_emb_ndim'] > 0:
        puzzle_emb_len = -(config['puzzle_emb_ndim'] // -config['hidden_size'])  # ceil div
    
    total_seq_len = config['seq_len'] + puzzle_emb_len
    
    carry = HRMInnerCarry(
        z_H=jax.random.normal(key3, (batch_size, total_seq_len, hidden_size)),
        z_L=jax.random.normal(key3, (batch_size, total_seq_len, hidden_size))
    )
    
    print("Initializing model...")
    try:
        # Initialize model parameters
        init_key = jax.random.PRNGKey(0)
        variables = model.init(init_key, carry, batch)
        print("✓ Model initialization successful")
        
        # Test forward pass
        print("Testing forward pass...")
        new_carry, lm_logits, q_logits = model.apply(variables, carry, batch)
        print(f"✓ Forward pass successful")
        print(f"  - LM logits shape: {lm_logits.shape}")
        print(f"  - Q logits shapes: {q_logits[0].shape}, {q_logits[1].shape}")
        print(f"  - New carry z_H shape: {new_carry.z_H.shape}")
        print(f"  - New carry z_L shape: {new_carry.z_L.shape}")
        
        # Test JIT compilation (this would fail with tracer leaks)
        print("Testing JIT compilation...")
        jit_apply = jax.jit(model.apply)
        jit_new_carry, jit_lm_logits, jit_q_logits = jit_apply(variables, carry, batch)
        print(f"✓ JIT compilation successful")
        print(f"  - JIT LM logits shape: {jit_lm_logits.shape}")
        
        print("\n🎉 All tests passed! Tracer leak has been fixed.")
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_hrm_tracer_leak_fix()
    exit(0 if success else 1)