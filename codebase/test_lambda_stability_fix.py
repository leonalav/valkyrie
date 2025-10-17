#!/usr/bin/env python3
"""
Test script to verify the Lambda stability fix for ValkyrieS5.
This script tests that all eigenvalues have Re(λ) ≤ 0 after the fix.
"""

import jax
import jax.numpy as jnp
import numpy as np
from src.model.s5 import ValkyrieS5
from src.model.modules import ValkyrieConfig

def test_lambda_stability_fix():
    """Test that the Lambda parameterization fix ensures all eigenvalues are stable."""
    print("🧪 Testing Lambda stability fix for ValkyrieS5...")
    
    # Initialize model with corrected parameterization
    config = ValkyrieConfig(d_model=64)  # Ensure d_model is compatible
    model = ValkyrieS5(config=config, state_dim=64)
    
    # Initialize parameters
    rng = jax.random.PRNGKey(42)
    dummy_input = jnp.ones((1, 10, config.d_model))  # (batch, seq_len, d_model)
    params = model.init(rng, dummy_input, training=True)
    
    print("✅ Model initialized successfully")
    
    # Apply the parameters to get the actual Lambda values using bound method
    bound_method = model.bind(params)
    Lambda_actual, _, _ = bound_method._get_complex_params()
    
    print(f"📊 Lambda shape: {Lambda_actual.shape}")
    print(f"📊 Lambda dtype: {Lambda_actual.dtype}")
    
    # Analyze Lambda real parts
    Lambda_real_parts = jnp.real(Lambda_actual)
    Lambda_imag_parts = jnp.imag(Lambda_actual)
    
    print(f"📊 Lambda real part range: [{jnp.min(Lambda_real_parts):.6f}, {jnp.max(Lambda_real_parts):.6f}]")
    print(f"📊 Lambda imag part range: [{jnp.min(Lambda_imag_parts):.6f}, {jnp.max(Lambda_imag_parts):.6f}]")
    
    # Check stability: all real parts should be ≤ 0
    stable_eigenvalues = jnp.sum(Lambda_real_parts <= 0)
    total_eigenvalues = len(Lambda_real_parts)
    stability_ratio = stable_eigenvalues / total_eigenvalues
    
    print(f"🎯 Stable eigenvalues (Re(λ) ≤ 0): {stable_eigenvalues}/{total_eigenvalues}")
    print(f"🎯 Stability ratio: {stability_ratio:.2%}")
    
    # Test forward pass to ensure the fix doesn't break functionality
    print("\n🔄 Testing forward pass with fixed parameterization...")
    try:
        output, final_state = model.apply(params, dummy_input, training=True)
        print(f"✅ Forward pass successful! Output shape: {output.shape}")
        print(f"📊 Output range: [{jnp.min(output):.6f}, {jnp.max(output):.6f}]")
        print(f"📊 Final state shape: {final_state.shape}")
        
        # Check for NaN or Inf values
        if jnp.any(jnp.isnan(output)) or jnp.any(jnp.isinf(output)):
            print("❌ Output contains NaN or Inf values!")
            return False
        else:
            print("✅ Output is numerically stable (no NaN/Inf)")
            
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        return False
    
    # Verify the fix worked
    if stability_ratio == 1.0:
        print("\n🎉 SUCCESS: All eigenvalues are stable (Re(λ) ≤ 0)!")
        print("🎉 The Lambda parameterization fix is working correctly!")
        return True
    else:
        print(f"\n❌ FAILURE: Only {stability_ratio:.2%} of eigenvalues are stable")
        print("❌ The Lambda parameterization fix needs further investigation")
        return False

if __name__ == "__main__":
    success = test_lambda_stability_fix()
    if success:
        print("\n✅ All tests passed! The S5 model is now stable.")
    else:
        print("\n❌ Tests failed! The S5 model still has stability issues.")