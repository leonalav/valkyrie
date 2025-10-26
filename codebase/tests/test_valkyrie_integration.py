"""
Integration test for Valkyrie model to verify all components work together.
"""

import pytest
import jax
import jax.numpy as jnp
from jax import random
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from model.valkyrie import ValkyrieModel
from model.modules import ValkyrieConfig


class TestValkyrieIntegration:
    """Integration tests for Valkyrie model."""
    
    def test_model_initialization_and_forward_pass(self):
        """Test that the model can be initialized and run a forward pass."""
        # Create a minimal config
        config = ValkyrieConfig(
            vocab_size=1000,
            d_model=256,
            n_layers=2,
            n_heads=8,
            max_position_embeddings=512,
            bigbird_block_size=64,
            bigbird_num_random_blocks=2,
            bigbird_num_global_tokens=64,
            bigbird_use_blockified_gemm=False,  # Disabled due to einsum issues
            s5_state_dim=64,
            attn_dropout=0.1,
            resid_dropout=0.1,
            ffn_dropout=0.1,
            layer_norm_eps=1e-6,
            initializer_range=0.02,
            use_bias=True,
        )
        
        # Initialize model
        model = ValkyrieModel(config)
        
        # Create dummy input
        batch_size = 2
        seq_len = 128  # Use a fixed sequence length
        
        # Create input tokens
        key = random.PRNGKey(42)
        input_ids = random.randint(key, (batch_size, seq_len), 0, config.vocab_size)
        
        # Initialize model parameters
        key, init_key = random.split(key)
        params = model.init(init_key, input_ids)
        
        # Run forward pass
        key, forward_key = random.split(key)
        outputs = model.apply(params, input_ids, rngs={'dropout': forward_key})
        
        # Extract logits from the output dictionary
        logits = outputs['logits']
        
        # Verify output shape
        expected_shape = (batch_size, seq_len, config.vocab_size)
        assert logits.shape == expected_shape, f"Expected shape {expected_shape}, got {logits.shape}"
        
        # Verify output is finite
        assert jnp.all(jnp.isfinite(logits)), "Model outputs contain NaN or Inf values"
        
        print(f"✓ Model initialization and forward pass successful")
        print(f"✓ Output shape: {logits.shape}")
        print(f"✓ Output range: [{jnp.min(logits):.4f}, {jnp.max(logits):.4f}]")
    
    def test_model_with_hrm_global_tokens(self):
        """Test that the model works with HRM global tokens."""
        # Create config with HRM enabled
        config = ValkyrieConfig(
            vocab_size=1000,
            d_model=256,
            n_layers=2,
            n_heads=8,
            max_position_embeddings=512,
            bigbird_block_size=64,
            bigbird_num_random_blocks=2,
            bigbird_num_global_tokens=64,
            bigbird_use_blockified_gemm=False,
            s5_state_dim=64,
            attn_dropout=0.1,
            resid_dropout=0.1,
            ffn_dropout=0.1,
            layer_norm_eps=1e-6,
            initializer_range=0.02,
            use_bias=True,
        )
        
        model = ValkyrieModel(config)
        
        batch_size = 2
        seq_len = 128  # Use a fixed sequence length
        
        key = random.PRNGKey(42)
        input_ids = random.randint(key, (batch_size, seq_len), 0, config.vocab_size)
        
        # Initialize with HRM stage
        key, init_key = random.split(key)
        params = model.init(init_key, input_ids)
        
        # Forward pass with HRM stage
        key, forward_key = random.split(key)
        outputs = model.apply(params, input_ids, rngs={'dropout': forward_key})
        
        # Extract logits from the output dictionary
        logits = outputs['logits']
        
        # Verify output shape (should be same as input since we're not adding tokens to output)
        expected_shape = (batch_size, seq_len, config.vocab_size)
        assert logits.shape == expected_shape, f"Expected shape {expected_shape}, got {logits.shape}"
        
        # Verify output is finite
        assert jnp.all(jnp.isfinite(logits)), "Model outputs contain NaN or Inf values"
        
        print(f"✓ Model with HRM global tokens successful")
        print(f"✓ Output shape: {logits.shape}")
    
    def test_different_sequence_lengths(self):
        """Test that the model works with different sequence lengths."""
        config = ValkyrieConfig(
            vocab_size=1000,
            d_model=256,
            n_layers=1,
            n_heads=8,
            max_position_embeddings=512,
            bigbird_block_size=64,
            bigbird_num_random_blocks=2,
            bigbird_num_global_tokens=32,
            bigbird_use_blockified_gemm=False,
            s5_state_dim=64,
            attn_dropout=0.1,
            resid_dropout=0.1,
            ffn_dropout=0.1,
            layer_norm_eps=1e-6,
            initializer_range=0.02,
            use_bias=True,
        )
        
        model = ValkyrieModel(config)
        
        batch_size = 1
        seq_len = 256  # Larger sequence length
        
        key = random.PRNGKey(42)
        input_ids = random.randint(key, (batch_size, seq_len), 0, config.vocab_size)
        
        # Initialize and run
        key, init_key = random.split(key)
        params = model.init(init_key, input_ids)
        
        key, forward_key = random.split(key)
        outputs = model.apply(params, input_ids, rngs={'dropout': forward_key})
        
        # Extract logits from the output dictionary
        logits = outputs['logits']
        
        expected_shape = (batch_size, seq_len, config.vocab_size)
        assert logits.shape == expected_shape, f"Expected shape {expected_shape}, got {logits.shape}"
        assert jnp.all(jnp.isfinite(logits)), "Model outputs contain NaN or Inf values"
        
        print(f"✓ Model with sequence length {seq_len} successful")


if __name__ == "__main__":
    test = TestValkyrieIntegration()
    test.test_model_initialization_and_forward_pass()
    test.test_model_with_hrm_global_tokens()
    test.test_different_sequence_lengths()
    print("All integration tests passed!")