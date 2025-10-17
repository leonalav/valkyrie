"""Unit tests for boundary cases and edge conditions.

This module tests edge cases, boundary conditions, and error handling
in the BigBird attention implementation to ensure robustness.
"""

import jax
import jax.numpy as jnp
import pytest
from unittest.mock import Mock, patch

from ..bigbird_attention import (
    BigBirdAttention,
    get_rand_attn_plan_vectorized,
    bigbird_block_rand_mask_with_head_vectorized,
    create_attention_mask_from_input_mask,
    create_rand_mask_from_inputs,
    create_band_mask_from_inputs
)
from ..gryphon_utils import create_sparse_attention_mask, create_causal_mask, pad_to_block_size
from ..gryphon_config import GryphonConfig


class TestBoundaryConditions:
    """Test boundary conditions and edge cases."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = GryphonConfig(
            d_model=128,
            n_heads=4,
            block_size=32,
            max_seq_len=256,
            num_rand_blocks=2
        )
        
        self.attention = BigBirdAttention(self.config)
        self.rng_key = jax.random.PRNGKey(42)
    
    def test_minimum_sequence_length(self):
        """Test with minimum possible sequence length (one block)."""
        batch_size = 1
        seq_len = self.config.block_size  # Exactly one block
        
        hidden_states = jax.random.normal(
            self.rng_key, (batch_size, seq_len, self.config.d_model)
        )
        position_ids = jnp.arange(seq_len)[None, :]
        
        # Mock RoPE frequencies
        head_dim = self.config.d_model // self.config.n_heads
        cos_freqs = jnp.ones((self.config.max_seq_len, head_dim // 2))
        sin_freqs = jnp.zeros((self.config.max_seq_len, head_dim // 2))
        
        # Should work without errors
        output = self.attention(
            hidden_states=hidden_states,
            position_ids=position_ids,
            cos_freqs=cos_freqs,
            sin_freqs=sin_freqs,
            training=False
        )
        
        assert output.shape == hidden_states.shape
        assert not jnp.any(jnp.isnan(output))
    
    def test_maximum_sequence_length(self):
        """Test with maximum configured sequence length."""
        batch_size = 1
        seq_len = self.config.max_seq_len
        
        hidden_states = jax.random.normal(
            self.rng_key, (batch_size, seq_len, self.config.d_model)
        )
        position_ids = jnp.arange(seq_len)[None, :]
        
        # Mock RoPE frequencies
        head_dim = self.config.d_model // self.config.n_heads
        cos_freqs = jnp.ones((self.config.max_seq_len, head_dim // 2))
        sin_freqs = jnp.zeros((self.config.max_seq_len, head_dim // 2))
        
        # Should work without errors
        output = self.attention(
            hidden_states=hidden_states,
            position_ids=position_ids,
            cos_freqs=cos_freqs,
            sin_freqs=sin_freqs,
            training=False
        )
        
        assert output.shape == hidden_states.shape
        assert not jnp.any(jnp.isnan(output))
    
    def test_single_head_attention(self):
        """Test with single attention head."""
        config = GryphonConfig(
            d_model=64,
            n_heads=1,  # Single head
            block_size=32,
            max_seq_len=128,
            num_rand_blocks=2
        )
        
        attention = BigBirdAttention(config)
        
        batch_size = 2
        seq_len = 96
        
        hidden_states = jax.random.normal(
            self.rng_key, (batch_size, seq_len, config.d_model)
        )
        position_ids = jnp.arange(seq_len)[None, :].repeat(batch_size, axis=0)
        
        # Mock RoPE frequencies
        head_dim = config.d_model // config.n_heads
        cos_freqs = jnp.ones((config.max_seq_len, head_dim // 2))
        sin_freqs = jnp.zeros((config.max_seq_len, head_dim // 2))
        
        output = attention(
            hidden_states=hidden_states,
            position_ids=position_ids,
            cos_freqs=cos_freqs,
            sin_freqs=sin_freqs,
            training=False
        )
        
        assert output.shape == hidden_states.shape
        assert not jnp.any(jnp.isnan(output))
    
    def test_zero_random_blocks(self):
        """Test with zero random blocks (only local + global attention)."""
        config = GryphonConfig(
            d_model=128,
            n_heads=4,
            block_size=32,
            max_seq_len=256,
            num_rand_blocks=0  # No random blocks
        )
        
        attention = BigBirdAttention(config)
        
        batch_size = 2
        seq_len = 96
        
        hidden_states = jax.random.normal(
            self.rng_key, (batch_size, seq_len, config.d_model)
        )
        position_ids = jnp.arange(seq_len)[None, :].repeat(batch_size, axis=0)
        
        # Mock RoPE frequencies
        head_dim = config.d_model // config.n_heads
        cos_freqs = jnp.ones((config.max_seq_len, head_dim // 2))
        sin_freqs = jnp.zeros((config.max_seq_len, head_dim // 2))
        
        output = attention(
            hidden_states=hidden_states,
            position_ids=position_ids,
            cos_freqs=cos_freqs,
            sin_freqs=sin_freqs,
            training=False
        )
        
        assert output.shape == hidden_states.shape
        assert not jnp.any(jnp.isnan(output))
    
    def test_all_padding_sequence(self):
        """Test with sequence that is entirely padding."""
        batch_size = 2
        seq_len = 64
        
        hidden_states = jnp.zeros((batch_size, seq_len, self.config.d_model))
        position_ids = jnp.zeros((batch_size, seq_len), dtype=jnp.int32)
        
        # Create input mask with one sequence entirely padded
        input_mask = jnp.array([
            jnp.ones(seq_len),  # Valid sequence
            jnp.zeros(seq_len)  # Entirely padded
        ])
        
        # Mock RoPE frequencies
        head_dim = self.config.d_model // self.config.n_heads
        cos_freqs = jnp.ones((self.config.max_seq_len, head_dim // 2))
        sin_freqs = jnp.zeros((self.config.max_seq_len, head_dim // 2))
        
        output = self.attention(
            hidden_states=hidden_states,
            position_ids=position_ids,
            cos_freqs=cos_freqs,
            sin_freqs=sin_freqs,
            input_mask=input_mask,
            training=False
        )
        
        assert output.shape == hidden_states.shape
        assert not jnp.any(jnp.isnan(output))
        
        # Entirely padded sequence should produce zero output
        assert jnp.allclose(output[1], 0.0, atol=1e-6)
    
    def test_non_divisible_sequence_length(self):
        """Test with sequence length not divisible by block size."""
        batch_size = 2
        seq_len = 50  # Not divisible by block_size (32)
        
        hidden_states = jax.random.normal(
            self.rng_key, (batch_size, seq_len, self.config.d_model)
        )
        position_ids = jnp.arange(seq_len)[None, :].repeat(batch_size, axis=0)
        
        # Mock RoPE frequencies
        head_dim = self.config.d_model // self.config.n_heads
        cos_freqs = jnp.ones((self.config.max_seq_len, head_dim // 2))
        sin_freqs = jnp.zeros((self.config.max_seq_len, head_dim // 2))
        
        output = self.attention(
            hidden_states=hidden_states,
            position_ids=position_ids,
            cos_freqs=cos_freqs,
            sin_freqs=sin_freqs,
            training=False
        )
        
        # Output should be trimmed back to original length
        assert output.shape == (batch_size, seq_len, self.config.d_model)
        assert not jnp.any(jnp.isnan(output))
    
    def test_extreme_position_ids(self):
        """Test with extreme position ID values."""
        batch_size = 2
        seq_len = 64
        
        hidden_states = jax.random.normal(
            self.rng_key, (batch_size, seq_len, self.config.d_model)
        )
        
        # Test with very large position IDs
        position_ids = jnp.full((batch_size, seq_len), 10000)
        
        # Mock RoPE frequencies
        head_dim = self.config.d_model // self.config.n_heads
        cos_freqs = jnp.ones((self.config.max_seq_len, head_dim // 2))
        sin_freqs = jnp.zeros((self.config.max_seq_len, head_dim // 2))
        
        # Should handle extreme values gracefully (clipping)
        output = self.attention(
            hidden_states=hidden_states,
            position_ids=position_ids,
            cos_freqs=cos_freqs,
            sin_freqs=sin_freqs,
            training=False
        )
        
        assert output.shape == hidden_states.shape
        assert not jnp.any(jnp.isnan(output))
    
    def test_mask_creation_edge_cases(self):
        """Test mask creation functions with edge cases."""
        # Test with minimum dimensions
        min_seq_len = self.config.block_size
        input_mask = jnp.ones((1, min_seq_len))
        
        # Should not raise errors
        attn_mask = create_attention_mask_from_input_mask(input_mask)
        band_mask = create_band_mask_from_inputs(input_mask, self.config.block_size)
        rand_mask = create_rand_mask_from_inputs(
            input_mask, self.config.block_size, self.config.num_rand_blocks, self.rng_key
        )
        
        assert attn_mask.shape == (1, min_seq_len, min_seq_len)
        assert band_mask.shape == (1, min_seq_len, min_seq_len)
        assert rand_mask.shape == (1, min_seq_len, min_seq_len)
        
        # Test with single token sequences
        single_token_mask = jnp.ones((1, 1))
        
        # These should handle single token gracefully
        attn_mask_single = create_attention_mask_from_input_mask(single_token_mask)
        assert attn_mask_single.shape == (1, 1, 1)
        assert attn_mask_single[0, 0, 0] == 1
    
    def test_random_plan_edge_cases(self):
        """Test random attention plan generation edge cases."""
        block_size = 32
        
        # Test with very few blocks
        seq_len = 2 * block_size  # Only 2 blocks
        num_rand_blocks = 1
        
        plan = get_rand_attn_plan_vectorized(
            seq_len, block_size, num_rand_blocks, self.rng_key
        )
        
        assert plan.shape == (2, num_rand_blocks)
        assert jnp.all(plan >= 0)
        assert jnp.all(plan < 2)
        
        # Test with num_rand_blocks equal to num_blocks
        seq_len = 3 * block_size
        num_rand_blocks = 3  # Same as number of blocks
        
        plan = get_rand_attn_plan_vectorized(
            seq_len, block_size, num_rand_blocks, self.rng_key
        )
        
        assert plan.shape == (3, num_rand_blocks)
        assert jnp.all(plan >= 0)
        assert jnp.all(plan < 3)
    
    def test_memory_efficiency_large_batch(self):
        """Test memory efficiency with larger batch sizes."""
        # Use smaller dimensions to avoid OOM in tests
        config = GryphonConfig(
            d_model=64,
            n_heads=2,
            block_size=16,
            max_seq_len=128,
            num_rand_blocks=1
        )
        
        attention = BigBirdAttention(config)
        
        batch_size = 8  # Larger batch
        seq_len = 48
        
        hidden_states = jax.random.normal(
            self.rng_key, (batch_size, seq_len, config.d_model)
        )
        position_ids = jnp.arange(seq_len)[None, :].repeat(batch_size, axis=0)
        
        # Mock RoPE frequencies
        head_dim = config.d_model // config.n_heads
        cos_freqs = jnp.ones((config.max_seq_len, head_dim // 2))
        sin_freqs = jnp.zeros((config.max_seq_len, head_dim // 2))
        
        output = attention(
            hidden_states=hidden_states,
            position_ids=position_ids,
            cos_freqs=cos_freqs,
            sin_freqs=sin_freqs,
            training=False
        )
        
        assert output.shape == hidden_states.shape
        assert not jnp.any(jnp.isnan(output))
    
    def test_gradient_flow(self):
        """Test that gradients flow correctly through the attention mechanism."""
        batch_size = 2
        seq_len = 64
        
        def loss_fn(hidden_states):
            position_ids = jnp.arange(seq_len)[None, :].repeat(batch_size, axis=0)
            
            # Mock RoPE frequencies
            head_dim = self.config.d_model // self.config.n_heads
            cos_freqs = jnp.ones((self.config.max_seq_len, head_dim // 2))
            sin_freqs = jnp.zeros((self.config.max_seq_len, head_dim // 2))
            
            output = self.attention(
                hidden_states=hidden_states,
                position_ids=position_ids,
                cos_freqs=cos_freqs,
                sin_freqs=sin_freqs,
                training=True
            )
            return jnp.mean(output ** 2)
        
        hidden_states = jax.random.normal(
            self.rng_key, (batch_size, seq_len, self.config.d_model)
        )
        
        # Compute gradients
        loss, grads = jax.value_and_grad(loss_fn)(hidden_states)
        
        assert jnp.isfinite(loss)
        assert not jnp.any(jnp.isnan(grads))
        assert grads.shape == hidden_states.shape
        
        # Gradients should not be all zero (indicating gradient flow)
        assert not jnp.allclose(grads, 0.0, atol=1e-8)


class TestErrorHandling:
    """Test error handling and validation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = GryphonConfig(
            d_model=128,
            n_heads=4,
            block_size=32,
            max_seq_len=256,
            num_rand_blocks=2
        )
        
        self.attention = BigBirdAttention(self.config)
        self.rng_key = jax.random.PRNGKey(42)
    
    def test_invalid_input_shapes(self):
        """Test error handling for invalid input shapes."""
        batch_size = 2
        seq_len = 64
        
        # Test wrong hidden_states dimensions
        with pytest.raises(AssertionError):
            invalid_hidden = jax.random.normal(
                self.rng_key, (batch_size, seq_len)  # Missing d_model dimension
            )
            position_ids = jnp.arange(seq_len)[None, :].repeat(batch_size, axis=0)
            
            head_dim = self.config.d_model // self.config.n_heads
            cos_freqs = jnp.ones((self.config.max_seq_len, head_dim // 2))
            sin_freqs = jnp.zeros((self.config.max_seq_len, head_dim // 2))
            
            self.attention(
                hidden_states=invalid_hidden,
                position_ids=position_ids,
                cos_freqs=cos_freqs,
                sin_freqs=sin_freqs,
                training=False
            )
    
    def test_mismatched_batch_dimensions(self):
        """Test error handling for mismatched batch dimensions."""
        batch_size = 2
        seq_len = 64
        
        hidden_states = jax.random.normal(
            self.rng_key, (batch_size, seq_len, self.config.d_model)
        )
        # Wrong batch size for position_ids
        position_ids = jnp.arange(seq_len)[None, :].repeat(3, axis=0)  # batch_size=3
        
        head_dim = self.config.d_model // self.config.n_heads
        cos_freqs = jnp.ones((self.config.max_seq_len, head_dim // 2))
        sin_freqs = jnp.zeros((self.config.max_seq_len, head_dim // 2))
        
        with pytest.raises(AssertionError, match="position_ids batch dimension mismatch"):
            self.attention(
                hidden_states=hidden_states,
                position_ids=position_ids,
                cos_freqs=cos_freqs,
                sin_freqs=sin_freqs,
                training=False
            )
    
    def test_invalid_mask_parameters(self):
        """Test error handling for invalid mask parameters."""
        # Test invalid block_size
        with pytest.raises(AssertionError, match="block_size must be positive"):
            create_sparse_attention_mask(seq_len=128, block_size=0)
        
        with pytest.raises(AssertionError, match="block_size must be positive"):
            create_sparse_attention_mask(seq_len=128, block_size=-1)
        
        # Test invalid seq_len
        with pytest.raises(AssertionError, match="seq_len must be positive"):
            create_sparse_attention_mask(seq_len=0, block_size=32)
    
    def test_configuration_validation(self):
        """Test validation of configuration parameters."""
        # Test invalid n_heads (not divisor of d_model)
        with pytest.raises(AssertionError):
            invalid_config = GryphonConfig(
                d_model=127,  # Prime number
                n_heads=8,    # Doesn't divide evenly
                block_size=32,
                max_seq_len=256,
                num_rand_blocks=2
            )
            BigBirdAttention(invalid_config)
    
    def test_rope_frequency_shape_validation(self):
        """Test validation of RoPE frequency shapes."""
        batch_size = 2
        seq_len = 64
        
        hidden_states = jax.random.normal(
            self.rng_key, (batch_size, seq_len, self.config.d_model)
        )
        position_ids = jnp.arange(seq_len)[None, :].repeat(batch_size, axis=0)
        
        # Wrong shape for cos_freqs
        head_dim = self.config.d_model // self.config.n_heads
        cos_freqs = jnp.ones((self.config.max_seq_len, head_dim))  # Wrong last dim
        sin_freqs = jnp.zeros((self.config.max_seq_len, head_dim // 2))
        
        with pytest.raises((AssertionError, ValueError)):
            self.attention(
                hidden_states=hidden_states,
                position_ids=position_ids,
                cos_freqs=cos_freqs,
                sin_freqs=sin_freqs,
                training=False
            )


if __name__ == "__main__":
    pytest.main([__file__])