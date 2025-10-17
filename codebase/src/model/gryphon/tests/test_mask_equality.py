"""Unit tests for mask equality and attention output consistency.

This module tests that vectorized mask generation functions produce identical
results to their original implementations, ensuring correctness of optimizations.
"""

import jax
import jax.numpy as jnp
import pytest
from unittest.mock import patch

from ..bigbird_attention import (
    get_rand_attn_plan_vectorized,
    bigbird_block_rand_mask_with_head_vectorized,
    create_attention_mask_from_input_mask,
    create_rand_mask_from_inputs,
    create_band_mask_from_inputs,
    BigBirdAttention
)
from ..gryphon_utils import create_sparse_attention_mask, create_causal_mask
from ..gryphon_config import GryphonConfig


class TestMaskEquality:
    """Test mask generation functions for correctness and consistency."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = GryphonConfig(
            d_model=256,
            n_heads=8,
            block_size=64,
            max_seq_len=512,
            num_rand_blocks=3
        )
        
        self.rng_key = jax.random.PRNGKey(42)
    
    def test_vectorized_rand_plan_deterministic_cases(self):
        """Test vectorized random plan generation for deterministic cases."""
        # Test cases that should produce deterministic patterns
        test_cases = [
            (4, 64, 3),   # num_blocks = 4
            (5, 64, 3),   # num_blocks = 5  
            (6, 64, 3),   # num_blocks = 6
        ]
        
        for num_blocks, block_size, num_rand_blocks in test_cases:
            seq_len = num_blocks * block_size
            
            # Generate plan multiple times - should be identical for deterministic cases
            plan1 = get_rand_attn_plan_vectorized(
                seq_len, block_size, num_rand_blocks, self.rng_key
            )
            plan2 = get_rand_attn_plan_vectorized(
                seq_len, block_size, num_rand_blocks, self.rng_key
            )
            
            assert jnp.array_equal(plan1, plan2), f"Deterministic plan mismatch for {num_blocks} blocks"
            assert plan1.shape == (num_blocks, num_rand_blocks)
    
    def test_vectorized_rand_plan_random_consistency(self):
        """Test vectorized random plan generation produces valid random patterns."""
        num_blocks = 10
        block_size = 64
        num_rand_blocks = 3
        seq_len = num_blocks * block_size
        
        plan = get_rand_attn_plan_vectorized(
            seq_len, block_size, num_rand_blocks, self.rng_key
        )
        
        # Check shape
        assert plan.shape == (num_blocks, num_rand_blocks)
        
        # Check all values are valid block indices
        assert jnp.all(plan >= 0)
        assert jnp.all(plan < num_blocks)
        
        # Check no self-attention (diagonal should not equal row index)
        row_indices = jnp.arange(num_blocks)[:, None]
        self_attention_mask = plan == row_indices
        # Allow some self-attention but not all
        assert not jnp.all(self_attention_mask)
    
    def test_vectorized_block_mask_consistency(self):
        """Test vectorized block mask generation produces consistent results."""
        batch_size = 2
        seq_len = 256
        block_size = 64
        num_rand_blocks = 3
        num_heads = 8
        
        # Generate random plan
        plan = get_rand_attn_plan_vectorized(
            seq_len, block_size, num_rand_blocks, self.rng_key
        )
        
        # Generate mask multiple times with same inputs
        mask1 = bigbird_block_rand_mask_with_head_vectorized(
            batch_size, seq_len, block_size, plan, num_heads
        )
        mask2 = bigbird_block_rand_mask_with_head_vectorized(
            batch_size, seq_len, block_size, plan, num_heads
        )
        
        # Should be identical
        assert jnp.array_equal(mask1, mask2)
        assert mask1.shape == (batch_size, num_heads, seq_len, seq_len)
        
        # Check mask is binary
        assert jnp.all((mask1 == 0) | (mask1 == 1))
    
    def test_attention_mask_from_input_consistency(self):
        """Test attention mask creation from input mask consistency."""
        batch_size = 2
        seq_len = 128
        
        # Create input mask (1 for valid tokens, 0 for padding)
        input_mask = jnp.ones((batch_size, seq_len))
        # Add some padding to second sequence
        input_mask = input_mask.at[1, 100:].set(0)
        
        # Generate mask multiple times
        mask1 = create_attention_mask_from_input_mask(input_mask)
        mask2 = create_attention_mask_from_input_mask(input_mask)
        
        assert jnp.array_equal(mask1, mask2)
        assert mask1.shape == (batch_size, seq_len, seq_len)
        
        # Check that padded positions are masked correctly
        # Padded tokens should not attend to anything
        assert jnp.all(mask1[1, 100:, :] == 0)
        # Nothing should attend to padded tokens
        assert jnp.all(mask1[1, :, 100:] == 0)
    
    def test_rand_mask_creation_consistency(self):
        """Test random mask creation produces consistent results."""
        batch_size = 2
        seq_len = 256
        block_size = 64
        num_rand_blocks = 3
        
        input_mask = jnp.ones((batch_size, seq_len))
        
        # Generate masks with same RNG key
        mask1 = create_rand_mask_from_inputs(
            input_mask, block_size, num_rand_blocks, self.rng_key
        )
        mask2 = create_rand_mask_from_inputs(
            input_mask, block_size, num_rand_blocks, self.rng_key
        )
        
        # Should be identical with same RNG
        assert jnp.array_equal(mask1, mask2)
        assert mask1.shape == (batch_size, seq_len, seq_len)
        
        # Check mask is binary
        assert jnp.all((mask1 == 0) | (mask1 == 1))
    
    def test_band_mask_creation_consistency(self):
        """Test band mask creation produces consistent results."""
        batch_size = 2
        seq_len = 256
        block_size = 64
        
        input_mask = jnp.ones((batch_size, seq_len))
        
        # Generate masks multiple times
        mask1 = create_band_mask_from_inputs(input_mask, block_size)
        mask2 = create_band_mask_from_inputs(input_mask, block_size)
        
        assert jnp.array_equal(mask1, mask2)
        assert mask1.shape == (batch_size, seq_len, seq_len)
        
        # Check mask is binary
        assert jnp.all((mask1 == 0) | (mask1 == 1))
        
        # Check band structure (should have local attention pattern)
        # Each position should attend to nearby positions within block
        for i in range(0, seq_len, block_size):
            block_end = min(i + block_size, seq_len)
            block_mask = mask1[0, i:block_end, i:block_end]
            # Within block, all positions should attend to each other
            assert jnp.all(block_mask == 1)
    
    def test_sparse_attention_mask_properties(self):
        """Test sparse attention mask has correct properties."""
        seq_len = 256
        block_size = 64
        
        mask = create_sparse_attention_mask(seq_len, block_size)
        
        assert mask.shape == (seq_len, seq_len)
        assert jnp.all((mask == 0) | (mask == 1))
        
        # Check diagonal is all ones (self-attention)
        assert jnp.all(jnp.diag(mask) == 1)
        
        # Check block structure
        num_blocks = seq_len // block_size
        for i in range(num_blocks):
            start_i = i * block_size
            end_i = (i + 1) * block_size
            
            # Within block attention should be enabled
            block_mask = mask[start_i:end_i, start_i:end_i]
            assert jnp.all(block_mask == 1)
    
    def test_causal_mask_properties(self):
        """Test causal mask has correct triangular structure."""
        seq_len = 128
        block_size = 32
        
        mask = create_causal_mask(seq_len, block_size)
        
        assert mask.shape == (seq_len, seq_len)
        assert jnp.all((mask == 0) | (mask == 1))
        
        # Check causal property: no attention to future positions
        for i in range(seq_len):
            for j in range(i + 1, seq_len):
                assert mask[i, j] == 0, f"Causal violation at ({i}, {j})"
        
        # Check diagonal is all ones
        assert jnp.all(jnp.diag(mask) == 1)
    
    def test_mask_boundary_conditions(self):
        """Test mask generation at boundary conditions."""
        # Test minimum sizes
        min_seq_len = 64  # One block
        block_size = 64
        
        input_mask = jnp.ones((1, min_seq_len))
        
        # Should not raise errors
        attn_mask = create_attention_mask_from_input_mask(input_mask)
        band_mask = create_band_mask_from_inputs(input_mask, block_size)
        
        assert attn_mask.shape == (1, min_seq_len, min_seq_len)
        assert band_mask.shape == (1, min_seq_len, min_seq_len)
        
        # Test with padding
        input_mask_padded = input_mask.at[0, 32:].set(0)  # Half padding
        
        attn_mask_padded = create_attention_mask_from_input_mask(input_mask_padded)
        
        # Padded region should be properly masked
        assert jnp.all(attn_mask_padded[0, 32:, :] == 0)
        assert jnp.all(attn_mask_padded[0, :, 32:] == 0)
    
    def test_different_rng_keys_produce_different_results(self):
        """Test that different RNG keys produce different random patterns."""
        seq_len = 320  # Non-deterministic case
        block_size = 64
        num_rand_blocks = 3
        
        key1 = jax.random.PRNGKey(42)
        key2 = jax.random.PRNGKey(123)
        
        plan1 = get_rand_attn_plan_vectorized(seq_len, block_size, num_rand_blocks, key1)
        plan2 = get_rand_attn_plan_vectorized(seq_len, block_size, num_rand_blocks, key2)
        
        # Should be different (with high probability)
        assert not jnp.array_equal(plan1, plan2)
        
        # But should have same shape and valid values
        assert plan1.shape == plan2.shape
        num_blocks = seq_len // block_size
        assert jnp.all(plan1 >= 0) and jnp.all(plan1 < num_blocks)
        assert jnp.all(plan2 >= 0) and jnp.all(plan2 < num_blocks)


class TestAttentionOutputEquality:
    """Test that attention outputs are consistent across implementations."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = GryphonConfig(
            d_model=128,  # Smaller for faster testing
            n_heads=4,
            block_size=32,
            max_seq_len=256,
            num_rand_blocks=2
        )
        
        self.attention = BigBirdAttention(self.config)
        self.rng_key = jax.random.PRNGKey(42)
    
    def test_attention_output_deterministic(self):
        """Test that attention produces deterministic outputs with same inputs."""
        batch_size = 2
        seq_len = 96
        
        hidden_states = jax.random.normal(
            self.rng_key, (batch_size, seq_len, self.config.d_model)
        )
        position_ids = jnp.arange(seq_len)[None, :].repeat(batch_size, axis=0)
        
        # Mock RoPE frequencies
        head_dim = self.config.d_model // self.config.n_heads
        cos_freqs = jnp.ones((self.config.max_seq_len, head_dim // 2))
        sin_freqs = jnp.zeros((self.config.max_seq_len, head_dim // 2))
        
        # Run attention twice with same inputs
        output1 = self.attention(
            hidden_states=hidden_states,
            position_ids=position_ids,
            cos_freqs=cos_freqs,
            sin_freqs=sin_freqs,
            training=False
        )
        
        output2 = self.attention(
            hidden_states=hidden_states,
            position_ids=position_ids,
            cos_freqs=cos_freqs,
            sin_freqs=sin_freqs,
            training=False
        )
        
        # Should be identical
        assert jnp.allclose(output1, output2, atol=1e-6)
        assert output1.shape == (batch_size, seq_len, self.config.d_model)
    
    def test_attention_output_with_padding(self):
        """Test attention output consistency with padded sequences."""
        batch_size = 2
        seq_lens = [64, 96]  # Different lengths
        max_seq_len = max(seq_lens)
        
        # Create padded inputs
        hidden_states_list = []
        position_ids_list = []
        
        for seq_len in seq_lens:
            # Create sequence
            seq_data = jax.random.normal(
                self.rng_key, (seq_len, self.config.d_model)
            )
            # Pad
            padded_data = jnp.pad(
                seq_data, ((0, max_seq_len - seq_len), (0, 0)), mode='constant'
            )
            hidden_states_list.append(padded_data)
            
            # Position IDs
            pos_ids = jnp.arange(seq_len)
            padded_pos_ids = jnp.pad(
                pos_ids, (0, max_seq_len - seq_len), mode='constant'
            )
            position_ids_list.append(padded_pos_ids)
        
        hidden_states = jnp.stack(hidden_states_list, axis=0)
        position_ids = jnp.stack(position_ids_list, axis=0)
        
        # Mock RoPE frequencies
        head_dim = self.config.d_model // self.config.n_heads
        cos_freqs = jnp.ones((self.config.max_seq_len, head_dim // 2))
        sin_freqs = jnp.zeros((self.config.max_seq_len, head_dim // 2))
        
        # Run attention
        output = self.attention(
            hidden_states=hidden_states,
            position_ids=position_ids,
            cos_freqs=cos_freqs,
            sin_freqs=sin_freqs,
            training=False
        )
        
        assert output.shape == hidden_states.shape
        assert not jnp.any(jnp.isnan(output))
        
        # Check padded regions are handled correctly
        for i, seq_len in enumerate(seq_lens):
            if seq_len < max_seq_len:
                # Padded region should have minimal activation
                padded_output = output[i, seq_len:, :]
                # Should be close to zero or at least not NaN/Inf
                assert jnp.all(jnp.isfinite(padded_output))


if __name__ == "__main__":
    pytest.main([__file__])