"""Unit tests for RoPE padding logic with multi-length batches.

This module tests the robustness of Rotary Position Embedding (RoPE) padding
when handling batches with sequences of different lengths.
"""

import jax
import jax.numpy as jnp
import pytest
from unittest.mock import Mock

from ..bigbird_attention import BigBirdSparseAttention
from ..gryphon_config import GryphonConfig
from ...modules import precompute_rope_freqs


class TestRoPEPadding:
    """Test RoPE padding logic with multi-length batches."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = GryphonConfig(
            d_model=256,
            n_heads=8,
            block_size=64,
            max_position_embeddings=512,
            rope_theta=10000.0
        )
        
        # Initialize the attention module properly
        rng_key = jax.random.PRNGKey(42)
        self.attention = BigBirdSparseAttention(self.config)
        
        # Initialize the module with dummy inputs to trigger setup()
        dummy_input = jnp.ones((1, 64, self.config.d_model))
        self.params = self.attention.init(
            {'params': rng_key, 'random_attention': jax.random.PRNGKey(123)}, 
            dummy_input, 
            training=False
        )
        
        # Pre-compute RoPE frequencies
        self.cos_freqs, self.sin_freqs = precompute_rope_freqs(
            dim=self.config.d_model // self.config.n_heads,
            max_seq_len=self.config.max_position_embeddings,
            base=self.config.rope_theta
        )
    
    def test_single_length_batch(self):
        """Test RoPE with single sequence length batch."""
        batch_size = 2
        seq_len = 128
        
        # Create input data
        hidden_states = jax.random.normal(
            jax.random.PRNGKey(42), 
            (batch_size, seq_len, self.config.d_model)
        )
        position_ids = jnp.arange(seq_len)[None, :].repeat(batch_size, axis=0)
        
        # Forward pass should work without errors
        output = self.attention.apply(
            self.params,
            hidden_states=hidden_states,
            position_ids=position_ids,
            cos_freqs=self.cos_freqs,
            sin_freqs=self.sin_freqs,
            training=False,
            rngs={'random_attention': jax.random.PRNGKey(456)}
        )
        
        assert output.shape == hidden_states.shape
        assert not jnp.any(jnp.isnan(output))
    
    def test_multi_length_batch_with_padding(self):
        """Test RoPE with multi-length batch requiring padding."""
        batch_size = 3
        seq_lens = [100, 150, 200]  # Different lengths
        max_seq_len = max(seq_lens)
        
        # Create padded batch
        hidden_states_list = []
        position_ids_list = []
        
        for seq_len in seq_lens:
            # Create sequence data
            seq_data = jax.random.normal(
                jax.random.PRNGKey(42), 
                (seq_len, self.config.d_model)
            )
            # Pad to max length
            padded_data = jnp.pad(
                seq_data, 
                ((0, max_seq_len - seq_len), (0, 0)), 
                mode='constant'
            )
            hidden_states_list.append(padded_data)
            
            # Create position IDs (0-indexed)
            pos_ids = jnp.arange(seq_len)
            # Pad position IDs with zeros
            padded_pos_ids = jnp.pad(
                pos_ids, 
                (0, max_seq_len - seq_len), 
                mode='constant'
            )
            position_ids_list.append(padded_pos_ids)
        
        hidden_states = jnp.stack(hidden_states_list, axis=0)
        position_ids = jnp.stack(position_ids_list, axis=0)
        
        # Forward pass should handle multi-length batch correctly
        output = self.attention.apply(
            self.params,
            hidden_states=hidden_states,
            position_ids=position_ids,
            cos_freqs=self.cos_freqs,
            sin_freqs=self.sin_freqs,
            training=False,
            rngs={'random_attention': jax.random.PRNGKey(456)}
        )
        
        assert output.shape == hidden_states.shape
        assert not jnp.any(jnp.isnan(output))
        
        # Check that padded regions remain zero (approximately)
        for i, seq_len in enumerate(seq_lens):
            if seq_len < max_seq_len:
                padded_region = output[i, seq_len:, :]
                # Padded regions should have minimal values (close to zero)
                assert jnp.allclose(padded_region, 0.0, atol=1e-6)
    
    def test_position_ids_bounds_checking(self):
        """Test that position_ids are properly clipped to frequency bounds."""
        batch_size = 2
        seq_len = 100
        
        # Create position_ids that exceed frequency bounds
        max_freq_len = self.cos_freqs.shape[0]
        position_ids = jnp.full((batch_size, seq_len), max_freq_len + 10)  # Exceed bounds
        
        hidden_states = jax.random.normal(
            jax.random.PRNGKey(42), 
            (batch_size, seq_len, self.config.d_model)
        )
        
        # Should not raise errors due to bounds checking
        output = self.attention.apply(
            self.params,
            hidden_states=hidden_states,
            position_ids=position_ids,
            cos_freqs=self.cos_freqs,
            sin_freqs=self.sin_freqs,
            training=False,
            rngs={'random_attention': jax.random.PRNGKey(456)}
        )
        
        assert output.shape == hidden_states.shape
        assert not jnp.any(jnp.isnan(output))
    
    def test_empty_sequences_handling(self):
        """Test handling of sequences with all-zero position_ids."""
        batch_size = 2
        seq_len = 64
        
        # Create position_ids with one sequence being all zeros
        position_ids = jnp.array([
            jnp.arange(seq_len),  # Normal sequence
            jnp.zeros(seq_len, dtype=jnp.int32)  # All-zero sequence
        ])
        
        hidden_states = jax.random.normal(
            jax.random.PRNGKey(42), 
            (batch_size, seq_len, self.config.d_model)
        )
        
        # Should handle all-zero position_ids gracefully
        output = self.attention.apply(
            self.params,
            hidden_states=hidden_states,
            position_ids=position_ids,
            cos_freqs=self.cos_freqs,
            sin_freqs=self.sin_freqs,
            training=False,
            rngs={'random_attention': jax.random.PRNGKey(456)}
        )
        
        assert output.shape == hidden_states.shape
        assert not jnp.any(jnp.isnan(output))
    
    def test_block_size_alignment_with_rope(self):
        """Test that RoPE works correctly with block size alignment."""
        batch_size = 2
        seq_len = 100  # Not divisible by block_size (64)
        
        hidden_states = jax.random.normal(
            jax.random.PRNGKey(42), 
            (batch_size, seq_len, self.config.d_model)
        )
        position_ids = jnp.arange(seq_len)[None, :].repeat(batch_size, axis=0)
        
        # Forward pass should handle padding to block size correctly
        output = self.attention.apply(
            self.params,
            hidden_states=hidden_states,
            position_ids=position_ids,
            cos_freqs=self.cos_freqs,
            sin_freqs=self.sin_freqs,
            training=False,
            rngs={'random_attention': jax.random.PRNGKey(456)}
        )
        
        # Output should be trimmed back to original sequence length
        assert output.shape == (batch_size, seq_len, self.config.d_model)
        assert not jnp.any(jnp.isnan(output))
    
    def test_rope_consistency_across_batch_items(self):
        """Test that RoPE produces consistent results for identical sequences."""
        seq_len = 128
        
        # Create identical sequences in batch
        hidden_states = jax.random.normal(
            jax.random.PRNGKey(42), 
            (1, seq_len, self.config.d_model)
        )
        hidden_states = jnp.repeat(hidden_states, 3, axis=0)  # Repeat 3 times
        
        position_ids = jnp.arange(seq_len)[None, :].repeat(3, axis=0)
        
        output = self.attention.apply(
            self.params,
            hidden_states=hidden_states,
            position_ids=position_ids,
            cos_freqs=self.cos_freqs,
            sin_freqs=self.sin_freqs,
            training=False,
            rngs={'random_attention': jax.random.PRNGKey(456)}
        )
        
        # All batch items should produce identical results
        assert jnp.allclose(output[0], output[1], atol=1e-6)
        assert jnp.allclose(output[1], output[2], atol=1e-6)
    
    def test_invalid_position_ids_shape(self):
        """Test error handling for invalid position_ids shapes."""
        batch_size = 2
        seq_len = 64
        
        hidden_states = jax.random.normal(
            jax.random.PRNGKey(42), 
            (batch_size, seq_len, self.config.d_model)
        )
        
        # Test 1D position_ids (should fail)
        position_ids_1d = jnp.arange(seq_len)
        
        with pytest.raises(AssertionError, match="position_ids must be 2D"):
            self.attention.apply(
                self.params,
                hidden_states=hidden_states,
                position_ids=position_ids_1d,
                cos_freqs=self.cos_freqs,
                sin_freqs=self.sin_freqs,
                training=False,
                rngs={'random_attention': jax.random.PRNGKey(456)}
            )
        
        # Test mismatched batch dimension
        position_ids_wrong_batch = jnp.arange(seq_len)[None, :].repeat(3, axis=0)  # Wrong batch size
        
        with pytest.raises(AssertionError, match="position_ids batch dimension mismatch"):
            self.attention.apply(
                self.params,
                hidden_states=hidden_states,
                position_ids=position_ids_wrong_batch,
                cos_freqs=self.cos_freqs,
                sin_freqs=self.sin_freqs,
                training=False,
                rngs={'random_attention': jax.random.PRNGKey(456)}
            )


if __name__ == "__main__":
    pytest.main([__file__])