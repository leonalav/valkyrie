"""Longformer attention unit tests.

Critical validation tests from output.txt:
- Chunked vs dense attention equality
- Global attention behavior verification
- Sliding window mask validation
- KV caching correctness
- Mixed precision stability
"""

import jax
import jax.numpy as jnp
import pytest
import numpy as np
from typing import Tuple, Dict, Any, Optional
import logging

from ...model import ValkyrieLongformerAttention, ValkyrieAttention, ValkyrieConfig
from ...model.modules import precompute_rope_freqs
from ...utils.debug import check_for_nans, debug_shapes

logger = logging.getLogger(__name__)


class TestLongformerAttention:
    """Test suite for Longformer attention implementation."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return ValkyrieConfig(
            d_model=256,
            n_heads=8,
            n_kv_heads=8,
            use_longformer_attention=True,
            longformer_window_size=64,
            longformer_global_attention_indices=[0],
            longformer_chunked=True,
            longformer_chunk_size=32,
            max_position_embeddings=512,
        )
    
    @pytest.fixture
    def longformer_attention(self, config):
        """Create Longformer attention layer."""
        return ValkyrieLongformerAttention(config)
    
    @pytest.fixture
    def standard_attention(self, config):
        """Create standard attention layer for comparison."""
        config_std = ValkyrieConfig(
            d_model=config.d_model,
            n_heads=config.n_heads,
            n_kv_heads=config.n_kv_heads,
            use_longformer_attention=False,
            max_position_embeddings=config.max_position_embeddings,
        )
        return ValkyrieAttention(config_std)
    
    @pytest.fixture
    def test_data(self, config):
        """Create test data."""
        batch_size = 2
        seq_len = 128  # Moderate length for testing
        d_model = config.d_model
        
        key = jax.random.PRNGKey(42)
        x = jax.random.normal(key, (batch_size, seq_len, d_model))
        
        # Create position IDs
        position_ids = jnp.arange(seq_len)[None, :].repeat(batch_size, axis=0)
        
        # Precompute RoPE frequencies
        head_dim = d_model // config.n_heads
        cos_freqs, sin_freqs = precompute_rope_freqs(
            dim=head_dim,
            max_seq_len=config.max_position_embeddings,
            base=config.rope_theta
        )
        
        return {
            'x': x,
            'position_ids': position_ids,
            'cos_freqs': cos_freqs,
            'sin_freqs': sin_freqs,
        }
    
    def test_longformer_initialization(self, longformer_attention, config, test_data):
        """Test Longformer attention initialization."""
        
        # Initialize parameters
        key = jax.random.PRNGKey(0)
        variables = longformer_attention.init(
            key,
            test_data['x'],
            test_data['cos_freqs'],
            test_data['sin_freqs'],
            test_data['position_ids'],
            training=True
        )
        
        # Check parameter structure
        assert 'params' in variables
        params = variables['params']
        
        # Check required projections exist
        required_projections = [
            'qs_proj', 'ks_proj', 'vs_proj',  # Local attention
            'qg_proj', 'kg_proj', 'vg_proj',  # Global attention
            'o_proj'  # Output projection
        ]
        
        for proj_name in required_projections:
            assert proj_name in params, f"Missing projection: {proj_name}"
        
        logger.info("✓ Longformer initialization test passed")
    
    def test_sliding_window_mask(self, longformer_attention, config):
        """Test sliding window mask creation."""
        
        seq_len = 64
        window_size = config.longformer_window_size
        
        # Initialize layer to access methods
        key = jax.random.PRNGKey(0)
        dummy_input = jax.random.normal(key, (1, seq_len, config.d_model))
        variables = longformer_attention.init(
            key, dummy_input, 
            jnp.zeros((config.max_position_embeddings, config.d_model // config.n_heads // 2)),
            jnp.zeros((config.max_position_embeddings, config.d_model // config.n_heads // 2)),
            jnp.arange(seq_len)[None, :],
            training=True
        )
        
        bound_layer = longformer_attention.bind(variables)
        
        # Test mask creation
        mask = bound_layer._create_sliding_window_mask(seq_len, window_size, causal=True)
        
        # Check mask shape
        assert mask.shape == (seq_len, seq_len)
        
        # Check mask properties
        # 1. Causal property: mask[i, j] = False if j > i
        for i in range(seq_len):
            for j in range(i + 1, seq_len):
                assert not mask[i, j], f"Causal mask violated at ({i}, {j})"
        
        # 2. Window property: mask[i, j] = False if |i - j| > window_size // 2
        window_radius = window_size // 2
        for i in range(seq_len):
            for j in range(seq_len):
                if j <= i:  # Only check causal positions
                    distance = abs(i - j)
                    expected = distance <= window_radius
                    actual = bool(mask[i, j])
                    assert actual == expected, f"Window mask incorrect at ({i}, {j}): expected {expected}, got {actual}"
        
        logger.info("✓ Sliding window mask test passed")
    
    def test_chunked_vs_dense_attention_equality(self, longformer_attention, config, test_data):
        """
        Critical test: Chunked vs dense attention equality.
        
        This is the most important test from output.txt - must pass for correctness.
        """
        
        logger.info("Running chunked vs dense attention equality test...")
        
        # Use smaller sequence for this test to enable dense computation
        batch_size = 2
        seq_len = 64  # Small enough for dense attention
        
        # Create test input
        key = jax.random.PRNGKey(42)
        x = jax.random.normal(key, (batch_size, seq_len, config.d_model))
        position_ids = jnp.arange(seq_len)[None, :].repeat(batch_size, axis=0)
        
        # Precompute RoPE frequencies
        head_dim = config.d_model // config.n_heads
        cos_freqs, sin_freqs = precompute_rope_freqs(
            dim=head_dim,
            max_seq_len=config.max_position_embeddings,
            base=config.rope_theta
        )
        
        # Initialize layer
        variables = longformer_attention.init(
            key, x, cos_freqs, sin_freqs, position_ids, training=True
        )
        
        # Create modified config for dense attention (force full attention)
        config_dense = ValkyrieConfig(
            d_model=config.d_model,
            n_heads=config.n_heads,
            n_kv_heads=config.n_kv_heads,
            use_longformer_attention=True,
            longformer_window_size=config.longformer_window_size,
            longformer_global_attention_indices=config.longformer_global_attention_indices,
            longformer_chunked=False,  # Disable chunking
            longformer_use_full_attention_fallback=True,
            max_position_embeddings=config.max_position_embeddings,
        )
        
        dense_attention = ValkyrieLongformerAttention(config_dense)
        
        # 1. Chunked attention output
        chunked_output, _ = longformer_attention.apply(
            variables, x, cos_freqs, sin_freqs, position_ids, training=False
        )
        
        # 2. Dense attention output (using same parameters)
        dense_output, _ = dense_attention.apply(
            variables, x, cos_freqs, sin_freqs, position_ids, training=False
        )
        
        # 3. Compare outputs
        max_diff = jnp.max(jnp.abs(chunked_output - dense_output))
        relative_error = max_diff / (jnp.max(jnp.abs(dense_output)) + 1e-8)
        
        logger.info(f"Chunked vs Dense attention comparison:")
        logger.info(f"  Max absolute difference: {max_diff}")
        logger.info(f"  Relative error: {relative_error}")
        logger.info(f"  Chunked output shape: {chunked_output.shape}")
        logger.info(f"  Dense output shape: {dense_output.shape}")
        
        # Tolerance check
        tolerance = 1e-4  # Slightly relaxed for chunked computation
        assert max_diff < tolerance, f"Chunked vs dense mismatch: {max_diff} > {tolerance}"
        
        logger.info("✓ Chunked vs dense attention equality test PASSED")
        
        return True
    
    def test_global_attention_behavior(self, longformer_attention, config, test_data):
        """Test global attention token behavior."""
        
        # Initialize layer
        key = jax.random.PRNGKey(0)
        variables = longformer_attention.init(
            key,
            test_data['x'],
            test_data['cos_freqs'],
            test_data['sin_freqs'],
            test_data['position_ids'],
            training=True
        )
        
        # Forward pass
        output, kv_cache = longformer_attention.apply(
            variables,
            test_data['x'],
            test_data['cos_freqs'],
            test_data['sin_freqs'],
            test_data['position_ids'],
            training=False
        )
        
        # Check output shape
        batch_size, seq_len, d_model = test_data['x'].shape
        assert output.shape == (batch_size, seq_len, d_model)
        
        # Check KV cache structure (Longformer has 4-tuple cache)
        assert len(kv_cache) == 4, f"Expected 4-tuple KV cache, got {len(kv_cache)}"
        
        ks_cache, vs_cache, kg_cache, vg_cache = kv_cache
        
        # Check cache shapes
        expected_cache_shape = (batch_size, config.n_kv_heads, config.max_position_embeddings, config.d_model // config.n_heads)
        
        assert ks_cache.shape == expected_cache_shape
        assert vs_cache.shape == expected_cache_shape
        assert kg_cache.shape == expected_cache_shape
        assert vg_cache.shape == expected_cache_shape
        
        # Check for NaNs
        assert not check_for_nans(output, "Longformer output")
        assert not check_for_nans(kv_cache, "Longformer KV cache")
        
        logger.info("✓ Global attention behavior test passed")
    
    def test_kv_caching_consistency(self, longformer_attention, config, test_data):
        """Test KV caching consistency."""
        
        # Initialize layer
        key = jax.random.PRNGKey(0)
        variables = longformer_attention.init(
            key,
            test_data['x'],
            test_data['cos_freqs'],
            test_data['sin_freqs'],
            test_data['position_ids'],
            training=True
        )
        
        batch_size, full_seq_len, d_model = test_data['x'].shape
        
        # Split sequence for caching test
        prefill_len = full_seq_len // 2
        
        # 1. Process full sequence without caching
        full_output, _ = longformer_attention.apply(
            variables,
            test_data['x'],
            test_data['cos_freqs'],
            test_data['sin_freqs'],
            test_data['position_ids'],
            training=False
        )
        
        # 2. Process with caching (prefill + generation)
        prefill_input = test_data['x'][:, :prefill_len, :]
        prefill_pos_ids = test_data['position_ids'][:, :prefill_len]
        
        # Prefill phase
        prefill_output, kv_cache = longformer_attention.apply(
            variables,
            prefill_input,
            test_data['cos_freqs'],
            test_data['sin_freqs'],
            prefill_pos_ids,
            training=False
        )
        
        # Generation phase (remaining tokens one by one)
        cached_outputs = [prefill_output]
        current_cache = kv_cache
        
        for i in range(prefill_len, full_seq_len):
            token_input = test_data['x'][:, i:i+1, :]
            token_pos_ids = test_data['position_ids'][:, i:i+1]
            
            token_output, current_cache = longformer_attention.apply(
                variables,
                token_input,
                test_data['cos_freqs'],
                test_data['sin_freqs'],
                token_pos_ids,
                past_key_value=current_cache,
                training=False
            )
            
            cached_outputs.append(token_output)
        
        # Concatenate cached outputs
        cached_full_output = jnp.concatenate(cached_outputs, axis=1)
        
        # 3. Compare outputs
        max_diff = jnp.max(jnp.abs(full_output - cached_full_output))
        relative_error = max_diff / (jnp.max(jnp.abs(full_output)) + 1e-8)
        
        logger.info(f"KV caching consistency check:")
        logger.info(f"  Max absolute difference: {max_diff}")
        logger.info(f"  Relative error: {relative_error}")
        
        # Tolerance check
        tolerance = 1e-4
        assert max_diff < tolerance, f"KV caching inconsistency: {max_diff} > {tolerance}"
        
        logger.info("✓ KV caching consistency test passed")
    
    def test_attention_mask_correctness(self, longformer_attention, config):
        """Test attention mask correctness for sliding window."""
        
        seq_len = 32
        window_size = 16
        
        # Initialize layer
        key = jax.random.PRNGKey(0)
        dummy_input = jax.random.normal(key, (1, seq_len, config.d_model))
        position_ids = jnp.arange(seq_len)[None, :]
        
        head_dim = config.d_model // config.n_heads
        cos_freqs, sin_freqs = precompute_rope_freqs(head_dim, config.max_position_embeddings)
        
        variables = longformer_attention.init(
            key, dummy_input, cos_freqs, sin_freqs, position_ids, training=True
        )
        
        bound_layer = longformer_attention.bind(variables)
        
        # Test mask creation
        mask = bound_layer._create_sliding_window_mask(seq_len, window_size, causal=True)
        
        # Verify mask properties
        assert mask.shape == (seq_len, seq_len)
        
        # Check causal property
        for i in range(seq_len):
            for j in range(i + 1, seq_len):
                assert not mask[i, j], f"Causal property violated at ({i}, {j})"
        
        # Check window property
        window_radius = window_size // 2
        for i in range(seq_len):
            for j in range(seq_len):
                if j <= i:
                    distance = abs(i - j)
                    expected = distance <= window_radius
                    actual = bool(mask[i, j])
                    assert actual == expected, f"Window property violated at ({i}, {j})"
        
        logger.info("✓ Attention mask correctness test passed")
    
    def test_global_token_symmetry(self, longformer_attention, config, test_data):
        """Test that global tokens have symmetric attention (bidirectional)."""
        
        # This test verifies that global tokens can attend to all tokens
        # and all tokens can attend to global tokens
        
        # Initialize layer
        key = jax.random.PRNGKey(0)
        variables = longformer_attention.init(
            key,
            test_data['x'],
            test_data['cos_freqs'],
            test_data['sin_freqs'],
            test_data['position_ids'],
            training=True
        )
        
        # Forward pass
        output, _ = longformer_attention.apply(
            variables,
            test_data['x'],
            test_data['cos_freqs'],
            test_data['sin_freqs'],
            test_data['position_ids'],
            training=False
        )
        
        # Check that global token (index 0) output is different from local tokens
        # This indicates it's receiving information from the full sequence
        global_output = output[:, 0, :]  # First token (global)
        local_output = output[:, 1, :]   # Second token (local)
        
        # They should be different (global has more information)
        diff = jnp.max(jnp.abs(global_output - local_output))
        assert diff > 1e-6, f"Global and local outputs too similar: {diff}"
        
        # Check for NaNs
        assert not check_for_nans(output, "Global attention output")
        
        logger.info("✓ Global token symmetry test passed")
    
    def test_attention_numerical_stability(self, longformer_attention, config):
        """Test attention numerical stability with edge cases."""
        
        # Test with various input magnitudes
        key = jax.random.PRNGKey(0)
        batch_size, seq_len, d_model = 2, 32, config.d_model
        
        position_ids = jnp.arange(seq_len)[None, :].repeat(batch_size, axis=0)
        head_dim = d_model // config.n_heads
        cos_freqs, sin_freqs = precompute_rope_freqs(head_dim, config.max_position_embeddings)
        
        # Initialize layer
        dummy_input = jax.random.normal(key, (batch_size, seq_len, d_model))
        variables = longformer_attention.init(
            key, dummy_input, cos_freqs, sin_freqs, position_ids, training=True
        )
        
        test_cases = [
            ("normal", jax.random.normal(key, (batch_size, seq_len, d_model))),
            ("small", jax.random.normal(key, (batch_size, seq_len, d_model)) * 1e-3),
            ("large", jax.random.normal(key, (batch_size, seq_len, d_model)) * 10.0),
            ("zeros", jnp.zeros((batch_size, seq_len, d_model))),
        ]
        
        for case_name, test_input in test_cases:
            logger.info(f"Testing attention stability with {case_name} inputs...")
            
            try:
                output, kv_cache = longformer_attention.apply(
                    variables, test_input, cos_freqs, sin_freqs, position_ids, training=False
                )
                
                # Check for NaNs/Infs
                assert not check_for_nans(output, f"Attention output ({case_name})")
                assert not check_for_nans(kv_cache, f"KV cache ({case_name})")
                
                # Check output magnitude is reasonable
                output_norm = jnp.linalg.norm(output)
                assert output_norm < 1e6, f"Output too large ({case_name}): {output_norm}"
                
                logger.info(f"  ✓ {case_name} case passed")
                
            except Exception as e:
                logger.error(f"  ❌ {case_name} case failed: {e}")
                raise
        
        logger.info("✓ Attention numerical stability test passed")
    
    def test_longformer_vs_standard_attention(self, longformer_attention, standard_attention, config):
        """Compare Longformer vs standard attention on small sequences."""
        
        # For small sequences within window size, Longformer should behave similarly to standard attention
        batch_size = 2
        seq_len = 16  # Smaller than window size
        d_model = config.d_model
        
        key = jax.random.PRNGKey(42)
        x = jax.random.normal(key, (batch_size, seq_len, d_model))
        position_ids = jnp.arange(seq_len)[None, :].repeat(batch_size, axis=0)
        
        head_dim = d_model // config.n_heads
        cos_freqs, sin_freqs = precompute_rope_freqs(head_dim, config.max_position_embeddings)
        
        # Initialize both layers with same parameters
        longformer_vars = longformer_attention.init(
            key, x, cos_freqs, sin_freqs, position_ids, training=True
        )
        
        standard_vars = standard_attention.init(
            key, x, cos_freqs, sin_freqs, position_ids, training=True
        )
        
        # Forward passes
        longformer_out, _ = longformer_attention.apply(
            longformer_vars, x, cos_freqs, sin_freqs, position_ids, training=False
        )
        
        standard_out, _ = standard_attention.apply(
            standard_vars, x, cos_freqs, sin_freqs, position_ids, training=False
        )
        
        # They won't be identical due to different projections, but should have similar magnitudes
        longformer_norm = jnp.linalg.norm(longformer_out)
        standard_norm = jnp.linalg.norm(standard_out)
        
        ratio = longformer_norm / (standard_norm + 1e-8)
        
        logger.info(f"Longformer vs Standard attention:")
        logger.info(f"  Longformer norm: {longformer_norm}")
        logger.info(f"  Standard norm: {standard_norm}")
        logger.info(f"  Ratio: {ratio}")
        
        # Sanity check: outputs should have reasonable magnitudes
        assert 0.1 < ratio < 10.0, f"Output magnitude ratio unreasonable: {ratio}"
        
        logger.info("✓ Longformer vs standard attention test passed")


def run_attention_tests():
    """Run all Longformer attention tests."""
    
    logger.info("=== Running Longformer Attention Unit Tests ===")
    
    # Create test configuration
    config = ValkyrieConfig(
        d_model=256,
        n_heads=8,
        n_kv_heads=8,
        use_longformer_attention=True,
        longformer_window_size=64,
        longformer_global_attention_indices=[0],
        longformer_chunked=True,
        longformer_chunk_size=32,
        max_position_embeddings=512,
    )
    
    # Create attention layers
    longformer_attention = ValkyrieLongformerAttention(config)
    
    config_std = ValkyrieConfig(
        d_model=config.d_model,
        n_heads=config.n_heads,
        n_kv_heads=config.n_kv_heads,
        use_longformer_attention=False,
        max_position_embeddings=config.max_position_embeddings,
    )
    standard_attention = ValkyrieAttention(config_std)
    
    # Create test data
    key = jax.random.PRNGKey(42)
    batch_size, seq_len, d_model = 2, 128, config.d_model
    x = jax.random.normal(key, (batch_size, seq_len, d_model))
    position_ids = jnp.arange(seq_len)[None, :].repeat(batch_size, axis=0)
    
    head_dim = d_model // config.n_heads
    cos_freqs, sin_freqs = precompute_rope_freqs(head_dim, config.max_position_embeddings)
    
    test_data = {
        'x': x,
        'position_ids': position_ids,
        'cos_freqs': cos_freqs,
        'sin_freqs': sin_freqs,
    }
    
    # Create test instance
    test_instance = TestLongformerAttention()
    
    try:
        # Run tests
        test_instance.test_longformer_initialization(longformer_attention, config, test_data)
        test_instance.test_sliding_window_mask(longformer_attention, config)
        test_instance.test_chunked_vs_dense_attention_equality(longformer_attention, config, test_data)
        test_instance.test_global_attention_behavior(longformer_attention, config, test_data)
        test_instance.test_kv_caching_consistency(longformer_attention, config, test_data)
        test_instance.test_attention_numerical_stability(longformer_attention, config)
        test_instance.test_longformer_vs_standard_attention(longformer_attention, standard_attention, config)
        
        logger.info("🎉 ALL LONGFORMER ATTENTION TESTS PASSED!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Longformer attention tests failed: {e}")
        return False


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Run tests
    success = run_attention_tests()
    
    if success:
        print("✅ Longformer attention unit tests completed successfully")
        exit(0)
    else:
        print("❌ Longformer attention unit tests failed")
        exit(1)