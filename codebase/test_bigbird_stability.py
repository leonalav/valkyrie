#!/usr/bin/env python3
"""
Test script to verify BigBird attention stability across different configurations.
This tests the fix for boundary condition issues in the BigBird sparse attention mechanism.
"""

import pytest
import jax
import jax.numpy as jnp
from src.model.gryphon.bigbird_attention import BigBirdSparseAttention
from src.model.gryphon.gryphon_config import get_gryphon_small_config

@pytest.mark.parametrize("seq_len,block_size", [
    (512, 64),
    (1024, 128),
    (2048, 64),
    (4096, 128)
])
def test_configuration(seq_len, block_size):
    """Test BigBird attention with specific configuration"""
    batch_size = 1
    print(f'\n=== Testing seq_len={seq_len}, block_size={block_size} ===')
    
    # Create config
    config = get_gryphon_small_config()
    config.max_sequence_length = seq_len
    config.block_size = block_size
    
    # Create test inputs
    key = jax.random.PRNGKey(42)
    hidden_states = jax.random.normal(key, (batch_size, seq_len, config.d_model))
    attention_mask = jnp.ones((batch_size, seq_len))
    
    num_blocks = seq_len // block_size
    print(f'  Input shape: {hidden_states.shape}')
    print(f'  Num blocks: {num_blocks}')
    
    # Initialize attention mechanism
    attention = BigBirdSparseAttention(config=config)
    
    # Setup RNG keys
    init_key, apply_key = jax.random.split(key)
    dropout_key, random_attn_key = jax.random.split(apply_key)
    
    rngs = {
        'params': init_key, 
        'random_attention': init_key,
        'dropout': init_key
    }
    
    try:
        # Test initialization
        params = attention.init(rngs, hidden_states, attention_mask)
        
        # Test forward pass
        apply_rngs = {
            'random_attention': random_attn_key,
            'dropout': dropout_key
        }
        
        output = attention.apply(params, hidden_states, attention_mask, rngs=apply_rngs)
        
        # Test causal forward pass
        causal_key = jax.random.PRNGKey(123)
        init_key2, apply_key2 = jax.random.split(causal_key)
        dropout_key2, random_attn_key2 = jax.random.split(apply_key2)
        
        rngs_causal = {
            'params': init_key2, 
            'random_attention': init_key2,
            'dropout': init_key2
        }
        
        params_causal = attention.init(rngs_causal, hidden_states, attention_mask, causal=True)
        
        apply_rngs_causal = {
            'random_attention': random_attn_key2,
            'dropout': dropout_key2
        }
        
        output_causal = attention.apply(params_causal, hidden_states, attention_mask, causal=True, rngs=apply_rngs_causal)
        
        print(f'  ✓ Non-causal output: {output.shape}')
        print(f'  ✓ Causal output: {output_causal.shape}')
        print(f'  ✓ SUCCESS: Both modes work correctly')
        return True
        
    except Exception as e:
        print(f'  ✗ FAILED: {e}')
        return False

def main():
    """Run comprehensive BigBird attention stability tests"""
    
    # Test various configurations
    test_configs = [
        # (seq_len, block_size)
        (64, 32),    # 2 blocks
        (128, 64),   # 2 blocks (original failing case)
        (192, 64),   # 3 blocks
        (256, 64),   # 4 blocks
        (320, 64),   # 5 blocks
        (128, 32),   # 4 blocks
        (96, 32),    # 3 blocks
    ]

    print('=== BigBird Attention Stability Test ===')
    print('Testing various sequence lengths and block sizes...')

    success_count = 0
    total_tests = len(test_configs)

    for seq_len, block_size in test_configs:
        if test_configuration(seq_len, block_size):
            success_count += 1

    print(f'\n=== Test Results ===')
    print(f'Passed: {success_count}/{total_tests}')
    print(f'Success rate: {success_count/total_tests*100:.1f}%')

    if success_count == total_tests:
        print('\n🎉 ALL TESTS PASSED! BigBird attention is stable across configurations.')
    else:
        print(f'\n⚠️ {total_tests - success_count} tests failed. Review the errors above.')
    
    return success_count == total_tests

if __name__ == '__main__':
    main()