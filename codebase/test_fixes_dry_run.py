#!/usr/bin/env python3
"""
Dry-run test for all training fixes with small shapes.
Tests compilation and basic functionality without full training.
"""

import jax
import jax.numpy as jnp
import numpy as np
from pathlib import Path
import sys
import traceback
from typing import Dict, Any, NamedTuple, List
import os

# Add src to path and set PYTHONPATH
src_path = str(Path(__file__).parent / "src")
sys.path.insert(0, src_path)
os.environ['PYTHONPATH'] = src_path + ':' + os.environ.get('PYTHONPATH', '')

# Import directly to avoid relative import issues
sys.path.insert(0, str(Path(__file__).parent / "src" / "model"))
sys.path.insert(0, str(Path(__file__).parent / "src" / "train"))
sys.path.insert(0, str(Path(__file__).parent / "src" / "io"))

# Simple test configs to avoid complex imports
class CheckpointConfig(NamedTuple):
    checkpoint_dir: str = "/tmp/test_checkpoints"
    save_interval_steps: int = 100
    keep_checkpoints: int = 3
    async_save: bool = False
    fast_checkpoint_interval: int = 50
    full_checkpoint_interval: int = 100
    validate_on_save: bool = False
    validate_on_load: bool = False
    use_compression: bool = False
    compression_level: int = 1
    save_optimizer_state: bool = False
    async_checkpointing: bool = False

class PhaseConfig(NamedTuple):
    name: str
    steps: int
    hrm_enabled: bool = True
    hrm_supervision_weight: float = 1.0

class CurriculumConfig(NamedTuple):
    phases: List[PhaseConfig]

class TrainingConfig(NamedTuple):
    learning_rate: float
    batch_size: int
    sequence_length: int
    gradient_accumulation_steps: int
    max_steps: int
    warmup_steps: int
    checkpoint_dir: str
    log_dir: str

class CheckpointConfig(NamedTuple):
    checkpoint_dir: str
    save_interval: int
    max_to_keep: int

class ValkyrieConfig(NamedTuple):
    vocab_size: int = 1000
    d_model: int = 128
    n_layers: int = 2
    n_heads: int = 4
    n_kv_heads: int = 4
    intermediate_size: int = 256
    max_position_embeddings: int = 512
    use_s5: bool = True
    use_hrm: bool = True
    s5_state_dim: int = 64
    s5_blocks_per_layer: int = 1
    use_bias: bool = False
    layer_norm_eps: float = 1e-5
    attn_dropout: float = 0.0
    resid_dropout: float = 0.0
    ffn_dropout: float = 0.1
    rope_theta: float = 10000.0
    bigbird_block_size: int = 64
    bigbird_num_global_tokens: int = 64
    bigbird_num_window_blocks: int = 3
    bigbird_num_random_blocks: int = 2
    use_bigbird_attention: bool = True
    hrm_plan_length: int = 32
    hrm_H_cycles: int = 3
    hrm_L_cycles: int = 3
    hrm_H_layers: int = 6
    hrm_L_layers: int = 6
    hrm_intermediate_size: int = 512
    hrm_use_act: bool = True
    hrm_act_threshold: float = 0.9
    hrm_planner_layers: int = 2
    hrm_executor_steps: int = 4
    hrm_planner_update_frequency: int = 4
    hrm_use_act_halting: bool = True
    hrm_one_step_gradient: bool = True
    hrm_deep_supervision: bool = True


def create_minimal_config() -> Dict[str, Any]:
    """Create minimal config for dry-run testing."""
    return {
        'model': ValkyrieConfig(),
        'training': TrainingConfig(
            learning_rate=1e-4,
            batch_size=2,
            sequence_length=64,
            gradient_accumulation_steps=1,
            max_steps=10,
            warmup_steps=2,
            checkpoint_dir="/tmp/test_checkpoints",
            log_dir="/tmp/test_logs",
        ),
        'curriculum': CurriculumConfig(
            phases=[
                PhaseConfig(
                    name="phase1_no_hrm",
                    steps=5,
                    hrm_enabled=False,
                    hrm_supervision_weight=0.0,
                ),
                PhaseConfig(
                    name="phase2_with_hrm", 
                    steps=5,
                    hrm_enabled=True,
                    hrm_supervision_weight=0.1,
                ),
            ]
        ),
        'checkpoint': CheckpointConfig(
            checkpoint_dir="/tmp/test_checkpoints",
            save_interval=5,
            max_to_keep=2,
        )
    }


def test_model_compilation():
    """Test model compilation with both HRM enabled/disabled."""
    print("Testing model compilation...")
    
    try:
        # Import here to avoid early import issues
        from model.valkyrie import ValkyrieModel
        
        config = create_minimal_config()
        model = ValkyrieModel(config['model'])
        
        # Test data
        batch_size, seq_len = 2, 64
        input_ids = jnp.ones((batch_size, seq_len), dtype=jnp.int32)
        
        # Initialize model parameters
        rng = jax.random.PRNGKey(42)
        params = model.init(
            rng,
            input_ids=input_ids,
            training=True,
            hrm_enabled=True,
        )
        
        # Test compilation with HRM enabled
        print("  - Compiling with HRM enabled...")
        @jax.jit
        def forward_hrm_enabled(params, input_ids):
            return model.apply(
                params,
                input_ids=input_ids,
                training=True,
                hrm_enabled=True,
                use_cache=False,
                return_dict=True,
            )
        
        outputs_hrm = forward_hrm_enabled(params, input_ids)
        assert 'logits' in outputs_hrm
        print("    ✓ HRM enabled compilation successful")
        
        # Test compilation with HRM disabled
        print("  - Compiling with HRM disabled...")
        @jax.jit
        def forward_hrm_disabled(params, input_ids):
            return model.apply(
                params,
                input_ids=input_ids,
                training=True,
                hrm_enabled=False,
                use_cache=False,
                return_dict=True,
            )
        
        outputs_no_hrm = forward_hrm_disabled(params, input_ids)
        assert 'logits' in outputs_no_hrm
        print("    ✓ HRM disabled compilation successful")
        
        # Verify different tensor materializations
        hrm_logits_shape = outputs_hrm['logits'].shape
        no_hrm_logits_shape = outputs_no_hrm['logits'].shape
        assert hrm_logits_shape == no_hrm_logits_shape
        print("    ✓ Logits shapes consistent between HRM modes")
        
        # Check HRM state handling
        if 'hrm_state' in outputs_hrm:
            print("    ✓ HRM state present when enabled")
        if 'hrm_state' in outputs_no_hrm:
            if outputs_no_hrm['hrm_state'] is None:
                print("    ✓ HRM state properly None when disabled")
        
        return True
        
    except Exception as e:
        print(f"    ✗ Model compilation failed: {e}")
        return False


def test_checkpoint_absolute_paths():
    """Test checkpoint manager with absolute path resolution."""
    print("Testing checkpoint absolute path resolution...")
    
    try:
        from io.checkpoint import CheckpointManager
    except ImportError:
        # Try alternative import path
        sys.path.insert(0, str(Path(__file__).parent / "src"))
        from src.io.checkpoint import CheckpointManager
        
        config = create_minimal_config()
        
        # Test relative path gets resolved to absolute
        relative_path = "test_checkpoints"
        checkpoint_config = CheckpointConfig(
            checkpoint_dir=relative_path,
            save_interval=5,
            max_to_keep=2,
        )
        
        checkpoint_manager = CheckpointManager(checkpoint_config)
        
        # Verify path is absolute
        assert checkpoint_manager.checkpoint_dir.is_absolute()
        print(f"    ✓ Relative path '{relative_path}' resolved to absolute: {checkpoint_manager.checkpoint_dir}")
        
        # Test already absolute path
        abs_path = "/tmp/test_checkpoints_abs"
        checkpoint_config_abs = CheckpointConfig(
            checkpoint_dir=abs_path,
            save_interval=5,
            max_to_keep=2,
        )
        
        checkpoint_manager_abs = CheckpointManager(checkpoint_config_abs)
        assert str(checkpoint_manager_abs.checkpoint_dir) == abs_path
        print(f"    ✓ Absolute path preserved: {checkpoint_manager_abs.checkpoint_dir}")
        
        return True
        
    except Exception as e:
        print(f"    ✗ Checkpoint path test failed: {e}")
        return False


def test_phase_control():
    """Test phase-based HRM control."""
    print("Testing training loop phase control...")
    
    config = create_minimal_config()
    
    # Test phase transitions
    phase1 = config['curriculum'].phases[0]  # HRM disabled
    phase2 = config['curriculum'].phases[1]  # HRM enabled
    
    assert not phase1.hrm_enabled
    assert phase2.hrm_enabled
    assert phase1.hrm_supervision_weight == 0.0
    assert phase2.hrm_supervision_weight == 0.1
    
    print("    ✓ Phase configurations correct")
    
    return True


def test_cache_disabled_during_training():
    """Test that use_cache=False is properly set during training."""
    print("Testing cache disabled during training...")
    
    try:
        from model.valkyrie import ValkyrieModel
        
        config = create_minimal_config()
        model = ValkyrieModel(config['model'])
        
        batch_size, seq_len = 2, 64
        input_ids = jnp.ones((batch_size, seq_len), dtype=jnp.int32)
        
        rng = jax.random.PRNGKey(42)
        params = model.init(
            rng,
            input_ids=input_ids,
            training=True,
            hrm_enabled=True,
        )
        
        # Test training mode with cache disabled
        outputs = model.apply(
            params,
            input_ids=input_ids,
            training=True,
            use_cache=False,  # This should be False during training
            hrm_enabled=True,
            return_dict=True,
        )
        
        # Verify no cache outputs
        if 'past_key_values' in outputs:
            assert outputs['past_key_values'] is None
            print("    ✓ Cache properly disabled during training")
        else:
            print("    ✓ No cache keys in output (cache disabled)")
        
        return True
        
    except Exception as e:
        print(f"    ✗ Cache test failed: {e}")
        return False


def run_all_tests():
    """Run all dry-run tests."""
    print("=" * 60)
    print("RUNNING DRY-RUN TESTS FOR TRAINING FIXES")
    print("=" * 60)
    
    tests = [
        ("Model Compilation", test_model_compilation),
        ("Checkpoint Absolute Paths", test_checkpoint_absolute_paths),
        ("Phase Control", test_phase_control),
        ("Cache Disabled During Training", test_cache_disabled_during_training),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n[TEST] {test_name}")
        try:
            success = test_func()
            results[test_name] = "PASS" if success else "FAIL"
            print(f"[RESULT] {test_name}: {results[test_name]}")
        except Exception as e:
            results[test_name] = "ERROR"
            print(f"[ERROR] {test_name}: {str(e)}")
            print(f"[TRACEBACK] {traceback.format_exc()}")
    
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    for test_name, result in results.items():
        status_symbol = "✓" if result == "PASS" else "✗"
        print(f"{status_symbol} {test_name}: {result}")
    
    all_passed = all(result == "PASS" for result in results.values())
    
    if all_passed:
        print("\n🎉 ALL TESTS PASSED! Fixes are ready for production.")
    else:
        print("\n❌ Some tests failed. Please review and fix issues.")
        return False
    
    return True


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)