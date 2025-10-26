#!/usr/bin/env python3
"""Integration Test for PLAN Training Pipeline

Tests the complete integration of:
- HRM Training Loop with curriculum phases
- Multi-source data pipeline with PackedSequence
- Phase-based sampling and algorithmic tasks
- Model forward/backward passes with proper shapes
"""

import jax
import jax.numpy as jnp
import numpy as np
from pathlib import Path
import sys
import logging
from typing import Dict, Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.data import (
    create_plan_data_loader,
    create_plan_task_generator, 
    create_plan_sampler,
    create_plan_phases
)
from src.train.hrm_training_loop import HRMTrainingLoop
# Remove the missing import
# from src.train.curriculum_config import CurriculumConfig
from src.model.valkyrie import ValkyrieModel
from src.model.modules import ValkyrieConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_data_pipeline():
    """Test data pipeline components."""
    logger.info("Testing data pipeline...")
    
    # Test data loader creation
    data_loader = create_plan_data_loader(
        tokenizer_name="gpt2",
        vocab_size=32000,
        max_length=1024,  # Smaller for testing
        seed=42
    )
    assert data_loader is not None
    logger.info("✓ Data loader created successfully")
    
    # Test task generator
    task_generator = create_plan_task_generator(seed=42)
    assert task_generator is not None
    logger.info("✓ Task generator created successfully")
    
    # Test phase sampler
    sampler = create_plan_sampler(data_loader, task_generator)
    assert sampler is not None
    logger.info("✓ Phase sampler created successfully")
    
    # Test phase configuration
    phases = create_plan_phases()
    assert len(phases) == 3
    assert phases[0].name == "phase1_base_lm"
    assert phases[1].name == "phase2_enable_hrm"
    assert phases[2].name == "phase3_full_hrm"
    logger.info("✓ Phase configuration correct")
    
    return data_loader, task_generator, sampler, phases


def test_model_creation():
    """Test model creation and basic forward pass."""
    logger.info("Testing model creation...")
    
    # Create model config
    config = ValkyrieConfig(
        vocab_size=32000,
        d_model=512,
        n_layers=4,
        n_heads=8,
        max_position_embeddings=1024,
        use_s5=True,
        use_hrm=True
    )
    
    # Create model
    model = ValkyrieModel(config)
    assert model is not None
    logger.info("✓ Model created successfully")
    
    # Test parameter initialization
    key = jax.random.PRNGKey(42)
    batch_size, seq_len = 2, 128
    
    # Create dummy input
    input_ids = jnp.ones((batch_size, seq_len), dtype=jnp.int32)
    
    # Initialize parameters
    params = model.init(key, input_ids)
    assert params is not None
    logger.info("✓ Model parameters initialized")
    
    # Test forward pass
    outputs = model.apply(params, input_ids)
    assert outputs.logits.shape == (batch_size, seq_len, config.vocab_size)
    logger.info(f"✓ Forward pass successful, output shape: {outputs.logits.shape}")
    
    return model, params, config


def test_hrm_training_loop():
    """Test HRM training loop initialization and basic operations."""
    logger.info("Testing HRM training loop...")
    
    # Create curriculum config
    curriculum_config = CurriculumConfig(
        phases=[
            {
                "name": "phase1_base_lm",
                "max_steps": 1000,
                "hrm_enabled": False,
                "hrm_cycles": 2,
                "hrm_steps": 2,
                "hrm_use_act": False,
                "hrm_supervision_weight": 0.0
            },
            {
                "name": "phase2_enable_hrm", 
                "max_steps": 2000,
                "hrm_enabled": True,
                "hrm_cycles": 2,
                "hrm_steps": 2,
                "hrm_use_act": False,
                "hrm_supervision_weight": 0.1
            }
        ]
    )
    
    # Create model config
    model_config = ModelConfig(
        vocab_size=32000,
        hidden_size=256,
        num_layers=2,
        num_heads=4,
        max_position_embeddings=1024
    )
    
    # Create HRM training loop
    hrm_loop = HRMTrainingLoop(model_config, curriculum_config)
    assert hrm_loop is not None
    logger.info("✓ HRM training loop created successfully")
    
    # Test HRM state initialization
    key = jax.random.PRNGKey(42)
    batch_size, seq_len = 2, 64
    
    phase_config = curriculum_config.phases[1]  # HRM enabled phase
    hrm_state = hrm_loop.initialize_hrm_state(key, batch_size, seq_len, phase_config)
    assert hrm_state is not None
    logger.info("✓ HRM state initialized successfully")
    
    return hrm_loop, curriculum_config, model_config


def test_integration():
    """Test complete integration of all components."""
    logger.info("Testing complete integration...")
    
    # Test all components
    data_loader, task_generator, sampler, phases = test_data_pipeline()
    model, params, model_config = test_model_creation()
    hrm_loop, curriculum_config, _ = test_hrm_training_loop()
    
    logger.info("✓ All components integrated successfully")
    
    # Test phase transition logic
    current_phase = phases[0]  # Start with phase 1
    assert not current_phase.hrm_enabled
    logger.info(f"✓ Phase 1 configuration: HRM disabled")
    
    next_phase = phases[1]  # Phase 2
    assert next_phase.hrm_enabled
    assert next_phase.hrm_cycles == 2
    assert next_phase.hrm_steps == 2
    logger.info(f"✓ Phase 2 configuration: HRM enabled with N={next_phase.hrm_cycles}, T={next_phase.hrm_steps}")
    
    final_phase = phases[2]  # Phase 3
    assert final_phase.hrm_enabled
    assert final_phase.hrm_use_act
    assert final_phase.hrm_cycles == 4
    assert final_phase.hrm_steps == 4
    logger.info(f"✓ Phase 3 configuration: Full HRM with ACT, N={final_phase.hrm_cycles}, T={final_phase.hrm_steps}")
    
    return True


def test_shape_consistency():
    """Test shape consistency across the pipeline."""
    logger.info("Testing shape consistency...")
    
    batch_size, seq_len = 2, 128
    vocab_size = 32000
    
    # Create model
    # Create model config
    config = ValkyrieConfig(
        vocab_size=32000,
        d_model=512,
        n_layers=4,
        n_heads=8,
        max_position_embeddings=1024,
        use_s5=True,
        use_hrm=True
    )
    
    model = ValkyrieModel(config)
    key = jax.random.PRNGKey(42)
    
    # Create input batch (simulating PackedSequence format)
    input_ids = jnp.ones((batch_size, seq_len), dtype=jnp.int32)
    attention_mask = jnp.ones((batch_size, seq_len), dtype=jnp.int32)
    
    # Initialize and run forward pass
    params = model.init(key, input_ids)
    outputs = model.apply(params, input_ids, attention_mask=attention_mask)
    
    # Verify shapes
    assert outputs.logits.shape == (batch_size, seq_len, vocab_size)
    assert outputs.s5_states is not None
    logger.info(f"✓ Shape consistency verified: {outputs.logits.shape}")
    
    return True


def run_all_tests():
    """Run all integration tests."""
    logger.info("Starting PLAN Training Pipeline Integration Tests")
    logger.info("=" * 60)
    
    try:
        # Run individual tests
        test_data_pipeline()
        test_model_creation() 
        test_hrm_training_loop()
        test_integration()
        test_shape_consistency()
        
        logger.info("=" * 60)
        logger.info("🎉 ALL TESTS PASSED! Training pipeline integration successful.")
        logger.info("✓ Data pipeline components working")
        logger.info("✓ Model creation and forward pass working")
        logger.info("✓ HRM training loop integration working")
        logger.info("✓ Phase-based curriculum working")
        logger.info("✓ Shape consistency verified")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)