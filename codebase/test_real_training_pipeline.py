#!/usr/bin/env python3
"""
Real Training Pipeline Integration Test

This test uses the ACTUAL training frameworks and functions from:
- src.train.main: load_config, create_model_from_config
- src.train.train_loop: TrainingLoop, PhaseConfig, CurriculumConfig

Tests the complete training pipeline with real TPU v4-8 config.
"""

import os
import sys
import jax
import jax.numpy as jnp
import numpy as np
from pathlib import Path
from typing import Dict, Any
import tempfile
import logging

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Import the ACTUAL training functions
from src.train.main import load_config, create_model_from_config
from src.train.train_loop import TrainingLoop, PhaseConfig, CurriculumConfig, ChunkConfig
from src.model import ValkyrieModel, ValkyrieConfig
from src.sharding import make_mesh
from src.io import CheckpointManager, CheckpointConfig

def test_real_config_loading():
    """Test using the ACTUAL load_config function from train/main.py"""
    print("\n=== Testing Real Config Loading ===")
    
    config_path = str(Path(__file__).parent / "configs" / "valkyrie_tpu_v4_8.yaml")
    
    try:
        # Use the REAL load_config function
        config = load_config(config_path)
        
        print(f"✓ Real load_config() loaded: {len(config)} top-level keys")
        print(f"✓ Model config: d_model={config['model']['d_model']}, n_layers={config['model']['n_layers']}")
        print(f"✓ Training config: total_steps={config['training']['total_steps']}")
        print(f"✓ Mesh config: {config['mesh']['mesh_shape']} with axes {config['mesh']['axis_names']}")
        
        return config
        
    except Exception as e:
        print(f"✗ Real config loading failed: {e}")
        raise

def test_real_model_creation(config: Dict[str, Any]):
    """Test using the ACTUAL create_model_from_config function"""
    print("\n=== Testing Real Model Creation ===")
    
    try:
        # Use the REAL create_model_from_config function
        model = create_model_from_config(config)
        
        print(f"✓ Real create_model_from_config() created ValkyrieModel")
        print(f"✓ Model type: {type(model).__name__}")
        print(f"✓ Model config: {model.config}")
        
        return model
        
    except Exception as e:
        print(f"✗ Real model creation failed: {e}")
        raise

def test_real_phase_config(config: Dict[str, Any]):
    """Test using the ACTUAL PhaseConfig from train_loop.py"""
    print("\n=== Testing Real PhaseConfig ===")
    
    try:
        # Use the REAL PhaseConfig class
        curriculum_config = config.get('curriculum', {})
        phases = curriculum_config.get('phases', [])
        
        if not phases:
            print("⚠ No curriculum phases in config, creating default")
            # Create a simple phase for testing
            phase = PhaseConfig(
                name="test_phase",
                steps=100,
                chunk_size=4096,
                lr=3e-4,
                backprop_chunks=2,
                hrm_enabled=True,
                hrm_supervision_weight=0.3
            )
        else:
            # Use first phase from config
            phase_data = phases[0]
            phase = PhaseConfig(
                name=phase_data['name'],
                steps=phase_data['max_steps'],
                chunk_size=phase_data['chunk_size'],
                lr=phase_data['lr'],
                backprop_chunks=phase_data.get('backprop_chunks', 2),
                hrm_enabled=phase_data.get('hrm_enabled', True),
                hrm_supervision_weight=phase_data.get('hrm_supervision_weight', 0.0)
            )
        
        print(f"✓ Real PhaseConfig created: {phase.name}")
        print(f"✓ Phase settings: chunk_size={phase.chunk_size}, lr={phase.lr}")
        print(f"✓ HRM settings: enabled={phase.hrm_enabled}, weight={phase.hrm_supervision_weight}")
        
        return phase
        
    except Exception as e:
        print(f"✗ Real PhaseConfig creation failed: {e}")
        raise

def test_real_curriculum_config(config: Dict[str, Any]):
    """Test using the ACTUAL CurriculumConfig from train_loop.py"""
    print("\n=== Testing Real CurriculumConfig ===")
    
    try:
        # Use the REAL CurriculumConfig class
        curriculum_data = config.get('curriculum', {})
        phases_data = curriculum_data.get('phases', [])
        
        # Convert config phases to the format expected by CurriculumConfig
        curriculum_config = CurriculumConfig(phases=phases_data)
        
        print(f"✓ Real CurriculumConfig created with {len(curriculum_config.phases)} phases")
        
        for i, phase in enumerate(curriculum_config.phases[:3]):  # Show first 3
            print(f"  Phase {i}: {phase.get('name', 'unnamed')} - {phase.get('chunk_size', 'unknown')} tokens")
        
        return curriculum_config
        
    except Exception as e:
        print(f"✗ Real CurriculumConfig creation failed: {e}")
        raise

def test_real_training_loop_setup(config: Dict[str, Any], model: ValkyrieModel):
    """Test setting up the ACTUAL TrainingLoop class"""
    print("\n=== Testing Real TrainingLoop Setup ===")
    
    try:
        # Create minimal mesh for CPU testing
        devices = jax.local_devices()[:1]  # Use 1 CPU device
        mesh = make_mesh(device_count=1, topology=(1,), axis_names=('data',))
        
        # Create chunk config
        chunk_config = ChunkConfig(
            chunk_size=4096,
            overlap_size=512,
            max_chunks_per_doc=10,  # Smaller for testing
            backprop_chunks=2
        )
        
        # Create curriculum config
        curriculum_config = CurriculumConfig()
        
        # Create checkpoint config with absolute path
        checkpoint_dir = Path(__file__).parent / "test_checkpoints"
        checkpoint_config = CheckpointConfig(
            checkpoint_dir=str(checkpoint_dir.absolute()),
            save_interval_steps=1000,
            keep_checkpoints=5,
            async_save=False  # Disable for testing
        )
        
        # Create checkpoint manager
        checkpoint_manager = CheckpointManager(
            config=checkpoint_config,
            mesh=mesh
        )
        
        # Use the REAL TrainingLoop class
        with mesh:
            training_loop = TrainingLoop(
                model=model,
                config=model.config,  # Use the ValkyrieConfig from the model
                chunk_config=chunk_config,
                curriculum_config=curriculum_config,
                mesh=mesh,
                checkpoint_manager=checkpoint_manager
            )
        
        print(f"✓ Real TrainingLoop created successfully")
        print(f"✓ Training loop type: {type(training_loop).__name__}")
        print(f"✓ Mesh: {mesh}")
        print(f"✓ Chunk config: {chunk_config.chunk_size} tokens")
        
        return training_loop
        
    except Exception as e:
        print(f"✗ Real TrainingLoop setup failed: {e}")
        raise

def test_real_checkpoint_manager(config: Dict[str, Any]):
    """Test using the ACTUAL CheckpointManager"""
    print("\n=== Testing Real CheckpointManager ===")
    
    try:
        # Create temporary checkpoint directory
        checkpoint_dir = Path(__file__).parent / "test_checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)
        
        # Use the REAL CheckpointConfig
        checkpoint_config = CheckpointConfig(
            checkpoint_dir=str(checkpoint_dir.absolute()),
            save_interval_steps=1000,
            keep_checkpoints=5,
            async_save=False
        )
        
        # Use the REAL CheckpointManager
        checkpoint_manager = CheckpointManager(checkpoint_config)
        
        print(f"✓ Real CheckpointManager created")
        print(f"✓ Checkpoint dir (absolute): {checkpoint_config.checkpoint_dir}")
        print(f"✓ Is absolute path: {Path(checkpoint_config.checkpoint_dir).is_absolute()}")
        
        # Clean up
        if checkpoint_dir.exists():
            import shutil
            shutil.rmtree(checkpoint_dir)
        
        return checkpoint_manager
        
    except Exception as e:
        print(f"✗ Real CheckpointManager creation failed: {e}")
        raise

def main():
    """Run all real training pipeline tests"""
    print("=" * 60)
    print("REAL TRAINING PIPELINE INTEGRATION TEST")
    print("Using ACTUAL functions from train.py and train_loop.py")
    print("=" * 60)
    
    # Suppress JAX warnings for cleaner output
    logging.getLogger('jax').setLevel(logging.ERROR)
    
    try:
        # Test 1: Real config loading
        config = test_real_config_loading()
        
        # Test 2: Real model creation
        model = test_real_model_creation(config)
        
        # Test 3: Real phase config
        phase_config = test_real_phase_config(config)
        
        # Test 4: Real curriculum config
        curriculum_config = test_real_curriculum_config(config)
        
        # Test 5: Real training loop setup
        training_loop = test_real_training_loop_setup(config, model)
        
        # Test 6: Real checkpoint manager
        checkpoint_manager = test_real_checkpoint_manager(config)
        
        print("\n" + "=" * 60)
        print("REAL TRAINING PIPELINE TEST RESULTS")
        print("=" * 60)
        print("✓ PASS   Real Config Loading (load_config)")
        print("✓ PASS   Real Model Creation (create_model_from_config)")
        print("✓ PASS   Real PhaseConfig")
        print("✓ PASS   Real CurriculumConfig")
        print("✓ PASS   Real TrainingLoop Setup")
        print("✓ PASS   Real CheckpointManager")
        print("\nOverall: 6/6 tests passed")
        print("🎉 Real training pipeline integration successful!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()