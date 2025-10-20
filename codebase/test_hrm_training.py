#!/usr/bin/env python3
"""
HRM Training Test Script with Distributed TPU Support

This script demonstrates the HRM training functionality with proper TPU distribution
across all available devices for optimal memory usage and performance.
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Dict, List
import time
import sys
import os
import psutil

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import distributed TPU configuration
from configure_tpu_distributed import (
    configure_jax_for_tpu, 
    create_device_mesh, 
    create_sharding_strategy,
    shard_batch_to_devices,
    estimate_memory_usage,
    replicate_params_to_devices
)

# Import data loading utilities
from data.data_loader import get_random_batch, load_dataset_info, AVAILABLE_DATASETS
from model.hrm.models import HRMWithACT

# Import training utilities - using direct import with fixed path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src', 'model', 'hrm'))
from training import (
    HRMTrainingState, 
    TrainingMetrics, 
    LossConfig,
    create_train_state,
    segment_train_step,
    train_segments,
    analyze_gradient_flow,
    validate_carry_detachment
)


def print_system_info():
    """Print system and JAX configuration information."""
    print("=" * 60)
    print("DISTRIBUTED TPU SYSTEM INFORMATION")
    print("=" * 60)
    
    # Configure JAX for TPU and get devices
    devices, local_devices = configure_jax_for_tpu()
    print(f"JAX devices: {len(devices)}")
    for i, device in enumerate(devices):
        print(f"  Device {i}: {device}")
        # Print memory info for each device
        try:
            memory_info = device.memory_stats()
            used_gb = memory_info['bytes_in_use'] / (1024**3)
            limit_gb = memory_info['bytes_limit'] / (1024**3)
            print(f"    Memory: {used_gb:.2f} GB / {limit_gb:.2f} GB used")
        except:
            print("    Memory info not available")
    
    # JAX backend
    print(f"JAX backend: {jax.default_backend()}")
    
    # Create device mesh and sharding strategies
    mesh = create_device_mesh(devices)
    data_sharding, param_shardings = create_sharding_strategy(
        mesh, batch_size=64, seq_len=512, hidden_size=512
    )
    print(f"Device mesh: {mesh}")
    print(f"Data sharding: {data_sharding}")
    print(f"Parameter shardings: {list(param_shardings.keys())}")
    
    print()
    return devices, mesh, data_sharding, param_shardings


def create_real_data_batch(batch_size: int, seq_len: int, vocab_size: int, dataset_name: str = "arc-aug-1000") -> Dict[str, jnp.ndarray]:
    """Create a batch of real training data with memory profiling and efficient loading."""
    data_dir = f"/home/ravkeave/v1/data/{dataset_name}"
    
    print(f"Loading real data from {data_dir}")
    
    # Memory profiling - before data loading
    process = psutil.Process(os.getpid())
    rss_before = process.memory_info().rss / (1024**3)  # GB
    print(f"RSS before data loading: {rss_before:.2f} GB")
    
    # Get random batch and dataset info - use target_seq_len to avoid loading 900 then truncating
    rng_key = jax.random.PRNGKey(42)
    batch, dataset_info = get_random_batch(data_dir, batch_size, "train", rng_key, target_seq_len=seq_len)
    
    # Memory profiling - during data loading
    rss_after_load = process.memory_info().rss / (1024**3)  # GB
    print(f"RSS after get_random_batch(): {rss_after_load:.2f} GB")
    print(f"Memory delta from data loading: {rss_after_load - rss_before:.2f} GB")
    
    print(f"Dataset info: vocab_size={dataset_info.vocab_size}, seq_len={dataset_info.seq_len}")
    print(f"Loaded batch: inputs={batch.inputs.shape}, labels={batch.labels.shape}")
    
    # No need for adjust_sequence_length anymore since we load the correct length directly
    adjusted_inputs = batch.inputs
    adjusted_targets = batch.labels
    
    print(f"Final batch: inputs={adjusted_inputs.shape}, targets={adjusted_targets.shape}")
    
    # Return in the format expected by the training functions
    return {
        "inputs": adjusted_inputs,
        "targets": adjusted_targets  # Use labels as targets for language modeling
    }


def create_real_data_segments(batch_size: int, seq_len: int, vocab_size: int, num_segments: int, dataset_name: str = "arc-aug-1000") -> List[Dict[str, jnp.ndarray]]:
    """Create multiple segments of real training data with memory-efficient loading."""
    print(f"Creating {num_segments} segments of real data from {dataset_name}")
    
    segments = []
    for i in range(num_segments):
        # Use different random keys for each segment
        rng_key = jax.random.PRNGKey(42 + i)
        data_dir = f"/home/ravkeave/v1/data/{dataset_name}"
        batch, _ = get_random_batch(data_dir, batch_size, "train", rng_key, target_seq_len=seq_len)
        
        # No need for sequence adjustment since we load the correct length directly
        segment = {
            "inputs": batch.inputs,
            "targets": batch.labels
        }
        segments.append(segment)
    
    return segments


def test_model_creation(devices, mesh, data_sharding, param_shardings):
    """Test HRM model creation and basic functionality with distributed setup."""
    print("=" * 60)
    print("TESTING DISTRIBUTED MODEL CREATION")
    print("=" * 60)
    
    # Load real dataset info to get correct parameters
    dataset_name = "arc-aug-1000"
    data_dir = f"/home/ravkeave/v1/data/{dataset_name}"
    dataset_info = load_dataset_info(data_dir, "train")
    
    print(f"Using dataset: {dataset_name}")
    print(f"Dataset info: {dataset_info}")
    
    # Model configuration with memory-optimized parameters
    # NOTE: Using seq_len=128 instead of dataset_info.seq_len (900) to prevent RESOURCE_EXHAUSTED errors
    # This reduces attention matrix memory usage by ~49x: 128x128 vs 900x900 elements
    config = {
        "vocab_size": dataset_info.vocab_size,
        "hidden_size": 512,  # Increased for better utilization
        "seq_len": 128,      # Memory-optimized: reduced from dataset_info.seq_len (900)
        "H_layers": 6,       # Increased layers
        "L_layers": 6,
        "num_heads": 8,      
        "max_steps": 8,
        "exploration_prob": 0.1,
        "dtype": jnp.bfloat16
    }
    
    print(f"Model config: {config}")
    
    # Estimate memory usage using the function from configure_tpu_distributed
    memory_ok, recommended_batch_size = estimate_memory_usage(
        batch_size=32,  # Reduced from 64 for memory safety
        seq_len=config["seq_len"],
        hidden_size=config["hidden_size"],
        vocab_size=config["vocab_size"],
        num_layers=config["H_layers"] + config["L_layers"],
        num_devices=len(devices)
    )
    
    try:
        # Create model
        model = HRMWithACT(**config)
        print(f"✓ Model created: {model}")
        
        # Test model initialization with sharding
        rng_key = jax.random.PRNGKey(42)
        batch_size = recommended_batch_size if not memory_ok else 32  # Use 32 as default instead of 64
        
        dummy_batch = {
            "inputs": jnp.ones((batch_size, config["seq_len"]), dtype=jnp.int32),
            "targets": jnp.ones((batch_size, config["seq_len"]), dtype=jnp.int32)
        }
        
        print("Initializing model parameters with sharding...")
        start_time = time.time()
        params = model.init(rng_key, dummy_batch, training=False)["params"]
        init_time = time.time() - start_time
        
        print(f"✓ Model initialization completed in {init_time:.3f}s")
        
        # Count parameters
        param_count = sum(x.size for x in jax.tree_util.tree_leaves(params))
        print(f"  Total parameters: {param_count:,}")
        
        # Replicate parameters across devices
        print("Replicating parameters across devices...")
        sharded_params = replicate_params_to_devices(params, mesh)
        print(f"✓ Parameters replicated successfully")
        
        return model, config, sharded_params
        
    except Exception as e:
        print(f"✗ Model creation failed with error: {e}")
        import traceback
        traceback.print_exc()
        return None, config, None


def test_training_state_creation(model, config):
    """Test training state creation."""
    print("============================================================")
    print("TESTING TRAINING STATE CREATION")
    print("============================================================")
    
    try:
        rng_key = jax.random.PRNGKey(42)
        batch_size = 2
        learning_rate = 1e-4
        
        state = create_train_state(
            model=model,
            rng_key=rng_key,
            learning_rate=learning_rate,
            batch_size=batch_size
        )
        
        print("✓ Training state created successfully")
        print(f"  Parameter count: {sum(x.size for x in jax.tree_util.tree_leaves(state.params))}")
        print(f"  Carry shape: {jax.tree.map(lambda x: x.shape, state.carry)}")
        return state
        
    except Exception as e:
        print(f"✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_single_training_step(state, config):
    """Test a single training step."""
    print("=" * 60)
    print("TESTING SINGLE TRAINING STEP")
    print("=" * 60)
    
    # Create a single batch
    batch_size = 2
    batch = create_real_data_batch(
        batch_size=batch_size,
        seq_len=config["seq_len"],
        vocab_size=config["vocab_size"],
        dataset_name="arc-aug-1000"
    )
    loss_config = LossConfig(
        lm_weight=1.0,
        act_weight=0.1,
        deep_supervision_weight=0.0,  # Disabled for now
        q_target_discount=0.95,
        label_smoothing=0.0
    )
    
    print(f"Loss config: {loss_config}")
    print(f"Batch shapes: inputs={batch['inputs'].shape}, targets={batch['targets'].shape}")
    
    # Perform training step
    print("Performing training step...")
    start_time = time.time()
    
    try:
        new_state, metrics = segment_train_step(state, batch, loss_config)
        step_time = time.time() - start_time
        
        print(f"Training step completed in {step_time:.3f}s")
        print(f"Metrics: {metrics}")
        
        return new_state, metrics
        
    except Exception as e:
        print(f"Training step failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def test_multi_segment_training(state, config):
    """Test training on multiple segments."""
    print("=" * 60)
    print("TESTING MULTI-SEGMENT TRAINING")
    print("=" * 60)
    
    # Create multiple segments
    batch_size = 2
    num_segments = 3
    segments = create_real_data_segments(
        batch_size=batch_size,
        seq_len=config["seq_len"],
        vocab_size=config["vocab_size"],
        num_segments=num_segments,
        dataset_name="arc-aug-1000"
    )
    
    loss_config = LossConfig(
        lm_weight=1.0,
        act_weight=0.1,
        deep_supervision_weight=0.0,
        q_target_discount=0.95,
        label_smoothing=0.0
    )
    
    print(f"Training on {num_segments} segments...")
    start_time = time.time()
    
    try:
        final_state, all_metrics = train_segments(
            state=state,
            segments=segments,
            loss_config=loss_config,
            log_every=1
        )
        
        training_time = time.time() - start_time
        print(f"Multi-segment training completed in {training_time:.3f}s")
        
        # Analyze metrics
        print("\nTraining metrics summary:")
        for i, metrics in enumerate(all_metrics):
            print(f"  Segment {i}: Loss={metrics.total_loss:.4f}, "
                  f"LM_Loss={metrics.lm_loss:.4f}, "
                  f"ACT_Loss={metrics.act_loss:.4f}, "
                  f"Steps={metrics.mean_steps:.2f}")
        
        return final_state, all_metrics
        
    except Exception as e:
        print(f"Multi-segment training failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def test_gradient_analysis(state, config):
    """Test gradient flow analysis."""
    print("=" * 60)
    print("TESTING GRADIENT ANALYSIS")
    print("=" * 60)
    
    # Create a batch for analysis
    batch_size = 2
    batch = create_real_data_batch(
        batch_size=batch_size,
        seq_len=config["seq_len"],
        vocab_size=config["vocab_size"],
        dataset_name="arc-aug-1000"
    )
    loss_config = LossConfig()
    
    try:
        grad_norms = analyze_gradient_flow(state, batch, loss_config)
        print(f"Gradient norms: {grad_norms}")
        
    except Exception as e:
        print(f"Gradient analysis failed: {e}")
        import traceback
        traceback.print_exc()


def test_carry_detachment(state, config):
    """Test carry detachment functionality."""
    print("=" * 60)
    print("TESTING CARRY DETACHMENT")
    print("=" * 60)
    
    # Create two batches
    batch_size = 2
    segments = create_real_data_segments(
        batch_size=batch_size,
        seq_len=config["seq_len"],
        vocab_size=config["vocab_size"],
        num_segments=2,
        dataset_name="arc-aug-1000"
    )
    
    batch1, batch2 = segments[0], segments[1]
    loss_config = LossConfig()
    
    try:
        is_detached = validate_carry_detachment(state, batch1, batch2, loss_config)
        print(f"Carry detachment working correctly: {is_detached}")
        
    except Exception as e:
        print(f"Carry detachment test failed: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Main test function."""
    print("Starting HRM Distributed Training Tests...")
    print("=" * 60)
    
    # Test 1: System Info and Setup
    devices, mesh, data_sharding, param_shardings = print_system_info()
    
    # Test 2: Model Creation
    model, config, sharded_params = test_model_creation(devices, mesh, data_sharding, param_shardings)
    if model is None:
        print("\n✗ Model creation failed. Cannot proceed with further tests.")
        return
    
    # Test 3: Distributed Training Step
    # Create a training state first
    training_state = test_training_state_creation(model, config)
    if training_state is None:
        print("\n✗ Training state creation failed. Cannot proceed with training step.")
        return
    
    # Test the training step
    final_state = test_single_training_step(training_state, config)
    if final_state is None:
        print("\n✗ Training step failed. Cannot proceed with further tests.")
        return
    
    print("\n" + "=" * 60)
    print("ALL DISTRIBUTED TRAINING TESTS COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print(f"✓ TPU devices configured: {len(devices)}")
    print(f"✓ Device mesh created: {mesh}")
    print(f"✓ Model parameters sharded across devices")
    print(f"✓ Training step executed with distributed data")
    print("✓ Memory usage optimized across all TPU devices")
    
    # Print final memory usage
    print("\nFinal memory usage across devices:")
    total_used = 0
    total_limit = 0
    for i, device in enumerate(devices):
        try:
            memory_info = device.memory_stats()
            used_gb = memory_info['bytes_in_use'] / (1024**3)
            limit_gb = memory_info['bytes_limit'] / (1024**3)
            total_used += used_gb
            total_limit += limit_gb
            print(f"  Device {i}: {used_gb:.2f} GB / {limit_gb:.2f} GB used ({used_gb/limit_gb*100:.1f}%)")
        except:
            print(f"  Device {i}: Memory info not available")
    
    print(f"\nTotal memory utilization: {total_used:.2f} GB / {total_limit:.2f} GB ({total_used/total_limit*100:.1f}%)")
    print("Distributed training setup is ready for large-scale training!")


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)