"""Example: TPU v4-32 Training with 3D Mesh and Multi-Host Setup

This example demonstrates how to set up and run training on TPU v4-32 with:
- 4-host × 8-chip configuration (32 total chips)
- 3D mesh topology: 4×4×2 (data × model × fsdp)
- Multi-host distributed initialization
- Proper gradient synchronization across hosts

Usage:
    # Set environment variables for TPU v4-32
    export TPU_MESH_CONFIG="4,4,2"
    export TPU_AXIS_NAMES="data,model,fsdp"
    export JAX_PLATFORMS="tpu"
    
    # Run on each host (process_index 0-3)
    gcloud compute tpus tpu-vm ssh node-1 \
        --zone=us-central2-b \
        --worker=all \
        --command="python3.11 -m examples.tpu_v4_32_training"
"""

import sys
import os
import logging
import time
from typing import Dict, Any, Tuple

# Add the codebase directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))  # /home/ravkeave/valkyrie/examples
valkyrie_root = os.path.dirname(current_dir)  # /home/ravkeave/valkyrie
codebase_path = os.path.join(valkyrie_root, 'codebase')  # /home/ravkeave/valkyrie/codebase
sys.path.insert(0, codebase_path)

import jax
import jax.numpy as jnp
import numpy as np
from jax import random
from jax.experimental import multihost_utils
from jax.sharding import Mesh, PartitionSpec as P
import optax
from flax import linen as nn
from flax.training import train_state

# Import directly from specific modules to avoid circular imports
from src.sharding.distributed_init import setup_multi_host_environment, print_distributed_info
from src.sharding.mesh_setup import create_mesh_v4_32
from src.train.step_fn import create_train_step, create_eval_step

# Import get_model_specs directly to avoid circular dependency
from src.sharding.partition_specs import get_model_specs, get_training_specs
from src.model.config import ValkyrieConfig
from src.model.model import ValkyrieModel

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_synthetic_batch(config: ValkyrieConfig, batch_size: int = 32) -> Dict[str, jnp.ndarray]:
    """Create synthetic training batch for testing."""
    return {
        'input_ids': jnp.ones((batch_size, config.max_seq_len), dtype=jnp.int32),
        'attention_mask': jnp.ones((batch_size, config.max_seq_len), dtype=jnp.bool_),
        'labels': jnp.ones((batch_size, config.max_seq_len), dtype=jnp.int32),
    }


def create_model_and_config() -> Tuple[ValkyrieModel, ValkyrieConfig]:
    """Create model and configuration for TPU v4-32 training."""
    config = ValkyrieConfig(
        vocab_size=32000,
        hidden_size=4096,
        num_layers=32,
        num_heads=32,
        intermediate_size=11008,
        max_seq_len=2048,
        dropout_rate=0.1,
        use_bias=False,
        rope_theta=10000.0,
        rope_scaling=None,
        attention_dropout=0.0,
        use_flash_attention=True,
        use_mamba=False,
        mamba_config=None,
        use_s5=False,
        s5_config=None,
    )
    
    model = ValkyrieModel(config)
    return model, config


def initialize_training_state(
    model: ValkyrieModel, 
    config: ValkyrieConfig, 
    mesh: Any,
    learning_rate: float = 1e-4
) -> train_state.TrainState:
    """Initialize training state with proper sharding."""
    
    # Create optimizer
    optimizer = optax.adamw(
        learning_rate=learning_rate,
        weight_decay=0.01,
        b1=0.9,
        b2=0.95,
        eps=1e-8
    )
    
    # Initialize parameters
    rng = jax.random.PRNGKey(42)
    dummy_input = jnp.ones((1, config.max_seq_len), dtype=jnp.int32)
    
    with mesh:
        params = model.init(rng, dummy_input)
        state = train_state.TrainState.create(
            apply_fn=model.apply,
            params=params,
            tx=optimizer
        )
    
    return state


def run_training_step_test(
    state: train_state.TrainState,
    model: ValkyrieModel,
    config: ValkyrieConfig,
    mesh: Any,
    batch: Dict[str, jnp.ndarray]
) -> Tuple[train_state.TrainState, Dict[str, float]]:
    """Run a single training step test."""
    
    # Create training step function with 3D mesh support
    train_step_fn = create_train_step(
        model=model,
        optimizer=state.tx,
        config=config,
        mesh=mesh,
        mixed_precision=True,
        use_2d_sharding=True,  # Enable 2D tensor parallelism
        use_3d_mesh=True       # Enable 3D mesh (v4-32)
    )
    
    # Run training step
    start_time = time.time()
    new_state, metrics = train_step_fn(state, batch)
    step_time = time.time() - start_time
    
    # Add timing metrics
    metrics['step_time_ms'] = step_time * 1000
    metrics['tokens_per_second'] = (batch['input_ids'].size / step_time)
    
    return new_state, metrics


def run_eval_step_test(
    state: train_state.TrainState,
    model: ValkyrieModel,
    config: ValkyrieConfig,
    mesh: Any,
    batch: Dict[str, jnp.ndarray]
) -> Dict[str, float]:
    """Run a single evaluation step test."""
    
    # Create evaluation step function with 3D mesh support
    eval_step_fn = create_eval_step(
        model=model,
        config=config,
        mesh=mesh,
        mixed_precision=True,
        use_2d_sharding=True,  # Enable 2D tensor parallelism
        use_3d_mesh=True       # Enable 3D mesh (v4-32)
    )
    
    # Run evaluation step
    start_time = time.time()
    metrics = eval_step_fn(state, batch)
    step_time = time.time() - start_time
    
    # Add timing metrics
    metrics['eval_time_ms'] = step_time * 1000
    
    return metrics


def main():
    """Main training loop for TPU v4-32."""
    
    logger.info("Starting TPU v4-32 training example")
    
    try:
        # Step 1: Initialize multi-host distributed runtime
        logger.info("=== Step 1: Multi-Host Initialization ===")
        dist_config = setup_multi_host_environment()
        print_distributed_info()
        
        # Step 2: Create TPU mesh with 3D topology
        logger.info("=== Step 2: TPU Mesh Setup ===")
        mesh = setup_tpu_mesh(
            device_count=32,  # TPU v4-32
            use_global=True,
            validate=True,
            print_info=True
        )
        
        # Verify mesh configuration
        expected_shape = (4, 4, 2)  # data × model × fsdp
        expected_axes = ('data', 'model', 'fsdp')
        
        if mesh.shape != expected_shape:
            raise ValueError(f"Expected mesh shape {expected_shape}, got {mesh.shape}")
        if mesh.axis_names != expected_axes:
            raise ValueError(f"Expected axis names {expected_axes}, got {mesh.axis_names}")
        
        logger.info("✓ Mesh configuration verified for TPU v4-32")
        
        # Step 3: Create model and configuration
        logger.info("=== Step 3: Model Creation ===")
        model, config = create_model_and_config()
        logger.info(f"Model config: {config.hidden_size}H, {config.num_layers}L, {config.num_heads}A")
        
        # Step 4: Initialize training state
        logger.info("=== Step 4: Training State Initialization ===")
        with mesh:
            state = initialize_training_state(model, config, mesh)
        
        # Count parameters
        param_count = sum(x.size for x in jax.tree_util.tree_leaves(state.params))
        logger.info(f"Model parameters: {param_count:,} ({param_count/1e9:.2f}B)")
        
        # Step 5: Create synthetic batch
        logger.info("=== Step 5: Batch Creation ===")
        batch_size = 32  # Global batch size across all devices
        batch = create_synthetic_batch(config, batch_size)
        
        # Verify batch sharding
        with mesh:
            # Batch should be sharded along 'data' axis (4-way)
            local_batch_size = batch_size // mesh.shape[0]  # 32 // 4 = 8 per data parallel group
            logger.info(f"Global batch size: {batch_size}, Local batch size: {local_batch_size}")
        
        # Step 6: Run training step test
        logger.info("=== Step 6: Training Step Test ===")
        with mesh:
            new_state, train_metrics = run_training_step_test(state, model, config, mesh, batch)
        
        logger.info("Training step completed successfully!")
        logger.info(f"Loss: {train_metrics.get('loss', 'N/A'):.4f}")
        logger.info(f"Step time: {train_metrics.get('step_time_ms', 'N/A'):.1f}ms")
        logger.info(f"Tokens/sec: {train_metrics.get('tokens_per_second', 'N/A'):.0f}")
        
        # Step 7: Run evaluation step test
        logger.info("=== Step 7: Evaluation Step Test ===")
        with mesh:
            eval_metrics = run_eval_step_test(new_state, model, config, mesh, batch)
        
        logger.info("Evaluation step completed successfully!")
        logger.info(f"Eval loss: {eval_metrics.get('loss', 'N/A'):.4f}")
        logger.info(f"Accuracy: {eval_metrics.get('accuracy', 'N/A'):.4f}")
        logger.info(f"Perplexity: {eval_metrics.get('perplexity', 'N/A'):.2f}")
        logger.info(f"Eval time: {eval_metrics.get('eval_time_ms', 'N/A'):.1f}ms")
        
        # Step 8: Multi-step training test
        logger.info("=== Step 8: Multi-Step Training Test ===")
        num_steps = 5
        
        current_state = new_state
        for step in range(num_steps):
            with mesh:
                current_state, step_metrics = run_training_step_test(
                    current_state, model, config, mesh, batch
                )
            
            logger.info(f"Step {step+1}/{num_steps}: "
                       f"loss={step_metrics.get('loss', 'N/A'):.4f}, "
                       f"time={step_metrics.get('step_time_ms', 'N/A'):.1f}ms")
        
        logger.info("Multi-step training completed successfully!")
        
        # Step 9: Performance summary
        logger.info("=== Step 9: Performance Summary ===")
        final_metrics = {
            'distributed_config': dist_config,
            'mesh_shape': mesh.shape,
            'mesh_axes': mesh.axis_names,
            'model_params': param_count,
            'global_batch_size': batch_size,
            'local_batch_size': batch_size // mesh.shape[0],
            'final_loss': step_metrics.get('loss', 'N/A'),
            'avg_step_time_ms': step_metrics.get('step_time_ms', 'N/A'),
            'tokens_per_second': step_metrics.get('tokens_per_second', 'N/A'),
        }
        
        logger.info("=== FINAL RESULTS ===")
        for key, value in final_metrics.items():
            logger.info(f"{key}: {value}")
        
        logger.info("TPU v4-32 training example completed successfully! 🎉")
        
    except Exception as e:
        logger.error(f"Training example failed: {e}")
        raise


if __name__ == "__main__":
    main()