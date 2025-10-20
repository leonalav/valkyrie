#!/usr/bin/env python3
"""
Test HRM training with distributed TPU configuration.
This test uses the proper multi-device setup to utilize all 4 TPU devices.
"""

import os
import sys
import jax
import jax.numpy as jnp
import optax
from flax import linen as nn
from flax.training import train_state
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import our modules
from src.model.hrm.models.hrm_act import HRMWithACT
from src.training import TrainingState, TrainingMetrics, compute_total_loss, detach_carry
from configure_tpu_distributed import configure_jax_for_tpu, create_device_mesh, create_sharding_strategies, shard_batch_data

def create_test_batch(batch_size, seq_len, vocab_size):
    """Create a test batch with proper shapes."""
    return {
        'input_ids': jnp.ones((batch_size, seq_len), dtype=jnp.int32),
        'labels': jnp.ones((batch_size, seq_len), dtype=jnp.int32),
        'attention_mask': jnp.ones((batch_size, seq_len), dtype=jnp.bool_)
    }

def test_distributed_training():
    """Test HRM training with distributed TPU configuration."""
    
    print("=== Testing Distributed HRM Training ===")
    
    # Configure JAX for TPU
    devices = configure_jax_for_tpu()
    print(f"Using {len(devices)} TPU devices")
    
    # Create device mesh and sharding strategies
    mesh = create_device_mesh(devices)
    sharding_strategies = create_sharding_strategies(mesh)
    
    # Model configuration - using correct parameter names
    model_config = {
        'vocab_size': 32000,
        'hidden_size': 512,
        'seq_len': 512,
        'H_cycles': 2,
        'L_cycles': 2,
        'H_layers': 4,
        'L_layers': 4,
        'num_heads': 8,
        'max_steps': 5,
        'dtype': jnp.bfloat16
    }
    
    # Training configuration
    batch_size = 64  # Will be distributed across 4 devices (16 per device)
    seq_len = 512
    learning_rate = 1e-4
    
    print(f"Model config: {model_config}")
    print(f"Batch size: {batch_size} (distributed across {len(devices)} devices)")
    
    # Initialize model
    model = HRMWithACT(**model_config)
    
    # Create test batch
    batch = create_test_batch(batch_size, seq_len, model_config['vocab_size'])
    print(f"Created batch with shapes: {jax.tree.map(lambda x: x.shape, batch)}")
    
    # Shard the batch across devices
    sharded_batch = shard_batch_data(batch, sharding_strategies['data'])
    print(f"Sharded batch: {jax.tree.map(lambda x: x.sharding, sharded_batch)}")
    
    # Initialize parameters with proper sharding
    rng = jax.random.PRNGKey(42)
    
    # Create dummy input for initialization
    dummy_input = {
        'input_ids': jnp.ones((1, seq_len), dtype=jnp.int32),
        'labels': jnp.ones((1, seq_len), dtype=jnp.int32),
        'attention_mask': jnp.ones((1, seq_len), dtype=jnp.bool_)
    }
    
    print("Initializing model parameters...")
    params = model.init(rng, **dummy_input)
    print(f"Parameter structure: {jax.tree.map(lambda x: x.shape, params)}")
    
    # Create optimizer
    optimizer = optax.adamw(learning_rate=learning_rate)
    
    # Create training state
    state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optimizer
    )
    
    print("Training state created successfully")
    
    # Test forward pass with sharded data
    print("\n=== Testing Forward Pass ===")
    
    def forward_fn(params, batch):
        """Forward pass function."""
        return model.apply(params, **batch)
    
    # JIT compile the forward function
    forward_fn_jit = jax.jit(forward_fn)
    
    try:
        print("Running forward pass...")
        output = forward_fn_jit(state.params, sharded_batch)
        print(f"Forward pass successful!")
        print(f"Output shapes: {jax.tree.map(lambda x: x.shape, output)}")
        
        # Test loss computation
        print("\n=== Testing Loss Computation ===")
        
        def loss_fn(params, batch):
            """Compute loss."""
            output = model.apply(params, **batch)
            return compute_total_loss(output, batch['labels'])
        
        loss_fn_jit = jax.jit(loss_fn)
        
        loss_result = loss_fn_jit(state.params, sharded_batch)
        print(f"Loss computation successful!")
        print(f"Loss: {loss_result.loss}")
        print(f"Metrics: {loss_result.metrics}")
        
        # Test gradient computation
        print("\n=== Testing Gradient Computation ===")
        
        def train_step(state, batch):
            """Single training step."""
            def loss_fn(params):
                output = model.apply(params, **batch)
                return compute_total_loss(output, batch['labels']).loss
            
            loss, grads = jax.value_and_grad(loss_fn)(state.params)
            state = state.apply_gradients(grads=grads)
            return state, loss
        
        train_step_jit = jax.jit(train_step)
        
        new_state, loss = train_step_jit(state, sharded_batch)
        print(f"Gradient computation successful!")
        print(f"Training step loss: {loss}")
        
        # Memory usage check
        print("\n=== Memory Usage Check ===")
        for i, device in enumerate(devices):
            memory_info = device.memory_stats()
            used_gb = memory_info['bytes_in_use'] / (1024**3)
            limit_gb = memory_info['bytes_limit'] / (1024**3)
            print(f"Device {i}: {used_gb:.2f} GB / {limit_gb:.2f} GB used")
        
        print("\n✅ All tests passed! Distributed training is working correctly.")
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_distributed_training()
    if success:
        print("\n🎉 Distributed TPU training test completed successfully!")
    else:
        print("\n💥 Distributed TPU training test failed!")
        sys.exit(1)