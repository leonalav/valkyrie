"""Gryphon Model Usage Examples

Comprehensive examples demonstrating how to use the Gryphon hybrid model
for various tasks including training, inference, and generation.

This file serves as both documentation and a practical guide for using
the BigBird-S5 hybrid architecture effectively.
"""

import jax
import jax.numpy as jnp
import optax
from typing import Dict, Any, List, Tuple
import time

from .gryphon_config import GryphonConfig, get_gryphon_base_config
from .gryphon_model import GryphonModel, create_gryphon_base
from .training_utils import (
    create_gryphon_optimizer,
    compute_gryphon_loss,
    monitor_s5_stability,
    check_gradient_health,
    validate_training_config,
    GryphonTrainingState
)


def example_model_creation():
    """Example: Creating and inspecting a Gryphon model."""
    print("=== Gryphon Model Creation Example ===")
    
    # Create model with default configuration
    model = create_gryphon_base(vocab_size=50257)
    
    # Get model information
    model_info = model.get_model_info()
    
    print(f"Model Type: {model_info['model_type']}")
    print(f"Total Parameters: {model_info['total_parameters']}")
    print(f"Architecture: {model_info['architecture']}")
    print(f"Sparse Attention: {model_info['sparse_attention']}")
    print(f"Memory Estimates: {model_info['memory_estimates']}")
    print(f"Optimizations: {model_info['optimizations']}")
    
    return model


def example_forward_pass():
    """Example: Forward pass through Gryphon model."""
    print("\n=== Forward Pass Example ===")
    
    # Create model
    model = create_gryphon_base(vocab_size=1000)  # Small vocab for example
    
    # Initialize model parameters
    rng = jax.random.PRNGKey(42)
    batch_size, seq_len = 2, 512
    
    # Create dummy input
    input_ids = jax.random.randint(rng, (batch_size, seq_len), 0, 1000)
    attention_mask = jnp.ones((batch_size, seq_len))
    
    # Initialize parameters
    params = model.init(rng, input_ids, attention_mask, training=False)
    
    print(f"Input shape: {input_ids.shape}")
    print(f"Attention mask shape: {attention_mask.shape}")
    
    # Forward pass
    start_time = time.time()
    outputs = model.apply(params, input_ids, attention_mask, training=False)
    forward_time = time.time() - start_time
    
    print(f"Output logits shape: {outputs['logits'].shape}")
    print(f"Forward pass time: {forward_time:.3f}s")
    
    # Check S5 states
    if outputs['s5_states']:
        print(f"Number of layers with S5 states: {len(outputs['s5_states'])}")
        print(f"S5 state shape (first layer, first block): {outputs['s5_states'][0][0].shape}")
    
    return model, params


def example_training_setup():
    """Example: Setting up training for Gryphon model."""
    print("\n=== Training Setup Example ===")
    
    # Create configuration
    config = get_gryphon_base_config()
    config.vocab_size = 1000
    config.max_sequence_length = 1024
    config.use_gradient_checkpointing = True
    config.use_mixed_precision = True
    
    # Validate configuration
    validate_training_config(config)
    
    # Create model
    model = GryphonModel(config=config)
    
    # Initialize parameters
    rng = jax.random.PRNGKey(42)
    batch_size, seq_len = 4, 512
    input_ids = jax.random.randint(rng, (batch_size, seq_len), 0, config.vocab_size)
    
    params = model.init(rng, input_ids, training=True)
    
    # Create optimizer with parameter-specific learning rates
    optimizer = create_gryphon_optimizer(config, base_learning_rate=1e-3)
    opt_state = optimizer.init(params)
    
    # Create training state
    training_state = GryphonTrainingState(
        params=params,
        opt_state=opt_state,
        step=0,
        s5_states=None,
        metrics={}
    )
    
    print(f"Model parameters initialized")
    print(f"Optimizer created with parameter-specific learning rates:")
    print(f"  - S5 LR multiplier: {config.s5_learning_rate_multiplier}")
    print(f"  - Attention LR multiplier: {config.attention_learning_rate_multiplier}")
    print(f"Training state created")
    
    return model, training_state, optimizer


def example_training_step(model, training_state, optimizer):
    """Example: Single training step."""
    print("\n=== Training Step Example ===")
    
    # Create batch data
    rng = jax.random.PRNGKey(123)
    batch_size, seq_len = 4, 512
    input_ids = jax.random.randint(rng, (batch_size, seq_len), 0, 1000)
    attention_mask = jnp.ones((batch_size, seq_len))
    
    def loss_fn(params):
        """Compute loss for training step."""
        outputs = model.apply(params, input_ids, attention_mask, training=True)
        logits = outputs['logits']
        
        # Use input_ids as targets (shifted internally in compute_gryphon_loss)
        loss, metrics = compute_gryphon_loss(
            logits=logits,
            targets=input_ids,
            attention_mask=attention_mask,
            label_smoothing=0.1
        )
        
        return loss, (outputs, metrics)
    
    # Compute gradients
    start_time = time.time()
    (loss, (outputs, metrics)), grads = jax.value_and_grad(loss_fn, has_aux=True)(training_state.params)
    grad_time = time.time() - start_time
    
    # Check gradient health
    grad_health = check_gradient_health(grads)
    
    # Update parameters
    updates, new_opt_state = optimizer.update(grads, training_state.opt_state, training_state.params)
    new_params = optax.apply_updates(training_state.params, updates)
    
    # Monitor S5 stability
    s5_stability = monitor_s5_stability(new_params)
    
    # Update training state
    new_training_state = GryphonTrainingState(
        params=new_params,
        opt_state=new_opt_state,
        step=training_state.step + 1,
        s5_states=outputs['s5_states'],
        metrics={**metrics, **grad_health, **s5_stability}
    )
    
    print(f"Training step completed in {grad_time:.3f}s")
    print(f"Loss: {loss:.4f}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Perplexity: {metrics['perplexity']:.2f}")
    print(f"Global gradient norm: {grad_health['global_grad_norm']:.4f}")
    
    # Check for potential issues
    if grad_health['global_grad_norm'] > 10.0:
        print("WARNING: High gradient norm detected!")
    
    if any('nan_count' in k for k in new_training_state.metrics.keys()):
        print("WARNING: NaN values detected in parameters or gradients!")
    
    return new_training_state


def example_generation():
    """Example: Text generation with Gryphon model."""
    print("\n=== Generation Example ===")
    
    # Create model
    model = create_gryphon_base(vocab_size=1000)
    
    # Initialize parameters
    rng = jax.random.PRNGKey(42)
    batch_size = 1
    init_seq_len = 10
    
    # Create initial sequence
    input_ids = jax.random.randint(rng, (batch_size, init_seq_len), 0, 1000)
    
    params = model.init(rng, input_ids, training=False)
    
    # Initialize S5 states for generation
    s5_states = model.init_s5_states(batch_size)
    
    print(f"Initial sequence: {input_ids[0].tolist()}")
    print(f"Generating {20} additional tokens...")
    
    # Generate tokens
    generated_tokens = []
    current_input = input_ids
    current_s5_states = s5_states
    
    for step in range(20):
        # Get next token
        next_token, updated_s5_states = model.apply(
            params,
            current_input[:, -1:],  # Only use last token
            s5_states=current_s5_states,
            training=False,
            method=model.generate_step,
            temperature=0.8,
            top_k=50,
            top_p=0.9
        )
        
        generated_tokens.append(next_token[0, 0].item())
        
        # Update for next iteration
        current_input = jnp.concatenate([current_input, next_token], axis=1)
        current_s5_states = updated_s5_states
        
        if step % 5 == 0:
            print(f"Step {step}: Generated token {next_token[0, 0].item()}")
    
    print(f"Generated tokens: {generated_tokens}")
    print(f"Full sequence: {current_input[0].tolist()}")
    
    return current_input, current_s5_states


def example_memory_analysis():
    """Example: Memory usage analysis for different configurations."""
    print("\n=== Memory Analysis Example ===")
    
    configs = [
        ("Small", get_gryphon_base_config()),
        ("Large Sequence", GryphonConfig(
            d_model=1024, n_layers=12, max_sequence_length=8192,
            s5_state_dim=1024, block_size=64
        )),
        ("Large Model", GryphonConfig(
            d_model=2048, n_layers=24, max_sequence_length=4096,
            s5_state_dim=2048, block_size=64
        ))
    ]
    
    for name, config in configs:
        config.vocab_size = 50257
        memory_est = config.get_memory_estimates(batch_size=8)
        attention_info = config.get_attention_pattern_info()
        
        print(f"\n{name} Configuration:")
        print(f"  Model size: {config.d_model}d × {config.n_layers}L")
        print(f"  Sequence length: {config.max_sequence_length}")
        print(f"  S5 state dim: {config.s5_state_dim}")
        print(f"  Total parameters: {memory_est['total_params_millions']:.1f}M")
        print(f"  Estimated memory (batch=8): {memory_est['estimated_total_memory_gb']:.2f}GB")
        print(f"  Attention sparsity: {attention_info['sparsity_ratio']:.1%}")


def example_performance_comparison():
    """Example: Performance comparison between different attention patterns."""
    print("\n=== Performance Comparison Example ===")
    
    # Test different block sizes and sparsity patterns
    base_config = get_gryphon_base_config()
    base_config.vocab_size = 1000
    
    test_configs = [
        ("Dense Attention", GryphonConfig(
            **base_config.__dict__,
            num_global_blocks=base_config.num_blocks,  # All blocks are global (dense)
            window_size=0,
            num_random_blocks=0
        )),
        ("Standard Sparse", base_config),
        ("High Sparsity", GryphonConfig(
            **base_config.__dict__,
            num_global_blocks=1,
            window_size=2,
            num_random_blocks=1
        ))
    ]
    
    batch_size, seq_len = 2, 1024
    rng = jax.random.PRNGKey(42)
    input_ids = jax.random.randint(rng, (batch_size, seq_len), 0, 1000)
    
    for name, config in test_configs:
        try:
            model = GryphonModel(config=config)
            params = model.init(rng, input_ids, training=False)
            
            # Time forward pass
            start_time = time.time()
            outputs = model.apply(params, input_ids, training=False)
            forward_time = time.time() - start_time
            
            attention_info = config.get_attention_pattern_info()
            memory_est = config.get_memory_estimates(batch_size=batch_size)
            
            print(f"\n{name}:")
            print(f"  Forward time: {forward_time:.3f}s")
            print(f"  Sparsity ratio: {attention_info['sparsity_ratio']:.1%}")
            print(f"  Memory estimate: {memory_est['estimated_total_memory_gb']:.2f}GB")
            print(f"  Output shape: {outputs['logits'].shape}")
            
        except Exception as e:
            print(f"\n{name}: Failed with error: {e}")


def main():
    """Run all examples."""
    print("Gryphon Model Examples")
    print("=" * 50)
    
    # Basic model creation and inspection
    model = example_model_creation()
    
    # Forward pass example
    model, params = example_forward_pass()
    
    # Training setup
    model, training_state, optimizer = example_training_setup()
    
    # Training step
    new_training_state = example_training_step(model, training_state, optimizer)
    
    # Generation example
    generated_sequence, final_s5_states = example_generation()
    
    # Memory analysis
    example_memory_analysis()
    
    # Performance comparison
    example_performance_comparison()
    
    print("\n" + "=" * 50)
    print("All examples completed successfully!")
    print("\nKey Takeaways:")
    print("1. Gryphon combines S5's sequential processing with BigBird's global attention")
    print("2. Parameter-specific learning rates are crucial for stable training")
    print("3. Sparse attention provides significant memory and compute savings")
    print("4. S5 states enable efficient recurrent generation")
    print("5. Gradient checkpointing and mixed precision are essential for large models")


if __name__ == "__main__":
    main()