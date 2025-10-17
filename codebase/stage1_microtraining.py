#!/usr/bin/env python3
"""
Stage 1 — "Microtraining" sanity run for S5 model

Objective: verify correctness and forward/backward stability.

✅ Setup:
- Model: single S5 block, optionally inside a toy 1-layer MLP
- Dataset: synthetic sine/cosine regression or AR(1) process (something continuous)
- Input: batch=8, seq_len=512, d_model=128
- Optimizer: AdamW(1e-3), no weight decay
- Precision: float32 weights, bfloat16 activations (standard TPU mix)

✅ Success criteria:
- Forward pass runs entirely on TPU (no host fallbacks)
- Loss steadily decreases over 1k steps
- Gradient norms stay finite (≈ 0.01–10)
- No NaNs or infs after 1k steps
- Throughput ≥ 75% of equivalent RNN baseline
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
import numpy as np
import time
from typing import Tuple, Dict, Any
from dataclasses import dataclass
import warnings

# Import our S5 model and config
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from model.s5 import ValkyrieS5
from model.modules import ValkyrieConfig


@dataclass
class MicrotrainingConfig:
    """Configuration for microtraining experiment."""
    # Data parameters
    batch_size: int = 8
    seq_len: int = 512
    d_model: int = 128
    
    # Model parameters
    s5_state_dim: int = 64  # Smaller for microtraining
    
    # Training parameters
    learning_rate: float = 1e-3
    num_steps: int = 1000
    weight_decay: float = 0.0  # No weight decay as specified
    
    # Monitoring parameters
    log_every: int = 50
    gradient_norm_min: float = 0.01
    gradient_norm_max: float = 10.0


class SyntheticDataset:
    """Synthetic sine/cosine regression dataset for continuous sequence modeling."""
    
    def __init__(self, config: MicrotrainingConfig, seed: int = 42):
        self.config = config
        self.rng = np.random.RandomState(seed)
        
    def generate_batch(self) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Generate a batch of synthetic sine/cosine sequences.
        
        Returns:
            inputs: [batch_size, seq_len, d_model] - input sequences
            targets: [batch_size, seq_len, d_model] - target sequences (shifted by 1)
        """
        batch_size = self.config.batch_size
        seq_len = self.config.seq_len
        d_model = self.config.d_model
        
        # Generate random frequencies and phases for each sequence
        freqs = self.rng.uniform(0.01, 0.1, (batch_size, d_model // 2))
        phases = self.rng.uniform(0, 2 * np.pi, (batch_size, d_model // 2))
        
        # Time steps
        t = np.arange(seq_len + 1)[None, :, None]  # [1, seq_len+1, 1]
        
        # Generate sine and cosine components
        sine_components = np.sin(2 * np.pi * freqs[:, None, :] * t + phases[:, None, :])
        cosine_components = np.cos(2 * np.pi * freqs[:, None, :] * t + phases[:, None, :])
        
        # Interleave sine and cosine to get full d_model
        sequences = np.zeros((batch_size, seq_len + 1, d_model))
        sequences[:, :, 0::2] = sine_components  # Even indices: sine
        sequences[:, :, 1::2] = cosine_components  # Odd indices: cosine
        
        # Add small amount of noise for realism
        noise = self.rng.normal(0, 0.01, sequences.shape)
        sequences += noise
        
        # Split into inputs and targets (autoregressive prediction)
        inputs = sequences[:, :-1, :]  # [batch_size, seq_len, d_model]
        targets = sequences[:, 1:, :]  # [batch_size, seq_len, d_model]
        
        return jnp.array(inputs, dtype=jnp.float32), jnp.array(targets, dtype=jnp.float32)


class SimpleS5Regressor(nn.Module):
    """Simple MLP wrapper around single S5 block for regression."""
    
    config: ValkyrieConfig
    
    def setup(self):
        # Single S5 layer
        self.s5_layer = ValkyrieS5(
            config=self.config,
            state_dim=self.config.s5_state_dim,
            init_mode="hippo"
        )
        
        # Simple output projection (no bias as per config)
        self.output_proj = nn.Dense(
            features=self.config.d_model,
            use_bias=self.config.use_bias,
            kernel_init=nn.initializers.normal(stddev=self.config.initializer_range)
        )
        
        # Layer norm for stability
        self.layer_norm = nn.LayerNorm(epsilon=self.config.layer_norm_eps)
    
    def __call__(self, x: jnp.ndarray, training: bool = True) -> jnp.ndarray:
        """
        Forward pass through S5 regressor.
        
        Args:
            x: Input sequences [batch_size, seq_len, d_model]
            training: Whether in training mode
            
        Returns:
            predictions: [batch_size, seq_len, d_model]
        """
        # Apply mixed precision: convert inputs to bfloat16 for activations
        if training:
            x = x.astype(jnp.bfloat16)
        
        # S5 layer (weights remain float32, activations in bfloat16)
        s5_output, _ = self.s5_layer(x, training=training)
        
        # Layer norm
        normalized = self.layer_norm(s5_output)
        
        # Output projection back to float32 for loss computation
        predictions = self.output_proj(normalized)
        
        return predictions.astype(jnp.float32)


class RNNBaseline(nn.Module):
    """Simple RNN baseline for throughput comparison."""
    
    config: ValkyrieConfig
    
    def setup(self):
        self.rnn_cell = nn.GRUCell(features=self.config.d_model)
        self.output_proj = nn.Dense(
            features=self.config.d_model,
            use_bias=self.config.use_bias
        )
    
    def __call__(self, x: jnp.ndarray, training: bool = True) -> jnp.ndarray:
        """Forward pass through RNN baseline."""
        batch_size, seq_len, d_model = x.shape
        
        # Initialize hidden state - use correct Flax API
        carry = self.rnn_cell.initialize_carry(
            jax.random.PRNGKey(0), (batch_size, d_model)
        )
        
        # Sequential processing (no parallelization)
        outputs = []
        for t in range(seq_len):
            carry, output = self.rnn_cell(carry, x[:, t, :])
            outputs.append(output)
        
        # Stack outputs
        rnn_output = jnp.stack(outputs, axis=1)
        
        # Output projection
        predictions = self.output_proj(rnn_output)
        
        return predictions


def create_train_state(model: nn.Module, config: MicrotrainingConfig, input_shape: Tuple[int, ...], rng: jax.random.PRNGKey):
    """Create training state with optimizer."""
    # Initialize model parameters
    dummy_input = jnp.ones(input_shape, dtype=jnp.float32)
    params = model.init(rng, dummy_input, training=True)
    
    # Create optimizer (AdamW without weight decay as specified)
    optimizer = optax.adamw(
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    # Create training state
    opt_state = optimizer.init(params)
    
    return params, opt_state, optimizer


def compute_loss(params: Dict[str, Any], model: nn.Module, inputs: jnp.ndarray, targets: jnp.ndarray) -> jnp.ndarray:
    """Compute MSE loss for regression."""
    predictions = model.apply(params, inputs, training=True)
    loss = jnp.mean((predictions - targets) ** 2)
    return loss


def train_step(params: Dict[str, Any], opt_state: Any, optimizer: optax.GradientTransformation, 
               model: nn.Module, inputs: jnp.ndarray, targets: jnp.ndarray) -> Tuple[Dict[str, Any], Any, float, float]:
    """Single training step with gradient computation and monitoring."""
    
    # Compute loss and gradients
    loss, grads = jax.value_and_grad(compute_loss)(params, model, inputs, targets)
    
    # Compute gradient norm for monitoring
    grad_norm = jnp.sqrt(sum(jnp.sum(jnp.square(g)) for g in jax.tree_util.tree_leaves(grads)))
    
    # Update parameters
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    
    return params, opt_state, loss, grad_norm


# Remove the unused JIT function that was causing issues
# @jax.jit
# def jit_train_step_s5(params: Dict[str, Any], opt_state: Any, inputs: jnp.ndarray, targets: jnp.ndarray):
#     """JIT-compiled training step specifically for S5 model."""
#     
#     def loss_fn(params):
#         predictions = s5_model.apply(params, inputs, training=True)
#         return jnp.mean((predictions - targets) ** 2)
#     
#     loss, grads = jax.value_and_grad(loss_fn)(params)
#     grad_norm = jnp.sqrt(sum(jnp.sum(jnp.square(g)) for g in jax.tree_util.tree_leaves(grads)))
#     
#     return loss, grads, grad_norm


def check_for_nans_infs(params: Dict[str, Any], loss: float, grad_norm: float) -> Dict[str, bool]:
    """Check for NaNs and infs in parameters, loss, and gradients."""
    param_has_nan = any(jnp.any(jnp.isnan(p)) for p in jax.tree_util.tree_leaves(params))
    param_has_inf = any(jnp.any(jnp.isinf(p)) for p in jax.tree_util.tree_leaves(params))
    
    return {
        'param_nan': param_has_nan,
        'param_inf': param_has_inf,
        'loss_nan': jnp.isnan(loss),
        'loss_inf': jnp.isinf(loss),
        'grad_nan': jnp.isnan(grad_norm),
        'grad_inf': jnp.isinf(grad_norm)
    }


def measure_throughput(model: nn.Module, params: Dict[str, Any], inputs: jnp.ndarray, 
                      num_runs: int = 10) -> float:
    """Measure model throughput (sequences per second)."""
    
    # Compile the function
    @jax.jit
    def forward_pass(params, inputs):
        return model.apply(params, inputs, training=False)
    
    # Warmup
    for _ in range(3):
        _ = forward_pass(params, inputs)
    
    # Measure
    start_time = time.time()
    for _ in range(num_runs):
        _ = forward_pass(params, inputs)
    end_time = time.time()
    
    total_sequences = num_runs * inputs.shape[0]
    throughput = total_sequences / (end_time - start_time)
    
    return throughput


def run_microtraining():
    """Run the complete microtraining experiment."""
    print("🚀 Starting Stage 1 — S5 Microtraining Sanity Run")
    print("=" * 60)
    
    # Configuration
    config = MicrotrainingConfig()
    
    # Create model config
    model_config = ValkyrieConfig(
        d_model=config.d_model,
        s5_state_dim=config.s5_state_dim,
        use_bias=False,
        layer_norm_eps=1e-5,
        initializer_range=0.02
    )
    
    # Initialize models
    s5_model = SimpleS5Regressor(config=model_config)
    rnn_baseline = RNNBaseline(config=model_config)
    
    # Create dataset
    dataset = SyntheticDataset(config)
    
    # Initialize training
    rng = jax.random.PRNGKey(42)
    input_shape = (config.batch_size, config.seq_len, config.d_model)
    
    # Create training states
    s5_params, s5_opt_state, s5_optimizer = create_train_state(s5_model, config, input_shape, rng)
    rnn_params, _, _ = create_train_state(rnn_baseline, config, input_shape, rng)
    
    # Define JIT-compiled loss and gradient function for S5
    @jax.jit
    def compute_loss_and_grads(params, inputs, targets):
        def loss_fn(params):
            predictions = s5_model.apply(params, inputs, training=True)
            return jnp.mean((predictions - targets) ** 2)
        
        loss, grads = jax.value_and_grad(loss_fn)(params)
        grad_norm = jnp.sqrt(sum(jnp.sum(jnp.square(g)) for g in jax.tree_util.tree_leaves(grads)))
        return loss, grads, grad_norm
    
    print(f"📊 Configuration:")
    print(f"   Batch size: {config.batch_size}")
    print(f"   Sequence length: {config.seq_len}")
    print(f"   Model dimension: {config.d_model}")
    print(f"   S5 state dimension: {config.s5_state_dim}")
    print(f"   Learning rate: {config.learning_rate}")
    print(f"   Training steps: {config.num_steps}")
    print()
    
    # Check device
    devices = jax.devices()
    print(f"🖥️  Available devices: {[d.device_kind for d in devices]}")
    print(f"   Primary device: {jax.devices()[0].device_kind}")
    print()
    
    # Training loop
    print("🏋️  Starting training...")
    losses = []
    grad_norms = []
    
    for step in range(config.num_steps):
        # Generate batch
        inputs, targets = dataset.generate_batch()
        
        # Compute loss and gradients
        loss, grads, grad_norm = compute_loss_and_grads(s5_params, inputs, targets)
        
        # Update parameters
        updates, s5_opt_state = s5_optimizer.update(grads, s5_opt_state, s5_params)
        s5_params = optax.apply_updates(s5_params, updates)
        
        losses.append(float(loss))
        grad_norms.append(float(grad_norm))
        
        # Check for numerical issues
        nan_inf_status = check_for_nans_infs(s5_params, loss, grad_norm)
        
        if any(nan_inf_status.values()):
            print(f"❌ Numerical instability detected at step {step}:")
            for key, value in nan_inf_status.items():
                if value:
                    print(f"   {key}: {value}")
            break
        
        # Logging
        if step % config.log_every == 0 or step == config.num_steps - 1:
            print(f"Step {step:4d}: Loss = {loss:.6f}, Grad Norm = {grad_norm:.6f}")
    
    print()
    
    # Final evaluation
    print("📈 Final Results:")
    print("=" * 40)
    
    # Loss trend analysis
    initial_loss = np.mean(losses[:10]) if len(losses) >= 10 else losses[0]
    final_loss = np.mean(losses[-10:]) if len(losses) >= 10 else losses[-1]
    loss_reduction = (initial_loss - final_loss) / initial_loss * 100
    
    print(f"✅ Loss Analysis:")
    print(f"   Initial loss (avg first 10): {initial_loss:.6f}")
    print(f"   Final loss (avg last 10): {final_loss:.6f}")
    print(f"   Loss reduction: {loss_reduction:.2f}%")
    print(f"   Loss decreasing: {'✅ YES' if final_loss < initial_loss else '❌ NO'}")
    print()
    
    # Gradient norm analysis
    grad_norm_stats = {
        'min': np.min(grad_norms),
        'max': np.max(grad_norms),
        'mean': np.mean(grad_norms),
        'std': np.std(grad_norms)
    }
    
    grad_norms_finite = all(np.isfinite(grad_norms))
    grad_norms_in_range = (grad_norm_stats['min'] >= config.gradient_norm_min and 
                          grad_norm_stats['max'] <= config.gradient_norm_max)
    
    print(f"✅ Gradient Analysis:")
    print(f"   Gradient norms finite: {'✅ YES' if grad_norms_finite else '❌ NO'}")
    print(f"   Gradient norm range: [{grad_norm_stats['min']:.4f}, {grad_norm_stats['max']:.4f}]")
    print(f"   Target range: [{config.gradient_norm_min}, {config.gradient_norm_max}]")
    print(f"   In target range: {'✅ YES' if grad_norms_in_range else '❌ NO'}")
    print(f"   Mean ± std: {grad_norm_stats['mean']:.4f} ± {grad_norm_stats['std']:.4f}")
    print()
    
    # Throughput comparison
    print("🚀 Throughput Analysis:")
    test_inputs, _ = dataset.generate_batch()
    
    s5_throughput = measure_throughput(s5_model, s5_params, test_inputs)
    rnn_throughput = measure_throughput(rnn_baseline, rnn_params, test_inputs)
    
    throughput_ratio = s5_throughput / rnn_throughput
    throughput_target_met = throughput_ratio >= 0.75
    
    print(f"   S5 throughput: {s5_throughput:.2f} sequences/sec")
    print(f"   RNN throughput: {rnn_throughput:.2f} sequences/sec")
    print(f"   S5/RNN ratio: {throughput_ratio:.2f}x")
    print(f"   Target (≥75%): {'✅ YES' if throughput_target_met else '❌ NO'}")
    print()
    
    # Final numerical stability check
    final_nan_inf = check_for_nans_infs(s5_params, losses[-1], grad_norms[-1])
    no_final_nans_infs = not any(final_nan_inf.values())
    
    print("🔍 Final Stability Check:")
    print(f"   No NaNs/Infs in parameters: {'✅ YES' if not (final_nan_inf['param_nan'] or final_nan_inf['param_inf']) else '❌ NO'}")
    print(f"   No NaNs/Infs in loss: {'✅ YES' if not (final_nan_inf['loss_nan'] or final_nan_inf['loss_inf']) else '❌ NO'}")
    print(f"   No NaNs/Infs in gradients: {'✅ YES' if not (final_nan_inf['grad_nan'] or final_nan_inf['grad_inf']) else '❌ NO'}")
    print()
    
    # Success criteria summary
    print("🎯 Success Criteria Summary:")
    print("=" * 40)
    
    criteria = {
        "Forward pass runs on TPU": jax.devices()[0].device_kind.startswith('TPU'),  # Check for TPU (including version)
        "Loss steadily decreases": final_loss < initial_loss,
        "Gradient norms finite": grad_norms_finite,
        "Gradient norms in range": grad_norms_in_range,
        "No NaNs/Infs after 1k steps": no_final_nans_infs,
        "Throughput ≥ 75% of RNN": throughput_target_met
    }
    
    all_passed = all(criteria.values())
    
    for criterion, passed in criteria.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"   {criterion}: {status}")
    
    print()
    print(f"🏆 Overall Result: {'✅ ALL CRITERIA PASSED' if all_passed else '❌ SOME CRITERIA FAILED'}")
    
    if all_passed:
        print("🎉 Stage 1 microtraining completed successfully!")
        print("   The discretization + init pipeline is numerically stable.")
    else:
        print("⚠️  Stage 1 microtraining had issues. Review failed criteria above.")
    
    return {
        'success': all_passed,
        'criteria': criteria,
        'losses': losses,
        'grad_norms': grad_norms,
        'throughput_ratio': throughput_ratio,
        'loss_reduction': loss_reduction
    }


if __name__ == "__main__":
    # Suppress JAX warnings for cleaner output
    warnings.filterwarnings("ignore", category=UserWarning, module="jax")
    
    # Run the microtraining experiment
    results = run_microtraining()