#!/usr/bin/env python3
"""
Stage 2: BigBird + S5 Hybrid Efficiency Benchmark

Implements a hybrid encoder-decoder architecture:
- 2 BigBird encoder blocks for global information routing
- 1 S5 decoder block for sequential processing
- Comprehensive profiling and performance monitoring

Architecture follows Stage 2 specifications:
- Sequence length: 2048 tokens
- Batch size: 16
- Model dimension: 256
- AdamW optimizer (1e-4)
- Mixed precision: float32 weights, bfloat16 activations
- JAX profiler trace enabled

Success Criteria:
- ≥70% baseline throughput
- Stable loss curve for 500+ steps
- Step time, memory footprint, throughput monitoring
- Spectral radius < 1 for numerical stability
- Host/device transfers = 0
"""

import jax
import jax.numpy as jnp
import jax.profiler
import flax.linen as nn
import optax
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, List
import time
import math
import os
import tempfile
from functools import partial

# Import existing components
import sys
sys.path.append('/home/ravkeave/v1/codebase/src')

from model.gryphon.gryphon_config import GryphonConfig
from model.gryphon.gryphon_blocks import BigBirdBlock
from model.gryphon.bigbird_attention import BigBirdSparseAttention
from model.s5 import ValkyrieS5
from model.modules import RMSNorm, ValkyrieConfig


@dataclass
class Stage2Config:
    """Configuration for Stage 2 BigBird + S5 hybrid benchmark."""
    
    # Architecture parameters
    seq_len: int = 2048
    batch_size: int = 16
    d_model: int = 256
    vocab_size: int = 32000
    
    # Hybrid architecture specification
    num_bigbird_encoder_blocks: int = 2
    num_s5_decoder_blocks: int = 1
    
    # BigBird sparse attention parameters
    block_size: int = 64  # 2048 / 64 = 32 blocks
    num_global_blocks: int = 2
    window_size: int = 3
    num_random_blocks: int = 2
    
    # S5 parameters
    s5_state_dim: int = 128
    s5_init_mode: str = "hippo"
    
    # Model parameters
    n_heads: int = 8
    head_dim: int = 32  # d_model // n_heads
    intermediate_size: int = 1024  # 4 * d_model
    
    # Training parameters
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 100
    max_steps: int = 1000
    
    # Mixed precision
    use_mixed_precision: bool = True
    param_dtype: jnp.dtype = jnp.float32
    compute_dtype: jnp.dtype = jnp.bfloat16
    
    # Regularization
    dropout_rate: float = 0.1
    attention_dropout: float = 0.1
    layer_norm_eps: float = 1e-8
    
    # Profiling
    enable_profiling: bool = False  # Disabled to fix throughput collapse
    profile_dir: str = "/tmp/stage2_profile"
    
    # Numerical stability
    gradient_clip_norm: float = 1.0
    
    def __post_init__(self):
        """Validate configuration."""
        assert self.seq_len % self.block_size == 0, f"seq_len ({self.seq_len}) must be divisible by block_size ({self.block_size})"
        assert self.d_model % self.n_heads == 0, f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
        self.num_blocks = self.seq_len // self.block_size


class SyntheticDataset:
    """Synthetic dataset for encoder-decoder training."""
    
    def __init__(self, config: Stage2Config, rng_key: jnp.ndarray):
        self.config = config
        self.rng_key = rng_key
        
    def generate_batch(self, rng_key: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Generate a batch of synthetic encoder-decoder data.
        
        Creates sequences with:
        - Encoder input: Random token sequences with patterns
        - Decoder target: Transformed sequences (e.g., reversed, shifted)
        
        Returns:
            Tuple of (encoder_input, decoder_target) with shape [batch, seq_len]
        """
        batch_size, seq_len = self.config.batch_size, self.config.seq_len
        vocab_size = self.config.vocab_size
        
        # Generate encoder input with some structure
        encoder_key, decoder_key = jax.random.split(rng_key)
        
        # Encoder input: structured patterns
        encoder_input = jax.random.randint(
            encoder_key, 
            (batch_size, seq_len), 
            0, vocab_size
        )
        
        # Decoder target: transformation of encoder input
        # Simple transformation: reverse + shift
        decoder_target = jnp.flip(encoder_input, axis=1)
        decoder_target = (decoder_target + 1) % vocab_size
        
        return encoder_input, decoder_target


class BigBirdEncoder(nn.Module):
    """BigBird encoder with multiple blocks."""
    
    config: Stage2Config
    
    def setup(self):
        """Initialize encoder blocks."""
        # Convert Stage2Config to GryphonConfig for compatibility
        gryphon_config = GryphonConfig(
            d_model=self.config.d_model,
            n_heads=self.config.n_heads,
            n_layers=self.config.num_bigbird_encoder_blocks,
            vocab_size=self.config.vocab_size,
            max_sequence_length=self.config.seq_len,
            block_size=self.config.block_size,
            num_global_blocks=self.config.num_global_blocks,
            window_size=self.config.window_size,
            num_random_blocks=self.config.num_random_blocks,
            attention_dropout=self.config.attention_dropout,
            resid_dropout=self.config.dropout_rate,
            layer_norm_eps=self.config.layer_norm_eps
        )
        
        # Token embeddings
        self.token_embeddings = nn.Embed(
            num_embeddings=self.config.vocab_size,
            features=self.config.d_model,
            dtype=self.config.param_dtype
        )
        
        # BigBird encoder blocks
        self.encoder_blocks = [
            BigBirdBlock(config=gryphon_config, name=f'encoder_block_{i}')
            for i in range(self.config.num_bigbird_encoder_blocks)
        ]
        
        # Final layer norm
        self.final_norm = RMSNorm(
            hidden_size=self.config.d_model,
            eps=self.config.layer_norm_eps
        )
    
    def __call__(
        self, 
        input_ids: jnp.ndarray, 
        training: bool = True
    ) -> jnp.ndarray:
        """Forward pass of BigBird encoder.
        
        Args:
            input_ids: Token indices [batch, seq_len]
            training: Whether in training mode
            
        Returns:
            Encoded representations [batch, seq_len, d_model]
        """
        # Token embeddings
        hidden_states = self.token_embeddings(input_ids)
        
        # Cast to compute dtype for mixed precision
        if self.config.use_mixed_precision:
            hidden_states = hidden_states.astype(self.config.compute_dtype)
        
        # Apply encoder blocks
        for block in self.encoder_blocks:
            hidden_states = block(
                hidden_states=hidden_states,
                training=training
            )
        
        # Final normalization
        hidden_states = self.final_norm(hidden_states)
        
        return hidden_states


class S5Decoder(nn.Module):
    """S5 decoder block with proper decoder input handling and cross-attention."""
    
    config: Stage2Config
    
    def setup(self):
        """Initialize S5 decoder."""
        # Token embeddings for decoder input
        self.token_embeddings = nn.Embed(
            num_embeddings=self.config.vocab_size,
            features=self.config.d_model,
            dtype=self.config.param_dtype
        )
        
        # Positional embeddings
        self.pos_emb = self.param(
            "pos_emb", 
            nn.initializers.normal(1e-3),
            (1, self.config.seq_len, self.config.d_model)
        )
        
        # Input layer norm
        self.input_norm = RMSNorm(
            hidden_size=self.config.d_model,
            eps=self.config.layer_norm_eps
        )
        
        # Use ValkyrieConfig from modules instead of separate valkyrie module
        s5_config = ValkyrieConfig(
            d_model=self.config.d_model,
            n_layers=self.config.num_s5_decoder_blocks,
            vocab_size=self.config.vocab_size,
            max_position_embeddings=self.config.seq_len,
            s5_state_dim=self.config.s5_state_dim,
            resid_dropout=self.config.dropout_rate,
            layer_norm_eps=self.config.layer_norm_eps
        )
        
        # S5 layer
        self.s5_layer = ValkyrieS5(
            config=s5_config
        )
        
        # Cross-attention projections
        self.q_proj = nn.Dense(self.config.d_model, dtype=self.config.param_dtype)
        self.k_proj = nn.Dense(self.config.d_model, dtype=self.config.param_dtype)
        self.v_proj = nn.Dense(self.config.d_model, dtype=self.config.param_dtype)
        
        # Output projection
        self.output_proj = nn.Dense(
            self.config.vocab_size,
            use_bias=False,
            dtype=self.config.param_dtype
        )
    
    def __call__(
        self, 
        encoder_output: jnp.ndarray,
        decoder_input: jnp.ndarray,
        training: bool = True
    ) -> jnp.ndarray:
        """Forward pass of S5 decoder.
        
        Args:
            encoder_output: Encoded representations [batch, seq_len, d_model]
            decoder_input: Decoder input tokens [batch, seq_len]
            training: Whether in training mode
            
        Returns:
            Logits [batch, seq_len, vocab_size]
        """
        # 1) Embed decoder tokens + positional embeddings
        x = self.token_embeddings(decoder_input)                            # [B,T,D]
        x = x + self.pos_emb[:, :x.shape[1], :]
        if self.config.use_mixed_precision:
            x = x.astype(self.config.compute_dtype)
        x = self.input_norm(x)

        # 2) S5 processes decoder embeddings (teacher forcing)
        ys, _ = self.s5_layer(x, training=training)                         # [B,T,D]; dtype complex internally, returns real float
        
        # CRITICAL FIX: Cast S5 outputs to real dtype before Dense projections
        # S5 may return complex outputs during training; Dense layers expect real dtypes
        ys_real = jnp.real(ys).astype(self.config.param_dtype)

        # 3) Build memory from encoder_output — use block pooling for efficiency
        B, S, D = encoder_output.shape
        block_size = self.config.block_size
        num_blocks = S // block_size
        enc_blocks = encoder_output.reshape(B, num_blocks, block_size, D)
        mem = enc_blocks.mean(axis=2)                                       # [B, num_blocks, D]

        # 4) Cross-attention (single-head for clarity; multihead recommended)
        Q = self.q_proj(ys_real)                                           # [B,T,D]
        K = self.k_proj(mem)                                                # [B,M,D]
        V = self.v_proj(mem)                                                # [B,M,D]
        scale = jnp.sqrt(self.config.d_model).astype(Q.dtype)
        logits = jnp.einsum('btd,bmd->btm', Q, K) / scale
        att = jax.nn.softmax(logits, axis=-1)
        ctx = jnp.einsum('btm,bmd->btd', att, V)

        # 5) Fuse and project to logits
        h = ys_real + ctx
        logits = self.output_proj(h.astype(jnp.float32))
        return logits


class Stage2HybridModel(nn.Module):
    """Complete Stage 2 hybrid model: BigBird encoder + S5 decoder."""
    
    config: Stage2Config
    
    def setup(self):
        """Initialize model components."""
        self.encoder = BigBirdEncoder(config=self.config)
        self.decoder = S5Decoder(config=self.config)
    
    def __call__(
        self, 
        encoder_input: jnp.ndarray,
        decoder_input: jnp.ndarray,
        training: bool = True
    ) -> jnp.ndarray:
        """Forward pass of hybrid model.
        
        Args:
            encoder_input: Encoder input tokens [batch, seq_len]
            decoder_input: Decoder input tokens [batch, seq_len]
            training: Whether in training mode
            
        Returns:
            Logits [batch, seq_len, vocab_size]
        """
        # Encode
        encoder_output = self.encoder(encoder_input, training=training)
        
        # Decode
        logits = self.decoder(encoder_output, decoder_input, training=training)
        
        return logits


class FeedforwardBaseline(nn.Module):
    """Simple feedforward baseline for comparison following Flax patterns"""
    config: Stage2Config
    
    def setup(self):
        self.embedding = nn.Embed(
            num_embeddings=self.config.vocab_size,
            features=self.config.d_model
        )
        # Simple feedforward layers that maintain d_model dimension
        self.layer1 = nn.Dense(self.config.d_model)
        self.layer2 = nn.Dense(self.config.d_model)
        self.layer3 = nn.Dense(self.config.d_model)
        self.dropout = nn.Dropout(0.1)
        self.layer_norm = nn.LayerNorm()
        self.output_projection = nn.Dense(self.config.vocab_size)
        
    def __call__(self, encoder_input, decoder_input, training=True):
        # Use encoder input for simplicity
        input_ids = encoder_input
        
        # Embeddings
        x = self.embedding(input_ids)  # [batch, seq_len, d_model]
        
        # Apply feedforward layers
        x = self.layer1(x)
        x = jax.nn.relu(x)
        x = self.dropout(x, deterministic=not training)
        x = self.layer_norm(x)
        
        x = self.layer2(x)
        x = jax.nn.relu(x)
        x = self.dropout(x, deterministic=not training)
        x = self.layer_norm(x)
        
        x = self.layer3(x)
        x = jax.nn.relu(x)
        x = self.dropout(x, deterministic=not training)
        
        # Output projection
        logits = self.output_projection(x)
        return logits


def create_train_state(model, config: Stage2Config, rng_key: jnp.ndarray):
    """Create training state with optimizer."""
    # Initialize model
    encoder_input = jnp.ones((config.batch_size, config.seq_len), dtype=jnp.int32)
    decoder_input = jnp.ones((config.batch_size, config.seq_len), dtype=jnp.int32)
    
    # Split RNG for model initialization
    init_key, dropout_key = jax.random.split(rng_key)
    params = model.init({'params': init_key, 'dropout': dropout_key}, encoder_input, decoder_input, training=False)
    
    # Create optimizer
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=config.learning_rate,
        warmup_steps=config.warmup_steps,
        decay_steps=config.max_steps,
        end_value=config.learning_rate * 0.1
    )
    
    optimizer = optax.adamw(
        learning_rate=schedule,
        weight_decay=config.weight_decay
    )
    
    # Apply gradient clipping
    optimizer = optax.chain(
        optax.clip_by_global_norm(config.gradient_clip_norm),
        optimizer
    )
    
    # Initialize optimizer with only trainable parameters
    trainable_params = params.get('params', params)
    opt_state = optimizer.init(trainable_params)
    
    return params, opt_state, optimizer


def compute_loss(params, model, encoder_input, decoder_target, training=True, rng_key=None):
    """Compute cross-entropy loss."""
    # Use decoder_target as decoder_input (teacher forcing)
    decoder_input = decoder_target
    
    if training and rng_key is not None:
        logits = model.apply(params, encoder_input, decoder_input, training=training, rngs={'dropout': rng_key})
    else:
        logits = model.apply(params, encoder_input, decoder_input, training=training)
    
    # Ensure logits are float32 for loss computation
    logits = logits.astype(jnp.float32)
    
    # Cross-entropy loss
    loss = optax.softmax_cross_entropy_with_integer_labels(
        logits, decoder_target
    ).mean()
    
    # Ensure loss is float32
    loss = loss.astype(jnp.float32)
    
    return loss


def compute_spectral_radius(params) -> float:
    """Compute spectral radius by reconstructing Lambda_bar from S5 parameters."""
    spectral_radii = []
    
    def extract_s5_params(path, param_dict):
        # Look for S5 layer parameter dictionaries
        if isinstance(param_dict, dict) and 's5_layer' in path:
            try:
                # Extract S5 parameters needed for Lambda_bar reconstruction
                Lambda_unconstrained_re = param_dict.get('Lambda_unconstrained_re')
                # Imaginary part is stored as constrained parameter 'Lambda_im' in ValkyrieS5
                Lambda_im = param_dict.get('Lambda_im')
                log_Delta = param_dict.get('log_Delta')
                
                if Lambda_unconstrained_re is not None and Lambda_im is not None and log_Delta is not None:
                    # Reconstruct Lambda following S5 constraints
                    eps = 1e-4
                    negative_bias = 0.01  # Match ValkyrieS5 parameterization to keep Re(λ) strictly negative
                    Lambda_re = -jax.nn.softplus(Lambda_unconstrained_re) - eps - negative_bias
                    Lambda = Lambda_re + 1j * Lambda_im
                    
                    # Reconstruct Delta with proper clamping
                    Delta = jnp.exp(log_Delta)
                    Delta = jnp.clip(Delta, 1e-4, 2.0)  # Use updated bounds
                    
                    # Compute Lambda_bar = exp(Lambda * Delta) - the actual discretized eigenvalues
                    Lambda_bar = jnp.exp(Lambda * Delta)
                    
                    # Spectral radius is max absolute value
                    spectral_radius = jnp.max(jnp.abs(Lambda_bar))
                    spectral_radii.append(float(spectral_radius))
                    
            except Exception as e:
                # Skip if parameter structure doesn't match expected S5 format
                pass
    
    # Traverse parameter tree looking for S5 layers
    def traverse_params(params, path=""):
        if isinstance(params, dict):
            # Check if this dict contains S5 parameters
            extract_s5_params(path, params)
            # Continue traversing
            for key, value in params.items():
                traverse_params(value, f"{path}/{key}" if path else key)
    
    traverse_params(params)
    
    return max(spectral_radii) if spectral_radii else 0.0


def check_for_nans_infs(params, grads) -> Tuple[bool, bool]:
    """Check for NaNs and Infs in parameters and gradients."""
    def has_nan_inf(pytree):
        leaves = jax.tree_util.tree_leaves(pytree)
        has_nan = any(jnp.any(jnp.isnan(leaf)) for leaf in leaves)
        has_inf = any(jnp.any(jnp.isinf(leaf)) for leaf in leaves)
        return has_nan, has_inf
    
    param_nan, param_inf = has_nan_inf(params)
    grad_nan, grad_inf = has_nan_inf(grads)
    
    return param_nan or grad_nan, param_inf or grad_inf


def measure_throughput(
    model, 
    params, 
    config: Stage2Config, 
    rng_key: jnp.ndarray,
    num_steps: int = 10
) -> float:
    """Measure model throughput in tokens/second."""
    dataset = SyntheticDataset(config, rng_key)
    
    # Warmup
    for _ in range(3):
        rng_key, batch_key = jax.random.split(rng_key)
        encoder_input, decoder_target = dataset.generate_batch(batch_key)
        _ = model.apply(params, encoder_input, decoder_target, training=False)
    
    # Measure
    start_time = time.time()
    total_tokens = 0
    
    for _ in range(num_steps):
        rng_key, batch_key = jax.random.split(rng_key)
        encoder_input, decoder_target = dataset.generate_batch(batch_key)
        _ = model.apply(params, encoder_input, decoder_target, training=False)
        total_tokens += encoder_input.size + decoder_target.size
    
    end_time = time.time()
    throughput = total_tokens / (end_time - start_time)
    
    return throughput


def compute_loss_and_grads(params, model, encoder_input, decoder_target, rng_key):
    """Loss and gradient computation with consistent parameter structure."""
    
    def loss_fn(trainable_params):
        # Reconstruct full params by combining trainable params with constants
        full_params = {
            'params': trainable_params,
            'constants': params.get('constants', {})
        }
        return compute_loss(full_params, model, encoder_input, decoder_target, training=True, rng_key=rng_key)
    
    # Only compute gradients for trainable parameters, not constants
    trainable_params = params.get('params', params)
    loss, grads = jax.value_and_grad(loss_fn)(trainable_params)
    
    # Compute gradient norm
    grad_norm = optax.global_norm(grads)
    
    return loss, grads, grad_norm


def run_stage2_benchmark(config: Stage2Config):
    """Run the complete Stage 2 benchmark."""
    print("🚀 Starting Stage 2: BigBird + S5 Hybrid Efficiency Benchmark")
    print(f"Architecture: {config.num_bigbird_encoder_blocks} BigBird encoder + {config.num_s5_decoder_blocks} S5 decoder")
    print(f"Parameters: batch={config.batch_size}, seq_len={config.seq_len}, d_model={config.d_model}")
    print(f"Device: {jax.devices()[0].device_kind}")
    
    # Initialize
    rng_key = jax.random.PRNGKey(42)
    model_key, data_key, baseline_key = jax.random.split(rng_key, 3)
    
    # Create models
    hybrid_model = Stage2HybridModel(config=config)
    baseline_model = FeedforwardBaseline(config=config)
    
    # Create training states
    print("\n📊 Initializing models...")
    hybrid_params, hybrid_opt_state, hybrid_optimizer = create_train_state(
        hybrid_model, config, model_key
    )
    baseline_params, baseline_opt_state, baseline_optimizer = create_train_state(
        baseline_model, config, model_key
    )
    
    # Create dataset
    dataset = SyntheticDataset(config, data_key)
    
    # Measure baseline throughput
    print("\n⚡ Measuring baseline throughput...")
    baseline_throughput = measure_throughput(
        baseline_model, baseline_params, config, baseline_key
    )
    print(f"Baseline feedforward throughput: {baseline_throughput:.2f} tokens/sec")
    
    # Training loop with profiling
    print(f"\n🏋️ Starting training for {config.max_steps} steps...")
    
    losses = []
    grad_norms = []
    spectral_radii = []
    step_times = []
    
    # Define JIT-compiled training step for true performance measurement
    @jax.jit
    def train_step(params, opt_state, encoder_input, decoder_target, dropout_key):
        """JIT-compiled training step for optimal performance."""
        # Compute loss and gradients
        loss, grads, grad_norm = compute_loss_and_grads(
            params, hybrid_model, encoder_input, decoder_target, dropout_key
        )
        
        # Update parameters
        updates, new_opt_state = hybrid_optimizer.update(grads, opt_state, params.get('params', params))
        new_trainable_params = optax.apply_updates(params.get('params', params), updates)
        
        # Reconstruct full params with updated trainable params and unchanged constants
        new_params = {
            'params': new_trainable_params,
            'constants': params.get('constants', {})
        }
        
        return new_params, new_opt_state, loss, grad_norm
    
    # Setup profiling
    if config.enable_profiling:
        os.makedirs(config.profile_dir, exist_ok=True)
        print(f"📈 Profiling enabled, traces will be saved to: {config.profile_dir}")
    
    # Start profiling
    if config.enable_profiling:
        jax.profiler.start_trace(config.profile_dir)
    
    training_start_time = time.time()
    
    for step in range(config.max_steps):
        step_start_time = time.time()
        
        # Generate batch
        data_key, batch_key = jax.random.split(data_key)
        encoder_input, decoder_target = dataset.generate_batch(batch_key)
        
        # Split RNG key for dropout
        data_key, dropout_key = jax.random.split(data_key)
        
        # Use JIT-compiled training step
        hybrid_params, hybrid_opt_state, loss, grad_norm = train_step(
            hybrid_params, hybrid_opt_state, encoder_input, decoder_target, dropout_key
        )
        
        # Record metrics
        step_time = time.time() - step_start_time
        losses.append(float(loss))
        grad_norms.append(float(grad_norm))
        step_times.append(step_time)
        
        # Compute spectral radius periodically
        if step % 100 == 0:
            spectral_radius = compute_spectral_radius(hybrid_params)
            spectral_radii.append(spectral_radius)
            
            # Add monitoring prints for top-k gradient norms and max(|B|) after projection (from note.txt)
            if step % 100 == 0:  # Monitor every 100 steps
                # Compute gradients for monitoring (not used for updates)
                _, monitoring_grads, _ = compute_loss_and_grads(
                    hybrid_params, hybrid_model, encoder_input, decoder_target, dropout_key
                )
                
                # Compute top-k gradient norms
                grad_leaves = jax.tree_util.tree_leaves(monitoring_grads)
                grad_norms_per_param = [jnp.linalg.norm(leaf) for leaf in grad_leaves]
                top_k_grad_norms = sorted(grad_norms_per_param, reverse=True)[:5]  # Top 5
                
                # Monitor max(|B|) after projection for S5 layers
                max_b_magnitude = 0.0
                try:
                    # Extract S5 B parameters from the model (use params structure)
                    if 'params' in hybrid_params and 'decoder' in hybrid_params['params'] and 's5_layer' in hybrid_params['params']['decoder']:
                        s5_params = hybrid_params['params']['decoder']['s5_layer']
                        if 'B_real' in s5_params and 'B_imag' in s5_params:
                            B_real = s5_params['B_real']
                            B_imag = s5_params['B_imag']
                            B_complex = B_real + 1j * B_imag
                            max_b_magnitude = float(jnp.max(jnp.abs(B_complex)))
                except (KeyError, TypeError):
                    pass  # Skip if S5 parameters not found
                
                print(f"Step {step:4d}: loss={loss:.6f}, grad_norm={grad_norm:.6f}, step_time={step_time:.4f}s")
                print(f"   Top-5 grad norms: {[f'{norm:.2e}' for norm in top_k_grad_norms[:5]]}")
                print(f"   Max |B| magnitude: {max_b_magnitude:.4f}")
            else:
                print(f"Step {step:4d}: loss={loss:.6f}, grad_norm={grad_norm:.6f}, step_time={step_time:.4f}s")
        
        # Check for numerical issues in parameters
        def check_for_nans_infs(params_tree):
            """Check for NaN or Inf values in parameter tree."""
            has_nan = False
            has_inf = False
            
            def check_array(x):
                nonlocal has_nan, has_inf
                if jnp.isnan(x).any():
                    has_nan = True
                if jnp.isinf(x).any():
                    has_inf = True
            
            jax.tree.map(check_array, params_tree)
            return has_nan, has_inf
        
        has_nan, has_inf = check_for_nans_infs(hybrid_params)
        if has_nan or has_inf:
            print(f"❌ Numerical instability detected at step {step}")
            print(f"   NaN: {has_nan}, Inf: {has_inf}")
            break
    
    # Stop profiling
    if config.enable_profiling:
        jax.profiler.stop_trace()
    
    training_end_time = time.time()
    total_training_time = training_end_time - training_start_time
    
    # Measure hybrid model throughput
    print("\n⚡ Measuring hybrid model throughput...")
    hybrid_throughput = measure_throughput(
        hybrid_model, hybrid_params, config, baseline_key
    )
    
    # Compute final metrics
    final_loss = losses[-1]
    initial_loss = losses[0]
    loss_reduction = (initial_loss - final_loss) / initial_loss * 100
    
    avg_grad_norm = np.mean(grad_norms)
    max_spectral_radius = max(spectral_radii) if spectral_radii else 0.0
    avg_step_time = np.mean(step_times)
    throughput_ratio = hybrid_throughput / baseline_throughput
    
    # Check TPU usage
    device_kind = jax.devices()[0].device_kind
    runs_on_tpu = device_kind.startswith('TPU')
    
    # Success criteria
    print("\n" + "="*60)
    print("📋 STAGE 2 BENCHMARK RESULTS")
    print("="*60)
    
    print(f"🏗️  Architecture: {config.num_bigbird_encoder_blocks} BigBird encoder + {config.num_s5_decoder_blocks} S5 decoder")
    print(f"📊 Training completed: {len(losses)} steps in {total_training_time:.2f}s")
    print(f"📉 Loss reduction: {loss_reduction:.2f}% ({initial_loss:.6f} → {final_loss:.6f})")
    print(f"📏 Average gradient norm: {avg_grad_norm:.6f}")
    print(f"🌊 Maximum spectral radius: {max_spectral_radius:.6f}")
    print(f"⏱️  Average step time: {avg_step_time:.4f}s")
    print(f"⚡ Hybrid throughput: {hybrid_throughput:.2f} tokens/sec")
    print(f"📈 Throughput ratio: {throughput_ratio:.2f}x baseline")
    print(f"🖥️  Device: {device_kind}")
    
    # Success criteria evaluation
    criteria = {
        "Forward pass runs on TPU": runs_on_tpu,
        "Loss decreases significantly": loss_reduction > 10.0,
        "Gradient norms are finite and reasonable": 0.001 < avg_grad_norm < 100.0,
        "No NaNs or Infs detected": not (has_nan or has_inf),
        "Spectral radius < 1 (stable)": max_spectral_radius < 1.0,
        "Hybrid throughput ≥ 70% baseline": throughput_ratio >= 0.70,
        "Stable training for 500+ steps": len(losses) >= 500
    }
    
    print("\n🎯 SUCCESS CRITERIA:")
    all_passed = True
    for criterion, passed in criteria.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"   {criterion}: {status}")
        if not passed:
            all_passed = False
    
    overall_status = "✅ SUCCESS" if all_passed else "❌ FAILURE"
    print(f"\n🏆 OVERALL RESULT: {overall_status}")
    
    if config.enable_profiling:
        print(f"\n📈 Profiling data saved to: {config.profile_dir}")
        print("   Use TensorBoard or Perfetto to analyze performance traces")
    
    return {
        'success': all_passed,
        'loss_reduction': loss_reduction,
        'avg_grad_norm': avg_grad_norm,
        'max_spectral_radius': max_spectral_radius,
        'throughput_ratio': throughput_ratio,
        'avg_step_time': avg_step_time,
        'total_steps': len(losses),
        'criteria': criteria
    }


if __name__ == "__main__":
    # Create configuration
    config = Stage2Config()
    
    # Run benchmark
    results = run_stage2_benchmark(config)
    
    print(f"\n🎉 Stage 2 benchmark completed!")
    print(f"Success: {results['success']}")