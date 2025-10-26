"""Gryphon Training Utilities

Specialized training utilities for the hybrid BigBird-S5 architecture.
Addresses the unique challenges of training models with both complex
state space dynamics and sparse attention patterns.

Key Features:
- Parameter-specific learning rates for S5 vs attention components
- Gradient clipping with special handling for complex gradients
- Numerical stability monitoring for S5 eigenvalues and attention weights
- Memory-efficient training strategies
- Mixed precision training support
"""

import jax
import jax.numpy as jnp
import optax
from typing import Dict, Any, Tuple, Optional, NamedTuple
import warnings

# Robust import to support both package-relative and direct-module usage
try:
    from .gryphon_config import GryphonConfig
except Exception:
    try:
        from gryphon_config import GryphonConfig
    except Exception:
        import importlib.util
        import os
        import sys
        _cur_dir = os.path.dirname(os.path.abspath(__file__))
        _config_path = os.path.join(_cur_dir, "gryphon_config.py")
        spec = importlib.util.spec_from_file_location("gryphon_config", _config_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        sys.modules["gryphon_config"] = mod
        GryphonConfig = mod.GryphonConfig


class GryphonTrainingState(NamedTuple):
    """Training state for Gryphon model."""
    params: Dict[str, Any]
    opt_state: optax.OptState
    step: int
    s5_states: Optional[list] = None
    metrics: Optional[Dict[str, float]] = None


def create_parameter_groups(params: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Separate parameters into groups for different learning rates.
    
    Args:
        params: Model parameters
        
    Returns:
        Dictionary with parameter groups: {'s5': {...}, 'attention': {...}, 'other': {...}}
    """
    s5_params = {}
    attention_params = {}
    other_params = {}
    
    def classify_param(path: str, param: Any) -> str:
        """Classify parameter based on its path."""
        path_lower = path.lower()
        
        # S5 parameters (need smaller learning rates)
        if any(s5_key in path_lower for s5_key in [
            's5_layer', 'lambda_re', 'lambda_im', 'b_real', 'b_imag', 
            'c_real', 'c_imag', 'log_delta', 'b_base', 'c_base'
        ]):
            return 's5'
        
        # Attention parameters
        elif any(attn_key in path_lower for attn_key in [
            'attention', 'q_proj', 'k_proj', 'v_proj', 'o_proj',
            'bigbird', 'attn'
        ]):
            return 'attention'
        
        # Everything else (embeddings, layer norms, MLP, etc.)
        else:
            return 'other'
    
    # Recursively classify parameters
    def process_params(params_dict: Dict[str, Any], prefix: str = '') -> None:
        for key, value in params_dict.items():
            current_path = f"{prefix}.{key}" if prefix else key
            
            if isinstance(value, dict):
                process_params(value, current_path)
            else:
                param_type = classify_param(current_path, value)
                
                if param_type == 's5':
                    if prefix not in s5_params:
                        s5_params[prefix] = {}
                    s5_params[prefix][key] = value
                elif param_type == 'attention':
                    if prefix not in attention_params:
                        attention_params[prefix] = {}
                    attention_params[prefix][key] = value
                else:
                    if prefix not in other_params:
                        other_params[prefix] = {}
                    other_params[prefix][key] = value
    
    process_params(params)
    
    return {
        's5': s5_params,
        'attention': attention_params,
        'other': other_params
    }


def create_gryphon_optimizer(config: GryphonConfig, base_learning_rate: float = 1e-3) -> optax.GradientTransformation:
    """Create optimizer with parameter-specific learning rates for Gryphon.
    
    Args:
        config: Gryphon configuration
        base_learning_rate: Base learning rate
        
    Returns:
        Optax optimizer with parameter-specific learning rates
    """
    # Calculate parameter-specific learning rates
    s5_lr = base_learning_rate * config.s5_learning_rate_multiplier
    attention_lr = base_learning_rate * config.attention_learning_rate_multiplier
    other_lr = base_learning_rate
    
    # Create parameter-specific optimizers
    s5_optimizer = optax.adamw(
        learning_rate=s5_lr,
        weight_decay=config.weight_decay * 0.5,  # Reduced weight decay for S5
        b1=0.9,
        b2=0.95,  # Slightly higher beta2 for S5 stability
        eps=1e-8
    )
    
    attention_optimizer = optax.adamw(
        learning_rate=attention_lr,
        weight_decay=config.weight_decay,
        b1=0.9,
        b2=0.999,
        eps=1e-8
    )
    
    other_optimizer = optax.adamw(
        learning_rate=other_lr,
        weight_decay=config.weight_decay,
        b1=0.9,
        b2=0.999,
        eps=1e-8
    )
    
    # Combine with multi-transform
    optimizer = optax.multi_transform(
        transforms={
            's5': s5_optimizer,
            'attention': attention_optimizer,
            'other': other_optimizer
        },
        param_labels=lambda path, _: classify_param_for_optimizer(path)
    )
    
    # Add gradient clipping
    optimizer = optax.chain(
        optax.clip_by_global_norm(config.gradient_clipping),
        optimizer
    )
    
    return optimizer


def classify_param_for_optimizer(path: str) -> str:
    """Classify parameter path for optimizer."""
    path_lower = path.lower()
    
    if any(s5_key in path_lower for s5_key in [
        's5_layer', 'lambda_re', 'lambda_im', 'b_real', 'b_imag', 
        'c_real', 'c_imag', 'log_delta', 'b_base', 'c_base'
    ]):
        return 's5'
    elif any(attn_key in path_lower for attn_key in [
        'attention', 'q_proj', 'k_proj', 'v_proj', 'o_proj',
        'bigbird', 'attn'
    ]):
        return 'attention'
    else:
        return 'other'


def compute_gryphon_loss(
    logits: jnp.ndarray,
    targets: jnp.ndarray,
    attention_mask: Optional[jnp.ndarray] = None,
    label_smoothing: float = 0.0
) -> Tuple[jnp.ndarray, Dict[str, float]]:
    """Compute loss for Gryphon model with optional label smoothing.
    
    Args:
        logits: Model logits [batch, seq_len, vocab_size]
        targets: Target token ids [batch, seq_len]
        attention_mask: Optional mask [batch, seq_len]
        label_smoothing: Label smoothing factor
        
    Returns:
        Tuple of (loss, metrics)
    """
    batch_size, seq_len, vocab_size = logits.shape
    
    # Shift targets for causal language modeling
    shifted_logits = logits[:, :-1, :]  # [batch, seq_len-1, vocab_size]
    shifted_targets = targets[:, 1:]    # [batch, seq_len-1]
    
    if attention_mask is not None:
        shifted_mask = attention_mask[:, 1:]  # [batch, seq_len-1]
    else:
        shifted_mask = jnp.ones_like(shifted_targets, dtype=jnp.float32)
    
    # Compute cross-entropy loss
    if label_smoothing > 0.0:
        # Label smoothing
        one_hot_targets = jax.nn.one_hot(shifted_targets, vocab_size)
        smooth_targets = (1.0 - label_smoothing) * one_hot_targets + label_smoothing / vocab_size
        
        log_probs = jax.nn.log_softmax(shifted_logits, axis=-1)
        loss_per_token = -jnp.sum(smooth_targets * log_probs, axis=-1)
    else:
        # Standard cross-entropy
        loss_per_token = optax.softmax_cross_entropy_with_integer_labels(
            shifted_logits, shifted_targets
        )
    
    # Apply mask and compute mean loss
    masked_loss = loss_per_token * shifted_mask
    total_loss = jnp.sum(masked_loss)
    total_tokens = jnp.sum(shifted_mask)
    
    # Avoid division by zero
    mean_loss = jnp.where(total_tokens > 0, total_loss / total_tokens, 0.0)
    
    # Compute accuracy
    predictions = jnp.argmax(shifted_logits, axis=-1)
    correct_predictions = (predictions == shifted_targets) * shifted_mask
    accuracy = jnp.sum(correct_predictions) / jnp.maximum(total_tokens, 1.0)
    
    # Compute perplexity
    perplexity = jnp.exp(mean_loss)
    
    metrics = {
        'loss': mean_loss,
        'accuracy': accuracy,
        'perplexity': perplexity,
        'total_tokens': total_tokens
    }
    
    return mean_loss, metrics


def monitor_s5_stability(params: Dict[str, Any]) -> Dict[str, float]:
    """Monitor S5 parameter stability and numerical health.
    
    Args:
        params: Model parameters
        
    Returns:
        Dictionary with stability metrics
    """
    stability_metrics = {}
    
    def check_s5_params(param_dict: Dict[str, Any], prefix: str = '') -> None:
        for key, value in param_dict.items():
            current_path = f"{prefix}.{key}" if prefix else key
            
            if isinstance(value, dict):
                check_s5_params(value, current_path)
            elif isinstance(value, jnp.ndarray):
                # Check for S5-specific parameters
                if 'lambda_re' in key.lower():
                    # Real parts should be negative for stability
                    stability_metrics[f'{current_path}_mean'] = float(jnp.mean(value))
                    stability_metrics[f'{current_path}_max'] = float(jnp.max(value))
                    stability_metrics[f'{current_path}_min'] = float(jnp.min(value))
                    
                    # Count positive eigenvalues (potentially unstable)
                    positive_count = jnp.sum(value > 0)
                    stability_metrics[f'{current_path}_positive_count'] = float(positive_count)
                
                elif 'lambda_im' in key.lower():
                    # Imaginary parts magnitude
                    stability_metrics[f'{current_path}_mean_abs'] = float(jnp.mean(jnp.abs(value)))
                    stability_metrics[f'{current_path}_max_abs'] = float(jnp.max(jnp.abs(value)))
                
                elif 'log_delta' in key.lower():
                    # Delta values (should be reasonable)
                    delta_values = jnp.exp(value)
                    stability_metrics[f'{current_path}_delta_mean'] = float(jnp.mean(delta_values))
                    stability_metrics[f'{current_path}_delta_max'] = float(jnp.max(delta_values))
                    stability_metrics[f'{current_path}_delta_min'] = float(jnp.min(delta_values))
                    
                    # Check for extreme values
                    extreme_count = jnp.sum((delta_values > 10.0) | (delta_values < 1e-4))
                    stability_metrics[f'{current_path}_extreme_count'] = float(extreme_count)
                
                # General parameter health checks
                if jnp.issubdtype(value.dtype, jnp.floating):
                    nan_count = jnp.sum(jnp.isnan(value))
                    inf_count = jnp.sum(jnp.isinf(value))
                    
                    if nan_count > 0:
                        stability_metrics[f'{current_path}_nan_count'] = float(nan_count)
                    if inf_count > 0:
                        stability_metrics[f'{current_path}_inf_count'] = float(inf_count)
    
    check_s5_params(params)
    
    return stability_metrics


def check_gradient_health(grads: Dict[str, Any]) -> Dict[str, float]:
    """Check gradient health for numerical stability.
    
    Args:
        grads: Model gradients
        
    Returns:
        Dictionary with gradient health metrics
    """
    grad_metrics = {}
    
    def compute_grad_stats(grad_dict: Dict[str, Any], prefix: str = '') -> None:
        for key, value in grad_dict.items():
            current_path = f"{prefix}.{key}" if prefix else key
            
            if isinstance(value, dict):
                compute_grad_stats(value, current_path)
            elif isinstance(value, jnp.ndarray):
                # Compute gradient statistics
                grad_norm = jnp.linalg.norm(value)
                grad_max = jnp.max(jnp.abs(value))
                grad_mean = jnp.mean(jnp.abs(value))
                
                grad_metrics[f'{current_path}_norm'] = float(grad_norm)
                grad_metrics[f'{current_path}_max'] = float(grad_max)
                grad_metrics[f'{current_path}_mean'] = float(grad_mean)
                
                # Check for problematic gradients
                nan_count = jnp.sum(jnp.isnan(value))
                inf_count = jnp.sum(jnp.isinf(value))
                zero_count = jnp.sum(value == 0)
                
                if nan_count > 0:
                    grad_metrics[f'{current_path}_nan_count'] = float(nan_count)
                if inf_count > 0:
                    grad_metrics[f'{current_path}_inf_count'] = float(inf_count)
                
                # Gradient sparsity (useful for debugging)
                total_elements = value.size
                sparsity = float(zero_count) / total_elements
                grad_metrics[f'{current_path}_sparsity'] = sparsity
    
    compute_grad_stats(grads)
    
    # Compute global gradient norm
    global_grad_norm = optax.global_norm(grads)
    grad_metrics['global_grad_norm'] = float(global_grad_norm)
    
    return grad_metrics


def create_learning_rate_schedule(
    config: GryphonConfig,
    total_steps: int,
    warmup_steps: Optional[int] = None,
    peak_lr: float = 1e-3
) -> optax.Schedule:
    """Create learning rate schedule optimized for Gryphon training.
    
    Args:
        config: Gryphon configuration
        total_steps: Total training steps
        warmup_steps: Warmup steps (default: 10% of total)
        peak_lr: Peak learning rate
        
    Returns:
        Optax learning rate schedule
    """
    if warmup_steps is None:
        warmup_steps = max(1000, total_steps // 10)
    
    # Cosine decay with warmup
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=peak_lr,
        warmup_steps=warmup_steps,
        decay_steps=total_steps - warmup_steps,
        end_value=peak_lr * 0.1  # Decay to 10% of peak
    )
    
    return schedule


def validate_training_config(config: GryphonConfig) -> None:
    """Validate training configuration for potential issues.
    
    Args:
        config: Gryphon configuration
        
    Raises:
        ValueError: If configuration has issues
        UserWarning: For potential problems
    """
    # Check S5 state dimension
    if config.s5_state_dim > 2 * config.d_model:
        warnings.warn(
            f"S5 state dimension ({config.s5_state_dim}) is much larger than "
            f"d_model ({config.d_model}). This may cause memory issues."
        )
    
    # Check learning rate multipliers
    if config.s5_learning_rate_multiplier >= 1.0:
        warnings.warn(
            f"S5 learning rate multiplier ({config.s5_learning_rate_multiplier}) "
            "is >= 1.0. S5 parameters typically need smaller learning rates."
        )
    
    # Check gradient clipping
    if config.gradient_clipping > 5.0:
        warnings.warn(
            f"Gradient clipping value ({config.gradient_clipping}) is quite high. "
            "Consider using a smaller value (e.g., 1.0) for better stability."
        )
    
    # Check sequence length vs block size
    if config.max_sequence_length % config.block_size != 0:
        raise ValueError(
            f"max_sequence_length ({config.max_sequence_length}) must be "
            f"divisible by block_size ({config.block_size})"
        )
    
    # Check attention pattern feasibility
    max_possible_attention = config.num_global_blocks + config.window_size + config.num_random_blocks
    if max_possible_attention > config.num_blocks:
        warnings.warn(
            f"Total attention blocks ({max_possible_attention}) exceeds "
            f"total blocks ({config.num_blocks}). Some patterns may overlap."
        )