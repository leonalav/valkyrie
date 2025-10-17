"""Sharded optimizer implementation for TPU training.

Implements:
- AdamW optimizer with proper weight decay
- Learning rate schedules with warmup
- Gradient clipping and accumulation
- Sharded optimizer state across TPU mesh
- Mixed precision support
"""

import jax
import jax.numpy as jnp
import optax
from typing import Dict, Any, Optional, Callable, Union
import math
import logging

from ..model import ValkyrieConfig

logger = logging.getLogger(__name__)


def create_lr_schedule(
    base_lr: float,
    warmup_steps: int,
    total_steps: int,
    schedule_type: str = "cosine",
    min_lr_ratio: float = 0.1,
    **kwargs
) -> optax.Schedule:
    """
    Create learning rate schedule with warmup.
    
    Args:
        base_lr: Peak learning rate
        warmup_steps: Number of warmup steps
        total_steps: Total training steps
        schedule_type: Type of schedule ("cosine", "linear", "constant")
        min_lr_ratio: Minimum LR as ratio of base_lr
        
    Returns:
        Optax learning rate schedule
    """
    
    min_lr = base_lr * min_lr_ratio
    
    if schedule_type == "cosine":
        schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=base_lr,
            warmup_steps=warmup_steps,
            decay_steps=total_steps - warmup_steps,
            end_value=min_lr
        )
    elif schedule_type == "linear":
        schedule = optax.warmup_linear_decay_schedule(
            init_value=0.0,
            peak_value=base_lr,
            warmup_steps=warmup_steps,
            decay_steps=total_steps - warmup_steps,
            end_value=min_lr
        )
    elif schedule_type == "constant":
        schedule = optax.warmup_constant_schedule(
            init_value=0.0,
            peak_value=base_lr,
            warmup_steps=warmup_steps,
        )
    else:
        raise ValueError(f"Unknown schedule type: {schedule_type}")
    
    logger.info(f"Created {schedule_type} LR schedule: {base_lr} -> {min_lr} over {total_steps} steps")
    
    return schedule


def create_optimizer(
    config: ValkyrieConfig,
    learning_rate: Union[float, optax.Schedule],
    total_steps: Optional[int] = None,
    warmup_steps: Optional[int] = None,
    **kwargs
) -> optax.GradientTransformation:
    """
    Create AdamW optimizer with proper configuration for Valkyrie training.
    
    Args:
        config: Model configuration
        learning_rate: Learning rate or schedule
        total_steps: Total training steps (for schedule)
        warmup_steps: Warmup steps (defaults to 10% of total_steps)
        
    Returns:
        Optax optimizer
    """
    
    # Create learning rate schedule if needed
    if isinstance(learning_rate, float) and total_steps is not None:
        if warmup_steps is None:
            warmup_steps = max(1000, int(0.1 * total_steps))
        
        lr_schedule = create_lr_schedule(
            base_lr=learning_rate,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            schedule_type=kwargs.get('schedule_type', 'cosine'),
            min_lr_ratio=kwargs.get('min_lr_ratio', 0.1)
        )
    else:
        lr_schedule = learning_rate
    
    # Optimizer configuration
    b1 = kwargs.get('b1', 0.9)
    b2 = kwargs.get('b2', 0.95)
    eps = kwargs.get('eps', 1e-8)
    weight_decay = kwargs.get('weight_decay', config.weight_decay)
    gradient_clipping = kwargs.get('gradient_clipping', config.gradient_clipping)
    
    logger.info(f"Creating AdamW optimizer:")
    logger.info(f"  Learning rate: {learning_rate}")
    logger.info(f"  Weight decay: {weight_decay}")
    logger.info(f"  Gradient clipping: {gradient_clipping}")
    logger.info(f"  Beta1: {b1}, Beta2: {b2}")
    logger.info(f"  Epsilon: {eps}")
    
    # Create optimizer chain
    optimizer_chain = []
    
    # Gradient clipping (before other transformations)
    if gradient_clipping > 0:
        optimizer_chain.append(optax.clip_by_global_norm(gradient_clipping))
    
    # AdamW with weight decay
    optimizer_chain.append(
        optax.adamw(
            learning_rate=lr_schedule,
            b1=b1,
            b2=b2,
            eps=eps,
            weight_decay=weight_decay,
        )
    )
    
    # Combine transformations
    optimizer = optax.chain(*optimizer_chain)
    
    return optimizer


def create_sharded_optimizer(
    config: ValkyrieConfig,
    learning_rate: Union[float, optax.Schedule],
    partition_specs: Dict[str, Any],
    total_steps: Optional[int] = None,
    **kwargs
) -> optax.GradientTransformation:
    """
    Create optimizer with proper sharding for TPU training.
    
    Args:
        config: Model configuration
        learning_rate: Learning rate or schedule
        partition_specs: Parameter partition specifications
        total_steps: Total training steps
        
    Returns:
        Sharded optimizer
    """
    
    # Create base optimizer
    base_optimizer = create_optimizer(
        config=config,
        learning_rate=learning_rate,
        total_steps=total_steps,
        **kwargs
    )
    
    # Wrap with sharding (optimizer state will follow parameter sharding)
    # This is handled automatically by pjit when we initialize optimizer state
    
    return base_optimizer


def create_gradient_accumulation_optimizer(
    base_optimizer: optax.GradientTransformation,
    accumulation_steps: int,
) -> optax.GradientTransformation:
    """
    Wrap optimizer with gradient accumulation.
    
    Args:
        base_optimizer: Base optimizer to wrap
        accumulation_steps: Number of steps to accumulate gradients
        
    Returns:
        Optimizer with gradient accumulation
    """
    
    if accumulation_steps <= 1:
        return base_optimizer
    
    logger.info(f"Adding gradient accumulation over {accumulation_steps} steps")
    
    # Use optax.MultiSteps for gradient accumulation
    accumulated_optimizer = optax.MultiSteps(
        base_optimizer,
        every_k_schedule=accumulation_steps,
        use_grad_mean=True,  # Average gradients over accumulation steps
    )
    
    return accumulated_optimizer


def create_mixed_precision_optimizer(
    base_optimizer: optax.GradientTransformation,
    config: ValkyrieConfig,
) -> optax.GradientTransformation:
    """
    Wrap optimizer for mixed precision training.
    
    Args:
        base_optimizer: Base optimizer
        config: Model configuration
        
    Returns:
        Mixed precision optimizer
    """
    
    # For TPU, we handle mixed precision in the model forward pass
    # The optimizer works with fp32 gradients
    # Just add gradient scaling if needed
    
    # Add gradient scaling for stability (optional)
    # This is typically not needed on TPU with bfloat16
    
    return base_optimizer


def get_optimizer_info(optimizer: optax.GradientTransformation, step: int) -> Dict[str, Any]:
    """
    Get information about optimizer state.
    
    Args:
        optimizer: Optimizer instance
        step: Current training step
        
    Returns:
        Dictionary with optimizer information
    """
    
    info = {
        'step': step,
        'optimizer_type': type(optimizer).__name__,
    }
    
    # Try to get learning rate if available
    try:
        if hasattr(optimizer, '_schedule'):
            info['learning_rate'] = float(optimizer._schedule(step))
        elif hasattr(optimizer, 'learning_rate'):
            lr = optimizer.learning_rate
            if callable(lr):
                info['learning_rate'] = float(lr(step))
            else:
                info['learning_rate'] = float(lr)
    except:
        info['learning_rate'] = None
    
    return info


# Preset optimizer configurations
def get_longformer_optimizer_config() -> Dict[str, Any]:
    """Get optimizer configuration for Longformer training (from paper)."""
    return {
        'b1': 0.9,
        'b2': 0.98,
        'eps': 1e-6,
        'weight_decay': 0.01,
        'gradient_clipping': 0.25,
        'schedule_type': 'linear',
        'min_lr_ratio': 0.0,
    }


def get_s5_optimizer_config() -> Dict[str, Any]:
    """Get optimizer configuration for S5 training (from paper)."""
    return {
        'b1': 0.9,
        'b2': 0.999,
        'eps': 1e-8,
        'weight_decay': 0.05,
        'gradient_clipping': 1.0,
        'schedule_type': 'cosine',
        'min_lr_ratio': 0.1,
    }


def get_valkyrie_optimizer_config() -> Dict[str, Any]:
    """Get balanced optimizer configuration for Valkyrie (Longformer + S5)."""
    return {
        'b1': 0.9,
        'b2': 0.95,
        'eps': 1e-8,
        'weight_decay': 0.1,
        'gradient_clipping': 1.0,
        'schedule_type': 'cosine',
        'min_lr_ratio': 0.1,
    }


class OptimizerManager:
    """
    Manager for optimizer state and scheduling.
    
    Handles:
    - Optimizer creation and configuration
    - Learning rate scheduling
    - Gradient accumulation
    - Mixed precision
    - Checkpointing optimizer state
    """
    
    def __init__(
        self,
        config: ValkyrieConfig,
        total_steps: int,
        base_lr: float = 2e-4,
        optimizer_config: Optional[Dict[str, Any]] = None,
    ):
        self.config = config
        self.total_steps = total_steps
        self.base_lr = base_lr
        
        # Use default Valkyrie config if none provided
        if optimizer_config is None:
            optimizer_config = get_valkyrie_optimizer_config()
        
        self.optimizer_config = optimizer_config
        self.optimizer = None
        
    def create_optimizer(
        self,
        partition_specs: Optional[Dict[str, Any]] = None,
        accumulation_steps: int = 1,
    ) -> optax.GradientTransformation:
        """Create and configure optimizer."""
        
        # Create base optimizer
        if partition_specs is not None:
            optimizer = create_sharded_optimizer(
                config=self.config,
                learning_rate=self.base_lr,
                partition_specs=partition_specs,
                total_steps=self.total_steps,
                **self.optimizer_config
            )
        else:
            optimizer = create_optimizer(
                config=self.config,
                learning_rate=self.base_lr,
                total_steps=self.total_steps,
                **self.optimizer_config
            )
        
        # Add gradient accumulation if needed
        if accumulation_steps > 1:
            optimizer = create_gradient_accumulation_optimizer(
                optimizer, accumulation_steps
            )
        
        # Add mixed precision support
        optimizer = create_mixed_precision_optimizer(optimizer, self.config)
        
        self.optimizer = optimizer
        return optimizer
    
    def get_lr_at_step(self, step: int) -> float:
        """Get learning rate at specific step."""
        if self.optimizer is None:
            raise ValueError("Optimizer not created yet")
        
        info = get_optimizer_info(self.optimizer, step)
        return info.get('learning_rate', self.base_lr)
    
    def get_optimizer_state_info(self, opt_state: Any) -> Dict[str, Any]:
        """Get information about optimizer state."""
        
        def count_params(tree):
            return sum(x.size for x in jax.tree_leaves(tree))
        
        def get_memory_usage(tree):
            total_bytes = sum(x.nbytes for x in jax.tree_leaves(tree))
            return total_bytes / (1024**3)  # GB
        
        info = {
            'num_params': count_params(opt_state),
            'memory_gb': get_memory_usage(opt_state),
            'state_structure': jax.tree_map(lambda x: x.shape, opt_state),
        }
        
        return info