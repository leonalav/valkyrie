"""Alias package to maintain backwards compatibility.

Exports training utilities from src/train under the src.training namespace,
and provides convenience re-exports for HRM training and TPU configuration.
"""

# Core training loop utilities from src.train
from ..train import (
    TrainingLoop,
    create_train_step,
    create_eval_step,
    create_optimizer,
    create_lr_schedule,
)

# Training state used by the general training loop
from ..train.train_loop import TrainingState

# HRM training utilities
from ..model.hrm.training import (
    TrainingMetrics,
    LossConfig,
    compute_total_loss,
    segment_train_step,
    create_train_state as create_hrm_train_state,
    train_segments,
    analyze_gradient_flow,
    validate_carry_detachment,
)

# TPU/distributed helpers (available at repo root)
try:
    from configure_tpu_distributed import (
        configure_jax_for_tpu,
        create_device_mesh,
        create_sharding_strategy,
        estimate_memory_usage,
        shard_batch_to_devices,
        replicate_params_to_devices,
        all_reduce_gradients,
        setup_distributed_training,
    )
except Exception:
    # These helpers are optional and only needed in distributed tests
    configure_jax_for_tpu = None
    create_device_mesh = None
    create_sharding_strategy = None
    estimate_memory_usage = None
    shard_batch_to_devices = None
    replicate_params_to_devices = None
    all_reduce_gradients = None
    setup_distributed_training = None

# Convenience wrapper to detach HRM carry trees without requiring HRMTrainingState
import jax

def detach_carry(carry):
    """Detach a carry pytree to prevent gradient flow between segments."""
    return jax.tree.map(jax.lax.stop_gradient, carry)

# Backwards-compatible alias: expose create_train_state under a generic name
# Prefer using create_hrm_train_state when working directly with HRM models.
def create_train_state(*args, **kwargs):
    """Create training state for HRM models (alias to hrm.training.create_train_state)."""
    return create_hrm_train_state(*args, **kwargs)

__all__ = [
    # Core training loop
    "TrainingLoop",
    "TrainingState",
    "create_train_step",
    "create_eval_step",
    "create_optimizer",
    "create_lr_schedule",
    # HRM training
    "TrainingMetrics",
    "LossConfig",
    "compute_total_loss",
    "segment_train_step",
    "create_train_state",
    "train_segments",
    "analyze_gradient_flow",
    "validate_carry_detachment",
    # TPU/distributed helpers
    "configure_jax_for_tpu",
    "create_device_mesh",
    "create_sharding_strategy",
    "estimate_memory_usage",
    "shard_batch_to_devices",
    "replicate_params_to_devices",
    "all_reduce_gradients",
    "setup_distributed_training",
    # Utilities
    "detach_carry",
]