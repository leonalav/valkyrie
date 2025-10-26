"""
HRM Training Package

Training step implementations with deep supervision, one-step gradient updates,
and proper loss computation for the Hierarchical Reasoning Model.
"""

import importlib.util
import os

# Load the sibling training.py module directly to avoid relative import issues
_training_py_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'training.py')
_spec = importlib.util.spec_from_file_location("hrm_training_module", _training_py_path)
_hrm_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_hrm_mod)

# Re-export core HRM training symbols
HRMTrainingState = _hrm_mod.HRMTrainingState
TrainingMetrics = _hrm_mod.TrainingMetrics
LossConfig = _hrm_mod.LossConfig
compute_total_loss = _hrm_mod.compute_total_loss
create_train_state = _hrm_mod.create_train_state
segment_train_step = _hrm_mod.segment_train_step
train_segments = _hrm_mod.train_segments
analyze_gradient_flow = _hrm_mod.analyze_gradient_flow
validate_carry_detachment = _hrm_mod.validate_carry_detachment
HRMTrainingLoop = getattr(_hrm_mod, 'HRMTrainingLoop', None)

# Try to load TPU/distributed helpers from repository root
# Determine repository root: .../codebase from current file path
_def_path = os.path.dirname(__file__)  # .../src/model/hrm/training
_model_path = os.path.dirname(_def_path)  # .../src/model/hrm
_src_path = os.path.dirname(_model_path)  # .../src/model
_codebase_path = os.path.dirname(_src_path)  # .../src
_repo_root = os.path.dirname(_codebase_path)  # .../codebase
_ctd_path = os.path.join(_repo_root, 'configure_tpu_distributed.py')

configure_jax_for_tpu = None
create_device_mesh = None
create_sharding_strategy = None
estimate_memory_usage = None
shard_batch_to_devices = None
replicate_params_to_devices = None
all_reduce_gradients = None
setup_distributed_training = None

try:
    if os.path.exists(_ctd_path):
        _ctd_spec = importlib.util.spec_from_file_location('configure_tpu_distributed', _ctd_path)
        _ctd_mod = importlib.util.module_from_spec(_ctd_spec)
        _ctd_spec.loader.exec_module(_ctd_mod)
        configure_jax_for_tpu = getattr(_ctd_mod, 'configure_jax_for_tpu', None)
        create_device_mesh = getattr(_ctd_mod, 'create_device_mesh', None)
        create_sharding_strategy = getattr(_ctd_mod, 'create_sharding_strategy', None)
        estimate_memory_usage = getattr(_ctd_mod, 'estimate_memory_usage', None)
        shard_batch_to_devices = getattr(_ctd_mod, 'shard_batch_to_devices', None)
        replicate_params_to_devices = getattr(_ctd_mod, 'replicate_params_to_devices', None)
        all_reduce_gradients = getattr(_ctd_mod, 'all_reduce_gradients', None)
        setup_distributed_training = getattr(_ctd_mod, 'setup_distributed_training', None)
except Exception:
    # Optional: ignore if TPU helpers fail to import
    pass

__all__ = [
    'HRMTrainingState',
    'TrainingMetrics', 
    'LossConfig',
    'compute_total_loss',
    'create_train_state',
    'segment_train_step',
    'train_segments',
    'analyze_gradient_flow',
    'validate_carry_detachment',
    'HRMTrainingLoop',
    # TPU/distributed helpers
    'configure_jax_for_tpu',
    'create_device_mesh',
    'create_sharding_strategy',
    'estimate_memory_usage',
    'shard_batch_to_devices',
    'replicate_params_to_devices',
    'all_reduce_gradients',
    'setup_distributed_training',
]