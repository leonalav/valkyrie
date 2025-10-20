"""
HRM Training Package

Training step implementations with deep supervision, one-step gradient updates,
and proper loss computation for the Hierarchical Reasoning Model.
"""

# Import all classes and functions from the training.py file using absolute import
import importlib.util
import os

# Get the path to training.py in the parent directory
training_py_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'training.py')

# Load the training.py module directly
spec = importlib.util.spec_from_file_location("hrm_training_module", training_py_path)
training_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(training_module)

# Import all the classes and functions
HRMTrainingState = training_module.HRMTrainingState
TrainingMetrics = training_module.TrainingMetrics
LossConfig = training_module.LossConfig
create_train_state = training_module.create_train_state
segment_train_step = training_module.segment_train_step
train_segments = training_module.train_segments
analyze_gradient_flow = training_module.analyze_gradient_flow
validate_carry_detachment = training_module.validate_carry_detachment

# Make all imports available at package level
__all__ = [
    'HRMTrainingState',
    'TrainingMetrics', 
    'LossConfig',
    'create_train_state',
    'segment_train_step',
    'train_segments',
    'analyze_gradient_flow',
    'validate_carry_detachment'
]