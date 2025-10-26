"""
Comprehensive evaluation suite for BigBird+S5+HRM model.

This module provides evaluation capabilities for:
- Algorithmic tasks (non-CoT I/O tasks)
- Code tasks (unit test execution, code completion)
- Long-context tasks (document retrieval, cross-reference)
- HRM-specific metrics (computational efficiency, state utilization)

As specified in the PLAN training document.
"""

from .algorithmic_tasks import AlgorithmicEvaluator, AlgorithmicTask
from .code_tasks import CodeTaskEvaluator, CodeTask
from .long_context_tasks import LongContextEvaluator, LongContextTask
from .hrm_metrics import HRMMetricsEvaluator, HRMMetrics
from .evaluation_suite import EvaluationSuite, EvaluationConfig, EvaluationResults

__all__ = [
    "AlgorithmicEvaluator",
    "AlgorithmicTask", 
    "CodeTaskEvaluator",
    "CodeTask",
    "LongContextEvaluator",
    "LongContextTask",
    "HRMMetricsEvaluator", 
    "HRMMetrics",
    "EvaluationSuite",
    "EvaluationConfig",
    "EvaluationResults",
]