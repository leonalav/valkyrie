"""
Unified evaluation suite for BigBird+S5+HRM model.

Orchestrates all evaluation components and provides a single interface
for comprehensive model evaluation as specified in the PLAN.
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Callable
from dataclasses import dataclass
import logging
import time
import json
from pathlib import Path

from .algorithmic_tasks import AlgorithmicEvaluator, AlgorithmicTask
from .code_tasks import CodeTaskEvaluator, CodeTask
from .long_context_tasks import LongContextEvaluator, LongContextTask
from .hrm_metrics import HRMMetricsEvaluator, HRMMetrics

logger = logging.getLogger(__name__)


@dataclass
class EvaluationConfig:
    """Configuration for evaluation suite."""
    
    # Task configuration
    num_algorithmic_tasks: int = 10
    num_code_tasks: int = 8
    num_long_context_tasks: int = 5  # Fewer due to computational cost
    
    # Difficulty levels
    difficulties: List[str] = None
    
    # Context lengths for long-context tasks
    context_lengths: List[int] = None
    
    # Batch sizes
    algorithmic_batch_size: int = 8
    code_batch_size: int = 4
    long_context_batch_size: int = 2
    
    # HRM-specific settings
    track_hrm_metrics: bool = True
    max_cycles: int = 8
    max_steps: int = 10
    
    # Output settings
    save_detailed_results: bool = True
    save_task_outputs: bool = False  # Can be large
    output_dir: Optional[str] = None
    
    def __post_init__(self):
        if self.difficulties is None:
            self.difficulties = ["easy", "medium", "hard"]
        if self.context_lengths is None:
            self.context_lengths = [4096, 8192, 16384]


@dataclass
class EvaluationResults:
    """Container for comprehensive evaluation results."""
    
    # Overall metrics
    overall_accuracy: float
    overall_exact_match: float
    
    # Task-specific results
    algorithmic_results: Dict[str, float]
    code_results: Dict[str, float]
    long_context_results: Dict[str, float]
    
    # HRM-specific metrics
    hrm_metrics: Optional[HRMMetrics]
    
    # Performance metrics
    total_evaluation_time: float
    tasks_per_second: float
    
    # Detailed results (optional)
    detailed_results: Optional[Dict[str, Any]] = None


class EvaluationSuite:
    """
    Comprehensive evaluation suite for BigBird+S5+HRM model.
    
    Orchestrates algorithmic, code, and long-context evaluations,
    tracks HRM-specific metrics, and provides unified reporting.
    """
    
    def __init__(
        self,
        config: EvaluationConfig,
        vocab_size: int = 32000,
        seed: int = 42,
    ):
        self.config = config
        self.vocab_size = vocab_size
        self.seed = seed
        
        # Initialize evaluators
        self.algorithmic_evaluator = AlgorithmicEvaluator(
            vocab_size=vocab_size,
            seed=seed
        )
        
        self.code_evaluator = CodeTaskEvaluator(
            vocab_size=vocab_size,
            seed=seed + 1
        )
        
        self.long_context_evaluator = LongContextEvaluator(
            vocab_size=vocab_size,
            seed=seed + 2
        )
        
        if config.track_hrm_metrics:
            self.hrm_evaluator = HRMMetricsEvaluator(
                max_cycles=config.max_cycles,
                max_steps=config.max_steps,
                track_gradients=True,
            )
        else:
            self.hrm_evaluator = None
        
        # Task cache
        self._task_cache = {}
        
        logger.info(f"Initialized evaluation suite with config: {config}")
    
    def generate_evaluation_tasks(self) -> Dict[str, List]:
        """Generate all evaluation tasks."""
        
        if "all_tasks" in self._task_cache:
            return self._task_cache["all_tasks"]
        
        logger.info("Generating evaluation tasks...")
        start_time = time.time()
        
        # Generate algorithmic tasks
        algorithmic_tasks = self.algorithmic_evaluator.create_evaluation_suite(
            num_tasks_per_type=self.config.num_algorithmic_tasks // 3,
            difficulties=self.config.difficulties
        )
        
        # Generate code tasks
        code_tasks = self.code_evaluator.create_evaluation_suite(
            num_tasks_per_type=self.config.num_code_tasks // 3,
            difficulties=self.config.difficulties
        )
        
        # Generate long-context tasks
        long_context_tasks = self.long_context_evaluator.create_evaluation_suite(
            num_tasks_per_type=self.config.num_long_context_tasks // 3,
            difficulties=self.config.difficulties,
            context_lengths=self.config.context_lengths
        )
        
        all_tasks = {
            "algorithmic": algorithmic_tasks,
            "code": code_tasks,
            "long_context": long_context_tasks,
        }
        
        self._task_cache["all_tasks"] = all_tasks
        
        generation_time = time.time() - start_time
        total_tasks = sum(len(tasks) for tasks in all_tasks.values())
        
        logger.info(f"Generated {total_tasks} evaluation tasks in {generation_time:.2f}s")
        logger.info(f"  Algorithmic: {len(algorithmic_tasks)}")
        logger.info(f"  Code: {len(code_tasks)}")
        logger.info(f"  Long-context: {len(long_context_tasks)}")
        
        return all_tasks
    
    def evaluate_model(
        self,
        model_fn: Callable,
        params: Dict[str, Any],
        tasks: Optional[Dict[str, List]] = None,
    ) -> EvaluationResults:
        """
        Comprehensive model evaluation.
        
        Args:
            model_fn: Model forward function
            params: Model parameters
            tasks: Optional pre-generated tasks
            
        Returns:
            EvaluationResults with comprehensive metrics
        """
        
        logger.info("Starting comprehensive model evaluation...")
        start_time = time.time()
        
        # Generate tasks if not provided
        if tasks is None:
            tasks = self.generate_evaluation_tasks()
        
        # Track all results
        all_results = {}
        all_accuracies = []
        all_exact_matches = []
        
        # Evaluate algorithmic tasks
        logger.info("Evaluating algorithmic tasks...")
        algorithmic_results = self.algorithmic_evaluator.evaluate_model(
            model_fn=model_fn,
            params=params,
            tasks=tasks["algorithmic"],
            batch_size=self.config.algorithmic_batch_size,
        )
        all_results["algorithmic"] = algorithmic_results
        all_accuracies.append(algorithmic_results["accuracy"])
        all_exact_matches.append(algorithmic_results["exact_match"])
        
        # Evaluate code tasks
        logger.info("Evaluating code tasks...")
        code_results = self.code_evaluator.evaluate_model(
            model_fn=model_fn,
            params=params,
            tasks=tasks["code"],
            batch_size=self.config.code_batch_size,
        )
        all_results["code"] = code_results
        all_accuracies.append(code_results["accuracy"])
        all_exact_matches.append(code_results["exact_match"])
        
        # Evaluate long-context tasks
        logger.info("Evaluating long-context tasks...")
        long_context_results = self.long_context_evaluator.evaluate_model(
            model_fn=model_fn,
            params=params,
            tasks=tasks["long_context"],
            batch_size=self.config.long_context_batch_size,
        )
        all_results["long_context"] = long_context_results
        all_accuracies.append(long_context_results["accuracy"])
        all_exact_matches.append(long_context_results["exact_match"])
        
        # Compute overall metrics
        overall_accuracy = np.mean(all_accuracies)
        overall_exact_match = np.mean(all_exact_matches)
        
        # Evaluate HRM-specific metrics (if enabled)
        hrm_metrics = None
        if self.hrm_evaluator is not None:
            logger.info("Computing HRM-specific metrics...")
            hrm_metrics = self._compute_hrm_metrics(
                model_fn, params, tasks
            )
        
        # Compute performance metrics
        total_time = time.time() - start_time
        total_tasks = sum(len(task_list) for task_list in tasks.values())
        tasks_per_second = total_tasks / total_time
        
        # Create results
        results = EvaluationResults(
            overall_accuracy=overall_accuracy,
            overall_exact_match=overall_exact_match,
            algorithmic_results=algorithmic_results,
            code_results=code_results,
            long_context_results=long_context_results,
            hrm_metrics=hrm_metrics,
            total_evaluation_time=total_time,
            tasks_per_second=tasks_per_second,
        )
        
        # Add detailed results if requested
        if self.config.save_detailed_results:
            results.detailed_results = {
                "task_counts": {k: len(v) for k, v in tasks.items()},
                "evaluation_config": self.config.__dict__,
                "per_task_type_metrics": all_results,
            }
        
        # Save results if output directory specified
        if self.config.output_dir:
            self._save_results(results, tasks)
        
        logger.info(f"Evaluation completed in {total_time:.2f}s")
        logger.info(f"Overall accuracy: {overall_accuracy:.4f}")
        logger.info(f"Overall exact match: {overall_exact_match:.4f}")
        logger.info(f"Tasks per second: {tasks_per_second:.2f}")
        
        return results
    
    def _compute_hrm_metrics(
        self,
        model_fn: Callable,
        params: Dict[str, Any],
        tasks: Dict[str, List],
    ) -> Optional[HRMMetrics]:
        """Compute HRM-specific metrics across all tasks."""
        
        if self.hrm_evaluator is None:
            return None
        
        # Sample a subset of tasks for HRM evaluation (computational efficiency)
        sample_tasks = []
        for task_type, task_list in tasks.items():
            # Sample up to 5 tasks per type
            sample_size = min(5, len(task_list))
            sampled = np.random.choice(task_list, sample_size, replace=False)
            sample_tasks.extend(sampled)
        
        # Collect HRM metrics from model outputs
        all_cycles_used = []
        all_steps_per_cycle = []
        all_z_H_states = []
        all_z_L_states = []
        all_act_regularizer = []
        all_early_stopping = []
        all_task_completion = []
        
        for task in sample_tasks[:10]:  # Limit to 10 tasks for efficiency
            try:
                # Prepare input
                if hasattr(task, 'input_sequence'):
                    input_seq = task.input_sequence
                else:
                    continue
                
                # Add batch dimension
                batch_input = jnp.expand_dims(input_seq, 0)
                
                # Run model
                outputs = model_fn(params, batch_input)
                
                # Extract HRM-specific outputs
                if isinstance(outputs, dict):
                    cycles_used = outputs.get('cycles_used', jnp.array([0]))
                    steps_per_cycle = outputs.get('steps_per_cycle', jnp.zeros((1, 8)))
                    z_H_states = outputs.get('z_H_states', jnp.zeros((1, 8, 512)))
                    z_L_states = outputs.get('z_L_states', jnp.zeros((1, 8, 10, 512)))
                    act_regularizer = outputs.get('act_regularizer', jnp.array([0.0]))
                    early_stopping = outputs.get('early_stopping', jnp.array([False]))
                    
                    # Determine task completion (simplified)
                    predictions = outputs.get('logits', outputs.get('predictions'))
                    if predictions is not None and hasattr(task, 'target_output'):
                        pred = jnp.argmax(predictions, axis=-1) if predictions.ndim > 1 else predictions
                        target = task.target_output
                        completed = jnp.all(pred[:len(target)] == target)
                    else:
                        completed = jnp.array([True])  # Assume completed
                    
                    all_cycles_used.append(cycles_used)
                    all_steps_per_cycle.append(steps_per_cycle)
                    all_z_H_states.append(z_H_states)
                    all_z_L_states.append(z_L_states)
                    all_act_regularizer.append(act_regularizer)
                    all_early_stopping.append(early_stopping)
                    all_task_completion.append(completed)
                
            except Exception as e:
                logger.warning(f"Error computing HRM metrics for task: {e}")
                continue
        
        if not all_cycles_used:
            logger.warning("No HRM metrics collected")
            return None
        
        # Concatenate all metrics
        cycles_used = jnp.concatenate(all_cycles_used)
        steps_per_cycle = jnp.concatenate(all_steps_per_cycle, axis=0)
        z_H_states = jnp.concatenate(all_z_H_states, axis=0)
        z_L_states = jnp.concatenate(all_z_L_states, axis=0)
        act_regularizer = jnp.concatenate(all_act_regularizer)
        early_stopping = jnp.concatenate(all_early_stopping)
        task_completion = jnp.concatenate(all_task_completion)
        
        # Compute metrics
        computational_metrics = self.hrm_evaluator.compute_computational_efficiency(
            cycles_used, steps_per_cycle
        )
        
        state_metrics = self.hrm_evaluator.compute_state_utilization(
            z_H_states, z_L_states, cycles_used
        )
        
        gradient_metrics = self.hrm_evaluator.compute_gradient_approximation_quality(
            {}, {}, jnp.zeros(len(cycles_used)), jnp.zeros(len(cycles_used))
        )
        
        adaptive_metrics = self.hrm_evaluator.compute_adaptive_computation_metrics(
            act_regularizer, cycles_used, early_stopping, task_completion
        )
        
        reasoning_metrics = self.hrm_evaluator.compute_reasoning_depth_metrics(
            None, z_H_states, cycles_used, jnp.full(len(cycles_used), 1000)
        )
        
        # Aggregate metrics
        hrm_metrics = self.hrm_evaluator.aggregate_hrm_metrics(
            computational_metrics,
            state_metrics,
            gradient_metrics,
            adaptive_metrics,
            reasoning_metrics,
        )
        
        return hrm_metrics
    
    def _save_results(
        self,
        results: EvaluationResults,
        tasks: Dict[str, List],
    ):
        """Save evaluation results to disk."""
        
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save main results
        results_dict = {
            "overall_accuracy": float(results.overall_accuracy),
            "overall_exact_match": float(results.overall_exact_match),
            "algorithmic_results": results.algorithmic_results,
            "code_results": results.code_results,
            "long_context_results": results.long_context_results,
            "total_evaluation_time": results.total_evaluation_time,
            "tasks_per_second": results.tasks_per_second,
        }
        
        # Add HRM metrics if available
        if results.hrm_metrics:
            results_dict["hrm_metrics"] = {
                "avg_cycles_used": results.hrm_metrics.avg_cycles_used,
                "computation_efficiency": results.hrm_metrics.computation_efficiency,
                "planner_state_utilization": results.hrm_metrics.planner_state_utilization,
                "executor_state_utilization": results.hrm_metrics.executor_state_utilization,
                "task_completion_rate": results.hrm_metrics.task_completion_rate,
                "reasoning_depth_achieved": results.hrm_metrics.reasoning_depth_achieved,
            }
        
        # Add detailed results if available
        if results.detailed_results:
            results_dict["detailed_results"] = results.detailed_results
        
        # Save to JSON
        results_file = output_dir / "evaluation_results.json"
        with open(results_file, 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        logger.info(f"Saved evaluation results to {results_file}")
        
        # Save task outputs if requested
        if self.config.save_task_outputs:
            tasks_file = output_dir / "evaluation_tasks.json"
            # Note: This could be large, so we save a summary
            task_summary = {
                task_type: {
                    "count": len(task_list),
                    "sample_names": [task.name for task in task_list[:5]]
                }
                for task_type, task_list in tasks.items()
            }
            
            with open(tasks_file, 'w') as f:
                json.dump(task_summary, f, indent=2)
            
            logger.info(f"Saved task summary to {tasks_file}")
    
    def create_evaluation_report(self, results: EvaluationResults) -> str:
        """Create a human-readable evaluation report."""
        
        report = []
        report.append("=" * 60)
        report.append("BigBird+S5+HRM Model Evaluation Report")
        report.append("=" * 60)
        report.append("")
        
        # Overall metrics
        report.append("OVERALL PERFORMANCE")
        report.append("-" * 20)
        report.append(f"Overall Accuracy: {results.overall_accuracy:.4f}")
        report.append(f"Overall Exact Match: {results.overall_exact_match:.4f}")
        report.append(f"Evaluation Time: {results.total_evaluation_time:.2f}s")
        report.append(f"Tasks per Second: {results.tasks_per_second:.2f}")
        report.append("")
        
        # Task-specific results
        report.append("TASK-SPECIFIC RESULTS")
        report.append("-" * 22)
        
        # Algorithmic tasks
        report.append("Algorithmic Tasks:")
        for metric, value in results.algorithmic_results.items():
            if isinstance(value, (int, float)):
                report.append(f"  {metric}: {value:.4f}")
        report.append("")
        
        # Code tasks
        report.append("Code Tasks:")
        for metric, value in results.code_results.items():
            if isinstance(value, (int, float)):
                report.append(f"  {metric}: {value:.4f}")
        report.append("")
        
        # Long-context tasks
        report.append("Long-Context Tasks:")
        for metric, value in results.long_context_results.items():
            if isinstance(value, (int, float)):
                report.append(f"  {metric}: {value:.4f}")
        report.append("")
        
        # HRM metrics
        if results.hrm_metrics:
            report.append("HRM-SPECIFIC METRICS")
            report.append("-" * 20)
            report.append(f"Average Cycles Used: {results.hrm_metrics.avg_cycles_used:.2f}")
            report.append(f"Computation Efficiency: {results.hrm_metrics.computation_efficiency:.4f}")
            report.append(f"Planner State Utilization: {results.hrm_metrics.planner_state_utilization:.4f}")
            report.append(f"Executor State Utilization: {results.hrm_metrics.executor_state_utilization:.4f}")
            report.append(f"Task Completion Rate: {results.hrm_metrics.task_completion_rate:.4f}")
            report.append(f"Reasoning Depth: {results.hrm_metrics.reasoning_depth_achieved:.4f}")
            report.append(f"ACT Regularizer: {results.hrm_metrics.act_regularizer_value:.6f}")
            report.append("")
        
        report.append("=" * 60)
        
        return "\n".join(report)