"""
HRM-specific evaluation metrics.

Implements metrics for tracking HRM computational efficiency, state utilization,
gradient approximation quality, and adaptive computation patterns.
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class HRMMetrics:
    """Container for HRM-specific metrics."""
    
    # Computational efficiency
    avg_cycles_used: float
    avg_steps_per_cycle: float
    computation_efficiency: float  # Useful computation / total computation
    
    # State utilization
    planner_state_utilization: float  # How much of z_H is used
    executor_state_utilization: float  # How much of z_L is used
    state_persistence: float  # How long states remain useful
    
    # Gradient approximation quality
    gradient_approximation_error: float
    one_step_gradient_quality: float
    deep_supervision_alignment: float
    
    # Adaptive computation patterns
    act_regularizer_value: float
    computation_distribution: Dict[str, float]  # Distribution of cycles used
    early_stopping_rate: float
    
    # Task-specific performance
    task_completion_rate: float
    reasoning_depth_achieved: float
    long_context_retention: float


class HRMMetricsEvaluator:
    """
    Evaluates HRM-specific metrics during training and inference.
    
    Tracks computational efficiency, state utilization, gradient quality,
    and adaptive computation patterns as specified in the PLAN.
    """
    
    def __init__(
        self,
        max_cycles: int = 8,
        max_steps: int = 10,
        state_dim: int = 512,
        track_gradients: bool = True,
    ):
        self.max_cycles = max_cycles
        self.max_steps = max_steps
        self.state_dim = state_dim
        self.track_gradients = track_gradients
        
        # Thresholds for state utilization
        self.utilization_threshold = 0.1  # Minimum activation to consider "used"
        self.persistence_window = 5  # Steps to track state persistence
    
    def compute_computational_efficiency(
        self,
        cycles_used: jnp.ndarray,
        steps_per_cycle: jnp.ndarray,
        task_difficulty: Optional[jnp.ndarray] = None,
    ) -> Dict[str, float]:
        """
        Compute computational efficiency metrics.
        
        Args:
            cycles_used: Number of cycles used per example [batch_size]
            steps_per_cycle: Steps used per cycle [batch_size, max_cycles]
            task_difficulty: Optional task difficulty scores [batch_size]
            
        Returns:
            Dictionary of efficiency metrics
        """
        
        batch_size = cycles_used.shape[0]
        
        # Basic efficiency metrics
        avg_cycles = jnp.mean(cycles_used)
        avg_steps = jnp.mean(steps_per_cycle[steps_per_cycle > 0])
        
        # Compute efficiency score
        # Efficiency = (task completion) / (computational cost)
        max_possible_computation = self.max_cycles * self.max_steps
        actual_computation = jnp.sum(steps_per_cycle, axis=1)
        
        # Normalize by task difficulty if available
        if task_difficulty is not None:
            expected_computation = task_difficulty * max_possible_computation
            efficiency = jnp.mean(jnp.minimum(expected_computation / (actual_computation + 1e-8), 1.0))
        else:
            efficiency = jnp.mean(1.0 - actual_computation / max_possible_computation)
        
        # Compute computation distribution
        cycle_distribution = jnp.bincount(cycles_used.astype(int), length=self.max_cycles + 1)
        cycle_distribution = cycle_distribution / jnp.sum(cycle_distribution)
        
        return {
            "avg_cycles_used": float(avg_cycles),
            "avg_steps_per_cycle": float(avg_steps),
            "computation_efficiency": float(efficiency),
            "cycle_distribution": {f"cycles_{i}": float(cycle_distribution[i]) 
                                 for i in range(len(cycle_distribution))},
            "computation_variance": float(jnp.var(actual_computation)),
        }
    
    def compute_state_utilization(
        self,
        z_H_states: jnp.ndarray,  # [batch, cycles, state_dim]
        z_L_states: jnp.ndarray,  # [batch, cycles, steps, state_dim]
        cycles_used: jnp.ndarray,  # [batch]
    ) -> Dict[str, float]:
        """
        Compute state utilization metrics.
        
        Args:
            z_H_states: Planner states
            z_L_states: Executor states
            cycles_used: Number of cycles used per example
            
        Returns:
            Dictionary of state utilization metrics
        """
        
        batch_size, max_cycles, state_dim = z_H_states.shape
        
        # Compute planner state utilization
        z_H_active = jnp.abs(z_H_states) > self.utilization_threshold
        planner_utilization = jnp.mean(z_H_active)
        
        # Compute executor state utilization
        z_L_active = jnp.abs(z_L_states) > self.utilization_threshold
        executor_utilization = jnp.mean(z_L_active)
        
        # Compute state persistence (how long states remain active)
        planner_persistence = self._compute_state_persistence(z_H_states, cycles_used)
        executor_persistence = self._compute_state_persistence(
            z_L_states.reshape(batch_size, -1, state_dim), cycles_used
        )
        
        # Compute state diversity (how different states are across cycles)
        planner_diversity = self._compute_state_diversity(z_H_states)
        executor_diversity = self._compute_state_diversity(z_L_states)
        
        # Compute state efficiency (useful activation / total activation)
        planner_efficiency = self._compute_state_efficiency(z_H_states, cycles_used)
        executor_efficiency = self._compute_state_efficiency(z_L_states, cycles_used)
        
        return {
            "planner_state_utilization": float(planner_utilization),
            "executor_state_utilization": float(executor_utilization),
            "planner_state_persistence": float(planner_persistence),
            "executor_state_persistence": float(executor_persistence),
            "planner_state_diversity": float(planner_diversity),
            "executor_state_diversity": float(executor_diversity),
            "planner_state_efficiency": float(planner_efficiency),
            "executor_state_efficiency": float(executor_efficiency),
        }
    
    def _compute_state_persistence(
        self,
        states: jnp.ndarray,  # [batch, time, state_dim]
        cycles_used: jnp.ndarray,
    ) -> float:
        """Compute how long states remain active/useful."""
        
        batch_size, max_time, state_dim = states.shape
        
        # Compute state activation over time
        state_active = jnp.abs(states) > self.utilization_threshold
        
        # For each batch element, compute persistence within used cycles
        persistence_scores = []
        
        for b in range(batch_size):
            used_cycles = int(cycles_used[b])
            if used_cycles <= 1:
                continue
                
            batch_states = state_active[b, :used_cycles]  # [used_cycles, state_dim]
            
            # Compute autocorrelation to measure persistence
            persistence = 0.0
            for lag in range(1, min(self.persistence_window, used_cycles)):
                if used_cycles - lag > 0:
                    corr = jnp.corrcoef(
                        batch_states[:-lag].flatten(),
                        batch_states[lag:].flatten()
                    )[0, 1]
                    if not jnp.isnan(corr):
                        persistence += corr / lag  # Weight by inverse lag
            
            persistence_scores.append(persistence)
        
        return jnp.mean(jnp.array(persistence_scores)) if persistence_scores else 0.0
    
    def _compute_state_diversity(self, states: jnp.ndarray) -> float:
        """Compute diversity of states across time."""
        
        # Flatten spatial dimensions, keep time dimension
        if states.ndim == 4:  # [batch, cycles, steps, state_dim]
            states = states.reshape(states.shape[0], -1, states.shape[-1])
        
        batch_size, max_time, state_dim = states.shape
        
        # Compute pairwise cosine similarities
        diversities = []
        
        for b in range(batch_size):
            batch_states = states[b]  # [max_time, state_dim]
            
            # Normalize states
            norms = jnp.linalg.norm(batch_states, axis=1, keepdims=True)
            normalized_states = batch_states / (norms + 1e-8)
            
            # Compute pairwise similarities
            similarities = jnp.dot(normalized_states, normalized_states.T)
            
            # Diversity = 1 - average off-diagonal similarity
            mask = 1 - jnp.eye(max_time)
            avg_similarity = jnp.sum(similarities * mask) / jnp.sum(mask)
            diversity = 1.0 - avg_similarity
            
            diversities.append(diversity)
        
        return jnp.mean(jnp.array(diversities))
    
    def _compute_state_efficiency(
        self,
        states: jnp.ndarray,
        cycles_used: jnp.ndarray,
    ) -> float:
        """Compute efficiency of state usage."""
        
        if states.ndim == 4:  # [batch, cycles, steps, state_dim]
            states = states.reshape(states.shape[0], -1, states.shape[-1])
        
        batch_size = states.shape[0]
        
        efficiencies = []
        
        for b in range(batch_size):
            used_cycles = int(cycles_used[b])
            if used_cycles == 0:
                continue
                
            # States used vs. total states
            used_states = states[b, :used_cycles]
            total_activation = jnp.sum(jnp.abs(used_states))
            
            # Efficiency = activation in used cycles / total possible activation
            max_possible_activation = used_cycles * states.shape[-1]
            efficiency = total_activation / (max_possible_activation + 1e-8)
            
            efficiencies.append(efficiency)
        
        return jnp.mean(jnp.array(efficiencies)) if efficiencies else 0.0
    
    def compute_gradient_approximation_quality(
        self,
        true_gradients: Dict[str, jnp.ndarray],
        approx_gradients: Dict[str, jnp.ndarray],
        deep_supervision_loss: jnp.ndarray,
        one_step_loss: jnp.ndarray,
    ) -> Dict[str, float]:
        """
        Compute gradient approximation quality metrics.
        
        Args:
            true_gradients: True gradients (if available)
            approx_gradients: 1-step approximated gradients
            deep_supervision_loss: Deep supervision loss values
            one_step_loss: One-step gradient loss values
            
        Returns:
            Dictionary of gradient quality metrics
        """
        
        if not self.track_gradients:
            return {
                "gradient_approximation_error": 0.0,
                "one_step_gradient_quality": 0.0,
                "deep_supervision_alignment": 0.0,
            }
        
        # Compute gradient approximation error
        grad_error = 0.0
        param_count = 0
        
        if true_gradients and approx_gradients:
            for param_name in true_gradients:
                if param_name in approx_gradients:
                    true_grad = true_gradients[param_name]
                    approx_grad = approx_gradients[param_name]
                    
                    # Compute relative error
                    error = jnp.linalg.norm(true_grad - approx_grad)
                    norm = jnp.linalg.norm(true_grad)
                    
                    if norm > 1e-8:
                        grad_error += error / norm
                        param_count += 1
            
            grad_error = grad_error / max(param_count, 1)
        
        # One-step gradient quality (lower loss = better quality)
        one_step_quality = 1.0 / (1.0 + jnp.mean(one_step_loss))
        
        # Deep supervision alignment (how well deep supervision aligns with final loss)
        deep_sup_alignment = 1.0 / (1.0 + jnp.mean(deep_supervision_loss))
        
        return {
            "gradient_approximation_error": float(grad_error),
            "one_step_gradient_quality": float(one_step_quality),
            "deep_supervision_alignment": float(deep_sup_alignment),
        }
    
    def compute_adaptive_computation_metrics(
        self,
        act_regularizer: jnp.ndarray,
        cycles_used: jnp.ndarray,
        early_stopping_flags: jnp.ndarray,
        task_completion_flags: jnp.ndarray,
    ) -> Dict[str, float]:
        """
        Compute adaptive computation metrics.
        
        Args:
            act_regularizer: ACT regularizer values
            cycles_used: Number of cycles used per example
            early_stopping_flags: Whether early stopping occurred
            task_completion_flags: Whether tasks were completed
            
        Returns:
            Dictionary of adaptive computation metrics
        """
        
        # ACT regularizer statistics
        avg_act_regularizer = jnp.mean(act_regularizer)
        
        # Early stopping rate
        early_stopping_rate = jnp.mean(early_stopping_flags)
        
        # Task completion rate
        task_completion_rate = jnp.mean(task_completion_flags)
        
        # Computation distribution
        cycle_counts = jnp.bincount(cycles_used.astype(int), length=self.max_cycles + 1)
        cycle_distribution = cycle_counts / jnp.sum(cycle_counts)
        
        # Adaptive efficiency (completing tasks with fewer cycles)
        adaptive_efficiency = jnp.mean(task_completion_flags / (cycles_used + 1))
        
        return {
            "act_regularizer_value": float(avg_act_regularizer),
            "early_stopping_rate": float(early_stopping_rate),
            "task_completion_rate": float(task_completion_rate),
            "adaptive_efficiency": float(adaptive_efficiency),
            "computation_distribution": {
                f"cycles_{i}": float(cycle_distribution[i])
                for i in range(len(cycle_distribution))
            },
        }
    
    def compute_reasoning_depth_metrics(
        self,
        attention_weights: Optional[jnp.ndarray],
        z_H_states: jnp.ndarray,
        cycles_used: jnp.ndarray,
        sequence_lengths: jnp.ndarray,
    ) -> Dict[str, float]:
        """
        Compute reasoning depth and long-context retention metrics.
        
        Args:
            attention_weights: Attention weights if available
            z_H_states: Planner states
            cycles_used: Number of cycles used
            sequence_lengths: Input sequence lengths
            
        Returns:
            Dictionary of reasoning depth metrics
        """
        
        batch_size = z_H_states.shape[0]
        
        # Reasoning depth: how much the planner state changes across cycles
        reasoning_depths = []
        
        for b in range(batch_size):
            used_cycles = int(cycles_used[b])
            if used_cycles <= 1:
                reasoning_depths.append(0.0)
                continue
            
            batch_states = z_H_states[b, :used_cycles]  # [used_cycles, state_dim]
            
            # Compute cumulative change in planner state
            state_changes = jnp.linalg.norm(
                batch_states[1:] - batch_states[:-1], axis=1
            )
            reasoning_depth = jnp.sum(state_changes)
            reasoning_depths.append(reasoning_depth)
        
        avg_reasoning_depth = jnp.mean(jnp.array(reasoning_depths))
        
        # Long-context retention: ability to maintain information across long sequences
        long_context_retention = 0.0
        
        if attention_weights is not None:
            # Measure attention to early tokens in long sequences
            for b in range(batch_size):
                seq_len = int(sequence_lengths[b])
                if seq_len > 1000:  # Only for long sequences
                    # Attention to first 10% of sequence
                    early_attention = jnp.mean(attention_weights[b, :, :seq_len//10])
                    long_context_retention += early_attention
            
            long_context_retention /= max(batch_size, 1)
        
        return {
            "reasoning_depth_achieved": float(avg_reasoning_depth),
            "long_context_retention": float(long_context_retention),
            "reasoning_consistency": float(jnp.std(jnp.array(reasoning_depths))),
        }
    
    def aggregate_hrm_metrics(
        self,
        computational_metrics: Dict[str, Any],
        state_metrics: Dict[str, float],
        gradient_metrics: Dict[str, float],
        adaptive_metrics: Dict[str, float],
        reasoning_metrics: Dict[str, float],
    ) -> HRMMetrics:
        """Aggregate all HRM metrics into a single container."""
        
        return HRMMetrics(
            # Computational efficiency
            avg_cycles_used=computational_metrics["avg_cycles_used"],
            avg_steps_per_cycle=computational_metrics["avg_steps_per_cycle"],
            computation_efficiency=computational_metrics["computation_efficiency"],
            
            # State utilization
            planner_state_utilization=state_metrics["planner_state_utilization"],
            executor_state_utilization=state_metrics["executor_state_utilization"],
            state_persistence=(state_metrics["planner_state_persistence"] + 
                             state_metrics["executor_state_persistence"]) / 2,
            
            # Gradient approximation quality
            gradient_approximation_error=gradient_metrics["gradient_approximation_error"],
            one_step_gradient_quality=gradient_metrics["one_step_gradient_quality"],
            deep_supervision_alignment=gradient_metrics["deep_supervision_alignment"],
            
            # Adaptive computation patterns
            act_regularizer_value=adaptive_metrics["act_regularizer_value"],
            computation_distribution=adaptive_metrics["computation_distribution"],
            early_stopping_rate=adaptive_metrics["early_stopping_rate"],
            
            # Task-specific performance
            task_completion_rate=adaptive_metrics["task_completion_rate"],
            reasoning_depth_achieved=reasoning_metrics["reasoning_depth_achieved"],
            long_context_retention=reasoning_metrics["long_context_retention"],
        )
    
    def log_hrm_metrics(self, metrics: HRMMetrics, step: int):
        """Log HRM metrics for monitoring."""
        
        logger.info(f"HRM Metrics at step {step}:")
        logger.info(f"  Computational Efficiency: {metrics.computation_efficiency:.4f}")
        logger.info(f"  Average Cycles Used: {metrics.avg_cycles_used:.2f}")
        logger.info(f"  State Utilization (P/E): {metrics.planner_state_utilization:.4f}/{metrics.executor_state_utilization:.4f}")
        logger.info(f"  Gradient Quality: {metrics.one_step_gradient_quality:.4f}")
        logger.info(f"  Task Completion Rate: {metrics.task_completion_rate:.4f}")
        logger.info(f"  Reasoning Depth: {metrics.reasoning_depth_achieved:.4f}")
        logger.info(f"  ACT Regularizer: {metrics.act_regularizer_value:.6f}")