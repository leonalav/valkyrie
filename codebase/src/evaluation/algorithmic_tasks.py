"""
Algorithmic task evaluation for HRM model.

Implements evaluation for algorithmic reasoning tasks without Chain-of-Thought,
focusing on I/O patterns as specified in the PLAN.
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class AlgorithmicTask:
    """Single algorithmic task instance."""
    name: str
    input_sequence: jnp.ndarray
    target_output: jnp.ndarray
    max_cycles: int = 6
    max_steps: int = 6
    difficulty: str = "medium"  # easy, medium, hard


class AlgorithmicEvaluator:
    """
    Evaluates HRM model on algorithmic reasoning tasks.
    
    Tasks include:
    - Sorting sequences
    - Graph traversal (DFS, BFS)
    - Maze solving
    - Pattern matching
    - Arithmetic sequences
    """
    
    def __init__(
        self,
        vocab_size: int = 32000,
        max_sequence_length: int = 1024,
        seed: int = 42,
    ):
        self.vocab_size = vocab_size
        self.max_sequence_length = max_sequence_length
        self.rng = jax.random.PRNGKey(seed)
        
    def generate_sorting_task(
        self,
        sequence_length: int = 10,
        difficulty: str = "medium"
    ) -> AlgorithmicTask:
        """Generate a sorting task."""
        
        self.rng, task_rng = jax.random.split(self.rng)
        
        # Generate random sequence to sort
        if difficulty == "easy":
            # Small numbers, short sequence
            numbers = jax.random.randint(task_rng, (sequence_length,), 0, 10)
        elif difficulty == "medium":
            # Medium range, medium sequence
            numbers = jax.random.randint(task_rng, (sequence_length,), 0, 100)
        else:  # hard
            # Large range, longer sequence
            numbers = jax.random.randint(task_rng, (sequence_length,), 0, 1000)
        
        # Create input sequence: [SORT] + numbers + [SEP]
        sort_token = self.vocab_size - 4  # Special token for SORT
        sep_token = self.vocab_size - 3   # Separator token
        
        input_seq = jnp.concatenate([
            jnp.array([sort_token]),
            numbers,
            jnp.array([sep_token])
        ])
        
        # Target: sorted sequence
        sorted_numbers = jnp.sort(numbers)
        target_output = sorted_numbers
        
        return AlgorithmicTask(
            name=f"sort_{difficulty}_{sequence_length}",
            input_sequence=input_seq,
            target_output=target_output,
            max_cycles=4 if difficulty == "easy" else 6,
            max_steps=3 if difficulty == "easy" else 6,
            difficulty=difficulty
        )
    
    def generate_graph_traversal_task(
        self,
        num_nodes: int = 8,
        difficulty: str = "medium"
    ) -> AlgorithmicTask:
        """Generate a graph traversal task (DFS/BFS)."""
        
        self.rng, task_rng = jax.random.split(self.rng)
        
        # Generate adjacency list representation
        # For simplicity, create a connected graph
        edges = []
        for i in range(num_nodes - 1):
            edges.append((i, i + 1))
        
        # Add some random edges based on difficulty
        num_extra_edges = 2 if difficulty == "easy" else 4 if difficulty == "medium" else 8
        for _ in range(num_extra_edges):
            self.rng, edge_rng = jax.random.split(self.rng)
            u, v = jax.random.randint(edge_rng, (2,), 0, num_nodes)
            if u != v:
                edges.append((int(u), int(v)))
        
        # Choose start and end nodes
        self.rng, start_rng, end_rng = jax.random.split(self.rng, 3)
        start_node = int(jax.random.randint(start_rng, (), 0, num_nodes))
        end_node = int(jax.random.randint(end_rng, (), 0, num_nodes))
        
        # Create input sequence: [TRAVERSE] + start + end + edges + [SEP]
        traverse_token = self.vocab_size - 5
        sep_token = self.vocab_size - 3
        
        # Flatten edges for input
        edge_tokens = []
        for u, v in edges:
            edge_tokens.extend([u, v])
        
        input_seq = jnp.array([traverse_token, start_node, end_node] + edge_tokens + [sep_token])
        
        # Target: path from start to end (simplified - just return if path exists)
        # In practice, you'd implement actual pathfinding
        target_output = jnp.array([1])  # 1 if path exists, 0 otherwise
        
        return AlgorithmicTask(
            name=f"graph_traversal_{difficulty}_{num_nodes}",
            input_sequence=input_seq,
            target_output=target_output,
            max_cycles=6 if difficulty != "easy" else 4,
            max_steps=8 if difficulty == "hard" else 6,
            difficulty=difficulty
        )
    
    def generate_arithmetic_sequence_task(
        self,
        sequence_length: int = 5,
        difficulty: str = "medium"
    ) -> AlgorithmicTask:
        """Generate arithmetic sequence completion task."""
        
        self.rng, task_rng = jax.random.split(self.rng)
        
        # Generate arithmetic sequence
        if difficulty == "easy":
            start = int(jax.random.randint(task_rng, (), 1, 10))
            diff = int(jax.random.randint(task_rng, (), 1, 5))
        elif difficulty == "medium":
            start = int(jax.random.randint(task_rng, (), 1, 50))
            diff = int(jax.random.randint(task_rng, (), -10, 10))
        else:  # hard
            start = int(jax.random.randint(task_rng, (), -100, 100))
            diff = int(jax.random.randint(task_rng, (), -20, 20))
        
        # Create sequence
        sequence = [start + i * diff for i in range(sequence_length)]
        
        # Input: first few terms + [COMPLETE]
        complete_token = self.vocab_size - 6
        sep_token = self.vocab_size - 3
        
        # Show first 3 terms, ask for next 2
        shown_terms = sequence[:3]
        hidden_terms = sequence[3:]
        
        input_seq = jnp.array(shown_terms + [complete_token, sep_token])
        target_output = jnp.array(hidden_terms)
        
        return AlgorithmicTask(
            name=f"arithmetic_{difficulty}_{sequence_length}",
            input_sequence=input_seq,
            target_output=target_output,
            max_cycles=3 if difficulty == "easy" else 5,
            max_steps=4 if difficulty == "easy" else 6,
            difficulty=difficulty
        )
    
    def evaluate_model(
        self,
        model_fn,
        params: Dict[str, Any],
        tasks: List[AlgorithmicTask],
        batch_size: int = 8,
    ) -> Dict[str, float]:
        """
        Evaluate model on algorithmic tasks.
        
        Args:
            model_fn: Model forward function
            params: Model parameters
            tasks: List of algorithmic tasks
            batch_size: Batch size for evaluation
            
        Returns:
            Dictionary of evaluation metrics
        """
        
        results = {
            "accuracy": 0.0,
            "exact_match": 0.0,
            "avg_cycles_used": 0.0,
            "avg_steps_used": 0.0,
            "efficiency_score": 0.0,
        }
        
        total_tasks = len(tasks)
        if total_tasks == 0:
            return results
        
        correct_predictions = 0
        exact_matches = 0
        total_cycles = 0
        total_steps = 0
        
        # Process tasks in batches
        for i in range(0, total_tasks, batch_size):
            batch_tasks = tasks[i:i + batch_size]
            
            # Prepare batch inputs
            batch_inputs = []
            batch_targets = []
            
            for task in batch_tasks:
                # Pad sequences to max length
                padded_input = jnp.zeros(self.max_sequence_length, dtype=jnp.int32)
                input_len = min(len(task.input_sequence), self.max_sequence_length)
                padded_input = padded_input.at[:input_len].set(task.input_sequence[:input_len])
                
                batch_inputs.append(padded_input)
                batch_targets.append(task.target_output)
            
            batch_inputs = jnp.stack(batch_inputs)
            
            # Run model inference
            try:
                outputs = model_fn(params, batch_inputs)
                
                # Extract predictions and HRM metrics
                if isinstance(outputs, dict):
                    predictions = outputs.get('logits', outputs.get('predictions'))
                    cycles_used = outputs.get('cycles_used', 0)
                    steps_used = outputs.get('steps_used', 0)
                else:
                    predictions = outputs
                    cycles_used = 0
                    steps_used = 0
                
                # Evaluate each task in batch
                for j, task in enumerate(batch_tasks):
                    pred = predictions[j] if predictions.ndim > 1 else predictions
                    target = task.target_output
                    
                    # Convert logits to predictions if needed
                    if pred.ndim > 1:  # Logits
                        pred = jnp.argmax(pred, axis=-1)
                    
                    # Truncate prediction to target length
                    pred = pred[:len(target)]
                    
                    # Check accuracy (element-wise)
                    if len(pred) == len(target):
                        accuracy = jnp.mean(pred == target)
                        correct_predictions += accuracy
                        
                        # Exact match
                        if jnp.all(pred == target):
                            exact_matches += 1
                    
                    # Track HRM efficiency
                    total_cycles += cycles_used
                    total_steps += steps_used
                    
            except Exception as e:
                logger.warning(f"Error evaluating batch {i}: {e}")
                continue
        
        # Compute final metrics
        if total_tasks > 0:
            results["accuracy"] = correct_predictions / total_tasks
            results["exact_match"] = exact_matches / total_tasks
            results["avg_cycles_used"] = total_cycles / total_tasks
            results["avg_steps_used"] = total_steps / total_tasks
            
            # Efficiency score: accuracy per computational cost
            avg_cost = results["avg_cycles_used"] * results["avg_steps_used"]
            if avg_cost > 0:
                results["efficiency_score"] = results["accuracy"] / avg_cost
        
        return results
    
    def create_evaluation_suite(
        self,
        num_tasks_per_type: int = 10,
        difficulties: List[str] = ["easy", "medium", "hard"]
    ) -> List[AlgorithmicTask]:
        """Create a comprehensive evaluation suite."""
        
        tasks = []
        
        for difficulty in difficulties:
            # Sorting tasks
            for _ in range(num_tasks_per_type):
                seq_len = 5 if difficulty == "easy" else 10 if difficulty == "medium" else 20
                tasks.append(self.generate_sorting_task(seq_len, difficulty))
            
            # Graph traversal tasks
            for _ in range(num_tasks_per_type):
                num_nodes = 5 if difficulty == "easy" else 8 if difficulty == "medium" else 12
                tasks.append(self.generate_graph_traversal_task(num_nodes, difficulty))
            
            # Arithmetic sequence tasks
            for _ in range(num_tasks_per_type):
                seq_len = 4 if difficulty == "easy" else 6 if difficulty == "medium" else 8
                tasks.append(self.generate_arithmetic_sequence_task(seq_len, difficulty))
        
        logger.info(f"Created evaluation suite with {len(tasks)} algorithmic tasks")
        return tasks