"""
Code task evaluation for HRM model.

Implements evaluation for code-related tasks including unit test execution,
code completion, and bug fixing as specified in the PLAN.
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union
from dataclasses import dataclass
import logging
import re
import ast
import subprocess
import tempfile
import os

logger = logging.getLogger(__name__)


@dataclass
class CodeTask:
    """Single code task instance."""
    name: str
    input_sequence: jnp.ndarray
    target_output: jnp.ndarray
    code_context: str
    test_cases: List[str]
    expected_result: Union[str, bool]  # Expected output or pass/fail
    max_cycles: int = 6
    max_steps: int = 8
    difficulty: str = "medium"


class CodeTaskEvaluator:
    """
    Evaluates HRM model on code-related tasks.
    
    Tasks include:
    - Unit test execution (pass/fail prediction)
    - Code completion
    - Bug detection and fixing
    - Function implementation from specification
    """
    
    def __init__(
        self,
        vocab_size: int = 32000,
        max_sequence_length: int = 2048,
        seed: int = 42,
    ):
        self.vocab_size = vocab_size
        self.max_sequence_length = max_sequence_length
        self.rng = jax.random.PRNGKey(seed)
        
        # Special tokens for code tasks
        self.special_tokens = {
            'CODE_START': vocab_size - 10,
            'CODE_END': vocab_size - 9,
            'TEST_START': vocab_size - 8,
            'TEST_END': vocab_size - 7,
            'PREDICT': vocab_size - 6,
            'COMPLETE': vocab_size - 5,
            'FIX': vocab_size - 4,
            'SEP': vocab_size - 3,
            'PASS': vocab_size - 2,
            'FAIL': vocab_size - 1,
        }
    
    def tokenize_code(self, code: str) -> List[int]:
        """Simple tokenization for code (in practice, use proper tokenizer)."""
        # This is a simplified tokenizer - in practice use SentencePiece/BPE
        tokens = []
        for char in code:
            tokens.append(min(ord(char), self.vocab_size - 20))
        return tokens
    
    def generate_unit_test_task(
        self,
        difficulty: str = "medium"
    ) -> CodeTask:
        """Generate a unit test execution task."""
        
        self.rng, task_rng = jax.random.split(self.rng)
        
        # Sample code snippets based on difficulty
        if difficulty == "easy":
            code_examples = [
                ("def add(a, b):\n    return a + b", "assert add(2, 3) == 5", True),
                ("def multiply(x, y):\n    return x * y", "assert multiply(4, 5) == 20", True),
                ("def is_even(n):\n    return n % 2 == 0", "assert is_even(4) == True", True),
            ]
        elif difficulty == "medium":
            code_examples = [
                ("def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n-1)", 
                 "assert factorial(5) == 120", True),
                ("def reverse_string(s):\n    return s[::-1]", 
                 "assert reverse_string('hello') == 'olleh'", True),
                ("def find_max(arr):\n    return max(arr)", 
                 "assert find_max([1, 5, 3, 9, 2]) == 9", True),
            ]
        else:  # hard
            code_examples = [
                ("def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[0]\n    less = [x for x in arr[1:] if x <= pivot]\n    greater = [x for x in arr[1:] if x > pivot]\n    return quicksort(less) + [pivot] + quicksort(greater)",
                 "assert quicksort([3, 1, 4, 1, 5, 9, 2, 6]) == [1, 1, 2, 3, 4, 5, 6, 9]", True),
                ("def binary_search(arr, target):\n    left, right = 0, len(arr) - 1\n    while left <= right:\n        mid = (left + right) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            left = mid + 1\n        else:\n            right = mid - 1\n    return -1",
                 "assert binary_search([1, 2, 3, 4, 5], 3) == 2", True),
            ]
        
        # Randomly select and potentially introduce bugs
        choice_idx = int(jax.random.randint(task_rng, (), 0, len(code_examples)))
        code, test, expected_pass = code_examples[choice_idx]
        
        # Randomly introduce bugs for some tasks
        self.rng, bug_rng = jax.random.split(self.rng)
        introduce_bug = jax.random.uniform(bug_rng) < 0.3  # 30% chance of bug
        
        if introduce_bug:
            # Simple bug introduction (change operators, etc.)
            if "+" in code:
                code = code.replace("+", "-", 1)
                expected_pass = False
            elif "==" in code:
                code = code.replace("==", "!=", 1)
                expected_pass = False
        
        # Create input sequence
        code_tokens = self.tokenize_code(code)
        test_tokens = self.tokenize_code(test)
        
        input_seq = jnp.array([
            self.special_tokens['CODE_START']
        ] + code_tokens + [
            self.special_tokens['CODE_END'],
            self.special_tokens['TEST_START']
        ] + test_tokens + [
            self.special_tokens['TEST_END'],
            self.special_tokens['PREDICT'],
            self.special_tokens['SEP']
        ])
        
        # Target: PASS or FAIL
        target_token = self.special_tokens['PASS'] if expected_pass else self.special_tokens['FAIL']
        target_output = jnp.array([target_token])
        
        return CodeTask(
            name=f"unit_test_{difficulty}_{choice_idx}",
            input_sequence=input_seq,
            target_output=target_output,
            code_context=code,
            test_cases=[test],
            expected_result=expected_pass,
            max_cycles=4 if difficulty == "easy" else 6,
            max_steps=6 if difficulty == "easy" else 8,
            difficulty=difficulty
        )
    
    def generate_code_completion_task(
        self,
        difficulty: str = "medium"
    ) -> CodeTask:
        """Generate a code completion task."""
        
        self.rng, task_rng = jax.random.split(self.rng)
        
        if difficulty == "easy":
            templates = [
                ("def add_numbers(a, b):\n    return ", "a + b"),
                ("def get_length(s):\n    return ", "len(s)"),
                ("def is_positive(x):\n    return ", "x > 0"),
            ]
        elif difficulty == "medium":
            templates = [
                ("def fibonacci(n):\n    if n <= 1:\n        return n\n    return ", "fibonacci(n-1) + fibonacci(n-2)"),
                ("def count_vowels(s):\n    vowels = 'aeiou'\n    count = 0\n    for char in s:\n        if ", "char.lower() in vowels:\n            count += 1\n    return count"),
                ("def binary_to_decimal(binary_str):\n    decimal = 0\n    for i, bit in enumerate(reversed(binary_str)):\n        decimal += ", "int(bit) * (2 ** i)\n    return decimal"),
            ]
        else:  # hard
            templates = [
                ("def merge_sort(arr):\n    if len(arr) <= 1:\n        return arr\n    mid = len(arr) // 2\n    left = merge_sort(arr[:mid])\n    right = merge_sort(arr[mid:])\n    return ", "merge(left, right)"),
                ("def dijkstra(graph, start):\n    distances = {node: float('inf') for node in graph}\n    distances[start] = 0\n    unvisited = set(graph.keys())\n    while unvisited:\n        current = ", "min(unvisited, key=lambda node: distances[node])"),
            ]
        
        # Select template
        choice_idx = int(jax.random.randint(task_rng, (), 0, len(templates)))
        incomplete_code, completion = templates[choice_idx]
        
        # Create input sequence
        code_tokens = self.tokenize_code(incomplete_code)
        completion_tokens = self.tokenize_code(completion)
        
        input_seq = jnp.array([
            self.special_tokens['CODE_START']
        ] + code_tokens + [
            self.special_tokens['COMPLETE'],
            self.special_tokens['SEP']
        ])
        
        target_output = jnp.array(completion_tokens)
        
        return CodeTask(
            name=f"code_completion_{difficulty}_{choice_idx}",
            input_sequence=input_seq,
            target_output=target_output,
            code_context=incomplete_code,
            test_cases=[],
            expected_result=completion,
            max_cycles=5 if difficulty == "easy" else 7,
            max_steps=6 if difficulty == "easy" else 10,
            difficulty=difficulty
        )
    
    def generate_bug_fixing_task(
        self,
        difficulty: str = "medium"
    ) -> CodeTask:
        """Generate a bug fixing task."""
        
        self.rng, task_rng = jax.random.split(self.rng)
        
        if difficulty == "easy":
            buggy_codes = [
                ("def add(a, b):\n    return a - b", "def add(a, b):\n    return a + b"),
                ("def is_even(n):\n    return n % 2 == 1", "def is_even(n):\n    return n % 2 == 0"),
            ]
        elif difficulty == "medium":
            buggy_codes = [
                ("def factorial(n):\n    if n <= 1:\n        return 0\n    return n * factorial(n-1)", 
                 "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n-1)"),
                ("def find_max(arr):\n    max_val = 0\n    for x in arr:\n        if x > max_val:\n            max_val = x\n    return max_val",
                 "def find_max(arr):\n    max_val = arr[0]\n    for x in arr:\n        if x > max_val:\n            max_val = x\n    return max_val"),
            ]
        else:  # hard
            buggy_codes = [
                ("def binary_search(arr, target):\n    left, right = 0, len(arr)\n    while left <= right:\n        mid = (left + right) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            left = mid + 1\n        else:\n            right = mid - 1\n    return -1",
                 "def binary_search(arr, target):\n    left, right = 0, len(arr) - 1\n    while left <= right:\n        mid = (left + right) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            left = mid + 1\n        else:\n            right = mid - 1\n    return -1"),
            ]
        
        # Select buggy code
        choice_idx = int(jax.random.randint(task_rng, (), 0, len(buggy_codes)))
        buggy_code, fixed_code = buggy_codes[choice_idx]
        
        # Create input sequence
        buggy_tokens = self.tokenize_code(buggy_code)
        fixed_tokens = self.tokenize_code(fixed_code)
        
        input_seq = jnp.array([
            self.special_tokens['CODE_START']
        ] + buggy_tokens + [
            self.special_tokens['CODE_END'],
            self.special_tokens['FIX'],
            self.special_tokens['SEP']
        ])
        
        target_output = jnp.array(fixed_tokens)
        
        return CodeTask(
            name=f"bug_fix_{difficulty}_{choice_idx}",
            input_sequence=input_seq,
            target_output=target_output,
            code_context=buggy_code,
            test_cases=[],
            expected_result=fixed_code,
            max_cycles=6 if difficulty == "easy" else 8,
            max_steps=8 if difficulty == "easy" else 12,
            difficulty=difficulty
        )
    
    def execute_code_test(self, code: str, test: str) -> bool:
        """Execute code and test to verify correctness."""
        try:
            # Create temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(code + '\n' + test)
                temp_file = f.name
            
            # Execute the code
            result = subprocess.run(
                ['python', temp_file],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            # Clean up
            os.unlink(temp_file)
            
            # Return True if no errors
            return result.returncode == 0
            
        except Exception as e:
            logger.warning(f"Error executing code test: {e}")
            return False
    
    def evaluate_model(
        self,
        model_fn,
        params: Dict[str, Any],
        tasks: List[CodeTask],
        batch_size: int = 4,
        execute_tests: bool = False,
    ) -> Dict[str, float]:
        """
        Evaluate model on code tasks.
        
        Args:
            model_fn: Model forward function
            params: Model parameters
            tasks: List of code tasks
            batch_size: Batch size for evaluation
            execute_tests: Whether to execute generated code for verification
            
        Returns:
            Dictionary of evaluation metrics
        """
        
        results = {
            "accuracy": 0.0,
            "exact_match": 0.0,
            "test_pass_rate": 0.0,
            "avg_cycles_used": 0.0,
            "avg_steps_used": 0.0,
            "code_execution_success": 0.0,
        }
        
        total_tasks = len(tasks)
        if total_tasks == 0:
            return results
        
        correct_predictions = 0
        exact_matches = 0
        test_passes = 0
        execution_successes = 0
        total_cycles = 0
        total_steps = 0
        
        # Process tasks in batches
        for i in range(0, total_tasks, batch_size):
            batch_tasks = tasks[i:i + batch_size]
            
            # Prepare batch inputs
            batch_inputs = []
            
            for task in batch_tasks:
                # Pad sequences to max length
                padded_input = jnp.zeros(self.max_sequence_length, dtype=jnp.int32)
                input_len = min(len(task.input_sequence), self.max_sequence_length)
                padded_input = padded_input.at[:input_len].set(task.input_sequence[:input_len])
                batch_inputs.append(padded_input)
            
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
                    
                    # Check accuracy
                    if len(pred) == len(target):
                        if task.name.startswith("unit_test"):
                            # For unit tests, check pass/fail prediction
                            if len(pred) > 0 and len(target) > 0:
                                correct = pred[0] == target[0]
                                correct_predictions += int(correct)
                                if correct:
                                    exact_matches += 1
                                    test_passes += 1
                        else:
                            # For code completion/fixing, check token accuracy
                            accuracy = jnp.mean(pred == target)
                            correct_predictions += accuracy
                            
                            if jnp.all(pred == target):
                                exact_matches += 1
                    
                    # Execute code if requested
                    if execute_tests and task.test_cases:
                        try:
                            # Convert prediction back to code (simplified)
                            pred_code = ''.join([chr(min(int(t), 127)) for t in pred if t > 0])
                            success = self.execute_code_test(task.code_context + pred_code, task.test_cases[0])
                            execution_successes += int(success)
                        except:
                            pass
                    
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
            results["test_pass_rate"] = test_passes / total_tasks
            results["avg_cycles_used"] = total_cycles / total_tasks
            results["avg_steps_used"] = total_steps / total_tasks
            
            if execute_tests:
                results["code_execution_success"] = execution_successes / total_tasks
        
        return results
    
    def create_evaluation_suite(
        self,
        num_tasks_per_type: int = 5,
        difficulties: List[str] = ["easy", "medium", "hard"]
    ) -> List[CodeTask]:
        """Create a comprehensive code evaluation suite."""
        
        tasks = []
        
        for difficulty in difficulties:
            # Unit test tasks
            for _ in range(num_tasks_per_type):
                tasks.append(self.generate_unit_test_task(difficulty))
            
            # Code completion tasks
            for _ in range(num_tasks_per_type):
                tasks.append(self.generate_code_completion_task(difficulty))
            
            # Bug fixing tasks
            for _ in range(num_tasks_per_type):
                tasks.append(self.generate_bug_fixing_task(difficulty))
        
        logger.info(f"Created code evaluation suite with {len(tasks)} tasks")
        return tasks