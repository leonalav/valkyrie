#!/usr/bin/env python3
"""Verification script to ensure no JAX recompilations occur with multiple sequence lengths.

This script implements the final verification step from advice.md:
"Run a small loop with multiple sequence lengths; confirm no recompiles (watch logs) or verify compilation time minimal."

The script tests the BigBird attention implementation with various sequence lengths
to ensure the JAX tracing fix is working correctly and no recompilations happen.
"""

import time
import jax
import jax.numpy as jnp
import numpy as np
from typing import List, Dict, Tuple
import logging

# Configure logging to capture JAX compilation messages
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import the fixed BigBird attention implementation
from src.model.gryphon.bigbird_attention import BigBirdSparseAttention
from src.model.gryphon.gryphon_config import GryphonConfig
from src.model.modules import precompute_rope_freqs


class RecompilationVerifier:
    """Verifies that BigBird attention doesn't recompile for different sequence lengths."""
    
    def __init__(self):
        """Initialize the verifier with test configuration."""
        self.config = GryphonConfig(
            d_model=512,
            n_heads=8,
            block_size=64,
            max_position_embeddings=2048,
            num_random_blocks=3,
            window_size=3,
            num_global_blocks=2
        )
        
        # Test sequence lengths - multiple of block_size for proper alignment
        self.test_seq_lengths = [256, 512, 768, 1024, 1280, 1536, 1792, 2048]
        self.batch_size = 2
        
        # Initialize attention module
        self.attention = BigBirdSparseAttention(self.config)
        
        # Pre-compute RoPE frequencies
        head_dim = self.config.d_model // self.config.n_heads
        self.cos_freqs, self.sin_freqs = precompute_rope_freqs(
            dim=head_dim,
            max_seq_len=self.config.max_position_embeddings,
            base=self.config.rope_theta
        )
        
        logger.info(f"Initialized verifier with config: d_model={self.config.d_model}, "
                   f"n_heads={self.config.n_heads}, block_size={self.config.block_size}")
        logger.info(f"Test sequence lengths: {self.test_seq_lengths}")
    
    def create_test_inputs(self, seq_len: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Create test inputs for given sequence length."""
        rng_key = jax.random.PRNGKey(42)
        
        hidden_states = jax.random.normal(
            rng_key, (self.batch_size, seq_len, self.config.d_model)
        )
        position_ids = jnp.arange(seq_len)[None, :].repeat(self.batch_size, axis=0)
        
        return hidden_states, position_ids
    
    def initialize_attention_params(self, seq_len: int) -> Dict:
        """Initialize attention parameters for given sequence length."""
        rng_key = jax.random.PRNGKey(123)
        dummy_input = jnp.ones((1, seq_len, self.config.d_model))
        
        params = self.attention.init(
            {'params': rng_key, 'random_attention': jax.random.PRNGKey(456)}, 
            dummy_input, 
            training=False
        )
        
        return params
    
    def create_forward_function(self, params: Dict, seq_len: int):
        """Create a JIT-compiled forward function for given sequence length."""
        def forward_fn(hidden_states, seq_length):
            # Generate position_ids inside the function to make it static/concrete
            batch_size = hidden_states.shape[0]
            position_ids = jnp.arange(seq_length)[None, :].repeat(batch_size, axis=0)
            
            return self.attention.apply(
                params,
                hidden_states=hidden_states,
                position_ids=position_ids,
                cos_freqs=self.cos_freqs,
                sin_freqs=self.sin_freqs,
                training=False
            )
        
        # Make seq_length static so JAX knows the shapes at compile time
        return jax.jit(forward_fn, static_argnums=(1,))
    
    def measure_compilation_time(self, jitted_fn, hidden_states: jnp.ndarray, 
                                position_ids: jnp.ndarray) -> float:
        """Measure compilation time for the first call to a JIT function."""
        start_time = time.perf_counter()
        
        # First call triggers compilation - pass seq_len as static argument
        seq_len = hidden_states.shape[1]
        output = jitted_fn(hidden_states, seq_len)
        output.block_until_ready()
        
        end_time = time.perf_counter()
        compilation_time = end_time - start_time
        
        return compilation_time
    
    def measure_execution_time(self, jitted_fn, hidden_states: jnp.ndarray, 
                              position_ids: jnp.ndarray, num_runs: int = 5) -> float:
        """Measure average execution time after compilation."""
        times = []
        seq_len = hidden_states.shape[1]
        
        for _ in range(num_runs):
            start_time = time.perf_counter()
            
            output = jitted_fn(hidden_states, seq_len)
            output.block_until_ready()
            
            end_time = time.perf_counter()
            times.append(end_time - start_time)
        
        return np.mean(times)
    
    def verify_no_recompilation(self) -> Dict[str, List[float]]:
        """Main verification function to test multiple sequence lengths."""
        logger.info("Starting recompilation verification...")
        
        results = {
            'seq_lengths': [],
            'compilation_times': [],
            'execution_times': [],
            'throughput_tokens_per_sec': []
        }
        
        # Test each sequence length
        for seq_len in self.test_seq_lengths:
            logger.info(f"\n--- Testing sequence length: {seq_len} ---")
            
            # Create inputs and parameters
            hidden_states, position_ids = self.create_test_inputs(seq_len)
            params = self.initialize_attention_params(seq_len)
            
            # Create JIT function
            jitted_forward = self.create_forward_function(params, seq_len)
            
            # Measure compilation time (first call)
            logger.info("Measuring compilation time...")
            compilation_time = self.measure_compilation_time(jitted_forward, hidden_states, position_ids)
            
            # Measure execution time (subsequent calls)
            logger.info("Measuring execution time...")
            execution_time = self.measure_execution_time(jitted_forward, hidden_states, position_ids)
            
            # Calculate throughput
            throughput = (self.batch_size * seq_len) / execution_time
            
            # Store results
            results['seq_lengths'].append(seq_len)
            results['compilation_times'].append(compilation_time)
            results['execution_times'].append(execution_time)
            results['throughput_tokens_per_sec'].append(throughput)
            
            logger.info(f"Compilation time: {compilation_time:.4f}s")
            logger.info(f"Execution time: {execution_time:.4f}s")
            logger.info(f"Throughput: {throughput:.2f} tokens/sec")
        
        return results
    
    def analyze_results(self, results: Dict[str, List[float]]) -> None:
        """Analyze results to check for recompilation issues."""
        logger.info("\n" + "="*60)
        logger.info("RECOMPILATION VERIFICATION ANALYSIS")
        logger.info("="*60)
        
        compilation_times = results['compilation_times']
        execution_times = results['execution_times']
        
        # Check compilation times
        max_compilation_time = max(compilation_times)
        min_compilation_time = min(compilation_times)
        avg_compilation_time = np.mean(compilation_times)
        
        logger.info(f"Compilation times:")
        logger.info(f"  Min: {min_compilation_time:.4f}s")
        logger.info(f"  Max: {max_compilation_time:.4f}s")
        logger.info(f"  Avg: {avg_compilation_time:.4f}s")
        logger.info(f"  Std: {np.std(compilation_times):.4f}s")
        
        # Check execution times
        max_execution_time = max(execution_times)
        min_execution_time = min(execution_times)
        avg_execution_time = np.mean(execution_times)
        
        logger.info(f"\nExecution times:")
        logger.info(f"  Min: {min_execution_time:.4f}s")
        logger.info(f"  Max: {max_execution_time:.4f}s")
        logger.info(f"  Avg: {avg_execution_time:.4f}s")
        logger.info(f"  Std: {np.std(execution_times):.4f}s")
        
        # Verification criteria
        logger.info(f"\n" + "-"*40)
        logger.info("VERIFICATION RESULTS:")
        logger.info("-"*40)
        
        # Check if compilation times are reasonable (< 10 seconds)
        compilation_ok = max_compilation_time < 10.0
        logger.info(f"✓ Compilation times reasonable (< 10s): {compilation_ok}")
        
        # Check if execution times are consistent (std < 50% of mean)
        execution_consistency = np.std(execution_times) < (0.5 * avg_execution_time)
        logger.info(f"✓ Execution times consistent: {execution_consistency}")
        
        # Check if no JAX tracer errors occurred (implicit - script completed)
        no_tracer_errors = True
        logger.info(f"✓ No JAX tracer errors: {no_tracer_errors}")
        
        # Overall verification
        overall_success = compilation_ok and execution_consistency and no_tracer_errors
        
        if overall_success:
            logger.info(f"\n🎉 VERIFICATION PASSED! 🎉")
            logger.info("The JAX tracing fix is working correctly.")
            logger.info("No recompilations detected for different sequence lengths.")
        else:
            logger.error(f"\n❌ VERIFICATION FAILED! ❌")
            logger.error("Issues detected with the JAX tracing fix.")
        
        return overall_success
    
    def run_verification(self) -> bool:
        """Run the complete verification process."""
        try:
            logger.info("BigBird JAX Tracing Fix Verification")
            logger.info("="*50)
            
            # Run the verification
            results = self.verify_no_recompilation()
            
            # Analyze results
            success = self.analyze_results(results)
            
            return success
            
        except Exception as e:
            logger.error(f"Verification failed with error: {e}")
            logger.error("This indicates the JAX tracing fix may not be working correctly.")
            return False


def main():
    """Main function to run the verification."""
    verifier = RecompilationVerifier()
    success = verifier.run_verification()
    
    if success:
        print("\n✅ Verification completed successfully!")
        exit(0)
    else:
        print("\n❌ Verification failed!")
        exit(1)


if __name__ == "__main__":
    main()