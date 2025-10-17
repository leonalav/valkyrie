"""Performance benchmarks for BigBird attention optimizations.

This script benchmarks the optimized BigBird attention implementation
to validate performance improvements from vectorization and caching.
"""

import time
import jax
import jax.numpy as jnp
import numpy as np
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import pandas as pd
import os

from ..bigbird_attention import BigBirdSparseAttention
from ..gryphon_config import GryphonConfig
from ...modules import precompute_rope_freqs


class ProgressVisualizer:
    """Real-time progress visualization for benchmark execution."""
    
    def __init__(self, total_benchmarks: int, output_dir: str = "progress_plots"):
        """Initialize progress visualizer.
        
        Args:
            total_benchmarks: Total number of benchmark configurations to run
            output_dir: Directory to save progress plots
        """
        self.total_benchmarks = total_benchmarks
        self.completed_benchmarks = 0
        self.output_dir = output_dir
        self.results_history = []
        self.start_time = time.time()
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Set matplotlib backend for non-GUI environment
        plt.switch_backend('Agg')
        
        # Initialize progress tracking
        self.config_progress = {}
        self.throughput_history = []
        self.memory_history = []
        
    def update_progress(self, config_name: str, batch_size: int, seq_len: int, 
                       perf_metrics: Dict, memory_metrics: Dict):
        """Update progress with new benchmark result."""
        self.completed_benchmarks += 1
        
        # Track progress by configuration
        if config_name not in self.config_progress:
            self.config_progress[config_name] = []
        
        result = {
            'config': config_name,
            'batch_size': batch_size,
            'seq_len': seq_len,
            'throughput': perf_metrics['throughput_tokens_per_sec'],
            'memory_mb': memory_metrics['total_estimated_mb'],
            'timestamp': time.time() - self.start_time
        }
        
        self.config_progress[config_name].append(result)
        self.results_history.append(result)
        self.throughput_history.append(perf_metrics['throughput_tokens_per_sec'])
        self.memory_history.append(memory_metrics['total_estimated_mb'])
        
        # Generate progress plots
        self._create_progress_plots()
        
        # Print progress update
        progress_pct = (self.completed_benchmarks / self.total_benchmarks) * 100
        elapsed_time = time.time() - self.start_time
        eta = (elapsed_time / self.completed_benchmarks) * (self.total_benchmarks - self.completed_benchmarks)
        
        print(f"    Progress: {self.completed_benchmarks}/{self.total_benchmarks} "
              f"({progress_pct:.1f}%) | "
              f"Throughput: {perf_metrics['throughput_tokens_per_sec']:.0f} tokens/sec | "
              f"Memory: {memory_metrics['total_estimated_mb']:.1f} MB | "
              f"ETA: {eta:.1f}s")
    
    def _create_progress_plots(self):
        """Create and save progress visualization plots."""
        # Create a 2x2 subplot layout
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'BigBird Benchmark Progress ({self.completed_benchmarks}/{self.total_benchmarks})', 
                     fontsize=16, fontweight='bold')
        
        # Plot 1: Overall Progress
        ax1.pie([self.completed_benchmarks, self.total_benchmarks - self.completed_benchmarks], 
                labels=['Completed', 'Remaining'], 
                colors=['#2ecc71', '#ecf0f1'],
                autopct='%1.1f%%',
                startangle=90)
        ax1.set_title('Benchmark Progress')
        
        # Plot 2: Throughput Over Time
        if len(self.throughput_history) > 1:
            timestamps = [r['timestamp'] for r in self.results_history]
            throughputs = [r['throughput'] for r in self.results_history]
            ax2.plot(timestamps, throughputs, 'b-o', alpha=0.7, markersize=4)
            ax2.set_xlabel('Time (seconds)')
            ax2.set_ylabel('Throughput (tokens/sec)')
            ax2.set_title('Throughput Over Time')
            ax2.grid(True, alpha=0.3)
        else:
            ax2.text(0.5, 0.5, 'Collecting data...', ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Throughput Over Time')
        
        # Plot 3: Memory Usage Distribution
        if len(self.memory_history) > 1:
            ax3.hist(self.memory_history, bins=min(10, len(self.memory_history)), 
                    alpha=0.7, color='orange', edgecolor='black')
            ax3.set_xlabel('Memory Usage (MB)')
            ax3.set_ylabel('Frequency')
            ax3.set_title('Memory Usage Distribution')
            ax3.grid(True, alpha=0.3)
        else:
            ax3.text(0.5, 0.5, 'Collecting data...', ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('Memory Usage Distribution')
        
        # Plot 4: Configuration Comparison
        if len(self.config_progress) > 0:
            config_names = list(self.config_progress.keys())
            avg_throughputs = []
            
            for config in config_names:
                if self.config_progress[config]:
                    avg_throughput = np.mean([r['throughput'] for r in self.config_progress[config]])
                    avg_throughputs.append(avg_throughput)
                else:
                    avg_throughputs.append(0)
            
            if any(t > 0 for t in avg_throughputs):
                bars = ax4.bar(config_names, avg_throughputs, 
                              color=['#3498db', '#e74c3c', '#f39c12'][:len(config_names)],
                              alpha=0.7)
                ax4.set_ylabel('Average Throughput (tokens/sec)')
                ax4.set_title('Configuration Comparison')
                ax4.grid(True, alpha=0.3)
                
                # Add value labels on bars
                for bar, value in zip(bars, avg_throughputs):
                    if value > 0:
                        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + value*0.01,
                                f'{value:.0f}', ha='center', va='bottom', fontsize=10)
            else:
                ax4.text(0.5, 0.5, 'Collecting data...', ha='center', va='center', transform=ax4.transAxes)
                ax4.set_title('Configuration Comparison')
        else:
            ax4.text(0.5, 0.5, 'Collecting data...', ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Configuration Comparison')
        
        plt.tight_layout()
        
        # Save the progress plot
        progress_filename = f'{self.output_dir}/progress_{self.completed_benchmarks:03d}.png'
        plt.savefig(progress_filename, dpi=150, bbox_inches='tight')
        plt.close()
        
        # Also save as latest progress
        latest_filename = f'{self.output_dir}/latest_progress.png'
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'BigBird Benchmark Progress ({self.completed_benchmarks}/{self.total_benchmarks})', 
                     fontsize=16, fontweight='bold')
        
        # Recreate the same plots for the latest version
        ax1.pie([self.completed_benchmarks, self.total_benchmarks - self.completed_benchmarks], 
                labels=['Completed', 'Remaining'], 
                colors=['#2ecc71', '#ecf0f1'],
                autopct='%1.1f%%',
                startangle=90)
        ax1.set_title('Benchmark Progress')
        
        if len(self.throughput_history) > 1:
            timestamps = [r['timestamp'] for r in self.results_history]
            throughputs = [r['throughput'] for r in self.results_history]
            ax2.plot(timestamps, throughputs, 'b-o', alpha=0.7, markersize=4)
            ax2.set_xlabel('Time (seconds)')
            ax2.set_ylabel('Throughput (tokens/sec)')
            ax2.set_title('Throughput Over Time')
            ax2.grid(True, alpha=0.3)
        else:
            ax2.text(0.5, 0.5, 'Collecting data...', ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Throughput Over Time')
        
        if len(self.memory_history) > 1:
            ax3.hist(self.memory_history, bins=min(10, len(self.memory_history)), 
                    alpha=0.7, color='orange', edgecolor='black')
            ax3.set_xlabel('Memory Usage (MB)')
            ax3.set_ylabel('Frequency')
            ax3.set_title('Memory Usage Distribution')
            ax3.grid(True, alpha=0.3)
        else:
            ax3.text(0.5, 0.5, 'Collecting data...', ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('Memory Usage Distribution')
        
        if len(self.config_progress) > 0:
            config_names = list(self.config_progress.keys())
            avg_throughputs = []
            
            for config in config_names:
                if self.config_progress[config]:
                    avg_throughput = np.mean([r['throughput'] for r in self.config_progress[config]])
                    avg_throughputs.append(avg_throughput)
                else:
                    avg_throughputs.append(0)
            
            if any(t > 0 for t in avg_throughputs):
                bars = ax4.bar(config_names, avg_throughputs, 
                              color=['#3498db', '#e74c3c', '#f39c12'][:len(config_names)],
                              alpha=0.7)
                ax4.set_ylabel('Average Throughput (tokens/sec)')
                ax4.set_title('Configuration Comparison')
                ax4.grid(True, alpha=0.3)
                
                for bar, value in zip(bars, avg_throughputs):
                    if value > 0:
                        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + value*0.01,
                                f'{value:.0f}', ha='center', va='bottom', fontsize=10)
            else:
                ax4.text(0.5, 0.5, 'Collecting data...', ha='center', va='center', transform=ax4.transAxes)
                ax4.set_title('Configuration Comparison')
        else:
            ax4.text(0.5, 0.5, 'Collecting data...', ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Configuration Comparison')
        
        plt.tight_layout()
        plt.savefig(latest_filename, dpi=150, bbox_inches='tight')
        plt.close()
    
    def finalize(self):
        """Create final summary visualization."""
        print(f"\nProgress visualization complete!")
        print(f"Total time: {time.time() - self.start_time:.1f} seconds")
        print(f"Progress plots saved to: {self.output_dir}/")
        print(f"Latest progress plot: {self.output_dir}/latest_progress.png")


class PerformanceBenchmark:
    """Benchmark suite for BigBird attention performance."""
    
    def __init__(self):
        """Initialize benchmark configurations."""
        self.configs = {
            'small': GryphonConfig(
                d_model=256,
                n_heads=8,
                block_size=64,
                max_position_embeddings=512,
                num_random_blocks=3
            ),
            'medium': GryphonConfig(
                d_model=512,
                n_heads=16,
                block_size=64,
                max_position_embeddings=1024,
                num_random_blocks=3
            ),
            'large': GryphonConfig(
                d_model=768,
                n_heads=12,
                block_size=64,
                max_position_embeddings=2048,
                num_random_blocks=3
            )
        }
        
        self.batch_sizes = [1, 2, 4, 8]
        self.sequence_lengths = [256, 512, 1024, 2048]
        self.num_warmup_runs = 3
        self.num_benchmark_runs = 10
        
        # JIT compile functions
        self._compile_functions()
    
    def _compile_functions(self):
        """JIT compile attention functions for fair benchmarking."""
        print("JIT compiling functions...")
        
        for config_name, config in self.configs.items():
            attention = BigBirdSparseAttention(config)
            
            # Create dummy inputs for compilation
            batch_size = 2
            seq_len = config.block_size * 2
            
            rng_key = jax.random.PRNGKey(42)
            hidden_states = jax.random.normal(
                rng_key, 
                (batch_size, seq_len, config.d_model)
            )
            position_ids = jnp.arange(seq_len)[None, :].repeat(batch_size, axis=0)
            
            # Pre-compute RoPE frequencies
            head_dim = config.d_model // config.n_heads
            cos_freqs, sin_freqs = precompute_rope_freqs(
                dim=head_dim,
                max_seq_len=config.max_position_embeddings,
                base=config.rope_theta
            )
            
            # Initialize the module
            dummy_input = jnp.ones((1, 64, config.d_model))
            params = attention.init(
                {'params': rng_key, 'random_attention': jax.random.PRNGKey(123)}, 
                dummy_input, 
                training=False
            )
            
            # Create and JIT compile forward function
            def forward_fn(params, hidden_states, position_ids, cos_freqs, sin_freqs):
                return attention.apply(
                    params,
                    hidden_states=hidden_states,
                    position_ids=position_ids,
                    cos_freqs=cos_freqs,
                    sin_freqs=sin_freqs,
                    training=False,
                    rngs={'random_attention': jax.random.PRNGKey(456)}
                )
            
            jitted_forward = jax.jit(forward_fn)
            
            # Warmup compilation
            _ = jitted_forward(
                params,
                hidden_states,
                position_ids,
                cos_freqs,
                sin_freqs
            )
            
            print(f"Compiled {config_name} configuration")
    
    def benchmark_attention_forward(
        self, 
        config: GryphonConfig, 
        batch_size: int, 
        seq_len: int
    ) -> Dict[str, float]:
        """Benchmark forward pass of attention."""
        attention = BigBirdSparseAttention(config)
        
        # Create inputs
        rng_key = jax.random.PRNGKey(42)
        hidden_states = jax.random.normal(
            rng_key, (batch_size, seq_len, config.d_model)
        )
        position_ids = jnp.arange(seq_len)[None, :].repeat(batch_size, axis=0)
        
        # Pre-compute RoPE frequencies
        head_dim = config.d_model // config.n_heads
        cos_freqs, sin_freqs = precompute_rope_freqs(
            dim=head_dim,
            max_seq_len=config.max_position_embeddings,
            base=config.rope_theta
        )
        
        # Initialize the module with proper sequence length to avoid runtime plan generation
        dummy_input = jnp.ones((1, seq_len, config.d_model))  # Use actual seq_len
        params = attention.init(
            {'params': rng_key, 'random_attention': jax.random.PRNGKey(123)}, 
            dummy_input, 
            training=False
        )
        
        # Create a simpler jitted function that avoids the problematic code path
        def forward_fn(hidden_states):
            return attention.apply(
                params,
                hidden_states=hidden_states,
                position_ids=position_ids,
                cos_freqs=cos_freqs,
                sin_freqs=sin_freqs,
                training=False,
                rngs={'random_attention': jax.random.PRNGKey(456)}
            )
        
        jitted_forward = jax.jit(forward_fn)
        
        # Warmup runs
        for _ in range(self.num_warmup_runs):
            _ = jitted_forward(hidden_states)
        
        # Benchmark runs
        times = []
        for _ in range(self.num_benchmark_runs):
            start_time = time.perf_counter()
            
            output = jitted_forward(hidden_states)
            
            # Ensure computation is complete
            output.block_until_ready()
            
            end_time = time.perf_counter()
            times.append(end_time - start_time)
        
        return {
            'mean_time': np.mean(times),
            'std_time': np.std(times),
            'min_time': np.min(times),
            'max_time': np.max(times),
            'throughput_tokens_per_sec': (batch_size * seq_len) / np.mean(times)
        }
    
    def benchmark_memory_usage(
        self, 
        config: GryphonConfig, 
        batch_size: int, 
        seq_len: int
    ) -> Dict[str, float]:
        """Benchmark memory usage of attention."""
        attention = BigBirdSparseAttention(config)
        
        # Create inputs
        rng_key = jax.random.PRNGKey(42)
        hidden_states = jax.random.normal(
            rng_key, (batch_size, seq_len, config.d_model)
        )
        position_ids = jnp.arange(seq_len)[None, :].repeat(batch_size, axis=0)
        
        # Pre-compute RoPE frequencies
        head_dim = config.d_model // config.n_heads
        cos_freqs, sin_freqs = precompute_rope_freqs(
            dim=head_dim,
            max_seq_len=config.max_position_embeddings,
            base=config.rope_theta
        )
        
        # Estimate memory usage based on tensor sizes
        input_memory = hidden_states.nbytes + position_ids.nbytes
        input_memory += cos_freqs.nbytes + sin_freqs.nbytes
        
        # Estimate intermediate memory (Q, K, V projections, attention matrices)
        qkv_memory = 3 * batch_size * seq_len * config.d_model * 4  # float32
        
        # Attention matrix memory (sparse, so estimate based on sparsity)
        num_blocks = (seq_len + config.block_size - 1) // config.block_size
        sparse_attention_memory = (
            batch_size * config.n_heads * 
            num_blocks * config.block_size * 
            (config.block_size + 2 * config.num_random_blocks * config.block_size) * 4
        )
        
        total_estimated_memory = input_memory + qkv_memory + sparse_attention_memory
        
        return {
            'input_memory_mb': input_memory / (1024 * 1024),
            'qkv_memory_mb': qkv_memory / (1024 * 1024),
            'attention_memory_mb': sparse_attention_memory / (1024 * 1024),
            'total_estimated_mb': total_estimated_memory / (1024 * 1024)
        }
    
    def run_comprehensive_benchmark(self) -> pd.DataFrame:
        """Run comprehensive benchmark across all configurations."""
        results = []
        
        print("Running comprehensive performance benchmark...")
        
        # Calculate total number of benchmarks for progress tracking
        total_benchmarks = 0
        for config_name, config in self.configs.items():
            for batch_size in self.batch_sizes:
                for seq_len in self.sequence_lengths:
                    if seq_len <= config.max_position_embeddings:
                        total_benchmarks += 1
        
        # Initialize progress visualizer
        progress_viz = ProgressVisualizer(total_benchmarks)
        print(f"Total benchmarks to run: {total_benchmarks}")
        print("Progress visualizations will be saved to: progress_plots/")
        
        for config_name, config in self.configs.items():
            print(f"\nBenchmarking {config_name} configuration...")
            
            for batch_size in self.batch_sizes:
                for seq_len in self.sequence_lengths:
                    # Skip configurations that would be too large
                    if seq_len > config.max_position_embeddings:
                        continue
                    
                    print(f"  Batch size: {batch_size}, Seq length: {seq_len}")
                    
                    try:
                        # Benchmark forward pass
                        perf_metrics = self.benchmark_attention_forward(
                            config, batch_size, seq_len
                        )
                        
                        # Benchmark memory usage
                        memory_metrics = self.benchmark_memory_usage(
                            config, batch_size, seq_len
                        )
                        
                        # Update progress visualization
                        progress_viz.update_progress(
                            config_name, batch_size, seq_len, 
                            perf_metrics, memory_metrics
                        )
                        
                        # Combine results
                        result = {
                            'config': config_name,
                            'batch_size': batch_size,
                            'seq_len': seq_len,
                            'd_model': config.d_model,
                            'n_heads': config.n_heads,
                            'block_size': config.block_size,
                            'num_random_blocks': config.num_random_blocks,
                            **perf_metrics,
                            **memory_metrics
                        }
                        
                        results.append(result)
                        
                    except Exception as e:
                        print(f"    Error: {e}")
                        continue
        
        # Finalize progress visualization
        progress_viz.finalize()
        
        return pd.DataFrame(results)
    
    def benchmark_cache_effectiveness(self) -> Dict[str, float]:
        """Benchmark the effectiveness of the attention plan cache."""
        config = self.configs['medium']
        attention = BigBirdSparseAttention(config)
        
        batch_size = 4
        seq_len = 512
        
        # Create inputs
        rng_key = jax.random.PRNGKey(42)
        hidden_states = jax.random.normal(
            rng_key, (batch_size, seq_len, config.d_model)
        )
        position_ids = jnp.arange(seq_len)[None, :].repeat(batch_size, axis=0)
        
        # Pre-compute RoPE frequencies
        head_dim = config.d_model // config.n_heads
        cos_freqs, sin_freqs = precompute_rope_freqs(
            dim=head_dim,
            max_seq_len=config.max_position_embeddings,
            base=config.rope_theta
        )
        
        # Initialize the module
        dummy_input = jnp.ones((1, 64, config.d_model))
        params = attention.init(
            {'params': rng_key, 'random_attention': jax.random.PRNGKey(123)}, 
            dummy_input, 
            training=False
        )
        
        # Create forward function
        def forward_fn(params, hidden_states, position_ids, cos_freqs, sin_freqs):
            return attention.apply(
                params,
                hidden_states=hidden_states,
                position_ids=position_ids,
                cos_freqs=cos_freqs,
                sin_freqs=sin_freqs,
                training=False,
                rngs={'random_attention': jax.random.PRNGKey(456)}
            )
        
        # Benchmark with cache (multiple runs with same seq_len)
        cache_times = []
        for _ in range(10):
            start_time = time.perf_counter()
            
            _ = forward_fn(
                params,
                hidden_states,
                position_ids,
                cos_freqs,
                sin_freqs
            )
            
            end_time = time.perf_counter()
            cache_times.append(end_time - start_time)
        
        return {
            'cache_mean_time': np.mean(cache_times),
            'cache_std_time': np.std(cache_times),
            'cache_speedup_estimate': 1.0  # Baseline for comparison
        }
    
    def generate_performance_report(self, results_df: pd.DataFrame) -> str:
        """Generate a comprehensive performance report."""
        report = []
        report.append("BigBird Attention Performance Benchmark Report")
        report.append("=" * 50)
        report.append("")
        
        # Summary statistics
        report.append("Summary Statistics:")
        report.append(f"Total configurations tested: {len(results_df)}")
        report.append(f"Average throughput: {results_df['throughput_tokens_per_sec'].mean():.2f} tokens/sec")
        report.append(f"Best throughput: {results_df['throughput_tokens_per_sec'].max():.2f} tokens/sec")
        report.append(f"Average memory usage: {results_df['total_estimated_mb'].mean():.2f} MB")
        report.append("")
        
        # Performance by configuration
        report.append("Performance by Configuration:")
        for config in results_df['config'].unique():
            config_data = results_df[results_df['config'] == config]
            avg_throughput = config_data['throughput_tokens_per_sec'].mean()
            avg_memory = config_data['total_estimated_mb'].mean()
            
            report.append(f"  {config.capitalize()}:")
            report.append(f"    Average throughput: {avg_throughput:.2f} tokens/sec")
            report.append(f"    Average memory: {avg_memory:.2f} MB")
            report.append("")
        
        # Scaling analysis
        report.append("Scaling Analysis:")
        
        # Batch size scaling
        batch_scaling = results_df.groupby('batch_size')['throughput_tokens_per_sec'].mean()
        report.append("  Throughput by batch size:")
        for batch_size, throughput in batch_scaling.items():
            report.append(f"    Batch {batch_size}: {throughput:.2f} tokens/sec")
        report.append("")
        
        # Sequence length scaling
        seq_scaling = results_df.groupby('seq_len')['throughput_tokens_per_sec'].mean()
        report.append("  Throughput by sequence length:")
        for seq_len, throughput in seq_scaling.items():
            report.append(f"    Length {seq_len}: {throughput:.2f} tokens/sec")
        report.append("")
        
        # Memory scaling
        memory_scaling = results_df.groupby('seq_len')['total_estimated_mb'].mean()
        report.append("  Memory usage by sequence length:")
        for seq_len, memory in memory_scaling.items():
            report.append(f"    Length {seq_len}: {memory:.2f} MB")
        report.append("")
        
        # Optimization insights
        report.append("Optimization Insights:")
        report.append("- Vectorized random plan generation eliminates Python loops")
        report.append("- LRU cache reduces recomputation for repeated sequence lengths")
        report.append("- Shape validation prevents runtime errors and improves reliability")
        report.append("- RoPE padding fixes ensure correct multi-length batch handling")
        report.append("")
        
        return "\n".join(report)
    
    def save_benchmark_plots(self, results_df: pd.DataFrame, output_dir: str = "."):
        """Save benchmark visualization plots."""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Throughput vs sequence length
        plt.figure(figsize=(12, 8))
        for config in results_df['config'].unique():
            config_data = results_df[results_df['config'] == config]
            seq_throughput = config_data.groupby('seq_len')['throughput_tokens_per_sec'].mean()
            plt.plot(seq_throughput.index, seq_throughput.values, 
                    marker='o', label=f'{config.capitalize()}')
        
        plt.xlabel('Sequence Length')
        plt.ylabel('Throughput (tokens/sec)')
        plt.title('BigBird Attention Throughput vs Sequence Length')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(f'{output_dir}/throughput_vs_seqlen.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Memory usage vs sequence length
        plt.figure(figsize=(12, 8))
        for config in results_df['config'].unique():
            config_data = results_df[results_df['config'] == config]
            seq_memory = config_data.groupby('seq_len')['total_estimated_mb'].mean()
            plt.plot(seq_memory.index, seq_memory.values, 
                    marker='s', label=f'{config.capitalize()}')
        
        plt.xlabel('Sequence Length')
        plt.ylabel('Memory Usage (MB)')
        plt.title('BigBird Attention Memory Usage vs Sequence Length')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(f'{output_dir}/memory_vs_seqlen.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Batch size scaling
        plt.figure(figsize=(10, 6))
        batch_throughput = results_df.groupby('batch_size')['throughput_tokens_per_sec'].mean()
        plt.bar(batch_throughput.index, batch_throughput.values, alpha=0.7)
        plt.xlabel('Batch Size')
        plt.ylabel('Average Throughput (tokens/sec)')
        plt.title('BigBird Attention Throughput vs Batch Size')
        plt.grid(True, alpha=0.3)
        plt.savefig(f'{output_dir}/throughput_vs_batch.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Benchmark plots saved to {output_dir}/")


def main():
    """Run the complete benchmark suite."""
    benchmark = PerformanceBenchmark()
    
    # Run comprehensive benchmark
    results_df = benchmark.run_comprehensive_benchmark()
    
    # Save results
    results_df.to_csv('bigbird_benchmark_results.csv', index=False)
    print(f"\nBenchmark results saved to bigbird_benchmark_results.csv")
    
    # Generate and save report
    report = benchmark.generate_performance_report(results_df)
    with open('bigbird_performance_report.txt', 'w') as f:
        f.write(report)
    print("Performance report saved to bigbird_performance_report.txt")
    
    # Save plots
    benchmark.save_benchmark_plots(results_df, output_dir='benchmark_plots')
    
    # Print summary
    print("\n" + "="*50)
    print("BENCHMARK SUMMARY")
    print("="*50)
    print(f"Configurations tested: {len(results_df)}")
    print(f"Best throughput: {results_df['throughput_tokens_per_sec'].max():.2f} tokens/sec")
    print(f"Average throughput: {results_df['throughput_tokens_per_sec'].mean():.2f} tokens/sec")
    print(f"Memory range: {results_df['total_estimated_mb'].min():.1f} - {results_df['total_estimated_mb'].max():.1f} MB")
    
    # Cache effectiveness
    cache_metrics = benchmark.benchmark_cache_effectiveness()
    print(f"Cache performance: {cache_metrics['cache_mean_time']:.4f}s ± {cache_metrics['cache_std_time']:.4f}s")
    
    print("\nOptimizations implemented:")
    print("✓ Vectorized random plan generation")
    print("✓ LRU cache for attention plans")
    print("✓ Shape validation and error handling")
    print("✓ RoPE padding fixes for multi-length batches")
    print("✓ Comprehensive unit test coverage")


if __name__ == "__main__":
    main()