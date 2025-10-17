"""S5 Training Statistics Collection and Visualization System.

This module provides a professional alternative to debug prints, collecting
S5 training statistics and providing matplotlib-based visualization.
"""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import time


@dataclass
class S5StatsSnapshot:
    """Single snapshot of S5 training statistics."""
    step: int
    timestamp: float
    
    # Lambda (eigenvalue) statistics
    lambda_real_min: float
    lambda_real_max: float
    lambda_imag_min: float
    lambda_imag_max: float
    
    # Delta (timescale) statistics
    delta_min: float
    delta_max: float
    delta_mean: float
    
    # Spectral radius and stability
    spectral_radius: float
    stability_correction_applied: bool
    scale_factor: float
    
    # B parameter statistics (optional)
    b_magnitude_max: Optional[float] = None
    b_magnitude_mean: Optional[float] = None
    
    # C parameter statistics (optional)
    c_magnitude_max: Optional[float] = None
    c_magnitude_mean: Optional[float] = None


class S5StatsCollector:
    """Professional statistics collection system for S5 training."""
    
    def __init__(self, max_snapshots: int = 10000):
        """Initialize the stats collector.
        
        Args:
            max_snapshots: Maximum number of snapshots to keep in memory
        """
        self.max_snapshots = max_snapshots
        self.snapshots: List[S5StatsSnapshot] = []
        self.step_counter = 0
        
        # Aggregated statistics for efficient querying
        self.stats_history = defaultdict(list)
        
    def collect_discretization_stats(self, 
                                   Lambda: jnp.ndarray, 
                                   Delta: jnp.ndarray, 
                                   Lambda_bar: jnp.ndarray,
                                   scale_factor: float,
                                   B_tilde: Optional[jnp.ndarray] = None,
                                   C: Optional[jnp.ndarray] = None) -> S5StatsSnapshot:
        """Collect statistics from S5 discretization step.
        
        Args:
            Lambda: Complex eigenvalues [N]
            Delta: Timescale parameters [N]
            Lambda_bar: Discretized eigenvalues [N]
            scale_factor: Stability correction scale factor
            B_tilde: Optional B parameters [N, d_model]
            C: Optional C parameters [d_model, N]
            
        Returns:
            S5StatsSnapshot with collected statistics
        """
        # Convert JAX arrays to numpy for host computation
        lambda_np = np.array(Lambda)
        delta_np = np.array(Delta)
        lambda_bar_np = np.array(Lambda_bar)
        
        # Compute statistics
        spectral_radius = float(np.max(np.abs(lambda_bar_np)))
        stability_correction_applied = abs(scale_factor - 1.0) > 1e-6
        
        # Optional B statistics
        b_magnitude_max = None
        b_magnitude_mean = None
        if B_tilde is not None:
            b_magnitudes = np.abs(np.array(B_tilde))
            b_magnitude_max = float(np.max(b_magnitudes))
            b_magnitude_mean = float(np.mean(b_magnitudes))
        
        # Optional C statistics
        c_magnitude_max = None
        c_magnitude_mean = None
        if C is not None:
            c_magnitudes = np.abs(np.array(C))
            c_magnitude_max = float(np.max(c_magnitudes))
            c_magnitude_mean = float(np.mean(c_magnitudes))
        
        # Create snapshot
        snapshot = S5StatsSnapshot(
            step=self.step_counter,
            timestamp=time.time(),
            lambda_real_min=float(np.min(lambda_np.real)),
            lambda_real_max=float(np.max(lambda_np.real)),
            lambda_imag_min=float(np.min(lambda_np.imag)),
            lambda_imag_max=float(np.max(lambda_np.imag)),
            delta_min=float(np.min(delta_np)),
            delta_max=float(np.max(delta_np)),
            delta_mean=float(np.mean(delta_np)),
            spectral_radius=spectral_radius,
            stability_correction_applied=stability_correction_applied,
            scale_factor=float(scale_factor),
            b_magnitude_max=b_magnitude_max,
            b_magnitude_mean=b_magnitude_mean,
            c_magnitude_max=c_magnitude_max,
            c_magnitude_mean=c_magnitude_mean
        )
        
        # Store snapshot
        self.snapshots.append(snapshot)
        
        # Maintain max_snapshots limit
        if len(self.snapshots) > self.max_snapshots:
            self.snapshots.pop(0)
        
        # Update aggregated history
        self._update_history(snapshot)
        
        self.step_counter += 1
        return snapshot
    
    def _update_history(self, snapshot: S5StatsSnapshot):
        """Update aggregated statistics history."""
        self.stats_history['steps'].append(snapshot.step)
        self.stats_history['spectral_radius'].append(snapshot.spectral_radius)
        self.stats_history['delta_min'].append(snapshot.delta_min)
        self.stats_history['delta_max'].append(snapshot.delta_max)
        self.stats_history['lambda_real_min'].append(snapshot.lambda_real_min)
        self.stats_history['lambda_real_max'].append(snapshot.lambda_real_max)
        self.stats_history['scale_factor'].append(snapshot.scale_factor)
        
        if snapshot.b_magnitude_max is not None:
            self.stats_history['b_magnitude_max'].append(snapshot.b_magnitude_max)
        if snapshot.c_magnitude_max is not None:
            self.stats_history['c_magnitude_max'].append(snapshot.c_magnitude_max)
    
    def get_recent_stats(self, n_steps: int = 100) -> List[S5StatsSnapshot]:
        """Get the most recent n_steps statistics."""
        return self.snapshots[-n_steps:]
    
    def get_stability_summary(self) -> Dict:
        """Get summary of stability-related statistics."""
        if not self.snapshots:
            return {}
        
        recent_snapshots = self.get_recent_stats(100)
        
        spectral_radii = [s.spectral_radius for s in recent_snapshots]
        corrections_applied = sum(1 for s in recent_snapshots if s.stability_correction_applied)
        
        return {
            'recent_spectral_radius_mean': np.mean(spectral_radii),
            'recent_spectral_radius_max': np.max(spectral_radii),
            'recent_spectral_radius_std': np.std(spectral_radii),
            'stability_corrections_pct': corrections_applied / len(recent_snapshots) * 100,
            'total_snapshots': len(self.snapshots),
            'current_step': self.step_counter
        }
    
    def plot_training_stats(self, save_path: Optional[str] = None, show: bool = True):
        """Create comprehensive matplotlib visualization of S5 training statistics.
        
        Args:
            save_path: Optional path to save the plot
            show: Whether to display the plot
        """
        if not self.snapshots:
            print("No statistics collected yet.")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('S5 Training Statistics', fontsize=16, fontweight='bold')
        
        steps = self.stats_history['steps']
        
        # 1. Spectral Radius Over Time
        ax = axes[0, 0]
        ax.plot(steps, self.stats_history['spectral_radius'], 'b-', linewidth=2, alpha=0.8)
        ax.axhline(y=1.0, color='r', linestyle='--', alpha=0.7, label='Stability Threshold')
        ax.set_title('Spectral Radius', fontweight='bold')
        ax.set_xlabel('Training Step')
        ax.set_ylabel('Spectral Radius')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # 2. Delta (Timescale) Range
        ax = axes[0, 1]
        ax.fill_between(steps, self.stats_history['delta_min'], self.stats_history['delta_max'], 
                       alpha=0.3, color='green', label='Delta Range')
        ax.plot(steps, self.stats_history['delta_min'], 'g-', linewidth=1, label='Delta Min')
        ax.plot(steps, self.stats_history['delta_max'], 'g-', linewidth=2, label='Delta Max')
        ax.set_title('Delta (Timescale) Range', fontweight='bold')
        ax.set_xlabel('Training Step')
        ax.set_ylabel('Delta Value')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # 3. Lambda Real Part Range
        ax = axes[0, 2]
        ax.fill_between(steps, self.stats_history['lambda_real_min'], self.stats_history['lambda_real_max'], 
                       alpha=0.3, color='purple', label='Lambda Real Range')
        ax.plot(steps, self.stats_history['lambda_real_min'], 'purple', linewidth=1, label='Real Min')
        ax.plot(steps, self.stats_history['lambda_real_max'], 'purple', linewidth=2, label='Real Max')
        ax.axhline(y=0, color='r', linestyle='--', alpha=0.7, label='Stability Boundary')
        ax.set_title('Lambda Real Parts', fontweight='bold')
        ax.set_xlabel('Training Step')
        ax.set_ylabel('Real(Lambda)')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # 4. Stability Correction Scale Factors
        ax = axes[1, 0]
        scale_factors = np.array(self.stats_history['scale_factor'])
        corrections_mask = scale_factors != 1.0
        
        ax.scatter(np.array(steps)[corrections_mask], scale_factors[corrections_mask], 
                  c='red', alpha=0.6, s=20, label='Corrections Applied')
        ax.axhline(y=1.0, color='black', linestyle='-', alpha=0.5, label='No Correction')
        ax.set_title('Stability Corrections', fontweight='bold')
        ax.set_xlabel('Training Step')
        ax.set_ylabel('Scale Factor')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # 5. B Parameter Magnitudes (if available)
        ax = axes[1, 1]
        if 'b_magnitude_max' in self.stats_history and self.stats_history['b_magnitude_max']:
            ax.plot(steps, self.stats_history['b_magnitude_max'], 'orange', linewidth=2, label='B Max Magnitude')
            ax.set_title('B Parameter Magnitudes', fontweight='bold')
            ax.set_xlabel('Training Step')
            ax.set_ylabel('Max |B|')
            ax.grid(True, alpha=0.3)
            ax.legend()
        else:
            ax.text(0.5, 0.5, 'B Parameter Stats\nNot Available', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title('B Parameter Magnitudes', fontweight='bold')
        
        # 6. C Parameter Magnitudes (if available)
        ax = axes[1, 2]
        if 'c_magnitude_max' in self.stats_history and self.stats_history['c_magnitude_max']:
            ax.plot(steps, self.stats_history['c_magnitude_max'], 'cyan', linewidth=2, label='C Max Magnitude')
            ax.set_title('C Parameter Magnitudes', fontweight='bold')
            ax.set_xlabel('Training Step')
            ax.set_ylabel('Max |C|')
            ax.grid(True, alpha=0.3)
            ax.legend()
        else:
            ax.text(0.5, 0.5, 'C Parameter Stats\nNot Available', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title('C Parameter Magnitudes', fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"S5 training statistics saved to: {save_path}")
        
        if show:
            plt.show()
    
    def export_stats_csv(self, filepath: str):
        """Export statistics to CSV for external analysis."""
        import csv
        
        if not self.snapshots:
            print("No statistics to export.")
            return
        
        with open(filepath, 'w', newline='') as csvfile:
            fieldnames = [
                'step', 'timestamp', 'lambda_real_min', 'lambda_real_max',
                'lambda_imag_min', 'lambda_imag_max', 'delta_min', 'delta_max',
                'delta_mean', 'spectral_radius', 'stability_correction_applied',
                'scale_factor', 'b_magnitude_max', 'b_magnitude_mean',
                'c_magnitude_max', 'c_magnitude_mean'
            ]
            
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for snapshot in self.snapshots:
                writer.writerow({
                    'step': snapshot.step,
                    'timestamp': snapshot.timestamp,
                    'lambda_real_min': snapshot.lambda_real_min,
                    'lambda_real_max': snapshot.lambda_real_max,
                    'lambda_imag_min': snapshot.lambda_imag_min,
                    'lambda_imag_max': snapshot.lambda_imag_max,
                    'delta_min': snapshot.delta_min,
                    'delta_max': snapshot.delta_max,
                    'delta_mean': snapshot.delta_mean,
                    'spectral_radius': snapshot.spectral_radius,
                    'stability_correction_applied': snapshot.stability_correction_applied,
                    'scale_factor': snapshot.scale_factor,
                    'b_magnitude_max': snapshot.b_magnitude_max,
                    'b_magnitude_mean': snapshot.b_magnitude_mean,
                    'c_magnitude_max': snapshot.c_magnitude_max,
                    'c_magnitude_mean': snapshot.c_magnitude_mean
                })
        
        print(f"S5 statistics exported to: {filepath}")


# Global stats collector instance
_global_stats_collector = None

def get_stats_collector() -> S5StatsCollector:
    """Get the global stats collector instance."""
    global _global_stats_collector
    if _global_stats_collector is None:
        _global_stats_collector = S5StatsCollector()
    return _global_stats_collector

def reset_stats_collector():
    """Reset the global stats collector."""
    global _global_stats_collector
    _global_stats_collector = S5StatsCollector()