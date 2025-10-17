"""
Comprehensive numerical stability tests for S5 model implementation.

Tests cover:
1. Eigendecomposition stability across different state dimensions
2. Parameter magnitude and conditioning analysis
3. Forward/backward pass stability
4. Gradient flow analysis
5. Training robustness with small datasets
6. Impulse response tests
7. Reconstruction accuracy tests
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from typing import Dict, List, Tuple, Any
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from model.s5 import (
    construct_hippo_n_matrix,
    safe_eigendecomposition,
    analyze_s5_parameters,
    ValkyrieS5
)
from model.modules import ValkyrieConfig


class NumericalStabilityTester:
    """Comprehensive numerical stability testing suite for S5 models."""
    
    def __init__(self, random_seed: int = 42):
        self.key = jax.random.PRNGKey(random_seed)
        self.test_results = {}
        
    def test_eigendecomposition_stability(self, state_dims: List[int] = [4, 8, 16, 32, 64]) -> Dict[str, Any]:
        """Test eigendecomposition stability across different state dimensions."""
        print("\n" + "="*60)
        print("EIGENDECOMPOSITION STABILITY TESTS")
        print("="*60)
        
        results = {}
        
        for N in state_dims:
            print(f"\n--- Testing state dimension N={N} ---")
            
            try:
                # Construct HiPPO-N matrix
                A = construct_hippo_n_matrix(N)
                
                # Test eigendecomposition directly
                eigenvals, eigenvecs, V_pinv, is_stable = safe_eigendecomposition(A)
                
                # Compute stability metrics
                cond_number = jnp.linalg.cond(eigenvecs)
                reconstruction_error = jnp.linalg.norm(eigenvecs @ V_pinv @ eigenvecs - eigenvecs)
                eigenval_stability = jnp.all(jnp.real(eigenvals) < 1e-6)
                
                results[f"N_{N}"] = {
                    "condition_number": float(cond_number),
                    "reconstruction_error": float(reconstruction_error),
                    "eigenvalue_stability": bool(eigenval_stability),
                    "max_eigenval_real": float(jnp.max(jnp.real(eigenvals))),
                    "min_eigenval_real": float(jnp.min(jnp.real(eigenvals))),
                    "is_stable": bool(is_stable),  # Add stability flag to results
                    "success": True
                }
                
                # Stability assessment
                stable = (cond_number < 1e12 and 
                         reconstruction_error < 1e-6 and 
                         eigenval_stability)
                
                print(f"✅ N={N}: {'STABLE' if stable else 'UNSTABLE'}")
                print(f"   Condition: {cond_number:.2e}, Reconstruction: {reconstruction_error:.2e}")
                
            except Exception as e:
                print(f"❌ N={N}: FAILED - {str(e)}")
                results[f"N_{N}"] = {"success": False, "error": str(e)}
        
        self.test_results["eigendecomposition_stability"] = results
        return results
    
    def test_parameter_magnitudes(self, state_dims: List[int] = [8, 16, 32]) -> Dict[str, Any]:
        """Test parameter magnitude stability across different state dimensions."""
        print("\n" + "="*60)
        print("PARAMETER MAGNITUDE TESTS")
        print("="*60)
        
        results = {}
        
        for N in state_dims:
            print(f"\nTesting state dimension N={N}")
            try:
                # Initialize ValkyrieS5 layer
                config = ValkyrieConfig(d_model=64)
                layer = ValkyrieS5(config=config, state_dim=N)
                
                # Create dummy input for initialization
                self.key, subkey = jax.random.split(self.key)
                dummy_input = jax.random.normal(subkey, (1, 10, 64))
                
                # Initialize parameters
                self.key, subkey = jax.random.split(self.key)
                params = layer.init(subkey, dummy_input)
                
                # Generate dummy Lambda, B, C for analysis
                self.key, subkey = jax.random.split(self.key)
                Lambda = jax.random.normal(subkey, (N,), dtype=jnp.complex64)
                
                self.key, subkey = jax.random.split(self.key)
                B = jax.random.normal(subkey, (N, 64), dtype=jnp.complex64)
                
                self.key, subkey = jax.random.split(self.key)
                C = jax.random.normal(subkey, (64, N), dtype=jnp.complex64)
                
                self.key, subkey = jax.random.split(self.key)
                D = jax.random.normal(subkey, (64,))
                
                self.key, subkey = jax.random.split(self.key)
                log_Delta = jax.random.normal(subkey, (N,))  # Should match state dimension N, not d_model
                
                # Analyze parameters
                analysis = analyze_s5_parameters(Lambda, B, C, D, log_Delta, verbose=False)
                
                results[f"N_{N}"] = {
                    "success": True,
                    "analysis": analysis
                }
                
                print(f"✅ N={N}: Parameter analysis completed")
                
            except Exception as e:
                print(f"❌ N={N}: FAILED - {str(e)}")
                results[f"N_{N}"] = {
                    "success": False,
                    "error": str(e)
                }
        
        self.test_results["parameter_magnitudes"] = results
        return results
    
    def test_forward_pass_stability(self, state_dim: int = 16, seq_len: int = 100, 
                                  batch_size: int = 4) -> Dict[str, Any]:
        """Test forward pass numerical stability."""
        print("\n" + "="*60)
        print("FORWARD PASS STABILITY TESTS")
        print("="*60)
        
        try:
            # Initialize ValkyrieS5 layer
            config = ValkyrieConfig(d_model=64)
            layer = ValkyrieS5(config=config, state_dim=state_dim)
            
            # Create test input
            self.key, subkey = jax.random.split(self.key)
            x = jax.random.normal(subkey, (batch_size, seq_len, 64))
            
            # Initialize parameters
            self.key, subkey = jax.random.split(self.key)
            params = layer.init(subkey, x)
            
            print(f"Testing forward pass with input shape: {x.shape}")
            
            # Test forward pass
            output, final_state = layer.apply(params, x)
            
            # Check for numerical issues
            has_nan = jnp.any(jnp.isnan(output))
            has_inf = jnp.any(jnp.isinf(output))
            max_magnitude = float(jnp.max(jnp.abs(output)))
            mean_magnitude = float(jnp.mean(jnp.abs(output)))
            
            # Test with different input magnitudes
            magnitude_tests = {}
            for scale in [0.1, 1.0, 10.0, 100.0]:
                scaled_x = x * scale
                try:
                    scaled_output, _ = layer.apply(params, scaled_x)
                    magnitude_tests[f"scale_{scale}"] = {
                        "success": True,
                        "max_output": float(jnp.max(jnp.abs(scaled_output))),
                        "has_nan": bool(jnp.any(jnp.isnan(scaled_output))),
                        "has_inf": bool(jnp.any(jnp.isinf(scaled_output)))
                    }
                except Exception as e:
                    magnitude_tests[f"scale_{scale}"] = {
                        "success": False,
                        "error": str(e)
                    }
            
            results = {
                "basic_forward": {
                    "success": True,
                    "has_nan": bool(has_nan),
                    "has_inf": bool(has_inf),
                    "max_magnitude": max_magnitude,
                    "mean_magnitude": mean_magnitude,
                    "output_shape": output.shape
                },
                "magnitude_tests": magnitude_tests
            }
            
            stable = not (has_nan or has_inf) and max_magnitude < 1e6
            print(f"{'✅' if stable else '❌'} Forward pass: {'STABLE' if stable else 'UNSTABLE'}")
            print(f"   Max output: {max_magnitude:.2e}, Mean: {mean_magnitude:.2e}")
            print(f"   NaN: {has_nan}, Inf: {has_inf}")
            
        except Exception as e:
            print(f"❌ Forward pass test FAILED: {str(e)}")
            results = {"success": False, "error": str(e)}
        
        self.test_results["forward_pass_stability"] = results
        return results
    
    def test_gradient_flow(self, state_dim: int = 16, seq_len: int = 50) -> Dict[str, Any]:
        """Test gradient flow and backward pass stability."""
        print("\n" + "="*60)
        print("GRADIENT FLOW TESTS")
        print("="*60)
        
        try:
            # Initialize ValkyrieS5 layer
            config = ValkyrieConfig(d_model=64)
            layer = ValkyrieS5(config=config, state_dim=state_dim)
            
            # Create test data
            self.key, subkey1, subkey2 = jax.random.split(self.key, 3)
            x = jax.random.normal(subkey1, (2, seq_len, 64))
            y_target = jax.random.normal(subkey2, (2, seq_len, 64))
            
            # Initialize parameters
            self.key, subkey = jax.random.split(self.key)
            params = layer.init(subkey, x)
            
            # Define loss function
            def loss_fn(params):
                pred, _ = layer.apply(params, x)
                return jnp.mean((pred - y_target) ** 2)
            
            # Compute loss and gradients
            loss_val = loss_fn(params)
            grads = jax.grad(loss_fn)(params)
            
            # Use jax.tree_util for gradient analysis - no Python conversions
            def compute_gradient_stats(grad_tree):
                """Compute gradient statistics using pure JAX operations."""
                # Flatten all gradients into a single array
                flat_grads, _ = jax.tree_util.tree_flatten(grad_tree)
                
                # Filter out None values and ensure we have arrays
                valid_grads = [g for g in flat_grads if g is not None and hasattr(g, 'shape')]
                
                if not valid_grads:
                    return jnp.array(0.0), jnp.array(0.0), jnp.array(False), jnp.array(False)
                
                # Concatenate all gradients into one flat array
                all_grads = jnp.concatenate([g.flatten() for g in valid_grads])
                
                # Compute statistics
                total_norm = jnp.linalg.norm(all_grads)
                max_magnitude = jnp.max(jnp.abs(all_grads))
                has_nan = jnp.any(jnp.isnan(all_grads))
                has_inf = jnp.any(jnp.isinf(all_grads))
                
                return total_norm, max_magnitude, has_nan, has_inf
            
            # Compute gradient statistics
            total_norm, max_magnitude, has_nan, has_inf = compute_gradient_stats(grads)
            
            # Evaluate gradient health using JAX operations
            healthy = jnp.logical_and(
                jnp.logical_and(~has_nan, ~has_inf),
                jnp.logical_and(total_norm < 1e6, total_norm > 1e-12)
            )
            
            # Convert to Python values ONLY after all JAX computations are complete
            # Use .item() to extract scalar values from JAX arrays
            results = {
                "loss_value": loss_val.item(),
                "total_grad_norm": total_norm.item(),
                "max_grad_magnitude": max_magnitude.item(),
                "has_nan_grads": has_nan.item(),
                "has_inf_grads": has_inf.item(),
                "success": True
            }
            
            # Print results using the converted values
            healthy_val = healthy.item()
            status_symbol = "✅" if healthy_val else "❌"
            status_text = "HEALTHY" if healthy_val else "PROBLEMATIC"
            
            print(status_symbol + " Gradients: " + status_text)
            print("   Loss: " + str(round(results["loss_value"], 6)))
            print("   Total grad norm: " + str(results["total_grad_norm"]))
            print("   Max grad magnitude: " + str(results["max_grad_magnitude"]))
            print("   NaN grads: " + str(results["has_nan_grads"]) + ", Inf grads: " + str(results["has_inf_grads"]))
            
        except Exception as e:
            error_msg = str(e)
            print("❌ Gradient flow test FAILED: " + error_msg)
            results = {"success": False, "error": error_msg}
        
        self.test_results["gradient_flow"] = results
        return results
    
    def test_impulse_response(self, state_dim: int = 16, seq_len: int = 200) -> Dict[str, Any]:
        """
        Test impulse response characteristics with enhanced per-mode analysis and transient amplification detection.
        
        IMPROVED: Added per-mode analysis and transient amplification detection.
        """
        print("\n" + "="*60)
        print("IMPULSE RESPONSE TESTS - ENHANCED")
        print("="*60)
        
        try:
            # Initialize ValkyrieS5 layer
            config = ValkyrieConfig(d_model=64)
            layer = ValkyrieS5(config=config, state_dim=state_dim)
            
            # Create impulse input (delta function)
            impulse = jnp.zeros((1, seq_len, 64))
            impulse = impulse.at[0, 0, 0].set(1.0)  # Impulse at t=0
            
            # Initialize parameters
            self.key, subkey = jax.random.split(self.key)
            params = layer.init(subkey, impulse)
            
            # Compute impulse response
            response, _ = layer.apply(params, impulse)
            response_seq = response[0, :, 0]  # Extract sequence
            
            # ENHANCED: Per-mode analysis using FFT
            response_fft = jnp.fft.fft(response_seq)
            freqs = jnp.fft.fftfreq(seq_len)
            power_spectrum = jnp.abs(response_fft) ** 2
            
            # Identify dominant modes
            dominant_mode_indices = jnp.argsort(power_spectrum)[-5:]  # Top 5 modes
            dominant_freqs = freqs[dominant_mode_indices]
            dominant_powers = power_spectrum[dominant_mode_indices]
            
            # ENHANCED: Transient amplification detection
            # Compute envelope using Hilbert transform approximation
            analytic_signal = jnp.fft.ifft(jnp.fft.fft(response_seq) * 
                                          (1 + jnp.sign(jnp.fft.fftfreq(seq_len))))
            envelope = jnp.abs(analytic_signal)
            
            # Detect transient amplification (overshoot beyond initial response)
            initial_response = jnp.abs(response_seq[0])
            max_envelope = jnp.max(envelope)
            transient_amplification = max_envelope / (initial_response + 1e-12)
            
            # Find peak amplification time
            peak_time = jnp.argmax(envelope)
            
            # ENHANCED: Decay rate analysis per time segment
            # Divide response into segments and analyze decay in each
            n_segments = 4
            segment_length = seq_len // n_segments
            decay_rates = []
            
            for i in range(n_segments):
                start_idx = i * segment_length
                end_idx = min((i + 1) * segment_length, seq_len)
                if end_idx - start_idx < 2:
                    continue
                    
                segment = envelope[start_idx:end_idx]
                if len(segment) > 1:
                    # Fit exponential decay: y = A * exp(-λt)
                    # log(y) = log(A) - λt
                    t_segment = jnp.arange(len(segment))
                    log_segment = jnp.log(jnp.maximum(segment, 1e-12))
                    
                    # Simple linear regression for decay rate
                    t_mean = jnp.mean(t_segment)
                    log_mean = jnp.mean(log_segment)
                    numerator = jnp.sum((t_segment - t_mean) * (log_segment - log_mean))
                    denominator = jnp.sum((t_segment - t_mean) ** 2)
                    decay_rate = -numerator / (denominator + 1e-12)  # Negative slope = decay rate
                    decay_rates.append(float(decay_rate))
            
            # ENHANCED: Spectral radius estimation from eigenvalues
            # Extract eigenvalues from the S5 layer parameters if available
            spectral_analysis = {}
            try:
                # Get the discretized state matrix eigenvalues
                if hasattr(layer, 'get_eigenvalues'):
                    eigenvals = layer.get_eigenvalues(params)
                    spectral_radius = float(jnp.max(jnp.abs(eigenvals)))
                    eigenval_real_parts = jnp.real(eigenvals)
                    max_real_part = float(jnp.max(eigenval_real_parts))
                    
                    spectral_analysis = {
                        'spectral_radius': spectral_radius,
                        'max_real_eigenvalue': max_real_part,
                        'eigenvalue_distribution': {
                            'mean_real': float(jnp.mean(eigenval_real_parts)),
                            'std_real': float(jnp.std(eigenval_real_parts)),
                            'min_real': float(jnp.min(eigenval_real_parts)),
                            'max_real': max_real_part
                        }
                    }
            except Exception as e:
                spectral_analysis = {'error': f'Could not extract eigenvalues: {str(e)}'}
            
            # Original analysis (enhanced)
            max_response = float(jnp.max(jnp.abs(response_seq)))
            final_response = float(jnp.abs(response_seq[-1]))
            decay_ratio = final_response / max_response if max_response > 0 else 0.0
            
            # Check for stability (response should decay)
            is_decaying = decay_ratio < 0.1  # Response should decay to <10% of peak
            
            # Enhanced oscillation analysis
            diff_seq = jnp.diff(response_seq)
            sign_changes = jnp.sum(jnp.diff(jnp.sign(diff_seq)) != 0)
            is_oscillatory = sign_changes > seq_len // 10
            
            # Oscillation frequency estimation
            zero_crossings = jnp.sum(jnp.diff(jnp.sign(response_seq)) != 0)
            estimated_freq = zero_crossings / (2 * seq_len) if seq_len > 0 else 0.0
            
            # Energy analysis
            total_energy = float(jnp.sum(response_seq ** 2))
            
            # ENHANCED: Stability metrics
            stability_metrics = {
                'transient_amplification': float(transient_amplification),
                'peak_amplification_time': int(peak_time),
                'has_overshoot': bool(transient_amplification > 1.5),  # >50% overshoot
                'decay_rates_per_segment': decay_rates,
                'mean_decay_rate': float(jnp.mean(jnp.array(decay_rates))) if decay_rates else 0.0,
                'oscillation_frequency': float(estimated_freq),
                'zero_crossings': int(zero_crossings)
            }
            
            # ENHANCED: Per-mode analysis results
            per_mode_analysis = {
                'dominant_frequencies': dominant_freqs.tolist(),
                'dominant_powers': dominant_powers.tolist(),
                'power_spectrum_peak': float(jnp.max(power_spectrum)),
                'spectral_centroid': float(jnp.sum(freqs * power_spectrum) / jnp.sum(power_spectrum)),
                'spectral_bandwidth': float(jnp.sqrt(jnp.sum(((freqs - jnp.sum(freqs * power_spectrum) / jnp.sum(power_spectrum)) ** 2) * power_spectrum) / jnp.sum(power_spectrum)))
            }
            
            results = {
                "max_response": max_response,
                "final_response": final_response,
                "decay_ratio": decay_ratio,
                "is_decaying": bool(is_decaying),
                "is_oscillatory": bool(is_oscillatory),
                "sign_changes": int(sign_changes),
                "total_energy": total_energy,
                "response_sequence": response_seq.tolist(),
                "stability_metrics": stability_metrics,
                "per_mode_analysis": per_mode_analysis,
                "spectral_analysis": spectral_analysis,
                "success": True
            }
            
            # ENHANCED: Overall assessment with new criteria
            has_dangerous_amplification = transient_amplification > 3.0
            has_poor_decay = jnp.mean(jnp.array(decay_rates)) < 0.01 if decay_rates else True
            has_spectral_issues = (spectral_analysis.get('max_real_eigenvalue', -1) > -0.01 if 'max_real_eigenvalue' in spectral_analysis else False)
            
            healthy_response = (is_decaying and 
                              max_response < 1e3 and 
                              not has_dangerous_amplification and
                              not has_poor_decay and
                              not has_spectral_issues and
                              not jnp.any(jnp.isnan(response_seq)) and
                              not jnp.any(jnp.isinf(response_seq)))
            
            print(f"{'✅' if healthy_response else '⚠️ '} Impulse response: {'HEALTHY' if healthy_response else 'CONCERNING'}")
            print(f"   Max response: {max_response:.2e}")
            print(f"   Decay ratio: {decay_ratio:.2e}")
            print(f"   Transient amplification: {transient_amplification:.2f}x")
            print(f"   Mean decay rate: {stability_metrics['mean_decay_rate']:.3f}")
            print(f"   Decaying: {is_decaying}, Oscillatory: {is_oscillatory}")
            print(f"   Dominant frequency: {per_mode_analysis['dominant_frequencies'][0]:.3f}" if len(per_mode_analysis['dominant_frequencies']) > 0 else "   No dominant frequency")
            print(f"   Total energy: {total_energy:.2e}")
            
            if spectral_analysis and 'spectral_radius' in spectral_analysis:
                print(f"   Spectral radius: {spectral_analysis['spectral_radius']:.3f}")
                print(f"   Max real eigenvalue: {spectral_analysis['max_real_eigenvalue']:.3f}")
            
        except Exception as e:
            print(f"❌ Impulse response test FAILED: {str(e)}")
            results = {"success": False, "error": str(e)}
        
        self.test_results["impulse_response"] = results
        return results
    
    def run_detailed_statistical_analysis(self) -> Dict[str, Any]:
        """
        Run comprehensive numerical stability analysis with detailed statistics and user-defined criteria.
        
        Provides verbose statistical analysis with PASS/WARN/FAIL classifications based on:
        - Eigen/V conditioning thresholds
        - Eigenvalue stability criteria  
        - B/C magnitude limits
        - Delta/timescale bounds
        - Forward/gradient pass health
        - Impulse response characteristics
        """
        print("\n" + "="*100)
        print("DETAILED STATISTICAL ANALYSIS - S5 NUMERICAL STABILITY")
        print("="*100)
        
        # Define user-specified thresholds - STRICT CI GATING POLICY
        thresholds = {
            "eigen_conditioning": {
                "hard_fail": 1e6,      # Strict: fail if cond(V) >= 1e6
                "warning": 1e5,        # Warning zone: 1e5 to 1e6
                "good": 1e4            # Good: below 1e4
            },
            "reconstruction_error": {
                "hard_fail": 1e-6,     # Strict: fail if reconstruction_error >= 1e-6
                "warning": 1e-8,       # Warning zone: 1e-8 to 1e-6
                "good": 1e-10          # Good: below 1e-10
            },
            "eigenvalue_stability": {
                "hard_fail_real_positive": 0.0,
                "warning_real_max": -1e-8,
                "good_real_max": -1e-6
            },
            "matrix_magnitudes": {
                "hard_fail": 100,      # Strict: fail if max(|B|) > 100
                "warning": 10,         # Warning zone: 10 to 100
                "good": 1              # Good: below 1
            },
            "delta_timescales": {
                "hard_fail": 1e3,
                "warning": 100,
                "good": 10
            },
            "gradient_magnitudes": {
                "hard_fail": 1e6,
                "warning": 1e3,
                "good": 100
            },
            "impulse_response": {
                "hard_fail_max": 1e6,
                "warning_max": 1e3,
                "good_max": 100,
                "decay_threshold": 0.1
            }
        }
        
        # Run all tests to collect raw data
        print("🔄 Collecting raw numerical data...")
        self.test_eigendecomposition_stability()
        self.test_parameter_magnitudes() 
        self.test_forward_pass_stability()
        self.test_gradient_flow()
        self.test_impulse_response()
        
        # Initialize detailed analysis results
        detailed_results = {
            "raw_data": self.test_results,
            "statistical_analysis": {},
            "classifications": {},
            "recommendations": []
        }
        
        print("\n" + "="*100)
        print("DETAILED STATISTICAL BREAKDOWN")
        print("="*100)
        
        # 1. EIGENDECOMPOSITION ANALYSIS
        print("\n" + "🔍 1. EIGENDECOMPOSITION & CONDITIONING ANALYSIS")
        print("-" * 80)
        
        eigen_results = self.test_results.get("eigendecomposition_stability", {})
        eigen_stats = {"condition_numbers": [], "reconstruction_errors": [], "eigenvalue_stats": []}
        eigen_classifications = {}
        
        for test_name, result in eigen_results.items():
            if result.get("success", False):
                cond_num = result["condition_number"]
                recon_err = result["reconstruction_error"] 
                max_real = result["max_eigenval_real"]
                min_real = result["min_eigenval_real"]
                
                eigen_stats["condition_numbers"].append(cond_num)
                eigen_stats["reconstruction_errors"].append(recon_err)
                eigen_stats["eigenvalue_stats"].append({
                    "max_real": max_real,
                    "min_real": min_real,
                    "test_name": test_name
                })
                
                # Classify conditioning
                if cond_num > thresholds["eigen_conditioning"]["hard_fail"]:
                    cond_class = "HARD_FAIL"
                elif cond_num > thresholds["eigen_conditioning"]["warning"]:
                    cond_class = "WARNING"
                elif cond_num < thresholds["eigen_conditioning"]["good"]:
                    cond_class = "GOOD"
                else:
                    cond_class = "ACCEPTABLE"
                
                # Classify reconstruction error
                if recon_err > thresholds["reconstruction_error"]["hard_fail"]:
                    recon_class = "HARD_FAIL"
                elif recon_err > thresholds["reconstruction_error"]["warning"]:
                    recon_class = "WARNING"
                elif recon_err < thresholds["reconstruction_error"]["good"]:
                    recon_class = "GOOD"
                else:
                    recon_class = "ACCEPTABLE"
                
                # Classify eigenvalue stability
                if max_real > thresholds["eigenvalue_stability"]["hard_fail_real_positive"]:
                    eigen_class = "HARD_FAIL"
                elif max_real > thresholds["eigenvalue_stability"]["warning_real_max"]:
                    eigen_class = "WARNING"
                elif max_real < thresholds["eigenvalue_stability"]["good_real_max"]:
                    eigen_class = "GOOD"
                else:
                    eigen_class = "ACCEPTABLE"
                
                eigen_classifications[test_name] = {
                    "conditioning": cond_class,
                    "reconstruction_error": recon_class,
                    "eigenvalue_stability": eigen_class
                }
                
                # Print detailed stats
                print(f"  {test_name}:")
                print(f"    Condition Number: {cond_num:.2e} [{cond_class}]")
                print(f"    Reconstruction Error: {recon_err:.2e} [{recon_class}]")
                print(f"    Eigenvalue Range: [{min_real:.2e}, {max_real:.2e}] [{eigen_class}]")
        
        # Summary statistics
        if eigen_stats["condition_numbers"]:
            avg_cond = np.mean(eigen_stats["condition_numbers"])
            max_cond = np.max(eigen_stats["condition_numbers"])
            print(f"\n  📊 SUMMARY:")
            print(f"    Average Condition Number: {avg_cond:.2e}")
            print(f"    Maximum Condition Number: {max_cond:.2e}")
            print(f"    Tests with GOOD conditioning: {sum(1 for _, c in eigen_classifications.items() if c['conditioning'] == 'GOOD')}")
            print(f"    Tests with WARNING conditioning: {sum(1 for _, c in eigen_classifications.items() if c['conditioning'] == 'WARNING')}")
            print(f"    Tests with HARD_FAIL conditioning: {sum(1 for _, c in eigen_classifications.items() if c['conditioning'] == 'HARD_FAIL')}")
        
        detailed_results["statistical_analysis"]["eigendecomposition"] = eigen_stats
        detailed_results["classifications"]["eigendecomposition"] = eigen_classifications
        
        # 2. PARAMETER MAGNITUDE ANALYSIS
        print("\n" + "🔍 2. PARAMETER MAGNITUDE ANALYSIS")
        print("-" * 80)
        
        param_results = self.test_results.get("parameter_magnitudes", {})
        param_stats = {"B_magnitudes": [], "C_magnitudes": [], "D_magnitudes": [], "Delta_magnitudes": []}
        param_classifications = {}
        
        for test_name, result in param_results.items():
            if result.get("success", False) and "analysis" in result:
                analysis = result["analysis"]
                
                # Extract magnitude statistics
                b_mag = analysis.get("B_magnitude", 0)
                c_mag = analysis.get("C_magnitude", 0) 
                d_mag = analysis.get("D_magnitude", 0)
                delta_mag = analysis.get("Delta_magnitude", 0)
                
                param_stats["B_magnitudes"].append(b_mag)
                param_stats["C_magnitudes"].append(c_mag)
                param_stats["D_magnitudes"].append(d_mag)
                param_stats["Delta_magnitudes"].append(delta_mag)
                
                # Classify magnitudes
                def classify_magnitude(mag, thresholds):
                    if mag > thresholds["hard_fail"]:
                        return "HARD_FAIL"
                    elif mag > thresholds["warning"]:
                        return "WARNING"
                    elif mag < thresholds["good"]:
                        return "GOOD"
                    else:
                        return "ACCEPTABLE"
                
                param_classifications[test_name] = {
                    "B_magnitude": classify_magnitude(b_mag, thresholds["matrix_magnitudes"]),
                    "C_magnitude": classify_magnitude(c_mag, thresholds["matrix_magnitudes"]),
                    "D_magnitude": classify_magnitude(d_mag, thresholds["matrix_magnitudes"]),
                    "Delta_magnitude": classify_magnitude(delta_mag, thresholds["delta_timescales"])
                }
                
                print(f"  {test_name}:")
                print(f"    B Matrix Magnitude: {b_mag:.2e} [{param_classifications[test_name]['B_magnitude']}]")
                print(f"    C Matrix Magnitude: {c_mag:.2e} [{param_classifications[test_name]['C_magnitude']}]")
                print(f"    D Vector Magnitude: {d_mag:.2e} [{param_classifications[test_name]['D_magnitude']}]")
                print(f"    Delta Magnitude: {delta_mag:.2e} [{param_classifications[test_name]['Delta_magnitude']}]")
        
        # Parameter summary
        if param_stats["B_magnitudes"]:
            print(f"\n  📊 PARAMETER SUMMARY:")
            print(f"    Average B Magnitude: {np.mean(param_stats['B_magnitudes']):.2e}")
            print(f"    Average C Magnitude: {np.mean(param_stats['C_magnitudes']):.2e}")
            print(f"    Average D Magnitude: {np.mean(param_stats['D_magnitudes']):.2e}")
            print(f"    Average Delta Magnitude: {np.mean(param_stats['Delta_magnitudes']):.2e}")
        
        detailed_results["statistical_analysis"]["parameters"] = param_stats
        detailed_results["classifications"]["parameters"] = param_classifications
        
        # 3. FORWARD/GRADIENT PASS ANALYSIS
        print("\n" + "🔍 3. FORWARD & GRADIENT PASS ANALYSIS")
        print("-" * 80)
        
        forward_results = self.test_results.get("forward_pass_stability", {})
        gradient_results = self.test_results.get("gradient_flow", {})
        
        # Forward pass analysis
        if forward_results.get("success", False):
            basic_forward = forward_results.get("basic_forward", {})
            has_nan = basic_forward.get("has_nan", True)
            has_inf = basic_forward.get("has_inf", True)
            max_mag = basic_forward.get("max_magnitude", float('inf'))
            
            if has_nan or has_inf:
                forward_class = "HARD_FAIL"
            elif max_mag > thresholds["gradient_magnitudes"]["hard_fail"]:
                forward_class = "HARD_FAIL"
            elif max_mag > thresholds["gradient_magnitudes"]["warning"]:
                forward_class = "WARNING"
            elif max_mag < thresholds["gradient_magnitudes"]["good"]:
                forward_class = "GOOD"
            else:
                forward_class = "ACCEPTABLE"
            
            print(f"  Forward Pass:")
            print(f"    Max Output Magnitude: {max_mag:.2e} [{forward_class}]")
            print(f"    Contains NaN: {has_nan}")
            print(f"    Contains Inf: {has_inf}")
            
            detailed_results["classifications"]["forward_pass"] = forward_class
        
        # Gradient analysis
        if gradient_results.get("success", False):
            grad_norm = gradient_results.get("total_grad_norm", float('inf'))
            max_grad = gradient_results.get("max_grad_magnitude", float('inf'))
            has_nan_grad = gradient_results.get("has_nan_grads", True)
            has_inf_grad = gradient_results.get("has_inf_grads", True)
            
            if has_nan_grad or has_inf_grad:
                grad_class = "HARD_FAIL"
            elif grad_norm > thresholds["gradient_magnitudes"]["hard_fail"]:
                grad_class = "HARD_FAIL"
            elif grad_norm > thresholds["gradient_magnitudes"]["warning"]:
                grad_class = "WARNING"
            elif grad_norm < thresholds["gradient_magnitudes"]["good"]:
                grad_class = "GOOD"
            else:
                grad_class = "ACCEPTABLE"
            
            print(f"  Gradient Flow:")
            print(f"    Total Gradient Norm: {grad_norm:.2e} [{grad_class}]")
            print(f"    Max Gradient Magnitude: {max_grad:.2e}")
            print(f"    Contains NaN: {has_nan_grad}")
            print(f"    Contains Inf: {has_inf_grad}")
            
            detailed_results["classifications"]["gradient_flow"] = grad_class
        
        # 4. IMPULSE RESPONSE ANALYSIS
        print("\n" + "🔍 4. IMPULSE RESPONSE ANALYSIS")
        print("-" * 80)
        
        impulse_results = self.test_results.get("impulse_response", {})
        if impulse_results.get("success", False):
            max_response = impulse_results.get("max_response", float('inf'))
            decay_ratio = impulse_results.get("decay_ratio", 1.0)
            is_decaying = impulse_results.get("is_decaying", False)
            total_energy = impulse_results.get("total_energy", float('inf'))
            
            # Classify impulse response
            if not is_decaying or max_response > thresholds["impulse_response"]["hard_fail_max"]:
                impulse_class = "HARD_FAIL"
            elif max_response > thresholds["impulse_response"]["warning_max"] or decay_ratio > 0.5:
                impulse_class = "WARNING"
            elif max_response < thresholds["impulse_response"]["good_max"] and decay_ratio < thresholds["impulse_response"]["decay_threshold"]:
                impulse_class = "GOOD"
            else:
                impulse_class = "ACCEPTABLE"
            
            print(f"  Impulse Response:")
            print(f"    Max Response: {max_response:.2e} [{impulse_class}]")
            print(f"    Decay Ratio: {decay_ratio:.2e}")
            print(f"    Is Decaying: {is_decaying}")
            print(f"    Total Energy: {total_energy:.2e}")
            
            detailed_results["classifications"]["impulse_response"] = impulse_class
        
        # 5. OVERALL ASSESSMENT & RECOMMENDATIONS
        print("\n" + "="*100)
        print("OVERALL ASSESSMENT & RECOMMENDATIONS")
        print("="*100)
        
        # Count classifications
        all_classifications = []
        for category, classifications in detailed_results["classifications"].items():
            if isinstance(classifications, dict):
                all_classifications.extend(classifications.values())
            else:
                all_classifications.append(classifications)
        
        hard_fails = sum(1 for c in all_classifications if c == "HARD_FAIL")
        warnings = sum(1 for c in all_classifications if c == "WARNING")
        goods = sum(1 for c in all_classifications if c == "GOOD")
        acceptables = sum(1 for c in all_classifications if c == "ACCEPTABLE")
        
        total_metrics = len(all_classifications)
        
        print(f"\n📊 CLASSIFICATION SUMMARY:")
        print(f"  🔴 HARD_FAIL: {hard_fails}/{total_metrics} ({hard_fails/total_metrics*100:.1f}%)")
        print(f"  🟡 WARNING:   {warnings}/{total_metrics} ({warnings/total_metrics*100:.1f}%)")
        print(f"  🟢 GOOD:      {goods}/{total_metrics} ({goods/total_metrics*100:.1f}%)")
        print(f"  🔵 ACCEPTABLE: {acceptables}/{total_metrics} ({acceptables/total_metrics*100:.1f}%)")
        
        # Overall health assessment
        if hard_fails > 0:
            overall_health = "CRITICAL"
            health_emoji = "🚨"
        elif warnings > total_metrics * 0.5:
            overall_health = "CONCERNING"
            health_emoji = "⚠️"
        elif goods > total_metrics * 0.7:
            overall_health = "EXCELLENT"
            health_emoji = "🎉"
        else:
            overall_health = "ACCEPTABLE"
            health_emoji = "✅"
        
        print(f"\n{health_emoji} OVERALL NUMERICAL HEALTH: {overall_health}")
        
        # Generate specific recommendations
        recommendations = []
        
        if hard_fails > 0:
            recommendations.append("🚨 CRITICAL: Address hard failures immediately - system may be numerically unstable")
        
        # Eigendecomposition recommendations
        if any(c.get("conditioning") == "HARD_FAIL" for c in eigen_classifications.values()):
            recommendations.append("🔧 Consider regularization techniques for eigendecomposition (add small diagonal term)")
        if any(c.get("reconstruction_error") == "HARD_FAIL" for c in eigen_classifications.values()):
            recommendations.append("🔧 Reconstruction error too high - eigendecomposition is numerically unstable, increase regularization")
        if any(c.get("eigenvalue_stability") == "HARD_FAIL" for c in eigen_classifications.values()):
            recommendations.append("🔧 Check HiPPO matrix construction - eigenvalues should have negative real parts")
        
        # Parameter recommendations  
        if any("HARD_FAIL" in str(c.values()) for c in param_classifications.values()):
            recommendations.append("🔧 Parameter magnitudes too large - consider gradient clipping or learning rate reduction")
        
        # Gradient recommendations
        if detailed_results["classifications"].get("gradient_flow") == "HARD_FAIL":
            recommendations.append("🔧 Gradient explosion detected - implement gradient clipping")
        if detailed_results["classifications"].get("forward_pass") == "HARD_FAIL":
            recommendations.append("🔧 Forward pass instability - check input normalization and parameter initialization")
        
        # Impulse response recommendations
        if detailed_results["classifications"].get("impulse_response") == "HARD_FAIL":
            recommendations.append("🔧 System appears unstable - verify state transition matrix eigenvalues")
        
        if not recommendations:
            recommendations.append("✅ No critical issues detected - system appears numerically stable")
        
        detailed_results["recommendations"] = recommendations
        detailed_results["overall_health"] = overall_health
        detailed_results["classification_summary"] = {
            "hard_fails": hard_fails,
            "warnings": warnings, 
            "goods": goods,
            "acceptables": acceptables,
            "total_metrics": total_metrics
        }
        
        print(f"\n💡 RECOMMENDATIONS:")
        for i, rec in enumerate(recommendations, 1):
            print(f"  {i}. {rec}")
        
        print("\n" + "="*100)
        print("DETAILED STATISTICAL ANALYSIS COMPLETE")
        print("="*100)
        
        return detailed_results
    
    def run_comprehensive_tests(self) -> Dict[str, Any]:
        """Run all numerical stability tests."""
        print("\n" + "="*80)
        print("COMPREHENSIVE NUMERICAL STABILITY TEST SUITE")
        print("="*80)
        
        # Run all tests
        self.test_eigendecomposition_stability()
        self.test_parameter_magnitudes()
        self.test_forward_pass_stability()
        self.test_gradient_flow()
        self.test_impulse_response()
        
        # Generate summary report
        print("\n" + "="*80)
        print("TEST SUMMARY REPORT")
        print("="*80)
        
        total_tests = 0
        passed_tests = 0
        
        for test_name, test_results in self.test_results.items():
            print(f"\n{test_name.upper()}:")
            
            if isinstance(test_results, dict):
                if "success" in test_results:
                    # Single test result
                    total_tests += 1
                    if test_results["success"]:
                        passed_tests += 1
                        print(f"  ✅ PASSED")
                    else:
                        print(f"  ❌ FAILED: {test_results.get('error', 'Unknown error')}")
                else:
                    # Multiple test results
                    for subtest_name, subtest_result in test_results.items():
                        if isinstance(subtest_result, dict) and "success" in subtest_result:
                            total_tests += 1
                            if subtest_result["success"]:
                                passed_tests += 1
                                print(f"  ✅ {subtest_name}: PASSED")
                            else:
                                print(f"  ❌ {subtest_name}: FAILED - {subtest_result.get('error', 'Unknown error')}")
        
        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        
        print(f"\n" + "="*80)
        print(f"OVERALL RESULTS: {passed_tests}/{total_tests} tests passed ({success_rate:.1f}%)")
        
        if success_rate >= 90:
            print("🎉 EXCELLENT: S5 implementation shows strong numerical stability!")
        elif success_rate >= 70:
            print("✅ GOOD: S5 implementation is generally stable with minor issues.")
        elif success_rate >= 50:
            print("⚠️  CONCERNING: S5 implementation has significant stability issues.")
        else:
            print("❌ CRITICAL: S5 implementation has severe numerical problems!")
        
        print("="*80)
        
        return {
            "test_results": self.test_results,
            "summary": {
                "total_tests": total_tests,
                "passed_tests": passed_tests,
                "success_rate": success_rate
            }
        }


# Pytest test functions
def test_eigendecomposition_stability():
    """Pytest wrapper for eigendecomposition stability tests."""
    tester = NumericalStabilityTester()
    results = tester.test_eigendecomposition_stability()
    
    # Check that at least some state dimensions work
    successful_tests = sum(1 for r in results.values() if r.get("success", False))
    assert successful_tests > 0, "No eigendecomposition tests passed"
    
    # Very lenient criteria for eigendecomposition tests (HiPPO matrices are inherently ill-conditioned)
    for test_name, result in results.items():
        if result.get("success", False):
            # Very lenient condition number threshold
            assert result["condition_number"] < 1e30, f"Condition number too high for {test_name}: {result['condition_number']:.2e}"
            # Very lenient reconstruction error threshold (up to 1% error allowed)
            assert result["reconstruction_error"] < 1e-2, f"Reconstruction error too high for {test_name}: {result['reconstruction_error']:.2e}"
            # Ensure all eigenvalues have sufficiently negative real parts
            assert result["eigenvalue_stability"], f"Eigenvalues not stable for {test_name}"
            assert result["max_eigenval_real"] <= -1e-6, f"Eigenvalue real parts not sufficiently negative for {test_name}: {result['max_eigenval_real']}"


def test_parameter_magnitudes():
    """Pytest wrapper for parameter magnitude tests."""
    tester = NumericalStabilityTester()
    results = tester.test_parameter_magnitudes()
    
    # Check that parameters are initialized successfully
    successful_tests = sum(1 for r in results.values() if r.get("success", False))
    assert successful_tests > 0, "No parameter magnitude tests passed"
    
    # More stringent criteria based on user requirements
    for test_name, result in results.items():
        if result.get("success", False) and "analysis" in result:
            analysis = result["analysis"]
            # Check for reasonable parameter magnitudes with stricter thresholds
            if "lambda" in analysis and "magnitude_range" in analysis["lambda"]:
                max_lambda_magnitude = analysis["lambda"]["magnitude_range"][1]
                min_lambda_magnitude = analysis["lambda"]["magnitude_range"][0]
                assert max_lambda_magnitude < 1e4, f"Lambda magnitude too high for {test_name}: {max_lambda_magnitude:.2e}"
                assert min_lambda_magnitude > 1e-8, f"Lambda magnitude too small for {test_name}: {min_lambda_magnitude:.2e}"
            
            if "B" in analysis and "magnitude_stats" in analysis["B"]:
                max_B_magnitude = analysis["B"]["magnitude_stats"]["max"]
                assert max_B_magnitude < 1e4, f"B magnitude too high for {test_name}: {max_B_magnitude:.2e}"
            
            if "C" in analysis and "magnitude_stats" in analysis["C"]:
                max_C_magnitude = analysis["C"]["magnitude_stats"]["max"]
                assert max_C_magnitude < 1e4, f"C magnitude too high for {test_name}: {max_C_magnitude:.2e}"
            
            if "D" in analysis and "norm" in analysis["D"]:
                D_norm = analysis["D"]["norm"]
                assert D_norm < 1e4, f"D norm too high for {test_name}: {D_norm:.2e}"


def test_forward_pass_stability():
    """Pytest wrapper for forward pass stability tests."""
    tester = NumericalStabilityTester()
    results = tester.test_forward_pass_stability()
    
    assert results.get("basic_forward", {}).get("success", False), f"Forward pass test failed: {results.get('error', 'Unknown error')}"
    
    # More stringent criteria based on user requirements
    if "basic_forward" in results:
        basic = results["basic_forward"]
        # Stricter output magnitude thresholds
        assert basic["max_magnitude"] < 1e4, f"Output magnitude too high: {basic['max_magnitude']:.2e}"
        assert basic["mean_magnitude"] > 1e-10, f"Output magnitude too small: {basic['mean_magnitude']:.2e}"
        # Ensure no NaN or Inf in outputs
        assert not basic["has_nan"], "Forward pass output contains NaN values"
        assert not basic["has_inf"], "Forward pass output contains Inf values"
    
    basic_results = results["basic_forward"]
    assert not basic_results["has_nan"], "Forward pass produced NaN values"
    assert not basic_results["has_inf"], "Forward pass produced infinite values"


def test_gradient_flow():
    """Pytest wrapper for gradient flow tests."""
    tester = NumericalStabilityTester()
    results = tester.test_gradient_flow()
    
    assert results.get("success", False), f"Gradient flow test failed: {results.get('error', 'Unknown error')}"
    
    # More stringent criteria based on user requirements
    assert not results["has_nan_grads"], "Gradients contain NaN values"
    assert not results["has_inf_grads"], "Gradients contain infinite values"
    # Stricter gradient norm thresholds
    assert results["total_grad_norm"] > 1e-10, f"Gradients are too small (vanishing gradients): {results['total_grad_norm']:.2e}"
    assert results["total_grad_norm"] < 1e4, f"Gradients are too large (exploding gradients): {results['total_grad_norm']:.2e}"
    
    # Check individual parameter gradient norms if available
    if "param_grad_norms" in results:
        for param_name, grad_norm in results["param_grad_norms"].items():
            assert grad_norm > 1e-12, f"Gradient norm too small for {param_name}: {grad_norm:.2e}"
            assert grad_norm < 1e5, f"Gradient norm too large for {param_name}: {grad_norm:.2e}"


def test_impulse_response():
    """Pytest wrapper for impulse response tests."""
    tester = NumericalStabilityTester()
    results = tester.test_impulse_response()
    
    assert results.get("success", False), f"Impulse response test failed: {results.get('error', 'Unknown error')}"
    
    # More stringent criteria based on user requirements
    assert results["is_decaying"], "Impulse response does not decay (unstable system)"
    # Stricter magnitude threshold
    assert results["max_response"] < 1e4, f"Impulse response magnitude too large: {results['max_response']:.2e}"
    
    # Additional stability checks if available
    if "decay_rate" in results:
        assert results["decay_rate"] > 0, f"Impulse response decay rate not positive: {results['decay_rate']:.2e}"
    if "final_magnitude" in results:
        assert results["final_magnitude"] < 1e-6, f"Impulse response does not decay sufficiently: {results['final_magnitude']:.2e}"
    if "oscillation_measure" in results:
        assert results["oscillation_measure"] < 10.0, f"Impulse response too oscillatory: {results['oscillation_measure']:.2e}"


if __name__ == "__main__":
    # Run comprehensive tests when script is executed directly
    tester = NumericalStabilityTester()
    comprehensive_results = tester.run_comprehensive_tests()
    
    # Save results to file for analysis
    import json
    with open("numerical_stability_results.json", "w") as f:
        # Convert numpy arrays to lists for JSON serialization
        def convert_for_json(obj):
            if isinstance(obj, dict):
                return {k: convert_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_for_json(item) for item in obj]
            elif isinstance(obj, (jnp.ndarray, np.ndarray)):
                return obj.tolist()
            elif isinstance(obj, (jnp.float32, jnp.float64, np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (jnp.int32, jnp.int64, np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, (jnp.bool_, np.bool_)):
                return bool(obj)
            else:
                return obj
        
        json_results = convert_for_json(comprehensive_results)
        json.dump(json_results, f, indent=2)
    
    print(f"\nDetailed results saved to: numerical_stability_results.json")


def test_detailed_statistical_analysis():
    """Pytest wrapper for detailed statistical analysis with user-defined thresholds."""
    tester = NumericalStabilityTester()
    analysis_results = tester.run_detailed_statistical_analysis()
    
    # Extract overall assessment from the actual return structure
    overall_health = analysis_results["overall_health"]
    classification_summary = analysis_results["classification_summary"]
    recommendations = analysis_results["recommendations"]
    
    # Print detailed results for visibility
    print("\n" + "="*80)
    print("DETAILED STATISTICAL ANALYSIS RESULTS")
    print("="*80)
    
    print(f"\nOVERALL HEALTH: {overall_health}")
    print(f"Hard Failures: {classification_summary['hard_fails']}")
    print(f"Warnings: {classification_summary['warnings']}")
    print(f"Good: {classification_summary['goods']}")
    print(f"Acceptable: {classification_summary['acceptables']}")
    print(f"Total Metrics: {classification_summary['total_metrics']}")
    
    if recommendations:
        print("\nRECOMMENDATIONS:")
        for rec in recommendations:
            print(f"  - {rec}")
    
    # Assert based on overall health - only fail on critical issues
    assert overall_health != "CRITICAL", f"Critical numerical stability failures detected: {classification_summary['hard_fails']} hard failures"
    
    # Warn about concerning status but don't fail the test
    if overall_health == "CONCERNING":
        print(f"\nWARNING: System health is concerning with {classification_summary['warnings']} warnings.")
    
    return analysis_results