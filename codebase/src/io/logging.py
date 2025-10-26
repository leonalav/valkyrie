"""Logging utilities for multi-host TPU training.

Implements:
- Multi-host coordinated logging
- Structured logging with metrics
- Performance monitoring
- Wandb integration
- Log aggregation across hosts
"""

import jax
import logging
import sys
import time
from typing import Dict, Any, Optional, Union
from pathlib import Path
import json
from dataclasses import dataclass, asdict
import threading
import queue
from datetime import datetime

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None


@dataclass
class LoggingConfig:
    """Configuration for logging setup."""
    log_level: str = "INFO"
    log_dir: str = "./logs"
    log_to_file: bool = True
    log_to_console: bool = True
    structured_logging: bool = False  # JSONL format
    
    # Metrics and monitoring
    log_interval_steps: int = 10
    eval_interval_steps: int = 1000
    
    # Performance monitoring
    log_throughput: bool = False
    log_memory_usage: bool = False
    log_device_utilization: bool = False
    log_system_metrics: bool = True
    metrics_interval: int = 60  # seconds
    
    # Multi-host coordination
    aggregate_logs: bool = True
    primary_host_only: bool = False
    
    # Wandb integration
    use_wandb: bool = False
    wandb_project: str = "valkyrie-training"
    wandb_entity: Optional[str] = None
    wandb_tags: Optional[list] = None


class MultiHostLogger:
    """
    Multi-host aware logger for TPU training.
    
    Features:
    - Coordinates logging across TPU hosts
    - Aggregates metrics from all hosts
    - Integrates with Wandb for experiment tracking
    - Monitors system performance
    - Structured logging with JSON output
    """
    
    def __init__(self, config: LoggingConfig, name: str = "valkyrie"):
        self.config = config
        self.name = name
        
        # Multi-host info
        self.process_index = jax.process_index()
        self.process_count = jax.process_count()
        self.is_primary_host = (self.process_index == 0)
        
        # Setup logging
        self._setup_logging()
        
        # Wandb setup
        self.wandb_run = None
        if config.use_wandb and self.is_primary_host:
            self._setup_wandb()
        
        # Metrics tracking
        self.metrics_buffer = queue.Queue()
        self.start_time = time.time()
        
        # Performance monitoring
        if config.log_system_metrics:
            self._start_system_monitoring()
        
        self.logger.info(f"Multi-host logger initialized for process {self.process_index}/{self.process_count}")
    
    def _setup_logging(self):
        """Setup Python logging with proper configuration."""
        
        # Create logger
        self.logger = logging.getLogger(f"{self.name}.{self.process_index}")
        self.logger.setLevel(getattr(logging, self.config.log_level.upper()))
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Create formatter
        formatter = logging.Formatter(
            f'%(asctime)s - P{self.process_index}/{self.process_count} - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Console handler
        if self.config.log_to_console:
            if not self.config.primary_host_only or self.is_primary_host:
                console_handler = logging.StreamHandler(sys.stdout)
                console_handler.setFormatter(formatter)
                self.logger.addHandler(console_handler)
        
        # File handler
        if self.config.log_to_file:
            log_dir = Path(self.config.log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)
            
            log_file = log_dir / f"valkyrie_p{self.process_index}.log"
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
        
        # Set root logger level to avoid duplicate messages
        logging.getLogger().setLevel(logging.WARNING)
    
    def _setup_wandb(self):
        """Setup Wandb experiment tracking."""
        
        if not WANDB_AVAILABLE:
            self.logger.warning("Wandb not available, skipping setup")
            return
        
        try:
            # Initialize wandb run
            self.wandb_run = wandb.init(
                project=self.config.wandb_project,
                entity=self.config.wandb_entity,
                tags=self.config.wandb_tags,
                name=f"valkyrie_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                config={
                    'process_count': self.process_count,
                    'logging_config': asdict(self.config),
                }
            )
            
            self.logger.info("✓ Wandb initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Wandb: {e}")
            self.wandb_run = None
    
    def _start_system_monitoring(self):
        """Start background system monitoring."""
        
        def monitor_system():
            """Background thread for system monitoring."""
            try:
                import psutil
                
                while True:
                    # Collect system metrics
                    metrics = {
                        'system/cpu_percent': psutil.cpu_percent(),
                        'system/memory_percent': psutil.virtual_memory().percent,
                        'system/disk_usage_percent': psutil.disk_usage('/').percent,
                        'timestamp': time.time(),
                    }
                    
                    # Add to metrics buffer
                    try:
                        self.metrics_buffer.put_nowait(('system_metrics', metrics))
                    except queue.Full:
                        pass  # Skip if buffer full
                    
                    time.sleep(self.config.metrics_interval)
                    
            except ImportError:
                self.logger.warning("psutil not available, system monitoring disabled")
            except Exception as e:
                self.logger.error(f"System monitoring error: {e}")
        
        # Start monitoring thread
        monitor_thread = threading.Thread(target=monitor_system, daemon=True)
        monitor_thread.start()
    
    def info(self, message: str, **kwargs):
        """Log info message."""
        self.logger.info(message)
        if kwargs:
            self._log_metrics('info', kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message."""
        self.logger.warning(message)
        if kwargs:
            self._log_metrics('warning', kwargs)
    
    def error(self, message: str, **kwargs):
        """Log error message."""
        self.logger.error(message)
        if kwargs:
            self._log_metrics('error', kwargs)
    
    def debug(self, message: str, **kwargs):
        """Log debug message."""
        self.logger.debug(message)
        if kwargs:
            self._log_metrics('debug', kwargs)
    
    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None):
        """
        Log training metrics.
        
        Args:
            metrics: Dictionary of metrics to log
            step: Training step (optional)
        """
        
        # Add metadata
        enriched_metrics = metrics.copy()
        enriched_metrics.update({
            'process_index': self.process_index,
            'timestamp': time.time(),
            'elapsed_time': time.time() - self.start_time,
        })
        
        if step is not None:
            enriched_metrics['step'] = step
        
        # Log to console/file
        self.logger.info(f"Metrics: {json.dumps(enriched_metrics, indent=2)}")
        
        # Add to metrics buffer
        self._log_metrics('training_metrics', enriched_metrics)
    
    def _log_metrics(self, metric_type: str, metrics: Dict[str, Any]):
        """Internal method to buffer metrics."""
        
        try:
            self.metrics_buffer.put_nowait((metric_type, metrics))
        except queue.Full:
            self.logger.warning("Metrics buffer full, dropping metrics")
        
        # Send to Wandb if available and primary host
        if self.wandb_run and self.is_primary_host and metric_type == 'training_metrics':
            try:
                # Filter out non-numeric values for Wandb
                wandb_metrics = {
                    k: v for k, v in metrics.items() 
                    if isinstance(v, (int, float, bool))
                }
                
                step = wandb_metrics.pop('step', None)
                self.wandb_run.log(wandb_metrics, step=step)
                
            except Exception as e:
                self.logger.warning(f"Failed to log to Wandb: {e}")
    
    def log_model_info(self, model_config: Any, total_params: int):
        """Log model configuration and parameter count."""
        
        model_info = {
            'model/total_parameters': total_params,
            'model/parameter_millions': total_params / 1e6,
            'model/parameter_billions': total_params / 1e9,
        }
        
        # Add config if it has useful attributes
        if hasattr(model_config, '__dict__'):
            config_dict = {
                f'config/{k}': v for k, v in model_config.__dict__.items()
                if isinstance(v, (int, float, str, bool))
            }
            model_info.update(config_dict)
        
        self.log_metrics(model_info)
        self.info(f"Model initialized with {total_params:,} parameters ({total_params/1e9:.2f}B)")
    
    def log_training_start(self, total_steps: int, dataset_info: Dict[str, Any]):
        """Log training start information."""
        
        training_info = {
            'training/total_steps': total_steps,
            'training/process_count': self.process_count,
            'training/start_time': time.time(),
        }
        
        # Add dataset info
        for key, value in dataset_info.items():
            if isinstance(value, (int, float, str, bool)):
                training_info[f'dataset/{key}'] = value
        
        self.log_metrics(training_info)
        self.info(f"Training started: {total_steps} steps on {self.process_count} processes")
    
    def log_phase_change(self, phase_index: int, phase_config: Dict[str, Any]):
        """Log curriculum phase change."""
        
        phase_info = {
            'curriculum/phase': phase_index,
            'curriculum/change_time': time.time(),
        }
        
        # Add phase config
        for key, value in phase_config.items():
            if isinstance(value, (int, float, str, bool)):
                phase_info[f'curriculum/{key}'] = value
        
        self.log_metrics(phase_info)
        self.info(f"Curriculum phase changed to {phase_index}: {phase_config}")
    
    def log_checkpoint_save(self, step: int, checkpoint_path: str, save_time: float):
        """Log checkpoint save event."""
        
        checkpoint_info = {
            'checkpoint/step': step,
            'checkpoint/save_time_seconds': save_time,
            'checkpoint/timestamp': time.time(),
        }
        
        self.log_metrics(checkpoint_info)
        self.info(f"Checkpoint saved at step {step} in {save_time:.2f}s: {checkpoint_path}")
    
    def log_performance_metrics(self, throughput: float, memory_usage: Dict[str, float]):
        """Log performance metrics."""
        
        perf_metrics = {
            'performance/throughput_tokens_per_sec': throughput,
            'performance/timestamp': time.time(),
        }
        
        # Add memory usage
        for key, value in memory_usage.items():
            perf_metrics[f'performance/memory_{key}'] = value
        
        self.log_metrics(perf_metrics)
    
    def aggregate_metrics_across_hosts(self, local_metrics: Dict[str, float]) -> Dict[str, float]:
        """
        Aggregate metrics across all hosts.
        
        Args:
            local_metrics: Metrics from current host
            
        Returns:
            Aggregated metrics (only on primary host)
        """
        
        if self.process_count == 1:
            return local_metrics
        
        try:
            # Convert to JAX arrays for aggregation
            metric_values = jnp.array([local_metrics.get(k, 0.0) for k in sorted(local_metrics.keys())])
            
            # All-reduce to sum across hosts
            total_values = jax.lax.psum(metric_values, axis_name='hosts')
            
            # Average across hosts
            avg_values = total_values / self.process_count
            
            # Convert back to dict (only on primary host)
            if self.is_primary_host:
                aggregated = {
                    k: float(v) for k, v in zip(sorted(local_metrics.keys()), avg_values)
                }
                return aggregated
            else:
                return {}
                
        except Exception as e:
            self.logger.warning(f"Failed to aggregate metrics: {e}")
            return local_metrics if self.is_primary_host else {}
    
    def flush_metrics(self):
        """Flush any buffered metrics."""
        
        if self.wandb_run:
            try:
                # Process any remaining metrics in buffer
                while not self.metrics_buffer.empty():
                    try:
                        metric_type, metrics = self.metrics_buffer.get_nowait()
                        if metric_type == 'training_metrics' and self.is_primary_host:
                            wandb_metrics = {
                                k: v for k, v in metrics.items() 
                                if isinstance(v, (int, float, bool))
                            }
                            step = wandb_metrics.pop('step', None)
                            self.wandb_run.log(wandb_metrics, step=step)
                    except queue.Empty:
                        break
                
                # Flush wandb
                self.wandb_run.finish()
                
            except Exception as e:
                self.logger.error(f"Failed to flush metrics: {e}")
    
    def close(self):
        """Close logger and cleanup resources."""
        
        self.flush_metrics()
        
        # Close handlers
        for handler in self.logger.handlers:
            handler.close()
        
        self.info("Logger closed")


def setup_logging(config: LoggingConfig, name: str = "valkyrie") -> MultiHostLogger:
    """
    Setup multi-host logging with given configuration.
    
    Args:
        config: Logging configuration
        name: Logger name
        
    Returns:
        Configured multi-host logger
    """
    
    return MultiHostLogger(config, name)


def get_logger(name: str) -> logging.Logger:
    """
    Get a standard Python logger.
    
    Args:
        name: Logger name
        
    Returns:
        Python logger
    """
    
    return logging.getLogger(name)


# Preset configurations
def get_development_logging_config() -> LoggingConfig:
    """Get logging configuration for development."""
    return LoggingConfig(
        log_level="DEBUG",
        log_to_console=True,
        log_to_file=True,
        use_wandb=False,
        log_system_metrics=True,
    )


def get_production_logging_config() -> LoggingConfig:
    """Get logging configuration for production training."""
    return LoggingConfig(
        log_level="INFO",
        log_to_console=True,
        log_to_file=True,
        use_wandb=True,
        log_system_metrics=True,
        primary_host_only=True,  # Reduce log volume
    )


def get_debug_logging_config() -> LoggingConfig:
    """Get logging configuration for debugging."""
    return LoggingConfig(
        log_level="DEBUG",
        log_to_console=True,
        log_to_file=True,
        use_wandb=False,
        log_system_metrics=False,
        primary_host_only=False,  # All hosts log for debugging
    )