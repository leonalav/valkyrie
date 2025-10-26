"""PLAN Training Orchestrator

Main orchestrator for the BigBird+S5+HRM training pipeline following PLAN specifications.
Coordinates multi-source data loading, phase-based training, and HRM supervision.

Features:
- Multi-phase training curriculum
- HRM one-step gradient training
- Algorithmic task integration
- Deterministic training with proper seeding
- Structured logging and checkpointing
"""

import jax
import jax.numpy as jnp
from flax.training import train_state
import optax
import orbax.checkpoint as ocp
from typing import Dict, List, Optional, Tuple
import logging
from pathlib import Path
import json
import time
from dataclasses import asdict

from ..model.gryphon.gryphon_hrm_model import GryphonHRMModel, GryphonHRMState
from ..model.gryphon.gryphon_config import get_gryphon_1_2b_config
from ..data import (
    create_plan_data_loader,
    create_plan_task_generator, 
    create_plan_sampler,
    PhaseBasedSampler
)
from ..sharding.mesh_setup import create_mesh, get_partition_specs
from ..io.checkpoint import create_checkpoint_manager
from ..io.logging import setup_structured_logging
from .hrm_training_loop import HRMTrainingLoop, HRMTrainingConfig, HRMTrainingState, create_hrm_training_config

logger = logging.getLogger(__name__)


class PLANTrainingOrchestrator:
    """Main orchestrator for PLAN training pipeline."""
    
    def __init__(
        self,
        config_path: str,
        output_dir: str,
        performance_track: bool = True,
        global_seed: int = 42
    ):
        """Initialize PLAN training orchestrator."""
        self.config_path = config_path
        self.output_dir = Path(output_dir)
        self.performance_track = performance_track
        self.global_seed = global_seed
        
        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.log_dir = self.output_dir / "logs"
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.log_dir.mkdir(exist_ok=True)
        
        # Initialize global RNG
        self.rng_key = jax.random.PRNGKey(global_seed)
        
        # Setup logging
        self.logger = setup_structured_logging(
            log_dir=str(self.log_dir),
            experiment_name="bigbird_s5_hrm_1_2b"
        )
        
        logger.info(f"Initialized PLAN Training Orchestrator")
        logger.info(f"  - Output directory: {output_dir}")
        logger.info(f"  - Performance track: {performance_track}")
        logger.info(f"  - Global seed: {global_seed}")
        
        # Initialize components
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize all training components."""
        logger.info("Initializing training components...")
        
        # 1. Model configuration
        self.model_config = get_gryphon_1_2b_config()
        logger.info(f"Model config: {self.model_config.d_model}x{self.model_config.n_layers}")
        
        # 2. Data pipeline
        logger.info("Initializing data pipeline...")
        self.data_loader = create_plan_data_loader(performance_track=self.performance_track)
        self.task_generator = create_plan_task_generator(seed=self.global_seed)
        self.phase_sampler = create_plan_sampler(self.data_loader, self.task_generator)
        
        # 3. Model and training state
        logger.info("Initializing model...")
        self.model = GryphonHRMModel(config=self.model_config)
        
        # 4. TPU mesh and sharding
        logger.info("Setting up TPU mesh...")
        self.mesh = create_mesh([4, 4, 2])  # TPU v4-32 mesh
        self.partition_specs = get_partition_specs(self.model_config)
        
        # 5. Optimizer
        self.optimizer = self._create_optimizer()
        
        # 6. HRM training loop
        logger.info("Initializing HRM training loop...")
        self.hrm_training_loop = None  # Will be initialized per phase
        # Initialize for the current phase immediately to avoid None usage
        current_phase_config = self.phase_sampler.get_current_config()
        self._update_hrm_training_loop(current_phase_config)
        
        # 7. Checkpoint manager
        self.checkpoint_manager = create_checkpoint_manager(
            checkpoint_dir=str(self.checkpoint_dir),
            max_to_keep=10
        )
        
        logger.info("All components initialized successfully")
    
    def _create_optimizer(self) -> optax.GradientTransformation:
        """Create optimizer following PLAN specifications."""
        # Get current phase config for learning rate
        current_config = self.phase_sampler.get_current_config()
        base_lr = current_config["learning_rate"]
        
        # Create learning rate schedule
        warmup_steps = 5000
        total_steps = sum(phase.max_steps for phase in self.phase_sampler.phases.values())
        
        schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=base_lr,
            warmup_steps=warmup_steps,
            decay_steps=total_steps,
            end_value=base_lr * 0.1
        )
        
        # Create optimizer with parameter-specific learning rates
        optimizer = optax.chain(
            optax.clip_by_global_norm(1.0),  # Gradient clipping
            optax.adamw(
                learning_rate=schedule,
                b1=0.9,
                b2=0.95,
                eps=1e-8,
                weight_decay=0.1
            )
        )
        
        return optimizer
    
    def _initialize_training_state(self, batch_size: int = 1) -> HRMTrainingState:
        """Initialize training state."""
        logger.info("Initializing training state...")
        
        # Initialize model parameters
        self.rng_key, init_key = jax.random.split(self.rng_key)
        
        # Create dummy inputs for initialization
        dummy_input_ids = jnp.ones((batch_size, self.model_config.max_sequence_length), dtype=jnp.int32)
        
        # Initialize model
        variables = self.model.init(
            init_key,
            input_ids=dummy_input_ids,
            deterministic=True
        )
        
        # Create train state
        model_state = train_state.TrainState.create(
            apply_fn=self.model.apply,
            params=variables['params'],
            tx=self.optimizer
        )
        
        # Initialize HRM runtime state via apply(method=self.model.init_state)
        gryphon_state = self.model.apply(
            {'params': variables['params']},
            batch_size,
            self.model_config.max_sequence_length,
            method=self.model.init_state
        )
        
        # Create HRM training state
        training_state = HRMTrainingState(
            model_state=model_state,
            hrm_carry=gryphon_state.hrm_carry,
            s5_state=gryphon_state.s5_state,
            global_tokens=gryphon_state.global_tokens,
            step=0,
            phase=self.phase_sampler.current_phase.name
        )
        
        logger.info("Training state initialized")
        return training_state
    
    def _update_hrm_training_loop(self, phase_config: Dict):
        """Update HRM training loop for current phase."""
        hrm_config = create_hrm_training_config(
            phase_config,
            use_act=phase_config.get("hrm_use_act", False)
        )
        
        self.hrm_training_loop = HRMTrainingLoop(
            model=self.model,
            config=hrm_config
        )
        
        logger.info(f"Updated HRM training loop for phase: {phase_config['phase_name']}")
    
    def _save_checkpoint(
        self,
        training_state: HRMTrainingState,
        step: int,
        phase: str
    ):
        """Save training checkpoint."""
        checkpoint_data = {
            "model_state": training_state.model_state,
            "hrm_carry": training_state.hrm_carry,
            "s5_state": training_state.s5_state,
            "global_tokens": training_state.global_tokens,
            "step": step,
            "phase": phase,
            "rng_key": self.rng_key,
            "phase_sampler_state": {
                "current_phase_idx": self.phase_sampler.current_phase_idx,
                "phase_step_count": self.phase_sampler.phase_step_count,
                "step_count": self.phase_sampler.step_count
            }
        }
        
        self.checkpoint_manager.save(step, checkpoint_data)
        logger.info(f"Saved checkpoint at step {step}")
    
    def _load_checkpoint(self, step: Optional[int] = None) -> Optional[HRMTrainingState]:
        """Load training checkpoint."""
        try:
            checkpoint_data = self.checkpoint_manager.restore(step)
            
            if checkpoint_data is None:
                return None
            
            # Restore training state
            training_state = HRMTrainingState(
                model_state=checkpoint_data["model_state"],
                hrm_carry=checkpoint_data["hrm_carry"],
                s5_state=checkpoint_data["s5_state"],
                global_tokens=checkpoint_data["global_tokens"],
                step=checkpoint_data["step"],
                phase=checkpoint_data["phase"]
            )
            
            # Restore RNG key
            self.rng_key = checkpoint_data["rng_key"]
            
            # Restore phase sampler state
            sampler_state = checkpoint_data["phase_sampler_state"]
            self.phase_sampler.current_phase_idx = sampler_state["current_phase_idx"]
            self.phase_sampler.phase_step_count = sampler_state["phase_step_count"]
            self.phase_sampler.step_count = sampler_state["step_count"]
            
            logger.info(f"Loaded checkpoint from step {checkpoint_data['step']}")
            return training_state
            
        except Exception as e:
            logger.warning(f"Failed to load checkpoint: {e}")
            return None
    
    def _log_metrics(self, metrics: Dict, step: int):
        """Log training metrics."""
        # Add timestamp
        metrics["timestamp"] = time.time()
        metrics["step"] = step
        
        # Log to structured logger
        self.logger.info("training_metrics", extra=metrics)
        
        # Log key metrics to console
        if "loss" in metrics:
            logger.info(f"Step {step}: loss={metrics['loss']:.4f}")
        
        if "phase" in metrics:
            logger.info(f"  Phase: {metrics['phase']}")
    
    def train(
        self,
        resume_from_checkpoint: bool = True,
        max_steps: Optional[int] = None
    ):
        """Run the complete PLAN training pipeline."""
        logger.info("Starting PLAN training pipeline")
        
        # Try to load checkpoint
        training_state = None
        if resume_from_checkpoint:
            training_state = self._load_checkpoint()
        
        # Initialize training state if not loaded
        if training_state is None:
            training_state = self._initialize_training_state()
        
        # Save initial configuration
        config_path = self.output_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump({
                "model_config": asdict(self.model_config),
                "global_seed": self.global_seed,
                "performance_track": self.performance_track,
                "phases": [asdict(phase) for phase in self.phase_sampler.phases.values()]
            }, f, indent=2)
        
        # Training loop
        try:
            step_count = training_state.step
            
            # Stream batches from phase sampler
            self.rng_key, data_key = jax.random.split(self.rng_key)
            
            for batch_data, batch_metadata in self.phase_sampler.stream_batches(
                rng_key=data_key,
                max_batches=max_steps
            ):
                # Update HRM training loop if phase changed
                current_config = self.phase_sampler.get_current_config()
                if training_state.phase != current_config["phase_name"]:
                    self._update_hrm_training_loop(current_config)
                    training_state = training_state._replace(phase=current_config["phase_name"])
                
                # Convert to HRM batch and train
                hrm_batch = self.hrm_training_loop._convert_to_hrm_batch(batch_data)
                
                # Training step
                self.rng_key, step_key = jax.random.split(self.rng_key)
                training_state, step_metrics = self.hrm_training_loop.train_step_fn(
                    training_state, hrm_batch, step_key
                )
                
                # Add batch metadata
                step_metrics.update(batch_metadata)
                step_metrics.update(current_config)
                
                # Log metrics
                self._log_metrics(step_metrics, step_count)
                
                # Save checkpoint periodically
                if step_count % 1000 == 0:
                    self._save_checkpoint(training_state, step_count, current_config["phase_name"])
                
                step_count += 1
                
                # Check for completion
                if max_steps and step_count >= max_steps:
                    break
            
            # Final checkpoint
            self._save_checkpoint(training_state, step_count, training_state.phase)
            
            logger.info(f"Training completed after {step_count} steps")
            
        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
            self._save_checkpoint(training_state, step_count, training_state.phase)
        
        except Exception as e:
            logger.error(f"Training failed with error: {e}")
            # Save crash checkpoint
            self._save_checkpoint(training_state, step_count, training_state.phase)
            raise
    
    def evaluate(
        self,
        checkpoint_step: Optional[int] = None,
        num_eval_batches: int = 100
    ) -> Dict:
        """Evaluate model on validation data."""
        logger.info("Starting evaluation...")
        
        # Load checkpoint
        training_state = self._load_checkpoint(checkpoint_step)
        if training_state is None:
            raise ValueError("No checkpoint found for evaluation")
        
        # Initialize HRM training loop for current phase
        current_config = self.phase_sampler.get_current_config()
        self._update_hrm_training_loop(current_config)
        
        # Evaluation loop
        eval_metrics = []
        self.rng_key, eval_key = jax.random.split(self.rng_key)
        
        for i, (batch_data, batch_metadata) in enumerate(
            self.phase_sampler.stream_batches(rng_key=eval_key, max_batches=num_eval_batches)
        ):
            if i >= num_eval_batches:
                break
            
            # Convert to HRM batch
            hrm_batch = self.hrm_training_loop._convert_to_hrm_batch(batch_data)
            
            # Evaluation step
            self.rng_key, step_key = jax.random.split(self.rng_key)
            step_metrics = self.hrm_training_loop.eval_step_fn(
                training_state, hrm_batch, step_key
            )
            
            eval_metrics.append(step_metrics)
        
        # Aggregate metrics
        aggregated_metrics = {}
        for key in eval_metrics[0].keys():
            values = [m[key] for m in eval_metrics]
            aggregated_metrics[f"avg_{key}"] = jnp.mean(jnp.array(values))
            aggregated_metrics[f"std_{key}"] = jnp.std(jnp.array(values))
        
        logger.info(f"Evaluation completed: {aggregated_metrics}")
        return aggregated_metrics


def main():
    """Main entry point for PLAN training."""
    import argparse
    
    parser = argparse.ArgumentParser(description="PLAN Training Pipeline")
    parser.add_argument("--config", type=str, required=True, help="Config file path")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--performance_track", action="store_true", help="Use performance track (120-160B tokens)")
    parser.add_argument("--seed", type=int, default=42, help="Global seed")
    parser.add_argument("--max_steps", type=int, help="Maximum training steps")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--eval_only", action="store_true", help="Run evaluation only")
    
    args = parser.parse_args()
    
    # Initialize orchestrator
    orchestrator = PLANTrainingOrchestrator(
        config_path=args.config,
        output_dir=args.output_dir,
        performance_track=args.performance_track,
        global_seed=args.seed
    )
    
    if args.eval_only:
        # Run evaluation
        results = orchestrator.evaluate()
        print(f"Evaluation results: {results}")
    else:
        # Run training
        orchestrator.train(
            resume_from_checkpoint=args.resume,
            max_steps=args.max_steps
        )


if __name__ == "__main__":
    main()