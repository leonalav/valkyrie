"""Main training script for Valkyrie model.

Entry point for training Valkyrie (Longformer + S5) on TPU v4-32.
Implements the complete training pipeline with:
- Configuration loading and validation
- Multi-host TPU coordination
- Chunked sequence processing
- Progressive curriculum
- Checkpointing and monitoring
"""

import jax
import jax.numpy as jnp
import yaml
import argparse
import logging
import time
import sys
from pathlib import Path
from typing import Dict, Any, Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from model import ValkyrieModel, ValkyrieConfig
from sharding import setup_tpu_mesh, get_model_specs, get_training_specs
from train import TrainingLoop, ChunkConfig, CurriculumConfig
from data import create_data_loader, FineWebConfig, TokenizerConfig, get_fineweb_edu_config, get_fineweb_tokenizer_config
from io import CheckpointManager, CheckpointConfig, setup_logging, LoggingConfig
from utils.debug import (
    get_tpu_mixed_precision_config, 
    MixedPrecisionPolicy,
    print_param_stats,
    check_for_nans,
    monitor_memory_usage
)

logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    logger.info(f"Configuration loaded from: {config_path}")
    return config


def create_model_from_config(config: Dict[str, Any]) -> ValkyrieModel:
    """Create Valkyrie model from configuration."""
    
    model_config = ValkyrieConfig(**config['model'])
    model = ValkyrieModel(model_config)
    
    logger.info("Model created:")
    logger.info(f"  Architecture: Longformer={model_config.use_longformer_attention}, S5={model_config.use_s5}")
    logger.info(f"  Dimensions: d_model={model_config.d_model}, n_layers={model_config.n_layers}")
    logger.info(f"  Attention: n_heads={model_config.n_heads}, window_size={model_config.longformer_window_size}")
    logger.info(f"  S5: state_dim={model_config.s5_state_dim}")
    
    return model


def setup_training_components(config: Dict[str, Any], model: ValkyrieModel):
    """Setup all training components."""
    
    # Setup TPU mesh
    logger.info("Setting up TPU mesh...")
    mesh = setup_tpu_mesh()
    
    # Setup mixed precision policy
    logger.info("Setting up mixed precision policy...")
    mp_config = get_tpu_mixed_precision_config()
    mixed_precision = MixedPrecisionPolicy(mp_config)
    
    # Setup data pipeline
    logger.info("Setting up data pipeline...")
    fineweb_config = FineWebConfig(**config['data'])
    tokenizer_config = TokenizerConfig(**config['data']['tokenizer'])
    
    data_loader = create_data_loader(
        config=fineweb_config,
        tokenizer_config=tokenizer_config,
    )
    
    # Setup checkpointing
    logger.info("Setting up checkpointing...")
    checkpoint_config = CheckpointConfig(**config['checkpointing'])
    checkpoint_manager = CheckpointManager(
        config=checkpoint_config,
        mesh=mesh,
        partition_specs=get_model_specs(model.config, use_2d_sharding=True),
    )
    
    # Setup training loop
    logger.info("Setting up training loop...")
    chunk_config = ChunkConfig(**config['training']['chunk_config'])
    curriculum_config = CurriculumConfig(phases=config['training']['curriculum']['phases'])
    
    training_loop = TrainingLoop(
        model=model,
        config=model.config,
        chunk_config=chunk_config,
        curriculum_config=curriculum_config,
        mesh=mesh,
        checkpoint_manager=checkpoint_manager,
    )
    
    return {
        'mesh': mesh,
        'mixed_precision': mixed_precision,
        'data_loader': data_loader,
        'checkpoint_manager': checkpoint_manager,
        'training_loop': training_loop,
    }


def run_validation_tests(model: ValkyrieModel, config: Dict[str, Any]) -> bool:
    """Run validation tests before training."""
    
    logger.info("Running pre-training validation tests...")
    
    try:
        # Import test modules
        from utils.tests.s5_unit_test import run_s5_tests
        from utils.tests.attention_test import run_attention_tests
        
        # Run S5 tests if using S5
        if config['model']['use_s5'] and config['validation']['run_s5_tests']:
            logger.info("Running S5 unit tests...")
            if not run_s5_tests():
                logger.error("S5 tests failed")
                return False
        
        # Run attention tests if using Longformer
        if config['model']['use_longformer_attention'] and config['validation']['run_attention_tests']:
            logger.info("Running Longformer attention tests...")
            if not run_attention_tests():
                logger.error("Attention tests failed")
                return False
        
        logger.info("✓ All validation tests passed")
        return True
        
    except Exception as e:
        logger.error(f"Validation tests failed: {e}")
        return False


def main():
    """Main training function."""
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="Valkyrie Training")
    parser.add_argument("--config", type=str, required=True, help="Configuration file path")
    parser.add_argument("--log_dir", type=str, default="./logs", help="Log directory")
    parser.add_argument("--checkpoint_dir", type=str, default=None, help="Checkpoint directory override")
    parser.add_argument("--resume_from", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--validate_only", action="store_true", help="Run validation tests only")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    
    args = parser.parse_args()
    
    # Load configuration
    try:
        config = load_config(args.config)
    except Exception as e:
        print(f"❌ Failed to load configuration: {e}")
        return 1
    
    # Override config with command line arguments
    if args.checkpoint_dir:
        config['checkpointing']['checkpoint_dir'] = args.checkpoint_dir
    
    if args.debug:
        config['logging']['log_level'] = "DEBUG"
        config['system']['jax_debug_nans'] = True
    
    # Setup logging
    logging_config = LoggingConfig(**config['logging'])
    logging_config.log_dir = args.log_dir
    
    multi_logger = setup_logging(logging_config)
    
    # Log startup info
    logger.info("=== Valkyrie Training Started ===")
    logger.info(f"Configuration: {args.config}")
    logger.info(f"JAX devices: {jax.device_count()}")
    logger.info(f"JAX processes: {jax.process_count()}")
    logger.info(f"Process index: {jax.process_index()}")
    
    # Set JAX configuration
    if config['system']['jax_debug_nans']:
        jax.config.update('jax_debug_nans', True)
        logger.info("JAX NaN debugging enabled")
    
    if config['system']['jax_disable_jit']:
        jax.config.update('jax_disable_jit', True)
        logger.info("JAX JIT disabled for debugging")
    
    if config['system']['jax_enable_x64']:
        jax.config.update('jax_enable_x64', True)
        logger.info("JAX x64 precision enabled for numerical stability")
    
    
    try:
        # Create model
        logger.info("Creating model...")
        model = create_model_from_config(config)
        
        # Setup training components
        logger.info("Setting up training components...")
        components = setup_training_components(config, model)
        
        # Initialize model parameters and print stats
        logger.info("Initializing model parameters...")
        key = jax.random.PRNGKey(config.get('seed', 42))
        
        with components['mesh']:
            training_state = components['training_loop'].initialize_training_state(key)
        
        # Print parameter statistics
        param_stats = print_param_stats(training_state.params, "Valkyrie Model")
        multi_logger.log_model_info(model.config, param_stats['total_parameters'])
        
        # Run validation tests
        if config['validation']['run_s5_tests'] or config['validation']['run_attention_tests']:
            logger.info("Running validation tests...")
            if not run_validation_tests(model, config):
                logger.error("Validation tests failed")
                return 1
        
        if args.validate_only:
            logger.info("Validation complete, exiting")
            return 0
        
        # Resume from checkpoint if specified
        if args.resume_from:
            logger.info(f"Resuming from checkpoint: {args.resume_from}")
            resumed_state = components['checkpoint_manager'].load(checkpoint_path=args.resume_from)
            if resumed_state:
                training_state = resumed_state
                logger.info(f"Resumed from step {training_state.step}")
            else:
                logger.error("Failed to resume from checkpoint")
                return 1
        
        # Log training start
        total_steps = config['training']['total_steps']
        dataset_info = components['data_loader'].get_stats()
        multi_logger.log_training_start(total_steps, dataset_info)
        
        # Main training loop
        logger.info("Starting training...")
        start_time = time.time()
        
        try:
            # Get data iterator
            data_iterator = components['data_loader'].get_batch_iterator()
            
            # Train for specified steps
            final_state = components['training_loop'].train_epoch(
                state=training_state,
                data_loader=data_iterator,
                max_steps=total_steps,
            )
            
            # Training completed successfully
            training_time = time.time() - start_time
            logger.info(f"Training completed in {training_time:.2f} seconds")
            
            # Save final checkpoint
            components['checkpoint_manager'].save(final_state, checkpoint_type="full")
            
            # Log final metrics
            final_metrics = components['training_loop'].get_training_metrics()
            multi_logger.log_metrics(final_metrics, step=final_state.step)
            
            logger.info("🎉 Training completed successfully!")
            return 0
            
        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
            
            # Save checkpoint before exiting
            logger.info("Saving checkpoint before exit...")
            components['checkpoint_manager'].save(training_state, checkpoint_type="full")
            
            return 0
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            
            # Save emergency checkpoint
            try:
                logger.info("Saving emergency checkpoint...")
                components['checkpoint_manager'].save(training_state, checkpoint_type="emergency")
            except:
                logger.error("Failed to save emergency checkpoint")
            
            return 1
    
    except Exception as e:
        logger.error(f"Setup failed: {e}")
        return 1
    
    finally:
        # Cleanup
        if 'multi_logger' in locals():
            multi_logger.close()


if __name__ == "__main__":
    exit(main())