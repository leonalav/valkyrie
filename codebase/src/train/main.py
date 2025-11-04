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
import os
from pathlib import Path
from typing import Dict, Any, Optional, List, Iterator

# Load environment variables from .env file (optional)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # dotenv not available, skip loading .env file
    pass

# Add codebase to path if running directly
current_file = Path(__file__).resolve()
codebase_dir = current_file.parent.parent.parent  # src/train/main.py -> src/train -> src -> codebase
if str(codebase_dir) not in sys.path:
    sys.path.insert(0, str(codebase_dir))

# Use absolute imports
from src.model import ValkyrieModel, ValkyrieConfig
from src.sharding import make_mesh, get_mesh_context, get_model_specs
# from src.data.multi_source_loader import PackedSequence  # Commented out until implemented
from src.train.train_loop import TrainingLoop, ChunkConfig, CurriculumConfig
# from src.data import create_data_loader, MultiSourceConfig, MultiSourceDataLoader, DataSourceConfig, TokenizerConfig, create_plan_data_loader  # Commented out until implemented
from src.data import create_data_loader, TokenizerConfig
from src.io import CheckpointManager, CheckpointConfig, setup_logging, LoggingConfig
from src.utils.debug import (
    get_tpu_mixed_precision_config, 
    MixedPrecisionPolicy,
    print_param_stats,
    check_for_nans,
    monitor_memory_usage
)

logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    import os
    import re
    
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config_text = f.read()
    
    # Substitute environment variables in the format ${VAR_NAME}
    def env_var_replacer(match):
        var_name = match.group(1)
        return os.environ.get(var_name, match.group(0))  # Return original if not found
    
    config_text = re.sub(r'\$\{([^}]+)\}', env_var_replacer, config_text)
    
    config = yaml.safe_load(config_text)
    
    logger.info(f"Configuration loaded from: {config_path}")
    return config


def create_model_from_config(config: Dict[str, Any]) -> ValkyrieModel:
    """Create Valkyrie model from configuration."""
    
    model_config = ValkyrieConfig(**config['model'])
    model = ValkyrieModel(model_config)
    
    logger.info("Model created:")
    logger.info(f"  Architecture: BigBird={model_config.use_bigbird_attention}, S5={model_config.use_s5}")
    logger.info(f"  Dimensions: d_model={model_config.d_model}, n_layers={model_config.n_layers}")
    logger.info(f"  Attention: n_heads={model_config.n_heads}, block_size={model_config.bigbird_block_size}")
    logger.info(f"  S5: state_dim={model_config.s5_state_dim}")
    logger.info(f"  HRM: enabled={model_config.use_hrm}, plan_length={model_config.hrm_plan_length}")
    
    return model


def convert_packed_sequences_to_batch(packed_sequences: List[Dict]) -> Dict[str, jnp.ndarray]:
    """Convert List[Dict] to batch dictionary with only fields used by training.
    
    Args:
        packed_sequences: List of sequence dictionaries
        
    Returns:
        Dictionary with only the fields used by train_step:
        - input_ids: [batch_size, pack_length]
    """
    if not packed_sequences:
        raise ValueError("Empty packed_sequences list")
    
    # Only stack input_ids since that's all the training code uses
    input_ids = jnp.stack([seq.input_ids for seq in packed_sequences])
    
    return {
        'input_ids': input_ids,
    }


def create_batch_iterator(data_loader) -> Iterator[Dict[str, jnp.ndarray]]:
#     """Create iterator that converts List[PackedSequence] to proper batch format."""
    for packed_batch in data_loader.get_batch_iterator():
        yield convert_packed_sequences_to_batch(packed_batch)


def setup_training_components(config: Dict[str, Any], model: ValkyrieModel):
    """Setup all training components."""
    
    # Setup TPU mesh
    logger.info("Setting up TPU mesh...")
    mesh_config = config.get('mesh', {})
    topology = tuple(mesh_config.get('mesh_shape', [4, 4, 2]))
    axis_names = tuple(mesh_config.get('axis_names', ['data', 'model', 'fsdp']))
    mesh = make_mesh(topology=topology, axis_names=axis_names)
    
    # Setup mixed precision policy
    logger.info("Setting up mixed precision policy...")
    mp_config = get_tpu_mixed_precision_config()
    mixed_precision = MixedPrecisionPolicy(mp_config)
    
    # Setup data pipeline
    logger.info("Setting up data pipeline...")
    
    # For now, use simple data loader since multi-source components are not available
    # TODO: Implement full multi-source pipeline when components are available
    tokenizer_config = TokenizerConfig(
        tokenizer_name=config['data']['tokenizer_name'],
        vocab_size=config['data']['vocab_size']
    )
    
    # Create FineWeb config for simple data loader testing
    from src.data.fineweb_reader import FineWebConfig
    fineweb_config = FineWebConfig(
        dataset_name="HuggingFaceFW/fineweb",
        dataset_config="sample-10BT",  # Use smaller sample for testing
        batch_size=config['training']['global_batch_size'],
        chunk_size=config['model']['max_position_embeddings'],
        streaming=True
    )
    
    # Create simple data loader for testing
    data_loader = create_data_loader(
        config=fineweb_config,
        tokenizer_config=tokenizer_config
    )
    
    # Setup checkpointing
    logger.info("Setting up checkpointing...")
    checkpoint_config = CheckpointConfig(**config['checkpointing'])
    checkpoint_manager = CheckpointManager(
        config=checkpoint_config,
        mesh=mesh,
        partition_specs=get_model_specs(model.config, use_2d_sharding=False),
    )
    
    # Setup training loop
    logger.info("Setting up training loop...")
    # Create ChunkConfig from training parameters
    chunk_config = ChunkConfig(
        chunk_size=config['training'].get('chunk_size', 8192),
        overlap_size=config['training'].get('overlap_size', 512),
        max_chunks_per_doc=config['training'].get('max_chunks_per_doc', 82),
        backprop_chunks=config['training'].get('backprop_chunks', 4),
        long_backprop_every=config['training'].get('long_backprop_every', 100),
        long_backprop_chunks=config['training'].get('long_backprop_chunks', 16)
    )
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


def run_tpu_v4_32_test(model: ValkyrieModel, training_state, config: Dict[str, Any], mesh) -> bool:
    """Run comprehensive TPU v4-32 validation test with enhanced functionality from examples."""
    
    logger.info("=== TPU v4-32 Comprehensive Validation Test ===")
    
    try:
        from src.train.step_fn import create_train_step, create_eval_step
        from src.sharding.distributed_init import print_distributed_info
        
        # Step 1: Validate distributed setup
        logger.info("=== Step 1: Distributed Setup Validation ===")
        print_distributed_info()
        
        # Step 2: Validate mesh configuration
        logger.info("=== Step 2: Mesh Configuration Validation ===")
        expected_shape = tuple(config['mesh']['mesh_shape'])  # (4, 4)
        expected_axes = tuple(config['mesh']['axis_names'])   # ('data', 'model')
        
        # Convert mesh.shape (OrderedDict) to tuple for comparison
        actual_shape = tuple(mesh.shape.values())
        
        if actual_shape != expected_shape:
            raise ValueError(f"Expected mesh shape {expected_shape}, got {actual_shape}")
        if mesh.axis_names != expected_axes:
            raise ValueError(f"Expected axis names {expected_axes}, got {mesh.axis_names}")
        
        logger.info(f"✓ Mesh configuration verified: shape={actual_shape}, axes={mesh.axis_names}")
        
        # Step 3: Create enhanced synthetic batch
        logger.info("=== Step 3: Synthetic Batch Creation ===")
        batch_size = config['training']['global_batch_size']  # Use global batch size
        seq_length = min(config['data']['max_length'], 2048)  # Cap for testing
        vocab_size = config['model']['vocab_size']
        
        # Create synthetic data with proper attention mask
        rng = jax.random.PRNGKey(42)
        input_ids = jax.random.randint(
            rng, (batch_size, seq_length), 0, vocab_size, dtype=jnp.int32
        )
        attention_mask = jnp.ones((batch_size, seq_length), dtype=jnp.bool_)
        labels = input_ids  # For language modeling
        
        batch = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
        }
        
        # Verify batch sharding
        with mesh:
            local_batch_size = batch_size // mesh.shape['data']  # Divide by data parallel dimension
            logger.info(f"Global batch size: {batch_size}, Local batch size: {local_batch_size}")
            logger.info(f"Batch shapes: input_ids={batch['input_ids'].shape}")
        
        # Step 4: Count model parameters
        logger.info("=== Step 4: Model Parameter Analysis ===")
        param_count = sum(x.size for x in jax.tree_util.tree_leaves(training_state.params))
        logger.info(f"Model parameters: {param_count:,} ({param_count/1e9:.2f}B)")
        
        # Step 5: Create step functions with enhanced configuration
        logger.info("=== Step 5: Step Function Creation ===")
        with mesh:
            # Create training step function with 3D mesh support
            train_step_fn = create_train_step(
                model=model,
                optimizer=None,  # We'll pass the optimizer separately since it's in opt_state
                config=model.config,
                mesh=mesh,
                mixed_precision=config['training'].get('use_mixed_precision', True),
                use_2d_sharding=config['sharding'].get('model_parallel', {}).get('type') == '2d_tensor_parallel',
                use_3d_mesh=config['sharding'].get('use_3d_sharding', True)
            )
            
            # Create evaluation step function
            eval_step_fn = create_eval_step(
                model=model,
                config=model.config,
                mesh=mesh,
                mixed_precision=config['training'].get('use_mixed_precision', True),
                use_2d_sharding=config['sharding'].get('model_parallel', {}).get('type') == '2d_tensor_parallel',
                use_3d_mesh=config['sharding'].get('use_3d_sharding', True)
            )
        
        # Step 6: Single training step test with timing
        logger.info("=== Step 6: Single Training Step Test ===")
        with mesh:
            start_time = time.time()
            # Generate dropout RNG key
            dropout_rng = jax.random.fold_in(training_state.rng, training_state.step)
            new_state, train_metrics = train_step_fn(training_state, batch, dropout_rng)
            step_time = time.time() - start_time
            
            # Add timing metrics
            train_metrics['step_time_ms'] = step_time * 1000
            train_metrics['tokens_per_second'] = (batch['input_ids'].size / step_time)
            
            logger.info("Training step completed successfully!")
            logger.info(f"  Loss: {train_metrics.get('loss', 'N/A'):.4f}")
            logger.info(f"  Learning rate: {train_metrics.get('learning_rate', 'N/A')}")
            logger.info(f"  Step time: {train_metrics.get('step_time_ms', 'N/A'):.1f}ms")
            logger.info(f"  Tokens/sec: {train_metrics.get('tokens_per_second', 'N/A'):.0f}")
        
        # Step 7: Evaluation step test with timing
        logger.info("=== Step 7: Evaluation Step Test ===")
        with mesh:
            start_time = time.time()
            eval_metrics = eval_step_fn(new_state, batch)
            eval_time = time.time() - start_time
            
            eval_metrics['eval_time_ms'] = eval_time * 1000
            
            logger.info("Evaluation step completed successfully!")
            logger.info(f"  Eval loss: {eval_metrics.get('loss', 'N/A'):.4f}")
            logger.info(f"  Accuracy: {eval_metrics.get('accuracy', 'N/A'):.4f}")
            logger.info(f"  Perplexity: {eval_metrics.get('perplexity', 'N/A'):.2f}")
            logger.info(f"  Eval time: {eval_metrics.get('eval_time_ms', 'N/A'):.1f}ms")
        
        # Step 8: Multi-step training test
        logger.info("=== Step 8: Multi-Step Training Test ===")
        num_steps = 5
        
        current_state = new_state
        step_times = []
        
        for step in range(num_steps):
            with mesh:
                start_time = time.time()
                current_state, step_metrics = train_step_fn(current_state, batch)
                step_time = time.time() - start_time
                step_times.append(step_time)
                
                logger.info(f"  Step {step+1}/{num_steps}: "
                           f"loss={step_metrics.get('loss', 'N/A'):.4f}, "
                           f"time={step_time*1000:.1f}ms")
        
        # Calculate average step time
        avg_step_time = sum(step_times) / len(step_times)
        logger.info(f"Multi-step training completed! Average step time: {avg_step_time*1000:.1f}ms")
        
        # Step 9: Performance summary
        logger.info("=== Step 9: Performance Summary ===")
        final_metrics = {
            'mesh_shape': mesh.shape,
            'mesh_axes': mesh.axis_names,
            'model_params': param_count,
            'global_batch_size': batch_size,
            'local_batch_size': batch_size // mesh.shape[0],
            'sequence_length': seq_length,
            'final_loss': step_metrics.get('loss', 'N/A'),
            'avg_step_time_ms': avg_step_time * 1000,
            'tokens_per_second': (batch['input_ids'].size / avg_step_time),
            'memory_efficient': config['sharding'].get('use_fsdp', False),
        }
        
        logger.info("=== FINAL TPU v4-32 TEST RESULTS ===")
        for key, value in final_metrics.items():
            logger.info(f"  {key}: {value}")
        
        logger.info("✓ TPU v4-32 comprehensive test completed successfully! 🎉")
        return True
            
    except Exception as e:
        import traceback
        logger.error(f"TPU v4-32 test failed: {e}")
        logger.error(f"Full traceback: {traceback.format_exc()}")
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
    parser.add_argument("--tpu_v4_32_test", action="store_true", help="Run TPU v4-32 specific validation test")
    parser.add_argument("--synthetic_data", action="store_true", help="Use synthetic data for testing")
    
    args = parser.parse_args()
    
    # Load configuration
    try:
        config = load_config(args.config)
    except Exception as e:
        print(f"❌ Failed to load configuration: {e}")
        return 1
    
    # Initialize JAX distributed system early (required for multi-host TPU)
    try:
        jax.distributed.initialize()
        print(f"✓ JAX distributed initialized - Process {jax.process_index()}/{jax.process_count()}")
    except Exception as e:
        print(f"⚠️  JAX distributed initialization failed: {e}")
        # Continue anyway for single-host setups
    
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
        
        # Run TPU v4-32 specific test if requested
        if args.tpu_v4_32_test:
            logger.info("Running TPU v4-32 specific validation test...")
            success = run_tpu_v4_32_test(model, training_state, config, components['mesh'])
            if success:
                logger.info("TPU v4-32 test completed successfully")
            else:
                logger.error("TPU v4-32 test failed")
                return 1
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
        # dataset_info = components['data_loader'].get_stats()  # Method doesn't exist, skip for now
#         dataset_info = {"sources": len(components['data_loader'].sources)}  # Simple fallback
#         multi_logger.log_training_start(total_steps, dataset_info)
        
        # Main training loop
        logger.info("Starting training...")
        start_time = time.time()
        
        try:
            # Create data iterator with batch conversion
            batch_size = config['training']['micro_batch_size']
            rng_key = jax.random.PRNGKey(42)  # Default seed since not in config
            data_iterator = create_batch_iterator(components['data_loader'])
            
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
            import traceback
            import sys
            
            # Get full traceback information
            exc_type, exc_value, exc_traceback = sys.exc_info()
            
            logger.error("=" * 80)
            logger.error("TRAINING FAILED - DETAILED ERROR INFORMATION")
            logger.error("=" * 80)
            logger.error(f"Exception Type: {exc_type.__name__}")
            logger.error(f"Exception Message: {str(e)}")
            logger.error("=" * 80)
            logger.error("FULL TRACEBACK:")
            logger.error("=" * 80)
            
            # Print full traceback with line numbers and context
            tb_lines = traceback.format_exception(exc_type, exc_value, exc_traceback)
            for line in tb_lines:
                logger.error(line.rstrip())
            
            logger.error("=" * 80)
            logger.error("STACK TRACE WITH LOCAL VARIABLES:")
            logger.error("=" * 80)
            
            # Print stack trace with local variables for debugging
            tb = exc_traceback
            while tb is not None:
                frame = tb.tb_frame
                filename = frame.f_code.co_filename
                line_no = tb.tb_lineno
                func_name = frame.f_code.co_name
                
                logger.error(f"File: {filename}")
                logger.error(f"Function: {func_name}")
                logger.error(f"Line: {line_no}")
                
                # Print local variables (be careful with large objects)
                logger.error("Local variables:")
                for var_name, var_value in frame.f_locals.items():
                    try:
                        # Limit string representation to avoid huge outputs
                        var_str = str(var_value)
                        if len(var_str) > 200:
                            var_str = var_str[:200] + "... (truncated)"
                        logger.error(f"  {var_name}: {var_str}")
                    except Exception as var_error:
                        logger.error(f"  {var_name}: <Error getting value: {var_error}>")
                
                logger.error("-" * 40)
                tb = tb.tb_next
            
            logger.error("=" * 80)
            
            # Save emergency checkpoint
            try:
                logger.info("Saving emergency checkpoint...")
                components['checkpoint_manager'].save(training_state, checkpoint_type="emergency")
                logger.info("Emergency checkpoint saved successfully")
            except Exception as checkpoint_error:
                logger.error(f"Failed to save emergency checkpoint: {checkpoint_error}")
                logger.error(f"Checkpoint error traceback: {traceback.format_exc()}")
            
            return 1
    
    except Exception as e:
        import traceback
        logger.error(f"Setup failed: {e}")
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return 1
    
    finally:
        # Cleanup
        if 'multi_logger' in locals():
            multi_logger.close()


if __name__ == "__main__":
    exit(main())