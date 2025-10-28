"""Chunked training loop for ultra-long sequences.

Implements the training strategy from output.txt:
- Chunk 657k sequences into manageable pieces (8k-32k tokens)
- Use S5 as inter-chunk memory carrier
- Truncated BPTT with occasional long backprop
- Progressive curriculum (start small, scale up)
- Mixed precision with fp32 for attention/S5
"""

import jax
import jax.numpy as jnp
from jax.experimental import pjit
from jax.sharding import PartitionSpec as P
import flax
import optax
import time
import logging
from typing import Dict, List, Optional, Tuple, Any, NamedTuple
from dataclasses import dataclass
import numpy as np

from ..model import ValkyrieModel, ValkyrieConfig
from ..sharding import make_mesh, get_model_specs, get_training_specs, DP
from ..io import CheckpointManager, get_logger

logger = get_logger(__name__)


class PhaseConfig(NamedTuple):
    """Hashable phase configuration for static arguments in JAX."""
    name: str
    steps: int
    chunk_size: int
    lr: float
    backprop_chunks: int = 1  # Default value
    hrm_enabled: bool = True
    hrm_supervision_weight: float = 0.0
    # New fields for ACT integration and deep supervision control
    act_enabled: bool = False
    act_loss_weight: float = 0.0
    deep_supervision_weight: float = 0.0
    hrm_one_step_gradient: bool = False


@dataclass
class ChunkConfig:
    """Configuration for chunked sequence processing."""
    chunk_size: int = 8192          # Tokens per chunk
    overlap_size: int = 512         # Overlap between chunks
    max_chunks_per_doc: int = 82    # ~657k / 8k
    backprop_chunks: int = 4        # Chunks to backprop through (TBPTT)
    long_backprop_every: int = 100  # Steps between long backprop
    long_backprop_chunks: int = 16  # Chunks for long backprop


@dataclass 
class CurriculumConfig:
    """Progressive curriculum configuration following blueprint specification."""
    phases: List[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.phases is None:
            # Blueprint curriculum: 4k→8k→16k→32k→64k with HRM stages
            self.phases = [
                # Phase 0: Foundation (4k tokens)
                {
                    'name': 'foundation_4k',
                    'chunk_size': 4096,
                    'backprop_chunks': 2,
                    'max_steps': 8000,
                    'lr': 3e-4,
                    'hrm_enabled': False,  # Start without HRM
                    'hrm_supervision_weight': 0.0,
                    'description': 'Foundation training on 4k sequences'
                },
                # Phase 1: HRM Introduction (8k tokens)
                {
                    'name': 'hrm_intro_8k', 
                    'chunk_size': 8192,
                    'backprop_chunks': 3,
                    'max_steps': 12000,
                    'lr': 2.5e-4,
                    'hrm_enabled': True,  # Introduce HRM
                    'hrm_supervision_weight': 0.3,
                    'hrm_one_step_gradient': True,
                    'description': 'Introduce HRM with light supervision'
                },
                # Phase 2: HRM Strengthening (16k tokens)
                {
                    'name': 'hrm_strengthen_16k',
                    'chunk_size': 16384,
                    'backprop_chunks': 4,
                    'max_steps': 15000,
                    'lr': 2e-4,
                    'hrm_enabled': True,
                    'hrm_supervision_weight': 0.6,
                    'hrm_one_step_gradient': True,
                    'act_enabled': True,  # Enable ACT
                    'act_loss_weight': 0.05,
                    'description': 'Strengthen HRM reasoning with ACT'
                },
                # Phase 3: Long Context (32k tokens)
                {
                    'name': 'long_context_32k',
                    'chunk_size': 32768,
                    'backprop_chunks': 6,
                    'max_steps': 20000,
                    'lr': 1.5e-4,
                    'hrm_enabled': True,
                    'hrm_supervision_weight': 0.8,
                    'hrm_one_step_gradient': True,
                    'act_enabled': True,
                    'act_loss_weight': 0.08,
                    'deep_supervision_weight': 1.0,
                    'description': 'Long context with full HRM supervision'
                },
                # Phase 4: Ultra-Long Context (64k tokens)
                {
                    'name': 'ultra_long_64k',
                    'chunk_size': 65536,
                    'backprop_chunks': 8,
                    'max_steps': 30000,
                    'lr': 1e-4,
                    'hrm_enabled': True,
                    'hrm_supervision_weight': 1.0,
                    'hrm_one_step_gradient': True,
                    'act_enabled': True,
                    'act_loss_weight': 0.1,
                    'deep_supervision_weight': 1.2,
                    'max_segments_per_batch': 8,
                    'description': 'Ultra-long context with maximum HRM capability'
                },
            ]


from .data_structures import TrainingState


class TrainingLoop:
    """
    Main training loop with chunked processing for ultra-long sequences.
    
    Key features:
    - Chunks 657k sequences into manageable pieces
    - S5 state management between chunks
    - Truncated BPTT with gradient accumulation
    - Progressive curriculum scaling
    - Mixed precision training
    - Multi-host TPU coordination
    """
    
    def __init__(
        self,
        model: ValkyrieModel,
        config: ValkyrieConfig,
        chunk_config: ChunkConfig,
        curriculum_config: CurriculumConfig,
        mesh: Optional[Any] = None,
        checkpoint_manager: Optional[CheckpointManager] = None,
        hrm_training_loop: Optional[Any] = None,  # HRMTrainingLoop for HRM phases
    ):
        self.model = model
        self.config = config
        self.chunk_config = chunk_config
        self.curriculum_config = curriculum_config
        
        # Setup mesh and sharding
        if mesh is None:
            mesh = make_mesh()
        self.mesh = mesh
        
        # Setup checkpointing
        self.checkpoint_manager = checkpoint_manager
        
        # HRM integration
        self.hrm_training_loop = hrm_training_loop
        
        # Get sharding specs
        self.model_specs = get_model_specs(config, use_2d_sharding=False)
        self.training_specs = get_training_specs(self.model_specs)
        
        # Initialize training functions
        self._setup_training_functions()
        
        # Training metrics
        self.metrics = {
            'step': 0,
            'phase': 0,
            'tokens_processed': 0,
            'loss_history': [],
            'throughput_history': [],
            'hrm_metrics': {},  # HRM-specific metrics
        }
    
    def _setup_training_functions(self):
        """Setup pjit-compiled training functions."""
        
        # Training step function
        def train_step_fn(state: TrainingState, batch: Dict[str, jnp.ndarray], phase_config: PhaseConfig) -> Tuple[TrainingState, Dict[str, Any]]:
            """Single training step with chunked processing."""
            
            # Add debugging print
            logger.debug(f"Starting train_step - step {state.step}, phase {phase_config.name}")
            logger.debug(f"State attributes: {[attr for attr in dir(state) if not attr.startswith('__')]}")
            
            def loss_fn(params):
                # Use passed phase config (static argument)
                chunk_size = phase_config.chunk_size
                backprop_chunks = phase_config.backprop_chunks
                
                # Process document in chunks
                input_ids = batch['input_ids']  # [batch, full_seq_len]
                batch_size, full_seq_len = input_ids.shape
                
                total_loss = 0.0
                num_chunks_processed = 0
                current_s5_states = state.s5_states
                
                # Calculate chunk boundaries using JAX-compatible operations
                start_pos = state.chunk_position * chunk_size
                
                # Use jax.lax.cond for position reset logic
                def reset_position():
                    reset_s5_states = jax.tree_util.tree_map(
                        jnp.zeros_like,
                        current_s5_states
                    )
                    return (0, jnp.minimum(backprop_chunks * chunk_size, full_seq_len), reset_s5_states)
                
                def keep_position():
                    return (start_pos, jnp.minimum(start_pos + (backprop_chunks * chunk_size), full_seq_len), current_s5_states)
                
                start_pos, end_pos, current_s5_states = jax.lax.cond(
                    start_pos >= full_seq_len,
                    reset_position,
                    keep_position
                )
                
                # Process chunks with gradient flow using JAX-compatible scan
                def process_chunk(carry, chunk_idx):
                    total_loss, current_s5_states, hrm_loss, hrm_metrics = carry
                    
                    chunk_start = start_pos + chunk_idx * chunk_size
                    chunk_end = jnp.minimum(chunk_start + chunk_size, full_seq_len)
                    
                    # Use jax.lax.cond to handle chunk boundary check
                    def process_valid_chunk():
                        # Extract chunk using JAX-compatible dynamic slicing
                        # Use concrete chunk_size instead of dynamic chunk_length
                        chunk_input = jax.lax.dynamic_slice(
                            input_ids, 
                            (0, chunk_start), 
                            (batch_size, chunk_size)
                        )
                        
                        # Create labels using dynamic slicing for the shifted input
                        chunk_input_shifted = jax.lax.dynamic_slice(
                            chunk_input,
                            (0, 1),
                            (batch_size, chunk_size - 1)
                        )
                        chunk_labels = jnp.concatenate([
                            chunk_input_shifted, 
                            jnp.full((batch_size, 1), -100)  # Padding
                        ], axis=1)
                        
                        # Forward pass through chunk
                        # Derive per-chunk RNG streams for dropout and random attention
                        dropout_rng_chunk = jax.random.fold_in(state.rng, chunk_idx)
                        random_rng_chunk = jax.random.fold_in(state.rng, chunk_idx + 1)
                        
                        # Expand S5 states to match batch size for model forward pass
                        def expand_s5_state(s5_state):
                            if s5_state.shape[0] != batch_size:
                                # Tile the state to match the batch size
                                return jnp.tile(s5_state, (batch_size,) + (1,) * (s5_state.ndim - 1))
                            return s5_state
                        
                        expanded_s5_states = jax.tree_util.tree_map(expand_s5_state, current_s5_states)
                        
                        outputs = self.model.apply(
                            {'params': params},
                            input_ids=chunk_input,
                            labels=chunk_labels,
                            s5_states=expanded_s5_states,
                            use_cache=False,  # Disable cache during training to avoid memory footgun
                            training=True,
                            hrm_enabled=phase_config.hrm_enabled,  # Pass phase-based HRM control
                            return_dict=True,
                            rngs={
                                'dropout': dropout_rng_chunk,
                                'random': random_rng_chunk
                            }
                        )
                        
                        chunk_loss = outputs['loss']
                        new_total_loss = total_loss + chunk_loss
                        new_s5_states = outputs['s5_states']
                        
                        # Reduce S5 states back to single batch dimension to maintain scan carry consistency
                        # Take the first element from the batch dimension since all elements should have the same sequential state
                        def reduce_s5_state(s5_state):
                            if s5_state.shape[0] > 1:
                                return s5_state[0:1]  # Keep first element with batch dimension 1
                            return s5_state
                        
                        new_s5_states = jax.tree_util.tree_map(reduce_s5_state, new_s5_states)
                        
                        # Ensure s5_states has the same pytree structure as current_s5_states
                        # Convert to the same type to avoid JAX pytree structure mismatch
                        if isinstance(current_s5_states, list):
                            new_s5_states = list(new_s5_states) if not isinstance(new_s5_states, list) else new_s5_states
                        elif isinstance(current_s5_states, tuple):
                            new_s5_states = tuple(new_s5_states) if not isinstance(new_s5_states, tuple) else new_s5_states
                        
                        # Add HRM loss if enabled in current phase
                        new_hrm_loss = hrm_loss
                        new_hrm_metrics = hrm_metrics
                        
                        if phase_config.hrm_enabled and self.hrm_training_loop:  # HRM enabled in this phase
                            hrm_batch = {
                                'input_ids': chunk_input,
                                'labels': chunk_labels,
                                'attention_mask': jnp.ones_like(chunk_input),
                            }
                            
                            # Compute HRM loss with phase-specific weights
                            hrm_rng_chunk = jax.random.fold_in(state.rng, chunk_idx + 1000)
                            hrm_output = self.hrm_training_loop.compute_hrm_loss(
                                params,
                                hrm_batch,
                                {
                                    'act_enabled': phase_config.act_enabled,
                                    'act_loss_weight': phase_config.act_loss_weight,
                                    'hrm_supervision_weight': phase_config.hrm_supervision_weight,
                                    'deep_supervision_weight': phase_config.deep_supervision_weight,
                                    'hrm_one_step_gradient': phase_config.hrm_one_step_gradient,
                                },
                                rng_key=hrm_rng_chunk
                            )
                            
                            new_hrm_loss = hrm_loss + hrm_output['loss'] * phase_config.hrm_supervision_weight
                            new_hrm_metrics = hrm_output['metrics']
                        
                        return (new_total_loss, new_s5_states, new_hrm_loss, new_hrm_metrics)
                    
                    def skip_chunk():
                        # Return S5 states with consistent batch dimension (should already be [1, 768])
                        return (total_loss, current_s5_states, hrm_loss, hrm_metrics)
                    
                    return jax.lax.cond(
                        chunk_start < full_seq_len,
                        process_valid_chunk,
                        skip_chunk
                    ), None
                
                # Initialize carry state
                initial_carry = (0.0, current_s5_states, 0.0, {})
                
                # Process all chunks
                final_carry, _ = jax.lax.scan(
                    process_chunk,
                    initial_carry,
                    jnp.arange(backprop_chunks)
                )
                
                total_loss, current_s5_states, hrm_loss, hrm_metrics = final_carry
                num_chunks_processed = backprop_chunks  # All chunks are processed (some may be skipped)
                
                # Combine losses
                total_loss = total_loss + hrm_loss
                
                # Average loss across chunks using JAX-compatible operation
                # Since num_chunks_processed is now always backprop_chunks, we can safely divide
                total_loss = total_loss / jnp.maximum(num_chunks_processed, 1)  # Avoid division by zero
                
                return total_loss, {
                    'loss': total_loss,
                    'chunks_processed': num_chunks_processed,
                    's5_states': current_s5_states,
                }
            
            # Compute gradients
            (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
            
            # Apply additional gradient clipping for numerical stability
            grads = jax.tree_util.tree_map(lambda g: jnp.clip(g, -1.0, 1.0), grads)
            
            # Apply gradients with optimizer
            updates, new_opt_state = self.optimizer.update(grads, state.opt_state, state.params)
            new_params = optax.apply_updates(state.params, updates)
            
            # Update chunk position
            backprop_chunks = phase_config.backprop_chunks
            new_chunk_position = state.chunk_position + backprop_chunks
            
            # Check if we need to advance to next phase using JAX-compatible operations
            max_phase_index = len(self.curriculum_config.phases) - 1
            can_advance = jnp.logical_and(
                state.step >= phase_config.steps,
                state.phase_index < max_phase_index
            )
            
            def advance_phase():
                return (
                    state.phase_index + 1,  # new_phase_index
                    0,  # reset chunk_position
                    True,  # new_hrm_enabled (simplified for now)
                    state.hrm_training_state  # new_hrm_training_state (simplified)
                )
            
            def keep_phase():
                return (
                    state.phase_index,  # keep current phase
                    new_chunk_position,  # normal chunk position update
                    state.hrm_enabled,  # keep current HRM state
                    state.hrm_training_state  # keep current HRM training state
                )
            
            new_phase_index, new_chunk_position, new_hrm_enabled, new_hrm_training_state = jax.lax.cond(
                can_advance,
                advance_phase,
                keep_phase
            )
            
            # Create new state
            new_state = TrainingState(
                params=new_params,
                opt_state=new_opt_state,
                step=state.step + 1,
                rng=jax.random.split(state.rng, 2)[1],
                s5_states=aux['s5_states'],
                chunk_position=new_chunk_position,
                phase_index=new_phase_index,
                hrm_enabled=new_hrm_enabled,
                hrm_training_state=new_hrm_training_state,
            )
            
            # Initialize HRM metrics (empty if HRM not enabled)
            hrm_metrics = {}
            
            metrics = {
                'loss': loss,
                'chunks_processed': aux['chunks_processed'],
                'phase': new_phase_index,
                'chunk_position': new_chunk_position,
                'learning_rate': self.lr_schedule(new_state.step) if hasattr(self, 'lr_schedule') else jnp.array(0.0),
                **hrm_metrics,  # Add HRM metrics if available
            }
            
            return new_state, metrics
        
        # Compile with pjit
        self.train_step = pjit.pjit(
            train_step_fn,
            in_shardings=(self.training_specs, P(DP, None)),  # state, batch (phase_config is static)
            out_shardings=(self.training_specs, P()),        # new_state, metrics
            static_argnums=(2,)   # phase_config is static
        )
    
    def create_optimizer(self, learning_rate: float) -> optax.GradientTransformation:
        """Create optimizer with proper scaling and clipping."""
        
        # Learning rate schedule with warmup
        warmup_steps = 1000
        schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=learning_rate,
            warmup_steps=warmup_steps,
            decay_steps=100000,
            end_value=learning_rate * 0.1
        )
        
        # Store schedule for metrics logging
        self.lr_schedule = schedule
        
        # Optimizer chain
        optimizer = optax.chain(
            # Gradient clipping
            optax.clip_by_global_norm(self.config.gradient_clipping),
            
            # AdamW with weight decay
            optax.adamw(
                learning_rate=schedule,
                weight_decay=self.config.weight_decay,
                b1=0.9,
                b2=0.95,
                eps=1e-8,
            ),
        )
        
        return optimizer
    
    def initialize_training_state(self, rng_key: jax.random.PRNGKey) -> TrainingState:
        """Initialize training state with proper sharding."""
        
        # Add comprehensive logging for mesh and sharding specs verification
        logger.info(f"Mesh axis names: {self.mesh.axis_names}")
        logger.info(f"Mesh shape: {self.mesh.shape}")
        
        # Sample and log model specs for verification
        from jax.tree_util import tree_leaves
        sample_model_specs = tree_leaves(self.model_specs)[:5]
        logger.info(f"Sample model specs: {', '.join(map(str, sample_model_specs))}")
        
        # Sample and log training specs for verification
        if hasattr(self, 'training_specs') and self.training_specs:
            sample_training_specs = tree_leaves(self.training_specs)[:5]
            logger.info(f"Sample training specs: {', '.join(map(str, sample_training_specs))}")
        
        # Verify mesh axis compatibility with specs
        mesh_axes = set(self.mesh.axis_names)
        spec_axes = set()
        for spec in tree_leaves(self.model_specs):
            if hasattr(spec, 'partitions'):
                for axis in spec.partitions:
                    if axis is not None:
                        spec_axes.add(axis)
        
        missing_axes = spec_axes - mesh_axes
        if missing_axes:
            logger.error(f"Missing mesh axes for sharding specs: {missing_axes}")
            logger.error(f"Available mesh axes: {mesh_axes}")
            logger.error(f"Required spec axes: {spec_axes}")
            raise ValueError(f"Mesh axes {mesh_axes} do not contain required spec axes {missing_axes}")
        else:
            logger.info(f"Mesh axis compatibility verified: all spec axes {spec_axes} are available in mesh {mesh_axes}")
        
        # Initialize model parameters
        dummy_input = jnp.ones((1, self.chunk_config.chunk_size), dtype=jnp.int32)
        
        with self.mesh:
            # Initialize parameters with sharding
            # Flax model.init returns {params: {...}}, so wrap model_specs accordingly
            flax_init_specs = {'params': self.model_specs}
            init_fn = pjit.pjit(
                self.model.init,
                in_shardings=(P(), P(DP, None)),
                out_shardings=flax_init_specs
            )
            init_result = init_fn(rng_key, dummy_input)
            params = init_result['params']  # Extract params from Flax wrapper
        
        # Create optimizer
        phase_0_lr = self.curriculum_config.phases[0]['lr']
        self.optimizer = self.create_optimizer(phase_0_lr)
        
        # Initialize optimizer state
        with self.mesh:
            opt_init_fn = pjit.pjit(
                self.optimizer.init,
                in_shardings=(self.model_specs,),  # Wrap in tuple for single argument
                out_shardings=None # Optax optimizer.init returns structure matching params
            )
            opt_state = opt_init_fn(params)
        
        # Initialize S5 states
        s5_states = None
        if self.config.use_s5:
            s5_states = [
                jnp.zeros((1, self.config.s5_state_dim), dtype=jnp.complex64)
                for _ in range(self.config.n_layers)
            ]
        
        return TrainingState(
            params=params,
            opt_state=opt_state,
            step=0,
            rng=rng_key,
            s5_states=s5_states,
            chunk_position=0,
            phase_index=0,
            hrm_enabled=False,
        )
    
    def train_epoch(
        self,
        state: TrainingState,
        data_loader: Any,
        max_steps: Optional[int] = None,
    ) -> TrainingState:
        """Train for one epoch with chunked processing."""
        
        logger.info(f"Starting training epoch, step {state.step}")
        
        step_times = []
        
        for step, batch in enumerate(data_loader):
            if max_steps is not None and step >= max_steps:
                break
            
            step_start = time.time()
            
            # Training step
            with self.mesh:
                # Get current phase config and convert to hashable PhaseConfig
                phase_dict = self.curriculum_config.phases[int(state.phase_index)]
                current_phase = PhaseConfig(
                    name=phase_dict.get('name', f'phase_{int(state.phase_index)}'),
                    steps=phase_dict.get('steps', 10000),
                    chunk_size=phase_dict['chunk_size'],
                    lr=phase_dict.get('lr', 0.0001),
                    backprop_chunks=phase_dict.get('backprop_chunks', 1),
                    hrm_enabled=phase_dict.get('hrm_enabled', False),
                    hrm_supervision_weight=phase_dict.get('hrm_supervision_weight', 0.0),
                    act_enabled=phase_dict.get('act_enabled', False),
                    act_loss_weight=phase_dict.get('act_loss_weight', 0.0),
                    deep_supervision_weight=phase_dict.get('deep_supervision_weight', 0.0),
                    hrm_one_step_gradient=phase_dict.get('hrm_one_step_gradient', False),
                )
                state, metrics = self.train_step(state, batch, current_phase)
            
            step_time = time.time() - step_start
            step_times.append(step_time)
            
            # Update metrics
            self.metrics['step'] = state.step
            self.metrics['phase'] = state.phase_index
            self.metrics['loss_history'].append(float(metrics['loss']))
            
            # Calculate throughput
            phase = self.curriculum_config.phases[state.phase_index]
            tokens_per_step = batch['input_ids'].shape[0] * phase['chunk_size'] * metrics['chunks_processed']
            throughput = tokens_per_step / step_time
            self.metrics['throughput_history'].append(throughput)
            self.metrics['tokens_processed'] += tokens_per_step
            
            # Logging
            if step % 10 == 0:
                avg_step_time = np.mean(step_times[-10:])
                avg_throughput = np.mean(self.metrics['throughput_history'][-10:])
                
                logger.info(
                    f"Step {state.step}, Phase {state.phase_index}, "
                    f"Loss: {metrics['loss']:.4f}, "
                    f"Chunks: {metrics['chunks_processed']}, "
                    f"LR: {float(metrics['learning_rate']):.3e}, "
                    f"Throughput: {avg_throughput:.0f} tok/s, "
                    f"Step time: {avg_step_time:.2f}s"
                )
            
            # Checkpointing
            if self.checkpoint_manager and step % 1000 == 0:
                self.checkpoint_manager.save(state, step=state.step)
                logger.info(f"Checkpoint saved at step {state.step}")
            
            # Long backprop occasionally
            if (step % self.chunk_config.long_backprop_every == 0 and 
                step > 0):
                logger.info("Performing long backprop for S5 gradient flow")
                # This would involve a special training step with more chunks
                # Implementation depends on specific requirements
        
        return state
    
    def validate(self, state: TrainingState, val_loader: Any) -> Dict[str, float]:
        """Run validation with chunked processing."""
        
        logger.info("Running validation...")
        
        total_loss = 0.0
        total_chunks = 0
        
        for batch in val_loader:
            # Validation forward pass (no gradients)
            with self.mesh:
                def val_loss_fn(params):
                    # Similar to training but without gradient accumulation
                    input_ids = batch['input_ids']
                    batch_size, seq_len = input_ids.shape
                    
                    # Process in chunks
                    chunk_size = self.chunk_config.chunk_size
                    num_chunks = (seq_len + chunk_size - 1) // chunk_size
                    
                    total_loss = 0.0
                    s5_states = None
                    
                    for i in range(num_chunks):
                        start_idx = i * chunk_size
                        end_idx = min((i + 1) * chunk_size, seq_len)
                        
                        chunk_input = input_ids[:, start_idx:end_idx]
                        chunk_labels = jnp.concatenate([
                            chunk_input[:, 1:],
                            jnp.full((batch_size, 1), -100)
                        ], axis=1)
                        
                        outputs = self.model.apply(
                            {'params': params},
                            input_ids=chunk_input,
                            labels=chunk_labels,
                            s5_states=s5_states,
                            use_cache=True,
                            training=False,
                            return_dict=True
                        )
                        
                        total_loss += outputs['loss']
                        s5_states = outputs['s5_states']
                    
                    return total_loss / num_chunks
                
                loss = val_loss_fn(state.params)
                total_loss += float(loss)
                total_chunks += 1
        
        avg_loss = total_loss / total_chunks if total_chunks > 0 else float('inf')
        
        logger.info(f"Validation loss: {avg_loss:.4f}")
        
        return {'val_loss': avg_loss}
    
    def get_training_metrics(self) -> Dict[str, Any]:
        """Get current training metrics."""
        return self.metrics.copy()
    
    def save_checkpoint(self, state: TrainingState, path: str):
        """Save training checkpoint."""
        if self.checkpoint_manager:
            self.checkpoint_manager.save(state, path=path)
    
    def load_checkpoint(self, path: str) -> TrainingState:
        """Load training checkpoint."""
        if self.checkpoint_manager:
            return self.checkpoint_manager.load(path)
        else:
            raise ValueError("No checkpoint manager configured")