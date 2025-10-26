"""HRM Training Loop

Implements the HRM training loop with one-step gradient approach and deep supervision.
Follows the PLAN training strategy with segment-level losses and state detaches.

Key features:
- One-step gradient: T-1 no-grad steps + 1 grad step
- Segment-level deep supervision with detaches
- O(1) memory complexity w.r.t. inner steps
- Proper gradient flow through final step only
- ACT integration for adaptive computation
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training import train_state
import optax
from typing import Dict, List, Tuple, Optional, NamedTuple, Callable, Iterator
import logging
from dataclasses import dataclass

from ..model.gryphon.gryphon_hrm_model import GryphonHRMModel, GryphonHRMState
from ..model.hrm.models.hrm_inner import HRMInnerCarry
from ..model.hrm.models.hrm_act import ACTOutput, compute_act_loss
from ..data.phase_sampler import PackedSequence

logger = logging.getLogger(__name__)


class HRMTrainingState(NamedTuple):
    """Training state for HRM model."""
    model_state: train_state.TrainState
    hrm_carry: HRMInnerCarry
    s5_state: jnp.ndarray
    global_tokens: jnp.ndarray
    step: int
    phase: str


class HRMBatch(NamedTuple):
    """Batch for HRM training."""
    input_ids: jnp.ndarray  # [batch, seq_len]
    attention_mask: jnp.ndarray  # [batch, seq_len]
    labels: jnp.ndarray  # [batch, seq_len]
    doc_ids: jnp.ndarray  # [batch, seq_len]
    segment_ids: jnp.ndarray  # [batch, seq_len]
    linked_next: jnp.ndarray  # [batch] - boolean flags
    source_ids: jnp.ndarray  # [batch] - source identifiers


@dataclass
class HRMTrainingConfig:
    """Configuration for HRM training."""
    # HRM-specific parameters
    hrm_enabled: bool = True  # gate HRM per phase (PLAN Stage A off, Stage B/C on)
    use_one_step_gradient: bool = True
    detach_states: bool = True
    deep_supervision_weight: float = 1.0
    segment_loss_weights: Optional[List[float]] = None  # Weights for each segment
    
    # ACT parameters
    use_act: bool = False
    act_loss_weight: float = 0.1
    act_efficiency_weight: float = 0.01
    
    # Training parameters
    max_segments_per_batch: int = 4  # Maximum segments to process
    gradient_accumulation_steps: int = 1
    
    # Loss configuration
    label_smoothing: float = 0.0
    ignore_index: int = -100


class HRMTrainingLoop:
    """HRM training loop with one-step gradient and deep supervision."""
    
    def __init__(
        self,
        model: GryphonHRMModel,
        config: HRMTrainingConfig
    ):
        """Initialize HRM training loop."""
        self.model = model
        self.config = config
        
        # Compile training functions
        self._compile_training_functions()
        
        logger.info("Initialized HRM training loop")
        logger.info(f"  - One-step gradient: {config.use_one_step_gradient}")
        logger.info(f"  - Deep supervision: {config.deep_supervision_weight}")
        logger.info(f"  - ACT enabled: {config.use_act}")
    
    def _compile_training_functions(self):
        """Compile JAX training functions."""
        # Compile the main training step
        self.train_step_fn = jax.jit(self._train_step)
        
        # Compile evaluation step
        self.eval_step_fn = jax.jit(self._eval_step)
        
        logger.info("Compiled training functions")
    
    def _compute_language_modeling_loss(
        self,
        logits: jnp.ndarray,
        labels: jnp.ndarray,
        attention_mask: jnp.ndarray
    ) -> jnp.ndarray:
        """Compute language modeling loss."""
        # Shift labels for causal LM
        shift_logits = logits[..., :-1, :]
        shift_labels = labels[..., 1:]
        shift_mask = attention_mask[..., 1:]
        
        # Compute cross-entropy loss
        loss = optax.softmax_cross_entropy_with_integer_labels(
            shift_logits, shift_labels
        )
        
        # Apply label smoothing if specified
        if self.config.label_smoothing > 0:
            num_classes = shift_logits.shape[-1]
            smooth_loss = optax.smooth_labels(
                jax.nn.one_hot(shift_labels, num_classes),
                self.config.label_smoothing
            )
            loss = optax.softmax_cross_entropy(shift_logits, smooth_loss)
        
        # Mask out padding tokens
        loss = loss * shift_mask
        
        # Return mean loss over valid tokens
        return jnp.sum(loss) / jnp.sum(shift_mask)
    
    def _segment_batch(
        self,
        batch: HRMBatch,
        max_segments: int
    ) -> List[HRMBatch]:
        """Segment batch for deep supervision."""
        batch_size, seq_len = batch.input_ids.shape
        
        # For simplicity, split sequence into equal segments
        segment_length = seq_len // max_segments
        segments = []
        
        for i in range(max_segments):
            start_idx = i * segment_length
            end_idx = (i + 1) * segment_length if i < max_segments - 1 else seq_len
            
            segment = HRMBatch(
                input_ids=batch.input_ids[:, start_idx:end_idx],
                attention_mask=batch.attention_mask[:, start_idx:end_idx],
                labels=batch.labels[:, start_idx:end_idx],
                doc_ids=batch.doc_ids[:, start_idx:end_idx],
                segment_ids=batch.segment_ids[:, start_idx:end_idx],
                linked_next=batch.linked_next,  # Keep same linking info
                source_ids=batch.source_ids
            )
            segments.append(segment)
        
        return segments
    
    def _hrm_forward_with_one_step_gradient(
        self,
        params: Dict,
        model_state: GryphonHRMState,
        batch: HRMBatch,
        rng_key: jax.Array,
        training: bool = True
    ) -> Tuple[jnp.ndarray, GryphonHRMState, Dict]:
        """
        Forward pass with HRM one-step gradient logic.
        
        This implements the core HRM training approach:
        1. Run T-1 steps with no gradients (detached)
        2. Run final step with gradients
        3. Compute loss only on final step
        """
        if (not self.config.hrm_enabled) or (not self.config.use_one_step_gradient):
            # Standard forward pass (HRM disabled or one-step gradient disabled)
            # Derive rng for dropout only
            std_dropout_rng = jax.random.fold_in(rng_key, 0)
            apply_outputs = self.model.apply(
                {'params': params},
                input_ids=batch.input_ids,
                state=model_state,
                attention_mask=batch.attention_mask,
                deterministic=not training,
                use_hrm=self.config.hrm_enabled,
                rngs={'dropout': std_dropout_rng},
                return_act_output=self.config.use_act,
            )
            if self.config.use_act:
                logits, new_state, act_output = apply_outputs
                return logits, new_state, {"hrm_enabled": self.config.hrm_enabled, "act_enabled": True, "act_output": act_output}
            else:
                logits, new_state = apply_outputs
                return logits, new_state, {"hrm_enabled": self.config.hrm_enabled, "act_enabled": False}
        
        # HRM one-step gradient approach
        current_state = model_state
        
        # Get HRM configuration from model
        hrm_cycles = getattr(self.model.config, 'hrm_executor_cycles', 2)
        hrm_steps = getattr(self.model.config, 'hrm_executor_steps', 2)
        
        # Run T-1 steps with detached states (no gradients)
        total_steps = hrm_cycles * hrm_steps
        
        for step in range(total_steps - 1):
            # Detach state to prevent gradient flow
            detached_state = GryphonHRMState(
                s5_state=jax.lax.stop_gradient(current_state.s5_state),
                hrm_carry=HRMInnerCarry(
                    z_H=jax.lax.stop_gradient(current_state.hrm_carry.z_H),
                    z_L=jax.lax.stop_gradient(current_state.hrm_carry.z_L)
                ),
                global_tokens=jax.lax.stop_gradient(current_state.global_tokens)
            )
            
            # Per-step rng (unique per forward)
            dropout_rng_step = jax.random.fold_in(rng_key, step)
            
            # Forward pass with detached state (no ACT outputs requested on non-final steps)
            _, current_state = self.model.apply(
                {'params': params},
                input_ids=batch.input_ids,
                state=detached_state,
                attention_mask=batch.attention_mask,
                deterministic=not training,
                use_hrm=True,
                rngs={'dropout': dropout_rng_step},
                return_act_output=False,
            )
        
        # Final step WITH gradients
        final_dropout_rng = jax.random.fold_in(rng_key, total_steps)
        apply_outputs = self.model.apply(
            {'params': params},
            input_ids=batch.input_ids,
            state=current_state,  # Not detached
            attention_mask=batch.attention_mask,
            deterministic=not training,
            use_hrm=True,
            rngs={'dropout': final_dropout_rng},
            return_act_output=self.config.use_act,
        )
        
        if self.config.use_act:
            logits, final_state, act_output = apply_outputs
        else:
            logits, final_state = apply_outputs
        
        # Detach final state for next iteration
        detached_final_state = GryphonHRMState(
            s5_state=jax.lax.stop_gradient(final_state.s5_state),
            hrm_carry=HRMInnerCarry(
                z_H=jax.lax.stop_gradient(final_state.hrm_carry.z_H),
                z_L=jax.lax.stop_gradient(final_state.hrm_carry.z_L)
            ),
            global_tokens=jax.lax.stop_gradient(final_state.global_tokens)
        )
        
        metrics = {
            "hrm_steps": total_steps,
            "gradient_steps": 1,  # Only final step has gradients
            "act_enabled": bool(self.config.use_act),
        }
        if self.config.use_act:
            metrics["act_output"] = act_output
        
        return logits, detached_final_state, metrics
    
    def _train_step(
        self,
        training_state: HRMTrainingState,
        batch: HRMBatch,
        rng_key: jax.Array
    ) -> Tuple[HRMTrainingState, Dict]:
        """Single training step with HRM logic."""
        
        def loss_fn(params):
            # Create model state
            model_state = GryphonHRMState(
                s5_state=training_state.s5_state,
                hrm_carry=training_state.hrm_carry,
                global_tokens=training_state.global_tokens
            )
            
            if self.config.deep_supervision_weight > 0:
                # Deep supervision: process segments separately
                segments = self._segment_batch(batch, self.config.max_segments_per_batch)
                
                total_loss = 0.0
                segment_losses = []
                current_state = model_state
                
                for i, segment in enumerate(segments):
                    # Forward pass for this segment
                    seg_rng = jax.random.fold_in(rng_key, i)
                    logits, new_state, hrm_metrics = self._hrm_forward_with_one_step_gradient(
                        params, current_state, segment, seg_rng, training=True
                    )
                    
                    # Compute segment loss
                    segment_loss = self._compute_language_modeling_loss(
                        logits, segment.labels, segment.attention_mask
                    )
                    
                    # Weight segment loss (later segments get higher weight)
                    if self.config.segment_loss_weights:
                        weight = self.config.segment_loss_weights[i]
                    else:
                        weight = (i + 1) / len(segments)  # Linear weighting
                    
                    weighted_loss = weight * segment_loss
                    total_loss += weighted_loss
                    segment_losses.append(segment_loss)
                    
                    # Detach state between segments (key for deep supervision)
                    if self.config.detach_states and i < len(segments) - 1:
                        current_state = GryphonHRMState(
                            s5_state=jax.lax.stop_gradient(new_state.s5_state),
                            hrm_carry=HRMInnerCarry(
                                z_H=jax.lax.stop_gradient(new_state.hrm_carry.z_H),
                                z_L=jax.lax.stop_gradient(new_state.hrm_carry.z_L)
                            ),
                            global_tokens=jax.lax.stop_gradient(new_state.global_tokens)
                        )
                    else:
                        current_state = new_state
                
                # Average loss across segments
                loss = total_loss / len(segments)
                
                # Store final state
                final_state = current_state
                
                aux_data = {
                    "segment_losses": jnp.array(segment_losses),
                    "num_segments": len(segments),
                    "final_state": final_state
                }
                
            else:
                # Standard training without deep supervision
                logits, final_state, hrm_metrics = self._hrm_forward_with_one_step_gradient(
                    params, model_state, batch, rng_key, training=True
                )
                
                loss = self._compute_language_modeling_loss(
                    logits, batch.labels, batch.attention_mask
                )
                
                aux_data = {
                    "final_state": final_state,
                    "hrm_metrics": hrm_metrics
                }
            
            return loss, aux_data
        
        # Compute gradients
        (loss, aux_data), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            training_state.model_state.params
        )
        
        # Update model parameters
        new_model_state = training_state.model_state.apply_gradients(grads=grads)
        
        # Update training state
        final_model_state = aux_data["final_state"]
        new_training_state = HRMTrainingState(
            model_state=new_model_state,
            hrm_carry=final_model_state.hrm_carry,
            s5_state=final_model_state.s5_state,
            global_tokens=final_model_state.global_tokens,
            step=training_state.step + 1,
            phase=training_state.phase
        )
        
        # Compute metrics
        metrics = {
            "loss": loss,
            "learning_rate": jnp.asarray(0.0),
            "step": new_training_state.step,
            "phase": training_state.phase
        }
        
        # Add segment-specific metrics if available
        if "segment_losses" in aux_data:
            metrics["segment_losses"] = aux_data["segment_losses"]
            metrics["num_segments"] = aux_data["num_segments"]
        
        # Add HRM metrics if available
        if "hrm_metrics" in aux_data:
            metrics.update(aux_data["hrm_metrics"])
        
        return new_training_state, metrics
    
    def _eval_step(
        self,
        training_state: HRMTrainingState,
        batch: HRMBatch,
        rng_key: jax.Array
    ) -> Dict:
        """Evaluation step."""
        # Create model state
        model_state = GryphonHRMState(
            s5_state=training_state.s5_state,
            hrm_carry=training_state.hrm_carry,
            global_tokens=training_state.global_tokens
        )
        
        # Forward pass
        logits, _, hrm_metrics = self._hrm_forward_with_one_step_gradient(
            training_state.model_state.params,
            model_state,
            batch,
            rng_key,
            training=False
        )
        
        # Compute loss
        loss = self._compute_language_modeling_loss(
            logits, batch.labels, batch.attention_mask
        )
        
        # Compute perplexity
        perplexity = jnp.exp(loss)
        
        metrics = {
            "eval_loss": loss,
            "eval_perplexity": perplexity
        }
        
        if hrm_metrics:
            metrics.update({f"eval_{k}": v for k, v in hrm_metrics.items()})
        
        return metrics
    
    def train_epoch(
        self,
        training_state: HRMTrainingState,
        data_iterator: Iterator[Tuple[List[PackedSequence], Dict]],
        rng_key: jax.Array,
        max_steps: Optional[int] = None
    ) -> Tuple[HRMTrainingState, List[Dict]]:
        """Train for one epoch."""
        metrics_history = []
        step_count = 0
        
        for batch_data, batch_metadata in data_iterator:
            if max_steps and step_count >= max_steps:
                break
            
            # Convert packed sequences to HRM batch
            hrm_batch = self._convert_to_hrm_batch(batch_data)
            
            # Training step
            rng_key, step_key = jax.random.split(rng_key)
            training_state, step_metrics = self.train_step_fn(
                training_state, hrm_batch, step_key
            )
            
            # Add batch metadata
            step_metrics.update(batch_metadata)
            metrics_history.append(step_metrics)
            
            step_count += 1
            
            # Log progress
            if step_count % 100 == 0:
                logger.info(f"Step {step_count}: loss={step_metrics['loss']:.4f}")
        
        return training_state, metrics_history
    
    def _convert_to_hrm_batch(self, packed_sequences: List[PackedSequence]) -> HRMBatch:
        """Convert packed sequences to HRM batch format."""
        batch_size = len(packed_sequences)
        seq_len = packed_sequences[0].input_ids.shape[0]
        
        # Stack sequences
        input_ids = jnp.stack([seq.input_ids for seq in packed_sequences])
        attention_mask = jnp.stack([seq.attention_mask for seq in packed_sequences])
        doc_ids = jnp.stack([seq.doc_ids for seq in packed_sequences])
        segment_ids = jnp.stack([seq.segment_ids for seq in packed_sequences])
        
        # Labels are same as input_ids for language modeling
        labels = input_ids
        
        # Extract metadata
        linked_next = jnp.array([seq.linked_next for seq in packed_sequences])
        source_ids = jnp.array([seq.source_id for seq in packed_sequences])
        
        return HRMBatch(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            doc_ids=doc_ids,
            segment_ids=segment_ids,
            linked_next=linked_next,
            source_ids=source_ids
        )
    
    def initialize_hrm_state(self, params: Dict, rng: jax.Array, batch_size: int = 1, seq_len: int = 1024) -> HRMTrainingState:
        """Initialize HRM training state for phase transition.
        
        Creates a TrainState for parameters and initializes the Gryphon HRM runtime state
        (s5_state, hrm_carry, global_tokens) via model.apply(method=self.model.init_state).
        """
        # Create a minimal optimizer for TrainState creation (no updates yet)
        tx = optax.adam(0.0)
        model_train_state = train_state.TrainState.create(
            apply_fn=self.model.apply,
            params=params,
            tx=tx
        )
        
        # Initialize HRM runtime state via method call
        hrm_init_state: GryphonHRMState = self.model.apply(
            {'params': params}, batch_size, seq_len, method=self.model.init_state
        )
        
        return HRMTrainingState(
            model_state=model_train_state,
            hrm_carry=hrm_init_state.hrm_carry,
            s5_state=hrm_init_state.s5_state,
            global_tokens=hrm_init_state.global_tokens,
            step=0,
            phase="init"
        )
    
    def compute_hrm_loss(
        self, 
        params: Dict, 
        batch: Dict, 
        phase_config: Dict,
        rng_key: jax.Array
    ) -> Dict:
        """Compute HRM loss for integration with main training loop.
        
        rng_key: base RNG key for this HRM forward; will be folded-in per inner step
        """
        # Convert batch format
        hrm_batch = HRMBatch(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            labels=batch['labels'],
            doc_ids=jnp.zeros_like(batch['input_ids']),  # Dummy doc_ids
            segment_ids=jnp.zeros_like(batch['input_ids']),  # Dummy segment_ids
            linked_next=jnp.zeros(batch['input_ids'].shape[0], dtype=bool),
            source_ids=jnp.zeros(batch['input_ids'].shape[0])
        )
        
        # Initialize HRM runtime state based on current batch shapes
        batch_size = hrm_batch.input_ids.shape[0]
        seq_len = hrm_batch.input_ids.shape[1]
        hrm_init_state: GryphonHRMState = self.model.apply(
            {'params': params}, batch_size, seq_len, method=self.model.init_state
        )
        
        # Forward pass with HRM
        logits, new_model_state, aux = self._hrm_forward_with_one_step_gradient(
            params, hrm_init_state, hrm_batch, rng_key, training=True
        )
        
        # Compute language modeling loss
        lm_loss = self._compute_language_modeling_loss(
            logits, hrm_batch.labels, hrm_batch.attention_mask
        )
        
        # ACT handling
        act_enabled = bool(phase_config.get('act_enabled', False)) or bool(self.config.use_act)
        act_loss_weight = phase_config.get('act_loss_weight', self.config.act_loss_weight)
        act_loss = jnp.asarray(0.0)
        act_metrics = {}
        if act_enabled and isinstance(aux, dict) and ('act_output' in aux):
            # compute_act_loss is expected to return (loss, metrics)
            act_loss_value, act_metrics_dict = compute_act_loss(aux['act_output'])
            act_loss = act_loss_value
            act_metrics.update(act_metrics_dict)
            act_metrics['act_enabled'] = True
        else:
            act_metrics['act_enabled'] = False
        
        # Combine losses
        total_loss = lm_loss + act_loss * act_loss_weight
        
        # Compose metrics
        metrics = {
            'hrm_loss': total_loss,
            'hrm_lm_loss': lm_loss,
            'act_loss': act_loss,
            'act_loss_weight': act_loss_weight,
        }
        # Include HRM forward aux metrics when available
        if isinstance(aux, dict):
            metrics.update({k: v for k, v in aux.items() if k != 'act_output'})
        # Include ACT metrics
        metrics.update(act_metrics)
        
        return {
            'loss': total_loss,
            'metrics': metrics
        }


def create_hrm_training_config(
    phase_config: Dict,
    use_act: bool = False
) -> HRMTrainingConfig:
    """Create HRM training configuration from phase config."""
    return HRMTrainingConfig(
        hrm_enabled=phase_config.get("hrm_enabled", False),
        use_one_step_gradient=True,
        detach_states=True,
        deep_supervision_weight=1.0,
        segment_loss_weights=None,  # Use linear weighting
        use_act=use_act,
        act_loss_weight=phase_config.get("act_loss_weight", 0.1),
        act_efficiency_weight=phase_config.get("act_efficiency_weight", 0.01),
        max_segments_per_batch=phase_config.get("max_segments_per_batch", 4),
        gradient_accumulation_steps=phase_config.get("gradient_accumulation_steps", 1),
        label_smoothing=phase_config.get("label_smoothing", 0.0),
        ignore_index=phase_config.get("ignore_index", -100)
    )