"""Compiled training and evaluation step functions.

Implements pjit-compiled functions for:
- Forward and backward passes with proper sharding
- Mixed precision training with fp32 for attention/S5
- Gradient accumulation and clipping
- Efficient memory usage with activation checkpointing
- S5 regularization for complex output handling
"""

import jax
import jax.numpy as jnp
from jax.experimental import pjit
from jax.sharding import PartitionSpec as P
import flax
import optax
from typing import Dict, Tuple, Any, Optional, Callable
import logging

from ..model import ValkyrieModel, ValkyrieConfig
from ..model.s5_regularization import s5_training_loss
from ..sharding import get_model_specs, get_training_specs, DP

logger = logging.getLogger(__name__)


def create_train_step(
    model: ValkyrieModel,
    optimizer: optax.GradientTransformation,
    config: ValkyrieConfig,
    mesh: Any,
    mixed_precision: bool = True,
) -> Callable:
    """
    Create pjit-compiled training step function.
    
    Args:
        model: Valkyrie model
        optimizer: Optax optimizer
        config: Model configuration
        mesh: JAX mesh for sharding
        mixed_precision: Whether to use mixed precision
        
    Returns:
        Compiled training step function
    """
    
    # Get sharding specs
    model_specs = get_model_specs(config, use_2d_sharding=False)
    training_specs = get_training_specs(model_specs)
    
    def train_step_fn(
        state: Dict[str, Any],
        batch: Dict[str, jnp.ndarray],
        dropout_rng: jax.random.PRNGKey,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Single training step with gradient computation and parameter update.
        
        Args:
            state: Training state (params, opt_state, step, etc.)
            batch: Input batch with 'input_ids' and optionally 'labels'
            dropout_rng: RNG key for dropout
            
        Returns:
            new_state: Updated training state
            metrics: Training metrics (loss, grad_norm, etc.)
        """
        
        def loss_fn(params):
            """Compute loss with mixed precision."""
            
            # Extract inputs
            input_ids = batch['input_ids']
            labels = batch.get('labels', None)
            
            # Early validation checks
            assert input_ids.dtype == jnp.int32, f"input_ids must be int32, got {input_ids.dtype}"
            max_token_id = int(input_ids.max())
            if max_token_id >= config.vocab_size:
                raise ValueError(f"Token id {max_token_id} >= vocab_size {config.vocab_size}. "
                               "Tokenizer/model mismatch (e.g., GPT-2 tokens with 32k model).")
            
            # If no labels provided, create them by shifting input_ids
            if labels is None:
                batch_size, seq_len = input_ids.shape
                labels = jnp.concatenate([
                    input_ids[:, 1:],
                    jnp.full((batch_size, 1), -100)  # Padding token
                ], axis=1)
            
            # Forward pass with mixed precision
            if mixed_precision:
                # Cast inputs to bfloat16 for TPU efficiency
                # But keep attention and S5 operations in fp32 (handled in model)
                input_ids = input_ids.astype(jnp.int32)  # Keep as int32
                labels = labels.astype(jnp.int32)
            
            # Model forward pass
            outputs = model.apply(
                params,
                input_ids=input_ids,
                labels=labels,
                s5_states=state.get('s5_states', None),
                use_cache=False,  # Don't use cache during training
                training=True,
                return_dict=True,
                rngs=(
                    {
                        'random': jax.random.fold_in(dropout_rng, state['step']),
                        'dropout': dropout_rng
                    }
                    if (config.attn_dropout > 0 or config.resid_dropout > 0 or config.ffn_dropout > 0)
                    else {
                        'random': jax.random.fold_in(dropout_rng, state['step'])
                    }
                ),
            )
            
            loss = outputs['loss']
            
            # Additional metrics
            aux_data = {
                'loss': loss,
                'logits': outputs['logits'],
                's5_states': outputs.get('s5_states', None),
            }
            
            return loss, aux_data
        
        # Compute gradients
        (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(state['params'])
        
        # Apply additional gradient clipping for numerical stability
        grads = jax.tree_util.tree_map(lambda g: jnp.clip(g, -1.0, 1.0), grads)
        
        # Gradient clipping and norm computation
        grad_norm = optax.global_norm(grads)
        grads = optax.clip_by_global_norm(config.gradient_clipping)(grads, state['opt_state'], state['params'])[0]
        
        # Apply optimizer update
        updates, new_opt_state = optimizer.update(grads, state['opt_state'], state['params'])
        new_params = optax.apply_updates(state['params'], updates)
        
        # Update training state
        new_state = {
            'params': new_params,
            'opt_state': new_opt_state,
            'step': state['step'] + 1,
            'rng': state['rng'],
        }
        
        # Add S5 states if using S5
        if config.use_s5 and aux['s5_states'] is not None:
            new_state['s5_states'] = aux['s5_states']
        
        # Compute metrics
        metrics = {
            'loss': loss,
            'grad_norm': grad_norm,
            'learning_rate': optimizer._schedule(state['step']) if hasattr(optimizer, '_schedule') else 0.0,
            'step': state['step'],
        }
        
        return new_state, metrics
    
    # Get training specs for explicit sharding (matches initialization)
    training_specs = get_training_specs(model_specs)
    
    # Determine batch sharding based on mesh configuration
    # If mesh.z == 1, batch should be replicated, otherwise shard on DP
    batch_spec = P(DP, None) if mesh.shape[mesh.axis_names.index('z')] > 1 else P()
    
    # Compile with pjit using explicit sharding specs (matches parameter initialization)
    compiled_train_step = pjit.pjit(
        train_step_fn,
        in_shardings=(
            training_specs,          # state - use explicit sharding specs that match initialization
            batch_spec,              # batch (sharded along batch dimension if multi-host)
            P(),                     # dropout_rng (replicated)
        ),
        out_shardings=(
            training_specs,          # new_state - use same explicit sharding specs
            P(),                     # metrics (replicated)
        ),
        donate_argnums=(0,),         # Donate state for memory efficiency
    )
    
    return compiled_train_step


def create_eval_step(
    model: ValkyrieModel,
    config: ValkyrieConfig,
    mesh: Any,
    mixed_precision: bool = True,
) -> Callable:
    """
    Create pjit-compiled evaluation step function.
    
    Args:
        model: Valkyrie model
        config: Model configuration
        mesh: JAX mesh for sharding
        mixed_precision: Whether to use mixed precision
        
    Returns:
        Compiled evaluation step function
    """
    
    # Get sharding specs
    model_specs = get_model_specs(config, use_2d_sharding=False)
    
    def eval_step_fn(
        params: Dict[str, Any],
        batch: Dict[str, jnp.ndarray],
        s5_states: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Single evaluation step (forward pass only).
        
        Args:
            params: Model parameters
            batch: Input batch
            s5_states: S5 states for sequence continuation
            
        Returns:
            metrics: Evaluation metrics
        """
        
        # Extract inputs
        input_ids = batch['input_ids']
        labels = batch.get('labels', None)
        
        # Create labels if not provided
        if labels is None:
            batch_size, seq_len = input_ids.shape
            labels = jnp.concatenate([
                input_ids[:, 1:],
                jnp.full((batch_size, 1), -100)
            ], axis=1)
        
        # Mixed precision casting
        if mixed_precision:
            input_ids = input_ids.astype(jnp.int32)
            labels = labels.astype(jnp.int32)
        
        # Forward pass (no gradients)
        outputs = model.apply(
            params,
            input_ids=input_ids,
            labels=labels,
            s5_states=s5_states,
            use_cache=False,
            training=False,  # Disable dropout
            return_dict=True,
        )
        
        loss = outputs['loss']
        logits = outputs['logits']
        
        # Compute additional metrics
        # Accuracy (for tokens that aren't padding)
        predictions = jnp.argmax(logits[..., :-1, :], axis=-1)
        targets = labels[..., 1:]
        mask = targets != -100
        
        correct = (predictions == targets) & mask
        accuracy = jnp.sum(correct) / jnp.sum(mask)
        
        # Perplexity
        perplexity = jnp.exp(loss)
        
        metrics = {
            'eval_loss': loss,
            'eval_accuracy': accuracy,
            'eval_perplexity': perplexity,
            's5_states': outputs.get('s5_states', None),
        }
        
        return metrics
    
    # Compile with pjit using param-inferred sharding
    compiled_eval_step = pjit.pjit(
        eval_step_fn,
        in_shardings=(
            None,                    # params - let arrays' own sharding drive compilation
            P('z', None),           # batch
            P() if config.use_s5 else None,  # s5_states
        ),
        out_shardings=P(),           # metrics (replicated)
    )
    
    return compiled_eval_step


def create_chunked_train_step(
    model: ValkyrieModel,
    optimizer: optax.GradientTransformation,
    config: ValkyrieConfig,
    mesh: Any,
    chunk_size: int = 8192,
    gradient_accumulation_steps: int = 1,
    mixed_precision: bool = True,
) -> Callable:
    """
    Create chunked training step for long sequences.
    
    This function processes long sequences in chunks and accumulates gradients
    across chunks before applying optimizer updates.
    
    Args:
        model: Valkyrie model
        optimizer: Optax optimizer
        config: Model configuration
        mesh: JAX mesh for sharding
        chunk_size: Size of each chunk in tokens
        gradient_accumulation_steps: Number of chunks to accumulate gradients over
        mixed_precision: Whether to use mixed precision
        
    Returns:
        Compiled chunked training step function
    """
    
    # Get sharding specs
    model_specs = get_model_specs(config, use_2d_sharding=False)
    training_specs = get_training_specs(model_specs)
    
    def chunked_train_step_fn(
        state: Dict[str, Any],
        batch: Dict[str, jnp.ndarray],
        dropout_rng: jax.random.PRNGKey,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Chunked training step with gradient accumulation.
        
        Processes long sequences by:
        1. Splitting into chunks
        2. Computing gradients for each chunk
        3. Accumulating gradients across chunks
        4. Applying optimizer update
        """
        
        input_ids = batch['input_ids']  # [batch, full_seq_len]
        batch_size, full_seq_len = input_ids.shape
        
        # Calculate number of chunks
        num_chunks = (full_seq_len + chunk_size - 1) // chunk_size
        
        # Initialize accumulated gradients
        accumulated_grads = jax.tree_map(jnp.zeros_like, state['params'])
        total_loss = 0.0
        s5_states = state.get('s5_states', None)
        
        # Process each chunk
        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx * chunk_size
            end_idx = min((chunk_idx + 1) * chunk_size, full_seq_len)
            
            # Extract chunk
            chunk_input = input_ids[:, start_idx:end_idx]
            chunk_labels = jnp.concatenate([
                chunk_input[:, 1:],
                jnp.full((batch_size, 1), -100)
            ], axis=1)
            
            # Define loss function for this chunk
            def chunk_loss_fn(params):
                outputs = model.apply(
                    params,
                    input_ids=chunk_input,
                    labels=chunk_labels,
                    s5_states=s5_states,
                    use_cache=True,
                    training=True,
                    return_dict=True,
                    rngs=(
                        {
                            'random': jax.random.fold_in(dropout_rng, chunk_idx),
                            'dropout': dropout_rng
                        }
                        if (config.attn_dropout > 0 or config.resid_dropout > 0 or config.ffn_dropout > 0)
                        else {
                            'random': jax.random.fold_in(dropout_rng, chunk_idx)
                        }
                    ),
                )
                
                return outputs['loss'], outputs.get('s5_states', None)
            
            # Compute gradients for this chunk
            (chunk_loss, new_s5_states), chunk_grads = jax.value_and_grad(
                chunk_loss_fn, has_aux=True
            )(state['params'])
            
            # Accumulate gradients
            accumulated_grads = jax.tree_map(
                lambda acc, new: acc + new / num_chunks,
                accumulated_grads,
                chunk_grads
            )
            
            total_loss += chunk_loss / num_chunks
            s5_states = new_s5_states
        
        # Apply gradient clipping (global norm only)
        grad_norm = optax.global_norm(accumulated_grads)
        accumulated_grads = optax.clip_by_global_norm(config.gradient_clipping)(
            accumulated_grads, state['opt_state'], state['params']
        )[0]
        
        # Apply optimizer update
        updates, new_opt_state = optimizer.update(
            accumulated_grads, state['opt_state'], state['params']
        )
        new_params = optax.apply_updates(state['params'], updates)
        
        # Update state
        new_state = {
            'params': new_params,
            'opt_state': new_opt_state,
            'step': state['step'] + 1,
            'rng': state['rng'],
        }
        
        if config.use_s5 and s5_states is not None:
            new_state['s5_states'] = s5_states
        
        # Metrics
        metrics = {
            'loss': total_loss,
            'grad_norm': grad_norm,
            'num_chunks': num_chunks,
            'chunk_size': chunk_size,
        }
        
        return new_state, metrics
    
    # Compile with pjit using param-inferred sharding
    compiled_chunked_step = pjit.pjit(
        chunked_train_step_fn,
        in_shardings=(
            None,                    # state - let arrays' own sharding drive compilation
            P('z', None),           # batch
            P(),                     # dropout_rng
        ),
        out_shardings=(
            None,                    # new_state - infer from computation
            P(),                     # metrics
        ),
        donate_argnums=(0,),
    )
    
    return compiled_chunked_step


def create_generation_step(
    model: ValkyrieModel,
    config: ValkyrieConfig,
    mesh: Any,
) -> Callable:
    """
    Create pjit-compiled generation step function.
    
    Args:
        model: Valkyrie model
        config: Model configuration
        mesh: JAX mesh for sharding
        
    Returns:
        Compiled generation step function
    """
    
    model_specs = get_model_specs(config, use_2d_sharding=False)
    
    def generation_step_fn(
        params: Dict[str, Any],
        input_ids: jnp.ndarray,
        past_key_values: Optional[Any] = None,
        s5_states: Optional[Any] = None,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 1.0,
        rng_key: Optional[jax.random.PRNGKey] = None,
    ) -> Tuple[jnp.ndarray, Any, Any]:
        """
        Single generation step.
        
        Returns:
            next_token: Generated token [batch, 1]
            new_past_key_values: Updated KV cache
            new_s5_states: Updated S5 states
        """
        
        # Forward pass
        outputs = model.apply(
            params,
            input_ids=input_ids,
            past_key_values=past_key_values,
            s5_states=s5_states,
            use_cache=True,
            training=False,
            return_dict=True,
        )
        
        logits = outputs['logits'][:, -1, :]  # Last token logits
        
        # Apply temperature
        if temperature != 1.0:
            logits = logits / temperature
        
        # Sample next token
        if rng_key is not None:
            # Sampling
            from ..model.valkyrie import sample_token
            next_token = sample_token(logits, temperature, top_k, top_p, rng_key)
        else:
            # Greedy
            next_token = jnp.argmax(logits, axis=-1, keepdims=True)
        
        return (
            next_token,
            outputs.get('past_key_values', None),
            outputs.get('s5_states', None)
        )
    
    # Compile with pjit using param-inferred sharding
    compiled_generation_step = pjit.pjit(
        generation_step_fn,
        in_shardings=(
            None,                          # params - let arrays' own sharding drive compilation
            P('z', None),                 # input_ids
            P() if config.use_longformer_attention else P(),  # past_key_values
            P() if config.use_s5 else None,  # s5_states
            P(),                           # temperature
            P(),                           # top_k
            P(),                           # top_p
            P(),                           # rng_key
        ),
        out_shardings=(
            P('z', None),                 # next_token
            P() if config.use_longformer_attention else P(),  # new_past_key_values
            P() if config.use_s5 else None,  # new_s5_states
        ),
    )
    
    return compiled_generation_step