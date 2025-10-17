"""Gryphon Model

Complete implementation of the hybrid BigBird-S5 architecture.
Combines the sequential processing power of S5 with the global
information routing capabilities of BigBird sparse attention.

Architecture Overview:
- Token embeddings with positional encoding
- Stack of Gryphon hybrid layers (S5 → BigBird)
- Output layer normalization and language modeling head

Key Features:
- Linear O(L) complexity for both S5 and BigBird components
- Mixed precision training support
- Gradient checkpointing for memory efficiency
- Support for both training and inference modes
- Recurrent generation with S5 state caching
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Optional, Tuple, Dict, Any
import math

from .gryphon_config import GryphonConfig
from .gryphon_blocks import GryphonLayer
from .gryphon_utils import pad_to_block_size
from ..modules import RMSNorm, precompute_rope_freqs


class GryphonEmbeddings(nn.Module):
    """Token and position embeddings for Gryphon model."""
    
    config: GryphonConfig
    
    def setup(self):
        """Initialize embedding layers."""
        # Token embeddings
        self.token_embeddings = nn.Embed(
            num_embeddings=self.config.vocab_size,
            features=self.config.d_model,
            embedding_init=nn.initializers.normal(stddev=self.config.initializer_range),
            name='token_embeddings'
        )
        
        # Embedding dropout
        self.dropout = nn.Dropout(
            rate=self.config.resid_dropout,
            deterministic=False
        )
        
        # Pre-compute RoPE frequencies for efficiency
        head_dim = self.config.d_model // self.config.n_heads
        self.cos_freqs, self.sin_freqs = precompute_rope_freqs(
            dim=head_dim,
            max_seq_len=self.config.max_sequence_length,
            base=self.config.rope_theta
        )
        
        # Store as non-trainable parameters
        self.cos_freqs = self.variable('constants', 'cos_freqs', lambda: self.cos_freqs).value
        self.sin_freqs = self.variable('constants', 'sin_freqs', lambda: self.sin_freqs).value
    
    def __call__(
        self,
        input_ids: jnp.ndarray,
        position_ids: Optional[jnp.ndarray] = None,
        training: bool = True
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Forward pass of embeddings.
        
        Args:
            input_ids: Token indices [batch, seq_len]
            position_ids: Position indices [batch, seq_len]
            training: Whether in training mode
            
        Returns:
            Tuple of (embeddings, position_ids, cos_freqs, sin_freqs)
        """
        batch_size, seq_len = input_ids.shape
        
        # Generate position_ids if not provided
        if position_ids is None:
            position_ids = jnp.arange(seq_len)[None, :].repeat(batch_size, axis=0)
        
        # Token embeddings
        embeddings = self.token_embeddings(input_ids)
        
        # Apply dropout
        if training:
            embeddings = self.dropout(embeddings)
        
        return embeddings, position_ids, self.cos_freqs, self.sin_freqs


class GryphonModel(nn.Module):
    """Complete Gryphon hybrid model.
    
    Implements the full hybrid architecture combining:
    - S5 state space modeling for sequential processing
    - BigBird sparse attention for global information routing
    
    The model follows Blueprint A from the architectural guide:
    alternating S5 and BigBird blocks for optimal synergy.
    
    Mathematical Foundation:
    - S5: dx/dt = Ax + Bu, y = Cx + Du (continuous-time state space)
    - BigBird: Sparse attention with O(L) complexity
    - Combined: S5 enriches → BigBird routes globally
    
    Key Optimizations:
    - Block-wise operations for TPU efficiency
    - Mixed precision training (bfloat16/float32)
    - Gradient checkpointing for memory efficiency
    - Parameter-specific learning rates
    """
    
    config: GryphonConfig
    
    def setup(self):
        """Initialize model components."""
        # Input embeddings
        self.embeddings = GryphonEmbeddings(config=self.config)
        
        # Stack of Gryphon hybrid layers
        self.layers = [
            GryphonLayer(
                config=self.config,
                layer_idx=i,
                name=f'layer_{i}'
            )
            for i in range(self.config.n_layers)
        ]
        
        # Final layer normalization
        self.final_norm = RMSNorm(
            hidden_size=self.config.d_model,
            eps=self.config.layer_norm_eps
        )
        
        # Language modeling head
        self.lm_head = nn.Dense(
            self.config.vocab_size,
            use_bias=False,
            kernel_init=nn.initializers.normal(stddev=self.config.initializer_range),
            name='lm_head'
        )
        
        # Initialize model statistics
        self._init_model_stats()
    
    def _init_model_stats(self):
        """Initialize model statistics for monitoring."""
        # Parameter count estimation
        embedding_params = self.config.vocab_size * self.config.d_model
        
        # Per-layer parameters
        s5_params_per_layer = (
            self.config.s5_state_dim * self.config.d_model * 4 +  # B, C matrices (real + imag)
            self.config.s5_state_dim * 2 +  # Lambda (real + imag)
            self.config.s5_state_dim +  # Delta
            self.config.d_model  # D
        )
        
        attention_params_per_layer = (
            self.config.d_model * self.config.d_model * 4 +  # Q, K, V, O projections
            self.config.d_model * 2  # Layer norms
        )
        
        mlp_params_per_layer = (
            self.config.d_model * (4 * self.config.d_model) * 3 +  # Gate, Up, Down
            self.config.d_model  # Layer norm
        )
        
        total_layer_params = self.config.n_layers * (
            s5_params_per_layer + attention_params_per_layer + mlp_params_per_layer
        )
        
        lm_head_params = self.config.vocab_size * self.config.d_model
        
        self.total_params = embedding_params + total_layer_params + lm_head_params
        
        # Memory estimates
        self.memory_estimates = self.config.get_memory_estimates()
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information."""
        attention_info = self.config.get_attention_pattern_info()
        
        return {
            'model_type': 'Gryphon (BigBird + S5 Hybrid)',
            'total_parameters': f"{self.total_params / 1e6:.1f}M",
            'architecture': {
                'n_layers': self.config.n_layers,
                'd_model': self.config.d_model,
                'n_heads': self.config.n_heads,
                's5_state_dim': self.config.s5_state_dim,
                'max_sequence_length': self.config.max_sequence_length
            },
            'sparse_attention': attention_info,
            'memory_estimates': self.memory_estimates,
            'optimizations': {
                'mixed_precision': self.config.use_mixed_precision,
                'gradient_checkpointing': self.config.use_gradient_checkpointing,
                'block_size': self.config.block_size,
                'sparsity_ratio': f"{attention_info['sparsity_ratio']:.1%}"
            }
        }
    
    def init_s5_states(self, batch_size: int) -> list:
        """Initialize S5 states for recurrent generation.
        
        Args:
            batch_size: Batch size
            
        Returns:
            List of S5 states for each layer and block
        """
        s5_states = []
        
        for layer_idx in range(self.config.n_layers):
            layer_states = []
            
            # Each layer can have multiple Gryphon blocks
            num_blocks_per_layer = max(1, self.config.s5_blocks_per_layer)
            
            for block_idx in range(num_blocks_per_layer):
                # Initialize S5 state with zeros (complex64)
                state_shape = (batch_size, self.config.s5_state_dim)
                initial_state = jnp.zeros(state_shape, dtype=jnp.complex64)
                layer_states.append(initial_state)
            
            s5_states.append(layer_states)
        
        return s5_states
    
    def __call__(
        self,
        input_ids: jnp.ndarray,
        attention_mask: Optional[jnp.ndarray] = None,
        position_ids: Optional[jnp.ndarray] = None,
        s5_states: Optional[list] = None,
        training: bool = True,
        return_dict: bool = True
    ) -> Dict[str, Any]:
        """Forward pass of Gryphon model.
        
        Args:
            input_ids: Token indices [batch, seq_len]
            attention_mask: Optional attention mask [batch, seq_len]
            position_ids: Position indices [batch, seq_len]
            s5_states: S5 states for recurrent generation
            training: Whether in training mode
            return_dict: Whether to return dictionary output
            
        Returns:
            Dictionary containing logits and states, or just logits if return_dict=False
        """
        batch_size, seq_len = input_ids.shape
        
        # === Input Processing ===
        
        # Get embeddings and RoPE frequencies
        hidden_states, position_ids, cos_freqs, sin_freqs = self.embeddings(
            input_ids, position_ids, training=training
        )
        
        # Pad sequence to block size if necessary for BigBird
        original_seq_len = seq_len
        if seq_len % self.config.block_size != 0:
            hidden_states, _ = pad_to_block_size(hidden_states, self.config.block_size, axis=1)
            
            if attention_mask is not None:
                attention_mask, _ = pad_to_block_size(attention_mask, self.config.block_size, axis=1)
            
            # Extend position_ids for padded tokens
            padded_seq_len = hidden_states.shape[1]
            if padded_seq_len > seq_len:
                max_pos = jnp.max(position_ids, axis=1, keepdims=True)
                pad_positions = jnp.arange(1, padded_seq_len - seq_len + 1)[None, :] + max_pos
                position_ids = jnp.concatenate([position_ids, pad_positions], axis=1)
        
        # === Layer Processing ===
        
        final_s5_states = []
        
        # Process through each Gryphon layer
        for layer_idx, layer in enumerate(self.layers):
            # Get S5 states for this layer
            layer_s5_states = s5_states[layer_idx] if s5_states is not None else None
            
            # Forward pass through hybrid layer
            hidden_states, layer_final_s5_states = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                cos_freqs=cos_freqs,
                sin_freqs=sin_freqs,
                s5_states=layer_s5_states,
                causal=True,  # Always use causal masking for language modeling
                training=training
            )
            
            final_s5_states.append(layer_final_s5_states)
        
        # === Output Processing ===
        
        # Final layer normalization
        hidden_states = self.final_norm(hidden_states)
        
        # Trim back to original sequence length if padded
        if hidden_states.shape[1] > original_seq_len:
            hidden_states = hidden_states[:, :original_seq_len, :]
        
        # Language modeling head
        logits = self.lm_head(hidden_states)
        
        # Return results
        if return_dict:
            return {
                'logits': logits,
                'hidden_states': hidden_states,
                's5_states': final_s5_states,
                'attention_mask': attention_mask,
                'position_ids': position_ids[:, :original_seq_len] if position_ids is not None else None
            }
        else:
            return logits
    
    def generate_step(
        self,
        input_ids: jnp.ndarray,
        s5_states: list,
        attention_mask: Optional[jnp.ndarray] = None,
        position_ids: Optional[jnp.ndarray] = None,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9
    ) -> Tuple[jnp.ndarray, list]:
        """Single generation step for autoregressive sampling.
        
        Args:
            input_ids: Current token [batch, 1]
            s5_states: Current S5 states
            attention_mask: Optional attention mask
            position_ids: Current position [batch, 1]
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Top-p (nucleus) sampling parameter
            
        Returns:
            Tuple of (next_token_ids, updated_s5_states)
        """
        # Forward pass with current token
        outputs = self(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            s5_states=s5_states,
            training=False,
            return_dict=True
        )
        
        # Get logits for the last (and only) position
        logits = outputs['logits'][:, -1, :]  # [batch, vocab_size]
        
        # Apply temperature
        if temperature != 1.0:
            logits = logits / temperature
        
        # Apply top-k filtering
        if top_k > 0:
            top_k_logits, top_k_indices = jax.lax.top_k(logits, top_k)
            logits = jnp.full_like(logits, float('-inf'))
            logits = logits.at[jnp.arange(logits.shape[0])[:, None], top_k_indices].set(top_k_logits)
        
        # Apply top-p (nucleus) filtering
        if top_p < 1.0:
            sorted_logits = jnp.sort(logits, axis=-1)[:, ::-1]
            sorted_probs = jax.nn.softmax(sorted_logits, axis=-1)
            cumulative_probs = jnp.cumsum(sorted_probs, axis=-1)
            
            # Find cutoff index
            cutoff_mask = cumulative_probs <= top_p
            cutoff_indices = jnp.sum(cutoff_mask, axis=-1, keepdims=True)
            
            # Create nucleus mask
            sorted_indices = jnp.argsort(logits, axis=-1)[:, ::-1]
            nucleus_mask = jnp.arange(logits.shape[-1])[None, :] < cutoff_indices
            
            # Apply nucleus filtering
            nucleus_logits = jnp.where(nucleus_mask, sorted_logits, float('-inf'))
            
            # Map back to original order
            original_order = jnp.argsort(sorted_indices, axis=-1)
            logits = jnp.take_along_axis(nucleus_logits, original_order, axis=-1)
        
        # Sample next token
        probs = jax.nn.softmax(logits, axis=-1)
        next_token_ids = jax.random.categorical(
            jax.random.PRNGKey(0),  # Use deterministic key for reproducibility
            logits,
            axis=-1
        )[:, None]  # [batch, 1]
        
        return next_token_ids, outputs['s5_states']


# Factory functions for different model sizes
def create_gryphon_small(vocab_size: int = 50257) -> GryphonModel:
    """Create small Gryphon model for experimentation."""
    from .gryphon_config import get_gryphon_small_config
    config = get_gryphon_small_config()
    config.vocab_size = vocab_size
    return GryphonModel(config=config)


def create_gryphon_base(vocab_size: int = 50257) -> GryphonModel:
    """Create base Gryphon model for general use."""
    from .gryphon_config import get_gryphon_base_config
    config = get_gryphon_base_config()
    config.vocab_size = vocab_size
    return GryphonModel(config=config)


def create_gryphon_large(vocab_size: int = 50257) -> GryphonModel:
    """Create large Gryphon model for high-performance applications."""
    from .gryphon_config import get_gryphon_large_config
    config = get_gryphon_large_config()
    config.vocab_size = vocab_size
    return GryphonModel(config=config)