"""Gryphon Hybrid Blocks

Implementation of hybrid S5 + BigBird blocks following Blueprint A:
alternating S5 and BigBird processing for optimal local/global information flow.

Architecture:
- S5Block: Sequential processing with state space modeling
- BigBirdBlock: Sparse global attention with feed-forward network
- GryphonBlock: Combined S5 → BigBird processing unit
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Optional, Tuple

from .gryphon_config import GryphonConfig
from .bigbird_attention import BigBirdSparseAttention, BigBirdMLP
from ..s5 import ValkyrieS5
from ..modules import RMSNorm


class GlobalTokenCoupling(nn.Module):
    """Global token coupling mechanism for S5 state management.
    
    This module enables BigBird's global tokens to influence S5 state updates,
    creating a feedback loop between global context and sequential processing.
    
    The coupling works by:
    1. Extracting global token representations from BigBird output
    2. Learning state modification signals via a small MLP
    3. Applying gated updates to S5 state: new_state = (1-gate) * old_state + gate * injection
    """
    
    config: GryphonConfig
    
    def setup(self):
        """Initialize coupling components."""
        # MLP for learning coupling transformations
        self.coupling_mlp = nn.Sequential([
            nn.Dense(
                features=self.config.coupling_hidden_dim,
                dtype=self.config.param_dtype,
                kernel_init=nn.initializers.xavier_uniform(),
                bias_init=nn.initializers.zeros
            ),
            nn.gelu,
            nn.Dense(
                features=self.config.s5_state_dim * 2,  # gate + injection
                dtype=self.config.param_dtype,
                kernel_init=nn.initializers.xavier_uniform(),
                bias_init=nn.initializers.zeros
            )
        ])
        
        # Layer norm for stability
        self.coupling_norm = RMSNorm(
            hidden_size=self.config.d_model,
            eps=self.config.layer_norm_eps
        )
    
    def __call__(
        self,
        bigbird_output: jnp.ndarray,
        s5_state: Optional[jnp.ndarray] = None,
        layer_idx: int = 0,
        training: bool = True
    ) -> Tuple[Optional[jnp.ndarray], jnp.ndarray]:
        """Apply global token coupling to S5 state.
        
        Args:
            bigbird_output: BigBird output [batch, seq_len, d_model]
            s5_state: Current S5 state [batch, state_dim] or None
            layer_idx: Current layer index for frequency control
            training: Whether in training mode
            
        Returns:
            Tuple of (modified_s5_state, coupling_info)
        """
        batch_size, seq_len, d_model = bigbird_output.shape
        
        # Check if coupling should be applied this layer
        if not self.config.use_global_coupling:
            return s5_state, jnp.zeros((batch_size,))
        
        # CRITICAL FIX: Use static evaluation to prevent layer_idx from being included in gradients
        # Convert to static values to avoid float0 gradient errors
        should_apply_coupling = (
            self.config.coupling_frequency <= 0 or 
            int(layer_idx) % int(self.config.coupling_frequency) == 0
        )
        
        if not should_apply_coupling:
            return s5_state, jnp.zeros((batch_size,))
        
        # Extract global tokens (first num_global_blocks tokens)
        # CRITICAL FIX: Use static conversion to prevent config parameters from being in gradients
        num_global_tokens = int(self.config.num_global_blocks)
        global_tokens = bigbird_output[:, :num_global_tokens, :]  # [batch, num_global, d_model]
        
        # Aggregate global tokens (mean pooling with normalization)
        global_repr = self.coupling_norm(jnp.mean(global_tokens, axis=1))  # [batch, d_model]
        
        # Generate coupling signals
        coupling_signals = self.coupling_mlp(global_repr)  # [batch, s5_state_dim * 2]
        
        # Split into gate and injection components
        gate_logits, injection = jnp.split(coupling_signals, 2, axis=-1)
        
        # Apply sigmoid to gate for [0,1] range, scale by coupling_strength
        coupling_gate = jax.nn.sigmoid(gate_logits) * self.config.coupling_strength
        
        # Apply coupling if S5 state exists
        if s5_state is not None:
            # Gated state update: new_state = (1-gate) * old_state + gate * injection
            modified_state = (1.0 - coupling_gate) * s5_state + coupling_gate * injection
            
            # Compute coupling strength for monitoring
            coupling_info = jnp.mean(coupling_gate, axis=-1)  # [batch,]
            
            return modified_state, coupling_info
        else:
            # No existing state, return injection as initial state
            coupling_info = jnp.mean(coupling_gate, axis=-1)  # [batch,]
            return injection * coupling_gate, coupling_info


class S5Block(nn.Module):
    """S5 processing block with layer normalization and residual connections.
    
    This block applies S5 state space modeling for sequential feature extraction.
    S5 excels at:
    - Local pattern recognition
    - Temporal dependency modeling
    - Long-range memory with HiPPO initialization
    - Linear O(L) complexity
    """
    
    config: GryphonConfig
    
    def setup(self):
        """Initialize S5 block components."""
        # Layer normalization (pre-norm architecture)
        self.input_norm = RMSNorm(
            hidden_size=self.config.d_model,
            eps=self.config.layer_norm_eps
        )
        
        # S5 state space model
        self.s5_layer = ValkyrieS5(
            config=self.config,
            state_dim=self.config.s5_state_dim,
            init_mode=self.config.s5_init_mode
        )
        
        # Residual dropout
        self.dropout = nn.Dropout(
            rate=self.config.resid_dropout,
            deterministic=False
        )
    
    def __call__(
        self,
        hidden_states: jnp.ndarray,
        training: bool = True,
        s5_state: Optional[jnp.ndarray] = None
    ) -> Tuple[jnp.ndarray, Optional[jnp.ndarray]]:
        """Forward pass of S5 block.
        
        Args:
            hidden_states: Input tensor [batch, seq_len, d_model]
            training: Whether in training mode
            s5_state: Previous S5 state for recurrent generation
            
        Returns:
            Tuple of (output, final_s5_state)
        """
        # Pre-normalization
        normed_input = self.input_norm(hidden_states)
        
        # S5 processing
        s5_output, final_s5_state = self.s5_layer(
            normed_input, 
            training=training, 
            state=s5_state
        )
        
        # Apply dropout
        if training:
            s5_output = self.dropout(s5_output)
        
        # Residual connection
        output = hidden_states + s5_output
        
        return output, final_s5_state


class BigBirdBlock(nn.Module):
    """BigBird attention block with sparse attention and feed-forward network.
    
    This block applies sparse global attention for information routing.
    BigBird excels at:
    - Global information routing
    - Long-range token interactions
    - Sparse O(L) attention complexity
    - Preserving Transformer representational power
    """
    
    config: GryphonConfig
    
    def setup(self):
        """Initialize BigBird block components."""
        # Layer normalizations (pre-norm architecture)
        self.attention_norm = RMSNorm(
            hidden_size=self.config.d_model,
            eps=self.config.layer_norm_eps
        )
        
        self.ffn_norm = RMSNorm(
            hidden_size=self.config.d_model,
            eps=self.config.layer_norm_eps
        )
        
        # BigBird sparse attention
        self.attention = BigBirdSparseAttention(config=self.config)
        
        # Feed-forward network
        self.mlp = BigBirdMLP(config=self.config)
        
        # Residual dropouts
        self.attn_dropout = nn.Dropout(
            rate=self.config.resid_dropout,
            deterministic=False
        )
        
        self.ffn_dropout = nn.Dropout(
            rate=self.config.resid_dropout,
            deterministic=False
        )
    
    def __call__(
        self,
        hidden_states: jnp.ndarray,
        attention_mask: Optional[jnp.ndarray] = None,
        position_ids: Optional[jnp.ndarray] = None,
        cos_freqs: Optional[jnp.ndarray] = None,
        sin_freqs: Optional[jnp.ndarray] = None,
        causal: bool = True,
        training: bool = True
    ) -> jnp.ndarray:
        """Forward pass of BigBird block.
        
        Args:
            hidden_states: Input tensor [batch, seq_len, d_model]
            attention_mask: Optional attention mask [batch, seq_len]
            position_ids: Position indices [batch, seq_len]
            cos_freqs: RoPE cosine frequencies
            sin_freqs: RoPE sine frequencies
            causal: Whether to apply causal masking
            training: Whether in training mode
            
        Returns:
            Block output [batch, seq_len, d_model]
        """
        # === Attention Sub-block ===
        
        # Pre-normalization
        normed_input = self.attention_norm(hidden_states)
        
        # Sparse attention
        attention_output = self.attention(
            normed_input,
            attention_mask=attention_mask,
            position_ids=position_ids,
            cos_freqs=cos_freqs,
            sin_freqs=sin_freqs,
            causal=causal,
            training=training
        )
        
        # Apply dropout and residual connection
        if training:
            attention_output = self.attn_dropout(attention_output)
        
        hidden_states = hidden_states + attention_output
        
        # === Feed-Forward Sub-block ===
        
        # Pre-normalization
        normed_input = self.ffn_norm(hidden_states)
        
        # MLP
        ffn_output = self.mlp(normed_input, training=training)
        
        # Apply dropout and residual connection
        if training:
            ffn_output = self.ffn_dropout(ffn_output)
        
        output = hidden_states + ffn_output
        
        return output


class GryphonBlock(nn.Module):
    """Hybrid Gryphon block combining S5 and BigBird processing.
    
    Implements Blueprint A: S5 → BigBird sequential processing.
    
    Information flow:
    1. S5 processes sequence for local/temporal feature extraction
    2. BigBird applies sparse global attention for information routing
    
    This creates a powerful synergy:
    - S5 enriches each token with contextual information
    - BigBird routes and compares these enriched representations globally
    """
    
    config: GryphonConfig
    
    def setup(self):
        """Initialize hybrid block components."""
        # S5 processing block
        self.s5_block = S5Block(config=self.config)
        
        # BigBird attention block
        self.bigbird_block = BigBirdBlock(config=self.config)
        
        # Global token coupling mechanism
        if self.config.use_global_coupling:
            self.global_coupling = GlobalTokenCoupling(config=self.config)
        else:
            self.global_coupling = None
        
        # Optional intermediate normalization for stability
        if self.config.gradient_checkpointing:
            self.intermediate_norm = RMSNorm(
                hidden_size=self.config.d_model,
                eps=self.config.layer_norm_eps
            )
        else:
            self.intermediate_norm = None
    
    def __call__(
        self,
        hidden_states: jnp.ndarray,
        attention_mask: Optional[jnp.ndarray] = None,
        position_ids: Optional[jnp.ndarray] = None,
        cos_freqs: Optional[jnp.ndarray] = None,
        sin_freqs: Optional[jnp.ndarray] = None,
        s5_state: Optional[jnp.ndarray] = None,
        layer_idx: int = 0,
        causal: bool = True,
        training: bool = True
    ) -> Tuple[jnp.ndarray, Optional[jnp.ndarray], Optional[jnp.ndarray]]:
        """Forward pass of Gryphon hybrid block.
        
        Args:
            hidden_states: Input tensor [batch, seq_len, d_model]
            attention_mask: Optional attention mask [batch, seq_len]
            position_ids: Position indices [batch, seq_len]
            cos_freqs: RoPE cosine frequencies
            sin_freqs: RoPE sine frequencies
            s5_state: Previous S5 state for recurrent generation
            layer_idx: Current layer index for coupling frequency control
            causal: Whether to apply causal masking
            training: Whether in training mode
            
        Returns:
            Tuple of (output, final_s5_state, coupling_info)
        """
        # === Phase 1: S5 Sequential Processing ===
        # S5 processes the sequence to extract local/temporal features
        # and build rich contextual representations at each position
        
        if self.config.use_gradient_checkpointing and training:
            s5_output, final_s5_state = jax.checkpoint(self.s5_block)(
                hidden_states, training=training, s5_state=s5_state
            )
        else:
            s5_output, final_s5_state = self.s5_block(
                hidden_states, training=training, s5_state=s5_state
            )
        
        # Optional intermediate normalization for numerical stability
        if self.intermediate_norm is not None:
            s5_output = self.intermediate_norm(s5_output)
        
        # === Phase 2: BigBird Global Attention ===
        # BigBird takes the S5-enriched representations and performs
        # sparse global attention to route information across the sequence
        
        if self.config.use_gradient_checkpointing and training:
            bigbird_output = jax.checkpoint(self.bigbird_block)(
                s5_output,
                attention_mask=attention_mask,
                position_ids=position_ids,
                cos_freqs=cos_freqs,
                sin_freqs=sin_freqs,
                causal=causal,
                training=training
            )
        else:
            bigbird_output = self.bigbird_block(
                s5_output,
                attention_mask=attention_mask,
                position_ids=position_ids,
                cos_freqs=cos_freqs,
                sin_freqs=sin_freqs,
                causal=causal,
                training=training
            )
        
        # === Phase 3: Global Token Coupling ===
        # Apply global token coupling to modify S5 state for next layer
        coupling_info = None
        if self.global_coupling is not None:
            modified_s5_state, coupling_info = self.global_coupling(
                bigbird_output=bigbird_output,
                s5_state=final_s5_state,
                layer_idx=layer_idx,
                training=training
            )
            final_s5_state = modified_s5_state
        
        return bigbird_output, final_s5_state, coupling_info


class GryphonLayer(nn.Module):
    """Complete Gryphon layer with multiple hybrid blocks.
    
    Allows for multiple S5 and BigBird blocks per layer for increased
    processing depth while maintaining the hybrid architecture benefits.
    """
    
    config: GryphonConfig
    layer_idx: int
    
    def setup(self):
        """Initialize layer components."""
        # Create multiple Gryphon blocks per layer
        self.gryphon_blocks = [
            GryphonBlock(config=self.config, name=f'gryphon_block_{i}')
            for i in range(max(1, self.config.s5_blocks_per_layer))
        ]
        
        # Layer-wise learning rate scaling (optional)
        self.layer_scale = self.param(
            'layer_scale',
            nn.initializers.ones,
            (self.config.d_model,)
        ) if hasattr(self.config, 'use_layer_scale') and self.config.use_layer_scale else None
    
    def __call__(
        self,
        hidden_states: jnp.ndarray,
        attention_mask: Optional[jnp.ndarray] = None,
        position_ids: Optional[jnp.ndarray] = None,
        cos_freqs: Optional[jnp.ndarray] = None,
        sin_freqs: Optional[jnp.ndarray] = None,
        s5_states: Optional[list] = None,
        causal: bool = True,
        training: bool = True
    ) -> Tuple[jnp.ndarray, Optional[list]]:
        """Forward pass of Gryphon layer.
        
        Args:
            hidden_states: Input tensor [batch, seq_len, d_model]
            attention_mask: Optional attention mask [batch, seq_len]
            position_ids: Position indices [batch, seq_len]
            cos_freqs: RoPE cosine frequencies
            sin_freqs: RoPE sine frequencies
            s5_states: List of S5 states for each block
            causal: Whether to apply causal masking
            training: Whether in training mode
            
        Returns:
            Tuple of (output, final_s5_states)
        """
        current_states = hidden_states
        final_s5_states = []
        
        # Process through each Gryphon block
        for i, gryphon_block in enumerate(self.gryphon_blocks):
            # Get S5 state for this block
            block_s5_state = s5_states[i] if s5_states is not None else None
            
            # Forward pass
            current_states, final_s5_state = gryphon_block(
                current_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                cos_freqs=cos_freqs,
                sin_freqs=sin_freqs,
                s5_state=block_s5_state,
                causal=causal,
                training=training
            )
            
            final_s5_states.append(final_s5_state)
        
        # Apply layer scaling if configured
        if self.layer_scale is not None:
            current_states = current_states * self.layer_scale
        
        return current_states, final_s5_states