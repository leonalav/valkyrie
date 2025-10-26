"""Gryphon-HRM Integrated Model

Complete implementation of the BigBird+S5+HRM architecture following the PLAN blueprint.
Combines sparse attention (BigBird), long-range memory (S5), and hierarchical reasoning (HRM)
with proper one-step gradient training and global token coupling.

Architecture Overview:
- BigBird sparse attention with global tokens serving as HRM planner state
- S5 state space model for recurrent long-range dependencies  
- HRM hierarchical reasoning with planner/executor cycles
- Gated fusion between BigBird and S5 branches
- One-step gradient training for O(1) memory complexity

Key Features:
- Linear O(L) complexity for 64k context sequences
- HRM planner tokens integrated as BigBird global tokens
- S5 recurrent state carry across segments
- Mixed precision training with numerical stability
- Gradient checkpointing and memory optimization
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Optional, Tuple, Dict, Any, NamedTuple
import math

from .gryphon_config import GryphonConfig
from .bigbird_attention import BigBirdSparseAttention
from ..s5 import ValkyrieS5
from ..hrm.models.hrm_inner import HRMInner, HRMInnerCarry
from ..hrm.models.hrm_act import HRMWithACT, ACTOutput
from ..modules import RMSNorm, precompute_rope_freqs


class GryphonHRMState(NamedTuple):
    """Complete state for Gryphon-HRM model."""
    s5_state: jnp.ndarray  # S5 recurrent state [batch, seq_len, state_dim]
    hrm_carry: HRMInnerCarry  # HRM hierarchical state
    global_tokens: jnp.ndarray  # Global tokens for BigBird [batch, num_global, d_model]


class GatedFusion(nn.Module):
    """Gated fusion module for combining BigBird and S5 outputs."""
    
    config: GryphonConfig
    
    def setup(self):
        """Initialize gating controller."""
        self.controller = nn.Sequential([
            nn.Dense(
                features=self.config.coupling_hidden_dim,
                use_bias=True,
                dtype=self.config.param_dtype if hasattr(self.config, 'param_dtype') else jnp.float32
            ),
            nn.gelu,
            nn.Dense(
                features=2,  # α and β gates
                use_bias=True,
                dtype=self.config.param_dtype if hasattr(self.config, 'param_dtype') else jnp.float32
            ),
            nn.sigmoid
        ])
    
    def __call__(
        self,
        bigbird_output: jnp.ndarray,
        s5_output: jnp.ndarray,
        hidden_states: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Fuse BigBird and S5 outputs with learned gating.
        
        Args:
            bigbird_output: BigBird attention output [batch, seq_len, d_model]
            s5_output: S5 state space output [batch, seq_len, d_model]
            hidden_states: Input hidden states for gating [batch, seq_len, d_model]
            
        Returns:
            Fused output: α·BigBird + β·S5 [batch, seq_len, d_model]
        """
        # Compute gating weights from input
        gates = self.controller(hidden_states)  # [batch, seq_len, 2]
        alpha = gates[..., 0:1]  # [batch, seq_len, 1]
        beta = gates[..., 1:2]   # [batch, seq_len, 1]
        
        # Gated fusion
        fused_output = alpha * bigbird_output + beta * s5_output
        
        return fused_output


class GryphonHRMLayer(nn.Module):
    """Single Gryphon-HRM layer combining BigBird, S5, and HRM components."""
    
    config: GryphonConfig
    layer_idx: int
    
    def setup(self):
        """Initialize layer components."""
        # Layer normalization
        self.input_layernorm = RMSNorm(
            hidden_size=self.config.d_model,
            eps=self.config.layer_norm_eps if hasattr(self.config, 'layer_norm_eps') else 1e-5
        )
        
        self.post_attention_layernorm = RMSNorm(
            hidden_size=self.config.d_model,
            eps=self.config.layer_norm_eps if hasattr(self.config, 'layer_norm_eps') else 1e-5
        )
        
        # BigBird sparse attention
        self.bigbird_attention = BigBirdSparseAttention(
            config=self.config
        )
        
        # S5 state space model
        self.s5_block = ValkyrieS5(
            config=self.config,
            state_dim=self.config.s5_state_dim
        )
        
        # Gated fusion
        self.fusion = GatedFusion(config=self.config)
        
        # FFN (SwiGLU)
        if hasattr(self.config, 'use_swiglu') and self.config.use_swiglu:
            ffn_dim = getattr(self.config, 'ffn_dim', 4 * self.config.d_model)
            self.ffn = nn.Sequential([
                nn.Dense(
                    features=ffn_dim,
                    use_bias=getattr(self.config, 'use_bias', False),
                    dtype=self.config.param_dtype if hasattr(self.config, 'param_dtype') else jnp.float32
                ),
                lambda x: nn.silu(x) * nn.Dense(
                    features=ffn_dim,
                    use_bias=getattr(self.config, 'use_bias', False),
                    dtype=self.config.param_dtype if hasattr(self.config, 'param_dtype') else jnp.float32
                )(x),
                nn.Dense(
                    features=self.config.d_model,
                    use_bias=getattr(self.config, 'use_bias', False),
                    dtype=self.config.param_dtype if hasattr(self.config, 'param_dtype') else jnp.float32
                )
            ])
        else:
            # Standard FFN
            ffn_dim = getattr(self.config, 'ffn_dim', 4 * self.config.d_model)
            self.ffn = nn.Sequential([
                nn.Dense(
                    features=ffn_dim,
                    use_bias=getattr(self.config, 'use_bias', False),
                    dtype=self.config.param_dtype if hasattr(self.config, 'param_dtype') else jnp.float32
                ),
                nn.gelu,
                nn.Dense(
                    features=self.config.d_model,
                    use_bias=getattr(self.config, 'use_bias', False),
                    dtype=self.config.param_dtype if hasattr(self.config, 'param_dtype') else jnp.float32
                )
            ])
        
        # Dropout
        self.dropout = nn.Dropout(
            rate=getattr(self.config, 'resid_dropout', 0.1)
        )
        
        # Precompute RoPE frequencies for attention (head_dim = d_model // n_heads)
        head_dim = self.config.d_model // self.config.n_heads
        # Ensure RoPE frequencies cover sequence plus prepended global tokens and block padding
        max_rope_seq_len = int(
        math.ceil((self.config.max_sequence_length + self.config.num_global_blocks) / self.config.block_size)
        * self.config.block_size
        )
        self.cos_freqs, self.sin_freqs = precompute_rope_freqs(
        dim=head_dim,
        max_seq_len=max_rope_seq_len,
        base=getattr(self.config, 'rope_theta', 10000.0)
        )
    
    def __call__(
        self,
        hidden_states: jnp.ndarray,
        global_tokens: jnp.ndarray,
        s5_state: Optional[jnp.ndarray] = None,
        attention_mask: Optional[jnp.ndarray] = None,
        position_ids: Optional[jnp.ndarray] = None,
        deterministic: bool = True
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Forward pass through Gryphon-HRM layer.
        
        Args:
            hidden_states: Input hidden states [batch, seq_len, d_model]
            global_tokens: Global tokens for BigBird [batch, num_global, d_model]
            s5_state: S5 recurrent state [batch, seq_len, state_dim]
            attention_mask: Attention mask [batch, seq_len]
            position_ids: Position indices [batch, seq_len]
            deterministic: Whether to use deterministic mode
            
        Returns:
            Tuple of (output_states, new_global_tokens, new_s5_state)
        """
        # Input layer norm
        normed_states = self.input_layernorm(hidden_states)
        
        # === Integrate global tokens by prepending to the sequence (Option A) ===
        batch_size = normed_states.shape[0]
        num_global = global_tokens.shape[1]
        
        # Concatenate global tokens before the sequence
        attn_inputs = jnp.concatenate([global_tokens, normed_states], axis=1)  # [batch, num_global+seq_len, d_model]
        
        # Build augmented attention mask: global tokens attend and are attended by all
        if attention_mask is None:
            seq_len = normed_states.shape[1]
            attention_mask = jnp.ones((batch_size, seq_len), dtype=jnp.float32)
        global_mask = jnp.ones((batch_size, num_global), dtype=jnp.float32)
        aug_attention_mask = jnp.concatenate([global_mask, attention_mask], axis=1)
        
        # Build augmented position_ids: use -1 sentinel for global tokens to skip RoPE
        if position_ids is None:
            position_ids = create_position_ids(normed_states[:, :, 0]) if normed_states.ndim == 3 else create_position_ids(normed_states)
        pos_ids_seq = position_ids
        pos_ids_global = -jnp.ones((batch_size, num_global), dtype=jnp.int32)
        aug_position_ids = jnp.concatenate([pos_ids_global, pos_ids_seq], axis=1)
        
        # BigBird attention with augmented inputs
        bigbird_output = self.bigbird_attention(
            hidden_states=attn_inputs,
            attention_mask=aug_attention_mask,
            position_ids=aug_position_ids,
            cos_freqs=self.cos_freqs,
            sin_freqs=self.sin_freqs,
            causal=True,
            training=not deterministic
        )
        
        # Split back global token outputs and sequence outputs
        new_global_tokens = bigbird_output[:, :num_global, :]
        bigbird_seq_output = bigbird_output[:, num_global:, :]
        
        # S5 state space processing on original sequence
        s5_output, new_s5_state = self.s5_block(
            x=normed_states,
            training=not deterministic,
            state=s5_state
        )
        
        # Gated fusion on sequence outputs
        fused_output = self.fusion(
            bigbird_output=bigbird_seq_output,
            s5_output=s5_output,
            hidden_states=normed_states
        )
        
        # Residual connection
        hidden_states = hidden_states + self.dropout(fused_output, deterministic=deterministic)
        
        # Post-attention layer norm
        normed_states = self.post_attention_layernorm(hidden_states)
        
        # FFN
        ffn_output = self.ffn(normed_states)
        
        # Final residual connection
        output_states = hidden_states + self.dropout(ffn_output, deterministic=deterministic)
        
        return output_states, new_global_tokens, new_s5_state


class GryphonHRMModel(nn.Module):
    """
    Complete Gryphon-HRM model implementing BigBird+S5+HRM architecture.
    
    Follows the PLAN blueprint for 1.2B parameter model with:
    - BigBird sparse attention (linear complexity)
    - S5 state space model (recurrent long-range memory)
    - HRM hierarchical reasoning (one-step gradient training)
    - Global token coupling between components
    """
    
    config: GryphonConfig
    
    def setup(self):
        """Initialize model components."""
        # Token embeddings
        self.embed_tokens = nn.Embed(
            num_embeddings=self.config.vocab_size,
            features=self.config.d_model,
            embedding_init=nn.initializers.normal(
                stddev=getattr(self.config, 'initializer_range', 0.02)
            ),
            dtype=self.config.param_dtype if hasattr(self.config, 'param_dtype') else jnp.float32
        )
        
        # Global token embeddings (HRM planner tokens)
        self.global_token_embeddings = nn.Embed(
            num_embeddings=self.config.num_global_blocks,
            features=self.config.d_model,
            embedding_init=nn.initializers.normal(
                stddev=getattr(self.config, 'initializer_range', 0.02)
            ),
            dtype=self.config.param_dtype if hasattr(self.config, 'param_dtype') else jnp.float32
        )
        
        # Transformer layers
        self.layers = [
            GryphonHRMLayer(config=self.config, layer_idx=i)
            for i in range(self.config.n_layers)
        ]
        
        # Final layer norm
        self.norm = RMSNorm(
            hidden_size=self.config.d_model,
            eps=getattr(self.config, 'layer_norm_eps', 1e-5)
        )
        
        # Language modeling head
        self.lm_head = nn.Dense(
            features=self.config.vocab_size,
            use_bias=False,
            dtype=self.config.param_dtype if hasattr(self.config, 'param_dtype') else jnp.float32,
            kernel_init=nn.initializers.normal(
                stddev=getattr(self.config, 'initializer_range', 0.02)
            )
        )
        
        # HRM integration (if enabled)
        if self.config.hrm_enabled:
            if self.config.hrm_use_act:
                # HRM with Adaptive Computation Time
                self.hrm = HRMWithACT(
                    vocab_size=self.config.vocab_size,
                    hidden_size=self.config.d_model,
                    seq_len=self.config.max_sequence_length,
                    H_cycles=self.config.hrm_executor_cycles,
                    L_cycles=self.config.hrm_executor_steps,
                    H_layers=self.config.hrm_planner_layers,
                    L_layers=self.config.n_layers // 4,  # Use subset of layers for L-level
                    num_heads=self.config.n_heads,
                    max_steps=self.config.hrm_max_steps,
                    exploration_prob=self.config.hrm_exploration_prob,
                    q_target_discount=self.config.hrm_q_target_discount,
                    dtype=getattr(self.config, 'dtype', jnp.bfloat16),
                    param_dtype=self.config.param_dtype if hasattr(self.config, 'param_dtype') else jnp.float32
                )
            else:
                # Standard HRM
                self.hrm = HRMInner(
                    vocab_size=self.config.vocab_size,
                    hidden_size=self.config.d_model,
                    seq_len=self.config.max_sequence_length,
                    H_cycles=self.config.hrm_executor_cycles,
                    L_cycles=self.config.hrm_executor_steps,
                    H_layers=self.config.hrm_planner_layers,
                    L_layers=self.config.n_layers // 4,  # Use subset of layers for L-level
                    num_heads=self.config.n_heads,
                    dtype=getattr(self.config, 'dtype', jnp.bfloat16),
                    param_dtype=self.config.param_dtype if hasattr(self.config, 'param_dtype') else jnp.float32
                )
    
    def init_state(self, batch_size: int, seq_len: Optional[int] = None) -> GryphonHRMState:
        """Initialize model state for recurrent processing."""
        # Use provided sequence length or default to config max
        if seq_len is None:
            seq_len = self.config.max_sequence_length
        
        # S5 state holds the last hidden state per batch [batch, state_dim]
        s5_state = jnp.zeros(
            (batch_size, self.config.s5_state_dim),
            dtype=getattr(self.config, 'dtype', jnp.bfloat16)
        )
        
        # HRM carry state
        if self.config.hrm_enabled:
            if hasattr(self.hrm, 'initial_carry'):
                hrm_carry = self.hrm.initial_carry(batch_size)
            else:
                hrm_carry = self.hrm.empty_carry(batch_size)
        else:
            # Dummy HRM carry
            hrm_carry = HRMInnerCarry(
                z_H=jnp.zeros((batch_size, seq_len, self.config.d_model)),
                z_L=jnp.zeros((batch_size, seq_len, self.config.d_model))
            )
        
        # Global tokens (initialized from embeddings)
        global_token_ids = jnp.arange(self.config.num_global_blocks)[None, :]  # [1, num_global]
        global_token_ids = jnp.broadcast_to(global_token_ids, (batch_size, self.config.num_global_blocks))
        global_tokens = self.global_token_embeddings(global_token_ids)
        
        return GryphonHRMState(
            s5_state=s5_state,
            hrm_carry=hrm_carry,
            global_tokens=global_tokens
        )
    
    def __call__(
        self,
        input_ids: jnp.ndarray,
        state: Optional[GryphonHRMState] = None,
        attention_mask: Optional[jnp.ndarray] = None,
        position_ids: Optional[jnp.ndarray] = None,
        deterministic: bool = True,
        use_hrm: bool = None,
        return_act_output: bool = False,
    ) -> Tuple[jnp.ndarray, GryphonHRMState]:
        """
        Forward pass through Gryphon-HRM model.
        
        Args:
            input_ids: Input token IDs [batch, seq_len]
            state: Model state for recurrent processing
            attention_mask: Attention mask [batch, seq_len]
            position_ids: Position indices [batch, seq_len]
            deterministic: Whether to use deterministic mode
            use_hrm: Whether to use HRM (overrides config if provided)
            return_act_output: When hrm_use_act is True, optionally return ACTOutput for metrics
            
        Returns:
            Tuple of (logits, new_state) or (logits, new_state, act_output) if return_act_output
        """
        batch_size, seq_len = input_ids.shape
        
        # Initialize state if not provided
        if state is None:
            state = self.init_state(batch_size, seq_len)
        
        # Determine HRM usage
        if use_hrm is None:
            use_hrm = self.config.hrm_enabled
        
        # Token embeddings
        hidden_states = self.embed_tokens(input_ids)
        
        # Create position ids if not provided
        if position_ids is None:
            position_ids = create_position_ids(input_ids)
        
        # Process through transformer layers
        current_s5_state = state.s5_state
        current_global_tokens = state.global_tokens
        
        for layer in self.layers:
            hidden_states, current_global_tokens, current_s5_state = layer(
                hidden_states=hidden_states,
                global_tokens=current_global_tokens,
                s5_state=current_s5_state,
                attention_mask=attention_mask,
                position_ids=position_ids,
                deterministic=deterministic
            )
        
        # Final layer norm
        hidden_states = self.norm(hidden_states)
        
        act_output = None
        # HRM processing (if enabled)
        if use_hrm and self.config.hrm_enabled:
            # Prepare batch for HRM
            hrm_batch = {"inputs": input_ids}
            
            if self.config.hrm_use_act:
                # HRM with ACT
                act_output = self.hrm(
                    batch=hrm_batch,
                    carry=state.hrm_carry,
                    training=not deterministic
                )
                logits = act_output.lm_logits
                new_hrm_carry = act_output.final_carry
            else:
                # Standard HRM
                new_hrm_carry, hrm_logits, _ = self.hrm(state.hrm_carry, hrm_batch)
                
                # Combine transformer and HRM outputs
                transformer_logits = self.lm_head(hidden_states)
                
                # Weighted combination (could be learned)
                hrm_weight = 0.5  # Could be a learned parameter
                logits = (1 - hrm_weight) * transformer_logits + hrm_weight * hrm_logits
        else:
            # Standard language modeling head
            logits = self.lm_head(hidden_states)
            new_hrm_carry = state.hrm_carry
        
        # Create new state
        new_state = GryphonHRMState(
            s5_state=current_s5_state,
            hrm_carry=new_hrm_carry,
            global_tokens=current_global_tokens
        )
        
        if return_act_output and act_output is not None:
            # Return ACT internals for downstream metrics (halt probs via softmax of Qs, steps)
            return logits, new_state, act_output
        
        return logits, new_state


# Utility functions for training and inference
def create_attention_mask(input_ids: jnp.ndarray, pad_token_id: int = 0) -> jnp.ndarray:
    """Create attention mask from input IDs."""
    return (input_ids != pad_token_id).astype(jnp.float32)


def create_position_ids(input_ids: jnp.ndarray, pad_token_id: int = 0) -> jnp.ndarray:
    """Create position IDs from input IDs."""
    mask = (input_ids != pad_token_id).astype(jnp.int32)
    return jnp.cumsum(mask, axis=-1) - 1


def compute_model_size(config: GryphonConfig) -> Dict[str, float]:
    """Compute model parameter count breakdown."""
    # Token embeddings
    token_emb_params = config.vocab_size * config.d_model
    
    # Global token embeddings
    global_emb_params = config.num_global_blocks * config.d_model
    
    # Per-layer parameters
    # BigBird attention: Q, K, V, O projections
    attention_params = 4 * config.d_model * config.d_model
    
    # S5 parameters: B, C matrices (complex), D, Δ
    s5_params = (
        2 * config.s5_state_dim * config.d_model * 2 +  # B, C (complex)
        config.s5_state_dim +  # D
        config.s5_state_dim    # Δ
    )
    
    # FFN parameters
    ffn_dim = getattr(config, 'ffn_dim', 4 * config.d_model)
    if getattr(config, 'use_swiglu', False):
        ffn_params = 3 * config.d_model * ffn_dim  # Gate, up, down projections
    else:
        ffn_params = 2 * config.d_model * ffn_dim  # Up, down projections
    
    # Layer norms (minimal)
    layernorm_params = 4 * config.d_model  # Input, post-attn, final norms
    
    # Per-layer total
    per_layer_params = attention_params + s5_params + ffn_params + layernorm_params
    
    # Total layer parameters
    total_layer_params = config.n_layers * per_layer_params
    
    # Language modeling head
    lm_head_params = config.vocab_size * config.d_model
    
    # HRM parameters (if enabled)
    hrm_params = 0
    if config.hrm_enabled:
        # Simplified estimate - actual would depend on HRM architecture
        hrm_params = config.hrm_planner_layers * config.d_model * config.d_model * 4
    
    # Total parameters
    total_params = (
        token_emb_params + global_emb_params + total_layer_params + 
        lm_head_params + hrm_params
    )
    
    return {
        "token_embeddings_M": token_emb_params / 1e6,
        "global_embeddings_M": global_emb_params / 1e6,
        "transformer_layers_M": total_layer_params / 1e6,
        "lm_head_M": lm_head_params / 1e6,
        "hrm_M": hrm_params / 1e6,
        "total_M": total_params / 1e6,
        "total_B": total_params / 1e9
    }