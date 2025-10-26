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

# Robust imports to support both package-relative and direct-module usage
try:
    from .gryphon_config import GryphonConfig
    from .bigbird_attention import BigBirdSparseAttention, BigBirdMLP
    from ..s5 import ValkyrieS5
    from ..modules import RMSNorm
except Exception:
    try:
        from gryphon_config import GryphonConfig
        from bigbird_attention import BigBirdSparseAttention, BigBirdMLP
    except Exception:
        import importlib.util, os, sys
        _cur_dir = os.path.dirname(os.path.abspath(__file__))
        
        def _load_module(_name: str, _file: str):
            _path = os.path.join(_cur_dir, _file)
            spec = importlib.util.spec_from_file_location(_name, _path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            sys.modules[_name] = mod
            return mod
        
        GryphonConfig = _load_module("gryphon_config", "gryphon_config.py").GryphonConfig
        _bb = _load_module("bigbird_attention", "bigbird_attention.py")
        BigBirdSparseAttention = getattr(_bb, "BigBirdSparseAttention")
        BigBirdMLP = getattr(_bb, "BigBirdMLP")
    
    # Load s5.py and modules.py from src/model
    try:
        from s5 import ValkyrieS5
        from modules import RMSNorm
    except Exception:
        import importlib.util, os
        _model_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        _s5_path = os.path.join(_model_dir, "s5.py")
        _modules_path = os.path.join(_model_dir, "modules.py")
        
        spec_s5 = importlib.util.spec_from_file_location("s5", _s5_path)
        s5mod = importlib.util.module_from_spec(spec_s5)
        spec_s5.loader.exec_module(s5mod)
        ValkyrieS5 = s5mod.ValkyrieS5
        
        spec_modules = importlib.util.spec_from_file_location("modules", _modules_path)
        modulesmod = importlib.util.module_from_spec(spec_modules)
        spec_modules.loader.exec_module(modulesmod)
        RMSNorm = modulesmod.RMSNorm


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
        """
        # Pre-normalization
        normed_input = self.attention_norm(hidden_states)
        
        # Sparse attention
        attn_output = self.attention(
            normed_input,
            attention_mask=attention_mask,
            position_ids=position_ids,
            cos_freqs=cos_freqs,
            sin_freqs=sin_freqs,
            causal=causal,
            training=training
        )
        
        if training:
            attn_output = self.attn_dropout(attn_output)
        
        # Residual connection
        hidden_states = hidden_states + attn_output
        
        # Feed-forward network with pre-norm
        ffn_input = self.ffn_norm(hidden_states)
        ffn_output = self.mlp(ffn_input, training=training)
        
        if training:
            ffn_output = self.ffn_dropout(ffn_output)
        
        # Residual connection
        output = hidden_states + ffn_output
        
        return output


class GryphonLayer(nn.Module):
    """Hybrid Gryphon layer combining S5 and BigBird blocks."""
    
    config: GryphonConfig
    layer_idx: int
    
    def setup(self):
        self.s5_block = S5Block(config=self.config)
        self.bigbird_block = BigBirdBlock(config=self.config)
        self.global_coupling = GlobalTokenCoupling(config=self.config)
    
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
        """Forward pass of hybrid layer.
        
        Returns:
            Tuple of (output, final_s5_states)
        """
        # S5 processing
        s5_output, final_s5_state = self.s5_block(
            hidden_states,
            training=training,
            s5_state=s5_states[0] if s5_states is not None and len(s5_states) > 0 else None
        )
        
        # BigBird processing
        attn_output = self.bigbird_block(
            s5_output,
            attention_mask=attention_mask,
            position_ids=position_ids,
            cos_freqs=cos_freqs,
            sin_freqs=sin_freqs,
            causal=causal,
            training=training
        )
        
        # Global token coupling
        modified_s5_state, coupling_info = self.global_coupling(
            attn_output,
            s5_state=final_s5_state,
            layer_idx=self.layer_idx,
            training=training
        )
        
        # Return final outputs
        return attn_output, [modified_s5_state]


# Backward-compatibility alias: historical API exposed GryphonBlock
GryphonBlock = GryphonLayer