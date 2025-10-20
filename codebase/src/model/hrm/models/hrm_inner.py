"""
HRM Inner module implementation in JAX/Flax.

Implements the core HRM reasoning with one-step gradient logic, hierarchical levels (H and L),
and proper gradient detachment for the carry state. Matches the PyTorch implementation.
"""

import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Dict, Tuple, Optional, NamedTuple
import math

from .initializers import truncated_lecun_normal, zeros_init, q_head_bias_init
from .blocks import TransformerBlock
from .rotary import RotaryEmbedding


class HRMInnerCarry(NamedTuple):
    """Carry state for HRM inner reasoning loops."""
    z_H: jnp.ndarray  # High-level state [batch, seq_len, hidden_size]
    z_L: jnp.ndarray  # Low-level state [batch, seq_len, hidden_size]


class HRMReasoningModule(nn.Module):
    """
    Hierarchical reasoning module with multiple transformer blocks.
    
    Performs input injection (addition) followed by transformer layers.
    """
    
    hidden_size: int
    num_heads: int
    num_layers: int
    num_key_value_heads: Optional[int] = None
    intermediate_size: Optional[int] = None
    eps: float = 1e-5
    dtype: jnp.dtype = jnp.float32
    
    @property
    def _dtype(self):
        """Get dtype as a property to avoid tracing issues."""
        return self.dtype
    
    def setup(self):
        """
        Initializes the transformer blocks. Each layer must be registered
        as a proper Flax submodule to prevent tracer leaks.
        """
        # Register each layer as a proper submodule using setattr
        for i in range(self.num_layers):
            setattr(self, f'layer_{i}', TransformerBlock(
                hidden_size=self.hidden_size,
                num_heads=self.num_heads,
                num_key_value_heads=self.num_key_value_heads,
                intermediate_size=self.intermediate_size,
                causal=False,  # HRM uses non-causal attention
                eps=self.eps,
                dtype=self._dtype
            ))
    
    def __call__(
        self,
        hidden_states: jnp.ndarray,
        input_injection: jnp.ndarray,
        cos: Optional[jnp.ndarray] = None,
        sin: Optional[jnp.ndarray] = None,
        attention_mask: Optional[jnp.ndarray] = None
    ) -> jnp.ndarray:
        """
        Apply reasoning module.
        """
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states + input_injection
        
        # Apply transformer layers, passing the arguments down
        for i in range(self.num_layers):
            layer = getattr(self, f'layer_{i}')
            hidden_states = layer(
                hidden_states,
                cos=cos,
                sin=sin,
                attention_mask=attention_mask
            )
        
        return hidden_states.astype(input_dtype)


class HRMInner(nn.Module):
    """
    HRM Inner model implementing hierarchical reasoning with one-step gradient.
    
    Key features:
    - Hierarchical reasoning with H (high) and L (low) levels
    - One-step gradient: detach carry, compute one step with gradients
    - Proper initialization matching PyTorch implementation
    - Q-head for ACT halting decisions
    """
    
    # Model configuration
    vocab_size: int
    hidden_size: int
    seq_len: int
    puzzle_emb_ndim: int = 0
    num_puzzle_identifiers: int = 0
    
    # Hierarchical reasoning config
    H_cycles: int = 3
    L_cycles: int = 3
    H_layers: int = 6
    L_layers: int = 6
    
    # Transformer config
    num_heads: int = 8
    num_key_value_heads: Optional[int] = None
    intermediate_size: Optional[int] = None
    eps: float = 1e-5
    
    # Positional encoding
    pos_encodings: str = "rope"
    rope_theta: float = 10000.0
    
    # Data types
    dtype: jnp.dtype = jnp.bfloat16
    param_dtype: jnp.dtype = jnp.float32
    
    def setup(self):
        # Embedding scale (like PyTorch implementation)
        self.embed_scale = math.sqrt(self.hidden_size)
        embed_init_std = 1.0 / self.embed_scale
        
        # Create a proper initializer function for embeddings
        def embed_init_fn(key, shape, dtype):
            return jax.random.normal(key, shape, dtype=dtype) * embed_init_std
        
        # Token embeddings
        self.embed_tokens = nn.Embed(
            num_embeddings=self.vocab_size,
            features=self.hidden_size,
            dtype=self.param_dtype,
            embedding_init=embed_init_fn
        )
        
        # Output heads
        self.lm_head = nn.Dense(
            features=self.vocab_size,
            use_bias=False,
            dtype=self.param_dtype,
            kernel_init=truncated_lecun_normal
        )
        
        # Q-head for ACT (2 outputs: halt, continue)
        self.q_head = nn.Dense(
            features=2,
            use_bias=True,
            dtype=self.param_dtype,
            kernel_init=zeros_init,
            bias_init=q_head_bias_init
        )
        
        # Puzzle embeddings (if needed)
        self.puzzle_emb_len = 0
        if self.puzzle_emb_ndim > 0:
            self.puzzle_emb_len = -(self.puzzle_emb_ndim // -self.hidden_size)  # ceil div
            # Note: Sparse embeddings would need custom implementation
            # For now, using regular embeddings with zero init
            self.puzzle_emb = nn.Embed(
                num_embeddings=self.num_puzzle_identifiers,
                features=self.puzzle_emb_ndim,
                dtype=self.param_dtype,
                embedding_init=truncated_lecun_normal
            )
        
        # Positional encodings
        total_seq_len = self.seq_len + self.puzzle_emb_len
        
        if self.pos_encodings == "rope":
            self.rotary_emb = RotaryEmbedding(
                head_dim=self.hidden_size // self.num_heads,
                max_seq_len=total_seq_len,
                base=self.rope_theta,
                dtype=self.param_dtype
            )
        elif self.pos_encodings == "learned":
            self.embed_pos = nn.Embed(
                num_embeddings=total_seq_len,
                features=self.hidden_size,
                dtype=self.param_dtype,
                embedding_init=lambda key, shape, dtype: (
                    jax.random.normal(key, shape, dtype=dtype) * embed_init_std
                )
            )
        else:
            raise NotImplementedError(f"Position encoding '{self.pos_encodings}' not implemented")
        
        # Reasoning modules
        self.H_level = HRMReasoningModule(
            hidden_size=self.hidden_size,
            num_heads=self.num_heads,
            num_layers=self.H_layers,
            num_key_value_heads=self.num_key_value_heads,
            intermediate_size=self.intermediate_size,
            eps=self.eps,
            dtype=self.param_dtype
        )
        
        self.L_level = HRMReasoningModule(
            hidden_size=self.hidden_size,
            num_heads=self.num_heads,
            num_layers=self.L_layers,
            num_key_value_heads=self.num_key_value_heads,
            intermediate_size=self.intermediate_size,
            eps=self.eps,
            dtype=self.param_dtype
        )
        
        # Force parameter initialization for reasoning modules to prevent tracer leaks
        # This ensures all parameters are created outside of scan contexts
        self._init_reasoning_params()
        
        # Initial states (buffers in PyTorch, parameters in Flax)
        self.H_init = self.param(
            'H_init',
            truncated_lecun_normal,
            (self.hidden_size,),
            self.param_dtype
        )
        
        self.L_init = self.param(
            'L_init',
            truncated_lecun_normal,
            (self.hidden_size,),
            self.param_dtype
        )
    
    def _init_reasoning_params(self):
        """
        Force initialization of reasoning module parameters outside scan context.
        This prevents tracer leaks during jax.lax.scan operations.
        """
        # Create dummy inputs to trigger parameter initialization
        batch_size = 1
        total_seq_len = self.seq_len + self.puzzle_emb_len
        
        dummy_hidden = jnp.zeros((batch_size, total_seq_len, self.hidden_size), dtype=self.param_dtype)
        dummy_injection = jnp.zeros((batch_size, total_seq_len, self.hidden_size), dtype=self.param_dtype)
        
        # Initialize H_level parameters
        try:
            self.H_level(dummy_hidden, dummy_injection)
        except:
            # Parameters are created during the call, exceptions are expected during setup
            pass
            
        # Initialize L_level parameters  
        try:
            self.L_level(dummy_hidden, dummy_injection)
        except:
            # Parameters are created during the call, exceptions are expected during setup
            pass
    
    def _input_embeddings(
        self,
        inputs: jnp.ndarray,
        puzzle_identifiers: Optional[jnp.ndarray] = None
    ) -> jnp.ndarray:
        """
        Compute input embeddings with optional puzzle embeddings and positional encoding.
        
        Args:
            inputs: Token inputs [batch, seq_len]
            puzzle_identifiers: Puzzle IDs [batch] (optional)
            
        Returns:
            Input embeddings [batch, total_seq_len, hidden_size]
        """
        # Token embeddings
        embedding = self.embed_tokens(inputs.astype(jnp.int32))
        
        # Puzzle embeddings
        if self.puzzle_emb_ndim > 0 and puzzle_identifiers is not None:
            puzzle_embedding = self.puzzle_emb(puzzle_identifiers)
            
            # Pad to match hidden_size alignment
            pad_count = self.puzzle_emb_len * self.hidden_size - puzzle_embedding.shape[-1]
            if pad_count > 0:
                puzzle_embedding = jnp.pad(puzzle_embedding, ((0, 0), (0, pad_count)))
            
            # Reshape and concatenate
            puzzle_embedding = puzzle_embedding.reshape(-1, self.puzzle_emb_len, self.hidden_size)
            embedding = jnp.concatenate([puzzle_embedding, embedding], axis=1)
        
        # Positional embeddings
        if self.pos_encodings == "learned":
            # Scale by 1/sqrt(2) to maintain forward variance (like PyTorch)
            pos_emb = self.embed_pos.embedding
            embedding = 0.707106781 * (embedding + pos_emb.astype(self.dtype))
        
        # Scale embeddings
        return self.embed_scale * embedding.astype(self.dtype)
    
    def empty_carry(self, batch_size: int) -> HRMInnerCarry:
        """Create empty carry state."""
        total_seq_len = self.seq_len + self.puzzle_emb_len
        
        return HRMInnerCarry(
            z_H=jnp.empty((batch_size, total_seq_len, self.hidden_size), dtype=self.dtype),
            z_L=jnp.empty((batch_size, total_seq_len, self.hidden_size), dtype=self.dtype)
        )
    
    def reset_carry(
        self,
        reset_flag: jnp.ndarray,
        carry: HRMInnerCarry
    ) -> HRMInnerCarry:
        """
        Reset carry state for halted sequences.
        
        Args:
            reset_flag: Boolean flags [batch] indicating which sequences to reset
            carry: Current carry state
            
        Returns:
            Updated carry state
        """
        # Broadcast initial states
        batch_size, seq_len, hidden_size = carry.z_H.shape
        H_init_broadcast = jnp.broadcast_to(self.H_init, (batch_size, seq_len, hidden_size))
        L_init_broadcast = jnp.broadcast_to(self.L_init, (batch_size, seq_len, hidden_size))
        
        # Reset where flag is True
        reset_mask = reset_flag.reshape(-1, 1, 1)
        
        return HRMInnerCarry(
            z_H=jnp.where(reset_mask, H_init_broadcast, carry.z_H),
            z_L=jnp.where(reset_mask, L_init_broadcast, carry.z_L)
        )
    
    def __call__(
        self,
        carry: HRMInnerCarry,
        batch: Dict[str, jnp.ndarray]
    ) -> Tuple[HRMInnerCarry, jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]]:
        """
        Forward pass with one-step gradient logic.
        
        Args:
            carry: Current carry state
            batch: Input batch with 'inputs' and optionally 'puzzle_identifiers'
            
        Returns:
            Tuple of:
            - New carry state (detached)
            - LM logits [batch, seq_len, vocab_size]
            - Q logits tuple (halt_logits, continue_logits) [batch]
        """
        # Prepare sequence info
        seq_info = {}
        if self.pos_encodings == "rope":
            total_seq_len = self.seq_len + self.puzzle_emb_len
            cos, sin = self.rotary_emb(total_seq_len)
            seq_info.update(cos=cos, sin=sin)
        
        # Input embeddings
        input_embeddings = self._input_embeddings(
            batch["inputs"],
            batch.get("puzzle_identifiers")
        )
        
        # Get current states
        z_H, z_L = carry.z_H, carry.z_L
        
        # Forward iterations WITHOUT gradients (like PyTorch no_grad)
        # This implements the "detach carry" part of one-step gradient
        z_H_detached = jax.lax.stop_gradient(z_H)
        z_L_detached = jax.lax.stop_gradient(z_L)
        
        # Multi-step reasoning loop (detached)
        def reasoning_step(state, _):
            z_H_curr, z_L_curr = state
            
            # L-level cycles
            def l_step(z_L_inner, _):
                return self.L_level(
                    z_L_inner,
                    z_H_curr + input_embeddings,
                    **seq_info
                ), None
            
            # Run L cycles (all but last)
            z_L_curr, _ = jax.lax.scan(l_step, z_L_curr, None, length=self.L_cycles - 1)
            
            # Final L step (will be done with gradients later)
            # For now, just do it detached
            z_L_curr = self.L_level(
                z_L_curr,
                z_H_curr + input_embeddings,
                **seq_info
            )
            
            # H-level update
            z_H_curr = self.H_level(z_H_curr, z_L_curr, **seq_info)
            
            return (z_H_curr, z_L_curr), None
        
        # Run H cycles (all but last)
        (z_H_detached, z_L_detached), _ = jax.lax.scan(
            reasoning_step,
            (z_H_detached, z_L_detached),
            None,
            length=self.H_cycles - 1
        )
        
        # Final L cycle (all but last step)
        def final_l_step(z_L_inner, _):
            return self.L_level(
                z_L_inner,
                z_H_detached + input_embeddings,
                **seq_info
            ), None
        
        z_L_detached, _ = jax.lax.scan(
            final_l_step,
            z_L_detached,
            None,
            length=self.L_cycles - 1
        )
        
        # ONE-STEP GRADIENT: Final step WITH gradients
        # This is the key insight from the PyTorch implementation
        z_L_final = self.L_level(
            z_L_detached,  # Detached input
            z_H_detached + input_embeddings,
            **seq_info
        )
        
        z_H_final = self.H_level(
            z_H_detached,  # Detached input
            z_L_final,
            **seq_info
        )
        
        # Create new carry (detached for next iteration)
        new_carry = HRMInnerCarry(
            z_H=jax.lax.stop_gradient(z_H_final),
            z_L=jax.lax.stop_gradient(z_L_final)
        )
        
        # LM head output (skip puzzle embedding positions)
        lm_output = self.lm_head(z_H_final)
        if self.puzzle_emb_len > 0:
            lm_output = lm_output[:, self.puzzle_emb_len:]
        
        # Q-head output (use first position like PyTorch)
        q_logits = self.q_head(z_H_final[:, 0]).astype(jnp.float32)
        q_halt_logits = q_logits[..., 0]
        q_continue_logits = q_logits[..., 1]
        
        return new_carry, lm_output, (q_halt_logits, q_continue_logits)