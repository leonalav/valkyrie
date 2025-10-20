"""
Core building blocks for HRM JAX implementation.

Includes RMSNorm, SwiGLU, and TransformerBlock (post-norm) matching the PyTorch implementation.
Optimized for TPU with proper initialization and efficient operations.
"""

import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Optional, Callable
import math

from .initializers import truncated_lecun_normal, zeros_init
from .rotary import apply_rotary


def rms_norm(
    x: jnp.ndarray,
    weight: jnp.ndarray,
    eps: float = 1e-6
) -> jnp.ndarray:
    """
    Root Mean Square Layer Normalization.
    
    Functional implementation matching the PyTorch version.
    More efficient than standard LayerNorm for TPU.
    
    Args:
        x: Input tensor [..., hidden_size]
        weight: Scale parameters [hidden_size]
        eps: Small constant for numerical stability
        
    Returns:
        Normalized tensor with same shape as input
    """
    # Compute RMS over the last dimension
    variance = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
    x_normalized = x * jax.lax.rsqrt(variance + eps)
    # Preserve input dtype
    return (x_normalized * weight).astype(x.dtype)


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization module.
    
    More efficient than standard LayerNorm, commonly used in modern LLMs.
    Matches the PyTorch implementation exactly.
    """
    
    hidden_size: int
    eps: float = 1e-6
    dtype: jnp.dtype = jnp.float32
    
    def setup(self):
        # Initialize weight to ones (no bias in RMSNorm)
        def ones_init(key, shape, dtype):
            return jnp.ones(shape, dtype=dtype)
        
        self.weight = self.param(
            'weight',
            ones_init,
            (self.hidden_size,),
            self.dtype
        )
    
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Apply RMS normalization."""
        return rms_norm(x, self.weight, self.eps)


class SwiGLU(nn.Module):
    """
    SwiGLU activation function (Swish-Gated Linear Unit).
    
    Combines SiLU (Swish) activation with gating mechanism.
    Matches the PyTorch implementation with proper intermediate dimension handling.
    """
    
    hidden_size: int
    intermediate_size: Optional[int] = None
    expansion_factor: float = 4.0
    dtype: jnp.dtype = jnp.float32
    
    def setup(self):
        # Calculate intermediate size with proper alignment (multiple of 256)
        if self.intermediate_size is None:
            intermediate_size = int(self.hidden_size * self.expansion_factor)
            # Round to nearest multiple of 256 for efficiency
            intermediate_size = ((intermediate_size + 255) // 256) * 256
        else:
            intermediate_size = self.intermediate_size
        
        # Gate and up projections combined (like PyTorch implementation)
        self.gate_up_proj = nn.Dense(
            features=2 * intermediate_size,  # Fixed: use local variable instead of self.intermediate_size
            use_bias=False,
            dtype=self.dtype,
            kernel_init=truncated_lecun_normal
        )
        
        self.down_proj = nn.Dense(
            features=self.hidden_size,
            use_bias=False,
            dtype=self.dtype,
            kernel_init=truncated_lecun_normal
        )
    
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Apply SwiGLU transformation.
        
        Args:
            x: Input tensor [..., hidden_size]
            
        Returns:
            Output tensor [..., hidden_size]
        """
        # Combined gate and up projection
        gate_up = self.gate_up_proj(x)  # [..., 2 * intermediate_size]
        
        # Split into gate and up components
        gate, up = jnp.split(gate_up, 2, axis=-1)  # Each: [..., intermediate_size]
        
        # Apply SiLU (Swish) to gate and multiply with up
        # SiLU(x) = x * sigmoid(x)
        gated = jax.nn.silu(gate) * up
        
        # Down projection
        output = self.down_proj(gated)
        
        return output


class TransformerBlock(nn.Module):
    """
    Transformer block with post-normalization.
    
    Architecture: x -> Attention -> Add -> RMSNorm -> SwiGLU -> Add -> RMSNorm
    This is the "post-norm" variant used in the HRM implementation.
    """
    
    hidden_size: int
    num_heads: int
    num_key_value_heads: Optional[int] = None
    intermediate_size: Optional[int] = None
    head_dim: Optional[int] = None
    causal: bool = True
    eps: float = 1e-6
    dtype: jnp.dtype = jnp.float32
    
    def setup(self):
        # Import here to avoid circular imports
        from .attention import Attention
        
        # Calculate head dimensions
        head_dim = self.head_dim
        if head_dim is None:
            assert self.hidden_size % self.num_heads == 0
            head_dim = self.hidden_size // self.num_heads
        
        num_key_value_heads = self.num_key_value_heads
        if num_key_value_heads is None:
            num_key_value_heads = self.num_heads
        
        # Self-attention layer
        self.self_attn = Attention(
            hidden_size=self.hidden_size,
            num_heads=self.num_heads,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
            causal=self.causal,
            dtype=self.dtype
        )
        
        # MLP layer (SwiGLU)
        self.mlp = SwiGLU(
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            dtype=self.dtype
        )
        
        # Layer norms (post-norm: after attention and MLP)
        self.attention_norm = RMSNorm(
            hidden_size=self.hidden_size,
            eps=self.eps,
            dtype=self.dtype
        )
        
        self.ffn_norm = RMSNorm(
            hidden_size=self.hidden_size,
            eps=self.eps,
            dtype=self.dtype
        )
    
    def __call__(
        self,
        x: jnp.ndarray,
        cos: Optional[jnp.ndarray] = None,
        sin: Optional[jnp.ndarray] = None,
        attention_mask: Optional[jnp.ndarray] = None
    ) -> jnp.ndarray:
        """
        Apply transformer block.
        
        Args:
            x: Input tensor [batch, seq_len, hidden_size]
            cos: Cosine rotary embeddings [seq_len, head_dim]
            sin: Sine rotary embeddings [seq_len, head_dim]
            attention_mask: Attention mask [batch, seq_len, seq_len]
            
        Returns:
            Output tensor [batch, seq_len, hidden_size]
        """
        # Self-attention with residual connection
        attn_output = self.self_attn(x, cos=cos, sin=sin, attention_mask=attention_mask)
        x = x + attn_output
        
        # Post-attention normalization
        x = self.attention_norm(x)
        
        # MLP with residual connection
        mlp_output = self.mlp(x)
        x = x + mlp_output
        
        # Post-MLP normalization
        x = self.ffn_norm(x)
        
        return x


# Utility function for creating causal attention mask
def make_causal_mask(seq_len: int, dtype: jnp.dtype = jnp.float32) -> jnp.ndarray:
    """
    Create causal attention mask.
    
    Args:
        seq_len: Sequence length
        dtype: Data type for the mask
        
    Returns:
        Causal mask [seq_len, seq_len] with -inf for masked positions
    """
    mask = jnp.tril(jnp.ones((seq_len, seq_len), dtype=dtype))
    mask = jnp.where(mask == 0, -jnp.inf, 0.0)
    return mask


# Utility function for finding multiples (from PyTorch implementation)
def find_multiple(n: int, k: int) -> int:
    """
    Find the smallest multiple of k that is >= n.
    
    Used for aligning intermediate dimensions to hardware-friendly sizes.
    
    Args:
        n: Target number
        k: Multiple to find
        
    Returns:
        Smallest multiple of k that is >= n
    """
    if n % k == 0:
        return n
    return n + k - (n % k)