"""
Attention mechanisms for HRM JAX implementation.

Includes efficient multi-head attention with rotary embeddings, causal masking,
and TPU-optimized dot_general operations. Matches the PyTorch implementation.
"""

import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Optional, Tuple
import math

from .initializers import truncated_lecun_normal, zeros_init
from .rotary import apply_rotary


class Attention(nn.Module):
    """
    Multi-head attention with rotary positional embeddings.
    
    Optimized for TPU with efficient dot_general operations and proper
    gradient flow. Supports both causal and non-causal attention.
    """
    
    hidden_size: int
    num_heads: int
    num_key_value_heads: Optional[int] = None
    head_dim: Optional[int] = None
    causal: bool = True
    dtype: jnp.dtype = jnp.float32
    
    def setup(self):
        # Calculate dimensions
        if self.head_dim is None:
            assert self.hidden_size % self.num_heads == 0
            self.head_dim = self.hidden_size // self.num_heads
        
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_heads
        
        # Grouped Query Attention support
        assert self.num_heads % self.num_key_value_heads == 0
        self.num_queries_per_kv = self.num_heads // self.num_key_value_heads
        
        # QKV projection (combined for efficiency like PyTorch)
        # Grouped query attention projections
        self.q_proj = nn.Dense(
            features=self.num_heads * self.head_dim,
            use_bias=False,
            dtype=self.dtype,
            kernel_init=truncated_lecun_normal
        )
        
        self.kv_proj = nn.Dense(
            features=2 * self.num_key_value_heads * self.head_dim,
            use_bias=False,
            dtype=self.dtype,
            kernel_init=truncated_lecun_normal
        )
        
        # Output projection
        self.o_proj = nn.Dense(
            features=self.hidden_size,
            use_bias=False,
            dtype=self.dtype,
            kernel_init=truncated_lecun_normal
        )
    
    def __call__(
        self,
        x: jnp.ndarray,
        cos: Optional[jnp.ndarray] = None,
        sin: Optional[jnp.ndarray] = None,
        attention_mask: Optional[jnp.ndarray] = None
    ) -> jnp.ndarray:
        """
        Apply multi-head attention.
        
        Args:
            x: Input tensor [batch, seq_len, hidden_size]
            cos: Cosine rotary embeddings [seq_len, head_dim]
            sin: Sine rotary embeddings [seq_len, head_dim]
            attention_mask: Attention mask [batch, seq_len, seq_len] or [seq_len, seq_len]
            
        Returns:
            Output tensor [batch, seq_len, hidden_size]
        """
        batch_size, seq_len, _ = x.shape
        
        # Separate Q and KV projections
        q = self.q_proj(x)  # [batch, seq_len, num_heads * head_dim]
        kv = self.kv_proj(x)  # [batch, seq_len, 2 * num_kv_heads * head_dim]
        
        # Split KV
        k_size = self.num_key_value_heads * self.head_dim
        k, v = jnp.split(kv, [k_size], axis=-1)
        
        # Reshape to separate heads
        q = q.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.reshape(batch_size, seq_len, self.num_key_value_heads, self.head_dim)
        v = v.reshape(batch_size, seq_len, self.num_key_value_heads, self.head_dim)
        
        # Apply rotary embeddings if provided
        if cos is not None and sin is not None:
            q = apply_rotary(q, cos, sin)
            k = apply_rotary(k, cos, sin)
        
        # Transpose for attention computation: [batch, num_heads, seq_len, head_dim]
        q = jnp.transpose(q, (0, 2, 1, 3))
        k = jnp.transpose(k, (0, 2, 1, 3))
        v = jnp.transpose(v, (0, 2, 1, 3))
        
        # Handle grouped query attention by repeating k, v
        if self.num_queries_per_kv > 1:
            k = jnp.repeat(k, self.num_queries_per_kv, axis=1)
            v = jnp.repeat(v, self.num_queries_per_kv, axis=1)
        
        # Compute attention scores using efficient dot_general
        # Scale by 1/sqrt(head_dim) for stability - ensure scale matches input dtype
        scale = jnp.array(1.0 / math.sqrt(self.head_dim), dtype=x.dtype)
        
        # Attention scores: [batch, num_heads, seq_len, seq_len]
        attn_weights = jax.lax.dot_general(
            q * scale, k,
            dimension_numbers=(((3,), (3,)), ((0, 1), (0, 1)))
        )
        
        # Apply causal mask if needed
        if self.causal:
            causal_mask = jnp.tril(jnp.ones((seq_len, seq_len), dtype=jnp.bool_))
            # Ensure -inf constant matches attention weights dtype
            neg_inf = jnp.array(-jnp.inf, dtype=attn_weights.dtype)
            attn_weights = jnp.where(causal_mask, attn_weights, neg_inf)
        
        # Apply additional attention mask if provided
        if attention_mask is not None:
            if attention_mask.ndim == 2:
                # Broadcast to [batch, num_heads, seq_len, seq_len]
                attention_mask = attention_mask[None, None, :, :]
            elif attention_mask.ndim == 3:
                # Broadcast to [batch, num_heads, seq_len, seq_len]
                attention_mask = attention_mask[:, None, :, :]
            
            # Ensure -inf constant matches attention weights dtype
            neg_inf = jnp.array(-jnp.inf, dtype=attn_weights.dtype)
            attn_weights = jnp.where(attention_mask, attn_weights, neg_inf)
        
        # Softmax over the last dimension
        attn_weights = jax.nn.softmax(attn_weights, axis=-1)
        
        # Apply attention to values using efficient dot_general
        # Output: [batch, num_heads, seq_len, head_dim]
        attn_output = jax.lax.dot_general(
            attn_weights, v,
            dimension_numbers=(((3,), (2,)), ((0, 1), (0, 1)))
        )
        
        # Transpose back and reshape: [batch, seq_len, hidden_size]
        attn_output = jnp.transpose(attn_output, (0, 2, 1, 3))
        attn_output = attn_output.reshape(batch_size, seq_len, self.hidden_size)
        
        # Output projection
        output = self.o_proj(attn_output)
        
        return output


class FlashAttention(nn.Module):
    """
    Flash Attention implementation for memory efficiency.
    
    Note: This is a simplified version. For production use, consider
    using specialized Flash Attention kernels when available.
    """
    
    hidden_size: int
    num_heads: int
    num_key_value_heads: Optional[int] = None
    head_dim: Optional[int] = None
    causal: bool = True
    block_size: int = 128  # Block size for memory-efficient computation
    dtype: jnp.dtype = jnp.float32
    
    def setup(self):
        # Same setup as regular Attention
        if self.head_dim is None:
            assert self.hidden_size % self.num_heads == 0
            self.head_dim = self.hidden_size // self.num_heads
        
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_heads
        
        self.num_queries_per_kv = self.num_heads // self.num_key_value_heads
        
        self.qkv_proj = nn.Dense(
            features=(self.num_heads + 2 * self.num_key_value_heads) * self.head_dim,
            use_bias=False,
            dtype=self.dtype,
            kernel_init=lambda key, shape, dtype: truncated_lecun_normal(key, shape, dtype)
        )
        
        self.o_proj = nn.Dense(
            features=self.hidden_size,
            use_bias=False,
            dtype=self.dtype,
            kernel_init=lambda key, shape, dtype: truncated_lecun_normal(key, shape, dtype)
        )
    
    def __call__(
        self,
        x: jnp.ndarray,
        cos: Optional[jnp.ndarray] = None,
        sin: Optional[jnp.ndarray] = None,
        attention_mask: Optional[jnp.ndarray] = None
    ) -> jnp.ndarray:
        """
        Apply Flash Attention (simplified version).
        
        For very long sequences, this would implement block-wise computation
        to reduce memory usage. For now, falls back to standard attention.
        """
        # For sequences shorter than block_size, use standard attention
        if x.shape[1] <= self.block_size:
            # Use standard attention implementation
            attention = Attention(
                hidden_size=self.hidden_size,
                num_heads=self.num_heads,
                num_key_value_heads=self.num_key_value_heads,
                head_dim=self.head_dim,
                causal=self.causal,
                dtype=self.dtype
            )
            return attention(x, cos=cos, sin=sin, attention_mask=attention_mask)
        
        # TODO: Implement block-wise Flash Attention for longer sequences
        # This would involve:
        # 1. Splitting Q, K, V into blocks
        # 2. Computing attention block by block
        # 3. Maintaining running statistics for numerical stability
        # 4. Combining results from all blocks
        
        # For now, fall back to standard attention with a warning
        import warnings
        warnings.warn(
            f"Sequence length {x.shape[1]} > block_size {self.block_size}. "
            "Using standard attention. Consider implementing block-wise Flash Attention."
        )
        
        attention = Attention(
            hidden_size=self.hidden_size,
            num_heads=self.num_heads,
            num_key_value_heads=self.num_key_value_heads,
            head_dim=self.head_dim,
            causal=self.causal,
            dtype=self.dtype
        )
        return attention(x, cos=cos, sin=sin, attention_mask=attention_mask)


# Utility functions for attention computation
def make_attention_mask(
    seq_len: int,
    causal: bool = True,
    dtype: jnp.dtype = jnp.float32
) -> jnp.ndarray:
    """
    Create attention mask.
    
    Args:
        seq_len: Sequence length
        causal: Whether to create causal (lower triangular) mask
        dtype: Data type for the mask
        
    Returns:
        Attention mask [seq_len, seq_len]
    """
    if causal:
        mask = jnp.tril(jnp.ones((seq_len, seq_len), dtype=jnp.bool_))
    else:
        mask = jnp.ones((seq_len, seq_len), dtype=jnp.bool_)
    
    return mask


def apply_attention_mask(
    attn_weights: jnp.ndarray,
    mask: jnp.ndarray,
    mask_value: float = -jnp.inf
) -> jnp.ndarray:
    """
    Apply attention mask to attention weights.
    
    Args:
        attn_weights: Attention weights [..., seq_len, seq_len]
        mask: Boolean mask [..., seq_len, seq_len] (True = keep, False = mask)
        mask_value: Value to use for masked positions
        
    Returns:
        Masked attention weights
    """
    return jnp.where(mask, attn_weights, mask_value)


def compute_attention_scores(
    q: jnp.ndarray,
    k: jnp.ndarray,
    scale: Optional[float] = None
) -> jnp.ndarray:
    """
    Compute attention scores using efficient dot_general.
    
    Args:
        q: Query tensor [batch, num_heads, seq_len, head_dim]
        k: Key tensor [batch, num_heads, seq_len, head_dim]
        scale: Optional scaling factor (defaults to 1/sqrt(head_dim))
        
    Returns:
        Attention scores [batch, num_heads, seq_len, seq_len]
    """
    if scale is None:
        scale = 1.0 / math.sqrt(q.shape[-1])
    
    # Efficient batched matrix multiplication
    scores = jax.lax.dot_general(
        q * scale, k,
        dimension_numbers=(((3,), (3,)), ((0, 1), (0, 1)))
    )
    
    return scores