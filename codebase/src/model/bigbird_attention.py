"""BigBird Sparse Attention for Valkyrie Model

JAX/Flax implementation of BigBird sparse attention optimized for TPU training.
Supports blockified GEMMs, global tokens for HRM integration, and efficient
window + random attention patterns.

Based on the BigBird paper: "Big Bird: Transformers for Longer Sequences"
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
import math
from typing import Optional, Tuple, List
from functools import partial

from .modules import RMSNorm, apply_rope


class BigBirdAttention(nn.Module):
    """BigBird sparse attention with blockified GEMM optimization.
    
    Implements three attention patterns:
    1. Window attention: Local sliding window
    2. Global attention: Special tokens that attend to all and are attended by all
    3. Random attention: Random connections between blocks
    
    Optimized for TPU with blockified operations and mixed precision.
    """
    config: 'ValkyrieConfig'
    
    def setup(self):
        self.d_model = self.config.d_model
        self.n_heads = self.config.n_heads
        self.head_dim = self.d_model // self.n_heads
        self.block_size = self.config.bigbird_block_size
        self.num_global_tokens = self.config.bigbird_num_global_tokens
        self.num_window_blocks = self.config.bigbird_num_window_blocks
        self.num_random_blocks = self.config.bigbird_num_random_blocks
        
        # Linear projections
        self.q_proj = nn.Dense(self.d_model, use_bias=self.config.use_bias)
        self.k_proj = nn.Dense(self.d_model, use_bias=self.config.use_bias)
        self.v_proj = nn.Dense(self.d_model, use_bias=self.config.use_bias)
        self.o_proj = nn.Dense(self.d_model, use_bias=self.config.use_bias)
        
        # Dropout
        self.dropout = nn.Dropout(self.config.attn_dropout)
        
    def _create_random_attention_mask(self, 
                                    num_blocks: int, 
                                    rng_key: jax.random.PRNGKey) -> jnp.ndarray:
        """Create random attention pattern for BigBird."""
        # Create random connections between blocks
        # Each block attends to num_random_blocks random blocks
        random_mask = jnp.zeros((num_blocks, num_blocks), dtype=jnp.bool_)
        
        for i in range(num_blocks):
            # Skip global tokens and window blocks
            available_blocks = []
            for j in range(num_blocks):
                # Skip self, global tokens, and window blocks
                if (j != i and 
                    j >= self.num_global_tokens and  # Skip global tokens
                    abs(j - i) > self.num_window_blocks):  # Skip window
                    available_blocks.append(j)
            
            if len(available_blocks) > 0:
                # Randomly select blocks to attend to
                num_select = min(self.num_random_blocks, len(available_blocks))
                selected = jax.random.choice(
                    rng_key, 
                    jnp.array(available_blocks), 
                    shape=(num_select,), 
                    replace=False
                )
                random_mask = random_mask.at[i, selected].set(True)
        
        return random_mask
    
    def _create_attention_mask(self, 
                             seq_len: int, 
                             global_tokens: Optional[jnp.ndarray] = None,
                             rng_key: Optional[jax.random.PRNGKey] = None) -> jnp.ndarray:
        """Create BigBird attention mask with window, global, and random patterns."""
        if rng_key is None:
            rng_key = jax.random.PRNGKey(42)  # Fixed seed for reproducibility
            
        # Pad sequence to block size
        padded_len = ((seq_len + self.block_size - 1) // self.block_size) * self.block_size
        num_blocks = padded_len // self.block_size
        
        # Initialize mask
        mask = jnp.zeros((padded_len, padded_len), dtype=jnp.bool_)
        
        # 1. Global attention: First num_global_tokens attend to all and are attended by all
        if self.num_global_tokens > 0:
            # Global tokens attend to everything
            mask = mask.at[:self.num_global_tokens, :].set(True)
            # Everything attends to global tokens
            mask = mask.at[:, :self.num_global_tokens].set(True)
        
        # 2. Window attention: Each token attends to local window (vectorized)
        if self.num_window_blocks > 0:
            window_size = self.num_window_blocks * self.block_size
            # Create position indices
            positions = jnp.arange(seq_len)
            # Calculate window bounds for all positions at once
            window_starts = jnp.maximum(0, positions - window_size)
            window_ends = jnp.minimum(seq_len, positions + window_size + 1)
            
            # Create window mask using broadcasting
            pos_i = positions[:, None]  # Shape: (seq_len, 1)
            pos_j = positions[None, :]  # Shape: (1, seq_len)
            
            # Each position i attends to positions j where window_starts[i] <= j < window_ends[i]
            window_mask = (pos_j >= window_starts[:, None]) & (pos_j < window_ends[:, None])
            mask = mask.at[:seq_len, :seq_len].set(mask[:seq_len, :seq_len] | window_mask)
        
        # 3. Random attention: Block-level random connections (vectorized)
        if self.num_random_blocks > 0:
            random_mask = self._create_random_attention_mask(num_blocks, rng_key)
            
            # Create a block-level mask and then expand it to token level
            # Each block is block_size x block_size
            block_mask_expanded = jnp.repeat(
                jnp.repeat(random_mask, self.block_size, axis=0), 
                self.block_size, axis=1
            )
            
            # Trim to actual sequence length
            block_mask_trimmed = block_mask_expanded[:seq_len, :seq_len]
            
            # Apply the random block mask
            mask = mask.at[:seq_len, :seq_len].set(mask[:seq_len, :seq_len] | block_mask_trimmed)
        
        # Apply causal mask
        causal_mask = jnp.tril(jnp.ones((padded_len, padded_len), dtype=jnp.bool_))
        mask = mask & causal_mask
        
        return mask[:seq_len, :seq_len]
    
    def _blockified_attention(self,
                            q: jnp.ndarray,
                            k: jnp.ndarray, 
                            v: jnp.ndarray,
                            attention_mask: jnp.ndarray) -> jnp.ndarray:
        """Compute attention using blockified GEMM operations for TPU efficiency."""
        batch_size, seq_len, n_heads, head_dim = q.shape
        
        # Pad to block size
        padded_len = ((seq_len + self.block_size - 1) // self.block_size) * self.block_size
        pad_len = padded_len - seq_len
        
        if pad_len > 0:
            q = jnp.pad(q, ((0, 0), (0, pad_len), (0, 0), (0, 0)))
            k = jnp.pad(k, ((0, 0), (0, pad_len), (0, 0), (0, 0)))
            v = jnp.pad(v, ((0, 0), (0, pad_len), (0, 0), (0, 0)))
            attention_mask = jnp.pad(attention_mask, ((0, pad_len), (0, pad_len)))
        
        num_blocks = padded_len // self.block_size
        
        # Reshape to blocks: [batch, num_blocks, block_size, n_heads, head_dim]
        q_blocks = q.reshape(batch_size, num_blocks, self.block_size, n_heads, head_dim)
        k_blocks = k.reshape(batch_size, num_blocks, self.block_size, n_heads, head_dim)
        v_blocks = v.reshape(batch_size, num_blocks, self.block_size, n_heads, head_dim)
        
        # Compute attention scores using blockified operations
        # This is where the TPU efficiency comes from - dense GEMMs on blocks
        scores = jnp.einsum('bqhd,bkhd->bqkh', q_blocks.reshape(-1, n_heads, head_dim), 
                           k_blocks.reshape(-1, n_heads, head_dim))
        scores = scores.reshape(batch_size, num_blocks, self.block_size, 
                               num_blocks, self.block_size, n_heads)
        
        # Apply scaling
        scores = scores / math.sqrt(head_dim)
        
        # Apply attention mask
        mask_blocks = attention_mask.reshape(num_blocks, self.block_size, 
                                           num_blocks, self.block_size)
        scores = jnp.where(mask_blocks[..., None], scores, -jnp.inf)
        
        # Softmax
        attn_weights = jax.nn.softmax(scores, axis=3)
        attn_weights = self.dropout(attn_weights, deterministic=not self.training)
        
        # Apply attention to values
        out = jnp.einsum('bqkvh,bvhd->bqhd', attn_weights, 
                        v_blocks.reshape(batch_size, num_blocks, self.block_size, n_heads, head_dim))
        
        # Reshape back to sequence
        out = out.reshape(batch_size, padded_len, n_heads, head_dim)
        
        # Remove padding
        if pad_len > 0:
            out = out[:, :seq_len]
            
        return out
    
    def __call__(self,
                 x: jnp.ndarray,
                 freqs_cos: jnp.ndarray,
                 freqs_sin: jnp.ndarray,
                 position_ids: jnp.ndarray,
                 attention_mask: Optional[jnp.ndarray] = None,
                 past_key_value: Optional[Tuple] = None,
                 global_tokens: Optional[jnp.ndarray] = None,
                 training: bool = False) -> Tuple[jnp.ndarray, Tuple]:
        """Forward pass of BigBird attention.
        
        Args:
            x: Input tensor [batch, seq_len, d_model]
            freqs_cos: RoPE cosine frequencies
            freqs_sin: RoPE sine frequencies  
            position_ids: Position indices
            attention_mask: Optional attention mask
            past_key_value: Cached key-value pairs
            global_tokens: HRM planner tokens to use as global tokens
            training: Whether in training mode
            
        Returns:
            Tuple of (output, present_key_value)
        """
        batch_size, orig_seq_len, _ = x.shape
        num_global_tokens = 0
        
        # If global tokens provided (from HRM planner), prepend them
        if global_tokens is not None:
            num_global_tokens = global_tokens.shape[1]
            x = jnp.concatenate([global_tokens, x], axis=1)
        
        total_seq_len = orig_seq_len + num_global_tokens
        
        # Linear projections
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Reshape for multi-head attention
        q = q.reshape(batch_size, total_seq_len, self.n_heads, self.head_dim)
        k = k.reshape(batch_size, total_seq_len, self.n_heads, self.head_dim)
        v = v.reshape(batch_size, total_seq_len, self.n_heads, self.head_dim)
        
        # Apply RoPE only to sequence tokens, not global tokens
        if num_global_tokens > 0:
            # Split into global and sequence parts
            q_global, q_seq = q[:, :num_global_tokens], q[:, num_global_tokens:]
            k_global, k_seq = k[:, :num_global_tokens], k[:, num_global_tokens:]
            
            # Apply RoPE only to sequence tokens
            q_seq = apply_rope(q_seq, freqs_cos, freqs_sin, position_ids)
            k_seq = apply_rope(k_seq, freqs_cos, freqs_sin, position_ids)
            
            # Concatenate back
            q = jnp.concatenate([q_global, q_seq], axis=1)
            k = jnp.concatenate([k_global, k_seq], axis=1)
        else:
            # No global tokens, apply RoPE normally
            q = apply_rope(q, freqs_cos, freqs_sin, position_ids)
            k = apply_rope(k, freqs_cos, freqs_sin, position_ids)
        
        # Handle KV caching
        if past_key_value is not None:
            past_k, past_v = past_key_value
            k = jnp.concatenate([past_k, k], axis=1)
            v = jnp.concatenate([past_v, v], axis=1)
        
        present_key_value = (k, v)
        
        # Create BigBird attention mask
        rng_key = self.make_rng('random') if training else jax.random.PRNGKey(42)
        bigbird_mask = self._create_attention_mask(total_seq_len, global_tokens, rng_key)
        
        # Apply additional attention mask if provided
        if attention_mask is not None:
            bigbird_mask = bigbird_mask & attention_mask
        
        # Compute attention using blockified operations
        # TODO: Fix blockified attention implementation - temporarily disabled
        if False and self.config.bigbird_use_blockified_gemm:
            attn_output = self._blockified_attention(q, k, v, bigbird_mask)
        else:
            # Fallback to standard attention
            scores = jnp.einsum('bqhd,bkhd->bhqk', q, k) / math.sqrt(self.head_dim)
            scores = jnp.where(bigbird_mask[None, None, :, :], scores, -jnp.inf)
            attn_weights = jax.nn.softmax(scores, axis=-1)
            attn_weights = self.dropout(attn_weights, deterministic=not training)
            attn_output = jnp.einsum('bhqk,bkhd->bqhd', attn_weights, v)
        
        # Reshape and project output
        attn_output = attn_output.reshape(batch_size, total_seq_len, self.d_model)
        output = self.o_proj(attn_output)
        
        # Remove global tokens from output if they were added
        if num_global_tokens > 0:
            output = output[:, num_global_tokens:]
        
        return output, present_key_value