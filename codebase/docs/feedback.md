1) Replace Python ops inside process_chunk with JAX ops and use fixed shapes

vmap requires each mapped invocation to return identically-shaped arrays. The last chunk is usually shorter — you need to pad windows so every chunk produces a fixed (chunk_size, window_len) shaped output.

Two good choices:

(A) Pad all chunks to fixed chunk_size / window_len and vmap over them — easiest.

(B) Use jax.lax.scan with fixed-size buffers (good if you want sequential memory reuse).

Below is a corrected pad-to-fixed-window sketch (you can drop in replacing _chunked_sliding_window_attention). It uses JAX ops, fixed shapes, jax.nn.softmax, and handles chunk padding:

import jax
import jax.numpy as jnp
import jax.nn as jnn
from jax import lax

def _chunked_sliding_window_attention_fixed(self, qs, ks, vs, causal: bool = True):
    # qs: [B, n_heads, T, head_dim]
    B, n_heads, T, head_dim = qs.shape
    chunk_size = int(self.config.longformer_chunk_size)
    half_w = self.window_size // 2

    # If needed, expand kv heads
    if self.n_kv_heads != self.n_heads:
        ks = jnp.repeat(ks, self.q_per_kv, axis=1)
        vs = jnp.repeat(vs, self.q_per_kv, axis=1)

    # compute number of chunks and padded length so everything is static
    num_chunks = (T + chunk_size - 1) // chunk_size
    padded_len = num_chunks * chunk_size

    # pad qs, ks, vs to padded_len on time axis
    pad_amt = padded_len - T
    qs_p = jnp.pad(qs, ((0,0),(0,0),(0,pad_amt),(0,0)))
    ks_p = jnp.pad(ks, ((0,0),(0,0),(0,pad_amt),(0,0)))
    vs_p = jnp.pad(vs, ((0,0),(0,0),(0,pad_amt),(0,0)))

    # For every chunk, we need a fixed-size window slice.
    # We'll take for chunk i the window covering [chunk_start - half_w, chunk_end + half_w]
    # that can be at most chunk_size + 2*half_w long -> fixed window_len
    window_len = chunk_size + 2 * half_w

    # Build a (num_chunks,) array of chunk starts (int)
    chunk_starts = jnp.arange(num_chunks) * chunk_size

    def process_one_chunk(chunk_start):
        # chunk_start is a scalar jnp int
        cs = jnp.int32(chunk_start)
        ce = cs + chunk_size  # chunk end (exclusive) in padded coords

        # window start & end (clamped to [0, padded_len])
        ws = jnp.maximum(0, cs - half_w)
        we = jnp.minimum(padded_len, ce + half_w)

        # We will always take a slice of length window_len by padding/clipping as needed:
        # compute offset into ks_p
        # If window is smaller at edges, we will pad it to window_len
        k_slice = ks_p[:, :, ws:we, :]  # [B, n_heads, w_real, dim]
        v_slice = vs_p[:, :, ws:we, :]

        # If w_real < window_len, pad on right (or left depending on clamp)
        pad_w = window_len - (we - ws)
        k_slice = jnp.pad(k_slice, ((0,0),(0,0),(0,pad_w),(0,0)))
        v_slice = jnp.pad(v_slice, ((0,0),(0,0),(0,pad_w),(0,0)))

        # q_chunk (chunk_size long)
        q_chunk = qs_p[:, :, cs:ce, :]  # [B, n_heads, chunk_size, dim]

        # compute scores: [B, n_heads, chunk_size, window_len]
        scale = 1.0 / jnp.sqrt(head_dim)
        scores = jnp.einsum('bhqd,bhkd->bhqk', q_chunk, k_slice) * scale

        # build mask: shape [chunk_size, window_len] with True where valid
        # q positions: cs..ce-1, k positions: ws..(we-1) then padded positions
        q_pos = jnp.arange(cs, cs + chunk_size)[:, None]  # [chunk_size,1]
        k_pos = jnp.arange(ws, ws + window_len)[None, :]  # [1, window_len]
        dist = jnp.abs(q_pos - k_pos)
        window_mask = dist <= half_w

        if causal:
            causal_mask = (k_pos <= q_pos)
            window_mask = window_mask & causal_mask

        # Broadcast mask to [B,n_heads,chunk_size,window_len]
        mask_b = window_mask[None, None, :, :]

        # apply mask: use very negative
        neg_inf = jnp.array(-1e9, dtype=scores.dtype)
        scores = jnp.where(mask_b, scores, neg_inf)

        weights = jnn.softmax(scores, axis=-1)
        # dropout: use deterministic flag and RNG management in Flax apply
        if self.config.attn_dropout > 0:
            weights = self.attn_dropout(weights, deterministic=not training)  # ensure RNG available outside

        chunk_out = jnp.einsum('bhqk,bhkd->bhqd', weights, v_slice)  # [B, n_heads, chunk_size, dim]
        return chunk_out

    # vmap process_one_chunk over chunk_starts - shapes are fixed now
    chunk_outs = jax.vmap(process_one_chunk)(chunk_starts)  # [num_chunks, B, n_heads, chunk_size, dim]

    # reorder and trim padding to original T
    chunk_outs = chunk_outs.transpose(1,2,0,3,4).reshape(B, n_heads, padded_len, head_dim)
    out = chunk_outs[:, :, :T, :]

    return out


Notes:

This forces fixed window_len and chunk_size so vmap is valid.

Use jnp.maximum/minimum and jnn.softmax (not Python min/max or nn.softmax).

You must ensure training flag and RNGs are passed into dropout properly.

2) Use jax.nn.softmax (not nn.softmax)

Replace occurrences of nn.softmax with jax.nn.softmax. Example:

import jax.nn as jnn
attn_weights = jnn.softmax(scores, axis=-1)

3) Dropout in Flax modules

Flax Dropout needs RNG. In __call__ ensure you use:

rng = self.make_rng('dropout')
attn_weights = self.attn_dropout(attn_weights, deterministic=not training, rng=rng)


And when applying the whole model, supply rngs={'dropout': dropout_rng} to model.apply(..., rngs=rngs).

4) Avoid Python control flow on arrays (use jnp / lax)

Anywhere you do min((chunk_idx + 1) * chunk_size, T) or max(0, chunk_start - w), use jnp.minimum / jnp.maximum. Use lax primitives if you need branching inside JIT.

5) Caching & memory on TPU

Preallocating full max_position_embeddings per layer per head may be huge. Consider:

using smaller per-shard caches, or

use relative offsets/rolling buffers (store only window_size for local attention + full global list for global tokens).

Keep cache dtype float32 for stability but consider bfloat16 elsewhere for TPU speed — be cautious with softmax/numeric stability.

6) Global attention edge cases

If global_indices can be empty, guard indexing (the function already returns zeros early — good).

When repeating kg/vg for GQA, verify broadcasting shapes carefully (you did repeat earlier, that’s correct).

RoPE & apply_rope

Make sure apply_rope is vectorized and supports shapes [B, T, heads, dim].

When using precomputed freqs_cos/freqs_sin, ensure their dtype matches qs/ks (float32) and that position_ids are used consistently in both training and caching. When caching keys, you either cache already-rotated keys or apply RoPE using position ids — caching pre-rotated K/V is simplest.

TPU-specific performance tips

Use pmap/pjit for data/model parallelism as needed; chunking strategy should be compatible with sharding.

Keep shapes static and small number of traced shapes — avoid Python variability in tensor sizes.

Use jax.lax.conv_general_dilated or einsum which XLA compiles well; avoid Python loops inside JIT.

Quick checklist for you to apply now

 Replace nn.softmax with jax.nn.softmax.

 Replace Python min/max with jnp.minimum/jnp.maximum.

 Pad chunks/windows to fixed sizes (see code sketch).

 Ensure self.attn_dropout(...) receives RNG and deterministic flag correctly.

 Make RoPE apply_rope vectorized and test caching behavior (cache rotated K/V if possible).

 Add unit tests comparing small-sequence full attention with chunked version to assert parity.

 Check memory of caches before running large TPU training.