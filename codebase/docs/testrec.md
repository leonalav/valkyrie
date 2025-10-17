1) Parity: Chunked vs Full sliding-window attention

Purpose: verify chunked implementation exactly matches full (reference) sliding-window attention for moderate-length sequences.

Why it matters: If chunking/padding or masking is wrong you’ll see visible, consistent output differences.

# Parity test: chunked vs full (JAX)
import jax, jax.numpy as jnp, jax.nn as jnn
from functools import partial
rng = jax.random.PRNGKey(0)

# dummy config and module
cfg = ValkyrieConfig(...)   # fill with small sizes: d_model=64, n_heads=4, n_kv_heads=2, longformer_window_size=256, ...
attn = ValkyrieLongformerAttention(config=cfg)

B, T, C = 2, 512, cfg.d_model
x = jax.random.normal(rng, (B, T, C))
freqs_cos, freqs_sin = make_rope_freqs(T, cfg.head_dim)  # use your apply_rope helper setup
pos_ids = jnp.arange(T)

# init params
variables = attn.init(rng, x, freqs_cos, freqs_sin, pos_ids, attention_mask=None, past_key_value=None, training=False)

# full/sliding output (reference) - uses small T so it's feasible
full_out, _ = attn.apply(variables, x, freqs_cos, freqs_sin, pos_ids, attention_mask=None, past_key_value=None, method=ValkyrieLongformerAttention._sliding_window_attention)

# chunked output
chunked_out, _ = attn.apply(variables, x, freqs_cos, freqs_sin, pos_ids, attention_mask=None, past_key_value=None, method=ValkyrieLongformerAttention._chunked_sliding_window_attention)

# compare
diff = jnp.max(jnp.abs(full_out - chunked_out))
rel = jnp.linalg.norm(full_out - chunked_out) / (jnp.linalg.norm(full_out) + 1e-12)
print("max_abs_diff", diff, "rel_diff", rel)


Pass criteria:

max_abs_diff < 1e-5 and rel_diff < 1e-6 for float32.
If fails: likely padding/edge-window handling or mask broadcasting bug. Fix by padding windows uniformly (see earlier sketch) and ensure mask uses jnp ops.

2) Global attention correctness (token <-> global symmetry)

Purpose: ensure global tokens (e.g., [CLS]) attend to all tokens and vice-versa.

# Global attention test
global_idx = jnp.array([0, 7])   # two global tokens
# create attention mask or set in config
cfg = cfg.replace(longformer_global_attention_indices=[0,7])
attn = ValkyrieLongformerAttention(cfg)
variables = attn.init(rng, x, freqs_cos, freqs_sin, pos_ids)

out, present = attn.apply(variables, x, freqs_cos, freqs_sin, pos_ids, attention_mask=None, past_key_value=None)

# quick check: at global positions, output is different than local-only
# compute local-only by zeroing qg/kg/vg (or use global_indices=empty and compute)
cfg_no_global = cfg.replace(longformer_global_attention_indices=[])
attn_no_global = ValkyrieLongformerAttention(cfg_no_global)
vars2 = attn_no_global.init(rng, x, freqs_cos, freqs_sin, pos_ids)
out_local_only, _ = attn_no_global.apply(vars2, x, freqs_cos, freqs_sin, pos_ids)

# compare global token outputs
g_out_with_global = out[:, global_idx, :]
g_out_local_only = out_local_only[:, global_idx, :]
print("global token diff", jnp.max(jnp.abs(g_out_with_global - g_out_local_only)))


Pass criteria:

Non-zero difference (global outputs must differ). Also manually inspect attention weights: global tokens should attend broadly; you can instrument _global_attention to return weights for inspection.

If fails: check query_global, key_global, value_global wiring and the code path that overwrites token positions with global_out.

3) KV cache correctness (incremental vs full)

Purpose: validate that incremental generation using past_key_value equals full forward pass on concatenated input. This is crucial for autoregressive inference.

# KV cache test
B, T1, T2 = 1, 8, 4  # small
x1 = jax.random.normal(rng, (B, T1, C))
x2 = jax.random.normal(rng, (B, T2, C))

# full run on combined input
x_all = jnp.concatenate([x1, x2], axis=1)
vars_init = attn.init(rng, x_all, freqs_cos, freqs_sin, pos_ids=jnp.arange(T1+T2))
out_full, _ = attn.apply(vars_init, x_all, freqs_cos, freqs_sin, jnp.arange(T1+T2), training=False)

# step1 - compute and cache
out1, present = attn.apply(vars_init, x1, freqs_cos[:, :T1,...], freqs_sin[:, :T1,...], jnp.arange(T1), training=False)
# step2 - use present_key_value to process x2 only
out2_inc, present2 = attn.apply(vars_init, x2, freqs_cos[:, T1:T1+T2,...], freqs_sin[:, T1:T1+T2,...], jnp.arange(T1, T1+T2), past_key_value=present, training=False)

# stitch
out_inc_all = jnp.concatenate([out1, out2_inc], axis=1)
# compare last T2 tokens
print("max_abs_diff incremental vs full (tail):", jnp.max(jnp.abs(out_full[:, T1:, :] - out_inc_all[:, T1:, :])))


Pass criteria:

max_abs_diff < 1e-5. If fails: cache write offsets or RoPE-on-cached-keys mismatch.

Fixes: ensure you store rotated (RoPE-applied) Ks in cache, or re-apply RoPE with consistent position ids during attention.

4) RoPE caching test (positioning / rotation invariance)

Purpose: confirm caching + pre-rotated keys produce identical attention as applying RoPE on-the-fly with absolute position ids.

Test idea:

Compute rotated keys for full sequence; compare using cached pre-rotated keys vs rotating per-step via position ids.

# RoPE test sketch
# Build full keys rotated:
qs_full = attn.apply(variables, x_all, freqs_cos, freqs_sin, jnp.arange(T1+T2), method=...)  # use your proj steps to extract qs/ks
# Alternatively, call internal helper to project and apply_rope
# Compare: if you cache rotated ks, they must match ks computed with apply_rope(pos_ids)


Pass criteria:

exact match for rotated cached keys.

If fails: mismatch in position_ids offsets or apply_rope implementation. Remedy: cache rotated keys at time of compute using absolute positions.

5) Dtype & numeric stability test (fp32 vs bfloat16 / float16)

Purpose: show numeric instability and NaNs in half precision; validate using fp32 for softmax steps.

# Numeric stability test
# run attention in float32 and in bfloat16 (TPU) or float16 (GPU) and check for NaNs/diffs
x_fp32 = x.astype(jnp.float32)
out32, _ = attn.apply(variables, x_fp32, freqs_cos, freqs_sin, jnp.arange(T), method=ValkyrieLongformerAttention.__call__)

x_bf16 = x_fp32.astype(jnp.bfloat16)
out_bf16, _ = attn.apply(variables, x_bf16, freqs_cos.astype(jnp.bfloat16), freqs_sin.astype(jnp.bfloat16), jnp.arange(T))
print("has_nan bf16:", jnp.any(jnp.isnan(out_bf16)))
print("max_abs_diff bf16 vs fp32:", jnp.max(jnp.abs(out32.astype(jnp.float32) - out_bf16.astype(jnp.float32))))


Pass criteria:

no NaNs for bfloat16; differences expected but not catastrophic. If bfloat16 shows NaNs, keep softmax & score accumulation in fp32 — cast to fp32 for score and softmax, then cast back.

Fix: ensure scores = (qs.astype(jnp.float32) @ ks.astype(jnp.float32).T), then softmax in float32.