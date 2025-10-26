# DynamicJaxprTracer _state Attribute Fix

## Problem
The training was failing with:
```
AttributeError: DynamicJaxprTracer has no attribute _state
```

This error occurred during gradient computation in the gradient-checkpointed model forward pass.

## Root Cause
The issue was in the gradient checkpointing implementation in `src/model/valkyrie.py`. When using `nn.remat` (Flax's gradient checkpointing), the `training` parameter was being passed as a keyword argument to the checkpointed function. During JAX's gradient computation, this parameter becomes a tracer object (`DynamicJaxprTracer`), and somewhere in the call chain, code was trying to access a `_state` attribute on this tracer.

JAX tracers only support array-like operations and don't have arbitrary attributes like `_state`, causing the AttributeError.

## Solution
Modified the gradient checkpointing implementation to make `training` a closure variable instead of a function parameter:

**Before (problematic):**
```python
def _block_call(
    x, freqs_cos, freqs_sin, position_ids,
    attention_mask, past_key_value, s5_state, global_tokens, *,
    training: bool,  # This becomes a tracer!
):
    return block(..., training)

checkpointed_call = nn.remat(lambda *a, **k: _block_call(*a, **k))
x, present_key_value, next_s5_state = checkpointed_call(
    ..., training=training  # Passing as parameter
)
```

**After (fixed):**
```python
def _block_call(
    x, freqs_cos, freqs_sin, position_ids,
    attention_mask, past_key_value, s5_state, global_tokens
):
    # training is captured from closure, not passed as parameter
    return block(..., training)  # training from outer scope

checkpointed_call = nn.remat(_block_call)
x, present_key_value, next_s5_state = checkpointed_call(
    x, self.cos_freqs, self.sin_freqs, position_ids,
    attention_mask, layer_past_key_value, layer_s5_state, global_tokens
    # No training parameter passed
)
```

## Key Differences from JAX `jax.jit`
- **JAX `jax.jit`**: Supports `static_argnames` to prevent certain arguments from becoming tracers
- **Flax `nn.remat`**: Does NOT support `static_argnames`, so we must use closure capture instead

## Verification
After applying this fix, the training script no longer produces the `DynamicJaxprTracer has no attribute _state` error. The gradient checkpointing now works correctly with Flax's `nn.remat`.

## Related Files
- `src/model/valkyrie.py`: Contains the fixed gradient checkpointing implementation
- `GRADIENT_CHECKPOINTING_FIX.md`: Documents the previous `static_argnames` fix