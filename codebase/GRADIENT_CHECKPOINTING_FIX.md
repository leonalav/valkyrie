# Gradient Checkpointing Fix: Flax nn.remat vs JAX jit

## Problem
Training failed with the error:
```
TypeError: checkpoint() got an unexpected keyword argument 'static_argnames'
```

This occurred in `src/model/valkyrie.py` at line 203 when calling `nn.remat` with `static_argnames` parameter.

## Root Cause
The issue was a confusion between Flax's `nn.remat` and JAX's `jax.jit` APIs:

- **JAX `jax.jit`**: Supports `static_argnames` parameter to mark arguments as static (not traced)
- **Flax `nn.remat`**: Does NOT support `static_argnames` parameter - this is Flax-specific gradient checkpointing

## Solution
Removed the unsupported `static_argnames` parameter from the `nn.remat` call:

### Before (Incorrect):
```python
checkpointed_call = nn.remat(
    lambda *a, **k: _block_call(*a, **k),
    static_argnames=("training",),  # ❌ Not supported by nn.remat
)
```

### After (Correct):
```python
checkpointed_call = nn.remat(
    lambda *a, **k: _block_call(*a, **k)  # ✅ Flax nn.remat syntax
)
```

## Key Differences

| Feature | JAX `jax.jit` | Flax `nn.remat` |
|---------|---------------|-----------------|
| Purpose | General JIT compilation | Gradient checkpointing |
| `static_argnames` | ✅ Supported | ❌ Not supported |
| Usage Context | Any JAX function | Flax modules/layers |
| Memory Trade-off | N/A | Trades compute for memory |

## Implementation Details

The fix maintains the pure functional approach:
1. Created a pure `_block_call` function that only operates on arrays
2. Used `nn.remat` without `static_argnames` 
3. Passed `training` as a regular argument (not static)

## Verification
- Training script now starts without the `TypeError`
- Gradient checkpointing is properly enabled for memory efficiency
- No `DynamicJaxprTracer` errors related to this change

## Files Modified
- `src/model/valkyrie.py`: Fixed `nn.remat` call in gradient checkpointing block

## Best Practices
1. Use `static_argnames` only with `jax.jit`, not with `nn.remat`
2. Keep gradient checkpointed functions purely functional over arrays
3. Avoid accessing private attributes (like `._state`) inside checkpointed blocks
4. Test gradient checkpointing with small examples before full training runs

## Related Issues
This fix resolves the immediate `TypeError` but the training may still encounter:
- Data loading issues (TPU configuration)
- Other tracer-related errors if private attributes are accessed
- Memory/performance issues that require further optimization