Got it — here’s a complete, soup-to-nuts fix plan for the training run that crashed, plus a few structural hardening steps so the entire training state is robust going forward. I’ve split this into (A) the immediate blocker (your AttributeError), (B) the emergency-checkpoint failure, and (C) correctness/perf hardening so you can actually train at your target context length and model size.

A) Immediate blocker — DynamicJaxprTracer has no attribute _state
What’s happening (plain English)

Inside ValkyrieModel.__call__ you’re doing an attribute probe like thing._state or hasattr(thing, "_state") on a traced JAX value (a DynamicJaxprTracer). During jit/grad, JAX forbids arbitrary attribute access on tracers; even hasattr(tracer, "_state") internally calls getattr and explodes — exactly the error you see.

From the stack, this happens when you iterate layers and pass an S5 state into a checkpointed_call of a ValkyrieBlock. In locals we see:

layer_s5_state: JitTracer<complex64[1,768]>

In the same frame, error mentions ._state — so somewhere you assume the S5 state is a wrapper object with a ._state field, but at trace time it’s just an array tracer.

Minimal, safe fix (avoid attribute access on tracers)

Never use hasattr(x, "_state") (or getattr(x, "_state", ...)) inside any jitted region.

Replace with an explicit type check against your wrapper type, not an attribute:

# src/model/valkyrie.py (near __call__)
from src.modules.s5 import S5State  # or wherever it lives

def _as_s5_array(s):
    # Safe: does not touch tracer attributes
    return s.state if isinstance(s, S5State) else s


Then, before calling the block:

s5_in = _as_s5_array(layer_s5_state)
x, present_key_value, next_s5_state = checkpointed_call(
    block, x, s5_in, position_ids=..., attention_mask=..., training=training, ...
)
# Normalize on the way out too:
next_s5_states.append(_as_s5_array(next_s5_state))


Search & destroy every tracer-unsafe probe:

grep -R "hasattr(.*_state" -n src
grep -R "\._state" -n src
grep -R "getattr(.*_state" -n src


Refactor each site with isinstance(..., S5State) → use .state, else leave as array.

Why this is correct
isinstance(tracer, S5State) returns False for tracers (they’re not your dataclass), so you’ll never poke into tracer attributes. When you do have the wrapper (e.g., outside jit or when constructing), it resolves cleanly.

Optional but recommended: make S5 state a crisp pytree

If you truly want a wrapper, define it as a Flax/JAX pytree so you can pass it through jitted code without manual unwrapping, and don’t use leading underscore field names (they often signal “private” and invite attr probing):

# src/modules/s5_state.py
import flax
import jax.numpy as jnp

@flax.struct.dataclass
class S5State:
    state: jnp.ndarray  # [*, d_state], complex64


Then always pass/return S5State consistently and remove any probing. Or, simplest: standardize on raw arrays for S5 state everywhere and delete the wrapper entirely.

B) Secondary crash — Orbax emergency checkpoint path

You got:

ValueError: Checkpoint path should be absolute. Got checkpoints/emergency_checkpoint_00000000.orbax-checkpoint-tmp


Orbax requires absolute paths, and your emergency writer is using a relative path. Fix in either config or code:

Config-level fix (fastest)

In your YAML (e.g., configs/valkyrie_tpu_v4_8.yaml) ensure:

checkpoint:
  base_dir: /home/ravkeave/v1/codebase/checkpoints  # absolute
  save_every_steps: 1000
  keep_last: 5
  emergency_dirname: emergency  # we'll join on base_dir

Code-level hardening (normalizes everywhere)

In your checkpoint util (looks like src/io/checkpoint.py):

import os

def _abs(path: str) -> str:
    return path if os.path.isabs(path) else os.path.abspath(path)

# Wherever you construct dirs
base_dir = _abs(cfg.checkpoint.base_dir)
emergency_dir = os.path.join(base_dir, cfg.checkpoint.emergency_dirname)
os.makedirs(emergency_dir, exist_ok=True)

Avoid double-fault on exceptions

When you’re already crashing, don’t use async commit for the emergency save — finish synchronously:

from orbax.checkpoint import Checkpointer  # not AsyncCheckpointer

with Checkpointer(...) as ckpt:
    ckpt.save(emergency_dir, save_args=..., options=...)


Also, clean stale temp dirs:

rm -rf /home/ravkeave/v1/codebase/checkpoints/*orbax-checkpoint-tmp

C) Make the whole training state reliable (correctness & perf)

These will keep you from hitting different walls after fixing A & B.

1) Don’t cache KV during training with grad checkpointing

You pass use_cache=True while gradient_checkpointing=True. That’s a classic “hold all activations AND re-materialize” footgun.

In process_valid_chunk(...) (train loop) pass use_cache=False to the model during training.

Or default use_cache = training and phase_config.use_cache and set use_cache: false in training phases.

outputs = self.model.apply(
    params,
    chunk_input,
    position_ids=...,
    attention_mask=...,
    s5_states=current_s5_states,
    labels=chunk_labels,
    training=True,
    use_cache=False,  # <—— change this
    rngs={"dropout": dropout_rng_chunk, "random": random_rng_chunk},
)

2) Guard optional features by phase

Your phase shows hrm_enabled=False, but the model still constructs next_hrm_state every call. Put the HRM path under an explicit guard so you don’t materialize big tensors you won’t use:

if self.config.hrm_enabled and training: 
    # run HRM planner/executor
else:
    next_hrm_state = None


Tie hrm_enabled to phase (runtime) not only to global model config.

3) Memory pressure & chunking sanity

Given: batch_size=8, full_seq_len=65536, chunk_size=4096, backprop_chunks=1, 36 layers, d=1536. That’s huge.

Quick knobs that don’t change the optimizer step semantics:

Increase backprop_chunks (e.g., 4 or 8) so each step backprops through fewer tokens at a time.

If needed, reduce chunk_size (e.g., 2048) while you validate correctness.

Keep gradient_checkpointing=True.

Consider activation dtype to bfloat16 on TPU: make sure your mixed precision policy does compute in bfloat16 and params in float32.

4) Make your logical mesh match devices

Log shows Mesh(axis_sizes=(4, 1)) on v4-8. If you intend to use all 8 chips on a single host, you likely want (8, 1) or another partition that matches your sharding. Double-check device count:

# During init
devices = jax.devices()
n = len(devices)  # should be 8 on v4-8 single host
mesh = Mesh(np.array(devices).reshape((n, 1)), ('x', 'z'))


If you deliberately use 4 chips for memory, fine — otherwise you’re leaving perf on the table.

5) Eliminate Python-side dynamic structures inside jit

Building Python lists/dicts inside a jitted loop is fragile. In __call__ you do:

next_key_values = []
next_s5_states = []
for i, ...:
    ...
    next_key_values.append(kv)
    next_s5_states.append(s5)
return x, next_key_values, next_s5_states


Make sure you return tuples, not lists, and you don’t mutate post-return. Even better, use jax.lax.scan over layers if you can (you’re already scanning over chunks; scanning over layers can help compiler too).

At minimum:

return x, tuple(next_key_values), tuple(next_s5_states)

6) Position handling: keep it pure

You have start_pos, end_pos as tracers and closures reset_position / keep_position. Ensure those functions don’t close over or mutate Python lists of states. They should return new pytrees, not mutate the incoming one. If you are slicing a list of S5 states, use jax.tree_util.tree_map over a tuple.

7) Data/labels alignment (quick check)

Your locals show:

chunk_input: [8,4096]

chunk_input_shifted: [8,4095]

chunk_labels: [8,4096]

Ensure your loss only evaluates on the 4095 predictive positions (or that labels are padded with -100 at index 0) so you don’t index out of bounds or inflate loss. (This hasn’t crashed you yet, but it’s a common source of silent bugs.)

8) Mixed precision policy (TPU)

Confirm your MixedPrecisionPolicy ends up with:

compute dtype: bfloat16

param dtype: float32

output/logits dtype: float32

If you’re still on float32 compute, memory & speed will both suffer.

9) Logging & debug hygiene

Set JAX_TRACEBACK_FILTERING=off while you iterate on crashes.

Log use_cache, hrm_enabled, chunk_size, backprop_chunks per phase at INFO right before compiling the train step — saves a ton of time later.

10) Emergency save behavior on exceptions

Wrap the training epoch in a try/except and do a synchronous save with an absolute path, then re-raise:

try:
    final_state = training_loop.train_epoch(...)
except Exception as e:
    safe_path = os.path.join(abs_base_dir, "emergency", f"emergency_checkpoint_{global_step:08d}")
    ckpt = Checkpointer(...)  # sync
    ckpt.save(safe_path, save_args=..., options=...)
    raise

D) Concrete diffs you can drop in
1) src/model/valkyrie.py (normalize S5 state & don’t probe tracers)
@@
-    checkpointed_call = _block_call  # grad ckpt toggled elsewhere
+    checkpointed_call = _block_call  # grad ckpt toggled elsewhere

+    from src.modules.s5_state import S5State  # adjust import path
+    def _as_s5_array(s):
+        # Avoid hasattr/getattr on tracers
+        return s.state if isinstance(s, S5State) else s
@@
-    for i, (block, layer_s5_state) in enumerate(zip(self.blocks, past_s5_states)):
-        x, present_key_value, next_s5_state = checkpointed_call(
-            block, x, layer_s5_state, position_ids=position_ids, attention_mask=attention_mask,
-            training=training, use_cache=use_cache, rngs=rngs
-        )
-        next_key_values.append(present_key_value)
-        next_s5_states.append(next_s5_state)
+    for i, (block, layer_s5_state) in enumerate(zip(self.blocks, past_s5_states)):
+        s5_in = _as_s5_array(layer_s5_state)
+        x, present_key_value, next_s5_state = checkpointed_call(
+            block, x, s5_in, position_ids=position_ids, attention_mask=attention_mask,
+            training=training, use_cache=use_cache, rngs=rngs
+        )
+        next_key_values.append(present_key_value)
+        next_s5_states.append(_as_s5_array(next_s5_state))
@@
-    return x, next_key_values, next_s5_states
+    return x, tuple(next_key_values), tuple(next_s5_states)


(Also remove any remaining hasattr(x, "_state") and getattr(x, "_state", ...) everywhere in the repo.)

2) src/train/train_loop.py (don’t cache during training)
-outputs = self.model.apply(
+outputs = self.model.apply(
     state.params,
     chunk_input,
     position_ids=position_ids_chunk,
     attention_mask=attention_mask_chunk,
     s5_states=current_s5_states,
     labels=chunk_labels,
-    training=True, use_cache=True,
+    training=True, use_cache=False,
     rngs={"dropout": dropout_rng_chunk, "random": random_rng_chunk},
)

3) src/io/checkpoint.py (force absolute paths)
+def _abs(path: str) -> str:
+    import os
+    return path if os.path.isabs(path) else os.path.abspath(path)

 def prepare_checkpoint_dirs(cfg):
-    base = cfg.checkpoint.base_dir
+    base = _abs(cfg.checkpoint.base_dir)
     os.makedirs(base, exist_ok=True)
     if hasattr(cfg.checkpoint, "emergency_dirname"):
-        emerg = os.path.join(base, cfg.checkpoint.emergency_dirname)
+        emerg = os.path.join(base, cfg.checkpoint.emergency_dirname)
         os.makedirs(emerg, exist_ok=True)
     return base


And in your emergency writer, join on prepare_checkpoint_dirs(cfg) and use the sync Checkpointer inside the exception path.

E) Sanity checklist to run after patching

Clean stale checkpoints

rm -rf /home/ravkeave/v1/codebase/checkpoints/*orbax-checkpoint-tmp


Dry-run compile with a tiny shape

Set chunk_size=1024, backprop_chunks=2, seq_len=8192, batch_size=1, steps=1.

use_cache=False, hrm_enabled=False in phase 0.

Goal: reach step 1, no attribute errors, no Orbax error on exit.

Bump to your target chunking

Gradually raise chunk_size→2048→4096, then increase backprop_chunks (4–8) instead of batch size.

Confirm mesh

Log available devices and your mesh axis sizes at startup. Ensure you’re using what you intend.

Quick numeric check

Over a fixed small batch, run two steps and ensure loss decreases or at least is finite and gradients aren’t NaN (log jnp.isnan counts).

If you apply the tracer-safe state handling and stop caching during training, the crash you posted will disappear. The absolute-path change will prevent the emergency-save cascade from masking the real error with a second failure. The rest of the checklist keeps you from hitting new walls as you scale the run back up.
