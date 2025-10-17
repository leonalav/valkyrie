— Repo / codebase blueprint (top-level layout)

Suggested repo tree (create this exactly):

valkyrie/
├─ README.md
├─ pyproject.toml / requirements.txt
├─ configs/
│  ├─ valkyrie_base.yaml
│  ├─ tpuv4_32_mesh.yaml
│  └─ data_pipeline.yaml
├─ src/
│  ├─ model/
│  │  ├─ __init__.py
│  │  ├─ longformer.py        # Flax/JAX model code (attention + RoPE)
│  │  ├─ s5.py                # S5 layer: complex params + discretize + scan
│  │  └─ modules.py           # layernorms, feedforward, positional
│  ├─ sharding/
│  │  ├─ mesh_setup.py        # create Mesh(np.array(...), ('x','y','z'))
│  │  └─ partition_specs.py   # canonical PartitionSpec objects
│  ├─ train/
│  │  ├─ train_loop.py
│  │  ├─ step_fn.py           # forward/backward pjit-ed functions
│  │  └─ optimizer.py         # optax wrapper with sharded states
│  ├─ data/
│  │  ├─ tfrecord_reader.py   # shard-aware reading: host shard = process_index
│  │  └─ tokenizer (use GPT2's)
│  ├─ io/
│  │  ├─ checkpoint.py        # Orbax-based multi-host checkpointing
│  │  └─ logging.py
│  ├─ utils/
│  │  ├─ tests/
│  │  │  ├─ s5_unit_test.py   # sequential vs parallel check
│  │  │  └─ attention_test.py # chunked vs dense attention
│  │  └─ debug.py
│  └─ launch/
│     └─ launcher.sh          # scripts to start multi-host process (gcloud/tpu_vm)
└─ docs/
   └─ design.md


Why:

Clear separation of model vs sharding vs training makes the pjit PartitionSpec decisions auditable.

Unit tests live with utils so you run them before scaling.

2 — Mesh & topology mapping (TPU-v4-32 specifics)

TPU facts to use (from TPU v4 paper):

TPU v4 is a 3D torus, and slices can be shaped to maximize bisection bandwidth for your parallelism pattern. The paper explicitly recommends mapping data parallel along one dimension and the two model parallel parameters on the other axes. Twisted/rectangular topologies can raise throughput. 

2304.01433v3

 

2304.01433v3

Each chip has ~32 GiB HBM (paper mentions HBM capacity; use this when planning memory per chip). 

2304.01433v3

For 32 chips choose a geometry that fits 32: e.g. 4 × 4 × 2 (other valid ones: 8×2×2, 4×8×1 but prefer balanced 3D if possible). The TPU paper suggests mapping data across one axis and the two model/tensor axes across the remaining axes. Twisted torus yields throughput gains for all-to-all but 4×4×2 is simple and effective. 

2304.01433v3

Mesh creation (code skeleton)
# src/sharding/mesh_setup.py
import numpy as np
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

def make_mesh():
    # devices should be arranged into a 3D shape consistent with TPU slice
    # For 32 devices, use shape (4,4,2)
    device_array = np.arange(32).reshape((4,4,2))
    mesh = Mesh(device_array, ('x','y','z'))
    return mesh


Mapping recommendation:

x axis -> model-parallel dim 1 (e.g., tensor-parallel width)

y axis -> model-parallel dim 2 (e.g., tensor-parallel length / 2D sharding)

z axis -> data-parallel

This mapping follows the TPU paper's recommendation for placing model parallelism along two torus dimensions and data across the other. 

2304.01433v3

3 — PartitionSpec canonical patterns

Define a small library partition_specs.py with canonical PartitionSpecs:

P('mp', None) — split rows (1D)

P(None, 'mp') — split columns (1D)

P('mp_w', 'mp_h') — 2D sharding for dense kernels (weight matrix split)

P('dp') — replicate across model shards (data parallel axis)

Example:

# src/sharding/partition_specs.py
from jax.sharding import PartitionSpec as P

# logical names: mp1, mp2, dp
MP1 = 'x'  # matches mesh axis names or logical names mapped later
MP2 = 'y'
DP  = 'z'

# Common specs
W_2D = P(MP1, MP2)   # 2D weight sharding
W_ROW = P(MP1, None) # row-wise shard
W_COL = P(None, MP2) # col-wise shard
EMBED_ROW = P(MP1)   # shard vocab rows across mp1


You will map these logical names to mesh axis names when you create the mesh.

4 — Model parallelism choices (practical rules & examples)

Which partitioning to pick? — three canonical options, with pros/cons:

1D (row or column) sharding

simple, low communication, easier to debug. Good for small models.

Preferred when you can split dense GEMMs along one dimension (e.g., split vocab rows for embedding). 

2304.01433v3

2D (mesh / checkerboard)

splits both input and output dims of dense matmul -> reduces memory/compute per device and gives better scaling, but requires more collective communication (all-reduce / all-gather). TPU paper shows 2D/1D partitions can improve throughput for LLMs. Use for the heavy linear layers. 

2304.01433v3

Pipeline parallelism

cut layers across device groups. Good to combine with tensor parallelism. For 32 chips you can do shallow pipelines (e.g., pipe depth 2–8 depending on model depth), but be mindful of bubble overhead. TPU paper documents pipeline+tensor mapping changes leading to 1.2x–2.3x throughput gains when tuned. 

2304.01433v3

Recommendation for Valkyrie (1.2B+):

Use 2D tensor parallelism for transformer feedforward / attention weight matrices across x,y axes.

Use z for data parallelism with gradient reduce (psums) across z.

Use small pipeline splitting if your layer count is high (e.g., 24+ layers) — pipeline depth 2–4 to reduce memory pressure. Tune with the PA-NAS style search (paper shows this pays off). 

2304.01433v3

5 — pjit / GSPMD step function design (exact)

Your train/step_fn.py will expose a pjit-ed train_step:

init_params() will create params sharded with PartitionSpecs (use with mesh: and pjit-wrapped init).

loss_and_grad = pjit(forward_and_loss, in_axis_resources=(P(...), P(...)), out_axis_resources=(P(...), None)) where arguments are sharded tensors.

Use jax.lax.with_sharding_constraint on intermediate partitions where necessary.

Important: pjit requires consistent in_axis_resources for all arguments. When wrapping the step function you must also specify how optimizer state and RNG are partitioned.

Also use jax.experimental.enable_x64(False) but keep complex parts in float32/complex64 as needed for S5.

Example snippet skeleton:

from jax import numpy as jnp
from jax import pmap, grad, value_and_grad
from jax.experimental import mesh_utils
from jax.sharding import Mesh
from jax.experimental.pjit import pjit

@pjit(in_axis_resources=(P('mp1','mp2','dp'), P('dp'), None),
      out_axis_resources=(None, None))
def train_step(params, batch, opt_state):
    # forward -> compute loss -> grads
    loss, grads = value_and_grad(forward)(params, batch)
    # all-reduce grads across mp axes if required
    grads = jax.lax.pmean(grads, axis_name='z')  # reduce across data replicas axis
    new_params, new_opt_state = optimizer_step(params, grads, opt_state)
    return new_params, new_opt_state, loss


(You’ll adapt in_axis_resources to the actual PartitionSpecs from partition_specs.py.)

6 — S5 & Longformer implementation notes for TPU specifics
S5 on TPU (critical):

TPU v4 prefers bfloat16 for speed. But S5 uses complex math internally (complex64). Implement S5 so:

Parameters (Λ, B̃, C̃) are stored in float32/complex64. Convert inputs (u) to complex64 during SSM computations, then reduce outputs to float32/bfloat16.

Keep discretization math in complex64 to prevent loss of precision. Only cast results to lower precision before leaving the S5 module.

Add with mesh/pjit wrappers around S5 apply so parameter shards line up with model partitions.

Unit test sequential vs associative_scan outputs on small sequences on single-device and then on mesh — must match.

Why: The TPU paper warns about sensitivity of numerics and the value of keeping numerically sensitive ops in higher precision; S5 is such an op. 

2304.01433v3

Longformer on TPU (critical)

Use chunked sliding-window attention implemented as fused matmuls where possible.

Keep QK softmax in float32 (or bfloat32) to avoid fp16 instability. Even on TPU bfloat16 softmax can be unstable for large logits; do softmax computation with jnp.float32 and convert outputs back to bfloat16 for the rest. (This mirrors Longformer recommendations for fp16-safe attention.) 

longformer

Partition attention heads across x,y axes so each device holds a subset of heads. Use with_sharding_constraint before and after attention to keep memory locality.

7 — Data pipeline: multi-host-aware, sharded input

Design rules:

Each host reads only its shard list of TFRecords. Use jax.process_index() & jax.process_count() to split the dataset across hosts deterministically. Example:

proc = jax.process_index()
nproc = jax.process_count()
# shard list of filenames ahead of time:
filenames_proc = all_filenames[proc::nproc]


Tokenization & chunking should be done offline or on CPU hosts with parallel workers. Do not attempt to tokenize within device kernels.

Use prefetch_to_device per host to keep device pipelines busy. Use micro-batches and gradient accumulation to reach desired effective batch size.

Why: multi-host TPU slices must avoid sending duplicate data across hosts. The TPU paper emphasizes scheduling slices and the need for coordinated data distribution. 

2304.01433v3

8 — Checkpointing, fault tolerance, and orchestration

TPU v4 slices are reliable thanks to OCS, but hosts can fail — the TPU paper recommends frequent checkpoints and scheduler-aware slices. Implement:

Orbax checkpointing (or equivalent) with sharded checkpoints. Save:

Sharded params (per-process shard)

Optimizer states (per-process shard)

S5 state / streaming memory snapshots (to resume streaming)

Training RNG/step + learning rate state

Two-level checkpointing:

Fast “micro-checkpoint” every N steps saving only minimal states (params shards + step) to local disk (VM).

Full checkpoint to GCS every M steps (larger) asynchronously to avoid blocking training.

Unified restore:

On restart, each process reads its own shard. Then call a global barrier (jax.process_index() / jax.process_count() and jax.distributed.initialize semantics) to ensure all processes are present. Only then resume training.

Checkpoint write strategy: use save-from-host-0 coordination to avoid concurrency issues OR let each host write its own shard to a known GCS path with unique suffix gs://bucket/job/checkpoint-step-<step>/shard-<proc>.ckpt. Then a small master JSON file lists the shards and their names. This is robust and parallelizable.

Why: TPU paper shows large-scale training needs strong reliability and frequent checkpoints to avoid losing days of progress; OCS helps but you still must checkpoint. 

2304.01433v3

9 — Optimizer state and memory: sharding & offload

Use optimizer libraries that support sharding (e.g., optax + manual sharding, or Flax + Alpa/Optax wrappers).

Shard optimizer states (first/second moments) across the same axes as the parameters to minimize all-to-all memory pressure.

If memory is tight on hosts, use offload strategies (parameter shard on device, master copy in host memory) — but this is slower. TPU v4 has limited HBM vs some GPU setups so partition carefully. The TPU paper notes HBM capacity may be a limiting factor for some LLMs — plan to partition across more chips rather than increasing per-chip memory. 

2304.01433v3

10 — Communication patterns and collectives (exact ops)

Use jax.lax.pmean for averaging gradients across z (data axis). Use jax.lax.psum for sums where appropriate.

For 2D tensor parallel matmuls, use x-reduce / y-allgather patterns (e.g., local matmul then all_gather on one axis, then reduction). Implement fused collectives to avoid extra copies.

Be careful with all_gather and all_to_all semantics: all_gather across large tensors can saturate bisection bandwidth — map it to the axis with best topology suited for large messages as per TPU paper: choose the axes with the best bisection for your communication patterns (TPU paper: 3D torus and twisting helps all-to-all). 

2304.01433v3

Practical rule: prefer smaller collectives across the model-parallel axes and keep large all-reduces on the data axis (z) where communication topologies are favorable.

11 — Multi-host launch, runtime & environment

Use tpu_vm or Cloud TPU provisioning that sets up jax.distributed automatically. Ensure:

--env JAX_ENABLE_XLA_PYTHON_CLIENT_ALLOCATOR=false if you face memory fragmentation.

Set host-level environment: XRT_TPU_CONFIG, JAX_PLATFORMS=cpu,gpu? (exact flags depend on your TPU launcher; if you're using gcloud alpha compute tpus or tpu vms, the vendor docs show exact flags.) The TPU paper describes scheduling topologies but the runtime flags vary by cloud setup. Use the provider docs for exact launch commands (not repeated here).

On start, each host should:

Set up Mesh consistent with device_count().

Initialize pjit/jax distributed runtime.

Load its dataset shard and begin prefetching.

Process-level invariants: every process must have same logical mesh names & axis mapping. If one process mis-declares the mesh, pjit fails with confusing errors.

12 — Tests & gating (must-run before large runs)

Unit tests (single host):

s5_unit_test.py: sequential vs parallel associativescan equality within tolerance.

attention_test.py: chunked vs masked dense attention equality for small seq lengths.

Small multi-host smoke (2 hosts / 8 devices):

Run a tiny model over 10 steps. Verify:

Parameter shapes & PartitionSpec matches across hosts.

jax.process_count() equals expected.

Gradients aggregated correctly (check pmean sums).

No NaNs.

Scaling test (32 chips dry-run):

Run for 1–5 steps with synthetic data but full sharding. Verify throughput stats + network usage. Monitor collectives latency.

Failure injection test:

Crash one host mid-training; ensure job can be re-scheduled and restore via checkpoint.

Run these before any substantive epoch.

13 — Important gotchas & hard truths (explicit, non-sugary)

Axis mismatch: the #1 source of cryptic pjit errors is inconsistent PartitionSpec/mesh mapping. Always create PartitionSpecs centrally and import them. Add assertions at model init: assert params_sharding_spec == expected_spec.

All-gather heap death: careless all_gather of large tensors across wrong axis will saturate bisection and kill throughput. Map heavy all_gathers to axes that align with torus links as recommended in TPU paper. Twisted torus can help but choose axes wisely. 

2304.01433v3

HBM limits: TPU v4’s per-chip HBM can be a limiting factor; partition more aggressively rather than trying to squeeze larger microbatch/chunk sizes. The paper warns HBM capacity might limit LLM performance; partition across chips. 

2304.01433v3

S5 numeric instability: do S5 discretization in complex64; on TPUs keep S5 ops in float32/complex64 until the last step. If you lower precision too early gradients explode.

Softmax precision: attention softmax in bfloat16 sometimes produces subtle instabilities — compute softmax in float32 (cast), then convert to bfloat16 for matmul. (Longformer paper warns fp16 attention → NaNs; similar logic applies for bfloat16.) 

longformer

Checkpoint metadata: Always write a tiny master manifest JSON listing shard filenames and process indexes — otherwise restore becomes a cliffwalk.

Debugging is hard on multi-host: enable debug flags on a small 2-host job first. Use jax_debug_nans during unit testing only — don’t leave on for full run.

14 — Concrete code snippets & patterns (copy/paste)

A. Mesh + context (canonical)

from jax.sharding import Mesh
import numpy as np
from src.sharding.partition_specs import MP1, MP2, DP

device_array = np.arange(32).reshape((4,4,2))
mesh = Mesh(device_array, ('x','y','z'))  # x->MP1, y->MP2, z->DP
# Use mesh as context manager
with mesh:
    # call pjit compiled in this mesh
    ...


B. Parameter init with sharding

from jax.experimental.pjit import pjit
from jax.sharding import PartitionSpec as P

@pjit(in_axis_resources=None, out_axis_resources=P('x','y'))
def init_w(key, shape):
    return jax.random.normal(key, shape)

# inside mesh context
W = init_w(key, (hidden, hidden))  # will be sharded across x,y


C. All-reduce grads across data axis z

# inside train_step
grads = compute_grads(params, batch)
grads = jax.lax.pmean(grads, axis_name='z')  # reduces across DP axis


D. S5 dispatching: keep computations in complex64

def s5_apply(params, u):
    # params.Lambda is complex64; u float32 -> cast
    u_c = u.astype(jnp.complex64)
    # run discretize and associative_scan in complex64
    xs_c = s5_scan(params, u_c)  # complex64
    out = jnp.real(jnp.dot(params.C_tilde, xs_c)).astype(jnp.float32)
    return out

15 — Telemetry, logging and performance tuning

Track: tokens/sec, TFLOPS utilization estimate, latency on all_gather, per-axis comm bytes.

Start with small micro-benchmarks to find the best MP1 × MP2 × DP mapping. The TPU paper shows searching topology + partitioning (PA-NAS) can yield 1.2–2.3× gains; do a small grid search for Valkyrie with a synthetic workload. 

2304.01433v3

16 — Checklist before first 32-chip run (copy/paste)

 Unit tests pass on single host (S5 checks + attention checks).

 Mesh mapping code returns device_count() == 32.

 PartitionSpecs created centrally and used for init, forward, and optimizer update.

 Data pipeline shards files by jax.process_index().

 Orbax checkpointing implemented, master manifest recorded.

 Softmax and S5 sensitive ops use high precision.

 Logging/telemetry ready (tokens/sec, comm sizes).

 Small multi-host smoke test (2 hosts) passed.

 32-chip dry run with synthetic data for 5 steps passed (no NaNs).

 Backup plan: smaller slice (e.g., 16 chips) verified in case the 32 slice is congested.

17 — Additional TPU v4 paper calls-to-action (from the paper)

Use the 3D torus topology to your advantage: map data vs model axes consistently for best bisection throughput. The paper explicitly says mapping data along one torus dimension and the two model parameters on the other dimensions yields best use of bandwidth. 

2304.01433v3

If your model uses heavy embedding lookups (not the case for vanilla LLM tokens but if you add large vocabulary embeddings), consider partitioning embeddings via row/column/table sharding — TPU v4 has SparseCore features (internal to TPU) that accelerate embeddings but may not be accessible on all clouds; still, design embedding sharding for row-slicing across MP axes. The paper documents embedding partitioning options and the SparseCore architecture benefits. 

2304.01433v3

18 — If anything goes wrong: immediate diagnostic checklist

NaNs appear:

Run one step locally w/ jax.config.update('jax_debug_nans', True) (or set in small debug run).

Recompute attention softmax in float32; re-check S5 complex64 sections.

Slow all_gather / comm saturation:

Re-map heavy collective to an axis with better bisection or reduce tensor size by switching to more aggressive 2D weight sharding.

Reduce batch per device and increase dp shards.

Out of memory:

Add activation checkpointing; reduce microbatch size; increase model parallelism (shard weights more), or reduce pipeline bubble size.

Mismatched PartitionSpec error:

Print params tree and expected PartitionSpecs; ensure consistent init and pjit wrappers; centralize PartitionSpec definitions.

19 — Tests I will provide if you want (I can code these for you immediately)

Minimal pjit example for 32-device mesh (4×4×2) that initializes a small transformer with 2D sharding and runs one pjit train step.

S5 sequential vs associative_scan equality unit test, ready to run.

Chunked Longformer vs dense masked attention equality test.
Tell me which of the above you want code for and I’ll produce the exact files to drop into src/utils/tests.

20 — Final, brutal truth (no sugar)

Training a 1.2B+ LLM across TPU-v4-32 is very doable — TPU v4 is built for it — but it requires meticulous partitioning, careful handling of numerics, a well-tested pjit mesh, and a resilient checkpoint/restore strategy. The TPU paper shows that correctly co-optimizing topology and partitioning can yield major speedups; you must invest time in that tuning (PA-NAS-style grid search pays). 

2304.01433v3

The most common failures are not algorithmic but engineering: mismatched PartitionSpecs, collectives over the wrong axis, precision errors in attention/S5, and checkpoint restore bugs. The blueprint above is built to intercept all of those early.