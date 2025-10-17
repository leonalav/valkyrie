# Valkyrie Training System Design

## Overview

The Valkyrie training system implements a hybrid architecture combining **Longformer sparse attention** with **S5 state space models** for ultra-long context training (up to 657k tokens). The system is optimized for TPU v4-32 clusters with sophisticated sharding, chunked processing, and numerical stability measures.

## Architecture

### Model Components

1. **Longformer Attention**
   - Sliding window local attention (linear complexity)
   - Global attention tokens for long-range connections
   - Chunked/vectorized implementation for memory efficiency
   - Dual projections: local (Qs/Ks/Vs) and global (Qg/Kg/Vg)

2. **S5 State Space Model**
   - Diagonal continuous-time SSM with complex eigenvalues
   - Zero-Order Hold (ZOH) discretization
   - Parallel scan using JAX's associative_scan
   - Conjugate symmetry for real-valued outputs

3. **Hybrid Design**
   - S5 handles inter-chunk memory and long-range dependencies
   - Longformer handles intra-chunk attention efficiently
   - Progressive curriculum for stable training

### Training Strategy

#### Chunked Processing (657k Token Sequences)

```
Document (657k tokens)
├── Chunk 0 (8k tokens) → Longformer attention → S5 state update
├── Chunk 1 (8k tokens) → Longformer attention → S5 state update  
├── Chunk 2 (8k tokens) → Longformer attention → S5 state update
└── ... (82 chunks total)
```

**Key Features:**
- **Overlap**: 512 tokens between chunks for context continuity
- **S5 Memory**: Persistent state carries information between chunks
- **Truncated BPTT**: Backprop through 4-16 chunks, occasional long unrolls
- **Progressive Scaling**: Start with 2k chunks, scale to 64k chunks

#### Progressive Curriculum

| Phase | Chunk Size | Backprop Chunks | Max Steps | Learning Rate |
|-------|------------|-----------------|-----------|---------------|
| 0     | 2,048      | 2               | 5,000     | 2e-4          |
| 1     | 8,192      | 4               | 10,000    | 1.5e-4        |
| 2     | 32,768     | 8               | 20,000    | 1e-4          |
| 3     | 65,536     | 16              | 50,000    | 5e-5          |

## TPU v4-32 Optimization

### Mesh Topology

```
TPU v4-32 Mesh: 4×4×2 (32 chips)
├── x-axis (4): Model parallel dimension 1 (tensor width)
├── y-axis (4): Model parallel dimension 2 (tensor height)
└── z-axis (2): Data parallel dimension (batch sharding)
```

### Sharding Strategy

- **2D Tensor Parallelism**: Weight matrices sharded across x,y axes
- **Data Parallelism**: Batches sharded across z axis
- **Embedding Sharding**: Vocabulary dimension sharded across x axis
- **S5 Parameters**: Kept replicated for numerical stability

### Communication Patterns

- **Gradient Reduction**: All-reduce across z axis (data parallel)
- **Parameter Gathering**: All-gather across x,y axes (model parallel)
- **Optimized Collectives**: Fused operations for 3D torus topology

## Numerical Stability

### Mixed Precision Policy

| Component | Precision | Reason |
|-----------|-----------|---------|
| Parameters | fp32 | Stability and gradient precision |
| General Compute | bfloat16 | TPU efficiency |
| Attention Softmax | fp32 | **Critical**: Prevents NaNs |
| S5 Complex Math | complex64 | **Critical**: S5 stability |
| Layer Normalization | fp32 | Numerical stability |
| Loss Computation | fp32 | Gradient precision |

### Critical Stability Measures

1. **S5 Complex Arithmetic**
   - All S5 parameters (Λ, B̃, C̃) in complex64
   - Conjugate symmetry enforced for real outputs
   - Discretization in complex64, cast to fp32 only at output

2. **Attention Precision**
   - QK^T matmul and softmax in fp32
   - Prevents NaN/overflow issues observed in fp16 attention
   - Cast back to bfloat16 for subsequent operations

3. **Gradient Flow**
   - Gradients computed and stored in fp32
   - Gradient clipping applied before mixed precision casting
   - S5 gradient paths validated with finite differences

## Data Pipeline

### FineWeb Dataset Processing

```python
# Dataset loading
fw = load_dataset("HuggingFaceFW/fineweb", name="CC-MAIN-2024-10", split="train", streaming=True)

# Multi-host sharding
dataset_shard = dataset.shard(num_shards=process_count, index=process_index)

# Document processing
document → tokenize → chunk (8k tokens) → overlap (512 tokens) → batch
```

### Multi-Host Coordination

- **Data Sharding**: Each host processes different document shards
- **Synchronization**: Gradient reduction across all hosts
- **Load Balancing**: Round-robin document distribution

## Implementation Details

### File Organization

```
src/
├── model/           # Exact math from 1_jax.py (NO MODIFICATIONS)
│   ├── modules.py   # ValkyrieConfig, RMSNorm, RoPE
│   ├── s5.py        # ValkyrieS5 with complex arithmetic
│   ├── longformer.py # ValkyrieLongformerAttention
│   └── valkyrie.py  # ValkyrieModel, ValkyrieBlock
├── sharding/        # TPU v4-32 mesh and partitioning
├── train/           # Chunked training loop
├── data/            # FineWeb pipeline
├── io/              # Checkpointing and logging
└── utils/           # Testing and debugging
```

### Key Algorithms

#### S5 Parallel Scan

```python
# Discretization: Λ̄ = exp(Λ * Δ), B̄ = (Λ̄ - I) / Λ * B̃
Lambda_bar = jnp.exp(Lambda * Delta)
B_bar = (Lambda_bar - 1.0) / Lambda * B_tilde

# Parallel scan with associative operator
elements = (A_elements, Bu_elements)
_, xs = jax.lax.associative_scan(binary_operator, elements, axis=1)

# Output: y = C̃ @ x + D * u
output = jnp.real(C_tilde @ xs) + D * u
```

#### Longformer Chunked Attention

```python
# Process sequence in chunks
for chunk_idx in range(num_chunks):
    # Extract chunk queries
    q_chunk = qs[:, :, chunk_start:chunk_end, :]
    
    # Define sliding window for keys/values
    window_start = max(0, chunk_start - window_size // 2)
    window_end = min(T, chunk_end + window_size // 2)
    
    # Compute attention within window
    scores = einsum('bhqd,bhkd->bhqk', q_chunk, k_window) * scale
    scores = where(window_mask, scores, -1e9)
    attn_weights = softmax(scores)
    chunk_output = einsum('bhqk,bhkd->bhqd', attn_weights, v_window)
```

## Performance Targets

### Throughput
- **Target**: >1000 tokens/sec/chip on TPU v4-32
- **Memory**: Support 657k context with 32 GiB/chip
- **Scaling**: Linear scaling across TPU pods

### Memory Budget (1.2B Parameter Model)

| Component | Memory (GB) | Notes |
|-----------|-------------|-------|
| Parameters (fp32) | 4.8 | Model weights |
| Parameters (bf16) | 2.4 | Mixed precision copy |
| Optimizer State | 9.6 | Adam m,v in fp32 |
| Gradients | 4.8 | fp32 gradients |
| **Total Static** | **21.6** | Before activations |
| Activations | Variable | Chunked to fit memory |

## Validation and Testing

### Critical Tests (Must Pass)

1. **S5 Parallel vs Sequential Equality**
   ```python
   parallel_output = s5_layer.parallel_scan(Lambda_bar, B_bar, u)
   sequential_output = sequential_recurrence(Lambda_bar, B_bar, u)
   assert jnp.allclose(parallel_output, sequential_output, rtol=1e-5)
   ```

2. **Longformer Chunked vs Dense Equality**
   ```python
   chunked_output = chunked_attention(qs, ks, vs)
   dense_output = dense_masked_attention(qs, ks, vs)
   assert jnp.allclose(chunked_output, dense_output, rtol=1e-4)
   ```

3. **Mixed Precision NaN Detection**
   ```python
   # Run with fp32 attention, verify no NaNs
   # Run with fp16 attention, expect NaNs (validates our fp32 policy)
   ```

### Performance Benchmarks

- **Overfit Test**: Single batch should reach near-zero loss in 1000 steps
- **Throughput Test**: Measure tokens/sec/chip under full load
- **Memory Test**: Verify 657k context fits within memory budget
- **Scaling Test**: Linear throughput scaling across multiple TPU pods

## Failure Modes and Mitigations

### Common Issues

1. **NaN/Inf in Attention**
   - **Cause**: fp16 softmax overflow
   - **Solution**: Force attention softmax to fp32

2. **S5 Complex Arithmetic Errors**
   - **Cause**: Dtype mixing in complex operations
   - **Solution**: Keep all S5 math in complex64 until final output

3. **Memory Overflow**
   - **Cause**: Trying to process full 657k sequence at once
   - **Solution**: Chunked processing with S5 state management

4. **TPU Communication Bottlenecks**
   - **Cause**: Inefficient collective operations
   - **Solution**: 2D sharding with optimized reduction patterns

### Recovery Strategies

- **Automatic Checkpointing**: Every 1000 steps with async saves
- **Emergency Checkpoints**: On training interruption
- **Host Failure Recovery**: Multi-host checkpoint coordination
- **Gradient Explosion**: Automatic gradient clipping and LR reduction

## Usage Examples

### Basic Training

```bash
# Launch training on TPU v4-32
python -m src.train.main --config configs/valkyrie_base.yaml
```

### Development/Debugging

```bash
# Run with validation tests
python -m src.train.main --config configs/valkyrie_base.yaml --validate_only

# Debug mode with detailed logging
python -m src.train.main --config configs/valkyrie_base.yaml --debug
```

### Resume Training

```bash
# Resume from specific checkpoint
python -m src.train.main --config configs/valkyrie_base.yaml --resume_from checkpoints/full_checkpoint_00010000
```

## References

- **Longformer**: Beltagy et al., "Longformer: The Long-Document Transformer"
- **S5**: Smith et al., "Efficiently Modeling Long Sequences with Structured State Spaces"  
- **TPU v4**: Jouppi et al., "TPU v4: An Optically Reconfigurable Supercomputer"
- **JAX**: Bradbury et al., "JAX: composable transformations of Python+NumPy programs"

## Mathematical Verification

All mathematical implementations are **EXACTLY COPIED** from `1_jax.py` without modifications:

- ✅ S5 discretization and parallel scan
- ✅ Longformer sliding window and global attention  
- ✅ RoPE rotary position embeddings
- ✅ Complex arithmetic with conjugate symmetry
- ✅ Gradient computation and backpropagation

**DO NOT MODIFY** the mathematical implementations - they are verified and tested.