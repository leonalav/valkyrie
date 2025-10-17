# Valkyrie Training Codebase

A high-performance training implementation for the Valkyrie model combining Longformer sparse attention with S5 state space models, optimized for TPU v4-32 clusters.

## Architecture

- **Longformer Attention**: Sliding window local attention + global attention tokens for linear complexity
- **S5 State Space Model**: Diagonal continuous-time SSM with parallel scan for long-range dependencies  
- **Hybrid Design**: S5 handles inter-chunk memory, Longformer handles intra-chunk attention
- **TPU Optimized**: 2D tensor parallelism + data parallelism on TPU v4-32 (4×4×2 mesh)

## Key Features

- **Ultra-Long Context**: Supports 657k+ token sequences via chunked processing
- **Numerical Stability**: fp32 precision for attention softmax and S5 complex arithmetic
- **Multi-Host Training**: Distributed data pipeline with proper TPU sharding
- **Progressive Curriculum**: Gradual sequence length scaling for stable training
- **State Management**: Persistent S5 states between chunks with checkpointing

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Configure TPU mesh and model settings
cp configs/valkyrie_base.yaml configs/my_config.yaml
# Edit configs/my_config.yaml as needed

# Launch training on TPU v4-32
python -m src.launch.launcher --config configs/my_config.yaml
```

## Directory Structure

```
├── src/
│   ├── model/          # Model architecture (Longformer + S5)
│   ├── sharding/       # TPU mesh setup and partitioning
│   ├── train/          # Training loop and optimization
│   ├── data/           # FineWeb data pipeline
│   ├── io/             # Checkpointing and logging
│   └── utils/          # Testing and debugging utilities
├── configs/            # YAML configuration files
└── docs/               # Documentation and design notes
```

## Training Strategy

1. **Chunked Processing**: Split 657k sequences into 8k-32k token chunks
2. **S5 Memory**: Use S5 states to carry information between chunks  
3. **Truncated BPTT**: Backprop through limited chunks, occasional long unrolls
4. **Mixed Precision**: fp16/fp8 for GEMMs, fp32 for numerically sensitive ops
5. **Progressive Scaling**: Start with short sequences, gradually increase length

## TPU Configuration

- **Mesh Topology**: 4×4×2 (32 chips total)
- **Parallelism**: 2D tensor parallel (x,y axes) + data parallel (z axis)  
- **Memory**: ~32 GiB HBM per chip, aggressive parameter sharding
- **Communication**: Optimized collectives for 3D torus topology

## Dataset

Training on HuggingFace FineWeb-edu dataset:
```python
from datasets import load_dataset
fw = load_dataset("HuggingFaceFW/fineweb", name="CC-MAIN-2024-10", split="train", streaming=True)
```

## Safety & Validation

- S5 parallel vs sequential equality tests
- Longformer chunked vs dense attention validation  
- Mixed precision NaN detection
- Multi-host synchronization checks
- Checkpoint integrity verification

## Performance Targets

- **Throughput**: >1000 tokens/sec/chip on TPU v4-32
- **Memory Efficiency**: Support 657k context with 32 GiB/chip
- **Scalability**: Linear scaling across TPU pods
- **Stability**: No NaNs, proper gradient flow through S5 complex arithmetic

## Citation

Based on:
- Longformer: The Long-Document Transformer (Beltagy et al.)
- Efficiently Modeling Long Sequences with Structured State Spaces (S5, Smith et al.)
- TPU v4: An Optically Reconfigurable Supercomputer (Jouppi et al.)