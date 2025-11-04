# TPU v4-32 Setup and Training Guide

This guide covers setting up and running Valkyrie model training on TPU v4-32 with multi-host distributed training.

## Overview

TPU v4-32 configuration:
- **4 hosts** × **8 chips per host** = **32 total chips**
- **3D mesh topology**: 4×4×2 (data × model × fsdp)
- **Multi-host distributed training** with gradient synchronization
- **Mixed precision**: BF16 activations, FP32 parameters

## Architecture

### Mesh Configuration
```
Logical Mesh: (4, 4, 2)
├── data axis (4-way): Data parallelism across hosts
├── model axis (4-way): Model/tensor parallelism within hosts  
└── fsdp axis (2-way): Fully Sharded Data Parallelism
```

### Sharding Strategy
- **Parameters**: Sharded across `model` and `fsdp` axes
- **Activations**: Sharded across `data` and `model` axes
- **Gradients**: Synchronized across `data` axis (multi-host)
- **Batch**: Sharded across `data` axis (4-way split)

## Environment Setup

### Required Environment Variables
```bash
# TPU configuration
export TPU_MESH_CONFIG="4,4,2"
export TPU_AXIS_NAMES="data,model,fsdp"
export JAX_PLATFORMS="tpu"

# Multi-host configuration (auto-detected if not set)
export JAX_COORDINATOR_ADDRESS="<coordinator-ip>:8476"
export JAX_PROCESS_INDEX="<0-3>"  # Host rank
export JAX_PROCESS_COUNT="4"      # Total hosts

# Optional: HuggingFace token for data access
export HF_TOKEN="your_token_here"
```

### TPU Pod Commands

#### Running on All Hosts
```bash
gcloud compute tpus tpu-vm ssh node-1 \
    --zone=us-central2-b \
    --worker=all \
    --command="python3.11 -m examples.tpu_v4_32_training"
```

#### Running Training Script
```bash
gcloud compute tpus tpu-vm ssh node-1 \
    --zone=us-central2-b \
    --worker=all \
    --command="cd /home/ravkeave/valkyrie && python3.11 -m src.train.main --config configs/tpu_v4_32.yaml"
```

#### File Transfer
```bash
gcloud compute tpus tpu-vm scp /local/path/file.py node-1://remote/path/ \
    --zone=us-central2-b \
    --worker=all
```

## Code Integration

### 1. Multi-Host Initialization
```python
from src.sharding.distributed_init import setup_multi_host_environment

# Initialize distributed runtime (call first)
dist_config = setup_multi_host_environment()
```

### 2. Mesh Setup
```python
from src.sharding.mesh_setup import setup_tpu_mesh

# Create 3D mesh for TPU v4-32
mesh = setup_tpu_mesh(
    device_count=32,
    use_global=True,
    validate=True
)
```

### 3. Training Step Creation
```python
from src.train.step_fn import create_train_step

# Create training step with 3D mesh support
train_step_fn = create_train_step(
    model=model,
    optimizer=optimizer,
    config=config,
    mesh=mesh,
    mixed_precision=True,
    use_2d_sharding=True,  # Enable 2D tensor parallelism
    use_3d_mesh=True       # Enable 3D mesh (v4-32)
)
```

### 4. Model Configuration
```python
from src.model.config import ValkyrieConfig

config = ValkyrieConfig(
    vocab_size=32000,
    hidden_size=4096,
    num_layers=32,
    num_heads=32,
    intermediate_size=11008,
    max_seq_len=2048,
    use_flash_attention=True,
    # ... other config options
)
```

## Performance Optimization

### Batch Size Guidelines
- **Global batch size**: 32-128 (adjust based on memory)
- **Local batch size**: Global / 4 (data parallel factor)
- **Sequence length**: Up to 2048 tokens
- **Gradient accumulation**: Use for larger effective batch sizes

### Memory Optimization
- **Mixed precision**: BF16 activations, FP32 parameters
- **Gradient checkpointing**: Enabled for large models
- **FSDP**: 2-way sharding reduces memory per chip
- **Flash attention**: Reduces memory for long sequences

### Communication Optimization
- **Gradient synchronization**: Only across `data` axis
- **Parameter sharding**: Across `model` and `fsdp` axes
- **Minimal host communication**: Optimized for TPU topology

## Troubleshooting

### Common Issues

#### 1. Coordinator Address Not Found
```bash
# Manually set coordinator address
export JAX_COORDINATOR_ADDRESS="10.0.0.2:8476"
```

#### 2. Process Index Detection Failed
```bash
# Manually set process index for each host
export JAX_PROCESS_INDEX="0"  # Host 0
export JAX_PROCESS_INDEX="1"  # Host 1
# ... etc
```

#### 3. Mesh Shape Mismatch
```bash
# Verify device count matches mesh topology
python -c "import jax; print(f'Devices: {len(jax.devices())}')"
# Should output: Devices: 32
```

#### 4. Out of Memory
- Reduce batch size or sequence length
- Enable gradient checkpointing
- Increase FSDP sharding (model axis)

### Debugging Commands

#### Check TPU Status
```bash
gcloud compute tpus tpu-vm describe node-1 --zone=us-central2-b
```

#### Monitor Training
```bash
# Check logs on all hosts
gcloud compute tpus tpu-vm ssh node-1 \
    --zone=us-central2-b \
    --worker=all \
    --command="tail -f /tmp/training.log"
```

#### Device Information
```python
import jax
print(f"Local devices: {jax.local_device_count()}")
print(f"Global devices: {jax.device_count()}")
print(f"Process index: {jax.process_index()}")
print(f"Process count: {jax.process_count()}")
```

## Example Training Script

See `examples/tpu_v4_32_training.py` for a complete example that demonstrates:
- Multi-host initialization
- 3D mesh setup
- Training and evaluation steps
- Performance monitoring
- Error handling

## Performance Expectations

### Typical Performance (7B model)
- **Training throughput**: ~2000-4000 tokens/second
- **Step time**: 100-200ms per step
- **Memory usage**: ~80% of available TPU memory
- **Communication overhead**: <10% of step time

### Scaling Characteristics
- **Linear scaling**: Up to 4-way data parallelism
- **Model parallelism**: Efficient for large models (>7B parameters)
- **FSDP**: Reduces memory usage by 2x
- **Multi-host**: Minimal overhead with proper setup

## Best Practices

1. **Always initialize distributed runtime first**
2. **Use environment variables for configuration**
3. **Validate mesh setup before training**
4. **Monitor memory usage and adjust batch size**
5. **Use mixed precision for optimal performance**
6. **Enable gradient checkpointing for large models**
7. **Test with synthetic data before real training**
8. **Save checkpoints frequently in multi-host setup**

## Support

For issues specific to TPU v4-32 setup:
1. Check the troubleshooting section above
2. Verify environment variables are set correctly
3. Test with the provided example script
4. Monitor TPU utilization and memory usage
5. Check logs on all hosts for distributed issues