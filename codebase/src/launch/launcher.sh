#!/bin/bash

# Valkyrie Training Launcher for TPU v4-32
# Multi-host training script with proper TPU setup

set -e  # Exit on any error

# Configuration
CONFIG_FILE=${1:-"configs/valkyrie_base.yaml"}
TPU_NAME=${TPU_NAME:-"valkyrie-tpu"}
ZONE=${ZONE:-"us-central2-b"}
PROJECT=${PROJECT:-"your-project-id"}

# Logging
LOG_DIR="./logs/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

echo "=== Valkyrie Training Launcher ==="
echo "Config: $CONFIG_FILE"
echo "TPU: $TPU_NAME"
echo "Zone: $ZONE"
echo "Project: $PROJECT"
echo "Log dir: $LOG_DIR"
echo "=================================="

# Validate configuration
if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ Configuration file not found: $CONFIG_FILE"
    exit 1
fi

# Check TPU availability
echo "Checking TPU availability..."
gcloud compute tpus describe "$TPU_NAME" --zone="$ZONE" --project="$PROJECT" > /dev/null 2>&1
if [ $? -ne 0 ]; then
    echo "❌ TPU not found or not accessible: $TPU_NAME"
    echo "Please create TPU with:"
    echo "  gcloud compute tpus create $TPU_NAME --zone=$ZONE --accelerator-type=v4-32 --version=tpu-vm-tf-2.13.0"
    exit 1
fi

echo "✓ TPU found: $TPU_NAME"

# Get TPU internal IPs
echo "Getting TPU worker addresses..."
TPU_WORKERS=$(gcloud compute tpus describe "$TPU_NAME" --zone="$ZONE" --project="$PROJECT" --format="value(networkEndpoints[].ipAddress)" | tr '\n' ',')
TPU_WORKERS=${TPU_WORKERS%,}  # Remove trailing comma

if [ -z "$TPU_WORKERS" ]; then
    echo "❌ Could not get TPU worker addresses"
    exit 1
fi

echo "✓ TPU workers: $TPU_WORKERS"

# Set environment variables for JAX/TPU
export JAX_PLATFORMS=tpu
export TPU_CHIPS_PER_HOST_BOUNDS=2,2,1
export TPU_HOST_BOUNDS=2,2,2
export JAX_ENABLE_XLA_PYTHON_CLIENT_ALLOCATOR=false

# XLA optimization flags
export XLA_FLAGS="--xla_tpu_enable_async_collective_fusion=true --xla_tpu_enable_async_collective_fusion_fuse_all_gather=true"

# Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

# Training command
TRAIN_CMD="python -m src.train.main --config $CONFIG_FILE --log_dir $LOG_DIR"

echo "Starting multi-host training..."
echo "Command: $TRAIN_CMD"

# Launch training on all TPU workers
gcloud compute tpus tpu-vm ssh "$TPU_NAME" \
    --zone="$ZONE" \
    --project="$PROJECT" \
    --worker=all \
    --command="
        set -e
        cd /tmp/valkyrie-training
        
        # Set environment
        export JAX_PLATFORMS=tpu
        export TPU_CHIPS_PER_HOST_BOUNDS=2,2,1
        export TPU_HOST_BOUNDS=2,2,2
        export JAX_ENABLE_XLA_PYTHON_CLIENT_ALLOCATOR=false
        export XLA_FLAGS='--xla_tpu_enable_async_collective_fusion=true --xla_tpu_enable_async_collective_fusion_fuse_all_gather=true'
        export PYTHONPATH=\${PYTHONPATH}:\$(pwd)/src
        
        # Install dependencies if needed
        if [ ! -f .deps_installed ]; then
            echo 'Installing dependencies...'
            pip install -r requirements.txt
            touch .deps_installed
        fi
        
        # Run training
        echo 'Starting training on worker \$HOSTNAME...'
        $TRAIN_CMD 2>&1 | tee $LOG_DIR/worker_\$(hostname).log
    "

echo "Training launched on all TPU workers"
echo "Logs will be saved to: $LOG_DIR"

# Monitor training (optional)
if [ "$MONITOR" = "true" ]; then
    echo "Monitoring training progress..."
    
    while true; do
        sleep 60
        
        # Check if training is still running
        RUNNING=$(gcloud compute tpus tpu-vm ssh "$TPU_NAME" --zone="$ZONE" --project="$PROJECT" --worker=0 --command="pgrep -f 'python.*train.main' | wc -l" 2>/dev/null || echo "0")
        
        if [ "$RUNNING" = "0" ]; then
            echo "Training completed or stopped"
            break
        fi
        
        echo "Training still running... ($(date))"
    done
fi

echo "=== Training Complete ==="