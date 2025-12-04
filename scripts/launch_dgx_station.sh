#!/bin/bash
# Launch script optimized for DGX Station (Tesla V100-DGS)
# Multi-user environment with direct access

set -e

echo "DGX Station Pneumonia Detection Training"
echo "=============================================="

# Configuration
CONFIG_FILE=${1:-"configs/dgx_station_config.yaml"}
WORLD_SIZE=${2:-4}
EXPERIMENT_NAME=${3:-"dgx_station_$(date +%Y%m%d_%H%M%S)"}

# Validate arguments
if [ ! -f "$CONFIG_FILE" ]; then
    echo "ERROR: Configuration file not found: $CONFIG_FILE"
    echo "Usage: $0 [config_file] [world_size] [experiment_name]"
    exit 1
fi

echo "📋 Configuration:"
echo "  Config file: $CONFIG_FILE"
echo "  World size: $WORLD_SIZE GPUs"
echo "  Experiment: $EXPERIMENT_NAME"
echo ""

# System checks
echo "System Verification:"

# Check GPU availability
if ! command -v nvidia-smi &> /dev/null; then
    echo "ERROR: nvidia-smi not found. CUDA drivers not installed?"
    exit 1
fi

# Check GPU count
GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
echo "  Available GPUs: $GPU_COUNT"

if [ "$GPU_COUNT" -lt "$WORLD_SIZE" ]; then
    echo "ERROR: Requested $WORLD_SIZE GPUs but only $GPU_COUNT available"
    exit 1
fi

# Check GPU model (should be V100)
GPU_MODEL=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -1)
echo "  GPU Model: $GPU_MODEL"

if [[ ! "$GPU_MODEL" == *"V100"* ]]; then
    echo "WARNING: Expected Tesla V100, found $GPU_MODEL"
    echo "  Proceeding anyway, but performance may not be optimal"
fi

# Check VRAM
VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
echo "  VRAM per GPU: ${VRAM} MiB"

if [ "$VRAM" -lt 30000 ]; then
    echo "WARNING: Expected 32GB V100, found ${VRAM}MiB"
    echo "  Consider reducing batch size if OOM occurs"
fi

# Check CUDA version
CUDA_VERSION=$(nvcc --version | grep "release" | sed 's/.*release //' | sed 's/,.*//')
echo "  CUDA Version: $CUDA_VERSION"

# Check available disk space
DISK_SPACE=$(df -h . | awk 'NR==2 {print $4}')
echo "  Available disk space: $DISK_SPACE"

echo "System checks passed"
echo ""

# Environment setup for DGX Station V100
echo "Setting up DGX Station environment..."

# V100-specific environment variables
export CUDA_VISIBLE_DEVICES=0,1,2,3
export NCCL_DEBUG=INFO
export NCCL_TREE_THRESHOLD=0
export NCCL_IB_DISABLE=1  # Use NVLink instead of InfiniBand
export NCCL_P2P_DISABLE=0  # Enable P2P over NVLink
export NCCL_SOCKET_IFNAME=^lo,docker0

# CPU affinity for V100 DGX Station (40 cores)
export OMP_NUM_THREADS=10  # 40 cores / 4 GPUs

# Memory optimizations
export CUDA_LAUNCH_BLOCKING=0
export CUDNN_BENCHMARK=1

# Multi-user considerations
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps
export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log

echo "  Environment configured for Tesla V100"

# Create output directory
OUTPUT_DIR="outputs/$EXPERIMENT_NAME"
mkdir -p "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR/tensorboard"
mkdir -p "$OUTPUT_DIR/checkpoints"

echo "Output directory: $OUTPUT_DIR"

# Copy config for reproducibility
cp "$CONFIG_FILE" "$OUTPUT_DIR/config_used.yaml"

# Resource monitoring setup
echo "Setting up monitoring..."

# Start GPU monitoring in background
nvidia-smi dmon -i 0,1,2,3 -s pucvmet -c 1000 > "$OUTPUT_DIR/gpu_monitor.log" 2>&1 &
MONITOR_PID=$!

# Function to cleanup monitoring on exit
cleanup() {
    echo ""
    echo "🧹 Cleaning up..."
    if [ ! -z "$MONITOR_PID" ]; then
        kill $MONITOR_PID 2>/dev/null || true
    fi
    echo "Cleanup completed"
}
trap cleanup EXIT

# Pre-training checks
echo "Pre-training validation..."

# Check data directory
DATA_DIR=$(grep "train_dir:" "$CONFIG_FILE" | awk '{print $2}' | tr -d '"')
if [ ! -z "$DATA_DIR" ] && [ ! -d "$DATA_DIR" ]; then
    echo "ERROR: Training data directory not found: $DATA_DIR"
    echo "Please update the config file or create the data directory"
    exit 1
fi

# Check Python environment
if ! python -c "import torch, torchvision, timm" 2>/dev/null; then
    echo "ERROR: Required Python packages not found"
    echo "Please install: pip install torch torchvision timm"
    exit 1
fi

# Check PyTorch CUDA support
TORCH_CUDA=$(python -c "import torch; print(torch.cuda.is_available())" 2>/dev/null)
if [ "$TORCH_CUDA" != "True" ]; then
    echo "ERROR: PyTorch CUDA support not available"
    echo "Please install CUDA-enabled PyTorch"
    exit 1
fi

echo "Pre-training checks passed"
echo ""

# Launch training
echo "Launching distributed training..."
echo "Time: $(date)"
echo "PID: $$"
echo "Output: $OUTPUT_DIR"
echo ""

# Log system info
{
    echo "=== DGX Station System Info ==="
    echo "Date: $(date)"
    echo "User: $(whoami)"
    echo "Host: $(hostname)"
    echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
    echo ""
    echo "=== GPU Information ==="
    nvidia-smi
    echo ""
    echo "=== Configuration Used ==="
    cat "$CONFIG_FILE"
} > "$OUTPUT_DIR/system_info.log"

# Start TensorBoard in background (multi-user friendly port)
TENSORBOARD_PORT=$((6006 + $RANDOM % 1000))
echo "🔗 Starting TensorBoard on port $TENSORBOARD_PORT"
tensorboard --logdir="$OUTPUT_DIR/tensorboard" --host=0.0.0.0 --port=$TENSORBOARD_PORT > "$OUTPUT_DIR/tensorboard.log" 2>&1 &
TENSORBOARD_PID=$!

# Add TensorBoard cleanup to trap
cleanup() {
    echo ""
    echo "🧹 Cleaning up..."
    if [ ! -z "$MONITOR_PID" ]; then
        kill $MONITOR_PID 2>/dev/null || true
    fi
    if [ ! -z "$TENSORBOARD_PID" ]; then
        kill $TENSORBOARD_PID 2>/dev/null || true
    fi
    echo "Cleanup completed"
}
trap cleanup EXIT

echo "TensorBoard available at: http://$(hostname):$TENSORBOARD_PORT"
echo ""

# Launch the actual training
echo "Starting training on $WORLD_SIZE Tesla V100 GPUs..."

# Use Python's multiprocessing for distributed training
python -m torch.distributed.launch \
    --nproc_per_node=$WORLD_SIZE \
    --master_addr=localhost \
    --master_port=12355 \
    src/training/distributed_trainer.py \
    --config "$CONFIG_FILE" \
    --output_dir "$OUTPUT_DIR" \
    2>&1 | tee "$OUTPUT_DIR/training.log"

TRAINING_EXIT_CODE=$?

# Post-training summary
echo ""
echo "📋 Training Summary"
echo "==================="
echo "Exit code: $TRAINING_EXIT_CODE"
echo "Duration: $(date)"
echo "Output directory: $OUTPUT_DIR"
echo "TensorBoard: http://$(hostname):$TENSORBOARD_PORT"
echo ""

if [ $TRAINING_EXIT_CODE -eq 0 ]; then
    echo "Training completed successfully!"

    # Show final results if available
    if [ -f "$OUTPUT_DIR/training.log" ]; then
        echo ""
        echo "Final Results:"
        tail -20 "$OUTPUT_DIR/training.log" | grep -E "(accuracy|loss|epoch)" || echo "No final metrics found in log"
    fi
else
    echo "ERROR: Training failed with exit code $TRAINING_EXIT_CODE"
    echo "Check the logs in $OUTPUT_DIR/ for details"
fi

echo ""
echo "All outputs saved to: $OUTPUT_DIR"
echo "DGX Station training session complete!"

exit $TRAINING_EXIT_CODE