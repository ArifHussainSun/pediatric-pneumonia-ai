#!/bin/bash

# Launch script for distributed training on DGX station
# Usage: ./scripts/launch_dgx_training.sh [config_file] [num_gpus]

set -e

# Default values
CONFIG_FILE=${1:-"configs/train_config.yaml"}
NUM_GPUS=${2:-4}
MASTER_PORT=${3:-12355}

echo "======================================="
echo "DGX Distributed Training Launch Script"
echo "======================================="
echo "Config file: $CONFIG_FILE"
echo "Number of GPUs: $NUM_GPUS"
echo "Master port: $MASTER_PORT"
echo ""

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file '$CONFIG_FILE' not found!"
    exit 1
fi

# Check GPU availability
echo "Checking GPU availability..."
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader,nounits

# Check if we have enough GPUs
AVAILABLE_GPUS=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
echo "Available GPUs: $AVAILABLE_GPUS"

if [ $NUM_GPUS -gt $AVAILABLE_GPUS ]; then
    echo "Error: Requested $NUM_GPUS GPUs but only $AVAILABLE_GPUS available!"
    exit 1
fi

# Set environment variables for optimal DGX performance
export CUDA_VISIBLE_DEVICES=0,1,2,3
export NCCL_DEBUG=INFO
export NCCL_TREE_THRESHOLD=0
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=^lo,docker0

# Create outputs directory if it doesn't exist
mkdir -p outputs

echo ""
echo "Starting distributed training..."
echo "======================================="

# Launch distributed training
python3 -m src.training.distributed_trainer \
    --config $CONFIG_FILE \
    --world-size $NUM_GPUS \
    --master-addr localhost \
    --master-port $MASTER_PORT

echo ""
echo "Training completed!"
echo "======================================="