# Deployment Guide for DGX Station

This guide provides step-by-step instructions for deploying the pediatric pneumonia detection system on DGX stations and production environments.

## Overview

This deployment guide covers:

- DGX station setup and configuration
- Multi-GPU distributed training
- Docker containerization
- Production deployment options
- Monitoring and maintenance

## DGX Station Deployment

### System Requirements

**Hardware:**

- DGX Station with 4x Tesla V100 GPUs (32GB VRAM each)
- 40+ CPU cores (Intel Xeon E5-2698 v4)
- 256GB+ system RAM
- NVMe SSD storage (recommended)

**Software:**

- Ubuntu 20.04+ or DGX OS
- CUDA 11.8+ with cuDNN
- Docker with nvidia-container-runtime
- Python 3.8+

### Initial Setup

#### 1. Verify GPU Configuration

```bash
# Check GPU status
nvidia-smi

# Expected output for DGX station:
# 4x Tesla V100-DGXS-32GB GPUs
# Driver Version: 470.57.02+
# CUDA Version: 11.8+
```

#### 2. Verify CUDA Installation

```bash
# Check CUDA version
nvcc --version

# Test CUDA functionality
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}')"
```

#### 3. Clone and Setup Repository

```bash
# Clone the repository
git clone <repository-url>
cd pediatric-pneumonia-ai

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import src; print('Installation successful!')"
```

### Data Preparation

#### 1. Organize Dataset

```bash
# Create data directory structure
mkdir -p data/{train,test,val}/{NORMAL,PNEUMONIA}

# Copy your dataset to the appropriate directories
# Example structure:
# data/
# ├── train/
# │   ├── NORMAL/     (1341 normal X-rays)
# │   └── PNEUMONIA/  (3875 pneumonia X-rays)
# ├── test/
# │   ├── NORMAL/     (234 normal X-rays)
# │   └── PNEUMONIA/  (390 pneumonia X-rays)
# └── val/            (optional validation set)
```

#### 2. Verify Data Integrity

```bash
# Run data verification script
python scripts/verify_data.py --data_dir data/

# Expected output:
# Data verification completed
# Training samples: 5,216
# Test samples: 624
# Class balance ratio: 2.89
```

### Configuration

#### 1. Training Configuration

Create or modify `configs/dgx_train_config.yaml`:

```yaml
# DGX-optimized configuration
model:
  type: "xception"
  params:
    num_classes: 2
    freeze_layers: 100
    dropout_rate: 0.5

training:
  epochs: 100
  batch_size: 32 # Per GPU (effective batch size = 32 * 4 = 128)
  gradient_accumulation_steps: 1
  use_amp: true # Essential for V100 optimization
  max_grad_norm: 1.0

optimizer:
  type: "adamw"
  lr: 0.0001
  weight_decay: 0.01
  betas: [0.9, 0.999]

scheduler:
  type: "cosine"
  min_lr: 0.000001

data:
  train_dir: "data/train"
  val_dir: "data/test"
  image_size: 224
  num_workers: 8 # Per GPU
  pin_memory: true
  prefetch_factor: 2

# DGX-specific settings
distributed:
  backend: "nccl"
  master_addr: "localhost"
  master_port: 12355

output_dir: "./outputs/dgx_experiment"
log_interval: 50
```

#### 2. Environment Variables

```bash
# Add to ~/.bashrc or create launch script
export CUDA_VISIBLE_DEVICES=0,1,2,3
export NCCL_DEBUG=INFO
export NCCL_TREE_THRESHOLD=0
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=^lo,docker0
```

### Training Execution

#### 1. Single Model Training

```bash
# Method 1: Using the launch script (recommended)
./scripts/launch_dgx_training.sh configs/dgx_train_config.yaml 4

# Method 2: Direct Python execution
python -m src.training.distributed_trainer \
    --config configs/dgx_train_config.yaml \
    --world-size 4 \
    --master-addr localhost \
    --master-port 12355
```

#### 2. Multiple Model Comparison

```bash
# Train multiple models in sequence
for model in xception vgg mobilenet fusion; do
    echo " Training $model model..."

    # Update config for current model
    sed -i "s/type: \".*\"/type: \"$model\"/" configs/dgx_train_config.yaml

    # Launch training
    ./scripts/launch_dgx_training.sh configs/dgx_train_config.yaml 4

    # Wait for completion
    wait
done
```

#### 3. Monitoring Training

```bash
# Monitor GPU utilization
watch -n 1 nvidia-smi

# Monitor training logs
tail -f outputs/dgx_experiment/training.log

# Launch TensorBoard (in separate terminal)
tensorboard --logdir outputs/dgx_experiment/tensorboard --host 0.0.0.0 --port 6006
```

### Performance Optimization

#### 1. Memory Optimization

```yaml
# In your config file
training:
  batch_size: 24 # Reduce if OOM
  gradient_accumulation_steps: 2 # Maintain effective batch size
  use_amp: true # Essential for memory efficiency

data:
  num_workers: 6 # Reduce if CPU bottleneck
  pin_memory: true
  prefetch_factor: 1 # Reduce if memory issues
```

#### 2. Speed Optimization

```bash
# Set optimal CPU affinity
export OMP_NUM_THREADS=10  # 40 cores / 4 GPUs

# Enable CUDA optimizations
export CUDA_LAUNCH_BLOCKING=0
export CUDNN_BENCHMARK=1
```

## Docker Deployment

### Building the Container

#### 1. Standard Deployment

```bash
# Build the container
docker build -t pneumonia-detection:dgx .

# Verify build
docker images | grep pneumonia-detection
```

#### 2. Multi-stage Build (Production)

```dockerfile
# Dockerfile.prod
FROM nvcr.io/nvidia/pytorch:23.10-py3 AS builder

# Install dependencies
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

FROM nvcr.io/nvidia/pytorch:23.10-py3

# Copy installed packages
COPY --from=builder /opt/conda /opt/conda

# Copy application
WORKDIR /workspace/pneumonia-detection
COPY src/ ./src/
COPY configs/ ./configs/
COPY scripts/ ./scripts/

# Set environment variables
ENV PYTHONPATH=/workspace/pneumonia-detection
ENV CUDA_VISIBLE_DEVICES=0,1,2,3

CMD ["python", "-m", "src.training.distributed_trainer", "--config", "configs/dgx_train_config.yaml"]
```

### Running Containers

#### 1. Training Container

```bash
# Run training container
docker run --gpus all \
    --ipc=host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -v $(pwd)/data:/workspace/data \
    -v $(pwd)/outputs:/workspace/outputs \
    -v $(pwd)/configs:/workspace/configs \
    pneumonia-detection:dgx \
    python -m src.training.distributed_trainer \
    --config /workspace/configs/dgx_train_config.yaml \
    --world-size 4
```

#### 2. Evaluation Container

```bash
# Run evaluation
docker run --gpus all \
    -v $(pwd)/data:/workspace/data \
    -v $(pwd)/outputs:/workspace/outputs \
    pneumonia-detection:dgx \
    python examples/evaluate_model.py \
    --model_path /workspace/outputs/model.pth \
    --data_dir /workspace/data
```

### Docker Compose (Advanced)

```yaml
# docker-compose.yml
version: "3.8"

services:
  training:
    build: .
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - CUDA_VISIBLE_DEVICES=0,1,2,3
      - NCCL_DEBUG=INFO
    volumes:
      - ./data:/workspace/data
      - ./outputs:/workspace/outputs
      - ./configs:/workspace/configs
    ipc: host
    ulimits:
      memlock:
        soft: -1
        hard: -1
      stack: 67108864
    command: >
      python -m src.training.distributed_trainer
      --config /workspace/configs/dgx_train_config.yaml
      --world-size 4

  tensorboard:
    image: tensorflow/tensorflow:latest-gpu
    ports:
      - "6006:6006"
    volumes:
      - ./outputs:/logs
    command: tensorboard --logdir=/logs --host=0.0.0.0
```

## Production Deployment

### Model Serving

#### 1. REST API Server

```python
# src/inference/server.py
from flask import Flask, request, jsonify
import torch
from src.models import create_model
from src.data import get_medical_transforms

app = Flask(__name__)

# Load model
model = create_model('xception', num_classes=2)
model.load_state_dict(torch.load('models/best_model.pth'))
model.eval()

@app.route('/predict', methods=['POST'])
def predict():
    # Process image and return prediction
    # Implementation details...
    pass

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
```

#### 2. Production Container

```dockerfile
# Dockerfile.serve
FROM nvcr.io/nvidia/pytorch:23.10-py3

WORKDIR /app

# Install production dependencies
COPY requirements-prod.txt .
RUN pip install --no-cache-dir -r requirements-prod.txt

# Copy application
COPY src/ ./src/
COPY models/ ./models/

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

EXPOSE 8080

CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "4", "src.inference.server:app"]
```

### Load Balancing

```yaml
# kubernetes/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: pneumonia-detection
spec:
  replicas: 3
  selector:
    matchLabels:
      app: pneumonia-detection
  template:
    metadata:
      labels:
        app: pneumonia-detection
    spec:
      containers:
        - name: pneumonia-detection
          image: pneumonia-detection:serve
          ports:
            - containerPort: 8080
          resources:
            limits:
              nvidia.com/gpu: 1
            requests:
              memory: "4Gi"
              cpu: "2"
```

## Monitoring and Maintenance

### System Monitoring

#### 1. GPU Monitoring

```bash
# Install nvidia-ml-py
pip install nvidia-ml-py3

# Monitor GPU usage
python scripts/monitor_gpus.py --interval 5
```

#### 2. Training Monitoring

```python
# Custom monitoring script
import wandb
from src.training import ModelTrainer

# Initialize monitoring
wandb.init(project="pneumonia-detection-dgx")

# Monitor training metrics
trainer = ModelTrainer(model, criterion, optimizer, log_wandb=True)
```

### Performance Benchmarking

```bash
# Benchmark script
python scripts/benchmark_dgx.py \
    --models xception vgg mobilenet \
    --batch_sizes 16 32 64 \
    --measure_throughput \
    --measure_memory
```

### Backup and Recovery

```bash
# Automated backup script
#!/bin/bash
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/backups/pneumonia-detection"

# Create backup
mkdir -p "$BACKUP_DIR/$DATE"
cp -r outputs/ "$BACKUP_DIR/$DATE/"
cp -r configs/ "$BACKUP_DIR/$DATE/"

# Compress
tar -czf "$BACKUP_DIR/backup_$DATE.tar.gz" -C "$BACKUP_DIR" "$DATE"
rm -rf "$BACKUP_DIR/$DATE"

# Cleanup old backups (keep last 7 days)
find "$BACKUP_DIR" -name "backup_*.tar.gz" -mtime +7 -delete
```

## Troubleshooting

### Common Issues

#### 1. Out of Memory Errors

```yaml
# Reduce batch size
training:
  batch_size: 16          # From 32
  gradient_accumulation_steps: 2  # Maintain effective batch size

# Enable memory optimization
training:
  use_amp: true
  max_grad_norm: 1.0
```

#### 2. Slow Training

```bash
# Check data loading bottleneck
python scripts/profile_data_loading.py

# Optimize data loading
data:
  num_workers: 8
  pin_memory: true
  prefetch_factor: 2
```

#### 3. NCCL Issues

```bash
# Set NCCL environment variables
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
```

### Performance Issues

#### 1. Low GPU Utilization

```bash
# Monitor GPU utilization
nvidia-smi dmon -i 0,1,2,3

# Solutions:
# - Increase batch size
# - Reduce num_workers if CPU bound
# - Enable pin_memory
# - Use AMP for memory efficiency
```

#### 2. Communication Bottlenecks

```bash
# Test inter-GPU bandwidth
nvidia-smi topo -m

# Optimize NCCL settings
export NCCL_TREE_THRESHOLD=0
export NCCL_ALGO=Ring
```

## Security Considerations

### Data Protection

```bash
# Encrypt data at rest
sudo cryptsetup luksFormat /dev/nvme0n1p1
sudo cryptsetup luksOpen /dev/nvme0n1p1 encrypted_data

# Mount encrypted partition
sudo mount /dev/mapper/encrypted_data /data
```

### Container Security

```dockerfile
# Use non-root user
RUN useradd -m -u 1000 appuser
USER appuser

# Limit container capabilities
docker run --cap-drop=ALL --cap-add=SYS_NICE pneumonia-detection:dgx
```

### Network Security

```yaml
# Network policies (Kubernetes)
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: pneumonia-detection-netpol
spec:
  podSelector:
    matchLabels:
      app: pneumonia-detection
  policyTypes:
    - Ingress
  ingress:
    - from:
        - podSelector:
            matchLabels:
              role: api-gateway
      ports:
        - protocol: TCP
          port: 8080
```

## Scaling Strategies

### Horizontal Scaling

```yaml
# Multiple DGX stations
apiVersion: v1
kind: ConfigMap
metadata:
  name: distributed-config
data:
  world_size: "16" # 4 nodes × 4 GPUs
  master_addr: "dgx-node-0"
  master_port: "12355"
```

### Vertical Scaling

```yaml
# Resource allocation per GPU
resources:
  limits:
    nvidia.com/gpu: 1
    memory: "32Gi" # Match GPU memory
    cpu: "10" # 40 cores / 4 GPUs
  requests:
    memory: "16Gi"
    cpu: "5"
```
