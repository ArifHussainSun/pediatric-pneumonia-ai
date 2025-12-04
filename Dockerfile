# Dockerfile for Pediatric Pneumonia Detection on DGX
# Optimized for Tesla V100 GPUs with CUDA 12.9

FROM nvcr.io/nvidia/pytorch:23.10-py3

# Set working directory
WORKDIR /workspace/pediatric-pneumonia-ai

# Set environment variables for optimal DGX performance
ENV PYTHONPATH=/workspace/pediatric-pneumonia-ai
ENV CUDA_VISIBLE_DEVICES=0,1,2,3
ENV NCCL_DEBUG=INFO
ENV NCCL_TREE_THRESHOLD=0
ENV NCCL_IB_DISABLE=1
ENV NCCL_SOCKET_IFNAME=^lo,docker0

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better Docker layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install additional packages for intelligent preprocessing and API
RUN pip install --no-cache-dir \
    aiohttp \
    aiofiles \
    python-multipart \
    pathlib2

# Copy source code
COPY src/ ./src/
COPY config/ ./config/
COPY scripts/ ./scripts/
COPY web/ ./web/
COPY validation/ ./validation/
COPY debug/ ./debug/

# Make scripts executable
RUN chmod +x scripts/*.sh

# Create directories for outputs, logs, and validation reports
RUN mkdir -p outputs logs tensorboard_logs validation/reports

# Expose API port
EXPOSE 8000

# Set default command to run API server
CMD ["python", "-m", "uvicorn", "src.api.server:app", "--host", "0.0.0.0", "--port", "8000"]