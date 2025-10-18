# SGLang Build & Deployment Runbook

**Last Verified:** 2025-10-01  
**Applicable Branch:** main (commit bfa27438)  
**Minimum Version:** 0.5.3rc0  
**Docker Image:** `lmsysorg/sglang:latest` (CUDA 12.9.1)

---

## Table of Contents
1. [Environment Setup](#environment-setup)
2. [Installation Methods](#installation-methods)
3. [Build from Source](#build-from-source)
4. [Docker Deployment](#docker-deployment)
5. [Launch Parameters Reference](#launch-parameters-reference)
6. [Scenario-Based Recipes](#scenario-based-recipes)
7. [Operations & Maintenance](#operations--maintenance)

---

## Environment Setup

### System Requirements

| Component | Minimum | Recommended | Notes |
|-----------|---------|-------------|-------|
| **OS** | Ubuntu 22.04 | Ubuntu 22.04 | Linux kernel 5.15+ |
| **Python** | 3.10 | 3.12 | Python 3.13 not yet supported |
| **CUDA** | 12.6.1 | 12.8.1 - 12.9.1 | Driver >= 550 |
| **GPU** | A100 (40GB) | H100/H200 (80GB+) | Min 24GB VRAM |
| **RAM** | 64GB | 128GB+ | For model loading |
| **Storage** | 100GB | 500GB+ SSD | Model weights + cache |

### Driver & CUDA Versions

```bash
# Check NVIDIA driver
nvidia-smi

# Required driver versions:
# - CUDA 12.6.1: >= 550.54.15
# - CUDA 12.8.1: >= 550.90.07
# - CUDA 12.9.1: >= 555.42.02

# Check CUDA toolkit
nvcc --version

# Recommended: Use CUDA 12.8.1 or 12.9.1
```

### NCCL Version

```bash
# Install recommended NCCL version
pip install nvidia-nccl-cu12==2.27.6 --force-reinstall --no-deps

# Verify NCCL
python -c "import torch; print(torch.cuda.nccl.version())"
# Expected: (2, 27, 6)
```

---

## Installation Methods

### Method 1: pip Install (Recommended for Production)

```bash
# Install SGLang with CUDA 12.8+
pip install "sglang[all]"

# Download FlashInfer cubins (required)
FLASHINFER_LOGGING_LEVEL=warning python -m flashinfer --download-cubin

# Verify installation
python -m sglang.launch_server --help
```

### Method 2: Docker (Recommended for Isolated Environments)

```bash
# Pull official image (CUDA 12.9.1)
docker pull lmsysorg/sglang:latest

# Run container with GPU support
docker run --gpus all -p 30000:30000 --shm-size 16g \
  lmsysorg/sglang:latest \
  python -m sglang.launch_server \
    --model-path meta-llama/Llama-3.1-8B-Instruct \
    --host 0.0.0.0 --port 30000
```

### Method 3: Build from Source (Development)

```bash
# Clone repository
git clone https://github.com/sgl-project/sglang.git
cd sglang

# Install dependencies
pip install --upgrade pip setuptools wheel
pip install -e "python[all]"

# Download FlashInfer cubins
FLASHINFER_LOGGING_LEVEL=warning python -m flashinfer --download-cubin

# Verify
python -m sglang.launch_server --version
```

---

## Build from Source

### Standard Build

```bash
git clone https://github.com/sgl-project/sglang.git
cd sglang

# Install Python dependencies
cd python
pip install -e ".[all]"

# Install sgl-kernel (custom CUDA kernels)
pip install sgl-kernel==0.3.13

# Download FlashInfer cubins
FLASHINFER_LOGGING_LEVEL=warning python -m flashinfer --download-cubin
```

### Build with DeepEP (for MoE Models)

```bash
# Prerequisites: See scripts/ci/ci_install_deepep.sh

# 1. Install system dependencies
apt install -y libibverbs-dev rdma-core infiniband-diags \
  libfabric-dev build-essential cmake

# 2. Install GDRCopy v2.4.4
cd /tmp
git clone https://github.com/NVIDIA/gdrcopy.git -b v2.4.4
cd gdrcopy/packages
CUDA=/usr/local/cuda ./build-deb-packages.sh
dpkg -i gdrdrv-dkms_*.deb libgdrapi_*.deb gdrcopy-tests_*.deb gdrcopy_*.deb
rm -rf /tmp/gdrcopy

# 3. Install NVSHMEM 3.3.9
export CUDA_HOME=/usr/local/cuda
export GDRCOPY_HOME=/usr/src/gdrdrv-2.4.4/
export NVSHMEM_DIR=/opt/nvshmem/install

cd /tmp
wget https://developer.download.nvidia.com/compute/redist/nvshmem/3.3.9/source/nvshmem_src_cuda12-all-all-3.3.9.tar.gz
tar -xf nvshmem_src_cuda12-all-all-3.3.9.tar.gz
cd nvshmem_src

NVSHMEM_SHMEM_SUPPORT=0 \
NVSHMEM_UCX_SUPPORT=0 \
NVSHMEM_USE_NCCL=0 \
NVSHMEM_MPI_SUPPORT=0 \
NVSHMEM_IBGDA_SUPPORT=1 \
NVSHMEM_PMIX_SUPPORT=0 \
NVSHMEM_TIMEOUT_DEVICE_POLLING=0 \
NVSHMEM_USE_GDRCOPY=1 \
cmake -S . -B build/ -DCMAKE_INSTALL_PREFIX=/opt/nvshmem/install -DCMAKE_CUDA_ARCHITECTURES=90
cd build && make -j$(nproc) install

export LD_LIBRARY_PATH="${NVSHMEM_DIR}/lib:$LD_LIBRARY_PATH"
export PATH="${NVSHMEM_DIR}/bin:$PATH"

# 4. Install DeepEP
git clone https://github.com/deepseek-ai/DeepEP.git /tmp/deepep
cd /tmp/deepep
git checkout 9af0e0d0e74f3577af1979c9b9e1ac2cad0104ee
python setup.py install

# 5. Verify
gdrcopy_copybw  # Should show bandwidth stats
nvshmem-info -a  # Should show NVSHMEM config
python -c "import deep_ep; print(deep_ep.__version__)"
```

### Build with Mooncake (for PD Disaggregation)

```bash
# Install Mooncake transfer engine
uv pip install mooncake-transfer-engine

# Or use pip
pip install mooncake-transfer-engine

# Verify
python -c "import mooncake; print(mooncake.__version__)"
```

---

## Docker Deployment

### Official Dockerfiles

| Dockerfile | Platform | Base Image |
|------------|----------|------------|
| `docker/Dockerfile` | NVIDIA CUDA | nvidia/cuda:12.9.1-cudnn-devel-ubuntu22.04 |
| `docker/Dockerfile.rocm` | AMD ROCm | rocm/dev-ubuntu-22.04 |
| `docker/Dockerfile.npu` | Ascend NPU | Custom |
| `docker/Dockerfile.b300` | Blackwell (GB200) | Custom ARM64 |

### Build Docker Image

```bash
cd sglang

# Build with CUDA 12.9.1
docker build -t sglang:local -f docker/Dockerfile \
  --build-arg CUDA_VERSION=12.9.1 \
  --build-arg BUILD_TYPE=all \
  .

# Build with DeepEP support (for MoE)
docker build -t sglang:deepep -f docker/Dockerfile \
  --build-arg CUDA_VERSION=12.9.1 \
  --build-arg BUILD_TYPE=all \
  --build-arg DEEPEP_COMMIT=9af0e0d0e74f3577af1979c9b9e1ac2cad0104ee \
  .
```

### Run with Docker Compose

```yaml
# docker-compose.yml
version: '3.8'
services:
  sglang:
    image: lmsysorg/sglang:latest
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    shm_size: 16gb
    ports:
      - "30000:30000"
    volumes:
      - ~/.cache/huggingface:/root/.cache/huggingface
    command: >
      python -m sglang.launch_server
      --model-path meta-llama/Llama-3.1-8B-Instruct
      --host 0.0.0.0 --port 30000
      --mem-fraction-static 0.85
```

```bash
docker compose up -d
```

---

## Launch Parameters Reference

### Must-Configure Parameters

#### Prefill Node (PD Disaggregation)

```bash
python -m sglang.launch_server \
  --model-path <MODEL> \
  --disaggregation-mode prefill \
  --disaggregation-ib-device <IB_DEVICE> \
  --disaggregation-bootstrap-port <PORT> \
  --host <LOCAL_IP> \
  --port 30000 \
  --dist-init-addr <MASTER_IP:PORT> \
  --nnodes <N> \
  --node-rank <RANK> \
  --tp-size <TP> \
  --dp-size <DP>
```

#### Decode Node (PD Disaggregation)

```bash
python -m sglang.launch_server \
  --model-path <MODEL> \
  --disaggregation-mode decode \
  --disaggregation-ib-device <IB_DEVICE> \
  --host <LOCAL_IP> \
  --port 30001 \
  --dist-init-addr <MASTER_IP:PORT> \
  --nnodes <N> \
  --node-rank <RANK> \
  --tp-size <TP> \
  --dp-size <DP> \
  --max-running-requests <MAX_REQS> \
  --base-gpu-id <BASE_ID>
```

### Recommended Defaults

| Parameter | Default | Prefill | Decode | Notes |
|-----------|---------|---------|--------|-------|
| `--page-size` | 1 | 1 | 1 | Token-level paging |
| `--chunked-prefill-size` | Auto | 8192 | N/A | Reduce for OOM |
| `--mem-fraction-static` | 0.9 | 0.8 | 0.85 | Model + KV pool |
| `--max-running-requests` | Auto | - | 128-512 | Critical for decode |
| `--max-total-tokens` | Auto | - | - | Usually auto-calculated |
| `--context-length` | From config | Override if needed | - | Max sequence length |
| `--decode-log-interval` | 40 | - | 40 | Log every N steps |
| `--max-prefill-tokens` | 16384 | 16384-32768 | - | Max tokens in prefill batch |

### Performance Tuning Flags

| Flag | When to Use | Impact |
|------|-------------|--------|
| `--enable-two-batch-overlap` | Experimental | May reduce TTFT, risk: instability |
| `--disable-radix-cache` | Debugging, determinism | No prefix caching, higher memory |
| `--enable-torch-compile` | Small models, low batch | JIT speedup, cold start penalty |
| `--enable-dp-attention` | DeepSeek/Qwen MoE | DP for attention, TP for FFN |
| `--enable-dp-lm-head` | With DP attention | Avoid all-gather across DP |
| `--moe-dense-tp-size` | MoE + high TP | Prevent GEMM dimension errors |

---

## Scenario-Based Recipes

### 1. Single-Machine H200×8 (TP=8)

```bash
python -m sglang.launch_server \
  --model-path meta-llama/Llama-3.1-405B-Instruct \
  --tp-size 8 \
  --mem-fraction-static 0.88 \
  --max-prefill-tokens 16384 \
  --host 0.0.0.0 --port 30000
```

**Expected Performance** (H200 80GB):
- TTFT: 50-200ms (input dependent)
- Decode: 40-60 tok/s per request
- Throughput: 2000+ req/s @ small outputs

### 2. 1P1D (Single Node, Educational)

```bash
# Find active IB device
for device in mlx5_{0..11}; do
  if ibv_devinfo $device 2>/dev/null | grep -q "PORT_ACTIVE"; then
    echo "Active device: $device"
    break
  fi
done

# Prefill on GPU 0
CUDA_VISIBLE_DEVICES=0 python -m sglang.launch_server \
  --model-path meta-llama/Llama-3.1-8B-Instruct \
  --disaggregation-mode prefill \
  --disaggregation-ib-device mlx5_0 \
  --disaggregation-bootstrap-port 9001 \
  --host 127.0.0.1 --port 30000 &

# Decode on GPU 1
CUDA_VISIBLE_DEVICES=1 python -m sglang.launch_server \
  --model-path meta-llama/Llama-3.1-8B-Instruct \
  --disaggregation-mode decode \
  --disaggregation-ib-device mlx5_0 \
  --host 127.0.0.1 --port 30001 \
  --base-gpu-id 1 &

# Wait for servers to start
sleep 60

# Router
python -m sglang_router.launch_router \
  --pd-disaggregation \
  --prefill http://127.0.0.1:30000 \
  --decode http://127.0.0.1:30001 \
  --host 0.0.0.0 --port 8000
```

**Health Check**:
```bash
curl http://127.0.0.1:30000/health  # Prefill
curl http://127.0.0.1:30001/health  # Decode
curl http://127.0.0.1:8000/health   # Router
```

### 3. 2P2D (2 Prefill Nodes + 2 Decode Nodes)

**Prerequisites**:
- 4 nodes with IB connectivity
- Same model path accessible on all nodes
- Firewall allows ports 5000, 30000-30001, 9001-9002

```bash
# Node 0 (Prefill Master)
export PREFILL_MASTER_IP=10.0.0.1
export DECODE_MASTER_IP=10.0.0.3
export IB_DEVICE=mlx5_0

python -m sglang.launch_server \
  --model-path meta-llama/Llama-3.1-70B-Instruct \
  --disaggregation-mode prefill \
  --disaggregation-ib-device $IB_DEVICE \
  --disaggregation-bootstrap-port 9001 \
  --host $PREFILL_MASTER_IP --port 30000 \
  --dist-init-addr $PREFILL_MASTER_IP:5000 \
  --nnodes 2 --node-rank 0 \
  --tp-size 8 \
  --mem-fraction-static 0.8

# Node 1 (Prefill Worker)
python -m sglang.launch_server \
  --model-path meta-llama/Llama-3.1-70B-Instruct \
  --disaggregation-mode prefill \
  --disaggregation-ib-device $IB_DEVICE \
  --disaggregation-bootstrap-port 9002 \
  --host 10.0.0.2 --port 30000 \
  --dist-init-addr $PREFILL_MASTER_IP:5000 \
  --nnodes 2 --node-rank 1 \
  --tp-size 8 \
  --mem-fraction-static 0.8

# Node 2 (Decode Master)
python -m sglang.launch_server \
  --model-path meta-llama/Llama-3.1-70B-Instruct \
  --disaggregation-mode decode \
  --disaggregation-ib-device $IB_DEVICE \
  --host $DECODE_MASTER_IP --port 30001 \
  --dist-init-addr $DECODE_MASTER_IP:5000 \
  --nnodes 2 --node-rank 0 \
  --tp-size 8 \
  --max-running-requests 256 \
  --mem-fraction-static 0.85

# Node 3 (Decode Worker)
python -m sglang.launch_server \
  --model-path meta-llama/Llama-3.1-70B-Instruct \
  --disaggregation-mode decode \
  --disaggregation-ib-device $IB_DEVICE \
  --host 10.0.0.4 --port 30001 \
  --dist-init-addr $DECODE_MASTER_IP:5000 \
  --nnodes 2 --node-rank 1 \
  --tp-size 8 \
  --max-running-requests 256 \
  --mem-fraction-static 0.85

# Router (any accessible node)
python -m sglang_router.launch_router \
  --pd-disaggregation \
  --prefill http://10.0.0.1:30000 http://10.0.0.2:30000 \
  --decode http://10.0.0.3:30001 http://10.0.0.4:30001 \
  --host 0.0.0.0 --port 8000
```

### 4. 4P-9D (DeepSeek-V3, Large Scale)

**Hardware**: 13 nodes × 8 H100/H200 GPUs

```bash
# Prefill Nodes (0-3): TP=16, DP=8, 2 nodes each
for RANK in 0 1 2 3; do
  ssh node-prefill-$RANK "
    python -m sglang.launch_server \
      --model-path deepseek-ai/DeepSeek-V3-0324 \
      --disaggregation-mode prefill \
      --disaggregation-ib-device mlx5_0 \
      --disaggregation-bootstrap-port $((9001 + RANK)) \
      --host \$(hostname -I | awk '{print \$1}') --port 30000 \
      --trust-remote-code \
      --dist-init-addr prefill-master:5000 \
      --nnodes 4 --node-rank $RANK \
      --tp-size 16 --dp-size 8 \
      --enable-dp-attention \
      --moe-a2a-backend deepep \
      --mem-fraction-static 0.8 \
      --max-prefill-tokens 32768
  "
done

# Decode Nodes (0-8): TP=16, DP=8
for RANK in 0 1 2 3 4 5 6 7 8; do
  ssh node-decode-$RANK "
    python -m sglang.launch_server \
      --model-path deepseek-ai/DeepSeek-V3-0324 \
      --disaggregation-mode decode \
      --disaggregation-ib-device mlx5_0 \
      --host \$(hostname -I | awk '{print \$1}') --port 30001 \
      --trust-remote-code \
      --dist-init-addr decode-master:5000 \
      --nnodes 9 --node-rank $RANK \
      --tp-size 16 --dp-size 8 \
      --enable-dp-attention \
      --moe-a2a-backend deepep \
      --mem-fraction-static 0.8 \
      --max-running-requests 512
  "
done

# Router: aggregate all endpoints
python -m sglang_router.launch_router \
  --pd-disaggregation \
  --prefill $(for i in {0..3}; do echo "http://node-prefill-$i:30000"; done) \
  --decode $(for i in {0..8}; do echo "http://node-decode-$i:30001"; done) \
  --host 0.0.0.0 --port 8000 \
  --load-balance-method minimum_tokens
```

**Expected Throughput** (DeepSeek-V3, H100):
- Input: 2048 tokens, Output: 256 tokens
- Throughput: 15,000+ req/s
- TTFT: 100-300ms (P50)
- Decode latency: 20-40ms/token

---

## Operations & Maintenance

### Service Management

#### Start Server (systemd example)

```ini
# /etc/systemd/system/sglang.service
[Unit]
Description=SGLang Inference Server
After=network.target

[Service]
Type=simple
User=sglang
WorkingDirectory=/opt/sglang
Environment="CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7"
ExecStart=/usr/bin/python3 -m sglang.launch_server \
  --model-path /models/Llama-3.1-405B-Instruct \
  --tp-size 8 \
  --host 0.0.0.0 --port 30000
Restart=on-failure
RestartSec=10s

[Install]
WantedBy=multi-user.target
```

```bash
systemctl daemon-reload
systemctl enable sglang
systemctl start sglang
systemctl status sglang
```

#### Health Check Script

```bash
#!/bin/bash
# health_check.sh

ENDPOINTS=(
  "http://127.0.0.1:30000/health"  # Prefill
  "http://127.0.0.1:30001/health"  # Decode
  "http://127.0.0.1:8000/health"   # Router
)

for endpoint in "${ENDPOINTS[@]}"; do
  if curl -sf "$endpoint" > /dev/null; then
    echo "✓ $endpoint"
  else
    echo "✗ $endpoint"
    exit 1
  fi
done
```

### Hot Reload / Graceful Restart

SGLang does not support hot reload. For zero-downtime updates:

1. Launch new instance on different port
2. Health check new instance
3. Update router/load balancer
4. Drain and stop old instance

### Port & Firewall Configuration

```bash
# Allow SGLang server
ufw allow 30000/tcp
ufw allow 30001/tcp

# Allow router
ufw allow 8000/tcp

# Allow distributed initialization
ufw allow 5000/tcp

# Allow IB/RDMA (typically handled at kernel level)
# Ensure RDMA ports are not blocked
```

### IB Connectivity Verification

```bash
# List IB devices
ibv_devices

# Device info
ibv_devinfo mlx5_0

# Check port state (must be ACTIVE)
ibv_devinfo mlx5_0 | grep state

# Test bandwidth (2 nodes required)
# Node 1:
ib_write_bw -d mlx5_0 -F

# Node 2:
ib_write_bw -d mlx5_0 -F <node1_ip>
```

### Common Commands

```bash
# Kill all SGLang processes
pkill -f "python.*sglang.launch_server"

# Check GPU utilization
nvidia-smi dmon -s u -c 1

# Monitor GPU memory
watch -n 1 nvidia-smi

# Check NCCL bandwidth (requires sglang running)
# Enable debug: export NCCL_DEBUG=INFO

# View server logs
journalctl -u sglang -f
```

---

## Failure Recovery

### Out-of-Memory (OOM)

```bash
# Reduce KV cache pool
--mem-fraction-static 0.7

# Enable chunked prefill
--chunked-prefill-size 4096

# Reduce max running requests (decode)
--max-running-requests 64
```

### NCCL Timeout

```bash
# Increase timeout
--dist-timeout 3600

# Disable CUDA graphs (temporary)
--disable-cuda-graph

# Check network connectivity
ping <other_node>
```

### IB Device Not Found

```bash
# Verify IB kernel module
lsmod | grep mlx

# Restart IB services
systemctl restart openibd

# Check device permissions
ls -l /dev/infiniband/
```

---

**End of BUILD_RUNBOOK memory**