# SGLang Hardware Topology Guide

**Last Verified:** 2025-10-01  
**Applicable Branch:** main (commit bfa27438)

---

## Table of Contents
1. [GPU Specifications](#gpu-specifications)
2. [Node Configurations](#node-configurations)
3. [Network Topology](#network-topology)
4. [NUMA & Affinity](#numa--affinity)
5. [Capacity Planning](#capacity-planning)

---

## GPU Specifications

### NVIDIA H200

| Spec | Value | Notes |
|------|-------|-------|
| **Memory** | 141 GB HBM3e | Effective ~80 GB for models |
| **Compute** | 67 TFLOPS (FP16) | 134 TFLOPS (FP8) |
| **Memory BW** | 4.8 TB/s | 2× H100 |
| **NVLink** | 900 GB/s per GPU | 7.2 TB/s aggregate (8 GPUs) |
| **TDP** | 700W | Thermal design power |
| **Architecture** | Hopper (sm_90) | CUDA Compute Capability 9.0 |

**Best For**: Large models (70B-405B), long context, high throughput

---

### NVIDIA H100

| Spec | Value | Notes |
|------|-------|-------|
| **Memory** | 80 GB HBM3 | SXM5 variant |
| **Compute** | 67 TFLOPS (FP16) | 134 TFLOPS (FP8) |
| **Memory BW** | 3.35 TB/s | |
| **NVLink** | 900 GB/s per GPU | 7.2 TB/s aggregate (8 GPUs) |
| **TDP** | 700W | SXM5 |
| **Architecture** | Hopper (sm_90) | CUDA Compute Capability 9.0 |

**Best For**: All workloads, good memory/compute balance

---

### NVIDIA A100

| Spec | Value | Notes |
|------|-------|-------|
| **Memory** | 80 GB HBM2e | 40GB variant also common |
| **Compute** | 312 TFLOPS (FP16 Tensor) | 624 TFLOPS (sparsity) |
| **Memory BW** | 2.0 TB/s | |
| **NVLink** | 600 GB/s per GPU | 4.8 TB/s aggregate (8 GPUs) |
| **TDP** | 400W | SXM4 |
| **Architecture** | Ampere (sm_80) | CUDA Compute Capability 8.0 |

**Best For**: Cost-effective, mature ecosystem, TP≤4

---

## Node Configurations

### DGX H100 / H200

**Configuration**:
- 8× H100/H200 80GB SXM5
- NVLink fully connected (all-to-all)
- 2× AMD EPYC 7742 (128 cores)
- 2TB RAM
- 8× 3.84TB NVMe SSD (RAID)
- 8× ConnectX-7 IB NDR (400 Gb/s)

**Topology**:
```
GPU0 ↔ GPU1 ↔ GPU2 ↔ GPU3
 ↕       ↕       ↕       ↕
GPU4 ↔ GPU5 ↔ GPU6 ↔ GPU7
```
All pairs connected via NVLink (900 GB/s bidirectional)

---

### Standard 8× H100 Node

**Configuration**:
- 8× H100 80GB PCIe or SXM
- 2× Intel Xeon or AMD EPYC
- 512GB - 1TB RAM
- 4× 1.92TB NVMe SSD
- 2× ConnectX-6 IB HDR (200 Gb/s)

**PCIe Topology** (example):
```
CPU0 ────┬──── GPU0, GPU1, GPU2, GPU3
         └──── NIC0

CPU1 ────┬──── GPU4, GPU5, GPU6, GPU7
         └──── NIC1
```

**NVSwitch Topology** (SXM):
```
All GPUs connected via NVSwitch (900 GB/s per link)
```

---

### GB200 NVL72 (Blackwell)

**Configuration**:
- 36× GB200 Superchips (72 GPUs total)
- Each Superchip: 2× B200 GPUs + 1× Grace CPU
- 1.44 TB HBM3e per Superchip
- NVLink Switch System (5th gen)

**Topology**: Fully connected via NVLink-Switch (1.8 TB/s per GPU)

**SGLang Support**: Experimental (Docker images for ARM64)

---

## Network Topology

### InfiniBand (Recommended for PD)

**HDR (200 Gb/s)**:
- Effective: ~23 GB/s per port
- Latency: <1 μs
- Use: PD disaggregation, small-scale (<16 nodes)

**NDR (400 Gb/s)**:
- Effective: ~47 GB/s per port
- Latency: <1 μs
- Use: Large-scale PD (>16 nodes)

**Topology Patterns**:

**Fat-Tree** (most common):
```
      Core Switches
         /    \
    Leaf Switches
    /  /  \  \  \
  Nodes (compute)
```

**Rail-Optimized** (for PD):
```
Prefill Rail  ↔  Core  ↔  Decode Rail
    ||||                      ||||
  P-Nodes                   D-Nodes
```

---

### Ethernet (Alternative)

**100 GbE**:
- Effective: ~11 GB/s per port
- Latency: ~10 μs
- Use: Small-scale, non-PD

**400 GbE**:
- Effective: ~45 GB/s per port
- Use: Can support PD with RDMA (RoCE)

---

## NUMA & Affinity

### NUMA Topology (Dual-Socket)

```
NUMA Node 0: CPU0, GPUs 0-3, RAM 0-511GB
NUMA Node 1: CPU1, GPUs 4-7, RAM 512-1023GB
```

**Best Practice**: Bind process to NUMA node matching GPU

---

### CPU Affinity for GPU Workers

```bash
# Bind to NUMA node 0 (GPUs 0-3)
numactl --cpunodebind=0 --membind=0 \
  CUDA_VISIBLE_DEVICES=0,1,2,3 \
  python -m sglang.launch_server --tp-size 4 ...

# Bind to NUMA node 1 (GPUs 4-7)
numactl --cpunodebind=1 --membind=1 \
  CUDA_VISIBLE_DEVICES=4,5,6,7 \
  python -m sglang.launch_server --tp-size 4 --base-gpu-id 4 ...
```

**Impact**: 5-10% throughput improvement from reduced cross-NUMA traffic

---

### Hugepages (Optional)

```bash
# Enable hugepages (2MB)
echo 8192 > /proc/sys/vm/nr_hugepages

# Mount hugetlbfs
mount -t hugetlbfs none /mnt/huge

# Use with SGLang (automatic if available)
```

**Impact**: Minor (<2%) improvement in memory-intensive workloads

---

## Capacity Planning

### Model Memory Requirements

**Formula**: `model_size_GB × 1.2 + context_length × batch_size × 2 × hidden_dim × num_layers × 2 / 1e9`

**Approximation**:

| Model | Params | FP16 Size | FP8 Size | TP=1 | TP=4 | TP=8 |
|-------|--------|-----------|----------|------|------|------|
| Llama-3.1-8B | 8B | 16 GB | 8 GB | 1×80GB ✓ | - | - |
| Llama-3.1-70B | 70B | 140 GB | 70 GB | 2×80GB ✓ | 1×80GB/GPU ✓ | - |
| Llama-3.1-405B | 405B | 810 GB | 405 GB | - | 3×80GB/GPU | 2×80GB/GPU |
| DeepSeek-V3 | 685B | 1370 GB | 685 GB | - | - | 2-3×80GB/GPU |

---

### KV Cache Memory

**Formula**: `batch_size × seq_length × 2 (K+V) × num_layers × hidden_dim × 2 bytes (FP16) / 1e9`

**Example** (Llama-3.1-70B, 128 requests, 8K tokens avg):
- KV cache: 128 × 8192 × 2 × 80 × 8192 × 2 / 1e9 ≈ **210 GB**

**Planning**: Leave 50-70% of GPU memory for KV cache after model weights

---

### Throughput Estimates

**Decode Throughput** (output tokens/s per GPU):

| GPU | Model Size | TP=1 | TP=4 | TP=8 |
|-----|------------|------|------|------|
| H200 | 8B | 10K-12K | - | - |
| H200 | 70B | 1.5K-2K | 6K-8K | - |
| H200 | 405B | - | 1K-1.5K | 3K-4K |
| H100 | 8B | 8K-10K | - | - |
| H100 | 70B | 1.2K-1.6K | 5K-6K | - |

**PD Disaggregation**: 1.5-2× improvement over unified (for decode-heavy workloads)

---

### Node Count Recommendations

| Deployment | Model | Prefill Nodes | Decode Nodes | Total GPUs |
|------------|-------|---------------|--------------|------------|
| **Small** | 70B | 1 (TP=8) | 1 (TP=8) | 16 |
| **Medium** | 70B | 2 (TP=16) | 2 (TP=16) | 32 |
| **Large** | DeepSeek-V3 | 4 (TP=16) | 9 (TP=16) | 104 |
| **X-Large** | DeepSeek-V3 | 8 (TP=16) | 16 (TP=16) | 192 |

---

### Power & Cooling

**DGX H100 Node**:
- Power: 10.2 kW (8×700W GPUs + system)
- Cooling: Liquid or high-CFM air

**Rack Planning** (42U rack):
- Max 4× DGX H100 nodes per rack (42 kW)
- Requires 3-phase 208V power, liquid cooling

---

## Verification Commands

### GPU Topology

```bash
# NVLink topology
nvidia-smi topo -m

# Expected output (DGX H100):
#      GPU0 GPU1 GPU2 GPU3 GPU4 GPU5 GPU6 GPU7
# GPU0  X   NV18 NV18 NV18 NV18 NV18 NV18 NV18
# GPU1 NV18  X   NV18 NV18 NV18 NV18 NV18 NV18
# ...
```

---

### IB Topology

```bash
# List IB devices
ibv_devices

# Device info
ibv_devinfo mlx5_0

# Bandwidth test (2 nodes)
# Node 1:
ib_write_bw -d mlx5_0

# Node 2:
ib_write_bw -d mlx5_0 <node1_ip>

# Expected: 180+ Gbps (HDR), 350+ Gbps (NDR)
```

---

### NUMA Topology

```bash
# Show NUMA nodes
numactl --hardware

# CPU-GPU affinity
nvidia-smi topo -m | grep CPU
```

---

**End of HARDWARE_TOPOLOGY memory**