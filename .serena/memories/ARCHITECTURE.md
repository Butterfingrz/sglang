# SGLang Architecture

**Last Verified:** 2025-10-01  
**Applicable Branch:** main (commit bfa27438)  
**Version:** 0.5.3rc0

---

## Table of Contents
1. [Overview](#overview)
2. [Key Components & Data Flow](#key-components--data-flow)
3. [Parallelism & Disaggregation](#parallelism--disaggregation)
4. [Critical Dependencies](#critical-dependencies)
5. [Directory Structure](#directory-structure)

---

## Overview

SGLang is a high-performance serving framework for large language models (LLMs) and vision-language models (VLMs). It achieves industry-leading throughput and latency through:

- **RadixAttention**: Prefix caching with LRU eviction for KV cache reuse
- **Prefill-Decode (PD) Disaggregation**: Separate compute clusters for prefill and decode phases
- **Zero-overhead CPU scheduler**: Overlapped scheduling with GPU execution
- **Continuous batching**: Dynamic request batching with chunked prefill
- **Multi-dimensional parallelism**: TP/DP/EP/PP support
- **Advanced optimizations**: CUDA graphs, torch.compile, FP8/INT4/AWQ quantization

---

## Key Components & Data Flow

### Request Path

```
Client Request
    ↓
HTTP Server (FastAPI/uvicorn)
    ↓
Tokenizer Manager
    ↓
Scheduler (CPU)
    ↓
Model Worker (GPU)
    ↓
RadixCache / KV Cache Pool
    ↓
Detokenizer
    ↓
Response Stream
```

### Core Modules

| Module | Location | Responsibility |
|--------|----------|----------------|
| **HTTP Server** | `python/sglang/srt/http_server.py` | OpenAI-compatible API endpoints (`/generate`, `/v1/completions`, `/v1/chat/completions`) |
| **Tokenizer Manager** | `python/sglang/srt/managers/tokenizer_manager.py` | Tokenization, chat template application, request preprocessing |
| **Scheduler** | `python/sglang/srt/managers/scheduler.py` | Batch formation, memory allocation, RadixCache management, chunked prefill |
| **Model Worker** | `python/sglang/srt/managers/model_worker.py` | GPU execution, forward passes, CUDA graphs |
| **Memory Pool** | `python/sglang/srt/mem_cache/` | Paged KV cache management, hierarchical cache (HiCache) |
| **Attention Backend** | `python/sglang/srt/layers/attention/` | FlashInfer (default), Triton, CUDNN fallback |
| **Router** | `sgl-router/` | Load balancing, PD disaggregation coordination, health checks |

### Prefill Phase

1. **Input Processing**: Full prompt tokens → attention computation
2. **Memory**: Computation-intensive, high FLOPS utilization
3. **KV Cache**: Allocated in memory pool, potentially chunked for large inputs
4. **Output**: First token + populated KV cache

### Decode Phase

1. **Input Processing**: Single new token per request
2. **Memory**: Memory-bandwidth intensive, lower FLOPS
3. **KV Cache**: Read-dominated, grows by 1 token per step
4. **Output**: One token per decode step

### RadixAttention (Prefix Caching)

- **Structure**: Radix tree indexed by token sequences
- **Hit**: Reuse precomputed KV cache for matching prefixes
- **Miss**: Compute and store new KV cache entries
- **Eviction**: LRU-based when memory pressure occurs
- **Disable**: `--disable-radix-cache` for deterministic debugging

---

## Parallelism & Disaggregation

### Tensor Parallelism (TP)

- **Scope**: Shards model weights across GPUs (column/row splits)
- **Communication**: All-reduce after matmul (NCCL)
- **Use Case**: Large models that don't fit on single GPU
- **Flag**: `--tp-size N`

### Data Parallelism (DP)

- **Scope**: Replicates full model across workers
- **Communication**: Independent workers, router distributes requests
- **Use Case**: High throughput when memory allows
- **Flag**: `--dp-size N`

### Expert Parallelism (EP)

- **Scope**: For MoE models, distributes experts across GPUs
- **Communication**: All-to-all for expert routing (DeepEP backend)
- **Use Case**: DeepSeek-V3, Qwen MoE models
- **Flags**: `--ep-size`, `--moe-a2a-backend deepep`, `--moe-dense-tp-size`

### Pipeline Parallelism (PP)

- **Scope**: Layer-wise sharding across GPUs
- **Communication**: P2P transfers between stages
- **Use Case**: Very large models, experimental
- **Flag**: `--pp-size N`

### DP Attention

- **Scope**: Hybrid - DP for attention, TP for FFN/MoE
- **Requirements**: `dp_size == tp_size`
- **Models**: DeepSeek-V2/V3, Qwen MoE
- **Flags**: `--enable-dp-attention`, `--enable-dp-lm-head`

### Prefill-Decode (PD) Disaggregation

**Motivation**: Prefill (compute-bound) and decode (memory-bound) have conflicting scheduling needs. Unified engines suffer from:
- Prefill interrupting decode → higher TTFT variance
- DP attention imbalance (one worker prefills while another decodes)

**Architecture**:
```
                    ┌─────────────┐
                    │   Router    │
                    └──────┬──────┘
                           │
              ┌────────────┴────────────┐
              ↓                         ↓
    ┌─────────────────┐       ┌─────────────────┐
    │ Prefill Cluster │       │ Decode Cluster  │
    │  (4-16 GPUs)    │──KV──→│   (8-32 GPUs)   │
    └─────────────────┘       └─────────────────┘
```

**Transfer Backends**:
- **Mooncake** (default): InfiniBand/RDMA, recommended for production
- **NIXL**: UCX-based alternative
- **ASCEND**: Huawei NPU backend

**Key Parameters**:
- `--disaggregation-mode {prefill|decode}`
- `--disaggregation-ib-device mlx5_X` (IB device name)
- `--disaggregation-bootstrap-port 8998` (coordination port)
- `--disaggregation-transfer-backend {mooncake|nixl|ascend}`

**Network Path**:
1. Prefill completes → KV cache ready
2. RDMA transfer to decode cluster (low latency)
3. Decode begins immediately after KV arrival
4. Router tracks prefill/decode health

---

## Critical Dependencies

### Core Runtime

| Dependency | Version | Purpose |
|------------|---------|---------|
| **PyTorch** | 2.8.0 | Model execution, autograd |
| **FlashInfer** | 0.4.0rc3 | Attention kernels (prefill/decode) |
| **sgl-kernel** | 0.3.13 | Custom CUDA kernels (MoE, quantization) |
| **Transformers** | 4.56.1 | Model loading, tokenization |
| **CUDA** | 12.6.1 - 12.9.1 | GPU runtime |

### PD Disaggregation

| Dependency | Purpose | Installation |
|------------|---------|--------------|
| **Mooncake** | Default KV transfer | `uv pip install mooncake-transfer-engine` |
| **NIXL** | Alternative transfer | `pip install nixl` |
| **InfiniBand** | RDMA networking | `libibverbs-dev`, `rdma-core`, `ibverbs-providers` |

### MoE / Expert Parallelism

| Dependency | Purpose | Installation |
|------------|---------|--------------|
| **DeepEP** | All-to-all backend | See `scripts/ci/ci_install_deepep.sh` |
| **NVSHMEM** | 3.3.9 | Symmetric memory for collectives |
| **GDRCopy** | v2.4.4 | GPU Direct RDMA |

**DeepEP Build Requirements**:
- `GDRCOPY_HOME=/usr/src/gdrdrv-2.4.4/`
- `NVSHMEM_DIR=/opt/nvshmem/install`
- Commit: `9af0e0d0e74f3577af1979c9b9e1ac2cad0104ee`

### Quantization

| Method | Dependency | Flag |
|--------|------------|------|
| FP8 Weight | Built-in | `--quantization fp8` |
| FP8 KV Cache | Built-in | `--kv-cache-dtype fp8_e5m2` |
| INT4/INT8 | TorchAO 0.9.0 | `--torchao-config int4wo-128` |
| AWQ/GPTQ | Built-in | `--quantization awq` |

---

## Directory Structure

```
sglang/
├── python/sglang/           # Main Python package
│   ├── srt/                 # SGLang Runtime (Server)
│   │   ├── managers/        # Scheduler, tokenizer, model worker
│   │   ├── layers/          # Attention, MoE, quantization kernels
│   │   ├── mem_cache/       # Memory pool, RadixCache, HiCache
│   │   ├── models/          # Model implementations (Llama, DeepSeek, Qwen, etc.)
│   │   ├── server_args.py   # CLI argument definitions
│   │   └── http_server.py   # FastAPI endpoints
│   ├── lang/                # Frontend DSL (programming interface)
│   ├── bench_serving.py     # Benchmark tool
│   └── launch_server.py     # Server entry point
│
├── sgl-router/              # Load balancer & PD coordinator
│   └── launch_router.py     # Router entry point
│
├── sgl-kernel/              # Custom CUDA/Triton kernels (separate package)
│
├── benchmark/               # Benchmarking suites
│   ├── deepseek_v3/         # DeepSeek-specific benchmarks
│   ├── hicache/             # Hierarchical cache tests
│   └── kernels/             # Kernel microbenchmarks
│
├── test/                    # Unit & integration tests
│   └── srt/                 # Runtime tests
│       └── ep/              # Expert parallelism tests
│
├── docs/                    # Sphinx documentation
│   ├── advanced_features/   # PD disaggregation, router, quantization
│   ├── basic_usage/         # Quick start guides
│   └── developer_guide/     # Contribution, profiling, benchmarking
│
├── scripts/                 # Utility scripts
│   ├── ci/                  # CI installation scripts
│   └── playground/          # Development experiments
│
├── docker/                  # Container definitions
│   ├── Dockerfile           # NVIDIA CUDA base
│   ├── Dockerfile.rocm      # AMD ROCm
│   └── compose.yaml         # Docker Compose examples
│
└── 3rdparty/                # Third-party integrations
    └── amd/                 # AMD-specific optimizations
```

### Key Entry Points

| File | Command | Purpose |
|------|---------|---------|
| `python/sglang/launch_server.py` | `python -m sglang.launch_server` | Start inference server |
| `sgl-router/launch_router.py` | `python -m sglang_router.launch_router` | Start load balancer/router |
| `python/sglang/bench_serving.py` | `python -m sglang.bench_serving` | Benchmark serving performance |

---

## Network & Communication

### NCCL (Default)

- All-reduce for TP
- Point-to-point for PP
- Flags: `--nccl-port`, `--enable-nccl-nvls`, `--enable-symm-mem`

### DeepEP (MoE All-to-All)

- Replaces NCCL for MoE expert routing
- Modes: `normal`, `low_latency`, `auto` (default)
- Flag: `--moe-a2a-backend deepep --deepep-mode auto`

### InfiniBand/RDMA

- Required for PD disaggregation with Mooncake/NIXL
- Device detection: `ibv_devinfo mlx5_X`
- Health check: `ibv_devinfo $device | grep "state:"`

---

## Memory Management

### KV Cache Pool

- **Paged**: Token-level granularity (default `--page-size 1`)
- **Static Fraction**: `--mem-fraction-static 0.9` (model + KV pool)
- **Max Tokens**: Auto-calculated or explicit `--max-total-tokens`
- **Max Requests**: `--max-running-requests` (critical for decode)

### Chunked Prefill

- **Purpose**: Split large prefills into chunks to avoid OOM
- **Flag**: `--chunked-prefill-size 4096`
- **Disable**: `--chunked-prefill-size -1`
- **Trade-off**: Lower memory peak vs. slight throughput reduction

### HiCache (Hierarchical Cache)

- **Purpose**: Offload cold KV cache to CPU/storage
- **Flags**: `--enable-hierarchical-cache`, `--hicache-ratio 2.0`
- **Backends**: Host memory, EIC storage

---

## Performance Features

| Feature | Flag | Impact |
|---------|------|--------|
| **CUDA Graphs** | Enabled by default | Reduce kernel launch overhead |
| **Torch Compile** | `--enable-torch-compile` | JIT optimization (small models/batches) |
| **Two-Batch Overlap** | `--enable-two-batch-overlap` | Overlap prefill/decode microbatches (⚠️ experimental) |
| **Mixed Chunk** | `--enable-mixed-chunk` | Mix prefill/decode in same batch |
| **Custom All-Reduce** | Enabled by default | Faster than NCCL for small TP |
| **MSCCLPP** | `--enable-mscclpp` | Microsoft collective library |

---

## Observability

### Metrics (Prometheus)

- Enable: `--enable-metrics`
- Endpoint: `http://host:port/metrics`
- Buckets: `--bucket-time-to-first-token`, `--bucket-inter-token-latency`

### Logging

- Levels: `--log-level {debug,info,warning,error}`
- Requests: `--log-requests --log-requests-level {0,1,2,3}`
- Decode Interval: `--decode-log-interval 40` (log every N decode steps)

### Health Checks

- `/health`: Server ready status
- `/get_server_info`: Model config, KV cache stats
- `/get_model_info`: Model metadata

---

## Common Workflows

### Single-GPU Development
```bash
python -m sglang.launch_server \
  --model-path meta-llama/Llama-3.1-8B-Instruct
```

### Multi-GPU TP
```bash
python -m sglang.launch_server \
  --model-path meta-llama/Llama-3.1-70B-Instruct \
  --tp-size 8
```

### Multi-Node TP
```bash
# Node 0
python -m sglang.launch_server \
  --model-path meta-llama/Llama-3.1-405B-Instruct \
  --tp-size 16 \
  --nnodes 2 --node-rank 0 \
  --dist-init-addr node0:50000

# Node 1
python -m sglang.launch_server \
  --model-path meta-llama/Llama-3.1-405B-Instruct \
  --tp-size 16 \
  --nnodes 2 --node-rank 1 \
  --dist-init-addr node0:50000
```

### PD Disaggregation (1P1D, Single Node)
```bash
# Prefill on GPU 0
CUDA_VISIBLE_DEVICES=0 python -m sglang.launch_server \
  --model-path meta-llama/Llama-3.1-8B-Instruct \
  --disaggregation-mode prefill \
  --disaggregation-ib-device mlx5_0

# Decode on GPU 1
CUDA_VISIBLE_DEVICES=1 python -m sglang.launch_server \
  --model-path meta-llama/Llama-3.1-8B-Instruct \
  --disaggregation-mode decode \
  --port 30001 \
  --base-gpu-id 1 \
  --disaggregation-ib-device mlx5_0

# Router
python -m sglang_router.launch_router \
  --pd-disaggregation \
  --prefill http://127.0.0.1:30000 \
  --decode http://127.0.0.1:30001 \
  --host 0.0.0.0 --port 8000
```

---

**End of ARCHITECTURE memory**