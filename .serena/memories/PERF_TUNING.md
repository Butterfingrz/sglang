# SGLang Performance Tuning Guide

**Last Verified:** 2025-10-01  
**Applicable Branch:** main (commit bfa27438)

---

## Table of Contents
1. [Critical Tuning Knobs](#critical-tuning-knobs)
2. [Memory Management](#memory-management)
3. [Scheduling & Batching](#scheduling--batching)
4. [Experimental Features](#experimental-features)
5. [Hardware-Specific Tuning](#hardware-specific-tuning)

---

## Critical Tuning Knobs

### 1. `--mem-fraction-static`

**What**: Fraction of GPU memory for model weights + KV cache pool  
**Default**: 0.9  
**Range**: 0.5 - 0.95

**Impact**:
- **Higher (0.9)**: More KV cache → higher throughput, risk of OOM during prefill
- **Lower (0.7)**: Less KV cache → lower throughput, safer for large inputs

**Tuning**:
```bash
# Start conservative
--mem-fraction-static 0.75

# Gradually increase if no OOM
--mem-fraction-static 0.8
--mem-fraction-static 0.85
--mem-fraction-static 0.9

# Monitor: nvidia-smi, server logs for "out of memory"
```

**Recommendations**:
| GPU | Model Size | Prefill | Decode |
|-----|------------|---------|--------|
| H100 80GB | 70B | 0.75-0.8 | 0.85-0.9 |
| H200 80GB | 70B | 0.8-0.85 | 0.88-0.92 |
| H100 80GB | 405B (TP=8) | 0.75 | 0.85 |

---

### 2. `--max-running-requests`

**What**: Maximum concurrent requests in decode phase  
**Default**: Auto-calculated based on memory  
**Range**: 32 - 2048

**Impact**:
- **Higher**: More parallelism → higher throughput, but more memory pressure
- **Lower**: Less memory usage → lower throughput

**Tuning**:
```bash
# Decode nodes (PD disaggregation)
--max-running-requests 256  # Start here
--max-running-requests 512  # High throughput
--max-running-requests 128  # Conservative

# Monitor: KV cache utilization, retract rate
```

**Recommendations**:
| Scenario | Value | Notes |
|----------|-------|-------|
| Latency-sensitive | 64-128 | Low variance |
| Throughput-focused | 256-512 | Decode-heavy workload |
| PD Decode cluster | 512-1024 | Dedicated decode GPUs |

---

### 3. `--chunked-prefill-size`

**What**: Maximum tokens in a prefill chunk (splits large prefills)  
**Default**: Auto (usually 8192-16384)  
**Range**: 2048 - 32768, or -1 to disable

**Impact**:
- **Smaller (4096)**: Lower memory peak, more scheduler overhead
- **Larger (16384)**: Higher memory peak, better GPU utilization
- **Disabled (-1)**: No chunking, highest risk of OOM

**Tuning**:
```bash
# Long context models
--chunked-prefill-size 8192

# OOM during prefill → reduce
--chunked-prefill-size 4096

# Disable for short contexts only
--chunked-prefill-size -1
```

**Recommendations**:
| Context Length | Chunk Size | Notes |
|----------------|------------|-------|
| < 8K | -1 or 8192 | Chunking optional |
| 8K - 32K | 8192 | Balance memory/perf |
| 32K - 128K | 4096-8192 | Prevent OOM |

---

### 4. `--page-size`

**What**: Tokens per KV cache page  
**Default**: 1  
**Range**: 1, 4, 8, 16

**Impact**:
- **1**: Fine-grained memory, higher management overhead
- **16**: Coarser memory, lower overhead, potential fragmentation

**Tuning**:
```bash
# Default (recommended for most cases)
--page-size 1

# Experimental: reduce memory management overhead
--page-size 16
```

**Recommendation**: Keep at **1** unless profiling shows memory management bottleneck

---

### 5. `--disable-radix-cache`

**What**: Disables prefix caching (RadixAttention)  
**Default**: false (enabled)  
**Impact**: No KV cache reuse → higher latency, higher memory

**When to disable**:
- ❌ Production (never disable unless specific reason)
- ✓ Debugging determinism issues
- ✓ Benchmarking cold cache performance
- ✓ Workloads with no shared prefixes

**Cache hit rate** (enable to see benefit):
```bash
# Shared system prompts → 80-95% hit rate
# Random prompts → 0-10% hit rate
```

---

### 6. `--enable-two-batch-overlap`

**What**: Overlap prefill and decode microbatches  
**Default**: false  
**Status**: ⚠️ **Experimental**

**Impact**:
- **Enabled**: Potentially lower TTFT, risk of instability/hangs
- **Disabled**: Stable, slightly higher TTFT

**Known Issues**:
- Occasional hangs with high concurrency (>256 requests)
- Increased TTFT variance under certain workloads
- Not recommended for production without extensive testing

**Tuning**:
```bash
# Enable only after baseline testing
--enable-two-batch-overlap

# Monitor for hangs, OOM, or increased TTFT variance
```

**Decision Matrix**:
| Use Case | Enable? | Notes |
|----------|---------|-------|
| Production | ❌ | Risk too high |
| Research/experiments | ✓ | Potential wins |
| Latency-critical | ❌ | Variance unacceptable |

---

### 7. `--enable-dp-attention` + `--enable-dp-lm-head`

**What**: Hybrid parallelism - DP for attention, TP for FFN/MoE  
**Default**: false  
**Requirement**: `dp_size == tp_size`

**Impact**:
- **Enabled**: Better expert utilization in MoE models, reduced all-gather
- **Disabled**: Standard TP behavior

**Supported Models**:
- DeepSeek-V2 / V3
- Qwen 2/3 MoE

**Tuning**:
```bash
# DeepSeek-V3 with TP=16, DP=8 (total 128 GPUs)
--tp-size 16 \
--dp-size 8 \
--enable-dp-attention \
--enable-dp-lm-head
```

**Performance Gain**: 15-30% higher throughput on MoE models

---

### 8. `--moe-dense-tp-size`

**What**: TP size for MoE dense layers (shared experts, gates)  
**Default**: Same as `--tp-size`  
**Use Case**: Prevent GEMM dimension errors with large TP

**Impact**:
- Smaller TP for dense layers → avoid dimension < 64 errors
- Larger TP for expert layers → full parallelism

**Tuning**:
```bash
# DeepSeek-V3 with TP=16
--tp-size 16 \
--moe-dense-tp-size 8  # Dense layers use TP=8 only
```

**When needed**: TP >= 8 on MoE models with small hidden dimensions

---

## Memory Management

### Symptom: OOM during Prefill

**Solutions (priority order)**:
1. Reduce `--mem-fraction-static` (e.g., 0.75)
2. Enable/reduce `--chunked-prefill-size` (e.g., 4096)
3. Reduce `--max-prefill-tokens` (e.g., 8192)
4. Increase TP size (if multi-GPU available)

### Symptom: OOM during Decode

**Solutions**:
1. Reduce `--max-running-requests`
2. Reduce `--mem-fraction-static`
3. Enable `--cpu-offload-gb 16` (experimental)

### Symptom: Low KV Cache Utilization

**Check**:
```bash
curl http://localhost:30000/get_server_info | jq '.kv_cache_utilization'
```

**If < 50%**: Increase `--max-running-requests` or `--max-total-tokens`

---

## Scheduling & Batching

### Decode Batching

**Key Metric**: Decode batch size (log output)

```bash
# Server log example:
Decode batch. #running-req: 256, #token: 12543, token usage: 0.82, ...
```

**Target**: `#running-req` close to `--max-running-requests`, `token usage` > 0.7

**If low**:
- Increase request rate (benchmark side)
- Increase `--max-running-requests`
- Check for retracted requests (memory pressure)

### Prefill Scheduling

**Flag**: `--schedule-policy fcfs`  
**Options**: `fcfs` (first-come-first-serve), `lpm` (longest-prefix-match)

**Recommendation**: Keep `fcfs` for most workloads

**Conservativeness**:
```bash
# More conservative (fewer retracts)
--schedule-conservativeness 1.2

# Less conservative (more aggressive)
--schedule-conservativeness 0.8
```

---

## Experimental Features

### Torch Compile

**Flag**: `--enable-torch-compile`  
**Impact**: 10-30% speedup for small models, small batch sizes  
**Overhead**: Cold start penalty (~1-2 minutes)

**When to enable**:
- Small models (< 13B)
- Batch size < 32
- Long-running server (amortize cold start)

**Cache**:
```bash
export TORCHINDUCTOR_CACHE_DIR=/tmp/torch_compile_cache
```

### Mixed Chunk

**Flag**: `--enable-mixed-chunk`  
**What**: Mix prefill and decode in same batch (with chunked prefill)  
**Status**: Experimental

**Impact**: Better GPU utilization, complexity in scheduling

### CUDA Graph Tuning

**Flags**:
- `--disable-cuda-graph`: Disable entirely (debugging)
- `--cuda-graph-max-bs 256`: Max batch size for CUDA graph
- `--cuda-graph-bs 1,2,4,8,16,32,64,128`: Explicit batch sizes

**Default**: Auto-capture common batch sizes

**When to disable**: Multi-node deadlock (rare), debugging

---

## Hardware-Specific Tuning

### H100 / H200

**Optimal**:
```bash
--mem-fraction-static 0.85
--max-running-requests 256
--chunked-prefill-size 8192
--enable-nccl-nvls  # NVLink-Switch for prefill-heavy
--enable-symm-mem   # Symmetric memory for fast collectives
```

**NCCL**:
```bash
export NCCL_P2P_LEVEL=NVL  # Use NVLink
export NCCL_IB_HCA=mlx5_0:1  # Specify IB device
```

### A100

**Optimal**:
```bash
--mem-fraction-static 0.8
--max-running-requests 128
--chunked-prefill-size 8192
```

### NVLink vs. PCIe

**NVLink** (TP friendly):
- Prefer TP over DP for multi-GPU
- Enable `--enable-symm-mem`

**PCIe** (limited bandwidth):
- Prefer DP over TP (less communication)
- Avoid high TP sizes

### InfiniBand (PD Disaggregation)

**Optimal**:
```bash
# Prefill
--disaggregation-ib-device mlx5_0

# Decode
--disaggregation-ib-device mlx5_0
--max-running-requests 512  # More decode parallelism
```

**Environment**:
```bash
export SGLANG_DISAGGREGATION_THREAD_POOL_SIZE=8
export SGLANG_DISAGGREGATION_QUEUE_SIZE=4
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=600  # 10min (if high TTFT acceptable)
export SGLANG_DISAGGREGATION_WAITING_TIMEOUT=600
```

---

## Performance Checklist

### Prefill Optimization
- [ ] `--mem-fraction-static` tuned for workload
- [ ] `--chunked-prefill-size` set appropriately
- [ ] `--max-prefill-tokens` matches context length needs
- [ ] RadixCache enabled (unless no shared prefixes)
- [ ] TP size appropriate for model size

### Decode Optimization
- [ ] `--max-running-requests` maximized without OOM
- [ ] CUDA graphs enabled (default)
- [ ] `--mem-fraction-static` leaves room for decode batches
- [ ] `--decode-log-interval` set to monitor batches

### MoE Models (DeepSeek, Qwen)
- [ ] `--moe-a2a-backend deepep` configured
- [ ] `--enable-dp-attention` + `--enable-dp-lm-head` enabled
- [ ] `--moe-dense-tp-size` set if TP > 8
- [ ] `--deepep-mode auto` (default)

### PD Disaggregation
- [ ] IB device active and specified
- [ ] Prefill `--mem-fraction-static 0.8` (more conservative)
- [ ] Decode `--mem-fraction-static 0.85-0.9` (more aggressive)
- [ ] Decode `--max-running-requests 256-512`
- [ ] Router health checks passing

---

## Profiling & Debugging

### Enable Profiling

```bash
# Server side
export SGLANG_TORCH_PROFILER_DIR=/tmp/profiles
python -m sglang.launch_server --model-path ... --enable-profile-cuda-graph

# Client side (bench_serving)
python -m sglang.bench_serving --backend sglang --profile

# Analyze traces
tensorboard --logdir /tmp/profiles
```

### Enable Debug Logging

```bash
python -m sglang.launch_server \
  --log-level debug \
  --log-requests \
  --log-requests-level 3  # Log all inputs/outputs
```

### NaN Detection

```bash
--enable-nan-detection  # Debugging only, significant overhead
```

---

**End of PERF_TUNING memory**