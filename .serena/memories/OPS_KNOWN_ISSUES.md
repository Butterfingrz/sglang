# SGLang Known Issues & Workarounds

**Last Verified:** 2025-10-01  
**Applicable Branch:** main (commit bfa27438)  
**Version:** 0.5.3rc0

---

## Table of Contents
1. [PD Disaggregation Issues](#pd-disaggregation-issues)
2. [Multi-Node & Distributed](#multi-node--distributed)
3. [Memory & Scheduling](#memory--scheduling)
4. [Experimental Features](#experimental-features)
5. [Hardware-Specific](#hardware-specific)

---

## PD Disaggregation Issues

### Issue 1: High TTFT Variance with PD

**Symptoms**:
- TTFT P50: 100ms
- TTFT P99: 2000ms (20× worse)
- Occasional very slow requests

**Root Cause**: Prefill queue congestion or decode node waiting for KV transfer

**Workarounds**:
1. Increase prefill capacity (more nodes or higher TP)
2. Increase timeout thresholds:
   ```bash
   export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=600
   export SGLANG_DISAGGREGATION_WAITING_TIMEOUT=600
   ```
3. Tune transfer thread pool:
   ```bash
   export SGLANG_DISAGGREGATION_THREAD_POOL_SIZE=12
   export SGLANG_DISAGGREGATION_QUEUE_SIZE=8
   ```

**Status**: Expected behavior under load; tune capacity to reduce variance

---

### Issue 2: PD Transfer Hangs with RadixCache

**Symptoms**:
- Requests hang indefinitely
- Decode logs show "waiting for KV cache"
- Prefill completes but decode never receives data

**Root Cause**: Race condition between RadixCache eviction and PD transfer (rare)

**Workaround**:
```bash
# Disable RadixCache temporarily
--disable-radix-cache
```

**Status**: Under investigation, workaround stable

---

### Issue 3: IB Device Auto-Detection Fails

**Symptoms**:
- Server fails to start with "IB device not found"
- IB device exists and is active

**Root Cause**: Auto-detection looks for `mlx5_0` to `mlx5_11` only

**Workaround**:
```bash
# Explicitly specify device
--disaggregation-ib-device mlx5_bond_0

# Or find correct device:
for dev in mlx5_*; do
  if ibv_devinfo $dev 2>/dev/null | grep -q "PORT_ACTIVE"; then
    echo $dev
  fi
done
```

**Status**: Expected; manual specification required for non-standard naming

---

## Multi-Node & Distributed

### Issue 4: NCCL Deadlock with CUDA Graphs (Multi-Node)

**Symptoms**:
- Server hangs during forward pass
- All nodes appear alive but no progress
- Watchdog timeout after 300s

**Root Cause**: CUDA graph capture + NCCL collective timing issue (rare)

**Workaround**:
```bash
# Disable CUDA graphs
--disable-cuda-graph
```

**Trade-off**: ~5-10% throughput loss, but eliminates deadlock

**Status**: Affects <1% of deployments, workaround stable

---

### Issue 5: Asymmetric Node Performance (TP)

**Symptoms**:
- One node consistently slower in logs
- Overall throughput limited by slowest node

**Root Causes**:
- Thermal throttling on one node
- PCIe vs. NVLink configuration mismatch
- Driver version mismatch

**Diagnostic**:
```bash
# Check GPU clocks
nvidia-smi --query-gpu=clocks.current.sm --format=csv

# Check PCIe link
nvidia-smi topo -m

# Check driver
nvidia-smi | grep "Driver Version"
```

**Workaround**: Fix hardware/driver asymmetry, or exclude slow node

---

### Issue 6: `--trust-remote-code` Required but Undocumented

**Symptoms**:
- Model fails to load with "Unrecognized model type"
- Model uses custom modeling code

**Affected Models**: DeepSeek-V3, some Qwen models, custom models

**Workaround**:
```bash
--trust-remote-code
```

**Status**: Expected for models with custom code; always use for DeepSeek/Qwen

---

## Memory & Scheduling

### Issue 7: High Retract Rate with Chunked Prefill

**Symptoms**:
- Many "Retracting requests" warnings
- Decode batch size << `--max-running-requests`
- Low throughput despite low utilization

**Root Cause**: Chunked prefill allocates memory conservatively, causing premature retracts

**Workaround**:
```bash
# Increase conservativeness
--schedule-conservativeness 1.3

# Or disable chunking (if memory allows)
--chunked-prefill-size -1
```

**Trade-off**: Higher conservativeness → lower throughput but fewer retracts

---

### Issue 8: OOM with `--mem-fraction-static 0.9` on Long Context

**Symptoms**:
- OOM during prefill with inputs >16K tokens
- Works fine with shorter inputs

**Root Cause**: Long inputs require temporary memory beyond KV cache pool

**Workaround**:
```bash
# Reduce static fraction
--mem-fraction-static 0.8

# Enable chunking
--chunked-prefill-size 8192
```

**Status**: Expected; tune based on workload

---

### Issue 9: RadixCache Not Sharing Between Nodes (TP)

**Symptoms**:
- Each TP group maintains separate RadixCache
- No cache hit for identical requests to different TP groups

**Root Cause**: RadixCache is per-worker, not global

**Status**: Expected behavior; use router with single TP group for cache sharing

---

## Experimental Features

### Issue 10: `--enable-two-batch-overlap` Causes Hangs

**Symptoms**:
- Server hangs after N requests (N varies)
- High concurrency (>256) increases likelihood
- Requires restart

**Root Cause**: Race condition in overlap scheduler (under investigation)

**Workaround**:
```bash
# Do not use in production
# If enabled, limit concurrency:
--max-running-requests 128
```

**Status**: ⚠️ **Not recommended for production** (v0.5.3rc0)

---

### Issue 11: Torch Compile Cold Start Penalty

**Symptoms**:
- First request takes 60-120s
- Subsequent requests fast

**Root Cause**: JIT compilation on first invocation

**Workaround**:
```bash
# Warm up before serving
--warmup-requests 10

# Or use persistent cache
export TORCHINDUCTOR_CACHE_DIR=/persistent/cache
```

**Status**: Expected; amortize over long-running servers

---

### Issue 12: DP Attention Imbalance without PD

**Symptoms**:
- One DP worker processes prefill while another decodes
- High decode latency variance

**Root Cause**: Unified engine can't prevent DP imbalance

**Workaround**:
```bash
# Use PD disaggregation (recommended)
# Or disable DP attention:
# (remove --enable-dp-attention)
```

**Status**: PD disaggregation resolves completely

---

## Hardware-Specific

### Issue 13: H200 Driver 550 Compatibility

**Symptoms**:
- Kernel panics or GPU hangs
- Driver 550.x on H200

**Root Cause**: Early driver versions unstable on H200

**Workaround**:
```bash
# Upgrade to 555.x or newer
apt install nvidia-driver-555
```

**Minimum**: Driver 555.42.02 for H200

---

### Issue 14: DeepEP Fails on Non-Hopper GPUs

**Symptoms**:
- "DeepEP not supported on this architecture"
- MoE models fail to load

**Root Cause**: DeepEP requires Hopper (H100/H200) with NVSHMEM

**Workaround**:
```bash
# Use fallback MoE backend
--moe-a2a-backend none

# Or use A100 with standard NCCL
# (no DeepEP support)
```

**Status**: Hopper-only feature

---

### Issue 15: PCIe Bandwidth Bottleneck with High TP

**Symptoms**:
- Low throughput with TP=8 on PCIe GPUs
- TP=4 faster than TP=8

**Root Cause**: PCIe Gen4 x16 = ~64 GB/s, insufficient for high TP all-reduce

**Workaround**:
```bash
# Use DP instead of TP for PCIe configs
--tp-size 2 --dp-size 4  # Instead of --tp-size 8

# Or use NVLink-enabled nodes
```

**Status**: Hardware limitation; prefer NVLink for TP > 4

---

## Debugging Paths

### Hang During Startup

**Check**:
1. Disk space for model download
2. Network connectivity (multi-node)
3. NCCL environment variables
4. GPU visibility (`nvidia-smi`)

**Emergency**:
```bash
# Disable all optimizations
--disable-cuda-graph \
--disable-radix-cache \
--disable-overlap-schedule
```

---

### Random Accuracy Issues

**Check**:
1. FP8/quantization enabled? (may reduce accuracy)
2. RadixCache enabled? (disable for determinism testing)
3. Torch compile? (numeric differences possible)

**Deterministic Mode**:
```bash
--disable-radix-cache \
--dtype float16  # Avoid FP8
# Send one request at a time
```

---

### Intermittent OOM

**Check**:
1. Varying input lengths → enable chunked prefill
2. Memory fragmentation → restart server
3. Concurrent requests spike → reduce `--max-running-requests`

**Diagnostic**:
```bash
# Monitor GPU memory continuously
nvidia-smi dmon -s mu -c 1000 > gpu_mem.log
```

---

### Router Not Forwarding Requests

**Check**:
1. Prefill/decode nodes healthy: `curl http://node:port/health`
2. Router can reach nodes: `ping node-ip`
3. Router logs for errors

**Workaround**:
```bash
# Restart router with debug logging
python -m sglang_router.launch_router \
  --pd-disaggregation \
  --prefill http://... \
  --decode http://... \
  --log-level debug
```

---

## Version-Specific Issues

### v0.5.3rc0

**Known Issues**:
- `--enable-two-batch-overlap` instability (Issue #10)
- PD + RadixCache rare hangs (Issue #2)

**Recommended Settings** (production):
```bash
# Disable experimental features
# (no --enable-two-batch-overlap)

# For PD, consider:
--disable-radix-cache  # If hangs occur
```

---

## Escalation Path

### Issue Not Listed

1. **Search GitHub Issues**: https://github.com/sgl-project/sglang/issues
2. **Check Slack**: https://slack.sglang.ai/
3. **File New Issue**: Include:
   - SGLang version (`python -m sglang.launch_server --version`)
   - Launch command (sanitize sensitive data)
   - Error logs (last 50 lines)
   - Hardware/driver info
   - Reproducible example

### Critical Production Issue

1. **Emergency Rollback**: Previous stable version
2. **Disable Features**: Start minimal, add incrementally
3. **Contact**: contact@sglang.ai (for enterprises)

---

**End of OPS_KNOWN_ISSUES memory**