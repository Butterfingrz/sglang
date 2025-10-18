# SGLang Launch Recipes by Scale

**Last Verified:** 2025-10-01  
**Applicable Branch:** main (commit bfa27438)

---

## Table of Contents
1. [Single-Node Configurations](#single-node-configurations)
2. [Multi-Node Standard TP](#multi-node-standard-tp)
3. [PD Disaggregation Recipes](#pd-disaggregation-recipes)
4. [Large-Scale Deployments](#large-scale-deployments)
5. [Parameter Field Explanations](#parameter-field-explanations)

---

## Single-Node Configurations

### Recipe 1: Single H200 (Development/Testing)

**Hardware**: 1× H200 80GB  
**Model**: Llama-3.1-8B  
**Use Case**: Development, testing, small-scale serving

```bash
python -m sglang.launch_server \
  --model-path meta-llama/Llama-3.1-8B-Instruct \
  --host 0.0.0.0 \
  --port 30000 \
  --mem-fraction-static 0.85 \
  --max-running-requests 128 \
  --enable-metrics \
  --log-level info
```

**Expected Performance**:
- TTFT P50: 20-50ms
- Decode: 80-120 tok/s per request
- Max throughput: 10,000 tok/s (output)

**Common Failures**:
- Port already in use → change `--port`
- Model download failure → check HuggingFace token

---

### Recipe 2: 8× H200 TP=8 (Single-Node Llama-3.1-405B)

**Hardware**: 8× H200 80GB (NVLink)  
**Model**: Llama-3.1-405B  
**Use Case**: Large model single-node serving

```bash
python -m sglang.launch_server \
  --model-path meta-llama/Llama-3.1-405B-Instruct \
  --tp-size 8 \
  --host 0.0.0.0 \
  --port 30000 \
  --mem-fraction-static 0.88 \
  --max-running-requests 128 \
  --chunked-prefill-size 8192 \
  --enable-metrics \
  --log-requests
```

**Expected Performance**:
- TTFT P50: 100-300ms (input-dependent)
- Decode: 40-60 tok/s per request
- Max throughput: 3,000-5,000 tok/s (output)

**Common Failures**:
- OOM → reduce `--mem-fraction-static 0.85`
- P2P errors → add `--enable-p2p-check`
- NCCL timeout → increase `--dist-timeout 1800`

---

## Multi-Node Standard TP

### Recipe 3: 2-Node TP=16 (Llama-3.1-405B)

**Hardware**: 2 nodes × 8 H100 80GB  
**Model**: Llama-3.1-405B  
**Use Case**: Multi-node tensor parallelism

**Node 0 (Master)**:
```bash
export MASTER_ADDR=10.0.0.1
export MASTER_PORT=29500

python -m sglang.launch_server \
  --model-path meta-llama/Llama-3.1-405B-Instruct \
  --tp-size 16 \
  --host 0.0.0.0 \
  --port 30000 \
  --dist-init-addr ${MASTER_ADDR}:${MASTER_PORT} \
  --nnodes 2 \
  --node-rank 0 \
  --mem-fraction-static 0.85 \
  --max-running-requests 128 \
  --chunked-prefill-size 8192
```

**Node 1 (Worker)**:
```bash
export MASTER_ADDR=10.0.0.1
export MASTER_PORT=29500

python -m sglang.launch_server \
  --model-path meta-llama/Llama-3.1-405B-Instruct \
  --tp-size 16 \
  --host 0.0.0.0 \
  --port 30000 \
  --dist-init-addr ${MASTER_ADDR}:${MASTER_PORT} \
  --nnodes 2 \
  --node-rank 1 \
  --mem-fraction-static 0.85 \
  --max-running-requests 128 \
  --chunked-prefill-size 8192
```

**Health Check**:
```bash
# Only node 0 responds (TP group has single endpoint)
curl http://10.0.0.1:30000/health
```

**Common Failures**:
- Nodes can't communicate → check firewall on port 29500
- NCCL init timeout → verify IB/network connectivity
- Asymmetric OOM → ensure identical `--mem-fraction-static`

---

## PD Disaggregation Recipes

### Recipe 4: 1P1D Single-Node (Educational)

**Hardware**: 1 node × 2+ GPUs  
**Model**: Llama-3.1-8B  
**Use Case**: PD feature demonstration

```bash
#!/bin/bash
# Find active IB device
IB_DEVICE=$(for dev in mlx5_{0..11}; do
  if ibv_devinfo $dev 2>/dev/null | grep -q "PORT_ACTIVE"; then
    echo $dev; break
  fi
done)

echo "Using IB device: $IB_DEVICE"

# Prefill on GPU 0
CUDA_VISIBLE_DEVICES=0 python -m sglang.launch_server \
  --model-path meta-llama/Llama-3.1-8B-Instruct \
  --disaggregation-mode prefill \
  --disaggregation-ib-device $IB_DEVICE \
  --disaggregation-bootstrap-port 9001 \
  --host 127.0.0.1 \
  --port 30000 &

PREFILL_PID=$!

# Decode on GPU 1
CUDA_VISIBLE_DEVICES=1 python -m sglang.launch_server \
  --model-path meta-llama/Llama-3.1-8B-Instruct \
  --disaggregation-mode decode \
  --disaggregation-ib-device $IB_DEVICE \
  --host 127.0.0.1 \
  --port 30001 \
  --base-gpu-id 1 &

DECODE_PID=$!

# Wait for servers
sleep 60

# Health check
curl -f http://127.0.0.1:30000/health && echo "✓ Prefill OK"
curl -f http://127.0.0.1:30001/health && echo "✓ Decode OK"

# Router
python -m sglang_router.launch_router \
  --pd-disaggregation \
  --prefill http://127.0.0.1:30000 \
  --decode http://127.0.0.1:30001 \
  --host 0.0.0.0 \
  --port 8000
```

**Test**:
```bash
curl http://localhost:8000/health
curl -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"meta-llama/Llama-3.1-8B-Instruct","prompt":"Once upon a time","max_tokens":50}'
```

**Common Failures**:
- IB device not found → verify `ibv_devices`
- Prefill/decode can't connect → check bootstrap port, IB state
- Router 502 → check prefill/decode endpoints

---

### Recipe 5: 2P2D Multi-Node (Production)

**Hardware**: 4 nodes × 8 H100 (2 prefill, 2 decode)  
**Model**: Llama-3.1-70B  
**Topology**: TP=8 per node group

```bash
# ========== Prefill Node 0 (Master) ==========
python -m sglang.launch_server \
  --model-path meta-llama/Llama-3.1-70B-Instruct \
  --disaggregation-mode prefill \
  --disaggregation-ib-device mlx5_0 \
  --disaggregation-bootstrap-port 9001 \
  --host 10.0.0.1 \
  --port 30000 \
  --dist-init-addr 10.0.0.1:29500 \
  --nnodes 2 \
  --node-rank 0 \
  --tp-size 8 \
  --mem-fraction-static 0.75 \
  --chunked-prefill-size 8192

# ========== Prefill Node 1 (Worker) ==========
python -m sglang.launch_server \
  --model-path meta-llama/Llama-3.1-70B-Instruct \
  --disaggregation-mode prefill \
  --disaggregation-ib-device mlx5_0 \
  --disaggregation-bootstrap-port 9002 \
  --host 10.0.0.2 \
  --port 30000 \
  --dist-init-addr 10.0.0.1:29500 \
  --nnodes 2 \
  --node-rank 1 \
  --tp-size 8 \
  --mem-fraction-static 0.75 \
  --chunked-prefill-size 8192

# ========== Decode Node 0 (Master) ==========
python -m sglang.launch_server \
  --model-path meta-llama/Llama-3.1-70B-Instruct \
  --disaggregation-mode decode \
  --disaggregation-ib-device mlx5_0 \
  --host 10.0.0.3 \
  --port 30001 \
  --dist-init-addr 10.0.0.3:29500 \
  --nnodes 2 \
  --node-rank 0 \
  --tp-size 8 \
  --mem-fraction-static 0.85 \
  --max-running-requests 256

# ========== Decode Node 1 (Worker) ==========
python -m sglang.launch_server \
  --model-path meta-llama/Llama-3.1-70B-Instruct \
  --disaggregation-mode decode \
  --disaggregation-ib-device mlx5_0 \
  --host 10.0.0.4 \
  --port 30001 \
  --dist-init-addr 10.0.0.3:29500 \
  --nnodes 2 \
  --node-rank 1 \
  --tp-size 8 \
  --mem-fraction-static 0.85 \
  --max-running-requests 256

# ========== Router ==========
python -m sglang_router.launch_router \
  --pd-disaggregation \
  --prefill http://10.0.0.1:30000 \
  --decode http://10.0.0.3:30001 \
  --host 0.0.0.0 \
  --port 8000 \
  --load-balance-method minimum_tokens
```

**Expected Performance**:
- TTFT P50: 80-150ms
- TTFT P99: 200-400ms
- Throughput: 8,000-12,000 tok/s (output)

---

### Recipe 6: 4P-9D Large-Scale (DeepSeek-V3)

**Hardware**: 13 nodes × 8 H100/H200 (52 GPUs prefill, 72 GPUs decode)  
**Model**: DeepSeek-V3 (685B)  
**Topology**: TP=16, DP=8 (prefill), TP=16, DP=8 (decode)

```bash
# ========== Prefill Nodes (0-3) ==========
for RANK in 0 1 2 3; do
  ssh prefill-node-${RANK} "
    python -m sglang.launch_server \
      --model-path deepseek-ai/DeepSeek-V3-0324 \
      --disaggregation-mode prefill \
      --disaggregation-ib-device mlx5_0 \
      --disaggregation-bootstrap-port $((9001 + RANK)) \
      --host \$(hostname -I | awk '{print \$1}') \
      --port 30000 \
      --trust-remote-code \
      --dist-init-addr prefill-master:29500 \
      --nnodes 4 \
      --node-rank ${RANK} \
      --tp-size 16 \
      --dp-size 8 \
      --enable-dp-attention \
      --moe-a2a-backend deepep \
      --mem-fraction-static 0.8 \
      --max-prefill-tokens 32768
  " &
done

# ========== Decode Nodes (0-8) ==========
for RANK in 0 1 2 3 4 5 6 7 8; do
  ssh decode-node-${RANK} "
    python -m sglang.launch_server \
      --model-path deepseek-ai/DeepSeek-V3-0324 \
      --disaggregation-mode decode \
      --disaggregation-ib-device mlx5_0 \
      --host \$(hostname -I | awk '{print \$1}') \
      --port 30001 \
      --trust-remote-code \
      --dist-init-addr decode-master:29500 \
      --nnodes 9 \
      --node-rank ${RANK} \
      --tp-size 16 \
      --dp-size 8 \
      --enable-dp-attention \
      --enable-dp-lm-head \
      --moe-a2a-backend deepep \
      --mem-fraction-static 0.8 \
      --max-running-requests 512
  " &
done

wait

# ========== Router ==========
python -m sglang_router.launch_router \
  --pd-disaggregation \
  --prefill \
    http://prefill-node-0:30000 \
    http://prefill-node-1:30000 \
    http://prefill-node-2:30000 \
    http://prefill-node-3:30000 \
  --decode \
    http://decode-node-0:30001 \
    http://decode-node-1:30001 \
    http://decode-node-2:30001 \
    http://decode-node-3:30001 \
    http://decode-node-4:30001 \
    http://decode-node-5:30001 \
    http://decode-node-6:30001 \
    http://decode-node-7:30001 \
    http://decode-node-8:30001 \
  --host 0.0.0.0 \
  --port 8000 \
  --load-balance-method minimum_tokens
```

**Expected Performance** (DeepSeek-V3, Input=2048, Output=256):
- TTFT P50: 150-300ms
- TTFT P99: 400-800ms
- Throughput: 15,000-20,000 tok/s (output)

---

## Large-Scale Deployments

### Recipe 7: Data Parallel (8× Independent Workers)

**Hardware**: 8 nodes × 8 H100 (64 GPUs total)  
**Model**: Llama-3.1-70B  
**Use Case**: Maximum throughput, stateless requests

```bash
# Launch 8 independent servers (one per node)
for NODE in 0 1 2 3 4 5 6 7; do
  ssh node-${NODE} "
    python -m sglang.launch_server \
      --model-path meta-llama/Llama-3.1-70B-Instruct \
      --tp-size 8 \
      --host 0.0.0.0 \
      --port 30000 \
      --mem-fraction-static 0.88 \
      --max-running-requests 256
  " &
done

# Router aggregates all endpoints
python -m sglang_router.launch_router \
  --model-path meta-llama/Llama-3.1-70B-Instruct \
  --worker-urls \
    http://node-0:30000 \
    http://node-1:30000 \
    http://node-2:30000 \
    http://node-3:30000 \
    http://node-4:30000 \
    http://node-5:30000 \
    http://node-6:30000 \
    http://node-7:30000 \
  --host 0.0.0.0 \
  --port 8000
```

**Expected Performance**:
- Throughput: 50,000+ tok/s (aggregate)
- Linear scaling with workers

---

## Parameter Field Explanations

### Required Parameters

| Parameter | Description | Example | When to Set |
|-----------|-------------|---------|-------------|
| `--model-path` | HF model ID or local path | `meta-llama/Llama-3.1-8B-Instruct` | Always |
| `--host` | Bind address | `0.0.0.0` | Always |
| `--port` | HTTP port | `30000` | Always |

### Parallelism Parameters

| Parameter | Description | Default | Notes |
|-----------|-------------|---------|-------|
| `--tp-size` | Tensor parallelism | 1 | Must divide GPU count |
| `--dp-size` | Data parallelism | 1 | Use router instead |
| `--ep-size` | Expert parallelism | 1 | MoE only |
| `--pp-size` | Pipeline parallelism | 1 | Experimental |

### Multi-Node Parameters

| Parameter | Description | Example | Notes |
|-----------|-------------|---------|-------|
| `--dist-init-addr` | Master address:port | `10.0.0.1:29500` | Required for multi-node |
| `--nnodes` | Total nodes | `2` | Required for multi-node |
| `--node-rank` | Node index (0-based) | `0` | Required for multi-node |

### PD Disaggregation Parameters

| Parameter | Description | Example | Notes |
|-----------|-------------|---------|-------|
| `--disaggregation-mode` | `prefill` or `decode` | `prefill` | Required for PD |
| `--disaggregation-ib-device` | InfiniBand device | `mlx5_0` | Required for Mooncake |
| `--disaggregation-bootstrap-port` | Coordination port | `9001` | Prefill only |
| `--base-gpu-id` | GPU offset | `1` | Decode when GPUs shared |

### Memory Parameters

| Parameter | Description | Default | Range |
|-----------|-------------|---------|-------|
| `--mem-fraction-static` | Model + KV pool | 0.9 | 0.5-0.95 |
| `--max-running-requests` | Max concurrent | Auto | 32-2048 |
| `--max-total-tokens` | KV pool size | Auto | - |
| `--chunked-prefill-size` | Chunk size | Auto | 2048-32768 or -1 |

### MoE Parameters (DeepSeek, Qwen)

| Parameter | Description | Default | Notes |
|-----------|-------------|---------|-------|
| `--moe-a2a-backend` | All-to-all backend | `none` | Use `deepep` |
| `--enable-dp-attention` | DP for attention | false | Requires `dp_size == tp_size` |
| `--enable-dp-lm-head` | Vocab parallel | false | Use with DP attention |
| `--moe-dense-tp-size` | Dense layer TP | `--tp-size` | Smaller for large TP |

---

## Common Failure Scenarios

### Failure 1: "NCCL Init Timeout"

**Symptoms**: Hangs during launch, "NCCL init" in logs

**Causes**:
- Firewall blocking `--nccl-port` or `--dist-init-addr` port
- Network connectivity issues between nodes
- Wrong `--dist-init-addr` (unreachable or typo)

**Solutions**:
```bash
# Check connectivity
ping <other_node_ip>

# Open ports
ufw allow 29500/tcp

# Increase timeout
--dist-timeout 1800
```

---

### Failure 2: "OOM During Prefill"

**Symptoms**: CUDA OOM error when processing large inputs

**Solutions** (priority order):
```bash
# 1. Reduce KV pool
--mem-fraction-static 0.75

# 2. Enable/reduce chunking
--chunked-prefill-size 4096

# 3. Reduce max prefill batch
--max-prefill-tokens 8192
```

---

### Failure 3: "PD Transfer Timeout"

**Symptoms**: Requests timeout, "bootstrap timeout" or "waiting timeout" in logs

**Causes**:
- IB device not active
- Prefill/decode can't communicate over RDMA
- High load causing slow KV transfer

**Solutions**:
```bash
# Increase timeouts
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=600
export SGLANG_DISAGGREGATION_WAITING_TIMEOUT=600

# Verify IB
ibv_devinfo mlx5_0 | grep state  # Should show ACTIVE
```

---

**End of LAUNCH_RECIPES memory**