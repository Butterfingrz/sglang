# SGLang Log Glossary & Interpretation

**Last Verified:** 2025-10-01  
**Applicable Branch:** main (commit bfa27438)

---

## Table of Contents
1. [Server Startup Logs](#server-startup-logs)
2. [Prefill Batch Logs](#prefill-batch-logs)
3. [Decode Batch Logs](#decode-batch-logs)
4. [Request Lifecycle](#request-lifecycle)
5. [Error Patterns](#error-patterns)

---

## Server Startup Logs

### Model Loading

```
INFO:     Loading model weights...
INFO:     Model weights loaded. Peak memory: 45.2 GB
INFO:     Initializing KV cache. Size: 64.5 GB
INFO:     Server ready. Model: meta-llama/Llama-3.1-8B-Instruct
```

**Fields**:
- **Peak memory**: Model weights + overhead
- **KV cache size**: `--mem-fraction-static` allocated pool

**Healthy**: Model loaded in 30-120s, no errors

---

### Distributed Init (Multi-Node)

```
INFO:     Initializing torch.distributed with backend=nccl
INFO:     Rank 0/2, local_rank 0/8
INFO:     All processes joined. World size: 16
```

**Fields**:
- **Rank**: Global rank in all processes
- **local_rank**: GPU index on current node
- **World size**: Total processes (nnodes × GPUs per node)

**Healthy**: All ranks join within 60s

---

### PD Disaggregation Init

```
INFO:     PD Disaggregation enabled. Mode: prefill
INFO:     Transfer backend: mooncake
INFO:     IB device: mlx5_0 (state: ACTIVE)
INFO:     Bootstrap server listening on: 0.0.0.0:9001
```

**Prefill-specific**:
- Bootstrap server must start successfully
- IB device must be ACTIVE

```
INFO:     PD Disaggregation enabled. Mode: decode
INFO:     Connecting to prefill bootstrap servers...
INFO:     Connected to prefill: 10.0.0.1:9001
```

**Decode-specific**:
- Must connect to at least one prefill node
- Heartbeat mechanism starts

---

## Prefill Batch Logs

### Standard Prefill

```
INFO:     Prefill batch. #seq: 8, #token: 8192, #req: 8, #chunk: 1
```

**Fields**:
- `#seq`: Number of sequences in batch
- `#token`: Total input tokens processed
- `#req`: Number of requests (usually == #seq)
- `#chunk`: Number of chunks (1 if no chunked prefill)

**Interpretation**:
- **High #seq**: Good batching efficiency
- **#chunk > 1**: Large inputs being chunked

---

### Chunked Prefill

```
INFO:     Prefill batch. #seq: 4, #token: 32768, #req: 4, #chunk: 4
```

**Interpretation**:
- 4 requests, 32K tokens total
- Each request split into multiple chunks
- Reduces memory peak vs. single-pass prefill

---

## Decode Batch Logs

### Standard Decode

```
INFO:     Decode batch. #running-req: 256, #token: 12543, token usage: 0.82, gen throughput (token/s): 8234, #queue-req: 0
```

**Fields**:
- `#running-req`: Active requests being decoded
- `#token`: Total tokens in KV cache for running requests
- `token usage`: `#token / max_total_tokens` (KV cache utilization)
- `gen throughput`: Output tokens per second
- `#queue-req`: Requests waiting to be scheduled

**Interpretation**:
- **`#running-req` close to `--max-running-requests`**: Good saturation
- **`token usage` > 0.8**: Memory well-utilized
- **`token usage` > 0.95**: Risk of memory pressure, retracts likely
- **`#queue-req` > 0**: Backlog, may need more decode capacity

---

### Decode with Retract

```
WARN:     Retracting 3 requests due to memory pressure
INFO:     Decode batch. #running-req: 253, #token: 12100, token usage: 0.79, ...
```

**Interpretation**:
- Scheduler evicted 3 requests to free KV cache
- Requests will be retried later (prefill rerun)
- **High retract rate**: Reduce `--max-running-requests` or increase memory

---

### Log Interval

**Controlled by**: `--decode-log-interval 40` (default)

```
# Logged every 40 decode steps
INFO:     Decode batch. #running-req: 256, ...
# (39 steps of silence)
INFO:     Decode batch. #running-req: 248, ...
```

**Interpretation**:
- Reduces log verbosity
- To log every step: `--decode-log-interval 1`

---

## Request Lifecycle

### Request Received

```
INFO:     Request received. ID: req_abc123, prompt_len: 1024, max_tokens: 256
```

**Fields**:
- **ID**: Unique request identifier
- **prompt_len**: Input tokens
- **max_tokens**: Requested output length

---

### Request Scheduled (Prefill)

```
INFO:     Scheduling request req_abc123 for prefill
```

**Interpretation**: Request admitted to prefill batch

---

### Request Completed

```
INFO:     Request completed. ID: req_abc123, latency: 3450ms, prompt_len: 1024, output_len: 256, TTFT: 98ms
```

**Fields**:
- **latency**: End-to-end time (ms)
- **output_len**: Actual generated tokens
- **TTFT**: Time to first token (ms)

**Healthy**: TTFT < 200ms, latency reasonable for output length

---

## Error Patterns

### OOM During Prefill

```
ERROR:    CUDA out of memory. Tried to allocate 2.5 GB (GPU 0; 79.2 GB total capacity; 76.8 GB already allocated)
```

**Root Cause**: Input too large, insufficient memory for prefill

**Solutions**:
- Reduce `--mem-fraction-static`
- Enable `--chunked-prefill-size 4096`
- Reduce `--max-prefill-tokens`

---

### OOM During Decode

```
ERROR:    CUDA out of memory in forward_extend
ERROR:    Cannot allocate new KV cache pages
```

**Root Cause**: Too many running requests, KV cache exhausted

**Solutions**:
- Reduce `--max-running-requests`
- Increase `--mem-fraction-static` (if prefill allows)

---

### NCCL Timeout

```
ERROR:    [Rank 1] Watchdog caught collective that timed out after 300 seconds
ERROR:    Aborting process group due to timeout
```

**Root Cause**:
- Network connectivity loss
- Deadlock (rare)
- One node slower/hung

**Solutions**:
- Check network connectivity
- Restart all nodes
- Increase `--dist-timeout 1800`
- Add `--disable-cuda-graph` (if deadlock suspected)

---

### PD Transfer Timeout (Prefill)

```
ERROR:    [Prefill] Bootstrap timeout waiting for decode indices
ERROR:    Request req_abc123 timed out after 300s
```

**Root Cause**:
- Decode node disconnected or slow
- Network congestion
- Decode node overloaded

**Solutions**:
- Increase `SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=600`
- Check decode node health
- Verify IB connectivity

---

### PD Transfer Timeout (Decode)

```
ERROR:    [Decode] KV cache transfer timeout for request req_abc123
ERROR:    Waited 300s but no data received
```

**Root Cause**:
- Prefill node disconnected or slow
- KV transfer queue congested
- Network issues

**Solutions**:
- Increase `SGLANG_DISAGGREGATION_WAITING_TIMEOUT=600`
- Check prefill node health
- Adjust `SGLANG_DISAGGREGATION_THREAD_POOL_SIZE`

---

### IB Device Inactive

```
ERROR:    IB device mlx5_0 is not in ACTIVE state
ERROR:    Current state: PORT_DOWN
```

**Root Cause**: InfiniBand link down or not configured

**Solutions**:
```bash
# Restart IB services
systemctl restart openibd

# Check cable/switch
ibv_devinfo mlx5_0
```

---

## Performance Indicators from Logs

### Good Performance

```
INFO:     Decode batch. #running-req: 512, #token: 25600, token usage: 0.88, gen throughput (token/s): 12000
INFO:     Decode batch. #running-req: 508, #token: 25400, token usage: 0.87, gen throughput (token/s): 11950
```

**Indicators**:
- High `#running-req` (close to max)
- High `token usage` (>0.8)
- Stable throughput (low variance)
- Low/zero `#queue-req`

---

### Poor Performance

```
INFO:     Decode batch. #running-req: 45, #token: 2500, token usage: 0.15, gen throughput (token/s): 1200, #queue-req: 200
WARN:     Retracting 5 requests due to memory pressure
```

**Indicators**:
- Low `#running-req` (far below max)
- Low `token usage` (<0.5)
- High `#queue-req` (backlog)
- Frequent retracts

**Likely Causes**:
- Insufficient load (increase request rate)
- Memory pressure causing retracts (tune memory params)
- Prefill bottleneck (use PD disaggregation)

---

## Calculating Metrics from Logs

### TTFT from Logs (Manual)

```
INFO:     Request received. ID: req_abc123, time: 10:00:00.000
INFO:     Request completed. ID: req_abc123, time: 10:00:03.450, TTFT: 98ms
```

**TTFT**: 98ms (from "TTFT" field)

---

### Throughput from Decode Logs

```
INFO:     Decode batch. gen throughput (token/s): 8234
INFO:     Decode batch. gen throughput (token/s): 8156
INFO:     Decode batch. gen throughput (token/s): 8301
```

**Average**: (8234 + 8156 + 8301) / 3 = 8230 tok/s

---

### Request Success Rate

```
INFO:     Requests completed: 950
ERROR:    Requests failed: 50
```

**Success rate**: 950 / 1000 = 95%

---

## Debugging Workflow

### Step 1: Identify Issue from Logs

```bash
# Check for errors
grep ERROR server.log | tail -20

# Check decode batch stats
grep "Decode batch" server.log | tail -10

# Check retracts
grep "Retracting" server.log | wc -l
```

---

### Step 2: Analyze Patterns

```bash
# Extract throughput over time
grep "gen throughput" server.log | awk '{print $(NF-1)}' > throughput.csv

# Count OOM errors
grep "out of memory" server.log | wc -l

# Check request latency distribution
grep "Request completed" server.log | awk '{print $NF}' | sort -n
```

---

### Step 3: Correlate with Metrics

```bash
# Prometheus metrics (if enabled)
curl http://localhost:30000/metrics | grep sglang_

# Server info
curl http://localhost:30000/get_server_info | jq '.kv_cache_utilization'
```

---

**End of LOG_GLOSSARY memory**