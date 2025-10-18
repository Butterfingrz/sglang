# SGLang Testing & Benchmarking Guide

**Last Verified:** 2025-10-01  
**Applicable Branch:** main (commit bfa27438)  
**Tool:** `python -m sglang.bench_serving`

---

## Table of Contents
1. [Benchmark Tool Overview](#benchmark-tool-overview)
2. [Dataset Options](#dataset-options)
3. [Metrics Explanation](#metrics-explanation)
4. [Common Benchmark Scenarios](#common-benchmark-scenarios)
5. [Regression Testing](#regression-testing)
6. [GPU/Node Scaling](#gpunode-scaling)

---

## Benchmark Tool Overview

### Command Structure

```bash
python -m sglang.bench_serving \
  --backend <BACKEND> \
  --host <HOST> --port <PORT> \
  --model <MODEL> \
  --dataset-name <DATASET> \
  --num-prompts <N> \
  --request-rate <RATE> \
  --max-concurrency <CONCURRENCY>
```

### Supported Backends

| Backend | Endpoint | Notes |
|---------|----------|-------|
| `sglang` | `/generate` | Native SGLang (recommended) |
| `sglang-native` | `/generate` | Alias for sglang |
| `sglang-oai` | `/v1/completions` | OpenAI-compatible |
| `sglang-oai-chat` | `/v1/chat/completions` | Chat endpoint |
| `vllm` | `/v1/completions` | vLLM comparison |
| `vllm-chat` | `/v1/chat/completions` | vLLM chat |
| `lmdeploy` | `/v1/completions` | LMDeploy comparison |
| `trt` | `/v2/models/ensemble/generate_stream` | TensorRT-LLM |

---

## Dataset Options

### 1. ShareGPT (Default)

Real conversational data from ShareGPT.

```bash
python -m sglang.bench_serving \
  --backend sglang \
  --dataset-name sharegpt \
  --num-prompts 1000 \
  --sharegpt-output-len 256
```

**Flags**:
- `--sharegpt-context-len`: Filter by max context length
- `--sharegpt-output-len`: Override output lengths
- `--dataset-path`: Custom ShareGPT JSON file

### 2. Random Dataset

Synthetic prompts with controlled lengths.

```bash
python -m sglang.bench_serving \
  --backend sglang \
  --dataset-name random \
  --num-prompts 3000 \
  --random-input-len 1024 \
  --random-output-len 1024 \
  --random-range-ratio 0.5
```

**Flags**:
- `--random-input-len`: Mean input tokens
- `--random-output-len`: Mean output tokens
- `--random-range-ratio`: Length variation (0.0-1.0)

**Use Case**: Controlled experiments, ablation studies

### 3. Random-IDs Dataset

Token IDs directly (no tokenization).

```bash
python -m sglang.bench_serving \
  --backend sglang \
  --dataset-name random-ids \
  --tokenize-prompt \
  --random-input-len 2048 \
  --random-output-len 256
```

**Use Case**: Strict length control, bypass tokenization overhead

### 4. Random-Image Dataset

Synthetic images for VLM benchmarking.

```bash
python -m sglang.bench_serving \
  --backend sglang-oai-chat \
  --dataset-name random-image \
  --num-prompts 500 \
  --random-image-num-images 3 \
  --random-image-resolution 720p \
  --random-input-len 512 \
  --random-output-len 512
```

**Resolutions**:
- Presets: `360p`, `720p`, `1080p`
- Custom: `512x768`, `1920x1080` (heightxwidth)

**Use Case**: LLaVA, Qwen2-VL, Phi-4 multimodal models

### 5. Generated-Shared-Prefix

Long system prompts + short questions (prefix caching test).

```bash
python -m sglang.bench_serving \
  --backend sglang \
  --dataset-name generated-shared-prefix \
  --gsp-num-groups 64 \
  --gsp-prompts-per-group 16 \
  --gsp-system-prompt-len 2048 \
  --gsp-question-len 128 \
  --gsp-output-len 256
```

**Use Case**: RadixAttention effectiveness, KV cache hit rate

### 6. MMMU Dataset

Real multimodal reasoning tasks (Math split).

```bash
python -m sglang.bench_serving \
  --backend sglang-oai-chat \
  --dataset-name mmmu \
  --num-prompts 200
```

**Requires**: `datasets`, `pillow` packages

---

## Metrics Explanation

### Core Metrics

| Metric | Unit | Description | Good Value |
|--------|------|-------------|------------|
| **Request Throughput** | req/s | Completed requests per second | Depends on workload |
| **Input Token Throughput** | tok/s | Total input tokens / duration | Higher is better |
| **Output Token Throughput** | tok/s | Generated tokens / duration | Key metric for decode |
| **Total Token Throughput** | tok/s | (Input + Output) / duration | Overall efficiency |
| **Concurrency** | - | Avg parallel requests in flight | Should match `--max-concurrency` |

### Latency Metrics

| Metric | Unit | Description | P50 Target | P99 Target |
|--------|------|-------------|------------|------------|
| **TTFT** | ms | Time to first token | <100ms | <500ms |
| **ITL** | ms | Inter-token latency | <20ms | <50ms |
| **E2E Latency** | ms | End-to-end per request | <3000ms | <10000ms |
| **TPOT** | ms | Time per output token | <20ms | <40ms |

**TPOT Formula**: `(E2E_latency - TTFT) / (output_tokens - 1)`

### Specialized Metrics

| Metric | Source | Meaning |
|--------|--------|---------|
| **Accept Length** | sglang only | Speculative decoding acceptance rate |
| **Retokenized Counts** | All | Verify output token counts via tokenizer |

### Percentiles Reported

- **Mean**: Average across all requests
- **Median (P50)**: 50th percentile
- **P95**: 95th percentile
- **P99**: 99th percentile
- **Max**: Worst-case latency
- **Std**: Standard deviation

---

## Common Benchmark Scenarios

### Scenario 1: Baseline Throughput Test

**Goal**: Maximum sustainable throughput

```bash
python -m sglang.bench_serving \
  --backend sglang \
  --host 127.0.0.1 --port 30000 \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --dataset-name random \
  --random-input-len 1024 \
  --random-output-len 256 \
  --random-range-ratio 0.5 \
  --num-prompts 3000 \
  --request-rate inf \
  --max-concurrency 512
```

**Expected** (H100, TP=1):
- Output Throughput: 8000-12000 tok/s
- Request Throughput: 40-50 req/s
- TTFT P50: 50-150ms

### Scenario 2: Latency-Sensitive (Online)

**Goal**: Low TTFT, controlled concurrency

```bash
python -m sglang.bench_serving \
  --backend sglang \
  --host 127.0.0.1 --port 30000 \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --dataset-name random \
  --random-input-len 512 \
  --random-output-len 128 \
  --num-prompts 1000 \
  --request-rate 50 \
  --max-concurrency 16
```

**Expected**:
- TTFT P50: <30ms
- TTFT P99: <100ms
- ITL P50: <15ms

### Scenario 3: Long Context (Needle-in-Haystack)

**Goal**: Test scaling with large inputs

```bash
python -m sglang.bench_serving \
  --backend sglang \
  --host 127.0.0.1 --port 30000 \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --dataset-name random \
  --random-input-len 32768 \
  --random-output-len 256 \
  --random-range-ratio 0.1 \
  --num-prompts 100 \
  --request-rate 10 \
  --max-concurrency 4
```

**Server Flags**: `--chunked-prefill-size 8192 --context-length 32768`

### Scenario 4: RadixCache Effectiveness

**Goal**: Measure prefix caching hit rate

```bash
# WITH RadixCache (default)
python -m sglang.bench_serving \
  --backend sglang \
  --dataset-name generated-shared-prefix \
  --gsp-num-groups 64 \
  --gsp-prompts-per-group 16 \
  --gsp-system-prompt-len 4096 \
  --gsp-question-len 64 \
  --gsp-output-len 128 \
  --num-prompts 1024 \
  --output-file radix_on.jsonl

# WITHOUT RadixCache
# Restart server with: --disable-radix-cache
python -m sglang.bench_serving \
  --backend sglang \
  --dataset-name generated-shared-prefix \
  --gsp-num-groups 64 \
  --gsp-prompts-per-group 16 \
  --gsp-system-prompt-len 4096 \
  --gsp-question-len 64 \
  --gsp-output-len 128 \
  --num-prompts 1024 \
  --output-file radix_off.jsonl

# Compare TTFT: should be ~10-50x faster with RadixCache
```

### Scenario 5: PD Disaggregation Comparison

**Goal**: Compare unified vs. disaggregated

```bash
# Unified (baseline)
python -m sglang.bench_serving \
  --backend sglang \
  --host 127.0.0.1 --port 30000 \
  --dataset-name random \
  --random-input-len 1024 \
  --random-output-len 512 \
  --num-prompts 2000 \
  --request-rate 100 \
  --output-file unified.jsonl

# PD Disaggregated (via router)
python -m sglang.bench_serving \
  --backend sglang \
  --base-url http://127.0.0.1:8000 \
  --dataset-name random \
  --random-input-len 1024 \
  --random-output-len 512 \
  --num-prompts 2000 \
  --request-rate 100 \
  --output-file pd_disagg.jsonl

# Expect: Lower TTFT variance, higher throughput with PD
```

---

## Regression Testing

### Baseline Collection

```bash
# Establish baselines for each model/config
python -m sglang.bench_serving \
  --backend sglang \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --dataset-name random \
  --random-input-len 1024 --random-output-len 256 \
  --num-prompts 3000 \
  --request-rate inf --max-concurrency 512 \
  --output-file baselines/llama31_8b_baseline.jsonl \
  --output-details
```

### Regression Check Script

```bash
#!/bin/bash
# regression_check.sh

BASELINE_FILE="baselines/llama31_8b_baseline.jsonl"
NEW_FILE="results/llama31_8b_current.jsonl"

# Extract key metrics
BASELINE_THROUGHPUT=$(jq -r '.total_output_throughput' $BASELINE_FILE)
NEW_THROUGHPUT=$(jq -r '.total_output_throughput' $NEW_FILE)

# Calculate percentage change
CHANGE=$(awk "BEGIN {print ($NEW_THROUGHPUT - $BASELINE_THROUGHPUT) / $BASELINE_THROUGHPUT * 100}")

echo "Baseline Throughput: $BASELINE_THROUGHPUT tok/s"
echo "Current Throughput:  $NEW_THROUGHPUT tok/s"
echo "Change: $CHANGE%"

# Fail if regression > 5%
if (( $(echo "$CHANGE < -5" | bc -l) )); then
  echo "❌ REGRESSION DETECTED"
  exit 1
else
  echo "✓ No regression"
fi
```

### Key Baselines to Track

| Model | Config | Input | Output | Baseline Throughput (H100) |
|-------|--------|-------|--------|----------------------------|
| Llama-3.1-8B | TP=1 | 1024 | 256 | 10,000 tok/s |
| Llama-3.1-70B | TP=8 | 1024 | 256 | 8,000 tok/s |
| DeepSeek-V3 | TP=16, EP | 2048 | 256 | 15,000 tok/s (4P-9D) |

---

## GPU/Node Scaling

### Single GPU Sweep

```bash
for BS in 1 2 4 8 16 32 64 128; do
  python -m sglang.bench_serving \
    --backend sglang \
    --dataset-name random \
    --random-input-len 512 --random-output-len 128 \
    --num-prompts 1000 \
    --max-concurrency $BS \
    --output-file scaling/single_gpu_bs${BS}.jsonl
done
```

### Multi-GPU Scaling (TP)

```bash
for TP in 1 2 4 8; do
  # Restart server with --tp-size $TP
  python -m sglang.launch_server \
    --model-path meta-llama/Llama-3.1-70B-Instruct \
    --tp-size $TP --port 30000 &
  
  sleep 120  # Wait for warmup
  
  python -m sglang.bench_serving \
    --backend sglang \
    --dataset-name random \
    --random-input-len 1024 --random-output-len 256 \
    --num-prompts 2000 \
    --output-file scaling/tp${TP}.jsonl
  
  pkill -f launch_server
done
```

### PD Scaling (Variable Decode Workers)

```bash
# Fixed prefill: 2 nodes
# Variable decode: 2, 4, 8, 16 nodes

for D in 2 4 8 16; do
  # Launch decode nodes (script omitted for brevity)
  # Update router with new decode endpoints
  
  python -m sglang.bench_serving \
    --backend sglang \
    --base-url http://router:8000 \
    --dataset-name random \
    --random-input-len 2048 --random-output-len 512 \
    --num-prompts 5000 \
    --request-rate 200 \
    --output-file scaling/2p${D}d.jsonl
done
```

---

## Advanced Benchmarking

### Warmup & Cache Flush

```bash
# Cold start (flush cache before run)
python -m sglang.bench_serving \
  --backend sglang \
  --flush-cache \
  --warmup-requests 10 \
  --dataset-name random \
  --num-prompts 1000
```

### Profiling Integration

```bash
# Server: export SGLANG_TORCH_PROFILER_DIR=/tmp/profiles
python -m sglang.bench_serving \
  --backend sglang \
  --profile \
  --dataset-name random \
  --num-prompts 100

# Profiles saved to: /tmp/profiles/*.pt.trace.json
```

### LoRA Benchmarking

```bash
# Server: --enable-lora --lora-paths adapter1=/path/to/adapter1 adapter2=/path/to/adapter2
python -m sglang.bench_serving \
  --backend sglang \
  --lora-name adapter1 adapter2 \
  --dataset-name random \
  --num-prompts 1000
```

---

## Troubleshooting Benchmarks

### All Requests Failed

**Check**:
1. Server is running: `curl http://host:port/health`
2. Model name matches: `curl http://host:port/v1/models`
3. Authentication: `export OPENAI_API_KEY=...`

### Low Throughput

**Check**:
1. `--request-rate` too conservative → increase or set to `inf`
2. `--max-concurrency` too low → increase to 256-512
3. Server batch size limits → check `--max-running-requests`

### High TTFT Variance

**Likely Causes**:
- Prefill interrupting decode (use PD disaggregation)
- Insufficient memory (reduce `--mem-fraction-static`)
- DP attention imbalance (enable `--enable-dp-attention`)

### Token Count Mismatch

**Check**:
- Ensure `--model` matches server model
- Verify tokenizer config (especially for custom models)
- Use `--tokenize-prompt` for exact control

---

## Output File Format (JSONL)

```json
{
  "backend": "sglang",
  "dataset_name": "random",
  "num_prompts": 3000,
  "request_rate": "inf",
  "max_concurrency": 512,
  "duration": 125.3,
  "total_input_tokens": 3072000,
  "total_output_tokens": 768000,
  "total_input_throughput": 24523.1,
  "total_output_throughput": 6130.8,
  "total_token_throughput": 30653.9,
  "request_throughput": 23.9,
  "concurrency": 12.4,
  "latency_mean": 4231.2,
  "latency_median": 3890.5,
  "latency_p99": 8450.1,
  "ttft_mean": 123.4,
  "ttft_median": 98.2,
  "ttft_p99": 456.7,
  "itl_mean": 15.3,
  "itl_median": 14.1,
  "itl_p99": 32.8
}
```

With `--output-details`:
```json
{
  ...,
  "input_lens": [1024, 980, 1100, ...],
  "output_lens": [256, 245, 270, ...],
  "ttfts": [98.2, 105.1, 89.3, ...],
  "generated_texts": ["...", "...", ...],
  "errors": [null, null, ...]
}
```

---

**End of TESTING memory**