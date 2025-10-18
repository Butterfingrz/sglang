# SGLang Terminology Glossary

**Last Verified:** 2025-10-01  
**Applicable Branch:** main (commit bfa27438)

---

## Core Concepts

### Prefill
The first phase of LLM inference where the entire input prompt is processed in parallel. Computes KV cache for all input tokens. Compute-intensive, high FLOPS utilization.

### Decode
The second phase of LLM inference where tokens are generated one at a time autoregressively. Memory-bandwidth intensive, lower FLOPS utilization. Also called "generation phase."

### PD Disaggregation
**Prefill-Decode Disaggregation**: Architectural pattern where prefill and decode run on separate GPU clusters. Avoids prefill interrupting decode, enables independent scaling. Requires KV cache transfer via RDMA.

### RadixAttention
SGLang's prefix caching mechanism using a radix tree data structure. Automatically detects and reuses KV cache for shared prompt prefixes. Significantly reduces TTFT for repeated patterns (e.g., system prompts).

### KV Cache
**Key-Value Cache**: Cached attention keys and values from previously processed tokens. Avoids recomputation during autoregressive generation. Memory footprint grows linearly with sequence length and batch size.

---

## Parallelism

### TP (Tensor Parallelism)
Shards model weights across GPUs within a single layer. Requires all-reduce communication after each matmul. Best for large models that don't fit on single GPU.

**Example**: TP=8 splits 70B model across 8 GPUs (~9GB per GPU)

### DP (Data Parallelism)
Replicates full model across GPUs. Each worker processes different requests independently. No inter-GPU communication during forward pass (except with DP Attention).

**Example**: DP=4 runs 4 independent copies of 8B model

### EP (Expert Parallelism)
For Mixture-of-Experts (MoE) models, distributes experts across GPUs. Requires all-to-all communication for expert routing. Used with DeepSeek-V3, Qwen MoE.

**Example**: EP=8 distributes 160 experts across 8 GPUs (~20 per GPU)

### PP (Pipeline Parallelism)
Splits model layers across GPUs vertically. GPU N processes layer N. Requires point-to-point transfers between stages. Less commonly used in SGLang.

### DP Attention
Hybrid parallelism: Data parallelism for attention computation, Tensor parallelism for FFN/MoE layers. Requires `dp_size == tp_size`. Improves expert utilization in MoE models.

**Example**: DeepSeek-V3 with TP=16, DP=8 (128 GPUs total)

---

## Performance Metrics

### TTFT (Time to First Token)
Latency from request submission to first generated token. Dominated by prefill time. Target: <100ms P50, <500ms P99.

**Units**: Milliseconds (ms)

### ITL (Inter-Token Latency)
Time between consecutive generated tokens during decode. Reflects decode batch efficiency. Target: <20ms P50, <50ms P99.

**Units**: Milliseconds (ms)

### TPOT (Time Per Output Token)
Average time per generated token excluding first token: `(E2E_latency - TTFT) / (output_tokens - 1)`. Similar to ITL but averaged over entire sequence.

**Units**: Milliseconds (ms)

### Throughput (Token/s)
Number of tokens processed per second. Separate metrics for input and output throughput.

**Units**: Tokens per second (tok/s)

### Goodput
Effective throughput accounting for failed/retried requests. `successful_tokens / total_wall_time`.

### Concurrency
Average number of parallel in-flight requests. `sum(request_latencies) / total_duration`.

---

## Memory & Scheduling

### Paged Attention
Memory management technique that allocates KV cache in fixed-size pages (typically 1 token per page in SGLang). Reduces fragmentation compared to contiguous allocation.

### Chunked Prefill
Splits large prefill operations into smaller chunks to reduce memory peak. Each chunk processes a subset of input tokens. Trades latency for memory efficiency.

**Flag**: `--chunked-prefill-size 8192`

### Retract
Scheduler evicts a running request from decode batch to free KV cache memory. Request will be rescheduled and re-prefilled later. High retract rate indicates memory pressure.

### Continuous Batching
Dynamic batching technique where new requests join decode batch as soon as prior requests finish, without waiting for batch boundary. Maximizes GPU utilization.

---

## Communication

### NCCL (NVIDIA Collective Communications Library)
Standard library for multi-GPU communication primitives (all-reduce, all-gather, etc.). Used for TP, EP, PP.

### All-Reduce
Collective operation that sums values across all GPUs and broadcasts result. Used in TP after matmul layers.

### All-to-All
Collective operation for scatter-gather patterns. Used in MoE expert routing. DeepEP provides optimized implementation.

### RDMA (Remote Direct Memory Access)
Network technology for zero-copy GPU-to-GPU transfers across nodes. Used in PD disaggregation for KV cache transfer. Requires InfiniBand or RoCE.

### IB (InfiniBand)
High-performance interconnect standard. Common speeds: HDR (200 Gb/s), NDR (400 Gb/s). Required for PD disaggregation with Mooncake.

### NVLink
NVIDIA's high-speed GPU-to-GPU interconnect. 900 GB/s per link on H100/H200. Enables efficient TP within a node.

---

## Backends & Kernels

### FlashInfer
Default attention kernel backend in SGLang. Highly optimized for both prefill (FlashAttention) and decode. Supports paged attention, MLA (Multi-Head Latent Attention).

### Triton
Alternative attention backend using Triton language (Python-like GPU programming). Slower than FlashInfer but more flexible.

### DeepEP
Optimized all-to-all communication library for MoE models. Requires Hopper GPUs (H100/H200) and NVSHMEM. Significantly faster than NCCL for expert routing.

### CUDA Graph
CUDA feature that captures and replays kernel sequences. Reduces kernel launch overhead. Enabled by default in SGLang for common batch sizes.

---

## Model Types

### MoE (Mixture of Experts)
Architecture where each layer contains multiple "expert" sub-networks. Router selects top-K experts per token. Examples: DeepSeek-V3 (256 experts), Mixtral (8 experts).

### MLA (Multi-Head Latent Attention)
Attention variant in DeepSeek models that compresses KV cache. Reduces memory footprint compared to standard multi-head attention.

### GQA (Grouped Query Attention)
Attention variant where multiple query heads share key/value heads. Reduces KV cache size. Used in Llama 3.x, Qwen 2.5.

---

## Router & Load Balancing

### Router
SGLang component that distributes requests across multiple backend workers. Supports PD disaggregation coordination, load balancing policies, health checks.

**Binary**: `python -m sglang_router.launch_router`

### Load Balancing Policies

**Round Robin**: Distributes requests evenly across workers in order. Simple, no state.

**Minimum Tokens**: Routes to worker with least KV cache tokens (lowest memory usage). Best for DP attention scenarios.

---

## Quantization

### FP16 (Float16)
16-bit floating point. Default precision for most models. Good balance of speed and accuracy.

### BF16 (BFloat16)
16-bit format with same exponent range as FP32. Better numerical stability than FP16. Preferred on Hopper GPUs.

### FP8 (Float8)
8-bit floating point. 2× memory reduction vs. FP16. Requires H100/H200. E4M3 (more precision) or E5M2 (wider range).

**Flags**: `--quantization fp8`, `--kv-cache-dtype fp8_e5m2`

### INT4/INT8
Integer quantization. Reduces model size and memory. Requires calibration for weight quantization. TorchAO provides implementation.

**Flags**: `--torchao-config int4wo-128`

### AWQ/GPTQ
Weight-only quantization techniques. Pre-quantized model checkpoints available on HuggingFace. INT4 weights, FP16 activations.

---

## Transfer Engines (PD)

### Mooncake
Default PD transfer engine. InfiniBand/RDMA-based. Developed by Moonshot AI. Best performance and stability.

**Install**: `uv pip install mooncake-transfer-engine`

### NIXL
Alternative PD transfer engine. UCX-based. Developed by Dynamo AI.

**Install**: `pip install nixl`

### ASCEND
PD transfer engine for Huawei Ascend NPUs. China region only.

---

## Optimization Techniques

### Speculative Decoding
Technique where a smaller "draft" model proposes tokens, verified by larger "target" model. Can improve decode throughput if acceptance rate is high. Experimental in SGLang.

### Torch Compile
PyTorch JIT compilation. Optimizes model execution by fusing operations. Cold start penalty but improves steady-state for small models.

**Flag**: `--enable-torch-compile`

### Two-Batch Overlap
Experimental technique to overlap prefill and decode microbatches on same GPU. Can reduce TTFT but has stability issues.

**Flag**: `--enable-two-batch-overlap` (⚠️ not recommended)

---

## Comparison to Other Frameworks

### SGLang vs. vLLM

| Feature | SGLang | vLLM |
|---------|--------|------|
| **Prefix Caching** | RadixAttention (automatic) | Prefix caching (manual) |
| **PD Disaggregation** | Native support | Not supported |
| **MoE Optimization** | DeepEP integration | Standard NCCL |
| **Frontend DSL** | Yes (Python) | No |
| **Maturity** | Newer, fast iteration | More mature |

### SGLang vs. TensorRT-LLM

| Feature | SGLang | TensorRT-LLM |
|---------|--------|--------------|
| **Ease of Use** | Simple, Python-first | Complex C++ build |
| **Model Support** | HuggingFace direct | Engine conversion required |
| **Performance** | Competitive | Slightly faster (optimized models) |
| **Flexibility** | High | Low (compiled engines) |

---

## Common Abbreviations

| Term | Full Name | Context |
|------|-----------|---------|
| **TP** | Tensor Parallelism | Multi-GPU sharding |
| **DP** | Data Parallelism | Worker replication |
| **EP** | Expert Parallelism | MoE distribution |
| **PP** | Pipeline Parallelism | Layer-wise sharding |
| **PD** | Prefill-Decode | Disaggregation architecture |
| **TTFT** | Time to First Token | Latency metric |
| **ITL** | Inter-Token Latency | Decode latency |
| **TPOT** | Time Per Output Token | Average token time |
| **KV** | Key-Value | Attention cache |
| **MoE** | Mixture of Experts | Model architecture |
| **MLA** | Multi-Head Latent Attention | DeepSeek attention |
| **GQA** | Grouped Query Attention | Llama 3 attention |
| **IB** | InfiniBand | Network interconnect |
| **RDMA** | Remote Direct Memory Access | Zero-copy transfer |
| **OOM** | Out of Memory | CUDA error |

---

## Related Projects

### FlashInfer
Kernel library providing optimized attention implementations. Core dependency of SGLang.

**GitHub**: https://github.com/flashinfer-ai/flashinfer

### Transformers
HuggingFace library for model loading and tokenization. Used by SGLang for model support.

**GitHub**: https://github.com/huggingface/transformers

### DeepEP
Optimized collective communication library for MoE models by DeepSeek.

**GitHub**: https://github.com/deepseek-ai/DeepEP

### Mooncake
Transfer engine for PD disaggregation by Moonshot AI.

**PyPI**: `mooncake-transfer-engine`

---

**End of GLOSSARY memory**