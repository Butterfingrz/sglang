# SGLang Data Pipeline & Benchmarking Datasets

**Last Verified:** 2025-10-01  
**Applicable Branch:** main (commit bfa27438)

---

## Table of Contents
1. [Dataset Sources](#dataset-sources)
2. [Data Preparation](#data-preparation)
3. [Benchmark Data Formats](#benchmark-data-formats)
4. [Log Output & Analysis](#log-output--analysis)

---

## Dataset Sources

### Built-in Synthetic Datasets

| Dataset | Source | Generation | Use Case |
|---------|--------|------------|----------|
| **random** | In-memory | Token sampling from vocab | Controlled length testing |
| **random-ids** | In-memory | Random token IDs | Strict length control |
| **random-image** | In-memory | PIL-generated images | VLM benchmarking |
| **generated-shared-prefix** | In-memory | Long system prompt + short questions | RadixCache testing |

**No external data required** - generated on-the-fly during benchmark run.

### Real-World Datasets

| Dataset | Source | Download | Records | Use Case |
|---------|--------|----------|---------|----------|
| **ShareGPT** | HuggingFace | Auto-download on first run | ~90K | Conversational workload |
| **MMMU** | HuggingFace | `datasets` library | ~11K | Multimodal reasoning |
| **Mooncake traces** | Custom | `--mooncake-workload` | Variable | KV cache sharing patterns |

### ShareGPT Data

```bash
# Auto-downloaded to: ~/.cache/sglang/ShareGPT_V3_unfiltered_cleaned_split.json
python -m sglang.bench_serving \
  --dataset-name sharegpt \
  --num-prompts 1000

# Custom ShareGPT file:
python -m sglang.bench_serving \
  --dataset-name sharegpt \
  --dataset-path /path/to/custom_sharegpt.json
```

**Format**:
```json
[
  {
    "conversations": [
      {"from": "human", "value": "Hello"},
      {"from": "gpt", "value": "Hi there!"}
    ]
  },
  ...
]
```

---

## Data Preparation

### Tokenization

```python
# Tokenizer is loaded based on --model or --tokenizer
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")

# Chat template applied when --apply-chat-template is set
messages = [{"role": "user", "content": "Hello"}]
prompt = tokenizer.apply_chat_template(messages, tokenize=False)
```

### Data Filtering

```bash
# ShareGPT: filter by context length
python -m sglang.bench_serving \
  --dataset-name sharegpt \
  --sharegpt-context-len 2048 \
  --num-prompts 500

# Random: control length distribution
python -m sglang.bench_serving \
  --dataset-name random \
  --random-input-len 1024 \
  --random-output-len 256 \
  --random-range-ratio 0.5  # ±50% variation
```

### Data Mixing (Manual)

```python
# Custom benchmark script combining datasets
import json

sharegpt_data = load_sharegpt()
random_data = generate_random(1000)

mixed = sharegpt_data[:500] + random_data[:500]
run_benchmark(mixed)
```

---

## Benchmark Data Formats

### Input Format (HTTP API)

**Native `/generate`**:
```json
{
  "text": "Once upon a time",
  "sampling_params": {
    "max_new_tokens": 256,
    "temperature": 1.0
  }
}
```

**OpenAI `/v1/completions`**:
```json
{
  "model": "meta-llama/Llama-3.1-8B-Instruct",
  "prompt": "Once upon a time",
  "max_tokens": 256,
  "temperature": 1.0,
  "stream": true
}
```

**OpenAI `/v1/chat/completions`**:
```json
{
  "model": "meta-llama/Llama-3.1-8B-Instruct",
  "messages": [
    {"role": "user", "content": "Hello"}
  ],
  "max_tokens": 256,
  "stream": true
}
```

### Output Format (JSONL)

**Single line per benchmark run**:
```jsonl
{"backend":"sglang","dataset_name":"random","num_prompts":3000,"duration":125.3,"total_output_throughput":6130.8,"ttft_median":98.2}
```

**With `--output-details`**:
```jsonl
{"backend":"sglang","dataset_name":"random","num_prompts":3000,"input_lens":[1024,980,1100],"output_lens":[256,245,270],"ttfts":[98.2,105.1,89.3],"generated_texts":["...","...","..."]}
```

---

## Log Output & Analysis

### Server Logs Location

```bash
# Stdout/stderr (default)
python -m sglang.launch_server --model-path ... 2>&1 | tee server.log

# Systemd journal
journalctl -u sglang -f

# Custom log file
python -m sglang.launch_server --model-path ... > /var/log/sglang/server.log 2>&1
```

### Benchmark Output Files

```bash
# Auto-named (timestamp)
python -m sglang.bench_serving \
  --backend sglang \
  --num-prompts 1000
# Output: benchmark_results_<timestamp>.jsonl

# Custom name
python -m sglang.bench_serving \
  --backend sglang \
  --output-file my_benchmark.jsonl
```

### Log Parsing Examples

**Extract throughput over time**:
```bash
grep "Decode batch" server.log | \
  awk '{print $1, $NF}' | \
  sed 's/,//g' > throughput_timeseries.csv
```

**Analyze TTFT distribution**:
```python
import json
import numpy as np

with open('benchmark.jsonl') as f:
    data = json.load(f)

ttfts = data['ttfts']
print(f"TTFT P50: {np.percentile(ttfts, 50):.2f}ms")
print(f"TTFT P99: {np.percentile(ttfts, 99):.2f}ms")
```

**Compare baselines**:
```bash
# Compare two benchmark runs
python -c "
import json

with open('baseline.jsonl') as f:
    baseline = json.load(f)
with open('current.jsonl') as f:
    current = json.load(f)

print(f'Throughput change: {(current[\"total_output_throughput\"] / baseline[\"total_output_throughput\"] - 1) * 100:.2f}%')
print(f'TTFT change: {(current[\"ttft_median\"] / baseline[\"ttft_median\"] - 1) * 100:.2f}%')
"
```

---

## Data Visualization (Optional)

### Plot Throughput

```python
import json
import matplotlib.pyplot as plt

results = []
for file in ['tp1.jsonl', 'tp2.jsonl', 'tp4.jsonl', 'tp8.jsonl']:
    with open(f'scaling/{file}') as f:
        results.append(json.load(f))

tp_sizes = [1, 2, 4, 8]
throughputs = [r['total_output_throughput'] for r in results]

plt.plot(tp_sizes, throughputs, marker='o')
plt.xlabel('TP Size')
plt.ylabel('Output Throughput (tok/s)')
plt.title('Scaling with Tensor Parallelism')
plt.grid(True)
plt.savefig('tp_scaling.png')
```

---

## Data Retention & Storage

### Benchmark Results

```bash
# Recommended structure
benchmark_results/
├── baselines/
│   ├── llama31_8b_tp1.jsonl
│   ├── llama31_70b_tp8.jsonl
│   └── deepseek_v3_4p9d.jsonl
├── experiments/
│   ├── 2025-10-01_radix_ablation/
│   │   ├── radix_on.jsonl
│   │   └── radix_off.jsonl
│   └── 2025-10-05_chunked_prefill/
│       ├── chunk_4k.jsonl
│       ├── chunk_8k.jsonl
│       └── chunk_16k.jsonl
└── regression/
    └── nightly_2025-10-01.jsonl
```

### Server Logs

```bash
# Rotate logs daily
logrotate /etc/logrotate.d/sglang

# Example logrotate config
/var/log/sglang/*.log {
    daily
    rotate 7
    compress
    missingok
    notifempty
}
```

---

**End of DATA_PIPELINE memory**