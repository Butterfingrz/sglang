# SGLang Compatibility Matrix

**Last Verified:** 2025-10-01  
**Applicable Branch:** main (commit bfa27438)  
**Current Version:** 0.5.3rc0

---

## Table of Contents
1. [Core Dependencies](#core-dependencies)
2. [Verified Configurations](#verified-configurations)
3. [DeepEP Compatibility](#deepep-compatibility)
4. [PD Disaggregation Backends](#pd-disaggregation-backends)
5. [Model Compatibility](#model-compatibility)

---

## Core Dependencies

### Python & PyTorch

| Component | Minimum | Recommended | Latest Tested | Notes |
|-----------|---------|-------------|---------------|-------|
| **Python** | 3.10 | 3.12 | 3.12 | Python 3.13 not supported |
| **PyTorch** | 2.5.0 | 2.8.0 | 2.8.0 | Must match CUDA version |
| **CUDA** | 12.1 | 12.8.1, 12.9.1 | 12.9.1 | Driver ≥555 for 12.9 |
| **cuDNN** | 8.9 | 9.0+ | 9.1.0 | Bundled in PyTorch |

**Installation**:
```bash
# CUDA 12.8
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128

# CUDA 12.9 (recommended)
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu129
```

---

### SGLang Ecosystem

| Package | Version | Purpose | Installation |
|---------|---------|---------|--------------|
| **sglang** | 0.5.3rc0 | Core framework | `pip install sglang[all]` |
| **sgl-kernel** | 0.3.13 | CUDA kernels | Auto-installed |
| **sgl-router** | 0.5.3rc0 | Load balancer | Bundled |
| **flashinfer** | 0.4.0rc3 | Attention kernels | Auto-installed |
| **transformers** | 4.56.1 | Model loading | Auto-installed |

---

### NVIDIA Software Stack

| Component | Minimum | Recommended | Notes |
|-----------|---------|-------------|-------|
| **NVIDIA Driver** | 550.54.15 | 555.42.02+ | For CUDA 12.9 |
| **NCCL** | 2.18.0 | 2.27.6 | `nvidia-nccl-cu12==2.27.6` |
| **cuBLAS** | 12.1 | 12.8+ | Bundled in CUDA toolkit |
| **cuDNN** | 8.9 | 9.0+ | Bundled in PyTorch |

**Verify**:
```bash
nvidia-smi  # Driver version
python -c "import torch; print(torch.cuda.nccl.version())"  # NCCL
```

---

## Verified Configurations

### Configuration 1: H200 Production

| Component | Version | Validated |
|-----------|---------|-----------|
| **GPU** | H200 80GB SXM5 | ✓ |
| **Driver** | 555.42.02 | ✓ |
| **CUDA** | 12.9.1 | ✓ |
| **PyTorch** | 2.8.0 | ✓ |
| **SGLang** | 0.5.3rc0 | ✓ |
| **sgl-kernel** | 0.3.13 | ✓ |
| **FlashInfer** | 0.4.0rc3 | ✓ |
| **NCCL** | 2.27.6 | ✓ |
| **OS** | Ubuntu 22.04 | ✓ |

**Use Case**: Large-scale production (DeepSeek-V3, 4P-9D)

---

### Configuration 2: H100 Standard

| Component | Version | Validated |
|-----------|---------|-----------|
| **GPU** | H100 80GB SXM5 | ✓ |
| **Driver** | 555.42.02 | ✓ |
| **CUDA** | 12.8.1 | ✓ |
| **PyTorch** | 2.8.0 | ✓ |
| **SGLang** | 0.5.3rc0 | ✓ |
| **sgl-kernel** | 0.3.13 | ✓ |
| **FlashInfer** | 0.4.0rc3 | ✓ |
| **NCCL** | 2.27.6 | ✓ |
| **OS** | Ubuntu 22.04 | ✓ |

**Use Case**: General production, multi-node TP

---

### Configuration 3: A100 (Legacy)

| Component | Version | Validated |
|-----------|---------|-----------|
| **GPU** | A100 80GB SXM4 | ✓ |
| **Driver** | 550.54.15 | ✓ |
| **CUDA** | 12.6.1 | ✓ |
| **PyTorch** | 2.8.0 | ✓ |
| **SGLang** | 0.5.3rc0 | ⚠️ |
| **sgl-kernel** | 0.3.12 | ✓ |
| **FlashInfer** | 0.4.0rc3 | ✓ |
| **NCCL** | 2.27.6 | ✓ |
| **OS** | Ubuntu 22.04 | ✓ |

**Limitations**: No DeepEP support, TP≤8 recommended

---

## DeepEP Compatibility

### Requirements

| Component | Version | Source | Notes |
|-----------|---------|--------|-------|
| **DeepEP** | 9af0e0d | [GitHub](https://github.com/deepseek-ai/DeepEP) | Specific commit required |
| **NVSHMEM** | 3.3.9 | [NVIDIA](https://developer.nvidia.com/nvshmem) | Must build from source |
| **GDRCopy** | v2.4.4 | [GitHub](https://github.com/NVIDIA/gdrcopy) | Kernel module + libs |
| **GPU** | Hopper (H100/H200) | - | Ampere not supported |
| **Driver** | ≥550.54.15 | - | For NVSHMEM |
| **IB/RDMA** | mlx5 driver | - | InfiniBand required |

### Verified DeepEP Setups

| Hardware | CUDA | NVSHMEM | GDRCopy | DeepEP | Status |
|----------|------|---------|---------|--------|--------|
| H200 × 8 | 12.9.1 | 3.3.9 | v2.4.4 | 9af0e0d | ✓ |
| H100 × 8 | 12.8.1 | 3.3.9 | v2.4.4 | 9af0e0d | ✓ |
| A100 × 8 | 12.6.1 | - | - | - | ✗ (not supported) |

**Build Script**: `scripts/ci/ci_install_deepep.sh`

---

## PD Disaggregation Backends

### Mooncake (Default)

| Component | Version | Platform | Status |
|-----------|---------|----------|--------|
| **mooncake-transfer-engine** | Latest | x86_64 | ✓ |
| **InfiniBand** | HDR/NDR | mlx5 driver | ✓ |
| **OS** | Ubuntu 22.04 | Linux 5.15+ | ✓ |

**Installation**:
```bash
uv pip install mooncake-transfer-engine
```

**Hardware**:
- ConnectX-6 (HDR, 200 Gb/s) or newer
- ConnectX-7 (NDR, 400 Gb/s) recommended

---

### NIXL (Alternative)

| Component | Version | Platform | Status |
|-----------|---------|----------|--------|
| **nixl** | Latest | x86_64 | ✓ |
| **UCX** | 1.14+ | Optional | ✓ |
| **InfiniBand** | HDR/NDR | mlx5 driver | ✓ |

**Installation**:
```bash
pip install nixl
```

---

### ASCEND (Huawei NPU)

| Component | Version | Platform | Status |
|-----------|---------|----------|--------|
| **mf_adapter** | 1.0.0 | aarch64 | ✓ |
| **NPU** | Ascend 910B | - | ✓ |
| **OS** | Ubuntu 22.04 ARM | - | ✓ |

**Installation**: Custom wheel (see docs)

---

## Model Compatibility

### Generative Models (LLMs)

| Model Family | Tested Sizes | TP | EP | DP Attention | Notes |
|--------------|-------------|----|----|--------------|-------|
| **Llama 3.1** | 8B, 70B, 405B | ✓ | - | - | Full support |
| **Llama 3.2** | 1B, 3B, 11B, 90B | ✓ | - | - | Full support |
| **DeepSeek V2** | 16B, 236B | ✓ | ✓ | ✓ | Requires DeepEP |
| **DeepSeek V3** | 685B | ✓ | ✓ | ✓ | Requires DeepEP |
| **Qwen 2.5** | 0.5B-72B | ✓ | - | - | Full support |
| **Qwen 2.5 MoE** | A14B, A116B | ✓ | ✓ | ✓ | Requires DeepEP |
| **Mistral** | 7B, 123B | ✓ | - | - | Full support |
| **Mixtral** | 8x7B, 8x22B | ✓ | ✓ | - | MoE support |
| **Gemma 2** | 2B, 9B, 27B | ✓ | - | - | Full support |

---

### Vision-Language Models (VLMs)

| Model | Tested Sizes | TP | Notes |
|-------|-------------|-----|-------|
| **LLaVA-OneVision** | 7B, 72B | ✓ | Multi-image |
| **Qwen2-VL** | 2B, 7B, 72B | ✓ | Video support |
| **Phi-4** | 14B | ✓ | Multimodal |
| **InternVL 2.5** | 1B-78B | ✓ | OCR, grounding |

---

### Embedding Models

| Model | Dimensions | TP | Notes |
|-------|-----------|-----|-------|
| **e5-mistral-7b** | 4096 | ✓ | Instruct-tuned |
| **gte-Qwen2** | 896-3584 | ✓ | Multilingual |

---

## Docker Images

### Official Images

| Image | CUDA | Python | SGLang | Platform |
|-------|------|--------|--------|----------|
| `lmsysorg/sglang:latest` | 12.9.1 | 3.12 | 0.5.3rc0 | x86_64 |
| `lmsysorg/sglang:v0.5.3rc0` | 12.9.1 | 3.12 | 0.5.3rc0 | x86_64 |
| `lmsysorg/sglang:rocm` | ROCm 6.2 | 3.12 | 0.5.3rc0 | AMD GPU |

**Pull**:
```bash
docker pull lmsysorg/sglang:latest
```

---

## Known Incompatibilities

### ❌ Not Supported

| Combination | Reason | Workaround |
|-------------|--------|------------|
| Python 3.13 | PyTorch not ready | Use Python 3.12 |
| CUDA 11.x | FlashInfer requires 12.1+ | Upgrade to CUDA 12.x |
| A100 + DeepEP | Requires Hopper | Use H100/H200 |
| Windows | Limited testing | Use WSL2 or Linux |
| ARM64 (non-GB200) | Limited support | Use x86_64 |

---

### ⚠️ Partial Support

| Feature | Limitation | Notes |
|---------|-----------|-------|
| AMD ROCm | Limited testing | Use ROCm 6.2+ |
| Intel XPU | Experimental | Community-maintained |
| Ascend NPU | China region | Custom builds |

---

## Version History

### v0.5.3rc0 (Current)

**Released**: 2025-10-01

**Key Changes**:
- FlashInfer 0.4.0rc3 (faster attention)
- PyTorch 2.8.0 support
- Improved PD disaggregation stability
- DeepEP optimizations

**Breaking Changes**: None

---

### v0.5.2 (Previous Stable)

**Released**: 2025-09-15

**Key Changes**:
- Initial PD disaggregation support
- DeepEP integration
- Router improvements

---

## Upgrade Path

### From v0.5.2 to v0.5.3rc0

```bash
# 1. Backup current installation
pip freeze > requirements_backup.txt

# 2. Upgrade SGLang
pip install --upgrade sglang[all]

# 3. Upgrade FlashInfer
pip install --upgrade flashinfer_python==0.4.0rc3

# 4. Download new cubins
FLASHINFER_LOGGING_LEVEL=warning python -m flashinfer --download-cubin

# 5. Verify
python -m sglang.launch_server --version
```

**No config changes required** for standard deployments.

---

**End of COMPAT_MATRIX memory**