# 🚀 gemma-4-e4b-on-6gb

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-brightgreen.svg)]()
[![vLLM](https://img.shields.io/badge/vLLM-0.31.0-orange.svg)]()
[![Hardware](https://img.shields.io/badge/GPU-RTX%203060%20(6GB)-purple.svg)]()
[![RAM](https://img.shields.io/badge/RAM-16GB%2B%20Required-blue.svg)]()
[![Context](https://img.shields.io/badge/Context%20Length-65536-red.svg)]()

> **Running Gemma 4 E4B on RTX 3060 Laptop using vLLM (Requires 16GB+ RAM)**

---

## 📌 Background & Motivation

**Gemma-4-E4B** uses an innovative **Per-Layer Embeddings (PLE)** architecture. However, in standard unquantized BF16 format:
- The single vocabulary embedding matrix alone takes **~5.25 GiB** of memory.
- Standard inference engines (such as stock vLLM) attempt to allocate the entire embedding matrix on the GPU, instantly causing a `CUDA out of memory (OOM)` crash on 6GB consumer graphic cards.
- GGUF quantizations for Gemma-4 are currently not natively mapped in vLLM's plugin system without specialized conversions.

This project delivers an automated, production-tested patch and serving pipeline that makes **full BF16 Gemma-4-E4B** run natively under **vLLM** on 6GB VRAM GPUs, preserving full **65,536 context** and OpenAI API compatibility.

---

## 🛠️ Architecture & Core Innovations

```
 +-------------------------------------------------------------------+
 |                   Gemma-4-E4B Architecture Flow                  |
 +-------------------------------------------------------------------+
                                    |
            [User Prompt / Input Tokens (up to 65,536)]
                                    |
                                    v
     +-------------------------------------------------------------+
     |   SYSTEM RAM (Host Memory)                                  |
     |   - 5.25 GiB PLE Embedding Matrix (CPU Look-up via Patch)   |
     |   - 12 GiB Transformer Weights (UVA Zero-Copy Offloading)   |
     +-------------------------------------------------------------+
                                    | (PCIe Streaming)
                                    v
     +-------------------------------------------------------------+
     |   NVIDIA GPU (RTX 3060 - 6GB VRAM)                          |
     |   - Active Transformer Computation Layers (~1.6 GiB)        |
     |   - PagedAttention KV Cache (~2.9 GiB -> 141,872 Tokens)    |
     |   - Full & Piecewise CUDA Graph Execution (Fast Decode)     |
     +-------------------------------------------------------------+
                                    |
                                    v
              [OpenAI-Compatible Streaming Output (Port 1234)]
```

1. **Selective Embedding CPU-Offloading**:
   - Patched `UnquantizedEmbeddingMethod` in `vllm/model_executor/layers/vocab_parallel_embedding.py` to initialize parameters `> 2 GiB` on CPU and perform lookups in host memory before passing embeddings to CUDA.
   - Guarded `device_loading_context` in `vllm/model_executor/model_loader/utils.py` with `requires_device_loading = False` to prevent vLLM from forcing huge weights back into VRAM after loading.
2. **UVA Offloading (`--cpu-offload-gb 12`)**:
   - Offloads remaining non-critical weights to host RAM using Unified Virtual Addressing.
3. **FlashInfer Sampler Conflict Workaround**:
   - Resolves CCCL / nvcc header incompatibilities by enabling PyTorch's native sampler via `VLLM_USE_FLASHINFER_SAMPLER=0`.
4. **Inductor & Dynamic Memory Reservation**:
   - Configured `PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"` and tuned `--gpu-memory-utilization 0.90` to guarantee adequate headroom for `torch.compile` graph capture while maximizing KV cache space (**2.89 GiB / 141,872 tokens**).

---

## 💻 Hardware Requirements

| Component | Minimum Specification | Tested Specification |
| :--- | :--- | :--- |
| **GPU** | NVIDIA GPU with **>= 6GB VRAM** | RTX 3060 Laptop (6144 MiB) |
| **RAM** | >= 24 GB System RAM | 32 GB DDR4/DDR5 |
| **OS** | Linux (Ubuntu 22.04+ recommended) | Linux x86_64 |
| **CUDA** | CUDA 12.4+ / 13.x | CUDA 13.2 / Driver 595.99 |

---

## ⚡ Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/gemma-4-e4b-on-6gb.git
cd gemma-4-e4b-on-6gb
```

### 2. Run Automated Setup
The `setup.sh` script creates the virtual environment, installs vLLM and dependencies, and automatically applies the memory optimization patches:
```bash
chmod +x setup.sh run_vllm_server.sh
./setup.sh
```
*What `./setup.sh` does automatically:*
- Creates the Python virtual environment.
- Installs pinned `vllm==0.31.0` and dependencies.
- Applies the low-VRAM CPU-Offloading patch.
- **Automatically downloads `google/gemma-4-E4B-it` via global `hf` CLI** into `model/` (skips if already downloaded).

### 3. Launch vLLM Server
```bash
./run_vllm_server.sh
```
*The server will start listening on `0.0.0.0:1234`.*

---

## 🧪 Testing the API

### Method 1: Using the Included Test Client
```bash
python3 test_client.py
```

### Method 2: Using cURL
```bash
curl http://localhost:1234/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "google/gemma-4-e4b",
    "messages": [
      {"role": "user", "content": "Hello! Introduce yourself in one sentence."}
    ],
    "max_tokens": 60,
    "temperature": 0.7
  }'
```

### Method 3: Using OpenAI Python SDK
```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:1234/v1",
    api_key="none"
)

response = client.chat.completions.create(
    model="google/gemma-4-e4b",
    messages=[{"role": "user", "content": "Tell me a short joke."}],
    max_tokens=100
)
print(response.choices[0].message.content)
```

---

## 📊 Benchmark & Performance

Tested on **Intel Core i7-12700H + NVIDIA RTX 3060 Laptop (6GB)**:

- **Max Context Length**: `65,536` tokens (Full length preserved)
- **KV Cache Size**: `141,872 tokens` (~2.89 GiB GPU memory)
- **Concurrency Support**: Up to **8 concurrent streams**
- **GPU Peak Memory**: ~5.1 GiB / 6.0 GiB (Clean headroom without OOM)
- **Throughput**: ~4 tok/s aggregate throughput across concurrent workers under PCIe bandwidth constraints

---

## 📁 Repository Structure

```
├── .gitignore             # Ignores model weights, virtual environments, logs
├── LICENSE                # Apache 2.0 Open Source License
├── README.md              # Complete documentation (This file)
├── requirements.txt       # Pinned dependencies (vLLM 0.31.0, PyTorch 2.13.0)
├── requirements-lock.txt  # Full frozen environment lockfile
├── patch_vllm.py          # Standalone, idempotent Python patcher for vLLM
├── patches/
│   └── vllm-0.31.0-gemma4-embedding-cpu-offload.patch  # Unified diff patch
├── setup.sh               # One-click environment bootstrap & patch installer
├── run_vllm_server.sh     # Production launch script with tuned parameters
└── test_client.py         # Verification client script
```

---

## 📄 License

This project is licensed under the [Apache 2.0 License](LICENSE).
Gemma model weights are subject to Google's [Gemma Terms of Use](https://ai.google.dev/gemma/terms).
