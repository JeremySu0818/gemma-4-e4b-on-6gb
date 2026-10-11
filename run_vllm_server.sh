#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PYTHON="${SCRIPT_DIR}/.venv/bin/python"
VENV_VLLM="${SCRIPT_DIR}/.venv/bin/vllm"

if [ ! -f "$VENV_VLLM" ]; then
    echo "[Error] vLLM is not installed yet."
    exit 1
fi

if lsof -i :1234 >/dev/null 2>&1 || nc -z 127.0.0.1 1234 >/dev/null 2>&1; then
    echo "================================================================="
    echo "[Notice] Port 1234 is currently in use!"
    echo "Please ensure LM Studio or any service occupying port 1234 is stopped."
    echo "================================================================="
    exit 1
fi

MODEL_DIR="${SCRIPT_DIR}/model"
if [ ! -f "${MODEL_DIR}/model.safetensors" ]; then
    echo "[*] Model weights not detected. Downloading via global hf CLI..."
    HF_CMD="hf"
    if ! command -v "$HF_CMD" &> /dev/null && [ -x "${HOME}/.local/bin/hf" ]; then
        HF_CMD="${HOME}/.local/bin/hf"
    fi
    mkdir -p "${MODEL_DIR}"
    "$HF_CMD" download google/gemma-4-E4B-it --local-dir "${MODEL_DIR}"
fi

export CUDA_HOME="${SCRIPT_DIR}/.venv/lib/python3.12/site-packages/nvidia/cu13"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib:${LD_LIBRARY_PATH}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export VLLM_USE_FLASHINFER_SAMPLER=0

echo "================================================================="
echo "Starting vLLM Server instance"
echo "Model: ${SCRIPT_DIR}/model (Local BF16 + CPU Offload)"
echo "Served model name: google/gemma-4-e4b"
echo "Listen address: 0.0.0.0:1234"
echo "Concurrency (max-num-seqs): 8"
echo "Context length (max-model-len): 65536 (preserved)"
echo "GPU memory utilization: 0.90"
echo "CPU offload (cpu-offload-gb): 12 GiB (optimized for 6GB VRAM)"
echo "================================================================="

exec "$VENV_VLLM" serve "${SCRIPT_DIR}/model" \
    --host 0.0.0.0 \
    --port 1234 \
    --served-model-name google/gemma-4-e4b \
    --gpu-memory-utilization 0.90 \
    --max-num-seqs 8 \
    --max-model-len 65536 \
    --dtype bfloat16 \
    --trust-remote-code \
    --enable-chunked-prefill \
    --performance-mode throughput \
    --generation-config vllm \
    --cpu-offload-gb 12
