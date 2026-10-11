#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PYTHON="${SCRIPT_DIR}/.venv/bin/python"
VENV_VLLM="${SCRIPT_DIR}/.venv/bin/vllm"

# 1. 檢查虛擬環境
if [ ! -f "$VENV_VLLM" ]; then
    echo "[錯誤] vLLM 尚未安裝完成。"
    exit 1
fi

# 2. 檢查 1234 port 是否被占用
if lsof -i :1234 >/dev/null 2>&1 || nc -z 127.0.0.1 1234 >/dev/null 2>&1; then
    echo "================================================================="
    echo "[提示] 偵測到 1234 port 正在被使用中！"
    echo "請確認 LM Studio 或其他占用 1234 port 的服務已關閉。"
    echo "================================================================="
    exit 1
fi

# 3. 檢查模型權重是否就緒 (若無則自動下載)
MODEL_DIR="${SCRIPT_DIR}/model"
if [ ! -f "${MODEL_DIR}/model.safetensors" ]; then
    echo "[*] 未偵測到模型權重，正透過全域 hf CLI 自動下載..."
    HF_CMD="hf"
    if ! command -v "$HF_CMD" &> /dev/null && [ -x "${HOME}/.local/bin/hf" ]; then
        HF_CMD="${HOME}/.local/bin/hf"
    fi
    mkdir -p "${MODEL_DIR}"
    "$HF_CMD" download google/gemma-4-E4B-it --local-dir "${MODEL_DIR}"
fi

# 設定 CUDA Toolkit 路徑 (FlashInfer JIT 需要 nvcc)
export CUDA_HOME="${SCRIPT_DIR}/.venv/lib/python3.12/site-packages/nvidia/cu13"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib:${LD_LIBRARY_PATH}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export VLLM_USE_FLASHINFER_SAMPLER=0

echo "================================================================="
echo "啟動 vLLM Server 實例"
echo "模型: ${SCRIPT_DIR}/model (本地下載 BF16 + CPU Offload)"
echo "服務模型名: google/gemma-4-e4b"
echo "監聽地址: 0.0.0.0:1234"
echo "併發容量 (max-num-seqs): 8"
echo "Context 長度 (max-model-len): 65536 (嚴格保留)"
echo "顯存利用率 (gpu-memory-utilization): 0.90"
echo "CPU 卸載 (cpu-offload-gb): 12 GiB (適配 6GB VRAM)"
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
