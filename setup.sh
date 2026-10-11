#!/usr/bin/env bash
# ==============================================================================
# Setup script for gemma-4-e4b-on-6gb
# Running Gemma 4 E4B on RTX 3060 Laptop using vLLM (Requires 16GB+ RAM)
# ==============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${SCRIPT_DIR}/.venv"

echo "================================================================="
echo "  gemma-4-e4b-on-6gb Setup"
echo "  Running Gemma 4 E4B on RTX 3060 Laptop using vLLM"
echo "================================================================="

# 1. 檢查 Python 版本
PYTHON_CMD="python3"
if ! command -v "$PYTHON_CMD" &> /dev/null; then
    echo "[!] 找不到 python3，請先安裝 Python (建議 >= 3.10)。"
    exit 1
fi

# 2. 建立虛擬環境 (優先使用 uv，若無則使用 python3 -m venv)
if [ ! -d "$VENV_DIR" ]; then
    echo "[*] 正在建立虛擬環境..."
    if command -v uv &> /dev/null; then
        echo "[*] 偵測到 uv，使用 uv 建立虛擬環境..."
        uv venv "$VENV_DIR" --python 3.12 2>/dev/null || uv venv "$VENV_DIR"
    else
        echo "[*] 使用 python3 -m venv 建立虛擬環境..."
        "$PYTHON_CMD" -m venv "$VENV_DIR"
    fi
else
    echo "[+] 虛擬環境已存在: $VENV_DIR"
fi

VENV_PYTHON="${VENV_DIR}/bin/python"
VENV_PIP="${VENV_DIR}/bin/pip"

# 3. 安裝依賴
echo "[*] 正在安裝必要套件 (vLLM 0.31.0 及相關工具)..."
if command -v uv &> /dev/null; then
    uv pip install --python "$VENV_PYTHON" -r "${SCRIPT_DIR}/requirements.txt"
else
    "$VENV_PIP" install --upgrade pip
    "$VENV_PIP" install -r "${SCRIPT_DIR}/requirements.txt"
fi

# 4. 套用 vLLM 記憶體優化補丁 (CPU-Offload Patch)
echo "[*] 正在對 vLLM 套用 Gemma-4-E4B 低顯存補丁..."
"$VENV_PYTHON" "${SCRIPT_DIR}/patch_vllm.py"

# 5. 自動下載模型權重 (若尚未下載)
MODEL_DIR="${SCRIPT_DIR}/model"
MODEL_ID="google/gemma-4-E4B-it"

if [ "$1" == "--skip-download" ]; then
    echo "[*] 依指定參數略過模型下載檢查 (--skip-download)。"
elif [ ! -f "${MODEL_DIR}/model.safetensors" ]; then
    echo ""
    echo "================================================================="
    echo "[*] 偵測到模型尚未下載，自動呼叫全域 hf CLI 下載權重..."
    echo "    模型 ID:   ${MODEL_ID}"
    echo "    目標路徑: ${MODEL_DIR}"
    echo "================================================================="
    
    HF_CMD="hf"
    if ! command -v "$HF_CMD" &> /dev/null; then
        if [ -x "${HOME}/.local/bin/hf" ]; then
            HF_CMD="${HOME}/.local/bin/hf"
        else
            echo "[!] 找不到全域 hf 指令，請先安裝: pip install --upgrade huggingface_hub"
            exit 1
        fi
    fi

    mkdir -p "${MODEL_DIR}"
    "$HF_CMD" download "${MODEL_ID}" --local-dir "${MODEL_DIR}"
    echo "[+] 模型權重已成功下載至: ${MODEL_DIR}"
else
    echo "[+] 偵測到完整模型權重已就緒於: ${MODEL_DIR}"
fi

echo ""
echo "================================================================="
echo "  安裝與配置完成！"
echo "  啟動伺服器: ./run_vllm_server.sh"
echo "  測試 API:   python3 test_client.py"
echo "================================================================="
