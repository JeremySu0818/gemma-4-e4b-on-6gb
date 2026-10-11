#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${SCRIPT_DIR}/.venv"

echo "================================================================="
echo "  gemma-4-e4b-on-6gb Setup"
echo "  Running Gemma 4 E4B on RTX 3060 Laptop using vLLM"
echo "================================================================="

PYTHON_CMD="python3"
if ! command -v "$PYTHON_CMD" &> /dev/null; then
    echo "[!] python3 not found. Please install Python (>= 3.10 recommended)."
    exit 1
fi

if [ ! -d "$VENV_DIR" ]; then
    echo "[*] Creating virtual environment..."
    if command -v uv &> /dev/null; then
        echo "[*] uv detected. Creating virtual environment with uv..."
        uv venv "$VENV_DIR" --python 3.12 2>/dev/null || uv venv "$VENV_DIR"
    else
        echo "[*] Creating virtual environment with python3 -m venv..."
        "$PYTHON_CMD" -m venv "$VENV_DIR"
    fi
else
    echo "[+] Virtual environment already exists: $VENV_DIR"
fi

VENV_PYTHON="${VENV_DIR}/bin/python"
VENV_PIP="${VENV_DIR}/bin/pip"

echo "[*] Installing dependencies (vLLM 0.31.0 and tools)..."
if command -v uv &> /dev/null; then
    uv pip install --python "$VENV_PYTHON" -r "${SCRIPT_DIR}/requirements.txt"
else
    "$VENV_PIP" install --upgrade pip
    "$VENV_PIP" install -r "${SCRIPT_DIR}/requirements.txt"
fi

echo "[*] Applying Gemma-4-E4B low-VRAM CPU-offload patch to vLLM..."
"$VENV_PYTHON" "${SCRIPT_DIR}/patch_vllm.py"

MODEL_DIR="${SCRIPT_DIR}/model"
MODEL_ID="google/gemma-4-E4B-it"

if [ "$1" == "--skip-download" ]; then
    echo "[*] Skipping model download check (--skip-download)."
elif [ ! -f "${MODEL_DIR}/model.safetensors" ]; then
    echo ""
    echo "================================================================="
    echo "[*] Model weights not detected. Downloading via global hf CLI..."
    echo "    Model ID:   ${MODEL_ID}"
    echo "    Target dir: ${MODEL_DIR}"
    echo "================================================================="
    
    HF_CMD="hf"
    if ! command -v "$HF_CMD" &> /dev/null; then
        if [ -x "${HOME}/.local/bin/hf" ]; then
            HF_CMD="${HOME}/.local/bin/hf"
        else
            echo "[!] Global hf command not found. Please install: pip install --upgrade huggingface_hub"
            exit 1
        fi
    fi

    mkdir -p "${MODEL_DIR}"
    "$HF_CMD" download "${MODEL_ID}" --local-dir "${MODEL_DIR}"
    echo "[+] Model weights downloaded successfully to: ${MODEL_DIR}"
else
    echo "[+] Model weights already present at: ${MODEL_DIR}"
fi

echo ""
echo "================================================================="
echo "  Setup completed successfully!"
echo "  Start server: ./run_vllm_server.sh"
echo "  Test API:     python3 test_client.py"
echo "================================================================="
