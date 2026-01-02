#!/bin/bash
# LunaVox CPU Environment Reset Script (Linux/macOS)
# This script removes GPU dependencies and restores the lightweight CPU-only environment.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "========================================"
echo " LunaVox CPU Environment Reset"
echo "========================================"
echo ""
echo "This will:"
echo "  1. Uninstall onnxruntime-gpu and NVIDIA libraries"
echo "  2. Install lightweight onnxruntime (CPU-only)"
echo ""
echo "This frees up ~500MB of disk space."
echo ""
read -p "Continue? (y/n): " confirm
if [[ "$confirm" != "y" && "$confirm" != "Y" ]]; then
    echo "Reset cancelled."
    exit 0
fi

echo ""
echo "[1/3] Uninstalling GPU dependencies..."
python3 -m pip uninstall onnxruntime-gpu nvidia-cudnn-cu12 nvidia-cublas-cu12 nvidia-cuda-runtime-cu12 nvidia-cuda-nvrtc-cu12 nvidia-cufft-cu12 nvidia-curand-cu12 nvidia-cusolver-cu12 nvidia-cusparse-cu12 -y || true

echo ""
echo "[2/3] Installing CPU runtime (1.22.1)..."
python3 -m pip install "onnxruntime==1.22.1" "numpy<2"

echo ""
echo "[3/3] Setting LunaVox mode to CPU..."
python3 -c "import sys; sys.path.insert(0, '$PROJECT_ROOT/src'); from lunavox_tts.Utils.EnvManager import env_manager; env_manager.set_mode('cpu')"

echo ""
echo "========================================"
echo " Reset Complete!"
echo "========================================"
echo ""
echo CPU-only mode is now active.
echo To enable GPU acceleration again, run setup_gpu.sh
