#!/bin/bash
# LunaVox GPU Environment Setup Script (Linux/macOS)
# This script installs GPU acceleration dependencies.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "========================================"
echo " LunaVox GPU Environment Setup"
echo "========================================"
echo ""
echo "This will:"
echo "  1. Uninstall CPU-only onnxruntime"
echo "  2. Install onnxruntime-gpu (CUDA 12)"
echo "  3. Install required NVIDIA runtime libraries"
echo ""
echo "Approximate download size: ~600MB"
echo ""
read -p "Continue? (y/n): " confirm
if [[ "$confirm" != "y" && "$confirm" != "Y" ]]; then
    echo "Installation cancelled."
    exit 0
fi

echo ""
echo "[1/4] Uninstalling onnxruntime packages..."
python3 -m pip uninstall onnxruntime onnxruntime-gpu -y || true

echo ""
echo "[2/4] Installing GPU dependencies..."
python3 -m pip install -r "$PROJECT_ROOT/requirements-gpu.txt"

echo ""
echo "[3/4] Verifying installation..."
python3 -c "import onnxruntime as ort; print('✓ available providers:', ort.get_available_providers())"

echo ""
echo "[4/4] Setting LunaVox mode to GPU..."
python3 -c "import sys; sys.path.insert(0, '$PROJECT_ROOT/src'); from lunavox_tts.Utils.EnvManager import env_manager; env_manager.set_mode('gpu')"

echo ""
echo "========================================"
echo " Setup Complete!"
echo "========================================"
echo ""
echo "GPU acceleration is now enabled."
echo "To switch back to CPU-only mode, run cleanup_gpu.sh"
