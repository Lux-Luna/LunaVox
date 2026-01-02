@echo off
REM LunaVox GPU Environment Setup Script (Windows)
REM This script installs GPU acceleration dependencies.

echo ========================================
echo  LunaVox GPU Environment Setup
echo ========================================
echo.
echo This will:
echo   1. Uninstall CPU-only onnxruntime
echo   2. Install onnxruntime-gpu (CUDA 12)
echo   3. Install required NVIDIA runtime libraries
echo.
echo Approximate download size: ~600MB
echo.
set /p confirm="Continue? (y/n): "
if /i not "%confirm%"=="y" (
    echo Installation cancelled.
    exit /b 0
)

echo.
echo [1/4] Uninstalling onnxruntime packages...
python -m pip uninstall onnxruntime onnxruntime-gpu -y

echo.
echo [2/4] Installing GPU dependencies...
python -m pip install -r "%~dp0..\requirements-gpu.txt"

echo.
echo [3/4] Verifying installation...
python -c "import onnxruntime as ort; print('✓ available providers:', ort.get_available_providers())"

echo.
echo [4/4] Setting LunaVox mode to GPU...
python -c "from lunavox_tts.Utils.EnvManager import env_manager; env_manager.set_mode('gpu')"

echo.
echo ========================================
echo  Setup Complete!
echo ========================================
echo.
echo GPU acceleration is now enabled.
echo To switch back to CPU-only mode, run cleanup_gpu.bat
echo.
pause
