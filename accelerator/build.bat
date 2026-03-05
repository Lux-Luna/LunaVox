@echo off
chcp 65001 >nul 2>&1
set PYTHONIOENCODING=utf-8

REM Build script for LunaVox C++ accelerator
call "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\Common7\Tools\VsDevCmd.bat" -arch=amd64

cd /d "%~dp0"

echo.
echo ===== Building LunaVox C++ Accelerator =====
echo.

"C:\Users\kwong\miniconda3\envs\lunavox\python.exe" setup.py build_ext --inplace

echo.
if %ERRORLEVEL% EQU 0 (
    echo ===== BUILD SUCCESS =====
    dir *.pyd 2>nul
) else (
    echo ===== BUILD FAILED (exit code: %ERRORLEVEL%) =====
)
