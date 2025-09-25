@echo off
setlocal ENABLEDELAYEDEXPANSION

REM Change to script directory
cd /d %~dp0

REM Prefer local package source
set "PYTHONPATH=%CD%\src;%PYTHONPATH%"

REM Use local models/dictionary if present
set "HUBERT_MODEL_PATH=%CD%\Data\chinese-hubert-base.onnx"
set "OPEN_JTALK_DICT_DIR=%CD%\Data\open_jtalk_dic_utf_8-1.11"

REM Ensure Output directory exists
if not exist "Output" mkdir "Output"

REM Launch WebUI (Gradio will auto-open browser)
python WebUI\webui.py

endlocal

