@echo off
set "PATH=%CD%\ggml\build\bin;%PATH%"
echo Running English smoke test...
build\qwen3-tts-cli.exe -m models -t "Hello from the Windows build smoke test." -o windows_smoke_en.wav -j 4
if %ERRORLEVEL% NEQ 0 (
    echo English test failed.
    exit /b %ERRORLEVEL%
)
echo Running Chinese smoke test...
build\qwen3-tts-cli.exe -m models -t "你好，这是一次 Windows 构建完成后的中文语音合成测试。" -l zh -o windows_smoke_cn.wav -j 4
if %ERRORLEVEL% NEQ 0 (
    echo Chinese test failed.
    exit /b %ERRORLEVEL%
)
echo All tests passed.
