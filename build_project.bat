@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
if %ERRORLEVEL% NEQ 0 exit /b %ERRORLEVEL%
cmake -S . -B build -G "NMake Makefiles" -DCMAKE_BUILD_TYPE=Release
if %ERRORLEVEL% NEQ 0 exit /b %ERRORLEVEL%
cmake --build build
if %ERRORLEVEL% NEQ 0 exit /b %ERRORLEVEL%
