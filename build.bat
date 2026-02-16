@echo off
REM Build microgpt on Windows (Release).
REM Usage: build.bat [simd]
REM   simd  - enable SIMD (AVX2) for faster build: build.bat simd

setlocal
cd /d "%~dp0"

if not exist build mkdir build
cd build

if /i "%1"=="simd" (
  echo Configuring with MICROGPT_SIMD=ON...
  cmake -DMICROGPT_SIMD=ON ..
) else (
  cmake ..
)

if errorlevel 1 exit /b 1

cmake --build . --config Release
if errorlevel 1 exit /b 1

echo.
echo Build done. Run from project root: build\Release\microgpt.exe
endlocal
