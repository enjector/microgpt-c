@echo off
REM Build microgpt with INT8 weight quantization (Windows, Release).
REM Usage: build_quantised.bat [simd]
REM   simd  - also enable SIMD (AVX2): build_quantised.bat simd

setlocal
cd /d "%~dp0"

if not exist build_quantised mkdir build_quantised
cd build_quantised

if /i "%1"=="simd" (
  echo Configuring with QUANTIZATION_INT8=ON and MICROGPT_SIMD=ON...
  cmake -DQUANTIZATION_INT8=ON -DMICROGPT_SIMD=ON ..
) else (
  echo Configuring with QUANTIZATION_INT8=ON...
  cmake -DQUANTIZATION_INT8=ON ..
)

if errorlevel 1 exit /b 1

cmake --build . --config Release
if errorlevel 1 exit /b 1

echo.
echo Build done (INT8 quantised). Run from project root: build_quantised\Release\microgpt.exe
endlocal
