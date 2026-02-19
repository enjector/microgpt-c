@echo off
echo [*] Configuring with CMake...
cmake -S . -B build -D CMAKE_BUILD_TYPE=Release
if %ERRORLEVEL% NEQ 0 (
    echo [!] CMake configuration failed.
    exit /b %ERRORLEVEL%
)

echo [*] Building project...
cmake --build build --config Release
if %ERRORLEVEL% NEQ 0 (
    echo [!] Build failed.
    exit /b %ERRORLEVEL%
)

echo [*] Success! Binaries are in build/Release/

