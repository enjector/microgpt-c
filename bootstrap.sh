#!/bin/bash
set -e
echo "[*] Configuring with CMake..."
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
echo "[*] Building project..."
cmake --build build --config Release
echo "[*] Success! Binaries are in build/Release"

