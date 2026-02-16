#!/usr/bin/env sh
# Build microgpt on Linux/macOS (Release).
# Usage: ./build.sh [--simd]
#   --simd  enable SIMD (-march=native) for faster build

set -e
cd "$(dirname "$0")"

SIMD=""
if [ "${1:-}" = "--simd" ]; then
  SIMD="-DMICROGPT_SIMD=ON"
  echo "Configuring with MICROGPT_SIMD=ON..."
fi

mkdir -p build
cd build

cmake $SIMD ..
cmake --build . --config Release

echo ""
echo "Build done. Run from project root: ./build/microgpt"
