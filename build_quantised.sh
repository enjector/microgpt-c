#!/usr/bin/env sh
# Build microgpt with INT8 weight quantization (Linux/macOS, Release).
# Usage: ./build_quantised.sh [--simd]
#   --simd  also enable SIMD (-march=native)

set -e
cd "$(dirname "$0")"

EXTRA=""
if [ "${1:-}" = "--simd" ]; then
  EXTRA="-DMICROGPT_SIMD=ON"
  echo "Configuring with QUANTIZATION_INT8=ON and MICROGPT_SIMD=ON..."
else
  echo "Configuring with QUANTIZATION_INT8=ON..."
fi

mkdir -p build_quantised
cd build_quantised

cmake -DQUANTIZATION_INT8=ON $EXTRA ..
cmake --build . --config Release

echo ""
echo "Build done (INT8 quantised). Run from project root: ./build_quantised/microgpt"
