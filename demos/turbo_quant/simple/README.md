# TurboQuant Standalone Kernels

This directory maintains the completely naked `turboquant.c` arithmetic validation logic. Unlike the parallel directories (`shakespeare` and `vm_codegen`) which test TurboQuant functionally inside the broader MicroGPT-C Transformer hierarchy, `simple/` assesses the pure matrix operations.

## Purpose
1. Check the MSE centroid clustering loop operations and alignment.
2. Measure matrix cosine similarity between an isolated continuous FP32 representation against its 3-bit QJL decoded expansion proxy.

## Build & Run
The main CMake framework automatically assigns this module under the `turbo_quant_test` target:
```bash
cd ../../../build
make turbo_quant_test
./turbo_quant_test
```

## Results
Independent testing validates reconstructive accuracy over dummy structures, scoring `MSE < 0.05` alongside Cosine Similarities hovering upwards of `0.94` per array. It exposes how minimal memory storage can accurately mock its source data mathematically.
