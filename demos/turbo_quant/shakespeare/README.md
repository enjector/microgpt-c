# TurboQuant: Word-Level Shakespeare Benchmark

This directory contains the integration benchmark testing the integration of the **TurboQuant** vector compression layer directly into the **Memory Sparse Attention (MSA) Paged Latent Pool** over an infinite sliding text context. 

## Benchmark Objectives
1. Profile pure autoregressive token generation speed (`words/second`).
2. Compare the computational overhead of MSA's default FP32 chunk indexing against the 4-bit TurboQuant-compressed lookup latency. 

## Build & Run
Ensure that the main CMake system has successfully built the corresponding demonstration layers:
```bash
# Build the baseline uncompressed setup and the TurboQuant configuration
cd ../../../build
make tq_shakespeare_base tq_shakespeare_tq

# Execute baseline
./tq_shakespeare_base

# Execute TurboQuant
./tq_shakespeare_tq
```

## Observations
When tested natively on an M-series CPU, the 4-bit compression enabled by TurboQuant actually **accelerated generative throughput**:

- **Baseline (Uncompressed FP32):** ~28,000 tok/sec
- **TurboQuant:** ~36,000 tok/sec

**Why is it Faster?** Even though recovering vector approximations from local residues and 3-bit clustering indices requires extra math in the inner MSA router functions, reducing the physical size of the latent sequence windows from 32-bits down to ~4-bits entirely solves L1/L2 cache bottlenecking bounds. Eliminating massive RAM trips drastically accelerates end-to-end generation beyond standard uncompressed operations.
