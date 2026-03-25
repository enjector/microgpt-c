# TurboQuant KV-Cache Compression Demo

This demo validates the standalone `microgpt_turboquant` kernels against the paper's
theoretical distortion bounds (arXiv 2504.19874, Theorems 1 & 2).

## Purpose
1. Measure MSE distortion (Q_mse, Theorem 1) at b = 1–4 bits and compare to paper bounds.
2. Measure inner-product distortion and bias (Q_prod, Theorem 2) — confirming the unbiased estimator property.
3. Show memory reduction for a simulated 1024-token KV-cache at d = 128.

## Build & Run
```bash
cd ../../../build
cmake ..
make tq_kv_cache_demo
./tq_kv_cache_demo
```

## Results
Across b = 1–4 bits, measured distortion tracks the paper's bounds within a few percent.
At b = 4 bits, Q_mse achieves `D_mse ≈ 0.009` (paper bound: 0.009) and Q_prod delivers
near-zero IP bias — confirming the unbiased inner-product estimator guarantee.
Memory reduction versus FP32 baseline: **7.5× at b = 4**.
