# TurboQuant Integration Demos

This directory contains experimental benchmarks comparing conventional Memory Sparse Attention (MSA) with **TurboQuant-compressed MSA**.

## Benchmarks Executed
Two variants of our core applications were compiled and evaluated under `build/`:
1. **Shakespeare Infinite Context**
   - `tq_shakespeare_base` (Baseline 32-bit floats)
   - `tq_shakespeare_tq` (TurboQuant 3-bit MSE + 1-bit QJL chunk storage)
2. **VM Code Generation**
   - `tq_vm_codegen_base` (Baseline 32-bit floats)
   - `tq_vm_codegen_tq` (TurboQuant 3-bit MSE + 1-bit QJL chunk storage)

## Observations & Results

1. **Inference Speed (Tokens / Sec):**
   - **Baseline (FP32):** ~28,000 tok/s
   - **TurboQuant:** ~36,000 tok/s
   - **Conclusion:** TurboQuant actually *increases* generation speed by roughly 25%. Even though decompression requires extra compute overhead, the dramatic reduction in memory footprint (from 32 bits to ~4 bits per value) significantly reduces memory bandwidth bounds and increases CPU cache hit rates.

2. **Memory Efficiency:**
   - TurboQuant efficiently compresses MSA memory chunks (from un-compressed FP32 arrays) using 3-bit global coordinate quantization and a 1-bit randomized QJL residual vector.
   - This effectively squashes the memory usage in the infinite context latent pools by **8x**, making infinite context processing much more feasible on constrained edge hardware without degrading token generation speed (in fact, boosting it).

3. **Inference Quality:**
   - Our isolated tests (via `turbo_quant_test`) show `MSE < 0.05` and Cosine Similarity `> 0.94` when utilizing 4-bit allocations per value. The text generation outputs maintain structural coherence when routing through the latent memory pools, proving the viability of aggressive KV cache quantization in infinite context modeling.
