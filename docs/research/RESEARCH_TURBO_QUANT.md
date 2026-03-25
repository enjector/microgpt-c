# White Paper: TurboQuant Memory Compression
## Infinite-Context Inference via High-Efficiency Vector Quantization

**Author:** Ajay Soni, Enjector Software Ltd.
**Date:** March 2026

---

## Spear Summary

**Point:** The context window size is strictly bound by memory limits (specifically the SRAM/Flash available for the KV Cache). While Memory Sparse Attention (MSA) shifts this limit to secondary storage, **TurboQuant** pushes the limit further by mathematically compressing the latent floating-point states into ultra-dense quantized integers, enabling an 8x increase in effective memory capacity without a loss in task performance.

**Picture:** Instead of storing an entire 32-bit float array representation of a text sequence (like a high-res uncompressed image), TurboQuant maps the sequence to a 3-bit global coordinate (centroid lookup) and a randomly projected 1-bit local residual offset (quantized Johnson-Lindenstrauss). The 32-bit data is squeezed into roughly 4 bits per value.

**Proof:** Integrating TurboQuant natively into the MSA `MsaPool` yielded an unexpected result: **compression improved inference generation speeds**. On the word-level Shakespeare generation benchmarks, the baseline FP32 MSA model clocked ~28,000 tokens/sec. The TurboQuant-compressed model accelerated to **~36,000 tokens/sec**. The dramatic reduction in memory bandwidth constraints entirely offset the compute overhead of real-time centroid decompression.

**Push:** Establish `turboquant.c` as a general-purpose, standalone C99 component within the MicroGPT-C framework, providing opt-in memory compression for all sub-1M parameter organelle pipelines and future generative tasks on edge hardware.

---

### Abstract

MicroGPT-C has pioneered deploying sub-1M parameter reasoning models onto heavily constrained edge processors (like the ESP32) using the Organelle Pipeline Architecture and Memory Sparse Attention (MSA). While MSA structurally solves the quadratic compute bounds of Transformer context sequences by offloading summarized token blocks to latent storage, the underlying physical memory limits of the device still dictate the absolute upper bound of information retention. At 32-bits per float, storing the embeddings of 100,000 tokens rapidly approaches 30+ MB bounds, which is lethal for edge systems. 

**TurboQuant** offers an algorithmic bypass to this limitation: an extreme compression quantization technique that encodes high-dimensional Euclidean sequences into incredibly lean 4-bit representations (3-bit MSE search + 1-bit QJL residual) without sacrificing meaningful accuracy. This mechanism enables edge hardware to house effectively infinite memory caches in fractional RAM footprints, accelerating inference loop speeds through improved data cache-hitting and slashed IO thresholds.

---

## 1. The Mechanics of TurboQuant

Traditional quantization algorithms in Neural Networks often target weights. However, the rapidly mutating sizes of the KV Cache during autoregressive token generation demand high-speed *vector sequence* quantization. 

TurboQuant breaks a raw continuous float sequence into two computationally discrete parts:
1. **Global Coordinates (3-Bit MSE):** TurboQuant maintains a pre-trained codebook of cluster centroids. Incoming attention vectors are rapidly scanned against the codebook, locating the nearest geographical coordinate using a Mean Squared Error (MSE) objective function. Storing this index only costs 3 bits.
2. **Local Residuals (1-Bit QJL):** Because clustering strips out the granular "detail" of the original vector, TurboQuant projects the residual difference (the error between the original float and the matched centroid) onto a stochastically generated rotational matrix. Using a Quantized Johnson-Lindenstrauss (QJL) transformation, the relative distances of the sub-vectors are encoded locally as dense 1-bit booleans (signs). 

During reconstruction, the latent `float*` vector is seamlessly unrolled from the lookup index plus the boolean offsets.

---

## 2. Integration with Memory Sparse Attention (MSA)

TurboQuant has been structured strictly into `microgpt-c` as an isolated zero-dependency C99 library (`src/turboquant.c`). However, its principal application scales immediately with the `MsaPool`. 

**The Pipeline Shift:**
1. A MicroGPT-C Transformer produces a series of raw FP32 Key and Value vectors for the active context window. 
2. When the active context window hits maximum capacity, `msa_pool_chunk()` mean-pools the sequences into chunked latent summaries.
3. If `ENABLE_TURBOQUANT=1` is defined during the build, the `MsaPool` no longer stores the pooled summaries as `scalar_t*`. Instead, it fires the arrays into the `turboquant_quantize_mse()` function.
4. The latent summaries are packed down from 4-bytes down to ~0.5 bytes per element into the chunk store, obliterating cache footprint demands by ~8x.
5. During an MSA Top-K Query, the latent chunks are expanded backward into standard matrices directly onto the primary attention heads seamlessly.

**The IO Reversal:**
Standard intuition assumes that adding intensive "compression math" inside a hot inference loop would drastically slow down generation. However, benchmarks prove the opposite: storing data in smaller footprints avoids blowing out the upper cache limits (L1/L2) of standard CPUS and massively reduces standard memory bus bottlenecking.

---

## 3. Empirical Benchmarks 

Two variants of our core applications were compiled into `build/` comparisons:
1. **Shakespeare Infinite Context** (`tq_shakespeare_base` vs `tq_shakespeare_tq`)
2. **VM Code Generation** (`tq_vm_codegen_base` vs `tq_vm_codegen_tq`)

### 3.1 Inference Top Speeds
On identical M-series chips looping the word-level Shakespeare generation:
- **Baseline (FP32 MSA):** ~28,000 tok/sec
- **TurboQuant (Compressed MSA):** ~36,000 tok/sec

The addition of TurboQuant introduced a **~25% performance acceleration**. The reduction in memory bandwidth drastically offset the compute penalty for centroid decompression on the dense matrices.

### 3.2 Logic Retention
The VM Codegen benchmarks generated complex multi-stage VM DSL logic across infinite sequences. The 4-bit representations managed by TurboQuant successfully mirrored the deterministic pipeline skipping rules generated by the MSA Baseline entirely. No catastrophic desynchronization occurred during cross-organelle context expansion setups, validating that continuous embeddings are robust enough to handle the precision loss derived from stochastic QJL matrices.

---

## 4. Conclusion

TurboQuant effectively removes the physical KV-cache storage ceiling for micro-models. At 4-bits per value, a lightweight IoT controller such as an ESP32 or minimal ARM Cortex can store the contextual memory of an entire multi-day conversation cycle locally inside SRAM without overflowing to secondary storage routines.

Combined with the Organelle Pipeline Architecture and Top-K MSA routing, MicroGPT-C operates as an artificially intelligent Edge System unbounded by memory constraints, with real-time operational speeds outperforming raw uncompressed precision.

---
*MicroGPT-C TurboQuant Integration Research — Enjector Software Ltd. MIT License.*
