# TDD_metal — Technical Design Document (Apple Metal GPU bridge)

**Document ID:** TDD-METAL-001
**Version:** 1.0
**Status:** DRAFT
**Paired BS:** `BS_metal.md`
**Sources:** `src/microgpt_metal.h`, `src/microgpt_metal.m`, `src/microgpt_metal.metal`

## 1. Overview

The Metal bridge offloads dense linear forward and backward (matrix-vector) operations to Apple's Metal compute shaders on macOS. It is opt-in via `MICROGPT_METAL=ON` and only relevant for `N_EMBD ≥ 512`; below that, GPU dispatch overhead exceeds compute time and the multi-threaded CPU path is faster.

Metal natively supports float32 only; the bridge converts `double ↔ float` at the CPU/GPU boundary. This is acceptable because gradient noise dominates the rounding error.

## 2. Architecture

```
   CPU side (microgpt.c::lin_fwd / lin_bwd)
            │  if MICROGPT_METAL && metal_available()
            ▼
   metal_lin_fwd / metal_lin_bwd   (microgpt_metal.h)
            │
            ▼
   Objective-C bridge (microgpt_metal.m)
   • copy double → float into shared MTLBuffer
   • dispatch compute pipeline
   • copy float → double back
            │
            ▼
   compute kernels (microgpt_metal.metal)
   • lin_fwd kernel: y = W @ x
   • lin_bwd kernel: dx = W^T @ dy ; dW += dy ⊗ x
```

On Apple Silicon the GPU and CPU share unified memory; `MTLBuffer` is created with `MTLResourceStorageModeShared` so the CPU pointer and GPU pointer reference the same physical bytes — no explicit copy is required, only the `double ↔ float` conversion.

## 3. Data flow

`metal_lin_fwd(x, W, nin, nout, y)`:
1. Convert `x[nin]` and `W[nout × nin]` from double to float into pre-allocated shared buffers.
2. Encode a compute command buffer dispatching the `lin_fwd` kernel with thread groups sized by `nout`.
3. Wait on completion.
4. Convert the float `y_float[nout]` back to `double y[nout]`.

`metal_lin_bwd(x, W, dy, nin, nout, dx, dW)`:
1. Copy `dy`, `W`, `x` to float buffers.
2. Dispatch `lin_bwd` kernel: writes `dx_float[nin]` and accumulates `dW_float[nout × nin]`.
3. Convert results back to double, accumulating into the caller's `dx` / `dW` (which may be NULL — the kernel skips the corresponding output).

## 4. Key data structures

### 4.1 Lazy-initialised globals (Objective-C side)

- `id<MTLDevice>` device — first GPU returned by `MTLCreateSystemDefaultDevice()`.
- `id<MTLCommandQueue>` queue.
- `id<MTLComputePipelineState>` for `lin_fwd` and `lin_bwd`.
- `id<MTLBuffer>` shared scratch buffers for the converted float operands.
- A `metal_initialised` boolean.

`metal_init` populates these and returns 0 on success, -1 on failure (no GPU, shader compile error). `metal_cleanup` releases them. `metal_available()` returns the cached state.

## 5. Algorithms

### 5.1 `lin_fwd` shader

```
threadgroup uint tid = thread_position_in_threadgroup
threadgroup uint gid = threadgroup_position_in_grid
uint j = gid * threadgroup_size + tid       // output row index
if (j >= nout) return
float acc = 0
for (uint i = 0; i < nin; i++)
    acc += W[j * nin + i] * x[i]
y[j] = acc
```

Tile sizes are chosen at pipeline-creation time based on `device.maxThreadsPerThreadgroup`.

### 5.2 `lin_bwd` shader

Two kernels (or one with branches) compute `dx = W^T @ dy` and `dW += dy ⊗ x`. The dW accumulation uses `atomic_fetch_add_explicit` on float when several threads might write to the same dW row; on Apple Silicon, native float atomics are used.

## 6. Concurrency model

Metal command buffers are submitted serially per `vm_engine`-equivalent. The bridge does not parallelise across CPU threads; for that, callers should use independent engines or the optional `MICROGPT_HEAD_PARALLEL` (CPU-side).

## 7. Trade-offs considered

| Decision | Chosen | Rejected | Rationale |
|---|---|---|---|
| Precision | Float32 on GPU, double on CPU | Double on GPU | Metal does not natively support double; emulation would lose the GPU benefit. The `double ↔ float` conversion at boundaries is fine for gradient computation. |
| Storage mode | `MTLResourceStorageModeShared` | Managed / private | Shared works on Apple Silicon (unified memory) and avoids explicit copy. Discrete-GPU Macs would require managed mode; current target is Apple Silicon. |
| Activation | Two kernels (`lin_fwd`, `lin_bwd`) | One generic matmul kernel | Specialisation removes branches; kernel size is small. |
| Thresholding | User-controlled (`N_EMBD ≥ 512` rule of thumb) | Auto-detect | The crossover depends on architecture; documenting the rule is more honest than auto-falling-back behind the user's back. |

## 8. Known limitations

- macOS only; non-Apple platforms get a stub that returns -1.
- `metal_init` must be called once before any `metal_lin_*` invocation; demos that opt into Metal SHOULD call it at startup.
- Float32 only on GPU; bit-identical CPU/GPU agreement is not expected.
- Pipeline state objects are global; concurrent dispatch from multiple threads must externally serialise.

## 9. References

- Apple Metal Programming Guide.
- `docs/BUILD_OPTIONS.md` "Apple Metal GPU Acceleration".
- `docs/research/RESEARCH_OPTIMISATIONS.md`.

## 10. Revision history

| Version | Date | Change |
|---|---|---|
| 1.0 | 2026-04-30 | Initial extraction. |
