# BS_metal — Behaviour Specification (Apple Metal GPU bridge)

**Document ID:** BS-METAL-001
**Version:** 1.0
**Status:** DRAFT

## RFC 2119

The key words MUST, MUST NOT, REQUIRED, SHALL, SHALL NOT, SHOULD, SHOULD NOT, RECOMMENDED, MAY, and OPTIONAL in this document are to be interpreted as described in RFC 2119.

## 1. Scope

Behavioural contract of the optional Apple Metal GPU bridge: `metal_init`, `metal_cleanup`, `metal_available`, `metal_lin_fwd`, `metal_lin_bwd`. Compiled in only when `MICROGPT_METAL=ON`.

## 2. Type contracts

There are no public types — the Metal state is encapsulated in `microgpt_metal.m`'s file-scope globals.

## 3. Operation contracts

### 3.1 `metal_init()`

**Postconditions:** Initialises the default Metal device, command queue, and compute pipelines for `lin_fwd` and `lin_bwd`. Returns 0 on success, -1 on failure (no GPU, shader compile error, framework missing). Idempotent — repeated calls return success without re-initialising.

### 3.2 `metal_cleanup()`

**Postconditions:** Releases all Metal resources. Safe to call multiple times.

### 3.3 `metal_available()`

**Postconditions:** Returns 1 if Metal was successfully initialised, 0 otherwise. Used by the engine's CPU/GPU dispatch decision.

### 3.4 `metal_lin_fwd(x, W, nin, nout, y)`

**Preconditions:** `metal_available() == 1`. `x` length `nin`, `W` length `nout × nin` (row-major), `y` length `nout`. All three are `double *`.

**Postconditions:** Computes `y[j] = sum_i W[j × nin + i] × x[i]`, with float32 conversion at the CPU/GPU boundary. The result is bitwise close to but not identical to a CPU double-precision computation.

### 3.5 `metal_lin_bwd(x, W, dy, nin, nout, dx, dW)`

**Postconditions:** Computes `dx = W^T @ dy` (writing into `dx` if non-NULL) and accumulates `dW += dy ⊗ x` (writing into `dW` if non-NULL). Both `dx` and `dW` may be NULL — the kernel skips the corresponding output.

## 4. Invariants table

| ID | Invariant |
|---|---|
| INV-METAL-001 | Functions are no-ops or stubs returning -1 on non-macOS platforms; the build flag must be OFF or the code is not compiled. |
| INV-METAL-002 | All GPU computation is float32; CPU side maintains double precision. |
| INV-METAL-003 | `metal_lin_fwd` / `_lin_bwd` are dispatched from the engine's CPU `lin_fwd` / `lin_bwd` only when `metal_available()` returns 1; otherwise the engine falls back to the CPU path. |
| INV-METAL-004 | Apple Silicon unified memory is exploited via `MTLResourceStorageModeShared`; no explicit copy across the CPU/GPU boundary is needed (only the double↔float conversion). |

## 5. Errors

`metal_init` returns -1 on initialisation failure. The compute primitives have no explicit error path; they assume `metal_available() == 1`. Misuse is undefined.

## 6. Concurrency

Metal command buffers are submitted serially per process. Multi-threaded callers MUST externally synchronise dispatches.

## 7. Performance SLOs

No explicit SLO. The bridge is recommended only for `N_EMBD ≥ 512`; below that, GPU dispatch overhead exceeds the compute time and the multi-threaded CPU path is faster.

## 8. Scenarios

### SCN-METAL-001: Train Shakespeare with Metal

A demo built with `MICROGPT_METAL=ON` calls `metal_init` at startup; the engine's `lin_fwd` / `lin_bwd` automatically dispatch to GPU. On `N_EMBD=512+` configs the wall-clock training time is reduced; on smaller configs the multi-threaded CPU path remains faster.

## 9. Acceptance criteria

| ID | Verifies | Test |
|---|---|---|
| ACC-METAL-001 | `metal_init` succeeds on macOS / Apple Silicon | Manual; CI on `macos-latest` |
| ACC-METAL-002 | Numeric agreement with CPU path within float32 tolerance | Manual |

## 10. Cross-references

- **TDD:** `TDD_metal.md`
- **Source:** `src/microgpt_metal.{h,m,metal}`
- **Build:** `docs/BUILD_OPTIONS.md` "Apple Metal GPU Acceleration"

## 11. Revision history

| Version | Date | Change |
|---|---|---|
| 1.0 | 2026-04-30 | Initial extraction. |
