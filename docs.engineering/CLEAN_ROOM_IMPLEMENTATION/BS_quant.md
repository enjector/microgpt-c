# BS_quant — Behaviour Specification (TurboQuant + RotorQuant)

**Document ID:** BS-QUANT-001
**Version:** 1.0
**Status:** DRAFT

## RFC 2119

The key words MUST, MUST NOT, REQUIRED, SHALL, SHALL NOT, SHOULD, SHOULD NOT, RECOMMENDED, MAY, and OPTIONAL in this document are to be interpreted as described in RFC 2119.

## 1. Scope

Behavioural contract of the 4-bit dual-state KV quantisers `TurboQuant` and `RotorQuant`. These quantisers compress an `n_embd`-dimensional float vector into an MSE codebook index + 1-bit QJL signature + residual norm, optionally preceded by a (dense or structured) random rotation.

## 2. Type contracts

### 2.1 `TurboQuant`

**Invariants:**
- INV-QUANT-001: `tq->d > 0`; `tq->b ∈ [2, 4]` (target bit-width for Qprod).
- INV-QUANT-002: `tq->codebook_mse[b]` SHALL contain `2^b` Lloyd-Max-optimal centroids (precomputed at init).
- INV-QUANT-003: `tq->use_rotation == true` iff a rotation matrix `Pi` is allocated and used at quant/dequant time.

### 2.2 `RotorQuant`

**Invariants:**
- INV-QUANT-010: `rq->d > 0`; `rq->b ∈ [2, 4]`.
- INV-QUANT-011: `rq->mode ∈ {RQ_MODE_PLANAR (0), RQ_MODE_ISO (1)}`.
- INV-QUANT-012: `rq->rotations` length SHALL be `⌊d/2⌋` for `RQ_MODE_PLANAR` (Givens angles) or `⌊d/4⌋ × 4` for `RQ_MODE_ISO` (quaternions).

## 3. Operation contracts

### 3.1 `turboquant_init(tq, d, b, use_rotation)`

**Postconditions:** Populates `tq->Pi` (if `use_rotation`), `tq->S` (random projection for QJL), `tq->codebook_mse[b]`, `tq->cb_sizes`. Idempotent — calling twice without `_free` leaks.

### 3.2 `turboquant_quant_prod(tq, x, &idx, qjl_signs, &rnorm)`

**Preconditions:** `x` length `tq->d`; `qjl_signs` capacity `tq->d` (one byte per dim).

**Postconditions:**
- Optional rotate: `x' = Π · x` if `tq->use_rotation`.
- MSE: `idx` is the index of the nearest centroid in `tq->codebook_mse[b]`.
- Compute residual `r = x' − dequant_mse(idx)`.
- QJL: `qjl_signs[i] = sign((S · r)[i])`.
- `rnorm = ||r||₂`.

INV-QUANT-020: This estimator is **inner-product-optimal** and **unbiased** in expectation: `E[<x_recon, y>] = <x, y>` for any `y`.

### 3.3 `turboquant_dequant_prod(tq, idx, qjl_signs, rnorm, out)`

**Postconditions:** `out` is the reconstructed vector (rotation inverted if `use_rotation`).

### 3.4 Pure-MSE variants

`turboquant_quant_mse(tq, x, &idx)` / `turboquant_dequant_mse(tq, &idx, out)` skip the QJL step and return only the codebook reconstruction. Reconstruction quality is worse but the storage is one `uint32_t` per vector.

### 3.5 RotorQuant API

Identical surface (`rotorquant_init`, `_quant_prod`, `_dequant_prod`, `_quant_mse`, `_dequant_mse`) with the `mode` argument selecting the rotation form.

## 4. Invariants table

| ID | Invariant |
|---|---|
| INV-QUANT-001..003 | TurboQuant: dim, bit-width, codebook, optional rotation. |
| INV-QUANT-010..012 | RotorQuant: dim, bit-width, mode, rotation parameter layout. |
| INV-QUANT-020 | `quant_prod`/`dequant_prod` is inner-product-optimal and unbiased. |
| INV-QUANT-021 | `MsaPool` storage MAY use either quantiser when its `ENABLE_*` flag is defined; the pool's getters / setters SHALL transparently quantise on write and dequantise on read. |
| INV-QUANT-022 | A clean-room implementation MAY use Lloyd-Max codebooks regenerated from a fixed seed; the bundled codebooks are RECOMMENDED for byte-stable comparison. |

## 5. Errors

The quantisers do not return error codes; misuse (invalid `b`, NULL pointers) is undefined behaviour.

## 6. Performance SLOs

Reference machine in `NFRD.md` §4.

| ID | Measured target |
|---|---|
| SLO-QUANT-001 | TurboQuant 4-bit ≥ 8× memory reduction vs raw `scalar_t` KV — see `NFRD.md` §4.4 |
| SLO-QUANT-002 | TurboQuant ≥ 1.3M encodes/s — see `NFRD.md` §4.4 |
| SLO-QUANT-003 | TurboQuant ≥ 25 % inference speedup on integrating demos under sufficient context — see `NFRD.md` §4.4 |

## 7. Scenarios

### SCN-QUANT-001: Quantised infinite Shakespeare

A demo builds an `MsaPool` with `ENABLE_TURBOQUANT=ON`; pool entries are stored as 4-bit triples; cosine routing reconstructs them on demand. End-to-end accuracy is unchanged on the Shakespeare corpus; memory footprint is 8× smaller.

## 8. Acceptance criteria

| ID | Verifies | Test |
|---|---|---|
| ACC-QUANT-001 | INV-QUANT-001..003, 020 | `tests/test_microgpt_turboquant.c` |
| ACC-QUANT-002 | INV-QUANT-010..012 | `tests/test_microgpt_rotorquant.c` |
| ACC-QUANT-003 | SLO-QUANT-002 | `tests/bench_microgpt_turboquant.c` |

## 9. Cross-references

- **TDD:** `TDD_quant.md`
- **Source:** `src/microgpt_turboquant.{h,c}`, `src/microgpt_rotorquant.{h,c}`
- **Integration point:** `BS_msa.md` (MsaPool storage when `ENABLE_*` is set)

## 10. Revision history

| Version | Date | Change |
|---|---|---|
| 1.0 | 2026-04-30 | Initial extraction. |
