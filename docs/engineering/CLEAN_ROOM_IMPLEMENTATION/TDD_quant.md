# TDD_quant — Technical Design Document (TurboQuant + RotorQuant)

**Document ID:** TDD-QUANT-001
**Version:** 1.0
**Status:** DRAFT
**Paired BS:** `BS_quant.md`
**Sources:** `src/microgpt_turboquant.{h,c}`, `src/microgpt_rotorquant.{h,c}`

## 1. Overview

TurboQuant and RotorQuant are two interchangeable 4-bit dual-state quantisers applied to the MSA pool's K/V vectors. Both compress an `n_embd`-dimensional float vector into:

- A 32-bit MSE codebook *index* (ceil(b−1) bits used; the remaining bits ride in the 32-bit slot).
- A 1-bit QJL signature: the sign of a random projection of the vector, packed across all dimensions.
- A residual norm (a single float capturing `||r||₂` after MSE quantisation).

Reconstruction is inner-product-optimal and unbiased — designed for cosine routing in MSA without accuracy regression, not for byte-identical recovery.

TurboQuant uses a learned per-vector random rotation (`Π`) to spread energy uniformly across dimensions before quantisation. RotorQuant replaces the dense rotation with structured rotors (2D Givens for `RQ_MODE_PLANAR`, 4D quaternion for `RQ_MODE_ISO`) — much lower memory cost for the rotation parameter at a small reconstruction quality penalty.

## 2. Architecture

```
                        K vector (n_embd × float)
                                  │
                                  ▼
         ┌────────────────────────────────────────┐
         │  Optional rotation (TurboQuant: dense Π│
         │   RotorQuant: planar / quaternion)     │
         └────────────────┬───────────────────────┘
                          │
        ┌─────────────────┼─────────────────┐
        ▼                 ▼                 ▼
  MSE quantise     QJL projection      residual norm
  → idx (b−1 bits) → ±1 signs (n bits)  → ||r||₂ (float)
        │                 │                 │
        └─────────────────┴─────────────────┘
                          │
                          ▼
              ┌─────────────────────────┐
              │ packed pool entry        │
              │   uint32_t idx           │
              │   int8_t   qjl_signs[]   │
              │   float    rnorm         │
              └─────────────────────────┘
```

## 3. Data flow

`turboquant_quant_prod(tq, x, &idx, qjl_signs, &rnorm)`:

1. Optional rotate: `x' = Π · x` if `tq->use_rotation` is true.
2. MSE: nearest centroid in `tq->codebook_mse[b]`; write `idx`.
3. Compute residual `r = x' − dequant_mse(idx)`.
4. QJL: sign of `S · r`, packed into `qjl_signs[]`.
5. Compute `rnorm = ||r||₂`.

`turboquant_dequant_prod(tq, idx, qjl_signs, rnorm, out)`:

1. Reconstruct base `b = codebook_mse[idx]`.
2. Add `rnorm × (S · qjl_signs) / ||S · qjl_signs||₂`.
3. Optional inverse rotate: `x = Π^T · x'` (orthonormal, so transpose is inverse).

The same flow applies to RotorQuant with the rotation mathematics replaced by Givens / quaternion composition.

## 4. Key data structures

### 4.1 `TurboQuant`

```c
typedef struct {
  int       d;             /* head_dim */
  int       b;             /* target bit-width (2.5–4 recommended) */
  float    *Pi;            /* d×d rotation, NULL if disabled */
  float    *S;             /* d×d random projection for QJL */
  float   **codebook_mse;  /* codebook[b][2^b], Lloyd-Max centroids */
  int      *cb_sizes;      /* number of centroids per bit-width */
  bool      use_rotation;
} TurboQuant;
```

Initialised once per model load (`turboquant_init`). The codebook is precomputed offline (Lloyd-Max), embedded as a static array.

### 4.2 `RotorQuant`

Same shape, but `rotations` replaces `Pi` with a parameter buffer whose layout depends on `mode`:
- `RQ_MODE_PLANAR`: angles for ⌊d/2⌋ Givens rotations.
- `RQ_MODE_ISO`: quaternions for ⌊d/4⌋ 4D rotations.

## 5. Algorithms

### 5.1 MSE quantisation

For each dimension (or grouped pair of dimensions in some configurations) the codebook holds `2^b` centroids (Lloyd-Max optimal under a unit-variance Gaussian assumption). The forward step is a nearest-neighbour search; cost is O(d × 2^b) per vector.

### 5.2 QJL signature

QJL = "Quaternion JL" projection (a random Gaussian projection). The 1-bit signature is the elementwise sign of `S · r`. Reconstructing the residual from the signature is unbiased in expectation: `E[r̂] = c × r` for some constant.

### 5.3 Codebook generation

Codebooks are generated offline by a Lloyd-Max iteration (split → assign → recompute centroids until convergence) and embedded as static C arrays. Implementations following this design SHOULD use the bundled codebooks rather than regenerating them at runtime to keep `turboquant_init` cheap.

## 6. Concurrency model

`TurboQuant` and `RotorQuant` instances are read-only after `init`. Multiple threads may share one instance; no internal locks are required.

The MSA pool's quantised state is per-pool, single-writer / many-reader as documented in `TDD_msa.md` §6.

## 7. Trade-offs considered

| Decision | Chosen | Rejected | Rationale |
|---|---|---|---|
| Rotation form | Dense (TurboQuant) + structured (RotorQuant) alternative | Single dense rotation only | Dense rotation is ~`d²` floats; structured rotors are ~`d` floats and recover most of the quality. Useful at edge memory budgets. |
| Bit budget | 4-bit total (3-bit MSE + 1-bit QJL) | 8-bit per dim | 4-bit dual-state matches the 8× memory reduction target without unbiased loss on cosine routing. |
| Codebooks | Pre-computed Lloyd-Max, embedded static | Online learning | Avoids runtime cost; the codebooks are deterministic and fully captured in source. |
| Reconstruction objective | Inner-product-optimal | MSE-optimal | Cosine routing in MSA depends on dot products, not absolute reconstruction; an inner-product-optimal estimator produces tighter routing scores. |

## 8. Known limitations

- A single global instance (`g_tq` / `g_rq`) is exported by `microgpt_msa.c` when `ENABLE_TURBOQUANT` / `ENABLE_ROTORQUANT` is defined. Multi-organelle deployments using different head dimensions need to arrange compatible instances.
- The codebooks assume unit-variance Gaussian K vectors; severe distribution shifts (e.g., post-prompt tokens with sharply different magnitudes) reduce reconstruction quality.
- TurboQuant's dense `Π` is `d × d × float` = 1 KB per d = 16; for d ≥ 64 the rotation matrix dominates the per-vector overhead, motivating RotorQuant.

## 9. References

- `docs/research/RESEARCH_TURBO_QUANT.md`, `RESEARCH_ROTOR_QUANT.md`.
- TurboQuant: Lloyd-Max codebook with QJL residual.
- RotorQuant: Givens / quaternion rotor variant.

## 10. Revision history

| Version | Date | Change |
|---|---|---|
| 1.0 | 2026-04-30 | Initial extraction. |
