# TDD_vr — Technical Design Document (Vietoris-Rips Persistent Cohomology)

**Document ID:** TDD-VR-001
**Version:** 1.0
**Status:** DRAFT
**Paired BS:** `BS_vr.md`
**Sources:** `src/microgpt_vr.{h,c}`

## 1. Overview

The Vietoris-Rips engine computes persistent cohomology over F₂ for a small point cloud (`VR_MAX_PTS = 64`, `VR_MAX_DIMS = 12`). It produces persistence diagrams and Betti numbers β₀, β₁, β₂. It is the topological-feature primitive intended for fraud / time-series anomaly detection (where a topological hole in feature space carries information that a distance metric does not).

Ported from the EnX-cpp `vr_engine.hpp`. All buffers are statically sized; no dynamic allocation in the hot path.

## 2. Architecture

```
   point cloud {p_1, ..., p_n}, n ≤ VR_MAX_PTS
                     │
                     ▼
        L2 distance matrix (n × n upper triangle)
                     │
                     ▼
        flag complex filtration via bitmask clique expansion
        (≤ VR_MAX_EDGES, VR_MAX_TRIANGLES, VR_MAX_SIMPLICES)
                     │
                     ▼
        F₂ persistent cohomology
        (apparent-pair shortcut + clearing)
                     │
                     ▼
        VRDiagram { intervals[VR_MAX_INTERVALS], count }
                     │
                     ▼
        β₀, β₁, β₂ at any filtration radius via vr_betti_at
```

## 3. Key data structures

### 3.1 `VRPoint`, `VRInterval`, `VRDiagram`, `VREngine`

```c
typedef struct {
  float coords[VR_MAX_DIMS];
  int   id;
  int   n_dims;       /* actual dimensionality, ≤ VR_MAX_DIMS */
} VRPoint;

typedef struct {
  int   dimension;    /* 0, 1, or 2 */
  float birth, death;
} VRInterval;

typedef struct {
  VRInterval intervals[VR_MAX_INTERVALS];
  int        count;
} VRDiagram;

typedef struct {
  float max_radius, max_radius_sq;
  int   max_dim;      /* 0, 1, or 2 */
  int   n_dims;
} VREngine;
```

`VR_MAX_EDGES = 2048`, `VR_MAX_TRIANGLES = 4096`, `VR_MAX_SIMPLICES = 8192`. These caps are fixed at build time; exceeding them fails the computation.

## 4. Algorithms

### 4.1 Distance matrix

All-pairs L2 distance. Squared distances are stored to avoid square roots in the hot loop; the comparison against `max_radius_sq` is what gates filtration.

### 4.2 Bitmask clique expansion

Edges within `max_radius` form the 1-skeleton. Triangles are 3-cliques: for each edge `(i,j)`, scan vertices `k > j` whose edges to both `i` and `j` are within `max_radius`; the bitmask representation makes the intersection cheap.

### 4.3 Apparent-pair shortcut + clearing

Ordinary persistent cohomology over F₂ using the standard reduction matrix algorithm, with two performance shortcuts:

- **Apparent pairs**: a simplex whose boundary's lowest term is also unique in the matrix can be paired immediately, skipping reduction.
- **Clearing**: once a column is paired, dependent columns above it can be cleared. Reduces the asymptotic cost from O(n⁶) to closer to O(n⁴) in practice.

### 4.4 Persistence diagram and Betti numbers

`VRInterval { dim, birth, death }` encodes "this dim-`dim` feature appeared at radius `birth` and disappeared at radius `death`". `vr_betti_at(diagram, dim, filtration)` counts intervals at the queried dim whose `birth ≤ filtration < death`.

Min-persistence filtering (`min_persistence` arg of `vr_compute`) drops short-lived noise features.

## 5. Concurrency model

The engine is read-only after `vr_engine_init`; multiple threads may share an engine and submit independent point clouds via separate `VRDiagram` outputs. The internal scratch buffers are stack-allocated per call.

## 6. Trade-offs considered

| Decision | Chosen | Rejected | Rationale |
|---|---|---|---|
| Coefficient field | F₂ | F₃ / ℚ / ℤ_p | F₂ is sufficient for Betti-number queries; field choice doesn't affect β. |
| Max dim | 2 (β₀, β₁, β₂) | Higher | β₃+ is rarely useful for the intended fraud / time-series use case; capping at 2 keeps the simplex budget tight. |
| Static caps | Fixed buffers | Dynamic alloc | Hot-path-friendly: zero malloc, predictable memory; cap exceedance is the caller's responsibility. |
| Reduction algorithm | Apparent + clearing | Plain Smith form | Standard speedups; no novel algorithmic claim. |

## 7. Known limitations

- Fixed `VR_MAX_PTS = 64`; larger clouds need recompilation with a higher cap (and proportionally more memory).
- `n_dims ≤ VR_MAX_DIMS = 12` per point.
- Min-persistence is the only noise filter; no advanced bottleneck-distance comparison is exposed.
- The Betti query is O(diagram->count) per call; for repeated queries at many radii, callers SHOULD pre-bucket.

## 8. References

- EnX-cpp `vr_engine.hpp`.
- Edelsbrunner & Harer, "Computational Topology" — apparent pairs, clearing.
- Bauer, "Ripser" reference for the persistent cohomology algorithm.

## 9. Revision history

| Version | Date | Change |
|---|---|---|
| 1.0 | 2026-04-30 | Initial extraction. |
