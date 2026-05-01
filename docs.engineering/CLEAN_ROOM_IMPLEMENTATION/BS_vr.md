# BS_vr — Behaviour Specification (Vietoris-Rips Persistent Cohomology)

**Document ID:** BS-VR-001
**Version:** 1.0
**Status:** DRAFT

## RFC 2119

The key words MUST, MUST NOT, REQUIRED, SHALL, SHALL NOT, SHOULD, SHOULD NOT, RECOMMENDED, MAY, and OPTIONAL in this document are to be interpreted as described in RFC 2119.

## 1. Scope

Behavioural contract of `microgpt_vr.{h,c}` — Vietoris-Rips filtration over a small point cloud, F₂ persistent cohomology, persistence diagram, and Betti numbers β₀, β₁, β₂.

## 2. Type contracts

### 2.1 `VREngine`

**Invariants:**
- INV-VR-001: `engine->n_dims ∈ [1, VR_MAX_DIMS]` (12).
- INV-VR-002: `engine->max_dim ∈ {0, 1, 2}`.
- INV-VR-003: `engine->max_radius_sq == max_radius²`.

### 2.2 `VRPoint`

**Invariants:**
- INV-VR-010: `point.coords` carries `point.n_dims ≤ VR_MAX_DIMS` valid components.

### 2.3 `VRDiagram`

**Invariants:**
- INV-VR-020: `diagram->count ≤ VR_MAX_INTERVALS` (512).
- INV-VR-021: For every interval, `dimension ∈ {0, 1, 2}`, `birth ≤ death`.

## 3. Operation contracts

### 3.1 `vr_engine_init(engine, max_radius, max_dim, n_dims)`

**Preconditions:** `max_radius > 0`; `max_dim ∈ {0, 1, 2}`; `n_dims ∈ [1, 12]`.

**Postconditions:** Engine fields populated.

### 3.2 `vr_compute(engine, points, n_points, min_persistence)`

**Preconditions:** `n_points ≤ VR_MAX_PTS` (64); each `points[i].n_dims == engine->n_dims`.

**Postconditions:** Returns a `VRDiagram` containing intervals whose persistence (`death − birth`) is ≥ `min_persistence`. Failure modes (cap exceedance internally) are silent — INV-VR-030 documents this caveat.

### 3.3 `vr_betti_numbers(engine, points, n_points, at_radius, min_persistence, betti_out[3])`

**Postconditions:** Writes β₀, β₁, β₂ at the queried filtration radius.

### 3.4 `vr_betti_at(diagram, dim, filtration)`

**Postconditions:** Returns the count of intervals at dimension `dim` with `birth ≤ filtration < death`.

### 3.5 `vr_make_point(coords, n_dims, id)`

**Postconditions:** Returns a `VRPoint` populated from the array (helper / convenience).

## 4. Invariants table

| ID | Invariant |
|---|---|
| INV-VR-001..003 | Engine field bounds. |
| INV-VR-010 | Per-point dimensionality bound. |
| INV-VR-020..021 | Diagram size and interval validity. |
| INV-VR-030 | Internal cap exceedance (`VR_MAX_EDGES`, `VR_MAX_TRIANGLES`, `VR_MAX_SIMPLICES`) silently truncates the computation; callers SHALL ensure their point clouds fit. |
| INV-VR-031 | Coefficient field is F₂; β computations are unchanged across coefficient choice. |

## 5. Errors

No error codes; misuse is undefined.

## 6. Concurrency

The engine is read-only after init; multiple threads MAY share it for independent point clouds.

## 7. Performance SLOs

No specific SLOs are published for the VR engine; the use case is offline / batched.

## 8. Scenarios

### SCN-VR-001: Topological feature on a small embedding

A demo computes Betti numbers of a 30-point embedding to detect a topological hole that distance metrics alone do not expose.

## 9. Acceptance criteria

| ID | Verifies | Test |
|---|---|---|
| ACC-VR-001 | INV-VR-001..021 | `tests/test_microgpt_vr.c` |

## 10. Cross-references

- **TDD:** `TDD_vr.md`
- **Source:** `src/microgpt_vr.{h,c}`

## 11. Revision history

| Version | Date | Change |
|---|---|---|
| 1.0 | 2026-04-30 | Initial extraction. |
