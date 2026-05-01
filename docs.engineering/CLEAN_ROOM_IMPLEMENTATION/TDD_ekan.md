# TDD_ekan — Technical Design Document (EKAN B-spline edge activations)

**Document ID:** TDD-EKAN-001
**Version:** 1.0
**Status:** DRAFT
**Paired BS:** `BS_ekan.md`
**Sources:** `src/microgpt_ekan.h`, `src/microgpt_ekan_network.h`

## 1. Overview

EKAN ("Edge Kolmogorov-Arnold Network") provides cubic B-spline activations on 1-D edges for use in tiny networks. The implementation is fixed-point (`int32_t` operations scaled by `BONSAI_FP_SCALE = 1,000,000`) so the activations can run on MCUs without floating-point hardware. All buffers are statically sized (`MAX_EKAN_EDGES = 128`, `MAX_SPLINE_GRID_SIZE = 64`) — zero allocation in the hot path.

The activation is the sum of:
- A linear bypass term: `base_weight × x`.
- A cubic B-spline term: linear combination of the four non-zero basis functions evaluated at `x` over the active knot span.

## 2. Architecture

```
   x (fixed-point input)
        │
        ▼
   ekan_find_knot_span_fp(x, n_points, knots)   ← binary search, O(log n_points)
        │
        ▼
   ekan_bspline_basis_fp(x, span, knots, N)     ← Cox-de Boor recurrence
        │                                          for cubic (4 non-zero basis fns)
        ▼
   ekan_edge_pulse(x, span, knots, control_points, base_weight)
        │
        ▼
   activation = base_activation + spline_activation
```

## 3. Key data structures

### 3.1 `EKAN_Organelle`

```c
typedef struct {
  int32_t control_points[MAX_EKAN_EDGES][MAX_SPLINE_GRID_SIZE];
  int32_t base_weights[MAX_EKAN_EDGES];
  int32_t knots[MAX_SPLINE_GRID_SIZE + EKAN_DEGREE + 1];
  int     num_edges;
  int     grid_size;
  int     num_points;   /* grid_size + EKAN_DEGREE */
} EKAN_Organelle;
```

`EKAN_DEGREE = 3` is fixed (cubic). The knot vector length is `grid_size + degree + 1` per the standard B-spline definition.

## 4. Algorithms

### 4.1 Fixed-point arithmetic

`fp_mul(a, b) = (int64_t)a × b / SCALE` (cast to int64 to avoid overflow during the multiplication step).
`fp_div(num, denom)` returns 0 if `denom == 0` (singularity protection at spline boundaries) — this is the only branch in the hot path.

### 4.2 `ekan_find_knot_span_fp`

Binary search for the index `i` such that `knots[i] ≤ x < knots[i+1]`. Edge cases:
- `x ≥ knots[n+1]` → return `n` (right boundary).
- `x ≤ knots[degree]` → return `degree` (left boundary).
- Otherwise standard binary search.

Cost: O(log(num_points)). Force-inlined.

### 4.3 `ekan_bspline_basis_fp`

The Cox-de Boor recurrence specialised for cubic with 4 non-zero basis functions. Two scratch arrays `left[degree+1]`, `right[degree+1]` (stack-allocated) hold intermediate distances. The recurrence runs `degree` outer iterations, each producing one more level of basis-function values.

### 4.4 `ekan_edge_pulse`

Linear bypass + spline blend:
```
base_activation = fp_mul(x, base_weight)
spline_activation = sum_{i=0..degree} fp_mul(control_points[span - degree + i], N[i])
return base_activation + spline_activation
```

Force-inlined; the entire pulse compiles to a tight straight-line block.

## 5. Concurrency model

The `EKAN_Organelle` is read-only after construction; many threads may evaluate edges concurrently. Each thread has its own stack-allocated `N[degree+1]` and `left/right` arrays, so there is no shared state.

## 6. Trade-offs considered

| Decision | Chosen | Rejected | Rationale |
|---|---|---|---|
| Numeric format | Fixed-point int32 (with int64 multiplier intermediate) | Float | Targeted for MCUs without FPU; deterministic across platforms. |
| Spline degree | Fixed at 3 (cubic) | Configurable | Cubic is the standard sweet spot; fixing it lets `ekan_bspline_basis_fp` be specialised and force-inlined. |
| Buffer sizing | Static `MAX_EKAN_EDGES`, `MAX_SPLINE_GRID_SIZE` | Heap | Zero-allocation hot path. |
| Singular knot handling | `fp_div` returns 0 on `denom == 0` | Throw / abort | Spline boundaries naturally produce zero denominators in the recurrence; quietly returning 0 keeps the hot path branch-free except for this guard. |

## 7. Known limitations

- Cubic only.
- 1-D edges only — multi-dimensional KAN-style layers are layered by the user via `microgpt_ekan_network.h` (which is currently a thin abstraction).
- Fixed-point precision is ~1 ppm at the operating range; numerically tight applications should validate.
- No backward pass — EKAN activations are inference-only in the shipped V1.0 (training-time learning of control points is left to the user).

## 8. References

- Liu et al., "KAN: Kolmogorov-Arnold Networks" — original idea.
- Standard cubic B-spline theory (Cox-de Boor).

## 9. Revision history

| Version | Date | Change |
|---|---|---|
| 1.0 | 2026-04-30 | Initial extraction. |
