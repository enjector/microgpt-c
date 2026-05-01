# BS_ekan — Behaviour Specification (EKAN B-spline activations)

**Document ID:** BS-EKAN-001
**Version:** 1.0
**Status:** DRAFT

## RFC 2119

The key words MUST, MUST NOT, REQUIRED, SHALL, SHALL NOT, SHOULD, SHOULD NOT, RECOMMENDED, MAY, and OPTIONAL in this document are to be interpreted as described in RFC 2119.

## 1. Scope

Behavioural contract of the fixed-point cubic B-spline edge activation primitives in `microgpt_ekan.h` and the small `microgpt_ekan_network.h` wrapper.

## 2. Type contracts

### 2.1 `EKAN_Organelle`

**Invariants:**
- INV-EKAN-001: `org->num_edges ≤ MAX_EKAN_EDGES` (128).
- INV-EKAN-002: `org->grid_size ≤ MAX_SPLINE_GRID_SIZE` (64).
- INV-EKAN-003: `org->num_points == org->grid_size + EKAN_DEGREE` (3).
- INV-EKAN-004: `org->knots[]` is a non-decreasing fixed-point knot vector of length `grid_size + EKAN_DEGREE + 1`.

### 2.2 Fixed-point arithmetic

**Invariants:**
- INV-EKAN-010: `BONSAI_FP_SCALE == 1,000,000` (a 64-bit-safe scale).
- INV-EKAN-011: `fp_mul(a, b) == (int32_t)((int64_t)a * b / BONSAI_FP_SCALE)` (cast to int64 to avoid overflow during multiplication).
- INV-EKAN-012: `fp_div(num, denom) == 0` when `denom == 0` (singularity protection at spline boundaries).

## 3. Operation contracts

### 3.1 `ekan_find_knot_span_fp(x, n_points, knots)`

**Postconditions:** Returns `i` such that `knots[i] ≤ x < knots[i+1]`. Edge cases:
- `x ≥ knots[n_points]` → return `n_points − 1` (right boundary).
- `x ≤ knots[EKAN_DEGREE]` → return `EKAN_DEGREE` (left boundary).
- Otherwise binary search.

Cost: O(log n_points). Force-inlined.

### 3.2 `ekan_bspline_basis_fp(x, span, knots, N)`

**Postconditions:** Writes `EKAN_DEGREE + 1 = 4` non-zero basis-function values into `N[]` for the cubic B-spline at `x` over the active knot span. Implementation is the Cox-de Boor recurrence.

### 3.3 `ekan_edge_pulse(x, span, knots, control_points, base_weight)`

**Postconditions:** Returns `fp_mul(x, base_weight) + sum_i fp_mul(control_points[span − degree + i], N[i])` for `i ∈ [0, EKAN_DEGREE]`. Force-inlined.

## 4. Invariants table

| ID | Invariant |
|---|---|
| INV-EKAN-001..004 | Organelle bounds and knot-vector length. |
| INV-EKAN-010..012 | Fixed-point arithmetic correctness. |
| INV-EKAN-020 | All primitives are pure functions of their inputs; no global state. |
| INV-EKAN-021 | All primitives are force-inlined to keep the hot path branch-free except for `fp_div`'s zero-denominator guard. |

## 5. Errors

No error codes; preconditions are documented per function.

## 6. Concurrency

All primitives are pure; multi-threaded callers are safe by construction.

## 7. Performance SLOs

No specific SLOs published; design target is "sub-nanosecond per edge pulse on modern x86-64". The fixed-point design is intended for MCUs without an FPU.

## 8. Scenarios

### SCN-EKAN-001: Single-edge activation

A caller passes a fixed-point market feature `x`, locates the span via `ekan_find_knot_span_fp`, evaluates the cubic basis via `ekan_bspline_basis_fp`, and computes the activation via `ekan_edge_pulse`.

## 9. Acceptance criteria

| ID | Verifies | Test |
|---|---|---|
| ACC-EKAN-001 | INV-EKAN-001..004, 010..012 | `tests/test_microgpt_ekan.c` |
| ACC-EKAN-002 | EKAN-Network harness | `tests/test_microgpt_ekan_network.c` |

## 10. Cross-references

- **TDD:** `TDD_ekan.md`
- **Source:** `src/microgpt_ekan.h`, `src/microgpt_ekan_network.h`

## 11. Revision history

| Version | Date | Change |
|---|---|---|
| 1.0 | 2026-04-30 | Initial extraction. |
