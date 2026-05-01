# BS_geodesic — Behaviour Specification

**Document ID:** BS-GEO-001
**Version:** 1.0
**Status:** DRAFT

## RFC 2119

The key words MUST, MUST NOT, REQUIRED, SHALL, SHALL NOT, SHOULD, SHOULD NOT, RECOMMENDED, MAY, and OPTIONAL in this document are to be interpreted as described in RFC 2119.

## 1. Scope

Behavioural contract of the fixed-dimension Riemannian geodesic solver in `microgpt_geodesic.{h,c}`.

## 2. Type contracts

### 2.1 `GeodesicSolver`

**Invariants:**
- INV-GEO-001: `solver->steps > 0`; `solver->epsilon > 0`; `solver->clamp > 0`.
- INV-GEO-002: `geo_solver_init(s, 15, 1e-4, 2.0)` is a documented sane default.

### 2.2 `GeoMetricFn`

**Invariants:**
- INV-GEO-010: Every metric function SHALL write a symmetric positive-definite (SPD) `GEO_DIMS × GEO_DIMS` tensor into `G_out`.
- INV-GEO-011: Built-in metrics `geo_metric_flat`, `geo_metric_diagonal`, `geo_metric_behavioral`, `geo_metric_fraud` SHALL be SPD by construction.

### 2.3 `GeodesicResult`

**Invariants:**
- INV-GEO-020: `result.tension`, `result.gauge_work`, `result.total_risk` are non-negative.
- INV-GEO-021: `result.steps_taken ≤ solver->steps`; equal in the typical case unless an early-termination criterion fires.

## 3. Operation contracts

### 3.1 `geo_solver_init(s, steps, epsilon, clamp)`

**Postconditions:** Populates the struct fields. Idempotent.

### 3.2 `geo_compute_tension(solver, metric_fn, metric_data, deviation, gauge, gauge_weight)`

**Preconditions:** `metric_fn` non-NULL; `deviation` length `GEO_DIMS`; `gauge` MAY be NULL.

**Postconditions:** Returns a `GeodesicResult`. On a flat metric (or `gauge == NULL`), `total_risk == tension`. With a non-NULL `gauge`, `total_risk = tension + gauge_weight × gauge_work`.

### 3.3 `geo_compute_euclidean(deviation, gauge, gauge_weight)`

**Postconditions:** Returns the closed-form Euclidean fast path; equivalent to `geo_compute_tension` with `geo_metric_flat`.

### 3.4 Public matrix utilities

`geo_dot`, `geo_norm`, `geo_norm_sq`, `geo_mat_vec`, `geo_quadratic_form`, `geo_identity`, `geo_is_identity(M, tol)`, `geo_invert_matrix`, `geo_christoffel(solver, metric_fn, metric_data, x, k, i, j)` — exposed for testing and for use by callers that need the primitives directly. Each is a pure function of its inputs.

## 4. Invariants table

| ID | Invariant |
|---|---|
| INV-GEO-001..002 | Solver field non-negativity and sane defaults. |
| INV-GEO-010..011 | Metric functions produce SPD tensors. |
| INV-GEO-020..021 | Result fields are non-negative; bounded `steps_taken`. |
| INV-GEO-030 | `geo_compute_tension` falls back to `geo_compute_euclidean` when `geo_is_identity(G, tol)` returns true. |
| INV-GEO-031 | `GEO_DIMS` is a compile-time constant; the entire engine constant-folds against it. |

## 5. Errors

The geodesic API does not return error codes; misuse (non-SPD metric, NULL `metric_fn`) is undefined behaviour.

## 6. Concurrency

`GeodesicSolver` is read-only after `geo_solver_init`; many threads MAY share an instance.

## 7. Performance SLOs

| ID | SLO | Notes |
|---|---|---|
| SLO-GEO-001 | A geodesic call at `GEO_DIMS=40`, `steps=15` SHOULD complete in ≤ 1 ms on a modern x86-64 / Apple Silicon CPU. | Empirical, no formal benchmark. |

## 8. Scenarios

### SCN-GEO-001: Wiring anchor classifier

The wiring layer embeds a prompt as keyword counts on each of `GEO_DIMS` axes, then calls `geo_compute_tension(solver, geo_metric_flat, NULL, embedding, NULL, 0)` and picks the axis of minimum tension as the predicted anchor family.

### SCN-GEO-002: Fraud deviation scoring

A fraud profile encodes the cardholder's baseline as the origin; the live transaction is a deviation. `geo_metric_fraud` (with its `GeoFraudMetricCtx { stiffness, nlp_coupling }` parameters) plus an optional `GeoGaugeField` produces a `total_risk` scalar.

## 9. Acceptance criteria

| ID | Verifies | Test |
|---|---|---|
| ACC-GEO-001 | INV-GEO-001..002, 020..021 | `tests/test_microgpt_geodesic.c` |
| ACC-GEO-002 | INV-GEO-030 (identity fast-path) | As above |
| ACC-GEO-003 | All four built-in metrics produce SPD G | As above |

## 10. Cross-references

- **TDD:** `TDD_geodesic.md`
- **Source:** `src/microgpt_geodesic.{h,c}`
- **Tests:** `tests/test_microgpt_geodesic.c` (16/16)
- **Downstream:** `BS_wiring.md` (anchor classifier)

## 11. Revision history

| Version | Date | Change |
|---|---|---|
| 1.0 | 2026-04-30 | Initial extraction. |
