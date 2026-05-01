# TDD_geodesic — Technical Design Document

**Document ID:** TDD-GEO-001
**Version:** 1.0
**Status:** DRAFT
**Paired BS:** `BS_geodesic.md`
**Sources:** `src/microgpt_geodesic.{h,c}`

## 1. Overview

The geodesic engine is a fixed-dimension Riemannian solver, ported from the EnX-cpp `geodesic_engine.hpp` (C++17, template-driven) into pure C99. It is the state-space-distance primitive used by the wiring organelle's anchor classifier and (prospectively) by the fraud-detection vertical's deviation-from-baseline scoring.

Default dimension `GEO_DIMS = 40` in this fork; the sibling fraud-detection codebase uses 12. The fork value was bumped from 12 to 20 (and again to 40) to give every held-out wiring template family a unique axis; this eliminated the slot collisions that capped Phase 2 anchor classification at 80 % (16/20).

## 2. Architecture

```
   query position x ∈ ℝ^GEO_DIMS
                     │
                     ▼
              GeoMetricFn(x, G_out)
              fills G[GEO_DIMS][GEO_DIMS] (SPD)
                     │
                     ▼
        Cholesky decomposition + cache
        (geo_invert_matrix, geo_is_identity fast-path)
                     │
                     ▼
        Christoffel symbols Γᵏᵢⱼ via finite differences
                     │
                     ▼
        RK4 integration of geodesic ODE
                     │
                     ▼
        GeodesicResult { tension, gauge_work, total_risk,
                         final_position, final_velocity, steps_taken }
```

Optional `GeoGaugeField` adds path-dependent work (used for coercion / romance-scam detection in the fraud metric).

## 3. Key data structures

### 3.1 `GeodesicSolver`

```c
typedef struct {
  int    steps;     /* RK4 step count */
  double epsilon;   /* finite-difference epsilon for Γ */
  double clamp;     /* maximum velocity magnitude per step */
} GeodesicSolver;
```

### 3.2 `GeodesicResult`

```c
typedef struct {
  double tension;
  double gauge_work;
  double total_risk;     /* tension + gauge_weight × gauge_work */
  double final_position[GEO_DIMS];
  double final_velocity[GEO_DIMS];
  int    steps_taken;
} GeodesicResult;
```

### 3.3 `GeoGaugeField`

```c
typedef struct {
  double potential[GEO_DIMS][GEO_DIMS];
  double charge[GEO_DIMS];
} GeoGaugeField;
```

`gauge_work` accumulates `charge · ∂A` along the path.

### 3.4 `GeoMetricFn`

```c
typedef void (*GeoMetricFn)(const double x[GEO_DIMS],
                            double G_out[GEO_DIMS][GEO_DIMS],
                            void *user_data);
```

Built-in metrics: `geo_metric_flat`, `geo_metric_diagonal`, `geo_metric_behavioral`, `geo_metric_fraud`. Each writes into the caller's `G_out` and may consult `user_data` for parameters (e.g., `GeoFraudMetricCtx { stiffness, nlp_coupling }`).

## 4. Algorithms

### 4.1 RK4 geodesic integration

Geodesic ODE in coordinates: `ẍᵏ = − Γᵏᵢⱼ ẋⁱ ẋʲ`. RK4 integrates `(x, ẋ)` with step size `1 / steps`. At each substep, Christoffels are computed from the metric via finite differences.

### 4.2 Identity-cache fast-path

If `geo_is_identity(G, tol)` returns true, the solver short-circuits to Euclidean distance (`geo_compute_euclidean`) and skips Christoffel evaluation. This handles the `geo_metric_flat` case at near-zero cost.

### 4.3 Cholesky decomposition

For SPD metrics, `geo_invert_matrix` does a Cholesky factorisation (no pivoting needed for SPD), then forward / back substitution. Costs O(N³) but N is fixed at GEO_DIMS, so the inner loop unrolls cleanly.

### 4.4 Tension and gauge work

`tension = ||final_velocity||² × something_steps_taken-dependent` (the precise normalisation matches the EnX-cpp port). `gauge_work` is the path integral of `charge · A · dx`. `total_risk = tension + gauge_weight × gauge_work` is the user-facing scalar that feeds into the wiring classifier or fraud-scoring path.

## 5. Concurrency model

`GeodesicSolver`, `GeoGaugeField`, and metric functions are read-only after construction. `geo_compute_tension` writes only to its return value and stack. Multiple threads may run independent computations against the same solver instance.

## 6. Trade-offs considered

| Decision | Chosen | Rejected | Rationale |
|---|---|---|---|
| Dimension | Compile-time `GEO_DIMS` macro | Runtime-variable | Constant-folding the inner-loop bounds saves ~30 % on the per-call cost; `GEO_DIMS` is set per-fork at build time. |
| Integrator | RK4 | Symplectic / leapfrog | Geodesic ODE is not Hamiltonian in the conserved-energy sense; RK4 is acceptable and well-understood. |
| Inversion | Cholesky | LU with pivoting | SPD assumption removes the need for pivoting; Cholesky halves the operation count. |
| Built-in metrics | `flat`, `diagonal`, `behavioral`, `fraud` | Single generic + user supplies all | Built-ins document the intended use cases without requiring callers to re-derive them. |

## 7. Known limitations

- Fixed dimension; cross-vertical reuse requires recompilation with a new `GEO_DIMS`.
- Finite-difference Christoffels are O(GEO_DIMS³ × steps) per call — at GEO_DIMS=40, this is ≈ 64K Christoffel evaluations per geodesic. Acceptable for batch-of-thousands; not for hot-path streaming.
- Gauge field is per-metric; coupling between gauge and metric (gauge-induced metric distortion) is not modelled.
- The metric must be SPD; non-SPD metrics fail Cholesky silently (the result is undefined).

## 8. References

- `docs/research/RESEARCH_MANIFOLD_LEARNING.md`.
- EnX-cpp `geodesic_engine.hpp` (template-driven C++17 reference).
- Standard differential geometry: Christoffel symbols, geodesic equation, RK4 integration.

## 9. Revision history

| Version | Date | Change |
|---|---|---|
| 1.0 | 2026-04-30 | Initial extraction. |
