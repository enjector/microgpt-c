/*
 * microgpt_geodesic.h — C99 Riemannian Geodesic Solver (12D)
 *
 * Ported from EnX-cpp geodesic_engine.hpp (C++17, template-driven).
 * Fixed at 12 dimensions for fraud detection feature space.
 *
 * Copyright (c) 2026 Ajay Soni (ajay.soni@enjector.com), Enjector Software Ltd.
 * MIT License — see LICENSE file for details.
 *
 * Features:
 *   - RK4 geodesic integration with metric identity caching
 *   - Cholesky decomposition for SPD metric inversion
 *   - Gauge field for coercion/romance-scam detection
 *   - Built-in metrics: flat, diagonal, behavioral, fraud
 *
 * Usage:
 *   GeodesicSolver solver;
 *   geo_solver_init(&solver, 15, 1e-4, 2.0);
 *   double deviation[GEO_DIMS] = { ... };
 *   GeodesicResult result = geo_compute_tension(&solver, geo_metric_fraud, NULL,
 *                                                deviation, NULL, 5.0);
 */

#ifndef MICROGPT_GEODESIC_H
#define MICROGPT_GEODESIC_H

#include <stddef.h>

/* Default 12 (sibling fraud-detection feature space). Phase 2b bumps to
 * 20 in this fork so each of the 20 held-out template families gets a
 * unique axis in the wiring_geo_classifier slot table, eliminating the
 * slot-collisions that capped Phase 2 at 80% (16/20). */
#ifndef GEO_DIMS
#define GEO_DIMS 40
#endif

/* =========================================================================
 * Core Types
 * ========================================================================= */

typedef struct {
    double tension;
    double gauge_work;
    double total_risk;
    double final_position[GEO_DIMS];
    double final_velocity[GEO_DIMS];
    int    steps_taken;
} GeodesicResult;

typedef struct {
    int    steps;
    double epsilon;
    double clamp;
} GeodesicSolver;

typedef struct {
    double potential[GEO_DIMS][GEO_DIMS];
    double charge[GEO_DIMS];
} GeoGaugeField;

/* Metric function: given position x, write metric tensor G[12][12] */
typedef void (*GeoMetricFn)(const double x[GEO_DIMS],
                            double G_out[GEO_DIMS][GEO_DIMS],
                            void *user_data);

/* Metric context for built-in metrics */
typedef struct {
    double stiffness;
    double nlp_coupling;
} GeoFraudMetricCtx;

/* =========================================================================
 * Solver API
 * ========================================================================= */

void geo_solver_init(GeodesicSolver *s, int steps, double epsilon, double clamp);

GeodesicResult geo_compute_tension(
    const GeodesicSolver *solver,
    GeoMetricFn metric_fn,
    void *metric_data,
    const double deviation[GEO_DIMS],
    const GeoGaugeField *gauge,    /* NULL for no gauge */
    double gauge_weight
);

GeodesicResult geo_compute_euclidean(
    const double deviation[GEO_DIMS],
    const GeoGaugeField *gauge,
    double gauge_weight
);

/* =========================================================================
 * Matrix Utilities (public for testing)
 * ========================================================================= */

double geo_dot(const double a[GEO_DIMS], const double b[GEO_DIMS]);
double geo_norm(const double v[GEO_DIMS]);
double geo_norm_sq(const double v[GEO_DIMS]);
void   geo_mat_vec(const double M[GEO_DIMS][GEO_DIMS],
                   const double v[GEO_DIMS], double out[GEO_DIMS]);
double geo_quadratic_form(const double v[GEO_DIMS],
                          const double M[GEO_DIMS][GEO_DIMS]);
void   geo_identity(double M[GEO_DIMS][GEO_DIMS]);
int    geo_is_identity(const double M[GEO_DIMS][GEO_DIMS], double tol);
void   geo_invert_matrix(const double M[GEO_DIMS][GEO_DIMS],
                         double out[GEO_DIMS][GEO_DIMS]);

/* Christoffel symbol Γᵏᵢⱼ at position x (public for testing) */
double geo_christoffel(const GeodesicSolver *solver,
                       GeoMetricFn metric_fn, void *metric_data,
                       const double x[GEO_DIMS],
                       int k, int i, int j);

/* =========================================================================
 * Built-in Metric Functions
 * ========================================================================= */

void geo_metric_flat(const double x[GEO_DIMS],
                     double G[GEO_DIMS][GEO_DIMS], void *user_data);

void geo_metric_behavioral(const double x[GEO_DIMS],
                           double G[GEO_DIMS][GEO_DIMS], void *user_data);

void geo_metric_fraud(const double x[GEO_DIMS],
                      double G[GEO_DIMS][GEO_DIMS], void *user_data);

/* Diagonal metric: user_data = double scales[GEO_DIMS] */
void geo_metric_diagonal(const double x[GEO_DIMS],
                         double G[GEO_DIMS][GEO_DIMS], void *user_data);

#endif /* MICROGPT_GEODESIC_H */
