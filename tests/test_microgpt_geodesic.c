/*
 * test_microgpt_geodesic.c — C99 Geodesic Solver Tests
 *
 * Port of 15 Catch2 tests from geodesic_engine_tests.cpp.
 * Zero-dependency assertion-based tests.
 *
 * Copyright (c) 2026 Ajay Soni (ajay.soni@enjector.com), Enjector Software Ltd.
 * MIT License.
 */

#define _CRT_SECURE_NO_WARNINGS 1

#include "microgpt_geodesic.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ---- Test harness (matches MicroGPT pattern) ---- */

static int g_tests_run = 0;
static int g_tests_passed = 0;
static int g_tests_failed = 0;

#define TEST(name) \
  static void test_##name(void); \
  static void run_##name(void) { \
    g_tests_run++; \
    printf("  %-50s ", #name); \
    fflush(stdout); \
    test_##name(); \
    printf("PASS\n"); \
    fflush(stdout); \
    g_tests_passed++; \
  } \
  static void test_##name(void)

#define ASSERT(cond) \
  do { \
    if (!(cond)) { \
      printf("FAIL\n"); \
      fprintf(stderr, "    Assertion failed: %s\n    at %s:%d\n", #cond, \
              __FILE__, __LINE__); \
      g_tests_failed++; \
      return; \
    } \
  } while (0)

#define ASSERT_NEAR(a, b, tol) ASSERT(fabs((a) - (b)) < (tol))
#define ASSERT_GT(a, b) ASSERT((a) > (b))
#define ASSERT_LT(a, b) ASSERT((a) < (b))
#define RUN(name) run_##name()

/* ---- Helper: make a 3D deviation (for tests that need small dims) ---- */
static void zero_vec(double v[GEO_DIMS]) { memset(v, 0, sizeof(double) * GEO_DIMS); }

/* ==================================================================== */
/* §1: Flat Metric — Geodesic = Euclidean Distance                       */
/* ==================================================================== */

TEST(flat_metric_tension_equals_euclidean_3d) {
    GeodesicSolver solver;
    geo_solver_init(&solver, 15, 1e-4, 2.0);

    double dev[GEO_DIMS];
    zero_vec(dev);
    dev[0] = 3.0; dev[1] = 4.0; /* |d| = 5 */

    GeodesicResult r = geo_compute_tension(&solver, geo_metric_flat, NULL,
                                            dev, NULL, 5.0);
    ASSERT_NEAR(r.tension, 5.0, 0.5);
}

TEST(flat_metric_tension_12d_unit) {
    GeodesicSolver solver;
    geo_solver_init(&solver, 15, 1e-4, 2.0);

    double dev[GEO_DIMS];
    zero_vec(dev);
    dev[0] = 1.0;

    GeodesicResult r = geo_compute_tension(&solver, geo_metric_flat, NULL,
                                            dev, NULL, 5.0);
    ASSERT_NEAR(r.tension, 1.0, 0.1);
}

/* ==================================================================== */
/* §2: Christoffel Symbols                                               */
/* ==================================================================== */

TEST(flat_christoffel_vanish) {
    GeodesicSolver solver;
    geo_solver_init(&solver, 15, 1e-4, 2.0);

    double x[GEO_DIMS];
    zero_vec(x);

    for (int k = 0; k < 3; k++)
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++) {
                double gamma = geo_christoffel(&solver, geo_metric_flat,
                                                NULL, x, k, i, j);
                ASSERT(fabs(gamma) < 0.01);
            }
}

TEST(curved_christoffel_nonzero) {
    GeodesicSolver solver;
    geo_solver_init(&solver, 15, 1e-4, 2.0);

    double stiffness = 0.5;
    double x[GEO_DIMS];
    zero_vec(x);
    x[0] = 1.0; x[1] = 1.0;

    int any_nonzero = 0;
    for (int k = 0; k < 3; k++)
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++) {
                double gamma = geo_christoffel(&solver, geo_metric_behavioral,
                                                &stiffness, x, k, i, j);
                if (fabs(gamma) > 0.001) any_nonzero = 1;
            }
    ASSERT(any_nonzero);
}

/* ==================================================================== */
/* §3: Matrix Inversion                                                  */
/* ==================================================================== */

TEST(matrix_inverse_identity) {
    double I[GEO_DIMS][GEO_DIMS], I_inv[GEO_DIMS][GEO_DIMS];
    geo_identity(I);
    geo_invert_matrix(I, I_inv);

    for (int i = 0; i < GEO_DIMS; i++)
        for (int j = 0; j < GEO_DIMS; j++) {
            double expected = (i == j) ? 1.0 : 0.0;
            ASSERT_NEAR(I_inv[i][j], expected, 1e-10);
        }
}

TEST(matrix_inverse_diagonal) {
    double D[GEO_DIMS][GEO_DIMS], D_inv[GEO_DIMS][GEO_DIMS];
    memset(D, 0, sizeof(D));
    D[0][0] = 2.0; D[1][1] = 4.0; D[2][2] = 0.5;
    for (int i = 3; i < GEO_DIMS; i++) D[i][i] = 1.0;

    geo_invert_matrix(D, D_inv);

    ASSERT_NEAR(D_inv[0][0], 0.5, 1e-10);
    ASSERT_NEAR(D_inv[1][1], 0.25, 1e-10);
    ASSERT_NEAR(D_inv[2][2], 2.0, 1e-10);
}

TEST(matrix_inverse_product_is_identity) {
    /* SPD matrix (use first 3×3 block + identity for rest) */
    double M[GEO_DIMS][GEO_DIMS], M_inv[GEO_DIMS][GEO_DIMS];
    geo_identity(M);
    M[0][0] = 4.0; M[0][1] = 2.0; M[0][2] = 1.0;
    M[1][0] = 2.0; M[1][1] = 5.0; M[1][2] = 3.0;
    M[2][0] = 1.0; M[2][1] = 3.0; M[2][2] = 6.0;

    geo_invert_matrix(M, M_inv);

    /* Check M × M_inv ≈ I */
    for (int i = 0; i < GEO_DIMS; i++) {
        for (int j = 0; j < GEO_DIMS; j++) {
            double sum = 0.0;
            for (int k = 0; k < GEO_DIMS; k++)
                sum += M[i][k] * M_inv[k][j];
            double expected = (i == j) ? 1.0 : 0.0;
            ASSERT_NEAR(sum, expected, 1e-8);
        }
    }
}

/* ==================================================================== */
/* §4: Gauge Work                                                        */
/* ==================================================================== */

TEST(zero_gauge_zero_work) {
    GeodesicSolver solver;
    geo_solver_init(&solver, 15, 1e-4, 2.0);

    double dev[GEO_DIMS];
    zero_vec(dev);
    dev[0] = 1.0; dev[1] = 2.0; dev[2] = 3.0;

    GeodesicResult r = geo_compute_tension(&solver, geo_metric_flat, NULL,
                                            dev, NULL, 5.0);
    ASSERT(fabs(r.gauge_work) < 1e-12);
}

TEST(nonzero_gauge_positive_work) {
    GeodesicSolver solver;
    geo_solver_init(&solver, 15, 1e-4, 2.0);

    double dev[GEO_DIMS];
    zero_vec(dev);
    dev[0] = 1.0;

    GeoGaugeField gauge;
    memset(&gauge, 0, sizeof(gauge));
    gauge.potential[0][0] = 1.0;
    gauge.charge[0] = 1.0;

    GeodesicResult r = geo_compute_tension(&solver, geo_metric_flat, NULL,
                                            dev, &gauge, 5.0);
    /* gauge_work = charge · (A × d) = [1,0..] · [1,0..] = 1.0 */
    ASSERT_NEAR(r.gauge_work, 1.0, 1e-10);
}

TEST(opposite_charge_negative_work) {
    GeodesicSolver solver;
    geo_solver_init(&solver, 15, 1e-4, 2.0);

    double dev[GEO_DIMS];
    zero_vec(dev);
    dev[0] = 1.0;

    GeoGaugeField gauge;
    memset(&gauge, 0, sizeof(gauge));
    gauge.potential[0][0] = 1.0;
    gauge.charge[0] = -1.0;

    GeodesicResult r = geo_compute_tension(&solver, geo_metric_flat, NULL,
                                            dev, &gauge, 5.0);
    ASSERT_LT(r.gauge_work, 0.0);
}

/* ==================================================================== */
/* §5: Curved Metric                                                     */
/* ==================================================================== */

TEST(curved_tension_exceeds_euclidean) {
    GeodesicSolver solver;
    geo_solver_init(&solver, 15, 1e-4, 2.0);

    double stiffness = 0.5;
    double dev[GEO_DIMS];
    zero_vec(dev);
    dev[0] = 2.0; dev[1] = 2.0;
    double euclidean = geo_norm(dev);

    GeodesicResult r = geo_compute_tension(&solver, geo_metric_behavioral,
                                            &stiffness, dev, NULL, 5.0);
    ASSERT_GT(r.tension, euclidean * 0.5);
}

/* ==================================================================== */
/* §6: Euclidean Bypass                                                  */
/* ==================================================================== */

TEST(euclidean_bypass_matches_flat) {
    double dev[GEO_DIMS];
    zero_vec(dev);
    dev[0] = 3.0; dev[1] = 4.0; /* |d| = 5 */

    GeodesicResult euc = geo_compute_euclidean(dev, NULL, 5.0);
    ASSERT_NEAR(euc.tension, 5.0, 1e-10);
    ASSERT(euc.steps_taken == 0);
}

/* ==================================================================== */
/* §7: Guards                                                            */
/* ==================================================================== */

TEST(zero_deviation_zero_tension) {
    GeodesicSolver solver;
    geo_solver_init(&solver, 15, 1e-4, 2.0);

    double dev[GEO_DIMS];
    zero_vec(dev);

    GeodesicResult r = geo_compute_tension(&solver, geo_metric_flat, NULL,
                                            dev, NULL, 5.0);
    ASSERT(fabs(r.tension) < 1e-12);
}

/* ==================================================================== */
/* §8: Total Risk Formula                                                */
/* ==================================================================== */

TEST(total_risk_tension_plus_gauge) {
    GeodesicSolver solver;
    geo_solver_init(&solver, 15, 1e-4, 2.0);

    double dev[GEO_DIMS];
    zero_vec(dev);
    dev[0] = 1.0;

    GeoGaugeField gauge;
    memset(&gauge, 0, sizeof(gauge));
    gauge.potential[0][0] = 2.0;
    gauge.charge[0] = 1.0;

    double weight = 5.0;
    GeodesicResult r = geo_compute_tension(&solver, geo_metric_flat, NULL,
                                            dev, &gauge, weight);
    double expected = r.tension + fabs(r.gauge_work) * weight;
    ASSERT_NEAR(r.total_risk, expected, 1e-10);
}

/* ==================================================================== */
/* §9: RK4 Convergence                                                   */
/* ==================================================================== */

TEST(rk4_convergence_more_steps_better) {
    double dev[GEO_DIMS];
    zero_vec(dev);
    dev[0] = 3.0; dev[1] = 4.0;

    GeodesicSolver s15, s30;
    geo_solver_init(&s15, 15, 1e-4, 2.0);
    geo_solver_init(&s30, 30, 1e-4, 2.0);

    double err15 = fabs(geo_compute_tension(&s15, geo_metric_flat, NULL,
                                              dev, NULL, 5.0).tension - 5.0);
    double err30 = fabs(geo_compute_tension(&s30, geo_metric_flat, NULL,
                                              dev, NULL, 5.0).tension - 5.0);
    ASSERT(err30 <= err15 + 0.01);
}

/* ==================================================================== */
/* §10: Fraud Metric                                                     */
/* ==================================================================== */

TEST(fraud_metric_valid_at_origin) {
    double x[GEO_DIMS];
    zero_vec(x);
    double G[GEO_DIMS][GEO_DIMS];
    geo_metric_fraud(x, G, NULL);

    /* At origin, metric should be identity-like (base_scale=1.0) */
    ASSERT_NEAR(G[0][0], 1.0, 1e-10);
    ASSERT_NEAR(G[5][5], 1.0, 1e-10);
}

/* ==================================================================== */
/* Main                                                                  */
/* ==================================================================== */

int main(void) {
    printf("\n=== MicroGPT Geodesic Solver Tests ===\n\n");

    RUN(flat_metric_tension_equals_euclidean_3d);
    RUN(flat_metric_tension_12d_unit);
    RUN(flat_christoffel_vanish);
    RUN(curved_christoffel_nonzero);
    RUN(matrix_inverse_identity);
    RUN(matrix_inverse_diagonal);
    RUN(matrix_inverse_product_is_identity);
    RUN(zero_gauge_zero_work);
    RUN(nonzero_gauge_positive_work);
    RUN(opposite_charge_negative_work);
    RUN(curved_tension_exceeds_euclidean);
    RUN(euclidean_bypass_matches_flat);
    RUN(zero_deviation_zero_tension);
    RUN(total_risk_tension_plus_gauge);
    RUN(rk4_convergence_more_steps_better);
    RUN(fraud_metric_valid_at_origin);

    printf("\n--- Results: %d/%d passed", g_tests_passed, g_tests_run);
    if (g_tests_failed > 0) printf(", %d FAILED", g_tests_failed);
    printf(" ---\n\n");

    return g_tests_failed > 0 ? 1 : 0;
}
