/*
 * microgpt_geodesic.c — C99 Riemannian Geodesic Solver (12D)
 *
 * Ported from EnX-cpp geodesic_engine.hpp (C++17, 842 lines).
 * Fixed at 12 dimensions for fraud detection feature space.
 *
 * Copyright (c) 2026 Ajay Soni (ajay.soni@enjector.com), Enjector Software Ltd.
 * MIT License — see LICENSE file for details.
 */

#include "microgpt_geodesic.h"
#include <math.h>
#include <string.h>
#include <float.h>

#define D GEO_DIMS

/* =========================================================================
 * Vector / Matrix Utilities
 * ========================================================================= */

double geo_dot(const double a[D], const double b[D]) {
    double sum = 0.0;
    for (int i = 0; i < D; i++) sum += a[i] * b[i];
    return sum;
}

double geo_norm_sq(const double v[D]) {
    return geo_dot(v, v);
}

double geo_norm(const double v[D]) {
    return sqrt(geo_norm_sq(v));
}

void geo_mat_vec(const double M[D][D], const double v[D], double out[D]) {
    for (int i = 0; i < D; i++) {
        double sum = 0.0;
        for (int j = 0; j < D; j++) sum += M[i][j] * v[j];
        out[i] = sum;
    }
}

double geo_quadratic_form(const double v[D], const double M[D][D]) {
    double sum = 0.0;
    for (int i = 0; i < D; i++)
        for (int j = 0; j < D; j++)
            sum += v[i] * M[i][j] * v[j];
    return sum;
}

void geo_identity(double M[D][D]) {
    memset(M, 0, sizeof(double) * D * D);
    for (int i = 0; i < D; i++) M[i][i] = 1.0;
}

int geo_is_identity(const double M[D][D], double tol) {
    for (int i = 0; i < D; i++)
        for (int j = 0; j < D; j++) {
            double expected = (i == j) ? 1.0 : 0.0;
            if (fabs(M[i][j] - expected) > tol) return 0;
        }
    return 1;
}

/* =========================================================================
 * Cholesky Decomposition — G = L L^T
 * ========================================================================= */

static int cholesky_decompose(const double G[D][D], double L[D][D]) {
    memset(L, 0, sizeof(double) * D * D);
    for (int i = 0; i < D; i++) {
        for (int j = 0; j <= i; j++) {
            double sum = 0.0;
            for (int k = 0; k < j; k++) sum += L[i][k] * L[j][k];
            if (i == j) {
                double diag = G[i][i] - sum;
                if (diag <= 0.0) return 0; /* Not SPD */
                L[i][j] = sqrt(diag);
            } else {
                L[i][j] = (G[i][j] - sum) / L[j][j];
            }
        }
    }
    return 1;
}

static void forward_solve(const double L[D][D], const double b[D],
                           double y[D]) {
    for (int i = 0; i < D; i++) {
        double sum = b[i];
        for (int j = 0; j < i; j++) sum -= L[i][j] * y[j];
        y[i] = sum / L[i][i];
    }
}

static void back_solve(const double L[D][D], const double y[D],
                        double x[D]) {
    for (int i = D - 1; i >= 0; i--) {
        double sum = y[i];
        for (int j = i + 1; j < D; j++) sum -= L[j][i] * x[j];
        x[i] = sum / L[i][i];
    }
}

/* Gauss-Jordan fallback for non-SPD */
static void gauss_jordan_invert(const double M[D][D], double out[D][D]) {
    double aug[D][D * 2];
    for (int i = 0; i < D; i++) {
        for (int j = 0; j < D; j++) aug[i][j] = M[i][j];
        for (int j = 0; j < D; j++) aug[i][D + j] = (i == j) ? 1.0 : 0.0;
    }
    for (int col = 0; col < D; col++) {
        int max_row = col;
        double max_val = fabs(aug[col][col]);
        for (int row = col + 1; row < D; row++) {
            double v = fabs(aug[row][col]);
            if (v > max_val) { max_val = v; max_row = row; }
        }
        if (max_val < DBL_EPSILON * 100.0) {
            geo_identity(out);
            return;
        }
        if (max_row != col) {
            double tmp[D * 2];
            memcpy(tmp, aug[max_row], sizeof(tmp));
            memcpy(aug[max_row], aug[col], sizeof(tmp));
            memcpy(aug[col], tmp, sizeof(tmp));
        }
        double pivot = aug[col][col];
        for (int j = 0; j < D * 2; j++) aug[col][j] /= pivot;
        for (int row = 0; row < D; row++) {
            if (row == col) continue;
            double factor = aug[row][col];
            for (int j = 0; j < D * 2; j++) aug[row][j] -= factor * aug[col][j];
        }
    }
    for (int i = 0; i < D; i++)
        for (int j = 0; j < D; j++)
            out[i][j] = aug[i][D + j];
}

void geo_invert_matrix(const double M[D][D], double out[D][D]) {
    double L[D][D];
    if (cholesky_decompose(M, L)) {
        /* Cholesky inversion: solve G * col = e_i for each column */
        for (int col = 0; col < D; col++) {
            double e[D], y[D], x[D];
            memset(e, 0, sizeof(e));
            e[col] = 1.0;
            forward_solve(L, e, y);
            back_solve(L, y, x);
            for (int row = 0; row < D; row++) out[row][col] = x[row];
        }
    } else {
        gauss_jordan_invert(M, out);
    }
}

/* =========================================================================
 * Partial Metric Derivative — central finite differences
 * ========================================================================= */

static void partial_metric(GeoMetricFn metric_fn, void *data,
                           const double x[D], int k, double epsilon,
                           double dG[D][D]) {
    double x_plus[D], x_minus[D];
    double G_plus[D][D], G_minus[D][D];

    memcpy(x_plus, x, sizeof(double) * D);
    memcpy(x_minus, x, sizeof(double) * D);
    x_plus[k] += epsilon;
    x_minus[k] -= epsilon;

    metric_fn(x_plus, G_plus, data);
    metric_fn(x_minus, G_minus, data);

    double inv_2eps = 1.0 / (2.0 * epsilon);
    for (int i = 0; i < D; i++)
        for (int j = 0; j < D; j++)
            dG[i][j] = (G_plus[i][j] - G_minus[i][j]) * inv_2eps;
}

/* =========================================================================
 * Christoffel Symbols — Γᵏᵢⱼ = ½ gᵏˡ (∂ⱼgₗᵢ + ∂ᵢgₗⱼ − ∂ₗgᵢⱼ)
 * ========================================================================= */

double geo_christoffel(const GeodesicSolver *solver,
                       GeoMetricFn metric_fn, void *metric_data,
                       const double x[D], int k, int i, int j) {
    double G[D][D], G_inv[D][D];
    metric_fn(x, G, metric_data);
    geo_invert_matrix(G, G_inv);

    double dG_i[D][D], dG_j[D][D], dG_l[D][D];
    partial_metric(metric_fn, metric_data, x, i, solver->epsilon, dG_i);
    partial_metric(metric_fn, metric_data, x, j, solver->epsilon, dG_j);

    double gamma = 0.0;
    for (int l = 0; l < D; l++) {
        partial_metric(metric_fn, metric_data, x, l, solver->epsilon, dG_l);
        gamma += 0.5 * G_inv[k][l] * (dG_j[l][i] + dG_i[l][j] - dG_l[i][j]);
    }
    if (gamma > solver->clamp) gamma = solver->clamp;
    if (gamma < -solver->clamp) gamma = -solver->clamp;
    return gamma;
}

/* =========================================================================
 * Geodesic Acceleration — aᵏ = −Γᵏᵢⱼ vⁱ vʲ
 * ========================================================================= */

static void geodesic_acceleration(const GeodesicSolver *solver,
                                  GeoMetricFn metric_fn, void *data,
                                  const double x[D], const double v[D],
                                  double accel[D]) {
    double G[D][D], G_inv[D][D];
    metric_fn(x, G, data);
    geo_invert_matrix(G, G_inv);

    /* Precompute all partial derivatives */
    double dG[D][D][D]; /* dG[d][i][j] = ∂G[i][j]/∂x[d] */
    for (int d = 0; d < D; d++)
        partial_metric(metric_fn, data, x, d, solver->epsilon,
                       dG[d]);

    for (int k = 0; k < D; k++) {
        double sum = 0.0;
        for (int i = 0; i < D; i++) {
            for (int j = 0; j < D; j++) {
                double gamma_kij = 0.0;
                for (int l = 0; l < D; l++) {
                    gamma_kij += 0.5 * G_inv[k][l] *
                        (dG[j][l][i] + dG[i][l][j] - dG[l][i][j]);
                }
                if (gamma_kij > solver->clamp) gamma_kij = solver->clamp;
                if (gamma_kij < -solver->clamp) gamma_kij = -solver->clamp;
                sum += gamma_kij * v[i] * v[j];
            }
        }
        accel[k] = -sum;
    }
}

/* =========================================================================
 * Gauge Utilities
 * ========================================================================= */

static int is_zero_gauge(const GeoGaugeField *gauge) {
    if (!gauge) return 1;
    double eps = DBL_EPSILON * 1e6;
    for (int i = 0; i < D; i++) {
        if (fabs(gauge->charge[i]) > eps) return 0;
        for (int j = 0; j < D; j++)
            if (fabs(gauge->potential[i][j]) > eps) return 0;
    }
    return 1;
}

static double compute_gauge_work(const double deviation[D],
                                 const GeoGaugeField *gauge) {
    if (!gauge) return 0.0;
    double Ad[D];
    geo_mat_vec(gauge->potential, deviation, Ad);
    return geo_dot(gauge->charge, Ad);
}

/* =========================================================================
 * Solver Initialisation
 * ========================================================================= */

void geo_solver_init(GeodesicSolver *s, int steps, double epsilon,
                     double clamp) {
    s->steps = (steps > 0) ? steps : 15;
    s->epsilon = (epsilon > 0.0) ? epsilon : 1e-4;
    s->clamp = (clamp > 0.0) ? clamp : 2.0;
}

/* =========================================================================
 * Euclidean Distance (flat-path bypass)
 * ========================================================================= */

GeodesicResult geo_compute_euclidean(const double deviation[D],
                                     const GeoGaugeField *gauge,
                                     double gauge_weight) {
    GeodesicResult r;
    memset(&r, 0, sizeof(r));
    r.tension = geo_norm(deviation);
    r.steps_taken = 0;
    memcpy(r.final_position, deviation, sizeof(double) * D);
    double t = (r.tension > DBL_EPSILON) ? r.tension : DBL_EPSILON;
    for (int i = 0; i < D; i++) r.final_velocity[i] = deviation[i] / t;
    r.gauge_work = compute_gauge_work(deviation, gauge);
    r.total_risk = r.tension + fabs(r.gauge_work) * gauge_weight;
    return r;
}

/* =========================================================================
 * RK4 Geodesic Solver — compute_tension
 * ========================================================================= */

GeodesicResult geo_compute_tension(
    const GeodesicSolver *solver,
    GeoMetricFn metric_fn, void *metric_data,
    const double deviation[D],
    const GeoGaugeField *gauge,
    double gauge_weight
) {
    GeodesicResult result;
    memset(&result, 0, sizeof(result));
    result.steps_taken = solver->steps;

    /* Guard: zero deviation */
    double dev_norm = geo_norm(deviation);
    if (dev_norm < DBL_EPSILON * 1e6) return result;

    double h = 1.0 / solver->steps;

    /* Initial position: origin */
    double x[D];
    memset(x, 0, sizeof(x));

    /* Initial velocity: normalized by local metric */
    double G0[D][D];
    metric_fn(x, G0, metric_data);
    double n_sq = geo_quadratic_form(deviation, G0);
    double metric_norm = sqrt((n_sq > DBL_EPSILON) ? n_sq : DBL_EPSILON);

    double v[D];
    for (int i = 0; i < D; i++) v[i] = deviation[i] / metric_norm;

    /* Precompute gauge force (constant) */
    int has_gauge = !is_zero_gauge(gauge);
    double gauge_force[D];
    memset(gauge_force, 0, sizeof(gauge_force));
    if (has_gauge) {
        geo_mat_vec(gauge->potential, deviation, gauge_force);
    }

    /* RK4 Integration */
    double length = 0.0;
    double zero_accel[D];
    memset(zero_accel, 0, sizeof(zero_accel));

    for (int s = 0; s < solver->steps; s++) {
        double G[D][D];
        metric_fn(x, G, metric_data);

        int metric_is_flat = geo_is_identity(G, 1e-6);

        double ds_sq;
        double accel[D];

        if (metric_is_flat) {
            ds_sq = geo_norm_sq(v);
            memcpy(accel, zero_accel, sizeof(accel));
        } else {
            ds_sq = geo_quadratic_form(v, G);
            geodesic_acceleration(solver, metric_fn, metric_data, x, v, accel);
        }

        length += sqrt((ds_sq > 0.0) ? ds_sq : 0.0) * h;

        /* RK4 Stage 1 */
        double k1_x[D], k1_v[D];
        for (int i = 0; i < D; i++) {
            k1_x[i] = v[i];
            k1_v[i] = accel[i] + gauge_force[i];
        }

        /* RK4 Stage 2 */
        double x2[D], v2[D], a2[D];
        for (int i = 0; i < D; i++) {
            x2[i] = x[i] + 0.5 * h * k1_x[i];
            v2[i] = v[i] + 0.5 * h * k1_v[i];
        }
        if (metric_is_flat) memcpy(a2, zero_accel, sizeof(a2));
        else geodesic_acceleration(solver, metric_fn, metric_data, x2, v2, a2);
        double k2_x[D], k2_v[D];
        for (int i = 0; i < D; i++) {
            k2_x[i] = v2[i];
            k2_v[i] = a2[i] + gauge_force[i];
        }

        /* RK4 Stage 3 */
        double x3[D], v3[D], a3[D];
        for (int i = 0; i < D; i++) {
            x3[i] = x[i] + 0.5 * h * k2_x[i];
            v3[i] = v[i] + 0.5 * h * k2_v[i];
        }
        if (metric_is_flat) memcpy(a3, zero_accel, sizeof(a3));
        else geodesic_acceleration(solver, metric_fn, metric_data, x3, v3, a3);
        double k3_x[D], k3_v[D];
        for (int i = 0; i < D; i++) {
            k3_x[i] = v3[i];
            k3_v[i] = a3[i] + gauge_force[i];
        }

        /* RK4 Stage 4 */
        double x4[D], v4[D], a4[D];
        for (int i = 0; i < D; i++) {
            x4[i] = x[i] + h * k3_x[i];
            v4[i] = v[i] + h * k3_v[i];
        }
        if (metric_is_flat) memcpy(a4, zero_accel, sizeof(a4));
        else geodesic_acceleration(solver, metric_fn, metric_data, x4, v4, a4);
        double k4_x[D], k4_v[D];
        for (int i = 0; i < D; i++) {
            k4_x[i] = v4[i];
            k4_v[i] = a4[i] + gauge_force[i];
        }

        /* RK4 combination */
        for (int i = 0; i < D; i++) {
            x[i] += h * (k1_x[i] + 2.0 * k2_x[i] + 2.0 * k3_x[i] + k4_x[i]) / 6.0;
            v[i] += h * (k1_v[i] + 2.0 * k2_v[i] + 2.0 * k3_v[i] + k4_v[i]) / 6.0;
        }
    }

    result.tension = length * metric_norm;
    memcpy(result.final_position, x, sizeof(double) * D);
    memcpy(result.final_velocity, v, sizeof(double) * D);
    result.gauge_work = compute_gauge_work(deviation, gauge);
    result.total_risk = result.tension + fabs(result.gauge_work) * gauge_weight;

    return result;
}

/* =========================================================================
 * Built-in Metric Functions
 * ========================================================================= */

void geo_metric_flat(const double x[D], double G[D][D], void *user_data) {
    (void)x; (void)user_data;
    geo_identity(G);
}

void geo_metric_diagonal(const double x[D], double G[D][D], void *user_data) {
    (void)x;
    double *scales = (double *)user_data;
    memset(G, 0, sizeof(double) * D * D);
    for (int i = 0; i < D; i++)
        G[i][i] = scales[i] * scales[i];
}

void geo_metric_behavioral(const double x[D], double G[D][D],
                           void *user_data) {
    double stiffness = user_data ? *(double *)user_data : 1.0;
    double r_sq = 0.0;
    for (int i = 0; i < D; i++) r_sq += x[i] * x[i];
    double scale = 1.0 + stiffness * r_sq;

    memset(G, 0, sizeof(double) * D * D);
    for (int i = 0; i < D; i++) {
        G[i][i] = scale;
        for (int j = 0; j < D; j++) {
            if (i != j) G[i][j] = stiffness * x[i] * x[j] * 0.1;
        }
    }
}

void geo_metric_fraud(const double x[D], double G[D][D], void *user_data) {
    GeoFraudMetricCtx ctx;
    if (user_data) {
        ctx = *(GeoFraudMetricCtx *)user_data;
    } else {
        ctx.stiffness = 0.1;
        ctx.nlp_coupling = 2.0;
    }

    double r_sq = 0.0;
    for (int i = 0; i < D; i++) r_sq += x[i] * x[i];
    double base_scale = 1.0 + ctx.stiffness * r_sq;

    memset(G, 0, sizeof(double) * D * D);
    for (int i = 0; i < D; i++) G[i][i] = base_scale;

    /* NLP cross-coupling: high dim1 stiffens dim0 */
    double nlp_val = x[1];
    G[0][0] += ctx.nlp_coupling * nlp_val * nlp_val;
    G[0][1] = ctx.nlp_coupling * nlp_val * 0.5;
    G[1][0] = G[0][1];

    /* Outflow–velocity coupling: fast dim5 amplifies dim3 cost */
    double outflow = x[5];
    G[3][3] += ctx.stiffness * outflow * outflow;
    G[3][5] = ctx.stiffness * outflow * 0.3;
    G[5][3] = G[3][5];
}
