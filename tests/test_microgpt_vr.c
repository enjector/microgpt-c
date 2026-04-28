/*
 * test_microgpt_vr.c — C99 Vietoris-Rips Persistent Cohomology Tests
 *
 * Port of 22 Catch2 tests from vr_engine_tests.cpp.
 * Zero-dependency assertion-based tests.
 *
 * Copyright (c) 2026 Ajay Soni (ajay.soni@enjector.com), Enjector Software Ltd.
 * MIT License.
 */

#define _CRT_SECURE_NO_WARNINGS 1

#include "microgpt_vr.h"

#include <math.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ---- Test harness ---- */

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

#define ASSERT_EQ(a, b) ASSERT((a) == (b))
#define ASSERT_GE(a, b) ASSERT((a) >= (b))
#define ASSERT_GT(a, b) ASSERT((a) > (b))
#define ASSERT_LT(a, b) ASSERT((a) < (b))
#define ASSERT_LE(a, b) ASSERT((a) <= (b))
#define RUN(name) run_##name()

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* Helper: create a 2D point */
static VRPoint pt2(float x, float y, int id) {
    float c[2] = {x, y};
    return vr_make_point(c, 2, id);
}

/* Helper: create a 3D point */
static VRPoint pt3(float x, float y, float z, int id) {
    float c[3] = {x, y, z};
    return vr_make_point(c, 3, id);
}

/* ==================================================================== */
/* §1: Single point and empty input                                      */
/* ==================================================================== */

TEST(single_point_betti) {
    VREngine engine;
    vr_engine_init(&engine, 100.0f, 2, 2);
    VRPoint pts[1] = { pt2(0.0f, 0.0f, 0) };
    int betti[3];
    vr_betti_numbers(&engine, pts, 1, -1.0f, 0.0f, betti);
    ASSERT_EQ(betti[0], 1);
    ASSERT_EQ(betti[1], 0);
    ASSERT_EQ(betti[2], 0);
}

TEST(empty_input) {
    VREngine engine;
    vr_engine_init(&engine, 100.0f, 2, 2);
    VRDiagram diag = vr_compute(&engine, NULL, 0, 0.0f);
    ASSERT_EQ(diag.count, 0);
}

/* ==================================================================== */
/* §2: Two points — β₀ transitions                                      */
/* ==================================================================== */

TEST(two_points_beta0_transitions) {
    VREngine engine;
    vr_engine_init(&engine, 100.0f, 1, 2);
    VRPoint pts[2] = { pt2(0.0f, 0.0f, 0), pt2(1.0f, 0.0f, 1) };
    VRDiagram diag = vr_compute(&engine, pts, 2, 0.0f);

    /* Before edge (d²=1): β₀=2 */
    ASSERT_EQ(vr_betti_at(&diag, 0, 0.5f), 2);
    /* After edge: β₀=1 */
    ASSERT_EQ(vr_betti_at(&diag, 0, 2.0f), 1);
}

/* ==================================================================== */
/* §3: Triangle — β₀=1, β₁=0                                           */
/* ==================================================================== */

TEST(equilateral_triangle_no_loop) {
    VREngine engine;
    vr_engine_init(&engine, 100.0f, 2, 2);
    float h = sqrtf(3.0f) / 2.0f;
    VRPoint pts[3] = {
        pt2(0.0f, 0.0f, 0), pt2(1.0f, 0.0f, 1), pt2(0.5f, h, 2)
    };
    VRDiagram diag = vr_compute(&engine, pts, 3, 0.0f);
    ASSERT_EQ(vr_betti_at(&diag, 0, 2.0f), 1);
    ASSERT_EQ(vr_betti_at(&diag, 1, 2.0f), 0);
}

/* ==================================================================== */
/* §4: Three clusters — β₀=3                                            */
/* ==================================================================== */

TEST(three_clusters_beta0) {
    VREngine engine;
    vr_engine_init(&engine, 1.0f, 1, 2);
    VRPoint pts[9] = {
        pt2(0.0f, 0.0f, 0), pt2(0.1f, 0.0f, 1), pt2(0.0f, 0.1f, 2),
        pt2(10.0f, 0.0f, 3), pt2(10.1f, 0.0f, 4), pt2(10.0f, 0.1f, 5),
        pt2(5.0f, 10.0f, 6), pt2(5.1f, 10.0f, 7), pt2(5.0f, 10.1f, 8)
    };
    int betti[3];
    vr_betti_numbers(&engine, pts, 9, -1.0f, 0.0f, betti);
    ASSERT_EQ(betti[0], 3);
}

/* ==================================================================== */
/* §5: Unit circle — β₁=1 (loop detection)                              */
/* ==================================================================== */

TEST(unit_circle_detects_loop) {
    VREngine engine;
    vr_engine_init(&engine, 100.0f, 2, 2);
    VRPoint pts[12];
    for (int i = 0; i < 12; i++) {
        float angle = 2.0f * (float)M_PI * i / 12.0f;
        pts[i] = pt2(cosf(angle), sinf(angle), i);
    }
    VRDiagram diag = vr_compute(&engine, pts, 12, 0.0f);

    int h1_count = 0;
    for (int i = 0; i < diag.count; i++)
        if (diag.intervals[i].dimension == 1 &&
            (diag.intervals[i].death - diag.intervals[i].birth) > 0.1f)
            h1_count++;
    ASSERT_GE(h1_count, 1);
}

/* ==================================================================== */
/* §6: Fraud — Mule cycle (β₁ detection)                                */
/* ==================================================================== */

TEST(fraud_mule_cycle_beta1) {
    VREngine engine;
    vr_engine_init(&engine, 100.0f, 2, 2);
    VRPoint mules[4] = {
        pt2(0.0f, 0.0f, 0), pt2(1.0f, 0.0f, 1),
        pt2(1.0f, 1.0f, 2), pt2(0.0f, 1.0f, 3)
    };
    VRDiagram diag = vr_compute(&engine, mules, 4, 0.0f);
    /* Between edges (d²=1) and diagonals (d²=2), β₁≥1 */
    ASSERT_GE(vr_betti_at(&diag, 1, 1.5f), 1);
}

/* ==================================================================== */
/* §7: Fraud — Velocity attack (outlier isolation)                       */
/* ==================================================================== */

TEST(fraud_velocity_attack_outlier) {
    VREngine engine;
    vr_engine_init(&engine, 5.0f, 1, 12);
    VRPoint pts[6];
    for (int i = 0; i < 5; i++) {
        float c[12];
        memset(c, 0, sizeof(c));
        c[0] = 0.1f * i;
        pts[i] = vr_make_point(c, 12, i);
    }
    /* Outlier */
    float outlier[12];
    memset(outlier, 0, sizeof(outlier));
    outlier[0] = 50.0f;
    outlier[3] = 20.0f;
    pts[5] = vr_make_point(outlier, 12, 5);

    int betti[3];
    vr_betti_numbers(&engine, pts, 6, -1.0f, 0.0f, betti);
    ASSERT_GE(betti[0], 2);
}

/* ==================================================================== */
/* §8: Radius filtering                                                  */
/* ==================================================================== */

TEST(max_radius_filters_edges) {
    VREngine engine;
    vr_engine_init(&engine, 2.0f, 1, 2);
    VRPoint pts[3] = {
        pt2(0.0f, 0.0f, 0), pt2(1.0f, 0.0f, 1), pt2(10.0f, 0.0f, 2)
    };
    int betti[3];
    vr_betti_numbers(&engine, pts, 3, -1.0f, 0.0f, betti);
    ASSERT_EQ(betti[0], 2); /* Point 2 disconnected */
}

/* ==================================================================== */
/* §9: Persistence API                                                   */
/* ==================================================================== */

TEST(persistence_interval_api) {
    VRInterval iv;
    iv.dimension = 1; iv.birth = 0.5f; iv.death = 1.5f;
    ASSERT(fabsf((iv.death - iv.birth) - 1.0f) < 0.001f);

    VRInterval essential;
    essential.dimension = 0; essential.birth = 0.0f;
    essential.death = FLT_MAX;
    ASSERT(essential.death > 1e30f);
}

/* ==================================================================== */
/* §10: Square transient H1                                              */
/* ==================================================================== */

TEST(square_transient_h1) {
    VREngine engine;
    vr_engine_init(&engine, 100.0f, 2, 2);
    VRPoint pts[4] = {
        pt2(0.0f, 0.0f, 0), pt2(1.0f, 0.0f, 1),
        pt2(1.0f, 1.0f, 2), pt2(0.0f, 1.0f, 3)
    };
    VRDiagram diag = vr_compute(&engine, pts, 4, 0.0f);

    int has_h1 = 0;
    for (int i = 0; i < diag.count; i++)
        if (diag.intervals[i].dimension == 1 &&
            (diag.intervals[i].death - diag.intervals[i].birth) > 0.01f)
            has_h1 = 1;
    ASSERT(has_h1);
}

/* ==================================================================== */
/* §11: Template instantiation — 12D                                     */
/* ==================================================================== */

TEST(twelve_dim_points) {
    VREngine engine;
    vr_engine_init(&engine, 100.0f, 1, 12);
    float a[12], b[12];
    memset(a, 0, sizeof(a)); memset(b, 0, sizeof(b));
    a[0] = 1.0f; b[0] = 2.0f;
    VRPoint pts[2] = { vr_make_point(a, 12, 0), vr_make_point(b, 12, 1) };
    int betti[3];
    vr_betti_numbers(&engine, pts, 2, -1.0f, 0.0f, betti);
    ASSERT_EQ(betti[0], 1);
}

/* ==================================================================== */
/* §12: Persistence threshold filtering                                  */
/* ==================================================================== */

TEST(min_persistence_filters) {
    VREngine engine;
    vr_engine_init(&engine, 100.0f, 2, 2);
    VRPoint pts[4] = {
        pt2(0.0f, 0.0f, 0), pt2(1.0f, 0.0f, 1),
        pt2(1.0f, 1.0f, 2), pt2(0.0f, 1.0f, 3)
    };
    VRDiagram full = vr_compute(&engine, pts, 4, 0.0f);
    VRDiagram filtered = vr_compute(&engine, pts, 4, 999.0f);

    int h1_full = 0, h1_filtered = 0;
    for (int i = 0; i < full.count; i++)
        if (full.intervals[i].dimension == 1) h1_full++;
    for (int i = 0; i < filtered.count; i++)
        if (filtered.intervals[i].dimension == 1) h1_filtered++;
    ASSERT_LE(h1_filtered, h1_full);
}

/* ==================================================================== */
/* §13: Determinism                                                      */
/* ==================================================================== */

TEST(deterministic_results) {
    VREngine e1, e2;
    vr_engine_init(&e1, 100.0f, 2, 2);
    vr_engine_init(&e2, 100.0f, 2, 2);
    VRPoint pts[4] = {
        pt2(0.0f, 0.0f, 0), pt2(1.0f, 0.0f, 1),
        pt2(0.5f, 0.866f, 2), pt2(2.0f, 0.0f, 3)
    };
    VRDiagram d1 = vr_compute(&e1, pts, 4, 0.0f);
    VRDiagram d2 = vr_compute(&e2, pts, 4, 0.0f);
    ASSERT_EQ(d1.count, d2.count);
    for (int i = 0; i < d1.count; i++) {
        ASSERT_EQ(d1.intervals[i].dimension, d2.intervals[i].dimension);
        ASSERT(d1.intervals[i].birth == d2.intervals[i].birth);
        ASSERT(d1.intervals[i].death == d2.intervals[i].death);
    }
}

/* ==================================================================== */
/* §14: Fuzzing — 3D random clouds                                       */
/* ==================================================================== */

/* Simple LCG for deterministic randomness */
static unsigned int lcg_state = 42;
static float lcg_float(float lo, float hi) {
    lcg_state = lcg_state * 1103515245u + 12345u;
    float t = (float)((lcg_state >> 16) & 0x7FFF) / 32767.0f;
    return lo + t * (hi - lo);
}

TEST(fuzz_3d_valid_output) {
    for (unsigned seed = 1; seed <= 5; seed++) {
        int ns[] = {3, 8, 15};
        for (int ni = 0; ni < 3; ni++) {
            int n = ns[ni];
            lcg_state = seed * 100 + n;

            VRPoint pts[15];
            for (int i = 0; i < n; i++)
                pts[i] = pt3(lcg_float(-10, 10), lcg_float(-10, 10),
                             lcg_float(-10, 10), i);

            VREngine engine;
            vr_engine_init(&engine, 1000.0f, 2, 3);
            VRDiagram diag = vr_compute(&engine, pts, n, 0.0f);

            for (int i = 0; i < diag.count; i++)
                ASSERT_LE(diag.intervals[i].birth, diag.intervals[i].death);
            ASSERT_GE(vr_betti_at(&diag, 0, 1e6f), 1);
        }
    }
}

/* ==================================================================== */
/* §15: Fuzzing — 12D random clouds                                      */
/* ==================================================================== */

TEST(fuzz_12d_valid_output) {
    for (unsigned seed = 1; seed <= 3; seed++) {
        int ns[] = {4, 8};
        for (int ni = 0; ni < 2; ni++) {
            int n = ns[ni];
            lcg_state = seed * 1000 + n;

            VRPoint pts[8];
            for (int i = 0; i < n; i++) {
                float c[12];
                for (int d = 0; d < 12; d++) c[d] = lcg_float(-5, 5);
                pts[i] = vr_make_point(c, 12, i);
            }

            VREngine engine;
            vr_engine_init(&engine, 1000.0f, 2, 12);
            VRDiagram diag = vr_compute(&engine, pts, n, 0.0f);

            for (int i = 0; i < diag.count; i++)
                ASSERT_LE(diag.intervals[i].birth, diag.intervals[i].death);
            ASSERT_GE(vr_betti_at(&diag, 0, 1e6f), 1);
        }
    }
}

/* ==================================================================== */
/* Main                                                                  */
/* ==================================================================== */

int main(void) {
    printf("\n=== MicroGPT VR Persistent Cohomology Tests ===\n\n");

    RUN(single_point_betti);
    RUN(empty_input);
    RUN(two_points_beta0_transitions);
    RUN(equilateral_triangle_no_loop);
    RUN(three_clusters_beta0);
    RUN(unit_circle_detects_loop);
    RUN(fraud_mule_cycle_beta1);
    RUN(fraud_velocity_attack_outlier);
    RUN(max_radius_filters_edges);
    RUN(persistence_interval_api);
    RUN(square_transient_h1);
    RUN(twelve_dim_points);
    RUN(min_persistence_filters);
    RUN(deterministic_results);
    RUN(fuzz_3d_valid_output);
    RUN(fuzz_12d_valid_output);

    printf("\n--- Results: %d/%d passed", g_tests_passed, g_tests_run);
    if (g_tests_failed > 0) printf(", %d FAILED", g_tests_failed);
    printf(" ---\n\n");

    return g_tests_failed > 0 ? 1 : 0;
}
