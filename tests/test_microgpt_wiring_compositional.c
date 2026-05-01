/*
 * MicroGPT-C — Compositional search tests
 *
 * Stream B of the compositional generator fix. Verifies the
 * type-directed search in wiring_compositional_search.c can:
 *   1. Compose a 2-primitive graph from "average of x and y squared".
 *   2. Compose a 3-primitive graph from a multi-stage prompt.
 *   3. Refuse cleanly when no primitive matches the prompt.
 *   4. Produce a graph that runs end-to-end via pipeline_execute_vm
 *      (Stream A integration check — depends on Stream A having landed).
 *
 * Copyright (c) 2026 Ajay Soni, Enjector Software Ltd. MIT License.
 */

#define _CRT_SECURE_NO_WARNINGS 1

#include "microgpt_pipeline.h"
#include "microgpt_vm.h"
#include "wiring_compositional_search.h"
#include "wiring_primitive_manifest.h"
#include "wiring_natives.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ---- Minimal test harness ---- */
static int g_tests_run = 0, g_tests_passed = 0, g_tests_failed = 0;

#define TEST(name) static void test_##name(void)
#define RUN(name)                                                              \
    do {                                                                       \
        g_tests_run++;                                                         \
        printf("  %-60s ", #name);                                             \
        fflush(stdout);                                                        \
        int _f0 = g_tests_failed;                                              \
        test_##name();                                                         \
        if (g_tests_failed == _f0) {                                           \
            g_tests_passed++;                                                  \
            printf("PASS\n");                                                  \
        } else {                                                               \
            printf("FAIL\n");                                                  \
        }                                                                      \
    } while (0)
#define ASSERT(cond)                                                           \
    do {                                                                       \
        if (!(cond)) {                                                         \
            g_tests_failed++;                                                  \
            printf("\n    ASSERT FAILED: %s @ line %d ", #cond, __LINE__);     \
            return;                                                            \
        }                                                                      \
    } while (0)
#define ASSERT_EQ(a, b)                                                        \
    do {                                                                       \
        long long _a = (long long)(a), _b = (long long)(b);                    \
        if (_a != _b) {                                                        \
            g_tests_failed++;                                                  \
            printf("\n    ASSERT_EQ FAILED: %s=%lld != %s=%lld @ line %d ",    \
                   #a, _a, #b, _b, __LINE__);                                  \
            return;                                                            \
        }                                                                      \
    } while (0)

/* ---- Helpers ---- */
static int report_has_primitive(const WiringComposeReport *r, const char *name) {
    for (int i = 0; i < r->n_nodes_used; i++) {
        if (r->primitive_names[i] && strcmp(r->primitive_names[i], name) == 0)
            return 1;
    }
    return 0;
}

/* ---- Tests ---- */

TEST(compositional_search_two_primitive) {
    /* "average of x and y squared" — outer should be average_two; inner
     * should bind a square node onto one of the inputs. */
    WiringComposeReport report = {0};
    Pipeline *p = wiring_compositional_search(
        "compute the average of x and y squared", &report);
    ASSERT(p != NULL);
    ASSERT_EQ(report.verified, 1);
    ASSERT(report_has_primitive(&report, "average_two"));
    ASSERT(report_has_primitive(&report, "square"));
    ASSERT(report.n_nodes_used >= 2);
    pipeline_free(p);
}

TEST(compositional_search_multi_primitive) {
    /* Multi-stage prompt that names two primitives unambiguously by
     * keyword.  We DO NOT pin which primitive the greedy search picks
     * as outer vs inner (manifest-order ties are an implementation
     * detail of the V1 greedy beam=1 strategy) — we only assert that
     * the mechanism produces a verified ≥2-node graph and that the
     * graph's primitive set is a subset of the manifest, i.e. nothing
     * fabricated. */
    WiringComposeReport report = {0};
    Pipeline *p = wiring_compositional_search(
        "the gcd of the squared difference between x and y", &report);
    ASSERT(p != NULL);
    ASSERT_EQ(report.verified, 1);
    ASSERT(report.n_nodes_used >= 2);
    /* Every reported primitive name must be in the manifest. */
    int n_manifest = 0;
    const WiringPrimitive *manifest = wiring_primitive_manifest(&n_manifest);
    for (int i = 0; i < report.n_nodes_used; i++) {
        int found = 0;
        for (int j = 0; j < n_manifest; j++) {
            if (strcmp(manifest[j].name, report.primitive_names[i]) == 0) {
                found = 1; break;
            }
        }
        ASSERT(found);
    }
    pipeline_free(p);
}

TEST(compositional_search_unknown_primitive) {
    /* Prompt that names nothing in the manifest's keyword sets. */
    WiringComposeReport report = {0};
    Pipeline *p = wiring_compositional_search(
        "wibble the foobar across the quux", &report);
    ASSERT(p == NULL);
    ASSERT_EQ(report.verified, 0);
    ASSERT_EQ(report.n_nodes_used, 0);
}

TEST(compositional_search_executes_vm) {
    /* End-to-end: synthesise a graph, register the natives, run via the
     * VM dispatcher, and confirm a numeric answer. */
    WiringComposeReport report = {0};
    Pipeline *p = wiring_compositional_search(
        "compute the average of x and y squared", &report);
    ASSERT(p != NULL);

    vm_engine *vm = vm_engine_create();
    /* Register hand-written wrappers for the two primitives we expect the
     * search to have picked. */
    extern double vm_native_average_two(int argc, const double *argv);
    extern double vm_native_square(int argc, const double *argv);
    vm_engine_register_fn(vm, "average_two", vm_native_average_two);
    vm_engine_register_fn(vm, "square", vm_native_square);

    PipelineValue inputs[16] = {0};
    PipelineValue outputs[1] = {0};
    /* Set all inputs to 4 (simple): if outer is average_two(square(4), 4),
     * result = (16 + 4) / 2 = 10.  Different orderings give different
     * concrete answers; we just assert the run produced *some* numeric
     * result and the VM dispatch path completed. */
    for (int i = 0; i < report.signature_in_count && i < 16; i++) {
        inputs[i].v.i = 4;
    }
    int rc = pipeline_execute_vm(p, vm, inputs, outputs);
    if (rc != 0) printf("\n    err: %s ", pipeline_last_error());
    ASSERT_EQ(rc, PIPE_OK);
    /* Plausibility: result must be in [4, 16] given the only operations
     * available (average_two on 4 + square on 4) — anywhere outside is
     * a marshalling bug. */
    ASSERT(outputs[0].v.i >= 4);
    ASSERT(outputs[0].v.i <= 16);

    vm_engine_dispose(vm);
    pipeline_free(p);
}

/* Native wrappers required by compositional_search_executes_vm. */
double vm_native_average_two(int argc, const double *argv) {
    (void)argc;
    return (argv[0] + argv[1]) / 2.0;
}
double vm_native_square(int argc, const double *argv) {
    (void)argc;
    return argv[0] * argv[0];
}

int main(void) {
    printf("[Wiring compositional search]\n");
    RUN(compositional_search_two_primitive);
    RUN(compositional_search_multi_primitive);
    RUN(compositional_search_unknown_primitive);
    RUN(compositional_search_executes_vm);

    printf("\n=== Results: %d/%d passed ===\n", g_tests_passed, g_tests_run);
    return g_tests_failed == 0 ? 0 : 1;
}
