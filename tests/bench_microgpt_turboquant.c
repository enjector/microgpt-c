#include "bench.h"
#include "microgpt_turboquant.h"
#include <math.h>
#include <stdio.h>

// -----------------------------------------------------------------------
// Shared state

static TurboQuant bench_tq;
static float      *bench_x;
static float      *bench_out;
static uint32_t   *bench_idx;
static int8_t     *bench_signs;
static float       bench_rnorm;

#define BENCH_D   128
#define BENCH_B   4
#define BENCH_N   1000000

static int setup(void) {
    turboquant_init(&bench_tq, BENCH_D, BENCH_B, true);
    bench_x    = (float   *)malloc(BENCH_D * sizeof(float));
    bench_out  = (float   *)malloc(BENCH_D * sizeof(float));
    bench_idx  = (uint32_t*)malloc(BENCH_D * sizeof(uint32_t));
    bench_signs = (int8_t  *)malloc(BENCH_D * sizeof(int8_t));
    // unit-norm test vector
    float norm = 0.0f;
    for (int i = 0; i < BENCH_D; i++) { bench_x[i] = (float)(i+1); norm += bench_x[i]*bench_x[i]; }
    norm = 1.0f / sqrtf(norm);
    for (int i = 0; i < BENCH_D; i++) bench_x[i] *= norm;
    return 0;
}

static void teardown(void) {
    free(bench_x); free(bench_out); free(bench_idx); free(bench_signs);
    turboquant_free(&bench_tq);
}

// -----------------------------------------------------------------------
// Throughput: MSE quant (b=4, d=128)

int bench_tq_mse_quant(void) {
    if (setup() != 0) return BENCHMARK_RESULT_CORE_FAILED;
    for (int i = 0; i < BENCH_N; i++)
        turboquant_quant_mse(&bench_tq, bench_x, bench_idx);
    teardown();
    return BENCH_N;
}

// -----------------------------------------------------------------------
// Throughput: MSE dequant (b=4, d=128)

int bench_tq_mse_dequant(void) {
    if (setup() != 0) return BENCHMARK_RESULT_CORE_FAILED;
    turboquant_quant_mse(&bench_tq, bench_x, bench_idx);   // warm state
    for (int i = 0; i < BENCH_N; i++)
        turboquant_dequant_mse(&bench_tq, bench_idx, bench_out);
    teardown();
    return BENCH_N;
}

// -----------------------------------------------------------------------
// Throughput: Prod quant (b=4, d=128)

int bench_tq_prod_quant(void) {
    if (setup() != 0) return BENCHMARK_RESULT_CORE_FAILED;
    for (int i = 0; i < BENCH_N; i++)
        turboquant_quant_prod(&bench_tq, bench_x, bench_idx, bench_signs, &bench_rnorm);
    teardown();
    return BENCH_N;
}

// -----------------------------------------------------------------------
// Throughput: Prod dequant (b=4, d=128)

int bench_tq_prod_dequant(void) {
    if (setup() != 0) return BENCHMARK_RESULT_CORE_FAILED;
    turboquant_quant_prod(&bench_tq, bench_x, bench_idx, bench_signs, &bench_rnorm);
    for (int i = 0; i < BENCH_N; i++)
        turboquant_dequant_prod(&bench_tq, bench_idx, bench_signs, bench_rnorm, bench_out);
    teardown();
    return BENCH_N;
}

// -----------------------------------------------------------------------
// Quality table: distortion vs bit-width vs the paper's Theorem 1/2 bounds.
// Printed after benchmarks; returns 1 always so the suite runner counts it.

int bench_tq_quality_table(void) {
    // Paper Theorem 1 (D_mse) and Theorem 2 (D_prod) per-unit-norm vector
    static const float paper_mse [5] = {0.0f, 0.36f, 0.117f, 0.030f, 0.009f};
    static const float paper_prod[5] = {0.0f, 1.57f, 0.56f,  0.18f,  0.047f}; // multiplied by d (d=128)

    int d = 128, N = 500;
    uint64_t seed_base = 0xFEEDC0FFEE1234ULL;

    printf("\n  %-6s  %-14s  %-14s  %-14s  %-14s  %-10s\n",
           "bits", "avg_D_mse", "paper_D_mse", "avg_D_prod*d", "paper_D_prod*d", "IP_bias");
    printf("  %s\n", "------------------------------------------------------------------------");

    for (int b = 1; b <= 4; b++) {
        TurboQuant tq;
        turboquant_init(&tq, d, b, true);

        float   *x    = malloc(d * sizeof(float));
        float   *y    = malloc(d * sizeof(float));
        float   *out  = malloc(d * sizeof(float));
        uint32_t *idx  = malloc(d * sizeof(uint32_t));
        int8_t  *sgn  = malloc(d * sizeof(int8_t));

        double sum_mse = 0.0, sum_prod = 0.0, sum_bias = 0.0;

        for (int n = 0; n < N; n++) {
            // random unit-norm x
            uint64_t s = seed_base ^ (uint64_t)(n * 6271 + b * 997);
            float nx = 0.0f;
            for (int i = 0; i < d; i++) {
                s = s * 6364136223846793005ULL + 1442695040888963407ULL;
                x[i] = (float)(int64_t)s / (float)(1LL << 32); nx += x[i]*x[i];
            }
            nx = 1.0f / sqrtf(nx);
            for (int i = 0; i < d; i++) x[i] *= nx;

            // random unit-norm y (query vector for inner-product test)
            s ^= 0xDEADBEEF;
            float ny = 0.0f;
            for (int i = 0; i < d; i++) {
                s = s * 6364136223846793005ULL + 1442695040888963407ULL;
                y[i] = (float)(int64_t)s / (float)(1LL << 32); ny += y[i]*y[i];
            }
            ny = 1.0f / sqrtf(ny);
            for (int i = 0; i < d; i++) y[i] *= ny;

            // MSE distortion
            turboquant_quant_mse(&tq, x, idx);
            turboquant_dequant_mse(&tq, idx, out);
            float err = 0.0f;
            for (int i = 0; i < d; i++) { float diff = x[i]-out[i]; err += diff*diff; }
            sum_mse += err;

            // Prod distortion + inner-product bias
            float rn;
            turboquant_quant_prod(&tq, x, idx, sgn, &rn);
            turboquant_dequant_prod(&tq, idx, sgn, rn, out);

            float true_ip = 0.0f, est_ip = 0.0f;
            for (int i = 0; i < d; i++) { true_ip += y[i]*x[i]; est_ip += y[i]*out[i]; }
            float prod_err = true_ip - est_ip;
            sum_prod += prod_err * prod_err;
            sum_bias += fabsf(est_ip - true_ip);
        }

        float avg_mse  = (float)(sum_mse  / N);
        float avg_prod = (float)(sum_prod / N) * d;   // scale by d to match paper's D_prod*d
        float avg_bias = (float)(sum_bias / N);

        printf("  b=%-4d  %-14.5f  %-14.5f  %-14.5f  %-14.5f  %-10.5f\n",
               b, avg_mse, paper_mse[b], avg_prod, paper_prod[b], avg_bias);

        free(x); free(y); free(out); free(idx); free(sgn);
        turboquant_free(&tq);
    }
    printf("\n  D_prod is inner-product error variance; D_prod*d should be <= paper_D_prod*d.\n"
           "  IP_bias for Q_prod should be near 0 (unbiased estimator).\n\n");
    return 1;
}

// -----------------------------------------------------------------------

int main(void) {
    benchmark tq_benchmarks[] = {
        BENCHMARK_CASE(bench_tq_mse_quant),
        BENCHMARK_CASE(bench_tq_mse_dequant),
        BENCHMARK_CASE(bench_tq_prod_quant),
        BENCHMARK_CASE(bench_tq_prod_dequant),
        BENCHMARK_CASE(bench_tq_quality_table),
        {BENCHMARK_END}
    };
    benchmark_suite suites[] = {
        {"TurboQuant Throughput & Quality (d=128, b=4)", tq_benchmarks},
        {BENCHMARK_END}
    };
    benchmark_suite_run(suites, NULL);
    return 0;
}
