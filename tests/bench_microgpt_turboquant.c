#include "bench.h"
#include "microgpt_turboquant.h"
#include <math.h>

static TurboQuant bench_tq;
static float *bench_x;
static float *bench_out;
static uint32_t *bench_centroid_idx;
#define BENCH_EMBD 96

static int setup_tq() {
    turboquant_init(&bench_tq, BENCH_EMBD, 3, true);
    
    bench_x = (float *)calloc(BENCH_EMBD, sizeof(float));
    bench_out = (float *)calloc(BENCH_EMBD, sizeof(float));
    bench_centroid_idx = (uint32_t *)calloc(BENCH_EMBD, sizeof(uint32_t));
    
    for (int i = 0; i < BENCH_EMBD; i++) {
        bench_x[i] = (float)i * 0.01f;
    }
    return 0;
}

static void teardown_tq() {
    free(bench_x);
    free(bench_out);
    free(bench_centroid_idx);
    turboquant_free(&bench_tq);
}

int bench_tq_quant_throughput() {
    if (setup_tq() != 0) return BENCHMARK_RESULT_CORE_FAILED;
    int iterations = 1000000;
    
    for (int i = 0; i < iterations; i++) {
        turboquant_quant_mse(&bench_tq, bench_x, bench_centroid_idx);
    }
    
    teardown_tq();
    return iterations;
}

int bench_tq_dequant_throughput() {
    if (setup_tq() != 0) return BENCHMARK_RESULT_CORE_FAILED;
    int iterations = 1000000;
    
    // Create an arbitrary state to dequantize
    turboquant_quant_mse(&bench_tq, bench_x, bench_centroid_idx);
    
    for (int i = 0; i < iterations; i++) {
        turboquant_dequant_mse(&bench_tq, bench_centroid_idx, bench_out);
    }
    
    teardown_tq();
    return iterations;
}

int main(void) {
    benchmark tq_benchmarks[] = {
        BENCHMARK_CASE(bench_tq_quant_throughput),
        BENCHMARK_CASE(bench_tq_dequant_throughput),
        {BENCHMARK_END}
    };
    
    benchmark_suite suites[] = {
        {"TurboQuant Array Compression Streams", tq_benchmarks},
        {BENCHMARK_END}
    };
    
    benchmark_suite_run(suites, NULL);
    return 0;
}
