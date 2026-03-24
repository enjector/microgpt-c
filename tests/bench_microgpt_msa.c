#include "bench.h"
#include "microgpt_msa.h"

static MsaPool *bench_pool;
static scalar_t **bench_keys;
static scalar_t **bench_values;
static scalar_t **bench_query;
#define BENCH_LAYER 4
#define BENCH_EMBD 96
#define BENCH_SEQ 32

static int setup_msa() {
    bench_pool = msa_pool_create(100000, BENCH_LAYER, BENCH_EMBD);
    if (!bench_pool) return BENCHMARK_RESULT_CORE_FAILED;
    
    bench_keys = (scalar_t **)malloc(BENCH_LAYER * sizeof(scalar_t *));
    bench_values = (scalar_t **)malloc(BENCH_LAYER * sizeof(scalar_t *));
    bench_query = (scalar_t **)malloc(BENCH_LAYER * sizeof(scalar_t *));
    for (int l = 0; l < BENCH_LAYER; l++) {
        bench_keys[l] = (scalar_t *)malloc(BENCH_SEQ * BENCH_EMBD * sizeof(scalar_t));
        bench_values[l] = (scalar_t *)malloc(BENCH_SEQ * BENCH_EMBD * sizeof(scalar_t));
        bench_query[l] = (scalar_t *)malloc(BENCH_EMBD * sizeof(scalar_t));
        for (int i = 0; i < BENCH_EMBD; i++) {
            bench_query[l][i] = 1.0f;
        }
    }
    return 0;
}

static void teardown_msa() {
    for (int l = 0; l < BENCH_LAYER; l++) {
        free(bench_keys[l]);
        free(bench_values[l]);
        free(bench_query[l]);
    }
    free(bench_keys);
    free(bench_values);
    free(bench_query);
    msa_pool_free(bench_pool);
}

int bench_msa_pool_chunk_throughput() {
    if (setup_msa() != 0) return BENCHMARK_RESULT_CORE_FAILED;
    int iterations = 100000;
    for (int i = 0; i < iterations; i++) {
        msa_pool_chunk(bench_pool, bench_keys, bench_values, BENCH_SEQ);
    }
    teardown_msa();
    return iterations;
}

int bench_msa_routing_sweep_throughput() {
    if (setup_msa() != 0) return BENCHMARK_RESULT_CORE_FAILED;
    
    /* Pre-fill pool with 1000 active compressed memory chunks */
    for (int i = 0; i < 1000; i++) {
        msa_pool_chunk(bench_pool, bench_keys, bench_values, BENCH_SEQ);
    }
    
    int sweeps = 40000;
    for (int i = 0; i < sweeps; i++) {
        msa_route_top_1(bench_pool, bench_query);
    }
    teardown_msa();
    return sweeps;
}

int main(void) {
    benchmark msa_benchmarks[] = {
        BENCHMARK_CASE(bench_msa_pool_chunk_throughput),
        BENCHMARK_CASE(bench_msa_routing_sweep_throughput),
        {BENCHMARK_END}
    };
    
    benchmark_suite suites[] = {
        {"Memory Sparse Attention Latency Volumes", msa_benchmarks},
        {BENCHMARK_END}
    };
    
    benchmark_suite_run(suites, NULL);
    return 0;
}
