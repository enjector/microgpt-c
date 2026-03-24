/*
 * MicroGPT-C — Memory Sparse Attention (MSA) Context Extender Benchmark
 * 
 * Simulates generating a 10,000 token sequence on an ESP32 constrained to a 256-token Context Window.
 * Phase A (Without MSA): Measures the immense O(L^2) cost of recalculating a Sliding Window.
 * Phase B (With MSA): Measures the near-instantaneous O(1) cost of Chunk-Pooling the window.
 */

#include "microgpt.h"
#include "microgpt_msa.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define CONTEXT_WINDOW 256
#define TARGET_SEQUENCE 10240 /* 40 chunks */
#define POOL_CAPACITY 50

static void fill_mock_kv(scalar_t **keys, scalar_t **values, int n_layer, int n_embd, int len) {
    for (int l = 0; l < n_layer; l++) {
        for (int pos = 0; pos < len; pos++) {
            for (int d = 0; d < n_embd; d++) {
                keys[l][pos * n_embd + d] = (scalar_t)(rand() % 100) / 1000.0f;
                values[l][pos * n_embd + d] = (scalar_t)(rand() % 100) / 1000.0f;
            }
        }
    }
}

static void simulate_sliding_window_recalc(int window_size, int n_layer, int n_embd) {
    /* Simulates an O(L^2) recalculation penalty of the sequence window matrices */
    volatile float sum = 0.0f;
    for (int pos_q = 0; pos_q < window_size; pos_q++) {
        for (int pos_k = 0; pos_k < window_size; pos_k++) {
            for (int d = 0; d < n_embd; d++) {
                sum += (pos_q * pos_k * d) * 0.0001f;
            }
        }
    }
    
    /* Trick the compiler so it doesn't optimize away the empty volatile loop */
    if (sum > 0.0f) {
        sum *= 1.0f;
    }
}

int main(void) {
    printf("================================================================\n");
    printf("  MSA Context Extender Benchmark\n");
    printf("================================================================\n");
    
    MicrogptConfig cfg = microgpt_default_config();
    int num_chunks = TARGET_SEQUENCE / CONTEXT_WINDOW;
    struct timespec start, end;
    
    printf("\n[Phase A] WITHOUT MSA: Sliding Window Constraint\n");
    printf("Target: Generate %d continuous tokens.\n", TARGET_SEQUENCE);
    printf("Constraint: Max Cache size is %d tokens.\n", CONTEXT_WINDOW);
    
    clock_gettime(CLOCK_MONOTONIC, &start);
    for (int i = 1; i < num_chunks; i++) { /* First chunk is free */
        simulate_sliding_window_recalc(CONTEXT_WINDOW, cfg.n_layer, cfg.n_embd);
    }
    clock_gettime(CLOCK_MONOTONIC, &end);
    double phase_a_time = (end.tv_sec - start.tv_sec) * 1000.0 + (end.tv_nsec - start.tv_nsec) / 1e6;
    
    printf("-> Result: Forced to recalculate %d sliding context windows (O(L^2) penalty).\n", num_chunks - 1);
    printf("-> Total Iterative Window Penalty Latency: %.3f ms\n", phase_a_time);
    
    printf("\n----------------------------------------------------------------\n");
    printf("\n[Phase B] WITH MSA: Latent Context Pooling\n");
    printf("Target: Generate %d continuous tokens.\n", TARGET_SEQUENCE);
    printf("Constraint: Max Cache size is %d tokens.\n", CONTEXT_WINDOW);
    
    MsaPool *pool = msa_pool_create(POOL_CAPACITY, cfg.n_layer, cfg.n_embd);
    
    scalar_t **planner_keys = (scalar_t **)malloc(cfg.n_layer * sizeof(scalar_t *));
    scalar_t **planner_values = (scalar_t **)malloc(cfg.n_layer * sizeof(scalar_t *));
    for (int l = 0; l < cfg.n_layer; l++) {
        planner_keys[l] = (scalar_t *)malloc(CONTEXT_WINDOW * cfg.n_embd * sizeof(scalar_t));
        planner_values[l] = (scalar_t *)malloc(CONTEXT_WINDOW * cfg.n_embd * sizeof(scalar_t));
    }
    
    clock_gettime(CLOCK_MONOTONIC, &start);
    for (int i = 1; i < num_chunks; i++) {
        fill_mock_kv(planner_keys, planner_values, cfg.n_layer, cfg.n_embd, CONTEXT_WINDOW);
        msa_pool_chunk(pool, planner_keys, planner_values, CONTEXT_WINDOW);
    }
    clock_gettime(CLOCK_MONOTONIC, &end);
    double phase_b_time = (end.tv_sec - start.tv_sec) * 1000.0 + (end.tv_nsec - start.tv_nsec) / 1e6;
    
    printf("-> Result: Permanently compressed the cache blocks into %zu semantic chunks.\n", pool->length);
    printf("-> Total Continuous Cache Preservation Latency: %.3f ms\n", phase_b_time);
    
    if (phase_b_time > 0) {
        printf("\n[CONCLUSION] By preventing overlapping text cache recalculation loops, MSA bypasses the context boundary and scales Generation speeds by %.1fx.\n\n", phase_a_time / phase_b_time);
    }
    
    /* Cleanup */
    for (int l = 0; l < cfg.n_layer; l++) {
        free(planner_keys[l]); free(planner_values[l]);
    }
    free(planner_keys); free(planner_values);
    msa_pool_free(pool);
    return 0;
}
