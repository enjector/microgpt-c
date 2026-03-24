/*
 * MicroGPT-C — Memory Sparse Attention (MSA) Neural Pipeline Benchmark
 * 
 * Simulates the Inter-Organelle communication overhead bottleneck.
 * Phase A (Without MSA): Measures the latency of stringifying sequences via `sprintf` and parsing them.
 * Phase B (With MSA): Measures the latency of passing continuous Key/Value floats straight to `msa_pool_chunk`.
 */

#include "microgpt.h"
#include "microgpt_msa.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define BOARD_STATE_SIZE 42
#define PIPELINE_ITERATIONS 1000

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

int main(void) {
    printf("================================================================\n");
    printf("  MSA Neural Pipeline Benchmark\n");
    printf("================================================================\n");
    
    MicrogptConfig cfg = microgpt_default_config();
    struct timespec start, end;
    
    printf("\n[Phase A] WITHOUT MSA: Kanban String Translation\n");
    printf("Target: Execute %d inter-organelle state exchanges.\n", PIPELINE_ITERATIONS);
    printf("Mechanism: `malloc` volatile char boundaries, simulate `sprintf` sequences, string-match parsing.\n");
    
    clock_gettime(CLOCK_MONOTONIC, &start);
    long volatile text_checksum = 0;
    
    for (int i = 0; i < PIPELINE_ITERATIONS; i++) {
        /* Simulate formatting a board state */
        char *buffer = (char *)malloc(128);
        sprintf(buffer, "board=.......................XXX.......|valid=1,2,3");
        
        /* Simulate parsing */
        char *valid_str = strstr(buffer, "valid=");
        if (valid_str) {
            text_checksum += valid_str[6] - '0';
        }
        free(buffer);
    }
    
    if (text_checksum == 0) {
        text_checksum = 1;
    }
    
    clock_gettime(CLOCK_MONOTONIC, &end);
    double phase_a_time = (end.tv_sec - start.tv_sec) * 1000.0 + (end.tv_nsec - start.tv_nsec) / 1e6;
    
    printf("-> Metric: Evaluated Volatile serialization / deserialization loop over char interfaces.\n");
    printf("-> Total Text serialization parsing overhead: %.3f ms\n", phase_a_time);
    
    printf("\n----------------------------------------------------------------\n");
    printf("\n[Phase B] WITH MSA: Latent Continuous Pooling Handshake\n");
    printf("Target: Execute %d inter-organelle mathematical chunk handoffs.\n", PIPELINE_ITERATIONS);
    printf("Mechanism: Seamless extraction and semantic consolidation via `msa_pool_chunk()`.\n");
    
    MsaPool *pool = msa_pool_create(PIPELINE_ITERATIONS + 10, cfg.n_layer, cfg.n_embd);
    
    scalar_t **planner_keys = (scalar_t **)malloc(cfg.n_layer * sizeof(scalar_t *));
    scalar_t **planner_values = (scalar_t **)malloc(cfg.n_layer * sizeof(scalar_t *));
    for (int l = 0; l < cfg.n_layer; l++) {
        planner_keys[l] = (scalar_t *)malloc(BOARD_STATE_SIZE * cfg.n_embd * sizeof(scalar_t));
        planner_values[l] = (scalar_t *)malloc(BOARD_STATE_SIZE * cfg.n_embd * sizeof(scalar_t));
    }
    
    fill_mock_kv(planner_keys, planner_values, cfg.n_layer, cfg.n_embd, BOARD_STATE_SIZE);
    
    clock_gettime(CLOCK_MONOTONIC, &start);
    for (int i = 0; i < PIPELINE_ITERATIONS; i++) {
        msa_pool_chunk(pool, planner_keys, planner_values, BOARD_STATE_SIZE);
    }
    clock_gettime(CLOCK_MONOTONIC, &end);
    double phase_b_time = (end.tv_sec - start.tv_sec) * 1000.0 + (end.tv_nsec - start.tv_nsec) / 1e6;
    
    printf("-> Metric: Synchronously flattened internal states straight into array structure.\n");
    printf("-> Total Latent Semantic Handoff latency: %.3f ms\n", phase_b_time);
    
    if (phase_b_time > 0) {
        printf("\n[CONCLUSION] Evading external text boundary limitations bypassing the Char Stringification IO bottleneck, accelerating frame rates by %.1fx.\n\n", phase_a_time / phase_b_time);
    }
    
    /* Cleanup */
    for (int l = 0; l < cfg.n_layer; l++) {
        free(planner_keys[l]); free(planner_values[l]);
    }
    free(planner_keys); free(planner_values);
    msa_pool_free(pool);
    return 0;
}
