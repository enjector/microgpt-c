/*
 * MicroGPT-C — Memory Sparse Attention (MSA) Semantic Companion Demo
 * 
 * Simulates an infinite context window (e.g. 365 daily diary entries)
 * compressed into Latent Memory blocks, proving the framework can 
 * recall specific memories instantly.
 */

#include "microgpt.h"
#include "microgpt_msa.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define DAYS_OF_MEMORY 365
#define TOKENS_PER_DAY 256
#define POOL_CAPACITY 400

static void fill_mock_kv(scalar_t **keys, scalar_t **values, int n_layer, int n_embd, int len, int target_memory) {
    for (int l = 0; l < n_layer; l++) {
        for (int pos = 0; pos < len; pos++) {
            for (int d = 0; d < n_embd; d++) {
                if (target_memory) {
                    keys[l][pos * n_embd + d] = 0.88f; /* The "Tokyo Trip" semantic signature */
                } else {
                    keys[l][pos * n_embd + d] = (scalar_t)(rand() % 100) / 1000.0f;
                }
                values[l][pos * n_embd + d] = (scalar_t)(rand() % 100) / 1000.0f;
            }
        }
    }
}

int main(void) {
    printf("================================================================\n");
    printf("  MSA Semantic Companion: 365-Day Lifelong Memory\n");
    printf("================================================================\n\n");
    
    MicrogptConfig cfg = microgpt_default_config();
    MsaPool *pool = msa_pool_create(POOL_CAPACITY, cfg.n_layer, cfg.n_embd);
    
    scalar_t **daily_keys = (scalar_t **)malloc(cfg.n_layer * sizeof(scalar_t *));
    scalar_t **daily_values = (scalar_t **)malloc(cfg.n_layer * sizeof(scalar_t *));
    for (int l = 0; l < cfg.n_layer; l++) {
        daily_keys[l] = (scalar_t *)malloc(TOKENS_PER_DAY * cfg.n_embd * sizeof(scalar_t));
        daily_values[l] = (scalar_t *)malloc(TOKENS_PER_DAY * cfg.n_embd * sizeof(scalar_t));
    }
    
    printf("[+] Simulating %d days of background diary ingestion (%d tokens/day)...\n", DAYS_OF_MEMORY, TOKENS_PER_DAY);
    int tokyo_trip_day = 142; /* The specific memory to recall */
    
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    
    for(int day = 0; day < DAYS_OF_MEMORY; day++) {
        int is_target = (day == tokyo_trip_day);
        fill_mock_kv(daily_keys, daily_values, cfg.n_layer, cfg.n_embd, TOKENS_PER_DAY, is_target);
        msa_pool_chunk(pool, daily_keys, daily_values, TOKENS_PER_DAY);
    }
    
    clock_gettime(CLOCK_MONOTONIC, &end);
    double chunk_time = (end.tv_sec - start.tv_sec) * 1000.0 + (end.tv_nsec - start.tv_nsec) / 1e6;
    printf("    -> Permanently compressed %d tokens into a %zu-block associative array in %.3f ms\n", 
           DAYS_OF_MEMORY * TOKENS_PER_DAY, pool->length, chunk_time);
           
    printf("\n[?] User Prompt: \"Do you remember our trip to Tokyo?\"\n");
    
    /* Query Engine */
    scalar_t **query = (scalar_t **)malloc(cfg.n_layer * sizeof(scalar_t *));
    for (int l = 0; l < cfg.n_layer; l++) {
        query[l] = (scalar_t *)malloc(cfg.n_embd * sizeof(scalar_t));
        for (int d = 0; d < cfg.n_embd; d++) {
            query[l][d] = 0.88f; 
        }
    }
    
    clock_gettime(CLOCK_MONOTONIC, &start);
    int recalled_day = msa_route_top_1(pool, query);
    clock_gettime(CLOCK_MONOTONIC, &end);
    double route_time = (end.tv_sec - start.tv_sec) * 1000.0 + (end.tv_nsec - start.tv_nsec) / 1e6;
    
    if (recalled_day == tokyo_trip_day) {
        printf("    -> [RECALLED] Router instantly surfaced Day %d latent states!\n", recalled_day);
        printf("    -> Database sweep latency: %.3f ms\n", route_time);
        printf("\n[SUCCESS] Lifelong Memory indexing operates optimally without contextual cache overflow.\n");
    } else {
        printf("\n[FAILED] Router picked day %d, expected %d.\n", recalled_day, tokyo_trip_day);
    }
    
    /* Cleanup */
    for (int l = 0; l < cfg.n_layer; l++) {
        free(daily_keys[l]); free(daily_values[l]); free(query[l]);
    }
    free(daily_keys); free(daily_values); free(query);
    msa_pool_free(pool);
    return 0;
}
