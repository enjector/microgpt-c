/*
 * MicroGPT-C — Memory Sparse Attention (MSA) Latent Handoff Demo
 * 
 * Demonstrates the structural viability of replacing string-based Kanbans 
 * with a shared MsaPool of `float*` vectors.
 */

#include "microgpt.h"
#include "microgpt_msa.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define CHUNK_SIZE 10
#define POOL_CAPACITY 100

static void fill_mock_kv(scalar_t **keys, scalar_t **values, int n_layer, int n_embd, int len) {
    for (int l = 0; l < n_layer; l++) {
        for (int pos = 0; pos < len; pos++) {
            for (int d = 0; d < n_embd; d++) {
                keys[l][pos * n_embd + d] = (scalar_t)(rand() % 100) / 100.0f;
                values[l][pos * n_embd + d] = (scalar_t)(rand() % 100) / 100.0f;
            }
        }
    }
}

int main(void) {
    printf("================================================================\n");
    printf("  MSA Latent Handoff Integration Demo\n");
    printf("================================================================\n\n");
    
    MicrogptConfig cfg = microgpt_default_config();
    
    /* 1. Create the MSA Latent Pool */
    MsaPool *pool = msa_pool_create(POOL_CAPACITY, cfg.n_layer, cfg.n_embd);
    if (!pool) {
        fprintf(stderr, "Failed to allocate MsaPool.\n");
        return 1;
    }
    printf("[+] Allocated MsaPool (Capacity: %zu chunks, Shape: [%d, %d])\n", 
           pool->capacity, pool->n_layer, pool->n_embd);
           
    /* Mock Active KV Cache for Planner */
    scalar_t **planner_keys = (scalar_t **)malloc(cfg.n_layer * sizeof(scalar_t *));
    scalar_t **planner_values = (scalar_t **)malloc(cfg.n_layer * sizeof(scalar_t *));
    for (int l = 0; l < cfg.n_layer; l++) {
        planner_keys[l] = (scalar_t *)malloc(CHUNK_SIZE * cfg.n_embd * sizeof(scalar_t));
        planner_values[l] = (scalar_t *)malloc(CHUNK_SIZE * cfg.n_embd * sizeof(scalar_t));
    }
    
    printf("[+] Simulating Planner Organelle execution over %d tokens...\n", CHUNK_SIZE);
    fill_mock_kv(planner_keys, planner_values, cfg.n_layer, cfg.n_embd, CHUNK_SIZE);
    
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    
    /* 2. Planner compresses sequence and dumps to Pool */
    int chunk_id = msa_pool_chunk(pool, planner_keys, planner_values, CHUNK_SIZE);
    if (chunk_id < 0) {
        printf("[-] Failed to pool chunk!\n");
        return 1;
    }
    
    clock_gettime(CLOCK_MONOTONIC, &end);
    double chunk_time = (end.tv_sec - start.tv_sec) * 1000.0 + (end.tv_nsec - start.tv_nsec) / 1e6;
    printf("    -> Pooled %d tokens into Latent Chunk %d in %.3f ms\n", CHUNK_SIZE, chunk_id, chunk_time);
    
    /* Pre-fill the pool with some noise chunks to test routing */
    for (int i = 0; i < 9; i++) {
        fill_mock_kv(planner_keys, planner_values, cfg.n_layer, cfg.n_embd, CHUNK_SIZE);
        msa_pool_chunk(pool, planner_keys, planner_values, CHUNK_SIZE);
    }
    printf("[+] Added 9 noise chunks to pool. Total pool length: %zu\n", pool->length);
    
    /* 3. Judge Organelle queries the Pool */
    /* To simulate a perfect match, Judge query is exactly the mean of Chunk 0 */
    scalar_t **judge_query = (scalar_t **)malloc(cfg.n_layer * sizeof(scalar_t *));
    for (int l = 0; l < cfg.n_layer; l++) {
        judge_query[l] = (scalar_t *)malloc(cfg.n_embd * sizeof(scalar_t));
        for (int d = 0; d < cfg.n_embd; d++) {
            judge_query[l][d] = pool->keys[0 * cfg.n_layer * cfg.n_embd + l * cfg.n_embd + d]; // exact match against chunk 0
        }
    }
    
    printf("[+] Simulating Judge Router Cosine Similarity sweep...\n");
    clock_gettime(CLOCK_MONOTONIC, &start);
    
    int best_chunk = msa_route_top_1(pool, judge_query);
    
    clock_gettime(CLOCK_MONOTONIC, &end);
    double route_time = (end.tv_sec - start.tv_sec) * 1000.0 + (end.tv_nsec - start.tv_nsec) / 1e6;
    
    printf("    -> Router selected Chunk %d (Expected: 0) in %.3f ms\n", best_chunk, route_time);
    
    if (best_chunk == 0) {
        printf("\n[SUCCESS] Pipeline 'Discretisation Wall' bypassed. Routing verified.\n");
    } else {
        printf("\n[FAILED] Router picked wrong chunk.\n");
    }
    
    /* Cleanup */
    for (int l = 0; l < cfg.n_layer; l++) {
        free(planner_keys[l]);
        free(planner_values[l]);
        free(judge_query[l]);
    }
    free(planner_keys);
    free(planner_values);
    free(judge_query);
    msa_pool_free(pool);
    
    return 0;
}
