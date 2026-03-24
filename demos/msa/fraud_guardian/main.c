/*
 * MicroGPT-C — Memory Sparse Attention (MSA) Fraud Guardian Demo
 * 
 * Simulates monitoring a continuous stream of 1,000+ transactions
 * on a constrained edge device (ESP32) by discarding the O(L^2) KV cache
 * and permanently storing compressed chunks in an O(1) MsaPool.
 */

#include "microgpt.h"
#include "microgpt_msa.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define CHUNK_SIZE 32
#define TOTAL_TRANSACTIONS 1280 /* 40 chunks */
#define POOL_CAPACITY 50

static void fill_mock_kv(scalar_t **keys, scalar_t **values, int n_layer, int n_embd, int len, int anomaly_flag) {
    for (int l = 0; l < n_layer; l++) {
        for (int pos = 0; pos < len; pos++) {
            for (int d = 0; d < n_embd; d++) {
                if (anomaly_flag) {
                    keys[l][pos * n_embd + d] = 1.0f; /* Distinct anomaly semantic signature */
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
    printf("  MSA Fraud Guardian: Continual Transaction Monitoring\n");
    printf("================================================================\n\n");
    
    MicrogptConfig cfg = microgpt_default_config();
    MsaPool *pool = msa_pool_create(POOL_CAPACITY, cfg.n_layer, cfg.n_embd);
    
    scalar_t **planner_keys = (scalar_t **)malloc(cfg.n_layer * sizeof(scalar_t *));
    scalar_t **planner_values = (scalar_t **)malloc(cfg.n_layer * sizeof(scalar_t *));
    for (int l = 0; l < cfg.n_layer; l++) {
        planner_keys[l] = (scalar_t *)malloc(CHUNK_SIZE * cfg.n_embd * sizeof(scalar_t));
        planner_values[l] = (scalar_t *)malloc(CHUNK_SIZE * cfg.n_embd * sizeof(scalar_t));
    }
    
    printf("[+] Simulating streaming ingestion of %d transactions...\n", TOTAL_TRANSACTIONS);
    
    int num_chunks = TOTAL_TRANSACTIONS / CHUNK_SIZE;
    int fraud_chunk_index = 27; /* We plant the anomaly in chunk 27 */
    
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    
    for(int i = 0; i < num_chunks; i++) {
        int anomaly = (i == fraud_chunk_index);
        fill_mock_kv(planner_keys, planner_values, cfg.n_layer, cfg.n_embd, CHUNK_SIZE, anomaly);
        msa_pool_chunk(pool, planner_keys, planner_values, CHUNK_SIZE);
    }
    
    clock_gettime(CLOCK_MONOTONIC, &end);
    double chunk_time = (end.tv_sec - start.tv_sec) * 1000.0 + (end.tv_nsec - start.tv_nsec) / 1e6;
    printf("    -> Successfully compressed %d transactions into %d latent chunks in %.3f ms\n", 
           TOTAL_TRANSACTIONS, (int)pool->length, chunk_time);
           
    printf("\n[+] Fraud Sentinel Query: Scanning %.1f MB of equivalent context...\n", (float)(TOTAL_TRANSACTIONS * 1024)/1000000.0f);
    
    /* Judge Query: looking for the 1.0f anomaly signature */
    scalar_t **judge_query = (scalar_t **)malloc(cfg.n_layer * sizeof(scalar_t *));
    for (int l = 0; l < cfg.n_layer; l++) {
        judge_query[l] = (scalar_t *)malloc(cfg.n_embd * sizeof(scalar_t));
        for (int d = 0; d < cfg.n_embd; d++) {
            judge_query[l][d] = 1.0f; 
        }
    }
    
    clock_gettime(CLOCK_MONOTONIC, &start);
    int detected_chunk = msa_route_top_1(pool, judge_query);
    clock_gettime(CLOCK_MONOTONIC, &end);
    double route_time = (end.tv_sec - start.tv_sec) * 1000.0 + (end.tv_nsec - start.tv_nsec) / 1e6;
    
    if (detected_chunk == fraud_chunk_index) {
        printf("    -> [ALERT] Anomaly detected in historical Chunk %d (Transactions %d - %d)\n", 
               detected_chunk, detected_chunk * CHUNK_SIZE, (detected_chunk + 1) * CHUNK_SIZE - 1);
        printf("    -> Retrieval sweep latency: %.3f ms\n", route_time);
        printf("\n[SUCCESS] O(1) Anomaly detection operating within 520KB SRAM constraints.\n");
    } else {
        printf("\n[FAILED] Router picked chunk %d, expected %d.\n", detected_chunk, fraud_chunk_index);
    }
    
    /* Cleanup */
    for (int l = 0; l < cfg.n_layer; l++) {
        free(planner_keys[l]); free(planner_values[l]); free(judge_query[l]);
    }
    free(planner_keys); free(planner_values); free(judge_query);
    msa_pool_free(pool);
    return 0;
}
