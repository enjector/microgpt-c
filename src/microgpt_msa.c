/*
 * MicroGPT-C — Memory Sparse Attention (MSA) Core Library
 *
 * MIT License
 */

#include "microgpt_msa.h"
#include <stdlib.h>
#include <math.h>

MsaPool *msa_pool_create(size_t capacity, int n_layer, int n_embd) {
    MsaPool *pool = (MsaPool *)calloc(1, sizeof(MsaPool));
    if (!pool) return NULL;
    
    pool->capacity = capacity;
    pool->n_layer = n_layer;
    pool->n_embd = n_embd;
    pool->length = 0;
    
    /* Allocate contiguous arrays for faster scanning */
    pool->keys = (scalar_t *)calloc(capacity * n_layer * n_embd, sizeof(scalar_t));
    pool->values = (scalar_t *)calloc(capacity * n_layer * n_embd, sizeof(scalar_t));
    
    if (!pool->keys || !pool->values) {
        msa_pool_free(pool);
        return NULL;
    }
    
    return pool;
}

void msa_pool_free(MsaPool *pool) {
    if (!pool) return;
    free(pool->keys);
    free(pool->values);
    free(pool);
}

int msa_pool_chunk(MsaPool *pool, scalar_t **active_keys, scalar_t **active_values, size_t chunk_len) {
    if (pool->length >= pool->capacity || chunk_len == 0) return -1;
    
    /* Find offset for the new chunk */
    size_t offset = pool->length * pool->n_layer * pool->n_embd;
    
    /* Mean pooling across the chunk window */
    for (int l = 0; l < pool->n_layer; l++) {
        for (int d = 0; d < pool->n_embd; d++) {
            scalar_t sum_k = 0.0f;
            scalar_t sum_v = 0.0f;
            
            for (size_t t = 0; t < chunk_len; t++) {
                sum_k += active_keys[l][t * pool->n_embd + d];
                sum_v += active_values[l][t * pool->n_embd + d];
            }
            
            pool->keys[offset + l * pool->n_embd + d] = sum_k / (scalar_t)chunk_len;
            pool->values[offset + l * pool->n_embd + d] = sum_v / (scalar_t)chunk_len;
        }
    }
    
    pool->length++;
    return (int)(pool->length - 1); // return index
}

int msa_route_top_1(const MsaPool *pool, scalar_t **query_keys) {
    if (pool->length == 0) return -1;
    
    int best_idx = -1;
    scalar_t best_sim = -999999.0f; /* We use a very low number to avoid fast-math INFINITY warnings */
    if (best_sim > 0) best_sim = -999999.0f; 
    
    /* We compute Cosine Similarity over the final layer to capture the highest semantic inference */
    int l = pool->n_layer - 1;
    if (l < 0) l = 0; // sanity fallback
    
    for (size_t i = 0; i < pool->length; i++) {
        scalar_t dot = 0.0f, norm_q = 0.0f, norm_p = 0.0f;
        size_t p_offset = i * pool->n_layer * pool->n_embd + l * pool->n_embd;
        
        for (int d = 0; d < pool->n_embd; d++) {
            scalar_t p_val = pool->keys[p_offset + d];
            scalar_t q_val = query_keys[l][d]; /* Assuming query is isolated at [d] */
            
            dot += p_val * q_val;
            norm_q += q_val * q_val;
            norm_p += p_val * p_val;
        }
        
        /* Add epsilon to avoid div by zero */
        scalar_t sim = dot / (M_SQRT(norm_q) * M_SQRT(norm_p) + 1e-8f);
        
        if (sim > best_sim) {
            best_sim = sim;
            best_idx = (int)i;
        }
    }
    
    return best_idx;
}

void msa_expand_context(const MsaPool *pool, int chunk_idx, scalar_t **active_keys, scalar_t **active_values, size_t pos) {
    if (chunk_idx < 0 || (size_t)chunk_idx >= pool->length) return;
    
    size_t offset = chunk_idx * pool->n_layer * pool->n_embd;
    
    /* Copy the static latent vectors into the specific position in the target KV arrays */
    for (int l = 0; l < pool->n_layer; l++) {
        for (int d = 0; d < pool->n_embd; d++) {
            active_keys[l][pos * pool->n_embd + d] = pool->keys[offset + l * pool->n_embd + d];
            active_values[l][pos * pool->n_embd + d] = pool->values[offset + l * pool->n_embd + d];
        }
    }
}
