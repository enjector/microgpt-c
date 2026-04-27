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
#ifdef ENABLE_TURBOQUANT
    pool->tq_keys_idx   = (uint32_t *)calloc(capacity * n_layer * n_embd, sizeof(uint32_t));
    pool->tq_keys_qjl   = (int8_t *)  calloc(capacity * n_layer * n_embd, sizeof(int8_t));
    pool->tq_keys_rnorm = (float *)   calloc(capacity * n_layer, sizeof(float));

    pool->tq_values_idx   = (uint32_t *)calloc(capacity * n_layer * n_embd, sizeof(uint32_t));
    pool->tq_values_qjl   = (int8_t *)  calloc(capacity * n_layer * n_embd, sizeof(int8_t));
    pool->tq_values_rnorm = (float *)   calloc(capacity * n_layer, sizeof(float));

    if (!pool->tq_keys_idx || !pool->tq_values_idx) {
        msa_pool_free(pool);
        return NULL;
    }
#elif defined(ENABLE_ROTORQUANT)
    pool->tq_keys_idx   = (uint32_t *)calloc(capacity * n_layer * n_embd, sizeof(uint32_t));
    pool->tq_keys_qjl   = (int8_t *)  calloc(capacity * n_layer * n_embd, sizeof(int8_t));
    pool->tq_keys_rnorm = (float *)   calloc(capacity * n_layer, sizeof(float));

    pool->tq_values_idx   = (uint32_t *)calloc(capacity * n_layer * n_embd, sizeof(uint32_t));
    pool->tq_values_qjl   = (int8_t *)  calloc(capacity * n_layer * n_embd, sizeof(int8_t));
    pool->tq_values_rnorm = (float *)   calloc(capacity * n_layer, sizeof(float));

    if (!pool->tq_keys_idx || !pool->tq_values_idx) {
        msa_pool_free(pool);
        return NULL;
    }
#else
    pool->keys = (scalar_t *)calloc(capacity * n_layer * n_embd, sizeof(scalar_t));
    pool->values = (scalar_t *)calloc(capacity * n_layer * n_embd, sizeof(scalar_t));
    
    if (!pool->keys || !pool->values) {
        msa_pool_free(pool);
        return NULL;
    }
#endif
    
    return pool;
}

void msa_pool_free(MsaPool *pool) {
    if (!pool) return;
#ifdef ENABLE_TURBOQUANT
    if (pool->tq_keys_idx) free(pool->tq_keys_idx);
    if (pool->tq_keys_qjl) free(pool->tq_keys_qjl);
    if (pool->tq_keys_rnorm) free(pool->tq_keys_rnorm);
    if (pool->tq_values_idx) free(pool->tq_values_idx);
    if (pool->tq_values_qjl) free(pool->tq_values_qjl);
    if (pool->tq_values_rnorm) free(pool->tq_values_rnorm);
#elif defined(ENABLE_ROTORQUANT)
    if (pool->tq_keys_idx) free(pool->tq_keys_idx);
    if (pool->tq_keys_qjl) free(pool->tq_keys_qjl);
    if (pool->tq_keys_rnorm) free(pool->tq_keys_rnorm);
    if (pool->tq_values_idx) free(pool->tq_values_idx);
    if (pool->tq_values_qjl) free(pool->tq_values_qjl);
    if (pool->tq_values_rnorm) free(pool->tq_values_rnorm);
#else
    free(pool->keys);
    free(pool->values);
#endif
    free(pool);
}

int msa_pool_chunk(MsaPool *pool, scalar_t **active_keys, scalar_t **active_values, size_t chunk_len) {
    if (pool->length >= pool->capacity || chunk_len == 0) return -1;
    
    /* Find offset for the new chunk */
    size_t offset = pool->length * pool->n_layer * pool->n_embd;
    
    /* Mean pooling across the chunk window */
    for (int l = 0; l < pool->n_layer; l++) {
#ifdef ENABLE_TURBOQUANT
        float *chunk_mean_k = pool->n_embd > 0 ? (float*)malloc(pool->n_embd * sizeof(float)) : NULL;
        float *chunk_mean_v = pool->n_embd > 0 ? (float*)malloc(pool->n_embd * sizeof(float)) : NULL;
#elif defined(ENABLE_ROTORQUANT)
        float *chunk_mean_k = pool->n_embd > 0 ? (float*)malloc(pool->n_embd * sizeof(float)) : NULL;
        float *chunk_mean_v = pool->n_embd > 0 ? (float*)malloc(pool->n_embd * sizeof(float)) : NULL;
#endif
        for (int d = 0; d < pool->n_embd; d++) {
            scalar_t sum_k = 0.0f;
            scalar_t sum_v = 0.0f;
            
            for (size_t t = 0; t < chunk_len; t++) {
                sum_k += active_keys[l][t * pool->n_embd + d];
                sum_v += active_values[l][t * pool->n_embd + d];
            }
            
            scalar_t tk = sum_k / (scalar_t)chunk_len;
            scalar_t tv = sum_v / (scalar_t)chunk_len;
#ifdef ENABLE_TURBOQUANT
            if (chunk_mean_k) chunk_mean_k[d] = (float)tk;
            if (chunk_mean_v) chunk_mean_v[d] = (float)tv;
#elif defined(ENABLE_ROTORQUANT)
            if (chunk_mean_k) chunk_mean_k[d] = (float)tk;
            if (chunk_mean_v) chunk_mean_v[d] = (float)tv;
#else
            pool->keys[offset + l * pool->n_embd + d] = tk;
            pool->values[offset + l * pool->n_embd + d] = tv;
#endif
        }
#ifdef ENABLE_TURBOQUANT
        uint32_t *k_idx = &pool->tq_keys_idx[offset + l * pool->n_embd];
        int8_t *k_qjl = &pool->tq_keys_qjl[offset + l * pool->n_embd];
        float *k_rnorm = &pool->tq_keys_rnorm[pool->length * pool->n_layer + l];
        turboquant_quant_prod(&g_tq, chunk_mean_k, k_idx, k_qjl, k_rnorm);

        uint32_t *v_idx = &pool->tq_values_idx[offset + l * pool->n_embd];
        int8_t *v_qjl = &pool->tq_values_qjl[offset + l * pool->n_embd];
        float *v_rnorm = &pool->tq_values_rnorm[pool->length * pool->n_layer + l];
        turboquant_quant_prod(&g_tq, chunk_mean_v, v_idx, v_qjl, v_rnorm);

        if (chunk_mean_k) free(chunk_mean_k);
        if (chunk_mean_v) free(chunk_mean_v);
#elif defined(ENABLE_ROTORQUANT)
        uint32_t *k_idx = &pool->tq_keys_idx[offset + l * pool->n_embd];
        int8_t *k_qjl = &pool->tq_keys_qjl[offset + l * pool->n_embd];
        float *k_rnorm = &pool->tq_keys_rnorm[pool->length * pool->n_layer + l];
        rotorquant_quant_prod(&g_rq, chunk_mean_k, k_idx, k_qjl, k_rnorm);

        uint32_t *v_idx = &pool->tq_values_idx[offset + l * pool->n_embd];
        int8_t *v_qjl = &pool->tq_values_qjl[offset + l * pool->n_embd];
        float *v_rnorm = &pool->tq_values_rnorm[pool->length * pool->n_layer + l];
        rotorquant_quant_prod(&g_rq, chunk_mean_v, v_idx, v_qjl, v_rnorm);

        if (chunk_mean_k) free(chunk_mean_k);
        if (chunk_mean_v) free(chunk_mean_v);
#endif
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
        
#ifdef ENABLE_TURBOQUANT
        float *dequant_k = pool->n_embd > 0 ? (float*)malloc(pool->n_embd * sizeof(float)) : NULL;
        if (dequant_k) {
            uint32_t *idx = &pool->tq_keys_idx[p_offset];
            int8_t *qjl = &pool->tq_keys_qjl[p_offset];
            float rnorm = pool->tq_keys_rnorm[i * pool->n_layer + l];
            turboquant_dequant_prod(&g_tq, idx, qjl, rnorm, dequant_k);
        }
#elif defined(ENABLE_ROTORQUANT)
        float *dequant_k = pool->n_embd > 0 ? (float*)malloc(pool->n_embd * sizeof(float)) : NULL;
        if (dequant_k) {
            uint32_t *idx = &pool->tq_keys_idx[p_offset];
            int8_t *qjl = &pool->tq_keys_qjl[p_offset];
            float rnorm = pool->tq_keys_rnorm[i * pool->n_layer + l];
            rotorquant_dequant_prod(&g_rq, idx, qjl, rnorm, dequant_k);
        }
#endif

        for (int d = 0; d < pool->n_embd; d++) {
#if defined(ENABLE_TURBOQUANT) || defined(ENABLE_ROTORQUANT)
            scalar_t p_val = dequant_k ? (scalar_t)dequant_k[d] : 0.0f;
#else
            scalar_t p_val = pool->keys[p_offset + d];
#endif
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

#if defined(ENABLE_TURBOQUANT) || defined(ENABLE_ROTORQUANT)
        if (dequant_k) free(dequant_k);
#endif
    }
    
    return best_idx;
}

/* ============================================================
 *  Sliding-Window Recency implementation
 * ============================================================ */

MsaRecency *msa_recency_create(size_t capacity, int n_layer, int n_embd) {
    if (capacity == 0 || n_layer <= 0 || n_embd <= 0) return NULL;
    MsaRecency *rec = (MsaRecency *)calloc(1, sizeof(MsaRecency));
    if (!rec) return NULL;
    rec->capacity = capacity;
    rec->n_layer = n_layer;
    rec->n_embd = n_embd;
    rec->length = 0;
    rec->head = 0;
    rec->keys   = (scalar_t *)calloc(capacity * (size_t)n_layer * (size_t)n_embd, sizeof(scalar_t));
    rec->values = (scalar_t *)calloc(capacity * (size_t)n_layer * (size_t)n_embd, sizeof(scalar_t));
    if (!rec->keys || !rec->values) {
        msa_recency_free(rec);
        return NULL;
    }
    return rec;
}

void msa_recency_free(MsaRecency *rec) {
    if (!rec) return;
    free(rec->keys);
    free(rec->values);
    free(rec);
}

void msa_recency_reset(MsaRecency *rec) {
    if (!rec) return;
    rec->length = 0;
    rec->head = 0;
}

/* Internal: write one token's K/V (across all layers) into the ring slot at
 * absolute index `slot`. Slot must be < capacity. */
static void _msa_recency_write_slot(MsaRecency *rec, size_t slot,
                                    scalar_t **token_keys,
                                    scalar_t **token_values) {
    const size_t stride_layer = (size_t)rec->n_embd;
    const size_t stride_token = (size_t)rec->n_layer * stride_layer;
    for (int l = 0; l < rec->n_layer; l++) {
        scalar_t *dst_k = rec->keys   + slot * stride_token + (size_t)l * stride_layer;
        scalar_t *dst_v = rec->values + slot * stride_token + (size_t)l * stride_layer;
        for (int d = 0; d < rec->n_embd; d++) {
            dst_k[d] = token_keys[l][d];
            dst_v[d] = token_values[l][d];
        }
    }
}

void msa_recency_push(MsaRecency *rec,
                      scalar_t **token_keys,
                      scalar_t **token_values) {
    if (!rec || !token_keys || !token_values) return;
    if (rec->length < rec->capacity) {
        /* Not yet full — append at end of the chronological sequence. */
        size_t slot = rec->length; /* head still 0 while filling */
        _msa_recency_write_slot(rec, slot, token_keys, token_values);
        rec->length++;
    } else {
        /* Full — overwrite oldest entry (at head), advance head. */
        _msa_recency_write_slot(rec, rec->head, token_keys, token_values);
        rec->head = (rec->head + 1) % rec->capacity;
    }
}

size_t msa_recency_inject(const MsaRecency *rec,
                          scalar_t **active_keys,
                          scalar_t **active_values,
                          size_t start_pos) {
    if (!rec || rec->length == 0) return 0;
    const size_t stride_layer = (size_t)rec->n_embd;
    const size_t stride_token = (size_t)rec->n_layer * stride_layer;
    /* Walk the ring chronologically: oldest = head when full, else 0. */
    size_t start = (rec->length < rec->capacity) ? 0 : rec->head;
    for (size_t i = 0; i < rec->length; i++) {
        size_t ring = (start + i) % rec->capacity;
        for (int l = 0; l < rec->n_layer; l++) {
            const scalar_t *src_k = rec->keys   + ring * stride_token + (size_t)l * stride_layer;
            const scalar_t *src_v = rec->values + ring * stride_token + (size_t)l * stride_layer;
            scalar_t *dst_k = active_keys[l]   + (start_pos + i) * stride_layer;
            scalar_t *dst_v = active_values[l] + (start_pos + i) * stride_layer;
            for (int d = 0; d < rec->n_embd; d++) {
                dst_k[d] = src_k[d];
                dst_v[d] = src_v[d];
            }
        }
    }
    return rec->length;
}

void msa_expand_context(const MsaPool *pool, int chunk_idx, scalar_t **active_keys, scalar_t **active_values, size_t pos) {
    if (chunk_idx < 0 || (size_t)chunk_idx >= pool->length) return;
    
    size_t offset = chunk_idx * pool->n_layer * pool->n_embd;
    
    /* Copy the static latent vectors into the specific position in the target KV arrays */
    for (int l = 0; l < pool->n_layer; l++) {
#ifdef ENABLE_TURBOQUANT
        float *dequant_k = pool->n_embd > 0 ? (float*)malloc(pool->n_embd * sizeof(float)) : NULL;
        float *dequant_v = pool->n_embd > 0 ? (float*)malloc(pool->n_embd * sizeof(float)) : NULL;
        if (dequant_k && dequant_v) {
            uint32_t *k_idx = &pool->tq_keys_idx[offset + l * pool->n_embd];
            int8_t *k_qjl = &pool->tq_keys_qjl[offset + l * pool->n_embd];
            float k_rnorm = pool->tq_keys_rnorm[chunk_idx * pool->n_layer + l];
            turboquant_dequant_prod(&g_tq, k_idx, k_qjl, k_rnorm, dequant_k);

            uint32_t *v_idx = &pool->tq_values_idx[offset + l * pool->n_embd];
            int8_t *v_qjl = &pool->tq_values_qjl[offset + l * pool->n_embd];
            float v_rnorm = pool->tq_values_rnorm[chunk_idx * pool->n_layer + l];
            turboquant_dequant_prod(&g_tq, v_idx, v_qjl, v_rnorm, dequant_v);
        }
#elif defined(ENABLE_ROTORQUANT)
        float *dequant_k = pool->n_embd > 0 ? (float*)malloc(pool->n_embd * sizeof(float)) : NULL;
        float *dequant_v = pool->n_embd > 0 ? (float*)malloc(pool->n_embd * sizeof(float)) : NULL;
        if (dequant_k && dequant_v) {
            uint32_t *k_idx = &pool->tq_keys_idx[offset + l * pool->n_embd];
            int8_t *k_qjl = &pool->tq_keys_qjl[offset + l * pool->n_embd];
            float k_rnorm = pool->tq_keys_rnorm[chunk_idx * pool->n_layer + l];
            rotorquant_dequant_prod(&g_rq, k_idx, k_qjl, k_rnorm, dequant_k);

            uint32_t *v_idx = &pool->tq_values_idx[offset + l * pool->n_embd];
            int8_t *v_qjl = &pool->tq_values_qjl[offset + l * pool->n_embd];
            float v_rnorm = pool->tq_values_rnorm[chunk_idx * pool->n_layer + l];
            rotorquant_dequant_prod(&g_rq, v_idx, v_qjl, v_rnorm, dequant_v);
        }
#endif

        for (int d = 0; d < pool->n_embd; d++) {
#if defined(ENABLE_TURBOQUANT) || defined(ENABLE_ROTORQUANT)
            if (dequant_k) active_keys[l][pos * pool->n_embd + d] = (scalar_t)dequant_k[d];
            if (dequant_v) active_values[l][pos * pool->n_embd + d] = (scalar_t)dequant_v[d];
#else
            active_keys[l][pos * pool->n_embd + d] = pool->keys[offset + l * pool->n_embd + d];
            active_values[l][pos * pool->n_embd + d] = pool->values[offset + l * pool->n_embd + d];
#endif
        }

#if defined(ENABLE_TURBOQUANT) || defined(ENABLE_ROTORQUANT)
        if (dequant_k) free(dequant_k);
        if (dequant_v) free(dequant_v);
#endif
    }
}
