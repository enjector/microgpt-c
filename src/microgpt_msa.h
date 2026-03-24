/*
 * MicroGPT-C — Memory Sparse Attention (MSA) Core Library
 *
 * Implements Chunk-Mean Pooling of KV states, Top-K Cosine Routing,
 * and Active Context Expansion for Edge devices.
 */

#ifndef MICROGPT_MSA_H
#define MICROGPT_MSA_H

#include "microgpt.h"
#include <stddef.h>

/*
 * MsaPool — Tiered latent memory arena.
 * Stores compressed KV chunks as continuous float vectors.
 */
typedef struct {
    scalar_t *keys;     /* shape: [capacity, n_layer, n_embd] */
    scalar_t *values;   /* shape: [capacity, n_layer, n_embd] */
    size_t capacity;    /* Max number of pooled chunks */
    size_t length;      /* Current number of pooled chunks */
    int n_layer;
    int n_embd;
} MsaPool;

/* Allocate a new MSA Pool in RAM */
MsaPool *msa_pool_create(size_t capacity, int n_layer, int n_embd);

/* Free the MSA Pool */
void msa_pool_free(MsaPool *pool);

/*
 * msa_pool_chunk: Compress a block of active KV attention states into a single latent vector
 * via mean-pooling, and append it to the MsaPool.
 * 
 * pool: the MsaPool instance
 * active_keys: The layer-separated Keys array from the transformer
 * active_values: The layer-separated Values array from the transformer
 * chunk_len: The number of tokens to average together
 *
 * Returns 0 on success, -1 on failure (pool full).
 */
int msa_pool_chunk(MsaPool *pool, scalar_t **active_keys, scalar_t **active_values, size_t chunk_len);

/*
 * msa_route_top_1: Evaluate a single-token Query vector against all compressed chunks in the pool
 * via Cosine Similarity, returning the index of the single most relevant chunk.
 *
 * pool: the MsaPool instance
 * query_keys: A single token's K vector across all layers. Used as the semantic query.
 *
 * Returns the index (0 to length-1) of the best block, or -1 if empty.
 */
int msa_route_top_1(const MsaPool *pool, scalar_t **query_keys);

/*
 * msa_expand_context: Loads the semantic essence of a chosen compressed chunk
 * back into a single token position in the active KV cache. 
 *
 * pool: the MsaPool instance
 * chunk_idx: The index returned from msa_route_top_1
 * active_keys: The layer-separated Keys array to write to
 * active_values: The layer-separated Values array to write to
 * pos: The offset inside the active arrays to inject the summary token
 */
void msa_expand_context(const MsaPool *pool, int chunk_idx, scalar_t **active_keys, scalar_t **active_values, size_t pos);

#endif /* MICROGPT_MSA_H */
