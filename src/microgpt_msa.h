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
#include <stdint.h>

#ifdef ENABLE_TURBOQUANT
#include "microgpt_turboquant.h"
extern TurboQuant g_tq;
#elif defined(ENABLE_ROTORQUANT)
#include "microgpt_rotorquant.h"
extern RotorQuant g_rq;
#endif

/*
 * MsaPool — Tiered latent memory arena.
 * Stores compressed KV chunks as continuous float vectors.
 */
typedef struct {
#if defined(ENABLE_TURBOQUANT) || defined(ENABLE_ROTORQUANT)
    uint32_t *tq_keys_idx;
    int8_t *tq_keys_qjl;
    float *tq_keys_rnorm;

    uint32_t *tq_values_idx;
    int8_t *tq_values_qjl;
    float *tq_values_rnorm;
#else
    scalar_t *keys;     /* shape: [capacity, n_layer, n_embd] */
    scalar_t *values;   /* shape: [capacity, n_layer, n_embd] */
#endif
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

/* ============================================================
 *  Sliding-Window Recency (DeepSeek-V4 §2.3.3 port)
 * ============================================================
 *
 * MsaRecency — a ring buffer of the most-recent n_win uncompressed K/V
 * tokens, surviving across chunking events. Ports the V4 idea of a
 * "supplementary attention branch in a sliding window manner ... for
 * better modelling of local dependencies."
 *
 * Why it matters for MicroGPT-C MSA. The default MSA flow chunks
 * block_size/2 oldest tokens at every overflow event. After chunking,
 * the model's only access to those tokens is via the (single) selected
 * compressed chunk that gets re-injected at position 0. Local detail
 * that was just-recent gets blurred into the chunk's mean-pool
 * summary. A sliding-window recency tail preserves the last n_win
 * tokens at FULL fidelity, independent of routing decisions, so the
 * attention has a guaranteed local-coherence signal even when the
 * pool's best-chunk routing is wrong.
 *
 * The recency buffer is intentionally a separate struct from MsaPool
 * — it composes orthogonally and doesn't break existing MSA users.
 *
 * See RESEARCH_DEEPSEEK_V4_MSA_SLIDING_WINDOW_RECENCY.md for the
 * benchmark and measured impact.
 */
typedef struct {
    scalar_t *keys;     /* shape: [capacity, n_layer, n_embd] */
    scalar_t *values;   /* shape: [capacity, n_layer, n_embd] */
    size_t capacity;    /* n_win = max number of recency tokens retained */
    size_t length;      /* current number of valid entries (<= capacity) */
    size_t head;        /* ring head: index of OLDEST entry when full */
    int n_layer;
    int n_embd;
} MsaRecency;

/* Allocate a sliding-window recency buffer holding capacity tokens. */
MsaRecency *msa_recency_create(size_t capacity, int n_layer, int n_embd);

/* Free buffer. */
void msa_recency_free(MsaRecency *rec);

/* Reset to empty (does not free memory). */
void msa_recency_reset(MsaRecency *rec);

/* Push one token's K/V across all layers into the ring buffer.
 * If full, the oldest entry is evicted. */
void msa_recency_push(MsaRecency *rec,
                      scalar_t **token_keys,
                      scalar_t **token_values);

/* Copy the entire recency window into active_keys/active_values at positions
 * [start_pos, start_pos + length). Tokens are written in chronological order
 * (oldest → newest) so the model sees a properly-ordered local context.
 * Returns the number of positions written. */
size_t msa_recency_inject(const MsaRecency *rec,
                          scalar_t **active_keys,
                          scalar_t **active_values,
                          size_t start_pos);

#endif /* MICROGPT_MSA_H */
