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

/* ============================================================
 *  Lightning Indexer + Top-K (DeepSeek-V4 §2.3.1 port)
 * ============================================================
 *
 * Replaces the single-layer cosine top-1 retrieval of msa_route_top_1
 * with a multi-layer ReLU-summed scoring function and top-k selection.
 *
 * V4's lightning indexer (eqs. 13-17) is:
 *   I_{t,s} = sum_h w_{t,h} * ReLU(q^I_{t,h} · K^IComp_s / sqrt(d))
 * where the per-head queries q^I and compressed-keys K^IComp come
 * from learned low-rank projections (W^DQ, W^IUQ, W^w).
 *
 * Our parameter-free port uses the model's own multi-LAYER K vectors
 * as the "multi-head" projection: layer 0..L-1 each contribute one
 * ReLU score, with uniform weights. This captures V4's two key
 * structural innovations without learned parameters:
 *
 *   1. ReLU-summed scoring (not raw cosine) — only positive matches
 *      contribute, so noisy negatively-aligned chunks don't drag the
 *      score down. Empirically more selective than cosine.
 *   2. Top-K retrieval (not top-1) — broader historical coverage.
 *      The benchmark injects top-k chunks at positions 0..k-1 in the
 *      active cache after each MSA chunking event.
 *
 * Why not the learnable form: same reason as CSA pooling. MSA routing
 * happens outside the training graph, so backprop through the indexer
 * isn't available without a substantial refactor. Fixed-form lets us
 * measure whether *the technique* helps before committing to a
 * learnable implementation. See RESEARCH_DEEPSEEK_V4_LIGHTNING_INDEXER_TOPK.md
 * for measurements.
 */

/*
 * msa_route_top_k — Lightning-indexer-style top-K retrieval.
 *
 *   For each chunk i in the pool, compute:
 *     score[i] = sum_l ReLU(K_q[l] · K_pool[i][l] / sqrt(n_embd))
 *   then return the top-k indices in descending score order.
 *
 * pool         : the MsaPool instance
 * query_keys   : single token's K vector across all layers (semantic query)
 * k            : number of top chunks to retain (capped at pool->length)
 * indices_out  : caller-allocated output array of length >= k.
 *                indices_out[0] is the highest-scoring chunk; ties broken
 *                by lower index (stable). Slots beyond returned count
 *                are filled with -1.
 * scores_out   : optional (may be NULL); same length as indices_out.
 *
 * Returns the number of valid indices written (<= k, == 0 if pool empty).
 */
int msa_route_top_k(const MsaPool *pool, scalar_t **query_keys,
                    int k, int *indices_out, scalar_t *scores_out);

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
 *  CSA-style Learnable Pooling (DeepSeek-V4 §2.3.1 port)
 * ============================================================
 *
 * Replaces the existing uniform mean pool in msa_pool_chunk with
 * weighted pooling of the form:
 *
 *   chunk_K[d] = sum_t  w[t] * active_K[t][d]
 *   chunk_V[d] = sum_t  w[t] * active_V[t][d]
 *   sum_t w[t] = 1
 *
 * V4's CSA (eqs. 9-12) computes w[t] via softmax of a learned
 * content-projection plus learned positional bias. We provide three
 * fixed-form alternatives that capture the same priors *without*
 * adding trainable parameters or backprop through the pool, so the
 * port can be measured against mean pool on identical training:
 *
 *   MSA_POOL_MODE = 0  (default) Uniform mean   — w[t] = 1/chunk_len
 *   MSA_POOL_MODE = 1  Linear ramp recency       — w[t] ∝ 1 + t/(L-1)
 *   MSA_POOL_MODE = 2  Exponential recency       — w[t] ∝ exp(t/tau)
 *   MSA_POOL_MODE = 3  Content-aware (softmax    — w[t] = softmax_t(
 *                      of cosine-to-anchor)        K[t]·K_last/√n_embd)
 *
 * Why fixed-form rather than learnable: MSA pooling happens *outside*
 * the model's training graph in MicroGPT-C — it's an inference-time
 * compression operation. Making it truly learnable requires
 * integrating MSA into the forward/backward training loop, which is a
 * much larger refactor. Fixed-form lets us measure whether *any*
 * non-uniform weighting helps before committing to that refactor.
 *
 * See RESEARCH_DEEPSEEK_V4_MSA_CSA_LEARNABLE_POOL.md for measurements.
 */
#ifndef MSA_POOL_MODE
#define MSA_POOL_MODE 0
#endif

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
    size_t *positions;  /* shape: [capacity] absolute pos_id at push time */
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

/* Push one token's K/V across all layers into the ring buffer along with
 * its absolute pos_id (the position used when RoPE-rotating its K).
 * If full, the oldest entry is evicted. */
void msa_recency_push(MsaRecency *rec,
                      scalar_t **token_keys,
                      scalar_t **token_values,
                      size_t pos);

/* Copy the entire recency window into active_keys/active_values at positions
 * [start_pos, start_pos + length). Tokens are written in chronological order
 * (oldest → newest) so the model sees a properly-ordered local context.
 * Returns the number of positions written.
 *
 * No rotation compensation. Use msa_recency_inject_rope() when MICROGPT_PARTIAL_ROPE
 * is enabled and the cached K vectors carry stale rotation angles. */
size_t msa_recency_inject(const MsaRecency *rec,
                          scalar_t **active_keys,
                          scalar_t **active_values,
                          size_t start_pos);

/* RoPE-aware injection: same as msa_recency_inject(), but for each cached
 * K vector at original position p_orig that gets re-injected at new slot
 * start_pos+i, rotates the K's last ROPE_DIMS dimensions by the angle
 * difference (start_pos+i − p_orig) per head. This makes each re-injected
 * K's rotation angle consistent with its new physical slot, restoring
 * relative-position attention semantics across MSA chunking events.
 *
 * Requires MICROGPT_PARTIAL_ROPE to be #define'd in the build. n_head is
 * needed because RoPE rotates per-head. Returns positions written. */
size_t msa_recency_inject_rope(const MsaRecency *rec,
                               scalar_t **active_keys,
                               scalar_t **active_values,
                               size_t start_pos,
                               int n_head);

#endif /* MICROGPT_MSA_H */
