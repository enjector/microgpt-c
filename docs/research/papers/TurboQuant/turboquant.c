#include "turboquant.h"
#include <stdlib.h>
#include <math.h>
#include <string.h>

// ------------------------------------------------------------------
// Simple fixed-seed RNG (zero dep, reproducible)
static uint64_t rng_state = 0x123456789ABCDEF0ULL;

static inline float rand_gauss(void) {
    // Box-Muller transform – advance state independently for u1 and u2
    rng_state = rng_state * 6364136223846793005ULL + 1442695040888963407ULL;
    float u1 = (float)((rng_state >> 33) + 1u) / (float)(1u << 31);
    rng_state = rng_state * 6364136223846793005ULL + 1442695040888963407ULL;
    float u2 = (float)(rng_state >> 33) / (float)(1u << 31);
    return sqrtf(-2.0f * logf(u1)) * cosf(2.0f * (float)M_PI * u2);
}

// ------------------------------------------------------------------
// Precomputed Lloyd-Max centroids for standard normal N(0,1)
// (scaled by 1/sqrt(d) at init time – exact for high-d Beta limit)
static const float normal_centroids[5][16] = {
    // b=0 unused (index 0 must be a dummy so normal_centroids[b] maps to bit-width b)
    {0},
    // b=1 (2 centroids) – N(0,1) Lloyd-Max: ±sqrt(2/pi)
    {-0.79788456f, +0.79788456f},
    // b=2 (4 centroids) – N(0,1) Lloyd-Max
    {-1.5104f, -0.4528f, +0.4528f, +1.5104f},
    // b=3 (8 centroids) – N(0,1) Lloyd-Max (Max 1960)
    {-2.1520f, -1.3439f, -0.7560f, -0.2451f, +0.2451f, +0.7560f, +1.3439f, +2.1520f},
    // b=4 (16 centroids) – N(0,1) Lloyd-Max (Max 1960)
    {-2.7326f,-2.0690f,-1.6180f,-1.2562f,-0.9423f,-0.6567f,-0.3880f,-0.1284f,
     +0.1284f,+0.3880f,+0.6567f,+0.9423f,+1.2562f,+1.6180f,+2.0690f,+2.7326f}
};

static const int cb_sizes[5] = {0, 2, 4, 8, 16};

// ------------------------------------------------------------------
void turboquant_init(TurboQuant *tq, int d, int b, bool use_rotation) {
    memset(tq, 0, sizeof(*tq));
    tq->d = d;
    tq->b = b;
    tq->use_rotation = use_rotation;

    float sigma = 1.0f / sqrtf((float)d);

    // Precomputed codebooks (scaled for this d).
    // Index 0 is a trivial 1-centroid quantizer (centroid = 0): used internally
    // by turboquant_quant_prod when the target bit-width is b=1.
    tq->codebook_mse = (float**)malloc((b+1) * sizeof(float*));
    tq->cb_sizes = (int*)malloc((b+1) * sizeof(int));
    tq->cb_sizes[0] = 1;
    tq->codebook_mse[0] = (float*)calloc(1, sizeof(float));
    for (int bb = 1; bb <= b; ++bb) {
        tq->cb_sizes[bb] = cb_sizes[bb];
        tq->codebook_mse[bb] = (float*)malloc(cb_sizes[bb] * sizeof(float));
        for (int i = 0; i < cb_sizes[bb]; ++i) {
            tq->codebook_mse[bb][i] = normal_centroids[bb][i] * sigma;
        }
    }

    if (!use_rotation) return;

    // Random rotation Π (QR of Gaussian matrix) – Gram-Schmidt for small d
    tq->Pi = (float*)malloc(d * d * sizeof(float));
    tq->S  = (float*)malloc(d * d * sizeof(float));

    float *G = (float*)malloc(d * d * sizeof(float)); // temporary Gaussian
    for (int i = 0; i < d*d; ++i) G[i] = rand_gauss();

    // Simple Gram-Schmidt orthogonalization (O(d^3), called once at init)
    for (int i = 0; i < d; ++i) {
        for (int j = 0; j < d; ++j) tq->Pi[i*d + j] = G[i*d + j];
        for (int k = 0; k < i; ++k) {
            float dot = 0.0f;
            for (int j = 0; j < d; ++j) dot += tq->Pi[k*d + j] * tq->Pi[i*d + j];
            for (int j = 0; j < d; ++j) tq->Pi[i*d + j] -= dot * tq->Pi[k*d + j];
        }
        float norm = 0.0f;
        for (int j = 0; j < d; ++j) norm += tq->Pi[i*d + j] * tq->Pi[i*d + j];
        norm = 1.0f / sqrtf(norm);
        for (int j = 0; j < d; ++j) tq->Pi[i*d + j] *= norm;
    }

    // QJL projection matrix S (i.i.d. N(0,1))
    for (int i = 0; i < d*d; ++i) tq->S[i] = rand_gauss();

    free(G);
}

void turboquant_free(TurboQuant *tq) {
    if (tq->Pi) free(tq->Pi);
    if (tq->S)  free(tq->S);
    if (tq->codebook_mse) {
        for (int i = 0; i <= tq->b; ++i) free(tq->codebook_mse[i]);
        free(tq->codebook_mse);
    }
    if (tq->cb_sizes) free(tq->cb_sizes);
}

// ------------------------------------------------------------------
// MSE quantizer (Algorithm 1)
static void quant_mse(const TurboQuant *tq, int bits, const float *y, uint32_t *idx) {
    const float *cb = tq->codebook_mse[bits];
    int k = tq->cb_sizes[bits];
    for (int j = 0; j < tq->d; ++j) {
        float v = y[j];
        int best = 0;
        float best_dist = fabsf(v - cb[0]);
        for (int i = 1; i < k; ++i) {
            float dist = fabsf(v - cb[i]);
            if (dist < best_dist) { best_dist = dist; best = i; }
        }
        idx[j] = (uint32_t)best;
    }
}

static void dequant_mse(const TurboQuant *tq, int bits, const uint32_t *idx, float *y) {
    const float *cb = tq->codebook_mse[bits];
    for (int j = 0; j < tq->d; ++j) y[j] = cb[idx[j]];
}

// ------------------------------------------------------------------
// Main Qprod (Algorithm 2)
void turboquant_quant_prod(const TurboQuant *tq, const float *x,
                           uint32_t *idx, int8_t *qjl_signs, float *residual_norm) {
    float *y = (float*)alloca(tq->d * sizeof(float));
    float *r = (float*)alloca(tq->d * sizeof(float));

    // 1. random rotation (or identity)
    if (tq->use_rotation) {
        for (int i = 0; i < tq->d; ++i) {
            y[i] = 0.0f;
            for (int j = 0; j < tq->d; ++j) y[i] += tq->Pi[i*tq->d + j] * x[j];
        }
    } else {
        memcpy(y, x, tq->d * sizeof(float));
    }

    // 2. MSE quant with b-1 bits
    quant_mse(tq, tq->b - 1, y, idx);

    // 3. residual r = y - dequant_mse
    dequant_mse(tq, tq->b - 1, idx, r);
    for (int j = 0; j < tq->d; ++j) r[j] = y[j] - r[j];

    // 4. residual norm
    *residual_norm = 0.0f;
    for (int j = 0; j < tq->d; ++j) *residual_norm += r[j] * r[j];
    *residual_norm = sqrtf(*residual_norm);

    // 5. 1-bit QJL on residual
    for (int i = 0; i < tq->d; ++i) {
        float dot = 0.0f;
        for (int j = 0; j < tq->d; ++j) dot += tq->S[i*tq->d + j] * r[j];
        qjl_signs[i] = (dot >= 0.0f) ? 1 : -1;
    }
}

void turboquant_dequant_prod(const TurboQuant *tq,
                             const uint32_t *idx, const int8_t *qjl_signs,
                             float residual_norm, float *out) {
    float *xmse = (float*)alloca(tq->d * sizeof(float));
    float *xqjl = (float*)alloca(tq->d * sizeof(float));

    dequant_mse(tq, tq->b - 1, idx, xmse);

    // QJL dequant
    const float scale = (sqrtf((float)M_PI / 2.0f) / (float)tq->d) * residual_norm;
    for (int i = 0; i < tq->d; ++i) {
        float dot = 0.0f;
        for (int j = 0; j < tq->d; ++j) dot += tq->S[j*tq->d + i] * (float)qjl_signs[j]; // transpose access
        xqjl[i] = scale * dot;
    }

    // sum
    for (int j = 0; j < tq->d; ++j) out[j] = xmse[j] + xqjl[j];

    // back-rotate if needed
    if (tq->use_rotation) {
        float tmp[tq->d];
        memcpy(tmp, out, tq->d * sizeof(float));
        for (int i = 0; i < tq->d; ++i) {
            out[i] = 0.0f;
            for (int j = 0; j < tq->d; ++j) out[i] += tq->Pi[j*tq->d + i] * tmp[j]; // Π^T
        }
    }
}

// MSE-only wrappers (optional)
void turboquant_quant_mse(const TurboQuant *tq, const float *x, uint32_t *idx) {
    float *y = (float*)alloca(tq->d * sizeof(float));
    if (tq->use_rotation) {
        for (int i = 0; i < tq->d; ++i) {
            y[i] = 0; for (int j = 0; j < tq->d; ++j) y[i] += tq->Pi[i*tq->d + j] * x[j];
        }
    } else memcpy(y, x, tq->d * sizeof(float));
    quant_mse(tq, tq->b, y, idx);
}

void turboquant_dequant_mse(const TurboQuant *tq, const uint32_t *idx, float *out) {
    dequant_mse(tq, tq->b, idx, out);
    if (tq->use_rotation) {
        float tmp[tq->d];
        memcpy(tmp, out, tq->d * sizeof(float));
        for (int i = 0; i < tq->d; ++i) {
            out[i] = 0; for (int j = 0; j < tq->d; ++j) out[i] += tq->Pi[j*tq->d + i] * tmp[j];
        }
    }
}