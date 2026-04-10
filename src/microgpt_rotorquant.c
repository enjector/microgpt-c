#include "microgpt_rotorquant.h"
#include <stdlib.h>
#include <math.h>
#include <string.h>

// ------------------------------------------------------------------
// Simple fixed-seed RNG (zero dep, reproducible)
static uint64_t rng_state = 0x123456789ABCDEF0ULL;

static inline float rand_uniform(void) {
    rng_state = rng_state * 6364136223846793005ULL + 1442695040888963407ULL;
    return (float)(rng_state >> 33) / (float)(1u << 31);
}

static inline float rand_gauss(void) {
    // Box-Muller transform
    float u1 = rand_uniform();
    float u2 = rand_uniform();
    if (u1 <= 1e-7f) u1 = 1e-7f; 
    return sqrtf(-2.0f * logf(u1)) * cosf(2.0f * (float)M_PI * u2);
}

// ------------------------------------------------------------------
// Precomputed Lloyd-Max centroids for standard normal N(0,1)
// (scaled by 1/sqrt(d) at init time)
static const float normal_centroids[5][16] = {
    // b=0 unused
    {0},
    // b=1 (2 centroids)
    {-0.79788456f, +0.79788456f},
    // b=2 (4 centroids)
    {-1.510f, -0.453f, +0.453f, +1.510f},
    // b=3 (8 centroids)
    {-2.1520f, -1.3439f, -0.7560f, -0.2451f, +0.2451f, +0.7560f, +1.3439f, +2.1520f},
    // b=4 (16 centroids)
    {-2.7326f,-2.0690f,-1.6180f,-1.2562f,-0.9423f,-0.6567f,-0.3880f,-0.1284f,
     +0.1284f,+0.3880f,+0.6567f,+0.9423f,+1.2562f,+1.6180f,+2.0690f,+2.7326f}
};

static const int cb_sizes[5] = {0, 2, 4, 8, 16};

// ------------------------------------------------------------------
void rotorquant_init(RotorQuant *rq, int d, int b, RotorQuantMode mode, bool use_rotation) {
    memset(rq, 0, sizeof(*rq));
    rq->d = d;
    rq->b = b;
    rq->mode = mode;
    rq->use_rotation = use_rotation;

    float sigma = 1.0f / sqrtf((float)d);

    rq->codebook_mse = (float**)malloc((b+1) * sizeof(float*));
    rq->cb_sizes = (int*)malloc((b+1) * sizeof(int));
    rq->cb_sizes[0] = 1;
    rq->codebook_mse[0] = (float*)calloc(1, sizeof(float)); 
    for (int bb = 1; bb <= b; ++bb) {
        rq->cb_sizes[bb] = cb_sizes[bb];
        rq->codebook_mse[bb] = (float*)malloc(cb_sizes[bb] * sizeof(float));
        for (int i = 0; i < cb_sizes[bb]; ++i) {
            rq->codebook_mse[bb][i] = normal_centroids[bb][i] * sigma;
        }
    }

    if (use_rotation) {
        if (mode == RQ_MODE_PLANAR) {
            int n_groups = (d + 1) / 2;
            rq->rotations = (float*)malloc(n_groups * 2 * sizeof(float));
            for (int i = 0; i < n_groups; ++i) {
                float angle = rand_uniform() * 2.0f * (float)M_PI;
                rq->rotations[i*2 + 0] = cosf(angle);
                rq->rotations[i*2 + 1] = sinf(angle);
            }
        } else if (mode == RQ_MODE_ISO) {
            int n_groups = (d + 3) / 4;
            rq->rotations = (float*)malloc(n_groups * 4 * sizeof(float));
            for (int i = 0; i < n_groups; ++i) {
                float w = rand_gauss();
                float x = rand_gauss();
                float y = rand_gauss();
                float z = rand_gauss();
                float norm = sqrtf(w*w + x*x + y*y + z*z);
                if (norm < 1e-8f) norm = 1e-8f;
                rq->rotations[i*4 + 0] = w / norm;
                rq->rotations[i*4 + 1] = x / norm;
                rq->rotations[i*4 + 2] = y / norm;
                rq->rotations[i*4 + 3] = z / norm;
            }
        }
    } else {
        rq->rotations = NULL;
    }

    rq->S = (float*)malloc(d * d * sizeof(float));
    for (int i = 0; i < d*d; ++i) rq->S[i] = rand_gauss();
}

void rotorquant_free(RotorQuant *rq) {
    if (rq->rotations) free(rq->rotations);
    if (rq->S) free(rq->S);
    if (rq->codebook_mse) {
        for (int i = 0; i <= rq->b; ++i) free(rq->codebook_mse[i]);
        free(rq->codebook_mse);
    }
    if (rq->cb_sizes) free(rq->cb_sizes);
}

// ------------------------------------------------------------------
// Rotation Operations
static void apply_rotation(const RotorQuant *rq, const float *x, float *y) {
    if (rq->mode == RQ_MODE_PLANAR) {
        int n_groups = (rq->d + 1) / 2;
        for (int i = 0; i < n_groups; ++i) {
            float c = rq->rotations[i*2 + 0];
            float s = rq->rotations[i*2 + 1];
            float v0 = x[i*2 + 0];
            float v1 = (i*2 + 1 < rq->d) ? x[i*2 + 1] : 0.0f;
            y[i*2 + 0] = c * v0 - s * v1;
            if (i*2 + 1 < rq->d) y[i*2 + 1] = s * v0 + c * v1;
        }
    } else if (rq->mode == RQ_MODE_ISO) {
        int n_groups = (rq->d + 3) / 4;
        for (int i = 0; i < n_groups; ++i) {
            float aw = rq->rotations[i*4 + 0];
            float ax = rq->rotations[i*4 + 1];
            float ay = rq->rotations[i*4 + 2];
            float az = rq->rotations[i*4 + 3];
            
            float bw = x[i*4 + 0];
            float bx = (i*4 + 1 < rq->d) ? x[i*4 + 1] : 0.0f;
            float by = (i*4 + 2 < rq->d) ? x[i*4 + 2] : 0.0f;
            float bz = (i*4 + 3 < rq->d) ? x[i*4 + 3] : 0.0f;

            y[i*4 + 0] = aw*bw - ax*bx - ay*by - az*bz;
            if (i*4 + 1 < rq->d) y[i*4 + 1] = aw*bx + ax*bw + ay*bz - az*by;
            if (i*4 + 2 < rq->d) y[i*4 + 2] = aw*by - ax*bz + ay*bw + az*bx;
            if (i*4 + 3 < rq->d) y[i*4 + 3] = aw*bz + ax*by - ay*bx + az*bw;
        }
    }
}

static void apply_inverse_rotation(const RotorQuant *rq, const float *x, float *y) {
    if (rq->mode == RQ_MODE_PLANAR) {
        int n_groups = (rq->d + 1) / 2;
        for (int i = 0; i < n_groups; ++i) {
            float c = rq->rotations[i*2 + 0];
            float s = rq->rotations[i*2 + 1];
            float v0 = x[i*2 + 0];
            float v1 = (i*2 + 1 < rq->d) ? x[i*2 + 1] : 0.0f;
            y[i*2 + 0] = c * v0 + s * v1;
            if (i*2 + 1 < rq->d) y[i*2 + 1] = -s * v0 + c * v1;
        }
    } else if (rq->mode == RQ_MODE_ISO) {
        int n_groups = (rq->d + 3) / 4;
        for (int i = 0; i < n_groups; ++i) {
            float aw = rq->rotations[i*4 + 0];
            float ax = -rq->rotations[i*4 + 1]; // conjugate
            float ay = -rq->rotations[i*4 + 2];
            float az = -rq->rotations[i*4 + 3];
            
            float bw = x[i*4 + 0];
            float bx = (i*4 + 1 < rq->d) ? x[i*4 + 1] : 0.0f;
            float by = (i*4 + 2 < rq->d) ? x[i*4 + 2] : 0.0f;
            float bz = (i*4 + 3 < rq->d) ? x[i*4 + 3] : 0.0f;

            y[i*4 + 0] = aw*bw - ax*bx - ay*by - az*bz;
            if (i*4 + 1 < rq->d) y[i*4 + 1] = aw*bx + ax*bw + ay*bz - az*by;
            if (i*4 + 2 < rq->d) y[i*4 + 2] = aw*by - ax*bz + ay*bw + az*bx;
            if (i*4 + 3 < rq->d) y[i*4 + 3] = aw*bz + ax*by - ay*bx + az*bw;
        }
    }
}

// ------------------------------------------------------------------
// MSE quantizer
static void quant_mse(const RotorQuant *rq, int bits, const float *y, uint32_t *idx) {
    const float *cb = rq->codebook_mse[bits];
    int k = rq->cb_sizes[bits];
    for (int j = 0; j < rq->d; ++j) {
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

static void dequant_mse(const RotorQuant *rq, int bits, const uint32_t *idx, float *y) {
    const float *cb = rq->codebook_mse[bits];
    for (int j = 0; j < rq->d; ++j) y[j] = cb[idx[j]];
}

// ------------------------------------------------------------------
// Main Qprod
#ifndef alloca
#ifdef __GNUC__
#define alloca __builtin_alloca
#else
#include <alloca.h>
#endif
#endif

void rotorquant_quant_prod(const RotorQuant *rq, const float *x,
                           uint32_t *idx, int8_t *qjl_signs, float *residual_norm) {
    float *y = (float*)alloca(rq->d * sizeof(float));
    float *r = (float*)alloca(rq->d * sizeof(float));

    if (rq->use_rotation) {
        apply_rotation(rq, x, y);
    } else {
        memcpy(y, x, rq->d * sizeof(float));
    }

    quant_mse(rq, rq->b - 1, y, idx);

    dequant_mse(rq, rq->b - 1, idx, r);
    for (int j = 0; j < rq->d; ++j) r[j] = y[j] - r[j];

    *residual_norm = 0.0f;
    for (int j = 0; j < rq->d; ++j) *residual_norm += r[j] * r[j];
    *residual_norm = sqrtf(*residual_norm);

    for (int i = 0; i < rq->d; ++i) {
        float dot = 0.0f;
        for (int j = 0; j < rq->d; ++j) dot += rq->S[i*rq->d + j] * r[j];
        qjl_signs[i] = (dot >= 0.0f) ? 1 : -1;
    }
}

void rotorquant_dequant_prod(const RotorQuant *rq,
                             const uint32_t *idx, const int8_t *qjl_signs,
                             float residual_norm, float *out) {
    float *xmse = (float*)alloca(rq->d * sizeof(float));
    float *xqjl = (float*)alloca(rq->d * sizeof(float));

    dequant_mse(rq, rq->b - 1, idx, xmse);

    // QJL dequant
    const float scale = (sqrtf((float)M_PI / 2.0f) / (float)rq->d) * residual_norm;
    for (int i = 0; i < rq->d; ++i) {
        float dot = 0.0f;
        for (int j = 0; j < rq->d; ++j) dot += rq->S[j*rq->d + i] * (float)qjl_signs[j]; 
        xqjl[i] = scale * dot;
    }

    // sum
    for (int j = 0; j < rq->d; ++j) out[j] = xmse[j] + xqjl[j];

    // back-rotate if needed
    if (rq->use_rotation) {
        float *tmp = (float*)alloca(rq->d * sizeof(float));
        memcpy(tmp, out, rq->d * sizeof(float));
        apply_inverse_rotation(rq, tmp, out);
    }
}

void rotorquant_quant_mse(const RotorQuant *rq, const float *x, uint32_t *idx) {
    float *y = (float*)alloca(rq->d * sizeof(float));
    if (rq->use_rotation) apply_rotation(rq, x, y);
    else memcpy(y, x, rq->d * sizeof(float));
    quant_mse(rq, rq->b, y, idx);
}

void rotorquant_dequant_mse(const RotorQuant *rq, const uint32_t *idx, float *out) {
    dequant_mse(rq, rq->b, idx, out);
    if (rq->use_rotation) {
        float *tmp = (float*)alloca(rq->d * sizeof(float));
        memcpy(tmp, out, rq->d * sizeof(float));
        apply_inverse_rotation(rq, tmp, out);
    }
}
