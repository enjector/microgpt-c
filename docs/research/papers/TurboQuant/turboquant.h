#ifndef TURBOQUANT_H
#define TURBOQUANT_H

#include <stdint.h>
#include <stdbool.h>

typedef struct {
    int d;                  // vector dimension (head_dim)
    int b;                  // target bit-width for Qprod (2.5–4 recommended)
    float *Pi;              // d×d random rotation matrix (row-major), NULL if disabled
    float *S;               // d×d random projection matrix for QJL (row-major)
    float **codebook_mse;   // codebooks[b][2^b] – precomputed Lloyd-Max centroids (scaled)
    int   *cb_sizes;        // number of centroids per bit-width (2^b)
    bool   use_rotation;
} TurboQuant;

// Init once at model load (call before any quant)
void turboquant_init(TurboQuant *tq, int d, int b, bool use_rotation);

// Free
void turboquant_free(TurboQuant *tq);

// Recommended: inner-product optimal (unbiased, perfect for MSA cosine routing)
void turboquant_quant_prod(const TurboQuant *tq,
                           const float *x,           // input vector (d)
                           uint32_t *idx,            // output: MSE indices (b-1 bits)
                           int8_t *qjl_signs,        // output: QJL signs (-1/+1)
                           float *residual_norm);    // output: ||r||₂

void turboquant_dequant_prod(const TurboQuant *tq,
                             const uint32_t *idx,
                             const int8_t *qjl_signs,
                             float residual_norm,
                             float *out);             // reconstructed vector

// Optional: pure MSE quantizer (if you only need reconstruction)
void turboquant_quant_mse(const TurboQuant *tq, const float *x, uint32_t *idx);
void turboquant_dequant_mse(const TurboQuant *tq, const uint32_t *idx, float *out);

#endif