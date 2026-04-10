#ifndef ROTORQUANT_H
#define ROTORQUANT_H

#include <stdint.h>
#include <stdbool.h>

typedef enum {
    RQ_MODE_PLANAR = 0, // 2D Givens rotation
    RQ_MODE_ISO = 1     // 4D Quaternion rotation
} RotorQuantMode;

typedef struct {
    int d;                  // vector dimension (head_dim)
    int b;                  // target bit-width for Qprod (2.5–4 recommended)
    RotorQuantMode mode;    // underlying rotation mathematical mode
    float *rotations;       // rotation parameters (angles for Planar, quaternions for Iso), NULL if disabled
    float *S;               // dxd random projection matrix for QJL (row-major)
    float **codebook_mse;   // codebooks[b][2^b] – precomputed Lloyd-Max centroids (scaled)
    int   *cb_sizes;        // number of centroids per bit-width (2^b)
    bool   use_rotation;
} RotorQuant;

// Init once at model load (call before any quant)
void rotorquant_init(RotorQuant *rq, int d, int b, RotorQuantMode mode, bool use_rotation);

// Free
void rotorquant_free(RotorQuant *rq);

// Recommended: inner-product optimal (unbiased, perfect for MSA cosine routing)
void rotorquant_quant_prod(const RotorQuant *rq,
                           const float *x,           // input vector (d)
                           uint32_t *idx,            // output: MSE indices (b-1 bits)
                           int8_t *qjl_signs,        // output: QJL signs (-1/+1)
                           float *residual_norm);    // output: ||r||₂

void rotorquant_dequant_prod(const RotorQuant *rq,
                             const uint32_t *idx,
                             const int8_t *qjl_signs,
                             float residual_norm,
                             float *out);             // reconstructed vector

// Optional: pure MSE quantizer (if you only need reconstruction)
void rotorquant_quant_mse(const RotorQuant *rq, const float *x, uint32_t *idx);
void rotorquant_dequant_mse(const RotorQuant *rq, const uint32_t *idx, float *out);

#endif
