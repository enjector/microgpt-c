/*
 * microgpt_metal.h - Metal GPU acceleration for MicroGPT-C
 *
 * Copyright (c) 2026 Ajay Soni (ajay.soni@enjector.com), Enjector Software Ltd.
 * MIT License — see LICENSE file for details.
 *
 * Provides GPU-accelerated matmul primitives via Apple Metal compute shaders.
 * On Apple Silicon, uses unified memory (zero-copy shared buffers).
 *
 * Note: Metal only supports float32. The bridge converts double↔float
 * at the CPU/GPU boundary. This is acceptable for training where gradients
 * are noisy, and the matmul precision isn't the bottleneck.
 *
 * Usage: compile with -DMICROGPT_METAL and link Metal.framework + Foundation.
 */

#ifndef MICROGPT_METAL_H
#define MICROGPT_METAL_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Initialise Metal device, command queue, and compile compute pipelines.
 * Returns 0 on success, -1 on failure (no GPU, shader compile error, etc.) */
int metal_init(void);

/* Release all Metal resources. Safe to call multiple times. */
void metal_cleanup(void);

/* Returns 1 if Metal was successfully initialized, 0 otherwise.
 * Used by lin_fwd/lin_bwd to decide whether to dispatch to GPU or fall back. */
int metal_available(void);

/* GPU-accelerated dense linear forward: y = W @ x
 *   x:    input vector  [nin]   (double, converted to float for GPU)
 *   W:    weight matrix [nout × nin] (row-major, converted to float)
 *   nin:  input dimension
 *   nout: output dimension
 *   y:    output vector [nout]  (float result converted back to double) */
void metal_lin_fwd(const double *x, const double *W, size_t nin, size_t nout,
                   double *y);

/* GPU-accelerated linear backward: dx = W^T@dy, dW += dy⊗x
 *   x:    saved input from forward [nin]
 *   W:    weight matrix [nout × nin]
 *   dy:   upstream gradient [nout]
 *   nin:  input dimension
 *   nout: output dimension
 *   dx:   gradient w.r.t. input [nin] (accumulated, may be NULL)
 *   dW:   gradient w.r.t. weights [nout × nin] (accumulated, may be NULL) */
void metal_lin_bwd(const double *x, const double *W, const double *dy,
                   size_t nin, size_t nout, double *dx, double *dW);

#ifdef __cplusplus
}
#endif

#endif /* MICROGPT_METAL_H */
