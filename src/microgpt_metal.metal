/*
 * microgpt_metal.metal - Metal compute shaders for MicroGPT-C
 *
 * Copyright (c) 2026 Ajay Soni (ajay.soni@enjector.com), Enjector Software Ltd.
 * MIT License — see LICENSE file for details.
 *
 * Three kernels for GPU-accelerated matmul operations:
 *   1. lin_fwd_kernel:    y = W @ x     (matrix-vector product)
 *   2. lin_bwd_dx_kernel: dx += W^T @ dy (transposed mat-vec, accumulated)
 *   3. lin_bwd_dW_kernel: dW += dy ⊗ x  (rank-1 outer product, accumulated)
 *
 * Metal Shading Language only supports float (32-bit).
 * The Objective-C bridge converts double↔float at the CPU/GPU boundary.
 * This introduces a small precision loss but is acceptable for training
 * where gradients are noisy anyway.
 */

#include <metal_stdlib>
using namespace metal;

/*
 * lin_fwd_kernel - y = W @ x
 *
 *   Each thread computes one output element y[j]:
 *     y[j] = Σ_i  W[j*nin + i] * x[i]
 *
 *   Grid: dispatch nout threads (one per output row).
 */
kernel void lin_fwd_kernel(
    device const float *x     [[buffer(0)]],
    device const float *W     [[buffer(1)]],
    device float       *y     [[buffer(2)]],
    constant uint      &nin   [[buffer(3)]],
    constant uint      &nout  [[buffer(4)]],
    uint                gid   [[thread_position_in_grid]])
{
    if (gid >= nout) return;

    float sum = 0.0f;
    device const float *Wrow = W + gid * nin;
    for (uint i = 0; i < nin; i++) {
        sum += Wrow[i] * x[i];
    }
    y[gid] = sum;
}

/*
 * lin_bwd_dx_kernel - dx += W^T @ dy
 *
 *   Each thread computes one element dx[i]:
 *     dx[i] += Σ_j  dy[j] * W[j*nin + i]
 *
 *   This is the column-wise dot product of W with dy.
 *   Grid: dispatch nin threads.
 */
kernel void lin_bwd_dx_kernel(
    device const float *dy    [[buffer(0)]],
    device const float *W     [[buffer(1)]],
    device float       *dx    [[buffer(2)]],
    constant uint      &nin   [[buffer(3)]],
    constant uint      &nout  [[buffer(4)]],
    uint                gid   [[thread_position_in_grid]])
{
    if (gid >= nin) return;

    float sum = 0.0f;
    for (uint j = 0; j < nout; j++) {
        sum += dy[j] * W[j * nin + gid];
    }
    dx[gid] += sum;  /* accumulate */
}

/*
 * lin_bwd_dW_kernel - dW += dy ⊗ x  (outer product)
 *
 *   Each thread computes one element dW[j*nin + i]:
 *     dW[j*nin + i] += dy[j] * x[i]
 *
 *   Grid: dispatch nout × nin threads (flattened 1D).
 */
kernel void lin_bwd_dW_kernel(
    device const float *dy    [[buffer(0)]],
    device const float *x     [[buffer(1)]],
    device float       *dW    [[buffer(2)]],
    constant uint      &nin   [[buffer(3)]],
    constant uint      &nout  [[buffer(4)]],
    uint                gid   [[thread_position_in_grid]])
{
    uint total = nout * nin;
    if (gid >= total) return;

    uint j = gid / nin;
    uint i = gid % nin;
    dW[gid] += dy[j] * x[i];  /* accumulate */
}
