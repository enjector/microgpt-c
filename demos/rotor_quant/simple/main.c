/*
 * RotorQuant: KV-Cache Compression Demo
 *
 * Simulates compressing a KV-cache (N key/value vectors of dimension d) at
 * increasing bit-widths and prints a quality table showing:
 *
 *   - Memory reduction vs full FP32 baseline
 *   - MSE distortion (reconstruction error)
 *   - Inner-product distortion (query-key dot-product error)
 *   - Whether Q_prod is unbiased (expected signed error near 0)
 *
 * Compare the measured distortion to the paper's Theorem 1/2 bounds:
 *   D_mse  ≈ 0.36 / 0.117 / 0.030 / 0.009  for b = 1/2/3/4
 *   D_prod ≈ 1.57/d, 0.56/d, 0.18/d, 0.047/d
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <string.h>
#include "microgpt_rotorquant.h"

#define HEAD_DIM    128     /* typical transformer head dimension */
#define N_TOKENS    1024    /* KV-cache entries (context length) */

// Simple deterministic vector generator (unit-sphere normalised).
static void make_unit_vec(float *v, int d, uint64_t seed) {
    uint64_t s = seed;
    float n = 0.0f;
    for (int i = 0; i < d; i++) {
        s = s * 6364136223846793005ULL + 1442695040888963407ULL;
        v[i] = (float)(int64_t)s / (float)(1LL << 32);
        n += v[i] * v[i];
    }
    n = 1.0f / sqrtf(n);
    for (int i = 0; i < d; i++) v[i] *= n;
}

int main(void) {
    int d = HEAD_DIM, N = N_TOKENS;

    printf("=================================================================\n");
    printf(" RotorQuant  —  KV-Cache Compression Demo\n");
    printf(" d = %d  |  N = %d tokens  |  FP32 baseline = %.1f KB\n",
           d, N, (float)(N * d * 4) / 1024.0f);
    printf("=================================================================\n\n");

    // Paper bounds for reference
    static const float p_mse[5]  = {0, 0.360f, 0.117f, 0.030f, 0.009f};
    static const float p_prod[5] = {0, 1.57f,  0.56f,  0.18f,  0.047f};

    printf("  %-5s  %-8s  %-10s  %-10s  %-10s  %-10s  %-10s\n",
           "bits", "mem(KB)", "D_mse", "D_mse*", "D_prod*d", "D_prod*d*", "IP_bias");
    printf("               (Q_mse)    (paper)   (Q_prod)   (paper)   (should≈0)\n");
    printf("  %s\n",
           "-------------------------------------------------------------------");

    float *x   = malloc(d * sizeof(float));
    float *q   = malloc(d * sizeof(float));   // query vector
    float *out = malloc(d * sizeof(float));
    uint32_t *idx  = malloc(d * sizeof(uint32_t));
    int8_t   *sgn  = malloc(d * sizeof(int8_t));

    // Fixed query vector (represents an attention query)
    make_unit_vec(q, d, 0xCAFEBABE00ABCDEFULL);

    for (int b = 1; b <= 4; b++) {
        RotorQuant tq;
        rotorquant_init(&tq, d, b, RQ_MODE_PLANAR, true);

        double sum_mse = 0.0, sum_prod = 0.0, sum_bias = 0.0;

        for (int t = 0; t < N; t++) {
            make_unit_vec(x, d, (uint64_t)(t * 7919 + b * 131 + 1));

            // MSE distortion: use Q_mse (Theorem 1)
            rotorquant_quant_mse(&tq, x, idx);
            rotorquant_dequant_mse(&tq, idx, out);
            float mse_v = 0.0f;
            for (int i = 0; i < d; i++) { float diff=x[i]-out[i]; mse_v+=diff*diff; }
            sum_mse += mse_v;

            // Inner-product distortion + bias: use Q_prod (Theorem 2)
            float rn;
            rotorquant_quant_prod(&tq, x, idx, sgn, &rn);
            rotorquant_dequant_prod(&tq, idx, sgn, rn, out);
            float ip_true = 0.0f, ip_est = 0.0f;
            for (int i = 0; i < d; i++) { ip_true+=q[i]*x[i]; ip_est+=q[i]*out[i]; }
            float ip_err = ip_true - ip_est;
            sum_prod += ip_err * ip_err;
            sum_bias += ip_est - ip_true;   // signed: unbiased means this → 0
        }

        float avg_mse  = (float)(sum_mse  / N);
        float avg_prod = (float)(sum_prod / N) * d;
        float avg_bias = (float)(sum_bias / N);

        // Memory: (b-1) bits for MSE indices + 1 sign bit per coord + 1 float norm
        float mem_kb = (float)N * ((float)(b * d) / 8.0f + 4.0f) / 1024.0f;

        printf("  b=%-3d  %-8.1f  %-10.5f  %-10.5f  %-10.5f  %-10.5f  %+.5f\n",
               b, mem_kb, avg_mse, p_mse[b], avg_prod, p_prod[b], avg_bias);

        rotorquant_free(&tq);
    }

    // FP32 baseline memory
    float fp32_kb = (float)(N * d * sizeof(float)) / 1024.0f;
    printf("\n  FP32 baseline:  %.1f KB\n", fp32_kb);
    printf("  b=4 Q_prod:    %.1f KB  (%.1fx reduction)\n",
           (float)N * ((float)(4*d)/8.0f + 4.0f) / 1024.0f,
           fp32_kb / ((float)N * ((float)(4*d)/8.0f + 4.0f) / 1024.0f));

    printf("\n  (*) paper bounds from Theorems 1 & 2 (arXiv 2504.19874)\n");
    printf("  IP_bias for Q_prod should be ~0 (unbiased inner-product estimator).\n");
    printf("\n=================================================================\n");

    free(x); free(q); free(out); free(idx); free(sgn);
    return 0;
}
