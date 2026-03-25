#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../../src/turboquant.h"

int main() {
    printf("--- TurboQuant Standalone Benchmark ---\n");

    int head_dim = 128;
    int bits = 4;
    bool use_rotation = false;

    TurboQuant tq;
    printf("Initializing TurboQuant (dim=%d, bits=%d, rotation=%d)...\n", head_dim, bits, use_rotation);
    turboquant_init(&tq, head_dim, bits, use_rotation);

    // Create a random high-dimensional vector
    float *x = (float*)malloc(head_dim * sizeof(float));
    float norm = 0.0f;
    for (int i = 0; i < head_dim; i++) {
        // pseudo-random N(0,1) for test
        float u1 = (float)rand() / RAND_MAX;
        float u2 = (float)rand() / RAND_MAX;
        x[i] = sqrtf(-2.0f * logf(u1 + 1e-10f)) * cosf(2.0f * (float)M_PI * u2);
        norm += x[i] * x[i];
    }
    float scale = 1.0f / sqrtf(norm);
    for (int i = 0; i < head_dim; i++) {
        x[i] *= scale;
    }

    uint32_t *idx = (uint32_t*)malloc(head_dim * sizeof(uint32_t));
    int8_t *qjl = (int8_t*)malloc(head_dim * sizeof(int8_t));
    float rnorm;
    float *out = (float*)malloc(head_dim * sizeof(float));

    printf("Quantizing with Qprod...\n");
    turboquant_quant_prod(&tq, x, idx, qjl, &rnorm);

    printf("Dequantizing with Qprod...\n");
    turboquant_dequant_prod(&tq, idx, qjl, rnorm, out);

    printf("rnorm: %f\n", rnorm);
    printf("First 5 elements:\n");
    for (int i = 0; i < 5; i++) {
        printf("x[%d]=%f, out[%d]=%f, idx[%d]=%d, qjl[%d]=%d\n", i, x[i], i, out[i], i, idx[i], i, qjl[i]);
    }

    float mse = 0.0f;
    float dot = 0.0f;
    float norm_x = 0.0f;
    float norm_out = 0.0f;
    for (int i = 0; i < head_dim; i++) {
        float diff = x[i] - out[i];
        mse += diff * diff;
        dot += x[i] * out[i];
        norm_x += x[i] * x[i];
        norm_out += out[i] * out[i];
    }
    mse /= head_dim;
    float cos_sim = dot / (sqrtf(norm_x) * sqrtf(norm_out) + 1e-8f);

    printf("MSE: %f\n", mse);
    printf("Cosine Similarity: %f\n", cos_sim);
    
    turboquant_free(&tq);
    free(x);
    free(idx);
    free(qjl);
    free(out);

    if (cos_sim > 0.90f) {
        printf("SUCCESS: High cosine similarity preserved!\n");
        return 0;
    } else {
        printf("WARNING: Cosine similarity lower than expected.\n");
        return 1;
    }
}
