#include "microgpt_turboquant.h"
#include "test.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

// ---------------------------------------------------------------------------
// Helpers

// Deterministic test-vector generator: fills x with pseudo-random floats,
// then normalises to unit-sphere so the paper's distortion bounds apply.
static void make_unit_vector(float *x, int d, uint64_t seed) {
    uint64_t s = seed;
    float norm = 0.0f;
    for (int i = 0; i < d; ++i) {
        s = s * 6364136223846793005ULL + 1442695040888963407ULL;
        float v = (float)(int64_t)s * (1.0f / (float)(1LL << 32));
        x[i] = v;
        norm += v * v;
    }
    norm = 1.0f / sqrtf(norm);
    for (int i = 0; i < d; ++i) x[i] *= norm;
}

// Mean-squared error between two vectors.
static float mse(const float *a, const float *b, int d) {
    float e = 0.0f;
    for (int i = 0; i < d; ++i) { float diff = a[i] - b[i]; e += diff * diff; }
    return e;
}

// ---------------------------------------------------------------------------
// Basic init / struct sanity

enx_test(test_tq_initialization) {
    TurboQuant tq;
    turboquant_init(&tq, 96, 3, true);

    enx_assert_equal_int(tq.d, 96);
    enx_assert_equal_int(tq.b, 3);
    enx_assert_ptr_not_null(tq.Pi);
    enx_assert_ptr_not_null(tq.S);
    enx_assert_ptr_not_null(tq.codebook_mse);

    turboquant_free(&tq);
}

// Π must be orthonormal: rows should have unit norm and mutual dot ≈ 0.
enx_test(test_rotation_matrix_orthonormal) {
    int d = 32;
    TurboQuant tq;
    turboquant_init(&tq, d, 2, true);

    // Check a sample of rows for unit norm
    for (int i = 0; i < d; i += 4) {
        float n = 0.0f;
        for (int j = 0; j < d; ++j) n += tq.Pi[i*d+j] * tq.Pi[i*d+j];
        enx_assert_true(fabsf(n - 1.0f) < 0.001f);
    }
    // Check two rows are orthogonal
    float dot = 0.0f;
    for (int j = 0; j < d; ++j) dot += tq.Pi[0*d+j] * tq.Pi[1*d+j];
    enx_assert_true(fabsf(dot) < 0.01f);

    turboquant_free(&tq);
}

// ---------------------------------------------------------------------------
// Codebook ordering: centroids must be strictly ascending per bit-width.

enx_test(test_codebook_ordering) {
    int d = 64;
    TurboQuant tq;
    turboquant_init(&tq, d, 4, false);

    for (int b = 1; b <= 4; ++b) {
        int k = tq.cb_sizes[b];
        for (int i = 1; i < k; ++i) {
            enx_assert_true(tq.codebook_mse[b][i] > tq.codebook_mse[b][i-1]);
        }
    }
    turboquant_free(&tq);
}

// ---------------------------------------------------------------------------
// MSE distortion bounds from the paper (Theorem 1):
//   b=1 → ≈0.36,  b=2 → ≈0.117,  b=3 → ≈0.03,  b=4 → ≈0.009  (per unit-norm x)
// We allow 2× margin so the test is robust to small-d finite-sample noise.

enx_test(test_mse_distortion_bounds) {
    // Paper bounds (Table from Theorem 1) with a 2× tolerance factor.
    static const float paper_bound[5] = { 0.0f, 0.36f, 0.117f, 0.03f, 0.009f };
    static const float margin        = 2.0f;

    int d = 256; // large-ish d so the Beta→Gaussian limit holds well
    int N = 200; // vectors to average over

    for (int b = 1; b <= 4; ++b) {
        TurboQuant tq;
        turboquant_init(&tq, d, b, true);

        float *x   = malloc(d * sizeof(float));
        float *out = malloc(d * sizeof(float));
        uint32_t *idx = malloc(d * sizeof(uint32_t));
        float total_mse = 0.0f;

        for (int n = 0; n < N; ++n) {
            make_unit_vector(x, d, (uint64_t)(n * 7919 + b * 131));
            turboquant_quant_mse(&tq, x, idx);
            turboquant_dequant_mse(&tq, idx, out);
            total_mse += mse(x, out, d);
        }
        float avg_mse = total_mse / N;

        // avg_mse should be ≤ paper_bound[b] * margin
        enx_assert_true(avg_mse <= paper_bound[b] * margin);
        // Also must be > 0 (lossy, not trivial)
        enx_assert_true(avg_mse > 0.0f);

        free(x); free(out); free(idx);
        turboquant_free(&tq);
    }
}

// ---------------------------------------------------------------------------
// Each bit-width should achieve strictly better MSE than the previous.

enx_test(test_mse_improves_with_bits) {
    int d = 128;
    int N = 100;

    float prev_mse = 1e9f;
    for (int b = 1; b <= 4; ++b) {
        TurboQuant tq;
        turboquant_init(&tq, d, b, true);

        float *x   = malloc(d * sizeof(float));
        float *out = malloc(d * sizeof(float));
        uint32_t *idx = malloc(d * sizeof(uint32_t));
        float total = 0.0f;

        for (int n = 0; n < N; ++n) {
            make_unit_vector(x, d, (uint64_t)(n * 2053 + b * 97));
            turboquant_quant_mse(&tq, x, idx);
            turboquant_dequant_mse(&tq, idx, out);
            total += mse(x, out, d);
        }
        float avg = total / N;
        enx_assert_true(avg < prev_mse);
        prev_mse = avg;

        free(x); free(out); free(idx);
        turboquant_free(&tq);
    }
}

// ---------------------------------------------------------------------------
// Q_prod must give an UNBIASED inner-product estimator (Theorem 2).
// E[<y, dequant_prod(quant_prod(x))>] = <y, x>  for any y.
// We test this by averaging over many Monte-Carlo runs (different S matrices).

enx_test(test_prod_unbiased_inner_product) {
    int d = 128;
    int trials = 300; // independent TurboQuant instances (different S,Π)

    // Fixed test vectors
    float *x = malloc(d * sizeof(float));
    float *y = malloc(d * sizeof(float));
    make_unit_vector(x, d, 0xDEADBEEF12345678ULL);
    make_unit_vector(y, d, 0xCAFEBABE87654321ULL);

    // True inner product
    float true_ip = 0.0f;
    for (int i = 0; i < d; ++i) true_ip += y[i] * x[i];

    uint32_t *idx   = malloc(d * sizeof(uint32_t));
    int8_t *signs   = malloc(d * sizeof(int8_t));
    float *out      = malloc(d * sizeof(float));

    for (int b = 2; b <= 4; ++b) {
        float sum_ip = 0.0f;
        for (int t = 0; t < trials; ++t) {
            TurboQuant tq;
            turboquant_init(&tq, d, b, true);

            float rn;
            turboquant_quant_prod(&tq, x, idx, signs, &rn);
            turboquant_dequant_prod(&tq, idx, signs, rn, out);

            float ip = 0.0f;
            for (int i = 0; i < d; ++i) ip += y[i] * out[i];
            sum_ip += ip;

            turboquant_free(&tq);
        }
        float mean_ip = sum_ip / trials;
        // Mean should be within 5% of the true inner product
        enx_assert_true(fabsf(mean_ip - true_ip) < 0.05f * fabsf(true_ip) + 0.01f);
    }

    free(x); free(y); free(idx); free(signs); free(out);
}

// ---------------------------------------------------------------------------
// MSE cosine similarity smoke test (legacy, kept for regression).

enx_test(test_tq_quantize_dequantize_mse) {
    int n_embd = 96;
    TurboQuant tq;
    turboquant_init(&tq, n_embd, 3, false);

    float *x = calloc(n_embd, sizeof(float));
    for (int i = 0; i < n_embd; i++) x[i] = (float)i * 0.01f;

    uint32_t *idx = calloc(n_embd, sizeof(uint32_t));
    turboquant_quant_mse(&tq, x, idx);

    float *out = calloc(n_embd, sizeof(float));
    turboquant_dequant_mse(&tq, idx, out);

    float dot = 0.0f, normx = 0.0f, normout = 0.0f;
    for (int i = 0; i < n_embd; i++) {
        dot += x[i] * out[i];
        normx += x[i] * x[i];
        normout += out[i] * out[i];
    }
    float cosine = dot / (sqrtf(normx) * sqrtf(normout) + 1e-8f);
    enx_assert_true(cosine > 0.85f);

    free(x); free(out); free(idx);
    turboquant_free(&tq);
}

// ---------------------------------------------------------------------------
// Prod quantizer round-trip: reconstruction should be close to original.

enx_test(test_prod_roundtrip) {
    int d = 128;
    TurboQuant tq;
    turboquant_init(&tq, d, 4, true);

    float *x   = malloc(d * sizeof(float));
    float *out = malloc(d * sizeof(float));
    uint32_t *idx = malloc(d * sizeof(uint32_t));
    int8_t *signs = malloc(d * sizeof(int8_t));

    make_unit_vector(x, d, 0xABCDEF0123456789ULL);

    float rn;
    turboquant_quant_prod(&tq, x, idx, signs, &rn);
    turboquant_dequant_prod(&tq, idx, signs, rn, out);

    // Cosine similarity should be high
    float dot = 0.0f, normx = 0.0f, normout = 0.0f;
    for (int i = 0; i < d; ++i) {
        dot    += x[i] * out[i];
        normx  += x[i] * x[i];
        normout += out[i] * out[i];
    }
    float cosine = dot / (sqrtf(normx) * sqrtf(normout) + 1e-8f);
    enx_assert_true(cosine > 0.80f);

    free(x); free(out); free(idx); free(signs);
    turboquant_free(&tq);
}

// ---------------------------------------------------------------------------

int main(void) {
    enx_test_case_t tq_cases[] = {
        enx_test_case(test_tq_initialization),
        enx_test_case(test_rotation_matrix_orthonormal),
        enx_test_case(test_codebook_ordering),
        enx_test_case(test_mse_distortion_bounds),
        enx_test_case(test_mse_improves_with_bits),
        enx_test_case(test_prod_unbiased_inner_product),
        enx_test_case(test_tq_quantize_dequantize_mse),
        enx_test_case(test_prod_roundtrip),
        enx_test_case_end()
    };
    test_suite suites[] = {
        {"TurboQuant Memory Compression Primitives", tq_cases},
        {NULL, NULL}
    };
    return test_suite_run(suites) ? EXIT_SUCCESS : EXIT_FAILURE;
}
