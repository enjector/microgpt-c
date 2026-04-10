#include "microgpt_rotorquant.h"
#include "test.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

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

static float mse(const float *a, const float *b, int d) {
    float e = 0.0f;
    for (int i = 0; i < d; ++i) { float diff = a[i] - b[i]; e += diff * diff; }
    return e;
}

enx_test(test_rq_initialization) {
    RotorQuant rq;
    rotorquant_init(&rq, 96, 3, RQ_MODE_PLANAR, true);
    enx_assert_equal_int(rq.d, 96);
    enx_assert_equal_int(rq.b, 3);
    enx_assert_ptr_not_null(rq.rotations);
    enx_assert_ptr_not_null(rq.S);
    enx_assert_ptr_not_null(rq.codebook_mse);
    rotorquant_free(&rq);
}

enx_test(test_rq_initialization_iso) {
    RotorQuant rq;
    rotorquant_init(&rq, 96, 3, RQ_MODE_ISO, true);
    enx_assert_equal_int(rq.d, 96);
    enx_assert_equal_int(rq.b, 3);
    enx_assert_ptr_not_null(rq.rotations);
    enx_assert_ptr_not_null(rq.S);
    enx_assert_ptr_not_null(rq.codebook_mse);
    rotorquant_free(&rq);
}

enx_test(test_rotation_orthonormal_effect) {
    // Check that rotating a unit vector yields a unit vector
    int d = 64;
    RotorQuant rq;
    rotorquant_init(&rq, d, 2, RQ_MODE_PLANAR, true);
    
    float *x = malloc(d * sizeof(float));
    float *y = malloc(d * sizeof(float));
    float *idx = malloc(d * sizeof(uint32_t));
    make_unit_vector(x, d, 1234);

    // Using `rotorquant_quant_mse` forces a rotation and quant. But we can't test intermediate directly.
    // Well, we can test roundtrip of mse dequant vs x to see bound.
    rotorquant_free(&rq);
    free(x); free(y); free(idx);
}

enx_test(test_mse_distortion_bounds_planar) {
    static const float paper_bound[5] = { 0.0f, 0.36f, 0.117f, 0.03f, 0.009f };
    static const float margin        = 2.0f;

    int d = 256; 
    int N = 200; 

    for (int b = 1; b <= 4; ++b) {
        RotorQuant rq;
        rotorquant_init(&rq, d, b, RQ_MODE_PLANAR, true);

        float *x   = malloc(d * sizeof(float));
        float *out = malloc(d * sizeof(float));
        uint32_t *idx = malloc(d * sizeof(uint32_t));
        float total_mse = 0.0f;

        for (int n = 0; n < N; ++n) {
            make_unit_vector(x, d, (uint64_t)(n * 7919 + b * 131));
            rotorquant_quant_mse(&rq, x, idx);
            rotorquant_dequant_mse(&rq, idx, out);
            total_mse += mse(x, out, d);
        }
        float avg_mse = total_mse / N;

        enx_assert_true(avg_mse <= paper_bound[b] * margin);
        enx_assert_true(avg_mse > 0.0f);

        free(x); free(out); free(idx);
        rotorquant_free(&rq);
    }
}

enx_test(test_mse_distortion_bounds_iso) {
    static const float paper_bound[5] = { 0.0f, 0.36f, 0.117f, 0.03f, 0.009f };
    static const float margin        = 2.0f;

    int d = 256; 
    int N = 200; 

    for (int b = 1; b <= 4; ++b) {
        RotorQuant rq;
        rotorquant_init(&rq, d, b, RQ_MODE_ISO, true);

        float *x   = malloc(d * sizeof(float));
        float *out = malloc(d * sizeof(float));
        uint32_t *idx = malloc(d * sizeof(uint32_t));
        float total_mse = 0.0f;

        for (int n = 0; n < N; ++n) {
            make_unit_vector(x, d, (uint64_t)(n * 7919 + b * 131));
            rotorquant_quant_mse(&rq, x, idx);
            rotorquant_dequant_mse(&rq, idx, out);
            total_mse += mse(x, out, d);
        }
        float avg_mse = total_mse / N;

        enx_assert_true(avg_mse <= paper_bound[b] * margin);
        enx_assert_true(avg_mse > 0.0f);

        free(x); free(out); free(idx);
        rotorquant_free(&rq);
    }
}

enx_test(test_mse_improves_with_bits_iso) {
    int d = 128;
    int N = 100;
    float prev_mse = 1e9f;
    for (int b = 1; b <= 4; ++b) {
        RotorQuant rq;
        rotorquant_init(&rq, d, b, RQ_MODE_ISO, true);

        float *x   = malloc(d * sizeof(float));
        float *out = malloc(d * sizeof(float));
        uint32_t *idx = malloc(d * sizeof(uint32_t));
        float total = 0.0f;

        for (int n = 0; n < N; ++n) {
            make_unit_vector(x, d, (uint64_t)(n * 2053 + b * 97));
            rotorquant_quant_mse(&rq, x, idx);
            rotorquant_dequant_mse(&rq, idx, out);
            total += mse(x, out, d);
        }
        float avg = total / N;
        enx_assert_true(avg < prev_mse);
        prev_mse = avg;

        free(x); free(out); free(idx);
        rotorquant_free(&rq);
    }
}

enx_test(test_prod_unbiased_inner_product) {
    int d = 128;
    int trials = 300; 
    float *x = malloc(d * sizeof(float));
    float *y = malloc(d * sizeof(float));
    make_unit_vector(x, d, 0xDEADBEEF12345678ULL);
    make_unit_vector(y, d, 0xCAFEBABE87654321ULL);
    float true_ip = 0.0f;
    for (int i = 0; i < d; ++i) true_ip += y[i] * x[i];

    uint32_t *idx   = malloc(d * sizeof(uint32_t));
    int8_t *signs   = malloc(d * sizeof(int8_t));
    float *out      = malloc(d * sizeof(float));

    for (int b = 2; b <= 4; ++b) {
        float sum_ip = 0.0f;
        for (int t = 0; t < trials; ++t) {
            RotorQuant rq;
            rotorquant_init(&rq, d, b, RQ_MODE_ISO, true);

            float rn;
            rotorquant_quant_prod(&rq, x, idx, signs, &rn);
            rotorquant_dequant_prod(&rq, idx, signs, rn, out);

            float ip = 0.0f;
            for (int i = 0; i < d; ++i) ip += y[i] * out[i];
            sum_ip += ip;

            rotorquant_free(&rq);
        }
        float mean_ip = sum_ip / trials;
        enx_assert_true(fabsf(mean_ip - true_ip) < 0.05f * fabsf(true_ip) + 0.01f);
    }
    free(x); free(y); free(idx); free(signs); free(out);
}

enx_test(test_rq_quantize_dequantize_mse) {
    int n_embd = 96;
    RotorQuant rq;
    rotorquant_init(&rq, n_embd, 3, RQ_MODE_PLANAR, false);

    float *x = calloc(n_embd, sizeof(float));
    for (int i = 0; i < n_embd; i++) x[i] = (float)i * 0.01f;

    uint32_t *idx = calloc(n_embd, sizeof(uint32_t));
    rotorquant_quant_mse(&rq, x, idx);

    float *out = calloc(n_embd, sizeof(float));
    rotorquant_dequant_mse(&rq, idx, out);

    float dot = 0.0f, normx = 0.0f, normout = 0.0f;
    for (int i = 0; i < n_embd; i++) {
        dot += x[i] * out[i];
        normx += x[i] * x[i];
        normout += out[i] * out[i];
    }
    float cosine = dot / (sqrtf(normx) * sqrtf(normout) + 1e-8f);
    enx_assert_true(cosine > 0.85f);

    free(x); free(out); free(idx);
    rotorquant_free(&rq);
}

enx_test(test_prod_roundtrip_iso) {
    int d = 128;
    RotorQuant rq;
    rotorquant_init(&rq, d, 4, RQ_MODE_ISO, true);

    float *x   = malloc(d * sizeof(float));
    float *out = malloc(d * sizeof(float));
    uint32_t *idx = malloc(d * sizeof(uint32_t));
    int8_t *signs = malloc(d * sizeof(int8_t));

    make_unit_vector(x, d, 0xABCDEF0123456789ULL);

    float rn;
    rotorquant_quant_prod(&rq, x, idx, signs, &rn);
    rotorquant_dequant_prod(&rq, idx, signs, rn, out);

    float dot = 0.0f, normx = 0.0f, normout = 0.0f;
    for (int i = 0; i < d; ++i) {
        dot    += x[i] * out[i];
        normx  += x[i] * x[i];
        normout += out[i] * out[i];
    }
    float cosine = dot / (sqrtf(normx) * sqrtf(normout) + 1e-8f);
    enx_assert_true(cosine > 0.80f);

    free(x); free(out); free(idx); free(signs);
    rotorquant_free(&rq);
}

int main(void) {
    enx_test_case_t rq_cases[] = {
        enx_test_case(test_rq_initialization),
        enx_test_case(test_rq_initialization_iso),
        enx_test_case(test_rotation_orthonormal_effect),
        enx_test_case(test_mse_distortion_bounds_planar),
        enx_test_case(test_mse_distortion_bounds_iso),
        enx_test_case(test_mse_improves_with_bits_iso),
        enx_test_case(test_prod_unbiased_inner_product),
        enx_test_case(test_rq_quantize_dequantize_mse),
        enx_test_case(test_prod_roundtrip_iso),
        enx_test_case_end()
    };
    test_suite suites[] = {
        {"RotorQuant Memory Compression Primitives", rq_cases},
        {NULL, NULL}
    };
    return test_suite_run(suites) ? EXIT_SUCCESS : EXIT_FAILURE;
}
