#include "microgpt_turboquant.h"
#include "test.h"
#include <math.h>

#define EPSILON 0.05f 

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

enx_test(test_tq_quantize_dequantize_mse) {
    int n_embd = 96;
    TurboQuant tq;
    turboquant_init(&tq, n_embd, 3, false);
    
    float *x = (float *)calloc(n_embd, sizeof(float));
    for (int i = 0; i < n_embd; i++) {
        x[i] = (float)i * 0.01f;
    }
    
    uint32_t *centroid_idx = (uint32_t *)calloc(n_embd, sizeof(uint32_t));
    
    turboquant_quant_mse(&tq, x, centroid_idx);
    
    float *out = (float *)calloc(n_embd, sizeof(float));
    turboquant_dequant_mse(&tq, centroid_idx, out);
    
    // Check cosine similarity
    float dot = 0.0f, normx = 0.0f, normout = 0.0f;
    for (int i = 0; i < n_embd; i++) {
        dot += x[i] * out[i];
        normx += x[i] * x[i];
        normout += out[i] * out[i];
    }
    float cosine = dot / (sqrtf(normx) * sqrtf(normout) + 1e-8f);
    
    enx_assert_true(cosine > 0.85f); // Should maintain general structural direction
    
    free(x);
    free(out);
    free(centroid_idx);
    turboquant_free(&tq);
}

int main(void) {
    enx_test_case_t tq_cases[] = {
        enx_test_case(test_tq_initialization),
        enx_test_case(test_tq_quantize_dequantize_mse),
        enx_test_case_end()
    };
    test_suite suites[] = {
        {"TurboQuant Memory Compression Primitives", tq_cases},
        {NULL, NULL}
    };
    return test_suite_run(suites) ? EXIT_SUCCESS : EXIT_FAILURE;
}
