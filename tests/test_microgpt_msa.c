#include "microgpt_msa.h"
#include "test.h"
#include <math.h>

#define EPSILON 0.001f

enx_test(test_msa_pool_initialization) {
    MsaPool *pool = msa_pool_create(10, 4, 96);
    enx_assert_ptr_not_null(pool);
    enx_assert_equal_int(pool->capacity, 10);
    enx_assert_equal_int(pool->length, 0);
    enx_assert_equal_int(pool->n_layer, 4);
    enx_assert_equal_int(pool->n_embd, 96);
    enx_assert_ptr_not_null(pool->keys);
    msa_pool_free(pool);
}

enx_test(test_msa_pool_chunking) {
    int n_layer = 1;
    int n_embd = 2;
    int chunk_size = 4;
    MsaPool *pool = msa_pool_create(10, n_layer, n_embd);
    
    scalar_t **keys = (scalar_t **)malloc(n_layer * sizeof(scalar_t *));
    scalar_t **values = (scalar_t **)malloc(n_layer * sizeof(scalar_t *));
    keys[0] = (scalar_t *)malloc(chunk_size * n_embd * sizeof(scalar_t));
    values[0] = (scalar_t *)malloc(chunk_size * n_embd * sizeof(scalar_t));
    
    /* 
     * Set predictable values:
     * keys[0] layer, token 0: {1.0, 2.0} -> token 1: {3.0, 4.0} -> token 2: {5.0, 6.0} -> token 3: {7.0, 8.0}
     * Mean of embd 0: (1+3+5+7)/4 = 4.0
     * Mean of embd 1: (2+4+6+8)/4 = 5.0
     */
    for(int pos=0; pos<4; pos++) {
        keys[0][pos*2 + 0] = (pos*2) + 1.0f;
        keys[0][pos*2 + 1] = (pos*2) + 2.0f;
        values[0][pos*2 + 0] = ((pos*2) + 1.0f) * 2.0f;
        values[0][pos*2 + 1] = ((pos*2) + 2.0f) * 2.0f;
    }
    
    msa_pool_chunk(pool, keys, values, chunk_size);
    
    enx_assert_equal_int(pool->length, 1);
    
    // offset = chunk_idx * (n_layer * n_embd) + l * n_embd + d
    // Chunk 0, Layer 0
    float diff_k0 = fabs(pool->keys[0 * (n_layer * n_embd) + 0 * n_embd + 0] - 4.0f);
    float diff_k1 = fabs(pool->keys[0 * (n_layer * n_embd) + 0 * n_embd + 1] - 5.0f);
    enx_assert_true(diff_k0 < EPSILON);
    enx_assert_true(diff_k1 < EPSILON);
    
    float diff_v0 = fabs(pool->values[0 * (n_layer * n_embd) + 0 * n_embd + 0] - 8.0f);
    float diff_v1 = fabs(pool->values[0 * (n_layer * n_embd) + 0 * n_embd + 1] - 10.0f);
    enx_assert_true(diff_v0 < EPSILON);
    enx_assert_true(diff_v1 < EPSILON);
    
    free(keys[0]); free(values[0]);
    free(keys); free(values);
    msa_pool_free(pool);
}

enx_test(test_msa_routing_precision) {
    int n_layer = 1;
    int n_embd = 4;
    MsaPool *pool = msa_pool_create(10, n_layer, n_embd);
    
    /* Create 3 chunks manually */
    pool->length = 3;
    
    /* Orthogonal Vectors */
    pool->keys[0 * (n_layer * n_embd) + 0 * n_embd + 0] = 1.0f;
    pool->keys[0 * (n_layer * n_embd) + 0 * n_embd + 1] = 0.0f;
    pool->keys[0 * (n_layer * n_embd) + 0 * n_embd + 2] = 0.0f;
    pool->keys[0 * (n_layer * n_embd) + 0 * n_embd + 3] = 0.0f;
    
    pool->keys[1 * (n_layer * n_embd) + 0 * n_embd + 0] = 0.0f;
    pool->keys[1 * (n_layer * n_embd) + 0 * n_embd + 1] = 1.0f;
    pool->keys[1 * (n_layer * n_embd) + 0 * n_embd + 2] = 0.0f;
    pool->keys[1 * (n_layer * n_embd) + 0 * n_embd + 3] = 0.0f;
    
    pool->keys[2 * (n_layer * n_embd) + 0 * n_embd + 0] = 0.0f;
    pool->keys[2 * (n_layer * n_embd) + 0 * n_embd + 1] = 0.0f;
    pool->keys[2 * (n_layer * n_embd) + 0 * n_embd + 2] = 1.0f;
    pool->keys[2 * (n_layer * n_embd) + 0 * n_embd + 3] = 0.0f;
    
    scalar_t **query_keys = (scalar_t **)malloc(n_layer * sizeof(scalar_t *));
    query_keys[0] = (scalar_t *)malloc(n_embd * sizeof(scalar_t));
    
    /* Query chunk perfectly matches Chunk 1 */
    query_keys[0][0] = 0.0f; query_keys[0][1] = 1.0f; query_keys[0][2] = 0.0f; query_keys[0][3] = 0.0f;
    
    int route = msa_route_top_1(pool, query_keys);
    
    enx_assert_equal_int(route, 1);
    
    free(query_keys[0]); free(query_keys);
    msa_pool_free(pool);
}

int main(void) {
    enx_test_case_t msa_cases[] = {
        enx_test_case(test_msa_pool_initialization),
        enx_test_case(test_msa_pool_chunking),
        enx_test_case(test_msa_routing_precision),
        enx_test_case_end()
    };
    test_suite suites[] = {
        {"MSA Memory Sparse Attention Primitives", msa_cases},
        {NULL, NULL}
    };
    return test_suite_run(suites) ? EXIT_SUCCESS : EXIT_FAILURE;
}
