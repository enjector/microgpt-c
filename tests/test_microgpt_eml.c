/*
 * MicroGPT-C — test_microgpt_eml: unit tests for the EML organelle evaluator.
 *
 * Validates the snapped-tree evaluator (microgpt_eml.{h,c}) against
 * hand-built trees for known elementary forms:
 *   - exp(x)   = eml(x, 1)                                    [depth 1]
 *   - ln(x)    = eml(1, eml(eml(1, x), 1))                    [depth 3]
 *   - eml(x,y) = eml(x, y)                                    [depth 1]
 *   - eml_d2   = eml(eml(y, x), 1) = e - log(exp(y) - log(x)) [depth 2]
 *
 * Each test builds the tree as static const arrays and asserts the
 * evaluator matches the canonical math.h form to within float64 epsilon.
 */

#include "microgpt_eml.h"
#include "test.h"

#include <math.h>

/* Tolerance: float32 in the default build, double in MICROGPT_USE_FLOAT=OFF.
 * The ln-recovery tree has wasted intermediate eml(x, x) = exp(x) − log(x)
 * subtrees that are computed but gated out; for x near 2.7 these grow to ~1e6
 * before being discarded, eating ~5 ULPs of relative precision in float32. */
#ifdef MICROGPT_USE_FLOAT
#define EML_TEST_TOL 1e-5
#else
#define EML_TEST_TOL 1e-12
#endif

/* --- exp(x) = eml(x, 1) ---------------------------------------------- */
/* depth=1: 2 leaves, 1 internal.
 *   leaf 0 = x, leaf 1 = constant 1
 *   internal 0: gates_left=use child(=x), gates_right=use child(=1)
 *   eml(x, 1) = exp(x) - log(1) = exp(x).
 */
static const unsigned char eml_exp_leaves[2]      = { EML_LEAF_X, EML_LEAF_CONST };
static const unsigned char eml_exp_gates_left[1]  = { EML_GATE_CHILD };
static const unsigned char eml_exp_gates_right[1] = { EML_GATE_CHILD };
static const EmlTree eml_exp_tree = {
    .depth = 1, .n_leaves = 2, .n_internal = 1,
    .leaves = eml_exp_leaves,
    .gates_left = eml_exp_gates_left,
    .gates_right = eml_exp_gates_right,
};

enx_test(test_eml_exp_x) {
    double xs[] = { -1.0, 0.0, 0.5, 1.0, 1.7 };
    for (size_t i = 0; i < sizeof(xs)/sizeof(xs[0]); ++i) {
        scalar_t pred = eml_eval(&eml_exp_tree, (scalar_t)xs[i], (scalar_t)0.0);
        scalar_t want = (scalar_t)exp(xs[i]);
        enx_assert_true(fabs((double)(pred - want)) < EML_TEST_TOL);
    }
}

/* --- ln(x) = eml(1, eml(eml(1, x), 1)) -------------------------------- */
/* depth=3, 8 leaves, 7 internals.  Embedding (per parent research §10.3.1):
 *   We need bottom-internal_2 = eml(1, x).  Its leaves are slots 4 and 5:
 *     leaf 4 = 1, leaf 5 = x.
 *   mid-internal_1 = eml(bottom_2, 1):
 *     left=child (uses bottom_2 = eml(1,x))
 *     right=const1
 *   root = eml(1, mid_1):
 *     left=const1
 *     right=child (uses mid_1)
 *   Other nodes are wasted; we set their leaves to safe positive values
 *   so the forward pass doesn't NaN.
 *
 * Internal index ordering (bottom-up):
 *   0..3: bottom internals  (parents of leaves)
 *   4: mid internal 0  (parent of bottom 0,1)
 *   5: mid internal 1  (parent of bottom 2,3)
 *   6: root            (parent of mid 0, mid 1)
 */
static const unsigned char eml_ln_leaves[8] = {
    EML_LEAF_X, EML_LEAF_X,   /* bottom 0 (wasted, but x>0 keeps log finite) */
    EML_LEAF_X, EML_LEAF_X,   /* bottom 1 (wasted) */
    EML_LEAF_CONST, EML_LEAF_X, /* bottom 2 = eml(1, x) <-- matters */
    EML_LEAF_X, EML_LEAF_X,   /* bottom 3 (wasted) */
};
static const unsigned char eml_ln_gates_left[7]  = {
    EML_GATE_CHILD, EML_GATE_CHILD, EML_GATE_CHILD, EML_GATE_CHILD,
    EML_GATE_CHILD,           /* mid 0: wasted */
    EML_GATE_CHILD,           /* mid 1: left = bottom 2 */
    EML_GATE_CONST,           /* root: left = constant 1 */
};
static const unsigned char eml_ln_gates_right[7] = {
    EML_GATE_CHILD, EML_GATE_CHILD, EML_GATE_CHILD, EML_GATE_CHILD,
    EML_GATE_CHILD,           /* mid 0: wasted */
    EML_GATE_CONST,           /* mid 1: right = constant 1 */
    EML_GATE_CHILD,           /* root: right = mid 1 */
};
static const EmlTree eml_ln_tree = {
    .depth = 3, .n_leaves = 8, .n_internal = 7,
    .leaves = eml_ln_leaves,
    .gates_left = eml_ln_gates_left,
    .gates_right = eml_ln_gates_right,
};

enx_test(test_eml_ln_x) {
    double xs[] = { 0.5, 1.0, 1.5, 2.7, 5.0 };
    for (size_t i = 0; i < sizeof(xs)/sizeof(xs[0]); ++i) {
        scalar_t pred = eml_eval(&eml_ln_tree, (scalar_t)xs[i], (scalar_t)0.0);
        scalar_t want = (scalar_t)log(xs[i]);
        enx_assert_true(fabs((double)(pred - want)) < EML_TEST_TOL);
    }
}

/* --- eml(x, y) = exp(x) - log(y), trivial depth-1 base case ----------- */
static const unsigned char eml_xy_leaves[2]      = { EML_LEAF_X, EML_LEAF_Y };
static const unsigned char eml_xy_gates_left[1]  = { EML_GATE_CHILD };
static const unsigned char eml_xy_gates_right[1] = { EML_GATE_CHILD };
static const EmlTree eml_xy_tree = {
    .depth = 1, .n_leaves = 2, .n_internal = 1,
    .leaves = eml_xy_leaves,
    .gates_left = eml_xy_gates_left,
    .gates_right = eml_xy_gates_right,
};

enx_test(test_eml_xy_basic) {
    double pts[][2] = { {1.0, 1.0}, {0.5, 2.0}, {2.0, 0.7}, {-0.3, 1.1} };
    for (size_t i = 0; i < sizeof(pts)/sizeof(pts[0]); ++i) {
        scalar_t pred = eml_eval(&eml_xy_tree, (scalar_t)pts[i][0],
                                 (scalar_t)pts[i][1]);
        scalar_t want = (scalar_t)(exp(pts[i][0]) - log(pts[i][1]));
        enx_assert_true(fabs((double)(pred - want)) < EML_TEST_TOL);
    }
}

/* --- eml_d2: e - log(exp(y) - log(x)) -------------------------------- */
/* Recovered tree from the parent research, leaves [1, y, y, x],
 * gates [(const, child), (child, child), (const, child)].
 * Decoded: eml(eml(1, y), eml(y, x)) routes via root.left=const, so the
 * answer is eml(1, eml(y, x)) = e - log(exp(y) - log(x)).
 */
static const unsigned char eml_d2_leaves[4] = {
    EML_LEAF_CONST, EML_LEAF_Y, EML_LEAF_Y, EML_LEAF_X
};
static const unsigned char eml_d2_gates_left[3]  = {
    EML_GATE_CONST,   /* bottom 0 left: use const 1 (so left input = 1) */
    EML_GATE_CHILD,   /* bottom 1 left: use child (= y leaf) */
    EML_GATE_CONST,   /* root left: use const 1 */
};
static const unsigned char eml_d2_gates_right[3] = {
    EML_GATE_CHILD,   /* bottom 0 right: use child (= y leaf) */
    EML_GATE_CHILD,   /* bottom 1 right: use child (= x leaf) */
    EML_GATE_CHILD,   /* root right: use child (= bottom 1's eml(y,x)) */
};
static const EmlTree eml_d2_tree = {
    .depth = 2, .n_leaves = 4, .n_internal = 3,
    .leaves = eml_d2_leaves,
    .gates_left = eml_d2_gates_left,
    .gates_right = eml_d2_gates_right,
};

enx_test(test_eml_d2_recovered) {
    double pts[][2] = { {1.5, 1.5}, {2.0, 2.0}, {1.2, 2.7}, {2.9, 1.1} };
    for (size_t i = 0; i < sizeof(pts)/sizeof(pts[0]); ++i) {
        double x = pts[i][0], y = pts[i][1];
        scalar_t pred = eml_eval(&eml_d2_tree, (scalar_t)x, (scalar_t)y);
        scalar_t want = (scalar_t)(M_E - log(exp(y) - log(x)));
        enx_assert_true(fabs((double)(pred - want)) < EML_TEST_TOL);
    }
}

/* --- API surface tests ------------------------------------------------ */

enx_test(test_eml_eval_batch_matches_scalar) {
    static const double xs_d[5] = { 0.5, 1.0, 1.5, 2.0, 2.5 };
    static const double ys_d[5] = { 1.1, 1.3, 1.7, 2.2, 2.8 };
    scalar_t xs[5], ys[5];
    for (int i = 0; i < 5; ++i) { xs[i] = (scalar_t)xs_d[i]; ys[i] = (scalar_t)ys_d[i]; }
    scalar_t out[5];
    eml_eval_batch(&eml_xy_tree, xs, ys, out, 5);
    for (int i = 0; i < 5; ++i) {
        scalar_t single = eml_eval(&eml_xy_tree, xs[i], ys[i]);
        enx_assert_true(fabs((double)(out[i] - single)) < EML_TEST_TOL);
    }
}

enx_test(test_eml_mse_zero_when_correct) {
    /* Use the eml(x,y) tree with targets = exp(x) - log(y); MSE must be 0. */
    static const double xs_d[6] = { 0.5, 1.0, 1.5, 2.0, 2.5, 0.3 };
    static const double ys_d[6] = { 1.1, 1.3, 1.7, 2.2, 2.8, 1.9 };
    scalar_t xs[6], ys[6], targets[6];
    for (int i = 0; i < 6; ++i) {
        xs[i] = (scalar_t)xs_d[i]; ys[i] = (scalar_t)ys_d[i];
        targets[i] = (scalar_t)(exp(xs_d[i]) - log(ys_d[i]));
    }
    scalar_t mse = eml_mse(&eml_xy_tree, xs, ys, targets, 6);
    /* "Zero to machine precision":
     *   - In float64 (double scalar_t) MSE is ≤ 1e-25 because the snapped tree
     *     IS the target and the only error is round-off in the last ULPs.
     *   - In float32 (float scalar_t) we round-trip through 32-bit rep, so
     *     MSE is on the order of (float32 epsilon * |output|)^2 ~ 1e-13. */
    const double tol = (sizeof(scalar_t) == 4) ? 1e-12 : 1e-25;
    enx_assert_true((double)mse < tol);
}

/* --- main --------------------------------------------------------------- */

int main(void) {
    enx_test_case_t eml_cases[] = {
        enx_test_case(test_eml_exp_x),
        enx_test_case(test_eml_ln_x),
        enx_test_case(test_eml_xy_basic),
        enx_test_case(test_eml_d2_recovered),
        enx_test_case(test_eml_eval_batch_matches_scalar),
        enx_test_case(test_eml_mse_zero_when_correct),
        enx_test_case_end(),
    };
    test_suite suites[] = {
        {"EML Organelle Evaluator", eml_cases},
        {NULL, NULL},
    };
    return test_suite_run(suites) ? EXIT_SUCCESS : EXIT_FAILURE;
}
