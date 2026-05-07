/*
 * Hand-coded EML tree for log(x), used as the log-price transform.
 *
 * Canonical Sheffer construction (verified in tests/test_microgpt_eml.c
 * test_eml_ln_x):
 *   log(x) = eml( 1, eml( eml(1, x), 1 ) )                     [depth 3]
 *
 * Embedding into the level-3 master tree (8 leaves, 7 internals):
 *   bottom-internal_2 must compute eml(1, x).
 *     Its leaves are slots 4 and 5: leaf 4 = 1, leaf 5 = x.
 *   mid-internal_1 = eml(bottom_2, 1):
 *     gates_left=child  (consume bottom_2 = eml(1,x))
 *     gates_right=const (consume the constant 1)
 *   root = eml(1, mid_1):
 *     gates_left=const  (output's first arg is 1)
 *     gates_right=child (output's second arg = mid_1 = eml(eml(1,x), 1))
 *   ⇒ root = eml( 1, eml( eml(1, x), 1 ) ) = log(x).
 *
 * Wasted subtrees (bottom 0, 1, 3 and mid 0): set leaves to "x" so that
 * eml(x, x) = exp(x) - log(x) is finite for x > 0.  These subtrees are
 * computed but their outputs are gated out by the upstream const choices.
 *
 * Internal indices (bottom-up, matching tree_prototype_torch_v16_final
 * convention):  0..3 = bottom internals, 4 = mid 0, 5 = mid 1, 6 = root.
 *
 * Quant context: log-price transform.  Given a positive scalar p, returns
 * log(p).  Used in the hybrid pipeline demo to convert raw prices into
 * log-prices for downstream return calculations.
 */

#ifndef MICROGPT_EML_TREE_EML_LOGPRICE_H
#define MICROGPT_EML_TREE_EML_LOGPRICE_H

#include "microgpt_eml.h"

static const unsigned char eml_logprice_leaves[8] = {
  EML_LEAF_X, EML_LEAF_X,    /* bottom 0: wasted, eml(x,x) finite for x>0 */
  EML_LEAF_X, EML_LEAF_X,    /* bottom 1: wasted */
  EML_LEAF_CONST, EML_LEAF_X,/* bottom 2: eml(1, x) — load-bearing */
  EML_LEAF_X, EML_LEAF_X,    /* bottom 3: wasted */
};
static const unsigned char eml_logprice_gates_left[7] = {
  EML_GATE_CHILD, EML_GATE_CHILD, EML_GATE_CHILD, EML_GATE_CHILD,
  EML_GATE_CHILD,            /* mid 0: wasted */
  EML_GATE_CHILD,            /* mid 1: left = bottom 2 */
  EML_GATE_CONST,            /* root: left = constant 1 */
};
static const unsigned char eml_logprice_gates_right[7] = {
  EML_GATE_CHILD, EML_GATE_CHILD, EML_GATE_CHILD, EML_GATE_CHILD,
  EML_GATE_CHILD,            /* mid 0: wasted */
  EML_GATE_CONST,            /* mid 1: right = constant 1 */
  EML_GATE_CHILD,            /* root: right = mid 1 */
};

static const EmlTree eml_logprice = {
  .depth = 3,
  .n_leaves = 8,
  .n_internal = 7,
  .leaves = eml_logprice_leaves,
  .gates_left = eml_logprice_gates_left,
  .gates_right = eml_logprice_gates_right,
};

#endif /* MICROGPT_EML_TREE_EML_LOGPRICE_H */
