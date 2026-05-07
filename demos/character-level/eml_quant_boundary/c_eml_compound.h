/*
 * Hand-coded EML tree for the continuous-compounding factor exp(x).
 *
 * Canonical Sheffer construction (verified in tests/test_microgpt_eml.c
 * test_eml_exp_x):  exp(x) = eml(x, 1).
 *
 * Tree shape: depth=1, n_leaves=2, n_internal=1.
 *   Leaves:        (x, 1)
 *   Gates_left:    (use child)         -> input = leaf 0 (x)
 *   Gates_right:   (use child)         -> input = leaf 1 (1)
 *   Output:        eml(x, 1) = exp(x) - log(1) = exp(x)
 *
 * Quant context: discount/compounding factor.  Given a single combined
 * input rt = r * t (rate * time), this organelle returns exp(rt).  The
 * multiplication r*t happens in the calling pipeline as the standard
 * *deterministic* Kanban step (microgpt_organelle.h convention) — the EML
 * organelle handles only the elementary transform itself, which is its
 * sweet spot.
 */

#ifndef MICROGPT_EML_TREE_EML_COMPOUND_H
#define MICROGPT_EML_TREE_EML_COMPOUND_H

#include "microgpt_eml.h"

static const unsigned char eml_compound_leaves[2]      = { EML_LEAF_X, EML_LEAF_CONST };
static const unsigned char eml_compound_gates_left[1]  = { EML_GATE_CHILD };
static const unsigned char eml_compound_gates_right[1] = { EML_GATE_CHILD };

static const EmlTree eml_compound = {
  .depth = 1,
  .n_leaves = 2,
  .n_internal = 1,
  .leaves = eml_compound_leaves,
  .gates_left = eml_compound_gates_left,
  .gates_right = eml_compound_gates_right,
};

#endif /* MICROGPT_EML_TREE_EML_COMPOUND_H */
