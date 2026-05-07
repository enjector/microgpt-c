/*
 * Hand-coded EML tree for the depth-2 elementary form
 *     y = e − log( exp(input_y) − log(input_x) ).
 *
 * Same tree as the parent eml_organelle demo's c_eml_tree.h, mirrored here
 * so the boundary demo is self-contained.  This is the trainer's
 * `eml_depth2` synthetic target — the recovered tree on 16/16 seeds at
 * every noise level σ ∈ {0.001, 0.01, 0.1} per the parent research's §9.1.
 *
 * Tree shape: depth=2, 4 leaves, 3 internals.
 *   Leaves:        (1, y, y, x)
 *   Gates_left:    (const, child, const)
 *   Gates_right:   (child, child, child)
 *   Decoded:       eml(1, eml(y, x)) = e − log(exp(y) − log(x))
 *
 * Quant context: a synthetic 2-input elementary transform.  Used in the
 * boundary-map demo as the "depth-2 frontier" case — exactly at the edge
 * of the trainer's reliable random-init regime.
 */

#ifndef MICROGPT_EML_TREE_EML_D2_ELEMENTARY_H
#define MICROGPT_EML_TREE_EML_D2_ELEMENTARY_H

#include "microgpt_eml.h"

static const unsigned char eml_d2_elementary_leaves[4]      = { EML_LEAF_CONST, EML_LEAF_Y, EML_LEAF_Y, EML_LEAF_X };
static const unsigned char eml_d2_elementary_gates_left[3]  = { EML_GATE_CONST, EML_GATE_CHILD, EML_GATE_CONST };
static const unsigned char eml_d2_elementary_gates_right[3] = { EML_GATE_CHILD, EML_GATE_CHILD, EML_GATE_CHILD };

static const EmlTree eml_d2_elementary = {
  .depth = 2,
  .n_leaves = 4,
  .n_internal = 3,
  .leaves = eml_d2_elementary_leaves,
  .gates_left = eml_d2_elementary_gates_left,
  .gates_right = eml_d2_elementary_gates_right,
};

#endif /* MICROGPT_EML_TREE_EML_D2_ELEMENTARY_H */
