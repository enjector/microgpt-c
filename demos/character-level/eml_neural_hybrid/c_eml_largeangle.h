/*
 * Placeholder snapped EML tree for the LARGE-ANGLE pendulum regime.
 *
 * Status:  PLACEHOLDER — pending offline training + export.
 *
 * Intended target:
 *
 *     T_large(L, theta_obs) ~ 2*pi*sqrt(L/g) * (1 + theta_obs^2 / 16)
 *
 * which is the first non-trivial term of the standard series expansion
 * of the elliptic-integral large-angle correction (Bernoulli, 1764);
 * the trainer's compile pass should fit it inside depth-4.
 *
 * Current placeholder behaviour:
 *   Reuses the depth-2 tree as a stand-in. See c_eml_smallangle.h for
 *   the same caveat. DEMO_USE_REFERENCE_PHYSICS bypasses this tree and
 *   computes the closed form via math.h until the offline export drops.
 */

#ifndef MICROGPT_EML_TREE_LARGEANGLE_PLACEHOLDER_H
#define MICROGPT_EML_TREE_LARGEANGLE_PLACEHOLDER_H

#include "microgpt_eml.h"

#define EML_LARGEANGLE_SYMPY  "2*pi*sqrt(L/g) * (1 + theta**2/16)"
#define EML_LARGEANGLE_PYTHON "2*math.pi*math.sqrt(L/9.81)*(1+theta**2/16)"

static const unsigned char eml_largeangle_leaves[4] = { 0, 2, 2, 1 };
static const unsigned char eml_largeangle_gates_left[3] = { 1, 0, 1 };
static const unsigned char eml_largeangle_gates_right[3] = { 0, 0, 0 };

static const EmlTree eml_largeangle = {
  .depth = 2,
  .n_leaves = 4,
  .n_internal = 3,
  .leaves = eml_largeangle_leaves,
  .gates_left = eml_largeangle_gates_left,
  .gates_right = eml_largeangle_gates_right,
};

#endif /* MICROGPT_EML_TREE_LARGEANGLE_PLACEHOLDER_H */
