/*
 * Placeholder snapped EML tree for the SMALL-ANGLE pendulum regime.
 *
 * Status:  PLACEHOLDER — pending offline retraining in the companion
 * eml research repo (~/dev/research/eml/) and re-export via
 * tools/eml_export.py.
 *
 * Intended target (to be recovered offline):
 *
 *     T_small(L, theta_obs) = 2*pi * sqrt(L / g)        (theta-independent)
 *
 * which compiles to depth-≤-4 EML via:
 *
 *     sqrt(L/g) = exp( 0.5 * (log(L) - log(g)) )
 *     T        = (2*pi) * sqrt(L/g)
 *
 * (Depth budget: log(L) and log(g) are depth-3 each per the Sheffer
 * construction in c_eml_logprice.h; the multiplication and the
 * trailing scale-by-2pi can be folded into a depth-4 wrapper. Pre-verify
 * via the trainer's compile pass before final export.)
 *
 * Current placeholder behaviour:
 *
 *   We reuse the existing depth-2 tree from eml_organelle/c_eml_tree.h,
 *   namely:
 *
 *       y = e - log(exp(input_y) - log(input_x))
 *
 *   This tree was trained for the paper's depth-2 target and does NOT
 *   compute the pendulum period. It is used here ONLY to exercise the
 *   pipeline IR + EML evaluator end-to-end. The hybrid demo's
 *   "in-domain accuracy" T1 will therefore reflect the *scaffold*, not
 *   the (yet-to-be-trained) pendulum tree.
 *
 *   The DEMO_USE_REFERENCE_PHYSICS flag (set to 1 by default in main.c)
 *   bypasses the placeholder tree and computes the closed-form pendulum
 *   period directly via math.h, so that T2/T4/T5/T6/T7 can be measured
 *   meaningfully today. When the offline-trained tree drops in, flip
 *   that flag to 0 and the pipeline routes through the EML evaluator
 *   only.
 *
 *   Once the offline-trained tree is exported here, this file's tree
 *   constants will be replaced and the placeholder caveat above will be
 *   removed from the docstring.
 */

#ifndef MICROGPT_EML_TREE_SMALLANGLE_PLACEHOLDER_H
#define MICROGPT_EML_TREE_SMALLANGLE_PLACEHOLDER_H

#include "microgpt_eml.h"

/* Audit-trail metadata: the sympy expression the EML node *will* decode
 * to once the real tree is exported. Used by the audit-trail printer
 * (T6) so the demo's audit output is testable today against the locked
 * pre-registration. */
#define EML_SMALLANGLE_SYMPY     "2*pi*sqrt(L/g)"
#define EML_SMALLANGLE_PYTHON    "2*math.pi*math.sqrt(L/9.81)"

/* === Placeholder tree constants (verbatim from eml_organelle/c_eml_tree.h). === */
static const unsigned char eml_smallangle_leaves[4] = { 0, 2, 2, 1 };
static const unsigned char eml_smallangle_gates_left[3] = { 1, 0, 1 };
static const unsigned char eml_smallangle_gates_right[3] = { 0, 0, 0 };

static const EmlTree eml_smallangle = {
  .depth = 2,
  .n_leaves = 4,
  .n_internal = 3,
  .leaves = eml_smallangle_leaves,
  .gates_left = eml_smallangle_gates_left,
  .gates_right = eml_smallangle_gates_right,
};

#endif /* MICROGPT_EML_TREE_SMALLANGLE_PLACEHOLDER_H */
