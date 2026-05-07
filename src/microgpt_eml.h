/*
 * MicroGPT-C — EML Organelle (Exp-Minus-Log Sheffer Operator)
 *
 * Copyright (c) 2026 Ajay Soni (ajay.soni@enjector.com), Enjector Software Ltd.
 * MIT License — see LICENSE file for details.
 *
 * Deployment-side evaluator for snapped EML trees produced by the EML
 * symbolic-regression trainer (PyTorch, in the companion research repo).
 *
 *   eml(a, b) = exp(a) - log(b)
 *
 * Per Odrzywolek (arXiv:2603.21852), eml + the constant 1 forms a Sheffer
 * basis for elementary functions.  The PyTorch trainer fits a fixed-depth
 * binary tree of eml nodes whose leaves softmax over {1, x, y} and whose
 * internal-node input slots sigmoid-gate between {1, child}.  After
 * training, the soft tree is "snapped" to a discrete tree with hard 0/1
 * choices.  This evaluator consumes that snapped form.
 *
 * Use cases for OPA:
 *   - Frozen organelle that recovers a shallow elementary closed-form law
 *     from noisy training data, then evaluates deterministically in C99.
 *   - Drop-in symbolic primitive on the deployment side of the pipeline,
 *     replacing a neural organelle when the underlying relation is known
 *     to be elementary and shallow (depth <= 4).
 *
 * Scope and non-scope:
 *   - This header ships ONLY the evaluator.  Training stays in PyTorch.
 *   - Suitable for shallow elementary targets with continuous (x, y)
 *     inputs.  Categorical / discrete tasks (the OPA games) are NOT
 *     candidates for this organelle — use neural organelles there.
 *   - See docs/research/RESEARCH_EML_ORGANELLE.md for the full story
 *     and the parent research doc at experiments/RESEARCH.md (eml repo).
 */

#ifndef MICROGPT_EML_H
#define MICROGPT_EML_H

#include "microgpt.h"  /* for scalar_t */
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ===================== Snapped EML tree representation =================== */
/*
 * A snapped EML tree at depth D has:
 *   - 2^D leaves, each a categorical choice from {const_1, x, y}.
 *   - 2^D - 1 internal nodes, each with two input slots; each slot is a
 *     binary choice between {const_1, child_value}.
 *
 * Leaf encoding (uint8_t):
 *   0 -> constant 1
 *   1 -> input variable x
 *   2 -> input variable y
 *
 * Gate encoding (uint8_t per slot):
 *   0 -> use the child's value
 *   1 -> use the constant 1 (the child's value is then computed but ignored)
 *
 * Internal nodes are stored in bottom-up flat order, identical to the
 * trainer's `tree_prototype_torch_v16_final.py` convention:
 *   index 0 .. n_internal-1, where lower indices correspond to the
 *   bottom-most internals.
 *
 * The struct is small enough to embed as a `const` literal in a generated
 * header.  See tools/eml_export.py.
 */

#define EML_LEAF_CONST  0
#define EML_LEAF_X      1
#define EML_LEAF_Y      2

#define EML_GATE_CHILD  0
#define EML_GATE_CONST  1

typedef struct {
  int depth;            /* tree depth; 2^depth leaves, 2^depth-1 internals */
  int n_leaves;         /* always 2^depth */
  int n_internal;       /* always 2^depth - 1 */
  const unsigned char *leaves;       /* length n_leaves */
  const unsigned char *gates_left;   /* length n_internal */
  const unsigned char *gates_right;  /* length n_internal */
} EmlTree;

/* ============================== Evaluation =============================== */
/*
 * eml_eval — evaluate the snapped tree at a single (x, y) point.
 *
 * Operates in real arithmetic via <math.h>'s exp() and log().  The trainer
 * uses complex128 internally because some elementary functions (e.g.
 * trigonometric forms) require it; deployment trees that recover purely
 * real-domain elementary laws can be evaluated in real arithmetic.
 *
 * Returns NAN if the tree's chain produces a non-finite intermediate, e.g.
 * because a log() argument went non-positive.  Callers should treat NAN
 * as a "tree out of valid domain" signal rather than a hard error.
 */
scalar_t eml_eval(const EmlTree *tree, scalar_t x, scalar_t y);

/*
 * eml_eval_batch — evaluate the tree on n input pairs.  Stores predictions
 * in `out` (length n).  Equivalent to a loop over eml_eval.
 */
void eml_eval_batch(const EmlTree *tree, const scalar_t *xs,
                    const scalar_t *ys, scalar_t *out, size_t n);

/* ============================== Diagnostics ============================== */
/*
 * eml_max_abs_err — convenience: return the max |pred - target| over the
 * supplied grid.  Useful for unit tests asserting that a hand-built tree
 * recovers a known function to machine precision.
 */
scalar_t eml_max_abs_err(const EmlTree *tree, const scalar_t *xs,
                         const scalar_t *ys, const scalar_t *targets,
                         size_t n);

/*
 * eml_mse — mean squared error over a batch.  Returns the residual that
 * the snapped tree leaves on the test set; on a clean test set with the
 * tree exactly equal to the data-generating relation, this is 0.
 */
scalar_t eml_mse(const EmlTree *tree, const scalar_t *xs, const scalar_t *ys,
                 const scalar_t *targets, size_t n);

#ifdef __cplusplus
}
#endif

#endif /* MICROGPT_EML_H */
