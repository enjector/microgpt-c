/*
 * MicroGPT-C — EML Organelle: snapped-tree evaluator.
 * Copyright (c) 2026 Ajay Soni, Enjector Software Ltd.  MIT License.
 *
 * Implementation of microgpt_eml.h.  Pure C99 + <math.h>.  No allocation.
 */

#include "microgpt_eml.h"

#include <math.h>
#include <stddef.h>

/* ----------------------------- internal helpers ------------------------- */

static scalar_t eml_op(scalar_t a, scalar_t b) {
  /* eml(a, b) = exp(a) - log(b).  In real arithmetic, log requires b > 0. */
  return (scalar_t)(exp((double)a) - log((double)b));
}

/*
 * leaf_value — resolve a leaf's symbolic choice to a numeric value at (x, y).
 */
static scalar_t leaf_value(unsigned char choice, scalar_t x, scalar_t y) {
  switch (choice) {
    case EML_LEAF_CONST: return (scalar_t)1.0;
    case EML_LEAF_X:     return x;
    case EML_LEAF_Y:     return y;
    default:             return (scalar_t)NAN;
  }
}

/*
 * gate_apply — given a gate choice and a child's value, return the value
 * fed into the parent's eml() input slot.
 */
static scalar_t gate_apply(unsigned char choice, scalar_t child_value) {
  if (choice == EML_GATE_CONST) return (scalar_t)1.0;
  return child_value;  /* EML_GATE_CHILD or any non-CONST encoding */
}

/* --------------------------------- API ---------------------------------- */

scalar_t eml_eval(const EmlTree *tree, scalar_t x, scalar_t y) {
  if (tree == NULL || tree->n_leaves <= 0) return (scalar_t)NAN;

  /*
   * Bottom-up evaluation.  We use a small fixed buffer indexed by the
   * "live" slot count at the current level.  Maximum supported depth here
   * is bounded by EML_MAX_DEPTH; deeper trees would overflow the stack
   * buffer.  In practice the trainer's reachable regime is depth <= 4
   * (16 leaves), so we cap at 7 (128 leaves) for safety headroom.
   */
  enum { EML_MAX_DEPTH = 7, EML_MAX_LEAVES = 1 << 7 };
  if (tree->depth > EML_MAX_DEPTH) return (scalar_t)NAN;

  scalar_t buf[EML_MAX_LEAVES];

  /* Resolve leaves to values. */
  for (int i = 0; i < tree->n_leaves; ++i) {
    buf[i] = leaf_value(tree->leaves[i], x, y);
  }

  int level_size = tree->n_leaves;
  int internal_idx = 0;

  while (level_size > 1) {
    int n_pairs = level_size / 2;
    for (int j = 0; j < n_pairs; ++j) {
      scalar_t left_child  = buf[2 * j];
      scalar_t right_child = buf[2 * j + 1];
      scalar_t left_input  = gate_apply(tree->gates_left[internal_idx],
                                        left_child);
      scalar_t right_input = gate_apply(tree->gates_right[internal_idx],
                                        right_child);
      buf[j] = eml_op(left_input, right_input);
      ++internal_idx;
    }
    level_size = n_pairs;
  }

  return buf[0];
}

void eml_eval_batch(const EmlTree *tree, const scalar_t *xs,
                    const scalar_t *ys, scalar_t *out, size_t n) {
  for (size_t i = 0; i < n; ++i) {
    out[i] = eml_eval(tree, xs[i], ys[i]);
  }
}

scalar_t eml_max_abs_err(const EmlTree *tree, const scalar_t *xs,
                         const scalar_t *ys, const scalar_t *targets,
                         size_t n) {
  scalar_t worst = (scalar_t)0.0;
  for (size_t i = 0; i < n; ++i) {
    scalar_t pred = eml_eval(tree, xs[i], ys[i]);
    scalar_t err = (scalar_t)fabs((double)(pred - targets[i]));
    if (!isfinite((double)err)) {
      return (scalar_t)INFINITY;
    }
    if (err > worst) worst = err;
  }
  return worst;
}

scalar_t eml_mse(const EmlTree *tree, const scalar_t *xs, const scalar_t *ys,
                 const scalar_t *targets, size_t n) {
  if (n == 0) return (scalar_t)0.0;
  double acc = 0.0;
  size_t finite_count = 0;
  for (size_t i = 0; i < n; ++i) {
    scalar_t pred = eml_eval(tree, xs[i], ys[i]);
    if (!isfinite((double)pred)) continue;
    double d = (double)pred - (double)targets[i];
    acc += d * d;
    ++finite_count;
  }
  if (finite_count == 0) return (scalar_t)NAN;
  return (scalar_t)(acc / (double)finite_count);
}
