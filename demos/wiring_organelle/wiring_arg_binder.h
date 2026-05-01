/*
 * MicroGPT-C — Argument-to-port binder (Phase 6b, Stream D).
 *
 * Closes R1 + R2 in COMPOSITIONAL_GENERATOR_FIX_PLAN.md v2.0:
 *   - prompt-noun → port-name binding
 *   - repeated-noun unification across ports
 *
 * Given an NL prompt, the chosen outer + inner primitives, and the
 * primitive manifest, the binder decides which signature input each
 * input port should consume. Two ports that bind to the same prompt
 * noun are aliased onto the SAME signature input — eliminating the
 * V1.0.6 duplicate-inner misrouting failure mode.
 *
 * Copyright (c) 2026 Ajay Soni, Enjector Software Ltd. MIT License.
 */

#ifndef WIRING_ARG_BINDER_H
#define WIRING_ARG_BINDER_H

#include "wiring_primitive_manifest.h"

#ifdef __cplusplus
extern "C" {
#endif

#define WIRING_ARG_MAX_NODES 8       /* outer + up to N-1 inners */
#define WIRING_ARG_MAX_SIG_INS 16    /* hard ceiling matching the search */
#define WIRING_ARG_NAME_LEN 24

/* One port binding decision: which signature input slot the port reads. */
typedef struct {
    int node_idx;     /* 0 = outer, 1+ = inner_picks ordered as in build_graph_for_outer */
    int port_idx;     /* index of the input port on that node */
    int sig_in_idx;   /* index into the signature_in[] array */
    char noun[WIRING_ARG_NAME_LEN]; /* the prompt noun bound to this slot, or "" if positional */
} WiringArgBinding;

typedef struct {
    int n_bindings;                          /* total port bindings (one per input port across all nodes) */
    WiringArgBinding bindings[32];           /* hard-cap; 8 nodes × 4 ports */

    int n_sig_inputs;                        /* unique signature inputs after unification */
    char sig_in_names[WIRING_ARG_MAX_SIG_INS][WIRING_ARG_NAME_LEN];
} WiringBindResult;

/*
 * Run the binder.
 *
 *   prompt:        NL prompt (lowercase already done by caller is fine, both work).
 *   outer_idx:     manifest index of the outer primitive.
 *   inner_picks:   length WIRING_PRIM_MAX_INPUTS, -1 for "no inner".
 *   result:        out — populated bindings + signature input list.
 *
 * Returns 1 on success, 0 on failure (e.g. too many ports for the
 * fixed-size buffers).
 */
int wiring_arg_bind(const char *prompt,
                    int outer_idx,
                    const int *inner_picks,
                    WiringBindResult *result);

#ifdef __cplusplus
}
#endif

#endif /* WIRING_ARG_BINDER_H */
