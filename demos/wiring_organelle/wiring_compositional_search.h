/*
 * MicroGPT-C — Type-directed compositional search (Stream B of compositional fix)
 *
 * Closes GAP-WIRE-005 in docs.engineering/CLEAN_ROOM_IMPLEMENTATION/TRACEABILITY.md.
 *
 * Given an NL prompt and a target output type, this module synthesises a
 * verified Pipeline IR graph by:
 *   1. Keyword-matching the prompt against the primitive manifest.
 *   2. Greedy reverse construction: from the target output type, pick a
 *      primitive whose output type matches; bind each of its inputs
 *      either to a graph signature input (consuming a fresh NL noun) or
 *      recursively to another primitive's output.
 *   3. Building the Pipeline IR graph and running pipeline_verify().
 *
 * Output: a verified `Pipeline *` (caller frees with pipeline_free()) plus
 *         the rendered @graph text (caller frees the string).
 *
 * Copyright (c) 2026 Ajay Soni, Enjector Software Ltd. MIT License.
 */

#ifndef WIRING_COMPOSITIONAL_SEARCH_H
#define WIRING_COMPOSITIONAL_SEARCH_H

#include "microgpt_pipeline.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Maximum number of primitives in a generated graph. */
#define WIRING_COMPOSE_MAX_NODES 8

typedef struct {
    int n_nodes_used;            /* number of primitives the search picked */
    const char *primitive_names[WIRING_COMPOSE_MAX_NODES];
    int verified;                /* 1 if pipeline_verify succeeded */
    int signature_in_count;
    int signature_out_count;
} WiringComposeReport;

/*
 * Run a compositional search.  Returns a verified Pipeline* on success,
 * NULL if no valid composition could be synthesised from the manifest.
 *
 * prompt:           NL prompt (e.g. "compute the average of x and y squared").
 * report:           Optional out — populated with metadata about the search.
 *                   May be NULL.
 *
 * Caller owns the returned Pipeline* and must free with pipeline_free().
 *
 * The current implementation is deterministic and beam-width 1 (greedy):
 * each step picks the primitive with the highest keyword-hit count whose
 * output type matches the current bind point.  Future revisions may
 * widen the beam.
 */
Pipeline *wiring_compositional_search(const char *prompt,
                                      WiringComposeReport *report);

/*
 * Convenience: run the search, render the resulting graph, and return
 * the heap-allocated text (caller frees).  Returns NULL on failure.
 * If pipeline_out is non-NULL, the verified Pipeline* is also returned
 * to the caller (otherwise it is freed internally).
 */
char *wiring_compositional_search_render(const char *prompt,
                                         Pipeline **pipeline_out,
                                         WiringComposeReport *report);

#ifdef __cplusplus
}
#endif

#endif /* WIRING_COMPOSITIONAL_SEARCH_H */
