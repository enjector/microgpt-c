/*
 * wiring_fragments.h — Phase 3b fragment composition
 *
 * Decomposes the existing 20 single-family anchors into reusable
 * sub-DAG fragments + a composition operator that retrieves K=2-3
 * fragments per prompt and chains them by output→input linkage.
 *
 * The §42 pre-registration commits to:
 *   - 5-7 of 10 multi-stage composition prompts correct
 *   - ≥18/20 no-regression on Phase 2c clean paraphrases
 *
 * Reference: docs/research/RESEARCH_PIPELINE_IR.md §42.
 */

#ifndef WIRING_FRAGMENTS_H
#define WIRING_FRAGMENTS_H

#include <stddef.h>

/* Try to compose a graph for `prompt` by retrieving 2-3 fragments
 * from the fragment table and chaining them by output→input linkage.
 *
 * Writes the rendered @graph text to `out_buf` (caller-allocated,
 * ≥2048 bytes). Returns 1 on success (graph rendered), 0 on no
 * composition (fewer than 2 fragments hit, or chain doesn't type-check).
 *
 * Caller is responsible for parsing/verifying/executing the result
 * via the same pipeline as wiring votes and anchors. */
int wiring_compose_for_prompt(const char *prompt, char *out_buf, size_t out_size);

#endif /* WIRING_FRAGMENTS_H */
