/*
 * wiring_anchor_graphs.h — Phase 2 anchor-retrieval generation
 *
 * Hardcoded canonical @graph DAGs, one per held-out reference family.
 * When the geodesic classifier identifies the right family for a
 * prompt, the corresponding canonical DAG is retrieved as a candidate
 * directly — no token-level generation.
 *
 * This bypasses the three-layer ceiling characterised in
 * RESEARCH_PIPELINE_IR.md §34: re-rank, family-name selection, and
 * primitive selection are all handled by retrieval.
 */

#ifndef WIRING_ANCHOR_GRAPHS_H
#define WIRING_ANCHOR_GRAPHS_H

/* Look up the canonical @graph DAG text for `family_name` (e.g.
 * "fib_fact_add"). Returns NULL if not in the table. The returned
 * pointer is a static literal; do NOT free. */
const char *wiring_anchor_graph_for(const char *family_name);

/* How many anchor families are stored. */
int wiring_anchor_count(void);

#endif /* WIRING_ANCHOR_GRAPHS_H */
