/*
 * wiring_geo_classifier.h — Phase 1c geodesic family classifier
 *
 * Lightweight family-prediction module: handcoded anchor table +
 * keyword bag + 12D Geodesic distance. Same logic as the
 * manifold_classifier_demo, packaged for reuse from
 * wiring_organelle's eval loop.
 *
 * Reference: docs/research/RESEARCH_PIPELINE_IR.md §33.
 */

#ifndef WIRING_GEO_CLASSIFIER_H
#define WIRING_GEO_CLASSIFIER_H

#include <stddef.h>

#define WIRING_GEO_TOP_K 3

/* Predict the top-K family names for `prompt`. Writes up to K
 * non-NULL pointers into out[] (statically-allocated strings, do
 * NOT free). Returns the number of families written. */
int wiring_geo_predict_top_k(const char *prompt,
                             const char *out[WIRING_GEO_TOP_K]);

/* Returns 1 if `family` (or `graph_name`'s family-prefix after
 * stripping trailing _<digits>) appears in the top-K set, 0 otherwise. */
int wiring_geo_in_top_k(const char *graph_name,
                        const char *top_k[WIRING_GEO_TOP_K],
                        int n_top_k);

#endif /* WIRING_GEO_CLASSIFIER_H */
