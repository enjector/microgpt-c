/*
 * MicroGPT-C — Wiring primitive manifest (Stream B of compositional fix)
 *
 * Catalogue of every primitive that wiring_natives.c can dispatch, with
 * its typed input/output signature and a small NL keyword set used by
 * the type-directed compositional search to map prompt nouns/verbs to
 * candidate primitives.
 *
 * The manifest is the load-bearing input to wiring_compositional_search.c:
 * given an NL prompt, the search picks primitives whose keywords appear
 * in the prompt, then composes them by output → input type linkage to
 * synthesise a new pipeline.
 *
 * Copyright (c) 2026 Ajay Soni, Enjector Software Ltd. MIT License.
 */

#ifndef WIRING_PRIMITIVE_MANIFEST_H
#define WIRING_PRIMITIVE_MANIFEST_H

#include "microgpt_pipeline.h"

#ifdef __cplusplus
extern "C" {
#endif

#define WIRING_PRIM_MAX_INPUTS   8
#define WIRING_PRIM_MAX_KEYWORDS 12
#define WIRING_PRIM_MAX_PORTS    8

typedef struct {
    /* Canonical primitive name (matches a key in wiring_natives.c registry). */
    const char *name;

    /* Number of inputs and the input port type kinds (PIPE_T_INT etc.). */
    int n_inputs;
    PipelineTypeKind input_types[WIRING_PRIM_MAX_INPUTS];

    /* Conventional input port names (used when constructing graphs). */
    const char *input_names[WIRING_PRIM_MAX_INPUTS];

    /* Output type kind (single output — matches the wiring_natives ABI). */
    PipelineTypeKind output_type;

    /* NL keywords (case-insensitive whole-word match). NULL-terminated;
     * fewer than WIRING_PRIM_MAX_KEYWORDS in practice. */
    const char *keywords[WIRING_PRIM_MAX_KEYWORDS];
} WiringPrimitive;

/* Returns a pointer to the static manifest array.  Length out_count. */
const WiringPrimitive *wiring_primitive_manifest(int *out_count);

/* Lookup a primitive by name; returns NULL if not in the manifest. */
const WiringPrimitive *wiring_primitive_find(const char *name);

#ifdef __cplusplus
}
#endif

#endif /* WIRING_PRIMITIVE_MANIFEST_H */
