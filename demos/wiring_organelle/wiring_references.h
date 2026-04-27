/*
 * MicroGPT-C — Wiring Organelle reference-answer suite (Phase 7)
 *
 * For each held-out NL prompt, a small C function computes the
 * canonical expected output given the same test input sequence
 * (5, 7, 3, 11, 2, 13, 4, 9, ...) the demo supplies to its
 * pipeline_execute() call. The demo compares the executed result
 * against the reference and reports a "correct" boolean.
 *
 * Headline (Phase 7): % of NL prompts producing the *correct*
 * numeric answer end-to-end.
 *
 * Copyright (c) 2026 Ajay Soni, Enjector Software Ltd. MIT License.
 */

#ifndef WIRING_REFERENCES_H
#define WIRING_REFERENCES_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Number of input sets the reference and demo should test against
 * for Phase 8 multi-input correctness. */
#define WIRING_INPUT_SETS 5

/* Maximum signature arity supported. */
#define WIRING_MAX_INPUTS 16

/* Fill `dst` with the kth test input sequence. dst length must be at
 * least WIRING_MAX_INPUTS. set_idx is taken mod WIRING_INPUT_SETS. */
void wiring_input_set(int set_idx, int64_t *dst);

/* Look up a reference function by name, run it with input set
 * `set_idx`, and write the answer to *out. Returns 1 if the name is
 * known; 0 otherwise. */
int wiring_reference_compute_at(const char *name, int set_idx, int64_t *out);

/* Backward-compat: equivalent to wiring_reference_compute_at(name, 0, out). */
int wiring_reference_compute(const char *name, int64_t *out);

#ifdef __cplusplus
}
#endif

#endif /* WIRING_REFERENCES_H */
