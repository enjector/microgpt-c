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

/* Look up a reference function by name, run it with the test input
 * sequence, and write the answer to *out. Returns 1 if the name is
 * known and out is set; 0 otherwise. */
int wiring_reference_compute(const char *name, int64_t *out);

#ifdef __cplusplus
}
#endif

#endif /* WIRING_REFERENCES_H */
