/*
 * MicroGPT-C — Wiring Organelle native primitive registry (Phase 6)
 *
 * Bridges the Pipeline IR's PipelineDispatchFn contract to a small
 * library of C implementations of the ~40 most common primitives in
 * demos/word-level/vm_codegen/w_vm_functions.txt. Verified Pipeline
 * IR graphs emitted by the Wiring Organelle execute end-to-end on
 * these natives via pipeline_execute().
 *
 * The Pipeline IR header has pipeline_execute_vm() as a deferred stub
 * (it would require synthesising VM scripts for arbitrary graphs).
 * This module bypasses that complication: native C functions plug
 * directly into pipeline_execute()'s dispatch callback.
 *
 * Copyright (c) 2026 Ajay Soni, Enjector Software Ltd. MIT License.
 */

#ifndef WIRING_NATIVES_H
#define WIRING_NATIVES_H

#include "microgpt_pipeline.h"

#ifdef __cplusplus
extern "C" {
#endif

/* PipelineDispatchFn implementation that resolves primitive names
 * against the wiring native registry. Pass as the `dispatch` arg to
 * pipeline_execute(). user_data is unused. Returns 0 on success or
 * a non-zero error code if the primitive isn't registered. */
int wiring_natives_dispatch(const char *primitive,
                            const PipelineConfig *config, int n_config,
                            const PipelineValue *inputs, int n_inputs,
                            PipelineValue *outputs, int n_outputs,
                            void *user_data);

/* Returns 1 if `primitive` is implemented by wiring_natives_dispatch,
 * 0 otherwise. Lets callers prune the held-out set to graphs whose
 * nodes are all executable. */
int wiring_natives_known(const char *primitive);

#ifdef __cplusplus
}
#endif

#endif /* WIRING_NATIVES_H */
