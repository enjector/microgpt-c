/*
 * oql_runtime_train.h  —  OQL TRAIN adapter (Experiment E10)
 *
 * Copyright (c) 2026 Ajay Soni.  MIT License.
 *
 * Adapter glue between OQL's TRAIN verb (parsed into OqlTrainSpec) and the
 * existing engine training API in src/microgpt.{h,c}.  No new engine surface;
 * no VM opcode changes; no modification to src/microgpt.{h,c}.  All glue
 * lives here so the engine remains the substrate and OQL stays a thin
 * declarative layer over it.
 *
 * Lifecycle (mirrors the names_demo training loop in
 * demos/character-level/names/main.c):
 *
 *   oql_run_train(rt, spec, out)
 *       1. resolve spec->organelle_name against the runtime registry
 *          (must be a non-loaded organelle — TRAIN creates the model)
 *       2. resolve spec->corpus_name → OqlCorpus + slurp file lazily
 *       3. load_docs / build_vocab on the corpus contents (in-memory)
 *       4. model_create with the runtime's MicrogptConfig
 *       5. seed_rng + srand(spec->seed)
 *       6. for step in 0..spec->steps:
 *             zero grads
 *             for b in 0..spec->batch_size: forward_backward_one over doc
 *             clip + average grads
 *             adam_step(...)
 *             if rt->loss_log: rt->loss_log[step] = mean_loss
 *       7. if spec->save_path: checkpoint_save(model, m, v, step, path)
 *       8. organelle->model = model; organelle->loaded = 1
 *
 * The adapter records summary metrics on rt->last_train_* and the per-step
 * loss curve on rt->loss_log (when attached by the caller).
 *
 * Returns OQL_OK on success, OQL_ERR_RUNTIME on misconfiguration / load
 * failure, OQL_ERR_OOM on alloc failure.
 *
 * Compile-time-macro caveat (E09 §3.4): TRAIN allocates the model with the
 * lib variant's compile-time dims (N_EMBD, N_HEAD, …).  A binary linking
 * the default OQL lib has names-incompatible dims; the E10 build registers
 * a dedicated `oql_names` variant for the loss-curve smoke test.  See the
 * RESEARCH_DEEPSEEK_V4_*-style note in experiments/E10-oql-train-wiring.md
 * §3 for the resolution.
 */

#ifndef OQL_RUNTIME_TRAIN_H
#define OQL_RUNTIME_TRAIN_H

#include "microgpt_oql.h"
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

/* TRAIN entry point — called by oql_execute_with_runtime when it
 * encounters an OQL_VERB_TRAIN statement.  `spec` is built from the
 * statement's AST; on completion, the organelle named spec->organelle_name
 * is populated with the trained model (organelle->loaded = 1).
 *
 * If spec->save_path is non-NULL, a checkpoint is also written via
 * checkpoint_save() so a follow-up CREATE ORGANELLE FROM CHECKPOINT can
 * round-trip the run (E10 T4).
 *
 * Returns:
 *   OQL_OK              — training completed; metrics on rt->last_train_*
 *   OQL_ERR_RUNTIME     — corpus or organelle missing / load failure
 *   OQL_ERR_OOM         — allocation failure
 *   OQL_ERR_NOT_IMPLEMENTED — quantised builds (INT8) reject training
 */
oql_status oql_run_train(OqlRuntime *rt,
                         const OqlTrainSpec *spec,
                         FILE *out);

#ifdef __cplusplus
}
#endif

#endif /* OQL_RUNTIME_TRAIN_H */
