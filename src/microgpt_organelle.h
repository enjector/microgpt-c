/*
 * MicroGPT-C — Organelle Pipeline Architecture (OPA) Library
 *
 * Copyright (c) 2026 Ajay Soni (ajay.soni@enjector.com), Enjector Software Ltd.
 * MIT License — see LICENSE file for details.
 *
 * Generic infrastructure for multi-organelle pipelines:
 *   - Organelle training (corpus → model + checkpoint)
 *   - Organelle inference (prompt → response)
 *   - Kanban state (blocked actions, move history, stall tracking)
 *   - Cycle detection (A↔B oscillation breaking)
 *   - Pipe-string parsing helpers
 *   - Multi-line corpus loader
 *
 * Domain-specific code (board representation, evaluation functions,
 * move validation, pipeline wiring) stays in each experiment's main.c.
 */

#ifndef MICROGPT_ORGANELLE_H
#define MICROGPT_ORGANELLE_H

#include "microgpt.h"
#include <stddef.h>

/* ======================== Organelle ====================================== */
/*
 * An Organelle bundles a trained model with its vocabulary and training docs.
 * Use organelle_train() to create one, organelle_generate() to infer,
 * and organelle_free() to release resources.
 */

typedef struct {
  Model *model;
  Vocab vocab;
  Docs docs;
} Organelle;

/*
 * Train an organelle from a corpus file.  Resumes from checkpoint if
 * available, otherwise trains for num_steps and saves a checkpoint.
 * Returns NULL on failure.
 */
Organelle *organelle_train(const char *name, const char *corpus_path,
                           const char *ckpt_path, MicrogptConfig *cfg,
                           int num_steps);

/*
 * Generate a response from a prompt.  Feeds the prompt character-by-character,
 * then samples up to max_len output characters at the given temperature.
 * Output is null-terminated and stops at newline or BOS.
 */
void organelle_generate(const Organelle *org, const MicrogptConfig *cfg,
                        const char *prompt, char *output, int max_len,
                        scalar_t temperature);

/*
 * Free all resources owned by an organelle (model, docs).
 * The Organelle pointer itself is also freed.
 */
void organelle_free(Organelle *org);

/* ======================== Kanban State =================================== */
/*
 * Tracks blocked actions, recent action history, stall counts, and replans.
 * Actions are stored as comma-separated strings for pipe-string compatibility.
 */

typedef struct {
  char blocked[64]; /* comma-separated blocked actions (e.g. "up,down") */
  char last[64];    /* comma-separated recent actions for history */
  int stalls;       /* consecutive failures without progress */
  int replans;      /* number of times Planner was re-invoked */
  int max_history;  /* max entries in last[] before oldest is dropped */
} OpaKanban;

/* Initialise kanban state.  max_history=0 disables last[] tracking. */
void opa_kanban_init(OpaKanban *kb, int max_history);

/* Add an action string to the blocked set (no-op if already present). */
void opa_kanban_add_blocked(OpaKanban *kb, const char *action);

/* Check whether an action is in the blocked set. */
int opa_kanban_is_blocked(const OpaKanban *kb, const char *action);

/* Clear all blocked actions. */
void opa_kanban_clear_blocked(OpaKanban *kb);

/* Add an action to the recent-history ring (drops oldest if full). */
void opa_kanban_add_last(OpaKanban *kb, const char *action);

/* ======================== Cycle Detector ================================= */
/*
 * Detects A↔B oscillation patterns in a sequence of integer-encoded actions.
 * Usage:
 *   1. opa_cycle_init() at start of each puzzle/game
 *   2. opa_cycle_detected(cd, proposed_action) before executing a move
 *   3. opa_cycle_record(cd, action) after a move is accepted
 */

#ifndef OPA_CYCLE_WINDOW
#define OPA_CYCLE_WINDOW 8
#endif

typedef struct {
  int history[OPA_CYCLE_WINDOW];
  int len;
} OpaCycleDetector;

void opa_cycle_init(OpaCycleDetector *cd);

/*
 * Returns 1 if executing proposed_action would continue an A,B,A,B cycle.
 * Caller should then pick an alternative action.
 */
int opa_cycle_detected(const OpaCycleDetector *cd, int proposed_action);

/*
 * Returns the other action in the detected cycle (the B in A,B,A,B when
 * proposed_action is A).  Only valid when opa_cycle_detected() returns 1.
 */
int opa_cycle_other(const OpaCycleDetector *cd, int proposed_action);

/* Record an accepted action in the history ring. */
void opa_cycle_record(OpaCycleDetector *cd, int action_id);

/* ======================== Pipe-String Helpers ============================= */

/*
 * Extract the value for a key from a pipe-separated string.
 * E.g. opa_extract_pipe_value("board=XO_|empties=6", "board") → "XO_"
 * WARNING: modifies buf in-place (inserts '\0' at delimiters).
 */
const char *opa_extract_pipe_value(char *buf, const char *key);

/* Returns 1 if buf starts with prefix. */
int opa_pipe_starts_with(const char *buf, const char *prefix);

/* ======================== Corpus Loader =================================== */

/*
 * Load a multi-line corpus (documents separated by blank lines).
 * Each document is a prompt line + newline + response line.
 * Returns 0 on success, -1 on failure.
 */
int opa_load_docs_multiline(const char *path, Docs *docs, int max_docs);

/* ======================== Ensemble Voting ================================ */
/*
 * Generate a response by running N inferences with temperature jitter and
 * majority-voting the result.  This is domain-agnostic: workers remain
 * generalists, but ensemble agreement filters out low-confidence outliers.
 *
 * n_votes:     number of inference runs (odd recommended, e.g. 3 or 5)
 * base_temp:   centre temperature; each vote jitters ±OPA_TEMP_JITTER
 * confidence:  if non-NULL, receives the fraction of votes that agreed
 *              with the winning answer (e.g. 0.67 for 2/3 agreement)
 */

#ifndef OPA_TEMP_JITTER
#define OPA_TEMP_JITTER 0.05
#endif

#ifndef OPA_MAX_VOTES
#define OPA_MAX_VOTES 7
#endif

void organelle_generate_ensemble(const Organelle *org,
                                 const MicrogptConfig *cfg, const char *prompt,
                                 char *output, int max_len, int n_votes,
                                 scalar_t base_temp, scalar_t *confidence);

/* ======================== Valid-Move Filter =============================== */
/*
 * Check whether a proposed action appears in a comma-separated valid list.
 * Returns 1 if action is found, 0 if not.
 * If valid_csv is NULL or empty, returns 1 (no constraint = all valid).
 *
 * Example: opa_valid_filter("3", "0,2,3,5") → 1
 *          opa_valid_filter("4", "0,2,3,5") → 0
 */
int opa_valid_filter(const char *action, const char *valid_csv);

/*
 * Pick the first valid action from valid_csv that is NOT blocked in kb.
 * Writes the result to fallback (must be at least 16 bytes).
 * Returns 1 if a fallback was found, 0 if all valid actions are blocked.
 */
int opa_valid_fallback(const OpaKanban *kb, const char *valid_csv,
                       char *fallback, size_t fallback_sz);

#endif /* MICROGPT_ORGANELLE_H */
