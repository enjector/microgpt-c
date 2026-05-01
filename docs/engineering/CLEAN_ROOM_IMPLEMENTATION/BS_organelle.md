# BS_organelle — Behaviour Specification

**Document ID:** BS-ORG-001
**Version:** 1.0
**Status:** DRAFT

## RFC 2119

The key words MUST, MUST NOT, REQUIRED, SHALL, SHALL NOT, SHOULD, SHOULD NOT, RECOMMENDED, MAY, and OPTIONAL in this document are to be interpreted as described in RFC 2119.

## 1. Scope

This document specifies the behavioural contract of the **organelle pipeline architecture** — `Organelle` lifecycle (train, generate, free), the deterministic scaffolding (`OpaKanban`, `OpaCycleDetector`, `OpaTrace`, valid-move filter), ensemble voting, speculative decoding, and word-level variants.

## 2. Type contracts

### 2.1 `Organelle`

**Invariants:**
- INV-ORG-001: An `Organelle` SHALL bundle a `Model *`, a `Vocab` (when `word_level == 0`), a `WordVocab` (when `word_level == 1`), a `Docs`, and the boolean `word_level` flag.
- INV-ORG-002: After `organelle_free(org)`, all owned resources SHALL be released and the `org` pointer SHALL NOT be dereferenced.

### 2.2 `OpaKanban`

**Invariants:**
- INV-ORG-010: `kb->blocked` is a comma-separated string of currently-blocked actions, ≤ 63 bytes plus NUL.
- INV-ORG-011: `kb->last` is a ring of recent actions; entries beyond `kb->max_history` are evicted oldest-first.
- INV-ORG-012: `kb->stalls` increments when a step yields `OPA_STEP_STALL`; resets when `OPA_STEP_REPLAN` is recorded.

### 2.3 `OpaCycleDetector`

**Invariants:**
- INV-ORG-020: `cd->history[]` holds the last `OPA_CYCLE_WINDOW` (default 8) action IDs.
- INV-ORG-021: `opa_cycle_detected(cd, a)` returns 1 iff appending `a` would extend an A-B-A pattern.

### 2.4 `OpaTrace`

**Invariants:**
- INV-ORG-030: `trace->num_steps ≤ OPA_TRACE_MAX_STEPS` (default 64).
- INV-ORG-031: After `opa_trace_finalise`, `trace->final_metric` and `trace->solved` reflect the run outcome.

## 3. Operation contracts

### 3.1 `organelle_train(name, corpus_path, ckpt_path, cfg, num_steps)`

**Preconditions:** `corpus_path` exists; `ckpt_path` is a writable path; `cfg` valid.

**Postconditions:**
- If `ckpt_path` exists, the function SHALL load the checkpoint and return without further training.
- Otherwise the function SHALL load the corpus via `opa_load_docs_multiline`, build the vocabulary, train the model for `num_steps` Adam steps with shuffled batches, save a checkpoint to `ckpt_path`, and return.
- Returns NULL on any failure (corpus missing, OOM, checkpoint save error).

**Errors:** ERR-ORG-001.

### 3.2 `organelle_generate(org, cfg, prompt, output, max_len, temperature)`

**Preconditions:** `org` valid; `prompt` NUL-terminated; `output` writable, capacity ≥ `max_len + 1`.

**Postconditions:** Implements the §2 protocol of `FS_organelle_wire.md`: feed BOS + each prompt char + newline, then auto-regressively sample. Output is NUL-terminated and stops at the first newline / BOS / `max_len` / `block_size`.

### 3.3 `organelle_generate_multiline(...)`, `organelle_generate_words(...)`, `organelle_generate_from_cache(...)`

Variants of the protocol per `FRD.md` REQ-ORG-005..007. Contracts mirror `organelle_generate` with the documented termination differences (blank line for multiline; word-level tokens for the words variant; pre-filled cache for the from-cache variant).

### 3.4 `organelle_generate_speculative(draft, target, cfg, prompt, output, max_len, temperature, spec_k, &accepted, &drafted)`

**Preconditions:** `draft` and `target` MUST share the same vocabulary. `spec_k > 0` (typical 4–8). `accepted_out` and `drafted_out` MAY be NULL.

**Postconditions:** Generates `output` by speculative decoding. On rejection, `target`'s sampled token is the recovery token. `accepted` and `drafted` (if non-NULL) tally tokens for an acceptance-rate report.

### 3.5 `organelle_generate_ensemble(org, cfg, prompt, output, max_len, n_votes, base_temp, &confidence)`

**Preconditions:** `n_votes ≤ OPA_MAX_VOTES` (default 7); `base_temp > 0`.

**Postconditions:** Runs `n_votes` inferences with temperature jitter `± OPA_TEMP_JITTER` (default 0.05), majority-votes the results, writes the winner to `output`, and (if non-NULL) writes the agreement fraction to `*confidence`.

### 3.6 `OpaKanban` API

`opa_kanban_init`, `_add_blocked`, `_is_blocked`, `_clear_blocked`, `_add_last`. Each is a small in-place mutation; failure is impossible (fixed-size buffers — exceeding capacity silently ignores the new entry, INV-ORG-040).

### 3.7 `OpaCycleDetector` API

`opa_cycle_init(cd)` zeroes the history. `opa_cycle_detected(cd, a)` is read-only. `opa_cycle_other(cd, a)` returns the alternative action when a cycle was detected. `opa_cycle_record(cd, a)` appends `a` to the history.

### 3.8 Pipe-string helpers

`opa_extract_pipe_value(buf, key)` mutates `buf` in place to NUL-terminate the value; returns NULL on miss. `opa_pipe_starts_with(buf, prefix)` is a read-only string comparison.

### 3.9 Multi-line corpus loader

`opa_load_docs_multiline(path, docs, max_docs)` per `FS_organelle_wire.md` §5. Returns 0 on success, -1 on `fopen` / OOM.

### 3.10 Valid-move filter

`opa_valid_filter(action, valid_csv)` returns 1 iff `action` appears in `valid_csv` (or `valid_csv` is NULL/empty). `opa_valid_fallback(kb, valid_csv, fallback, sz)` writes the first valid non-blocked action into `fallback` and returns 1, or returns 0 if all are blocked.

### 3.11 Reasoning trace

`opa_trace_init(trace, initial_metric)` zeroes the trace. `opa_trace_record(trace, action, outcome, mb, ma, blocked, from_model)` appends a step (silently dropped beyond `OPA_TRACE_MAX_STEPS`). `opa_trace_finalise`, `_to_corpus`, `_write`, `_count`, `_has_recovery` per the header.

## 4. Invariants table (consolidated)

| ID | Invariant |
|---|---|
| INV-ORG-001 | Organelle bundles model + vocab(s) + corpus + word-level flag. |
| INV-ORG-002 | `organelle_free` releases all owned resources. |
| INV-ORG-010..012 | Kanban state machine: blocked / last / stalls. |
| INV-ORG-020..021 | Cycle detector A↔B oscillation. |
| INV-ORG-030..031 | OpaTrace bounded length; finalise sets terminal fields. |
| INV-ORG-040 | Kanban / trace overflow silently ignores new entries. |
| INV-ORG-050 | `organelle_generate_speculative` REQUIRES draft and target to share a vocabulary. |
| INV-ORG-051 | Ensemble vote uses `± OPA_TEMP_JITTER` around `base_temp` (default 0.05). |

## 5. Errors

| ID | Function | Code | Conditions |
|---|---|---|---|
| ERR-ORG-001 | `organelle_train*` | NULL | Corpus missing, OOM, checkpoint save error |
| ERR-ORG-002 | `opa_load_docs_multiline` | -1 | `fopen` / OOM / file > 2 GiB |
| ERR-ORG-003 | `opa_trace_to_corpus` | -1 | Output buffer too small |

## 6. Concurrency

Organelles are read-only after training; multiple threads MAY perform inference against the same organelle with disjoint KV caches. The deterministic scaffolding (`OpaKanban`, `OpaCycleDetector`, `OpaTrace`) is per-pipeline-run and SHALL NOT be shared across concurrent runs without external synchronisation.

## 7. Performance SLOs

The relevant SLOs are end-to-end demo metrics in `NFRD.md` §4.3 (game leaderboard timings).

## 8. Scenarios

### SCN-ORG-001: Two-organelle Connect-4 pipeline

A demo trains `planner.ckpt` and `player.ckpt` from per-role corpora. At play time, the game engine encodes state as `STATE|board=...|empties=...|moves=...`. The planner emits `PLAN|next=4|reason=...`. The player emits `ACTION|move=4`. The valid-move filter checks against `moves=`; the kanban blocks bad moves; the cycle detector breaks oscillations. Result: ≥ 88 % win rate against random.

### SCN-ORG-002: Ensemble voting on Mastermind

A 92K-param organelle plays Mastermind with `organelle_generate_ensemble(n_votes=5)`. The agreement fraction gates whether to act on the vote winner or escalate.

### SCN-ORG-003: Reasoning trace for retraining

A pipeline run records its reasoning via `OpaTrace`; `opa_trace_to_corpus` serialises it to a `prompt|action|outcome|...` line. Concatenated traces feed `opa_load_docs_multiline` for a future "process retrieval" training run.

## 9. Acceptance criteria

| ID | Verifies | Test |
|---|---|---|
| ACC-ORG-001 | INV-ORG-001..002 | `tests/test_microgpt_organelle.c::test_lifecycle` |
| ACC-ORG-002 | INV-ORG-010..021 | `tests/test_microgpt_organelle.c::test_kanban_cycle` |
| ACC-ORG-003 | INV-ORG-030..031 | `tests/test_microgpt_organelle.c::test_trace` |
| ACC-ORG-004 | BREQ-012, BREQ-013 (game thresholds) | Game demos: `c_puzzle8_demo`, `c_connect4_demo`, etc. |

## 10. Cross-references

- **TDD:** `TDD_organelle.md`
- **FS:** `FS_organelle_wire.md`
- **Source:** `src/microgpt_organelle.{h,c}`
- **Tests:** `tests/test_microgpt_organelle.c`
- **Upstream:** `BS_core.md`, `BS_tokeniser.md`
- **Downstream:** Demo `main.c` files in `demos/character-level/*`, `demos/word-level/*`.

## 11. Revision history

| Version | Date | Change |
|---|---|---|
| 1.0 | 2026-04-30 | Initial extraction. |
