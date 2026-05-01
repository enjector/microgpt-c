# TDD_organelle — Technical Design Document

**Document ID:** TDD-ORG-001
**Version:** 1.0
**Status:** DRAFT
**Paired BS:** `BS_organelle.md`
**Sources:** `src/microgpt_organelle.{h,c}`

## 1. Overview

The organelle module turns one trained MicroGPT-C model + its vocabulary + its training corpus into a runnable specialist (`Organelle`), and provides the deterministic C scaffolding (`OpaKanban`, `OpaCycleDetector`, valid-move filter, ensemble vote, `OpaTrace`) that a multi-organelle pipeline needs to keep small models honest.

The intelligence claim of the platform — that tiny coordinated specialists outperform monoliths of similar parameter budget on focused tasks — rests on this module's scaffolding plus the per-domain `main.c` that wires the organelles together.

## 2. Architecture

```
                  ┌─────────────────────────────────┐
                  │        Organelle (per role)      │
                  │  ┌────────┐  ┌────────┐         │
                  │  │ Model  │  │ Vocab  │         │
                  │  └────────┘  └────────┘         │
                  │       (or WordVocab + word_level)│
                  │  ┌────────┐                      │
                  │  │ Docs   │ (training corpus)    │
                  │  └────────┘                      │
                  └─────────────────────────────────┘
                              ▲
                              │
   ┌──────────────────────────┴───────────────────────────────────┐
   │ Pipeline scaffolding (deterministic C, ~340 LoC overall)      │
   │                                                                │
   │  OpaKanban       — blocked actions, recent history, stalls    │
   │  OpaCycleDetector — A↔B oscillation breaker (8-step window)   │
   │  opa_valid_filter / opa_valid_fallback — legal-move guard     │
   │  organelle_generate_ensemble — N-vote majority                │
   │  OpaTrace — record every step's outcome for re-trainable logs │
   └─────────────────────────────────────────────────────────────-─┘
                              ▲
                              │
   ┌──────────────────────────┴───────────────────────────────────┐
   │  Domain-specific main.c (e.g. demos/character-level/connect4) │
   │   • Loads / trains 2–5 organelles                             │
   │   • Wires planner.output  →  player.prompt  →  judge.input    │
   │   • Calls organelle_generate / _ensemble at each step          │
   │   • Updates kanban, runs cycle detector, writes trace          │
   └─────────────────────────────────────────────────────────────-─┘
```

## 3. Data flow — typical OPA pipeline step

```
1. Game / puzzle engine encodes state as a pipe-string:
       "STATE|board=123456780|empties=1|moves=R,D,L"

2. Planner organelle reads the state, emits:
       "PLAN|next=R|reason=center"

3. Player organelle reads the plan + state, emits:
       "ACTION|move=R"

4. opa_valid_filter("R", "R,D,L")  → ok
   opa_kanban_is_blocked(&kb, "R") → no
   opa_cycle_detected(&cd, R)      → no
   apply move

5. opa_kanban_add_last(&kb, "R")
   opa_cycle_record(&cd, R)
   opa_trace_record(...)
```

Stalls (no progress for K steps) trigger `replan`: the planner is re-invoked. Cycles trigger `cycle_break`: the cycle detector forces an alternative action.

## 4. Key data structures

### 4.1 `Organelle`

```c
typedef struct {
  Model    *model;
  Vocab     vocab;       /* if word_level == 0 */
  WordVocab word_vocab;  /* if word_level == 1 */
  Docs      docs;
  int       word_level;
} Organelle;
```

The Organelle owns its model, its vocabulary, and the corpus it was trained from. The corpus is retained because some pipelines (model soup, transfer training) need it post-construction.

### 4.2 `OpaKanban`

A small fixed-size struct holding:

- `blocked[64]` — comma-separated string of currently-blocked actions (no allocation).
- `last[64]` — comma-separated string of the recent action history.
- `stalls`, `replans` — counters for stall-detection and planner re-invocation accounting.
- `max_history` — caps the `last[]` ring (0 disables).

Blocked actions and the last-action ring use string membership rather than hash sets; the cardinalities involved (≤ 64 chars) make this faster than hashing.

### 4.3 `OpaCycleDetector`

```c
typedef struct {
  int history[OPA_CYCLE_WINDOW];  /* default 8 */
  int len;
} OpaCycleDetector;
```

Detects A↔B oscillation by inspecting the last 4 entries (`A B A`-like patterns). `opa_cycle_other` returns the alternative action so the controller can break out.

### 4.4 `OpaTrace`

A bounded array of `OpaTraceStep` entries (default `OPA_TRACE_MAX_STEPS = 64`). Each step records the proposed action, outcome (`OpaStepOutcome`), metric before / after, kanban snapshot, and `from_model` flag. `opa_trace_to_corpus` and `opa_trace_write` serialise to a re-trainable corpus format (see `FS_organelle_wire.md` §6).

## 5. Algorithms

### 5.1 `organelle_train`

```
1. open ckpt_path; if exists, checkpoint_load(...) and return.
2. else:
     opa_load_docs_multiline(corpus_path, &org->docs, ...)
     build_vocab / build_word_vocab from docs
     model_create(...)
     allocate grads, m, v
     for step in 0..num_steps:
         shuffle_docs(&org->docs) every K steps
         for batch:
             for token:
                 forward_backward_one
         clip_gradients(grads)
         adam_step(...)
     checkpoint_save(...)
```

`organelle_train_transfer` is the same flow but pre-populates `wpe`, attention, and MLP weights via `model_transfer_weights` from a source model before the random init step. `organelle_train_soup` trains `n_seeds` independent models with different RNG seeds, then averages the weights via `model_soup_average`.

### 5.2 `organelle_generate`

Implements the inference protocol specified in `FS_organelle_wire.md` §2. Allocates KV caches per call (stateless inference), feeds BOS + prompt + newline, then auto-regressively decodes until newline / BOS / `max_len`.

### 5.3 Ensemble voting

`organelle_generate_ensemble(org, cfg, prompt, output, max_len, n_votes, base_temp, &confidence)`:

```
for v in 0..n_votes:
    temp = base_temp + jitter ∈ [-OPA_TEMP_JITTER, +OPA_TEMP_JITTER]
    organelle_generate(... output_v ...)
candidates[] = output_v
winner = mode(candidates)
confidence = count(winner) / n_votes
```

Recommended `n_votes` is 3 or 5 (odd). The confidence value gates downstream decisions — high confidence proceeds autonomously, low confidence escalates / requests training data.

A speculative-decoding variant (`organelle_generate_speculative`) lets a small "draft" organelle propose `spec_k` tokens; the larger "target" organelle verifies them in a single pass and recovers via KV-cache rollback on rejection.

### 5.4 Valid-move filter

`opa_valid_filter(action, valid_csv)` is a CSV membership test. `opa_valid_fallback(kb, valid_csv, fallback, sz)` returns the first valid action that is NOT in `kb->blocked`, used when the model output is invalid or blocked.

## 6. Concurrency model

Organelles are read-only after training; ensemble votes can be parallelised by the caller. The current shipped code is single-threaded inference per organelle. Training uses the core `TrainWorker` harness (see `TDD_core.md` §6).

## 7. Trade-offs considered

| Decision | Chosen | Rejected | Rationale |
|---|---|---|---|
| Wire format | Pipe strings (`KEY=value\|...`) | JSON / Protobuf | Pipe strings are tokenisable as plain text and cost ≤ 1 instruction per byte to parse. JSON would inflate the corpus 3–5×. |
| Coordination logic | Deterministic C (Kanban, cycle detector, valid filter) | Larger model that learns coordination | The Neural Algorithmic Reasoning argument: gradient descent is poor at exact coordination; deterministic C handles it in 30–80 lines per concern, freeing every model parameter for fuzzy pattern matching. |
| Inference statefulness | Stateless (KV cache allocated per call) | Persistent caches | Game pipelines re-enter inference frequently; a persistent cache would need explicit reset hooks. The cost is small at the model sizes shipped. |
| Confidence signal | Ensemble agreement fraction | Entropy of softmax | Ensemble jitter exposes the model's actual decision-boundary noise; raw entropy can be over-confident on memorised inputs. |

## 8. Known limitations

- `OpaKanban.blocked` and `last` are fixed 64-byte buffers; long action vocabularies need re-sizing or an alternate data structure.
- `opa_extract_pipe_value` mutates the buffer it reads from; callers MUST pass a writable copy if the same buffer is read repeatedly.
- The valid-move filter accepts any string in the CSV without sanitisation; the producer is responsible for not putting `|` or `,` in action names.
- The cycle detector handles only A↔B oscillation; longer cycles (A→B→C→A) are not detected.
- Speculative decoding currently requires both organelles to share a vocabulary; cross-vocab speculation is documented future work.

## 9. References

- `book/` Chapter 4 (Organelles), Chapter 8 (Pipelines), Chapter 11 (OpaBoard / Kanban).
- `docs/research/RESEARCH_ORGANELLE_PIPELINE.md` — white paper.
- `docs/research/RESEARCH_ORGANELLE_REASONING.md` — Neural Algorithmic Reasoning argument.
- `docs/research/RESEARCH_ORGANELLE_GAMES.md` — game leaderboard.

## 10. Revision history

| Version | Date | Change |
|---|---|---|
| 1.0 | 2026-04-30 | Initial extraction. |
