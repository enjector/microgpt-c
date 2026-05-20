# Experiment E10 — Wire OQL `TRAIN` so OQL scripts can train organelles from scratch, not just load checkpoints

**Status:** 📋 Proposal locked — 2026-05-20.
**Direction:** close the last remaining `OQL_ERR_NOT_IMPLEMENTED` stub from E07. With `TRAIN` wired, an OQL script becomes self-contained — corpus → trained organelle → composed pipeline → game loop, all in one `.oql` file.
**Cost estimate:** ~3-5 weeks (1 wk grammar + 1 wk interpreter struct + 1 wk binding to `TrainWorker` + 1 wk loss-curve fidelity check + 1 wk writeup).
**Falsification risk:** Medium — depends on whether the existing `TrainWorker` pthread harness in `microgpt.c` cleanly exposes the lifecycle (init / step / save) the OQL runtime needs to drive.

---

## Spear summary

**Point:** E09 closed three of E07's four stubbed verbs (`RUN`, `COMPOSE`, `CREATE ORGANELLE FROM CHECKPOINT`). The last stub, `TRAIN`, is what makes OQL **read-only** today: scripts can load and execute trained organelles but cannot produce new ones. Wiring `TRAIN` collapses the authorship loop from *"write OQL → fall back to C to train → return to OQL"* to *"write OQL → run."*

**Picture:** A researcher writes one `.oql` file: `CREATE CORPUS shakespeare_tiny FROM FILE 'corpus.txt'; CREATE ORGANELLE poet WITH (input_behaviour = ...); TRAIN poet ON shakespeare_tiny STEPS 5000 LR 1e-3 SAVE 'checkpoints/poet.ckpt'; CREATE PIPELINE generate AS COMPOSE @graph... @end; RUN generate ON stdin RETURNING (...);`. Today: impossible. After E10: yes.

**Proof (to be measured):** training loop drives end-to-end without errors; loss curve matches the existing C-demo equivalent (`names_demo` or `shakespeare_demo`) within ±10% on the same hyperparameters and seed; saved checkpoint round-trips through `CREATE ORGANELLE FROM CHECKPOINT` to identical inference outputs; **zero new VM opcodes** (E08 hard-lock preserved); E07/E08/E09 tests all pass.

**Push:** This is the experiment that makes OQL a fully self-sufficient research substrate. After E10, a published experiment ships as a single `.oql` file + a corpus file + the engine version pin — no C compilation required.

---

## 1. Proposal

### 1.1 Hypothesis (locked before measurement)

> *Adding a `TRAIN <organelle> ON <corpus> [WITH ROLE <role>] [STEPS <n>] [LR <rate>] [SAVE <path>];` clause to OQL's grammar and binding it to the existing `TrainWorker` pthread harness in `src/microgpt.c` is sufficient to train an organelle end-to-end from an OQL script, with: (a) loss-curve fidelity to the equivalent C demo within ±10% on the same seed and hyperparameters; (b) saved checkpoint round-trips bit-identically through `CREATE ORGANELLE FROM CHECKPOINT`; (c) the existing E07/E08/E09 tests all pass; (d) zero new VM opcodes added (E08 T3-lock preserved); (e) the `+6/-4` verb surface lock from E07 still holds (TRAIN was always in the +6).*

### 1.2 Why this matters

After E09, OQL can execute trained organelles but cannot produce new ones. That asymmetry breaks the EQL-derived "the query language is the product" framing: the *product* is incomplete — researchers still have to drop down to C99 to train any new model, then return to OQL to use it. The cognitive surface is still split across two languages.

After E10:
1. **The authorship loop closes.** Single-language workflow from raw corpus to trained organelle to running pipeline.
2. **E01's System C path is fully expressible in OQL.** Today System C uses `c4_player.ckpt` produced from the C demo; after E10, it could be produced inside the same OQL script that runs E01's measurement.
3. **The "single file reproduces the experiment" claim becomes literal.** No checkpoint dependency to ship separately.
4. **TRAIN's audit trail is queryable.** `SELECT loss, step FROM train_runs WHERE organelle = 'poet'` is meaningful after E10 — opens an audit dimension that the C demos don't natively expose.

### 1.3 Mechanism

**Phase 1 — Grammar (1 week).**

Extend `src/microgpt_oql.{l,y}` with the `TRAIN` clause. The clause already exists as a stubbed verb in E07's grammar; this phase adds the *arguments*:

```sql
TRAIN <organelle_name>
    ON <corpus_name>
    [WITH ROLE { planner | player | judge | <user_defined> }]
    [STEPS <integer>]
    [LR <float>]
    [BATCH_SIZE <integer>]
    [SAVE <path_string>]
    [SEED <integer>]
;
```

Defaults: `STEPS` = the engine's compile-time `NUM_STEPS` macro; `LR` = `LEARNING_RATE`; `BATCH_SIZE` = `BATCH_SIZE`; `SAVE` = no save (in-memory only); `SEED` = 1337.

`OqlTrainSpec` struct in `microgpt_oql.h` captures the parsed arguments. **One new production (`train_stmt` already exists from E07 as a stub; this phase adds the clause list); no new top-level verbs.** T1 (+6/-4 verb lock from E07) holds trivially.

**Phase 2 — Corpus first-class object (concurrent with Phase 1, ~3 days).**

OQL doesn't have a `CORPUS` object today. Add `CREATE CORPUS <name> FROM FILE <path>` as a new object type — note that this reuses the existing `CREATE` verb (does NOT add a 7th verb; same way `BEHAVIOUR` was a new object type, not a new verb in E08).

```c
typedef struct {
    char  name[OQL_MAX_NAME];
    char  file_path[OQL_MAX_PATH];
    /* lazy: read on first TRAIN reference */
    char *contents;
    size_t  contents_len;
} OqlCorpus;
```

**Phase 3 — Interpreter binding to `TrainWorker` (~1 week).**

Bind `oql_run_train(spec)` to the existing engine. The agent should *read* `src/microgpt.{h,c}` to find the actual `TrainWorker` API (pthread harness for training); the binding should not require any new engine surface.

Lifecycle expectations:
- `train_worker_init(model, corpus, hyperparameters)` → opaque handle
- `train_worker_step(handle)` → loss
- `train_worker_save(handle, path)` → checkpoint to disk
- `train_worker_destroy(handle)` → cleanup

If the existing C API doesn't fit this shape exactly, adapter layer in `src/oql_runtime_train.{c,h}` (new file) glues to whatever API does exist. **No new engine surface; no new VM opcodes.**

**Phase 4 — Loss-curve fidelity smoke test (~1 week).**

The smoke test uses one of the smallest existing demos (`names_demo`, ~4K params, < 1 second training) as the baseline. Two runs:

1. **Baseline:** `names_demo` with seed 1337, ~5000 steps.
2. **OQL:** A 20-line OQL script that does the same thing — `CREATE CORPUS`, `CREATE ORGANELLE`, `TRAIN ... STEPS 5000 LR ... SEED 1337 SAVE 'checkpoints/oql_names.ckpt'`.

Compare loss at steps 100, 500, 1000, 2500, 5000. Per-step delta must be ≤ ±10% relative (i.e. `|oql_loss - c_loss| / c_loss ≤ 0.10`). If the loss curves are bit-identical, even better — but ±10% is the locked floor because pthread scheduling and reduction order can introduce tiny float-summation differences.

**Phase 5 — Checkpoint round-trip test (~3 days).**

After TRAIN saves to `checkpoints/oql_names.ckpt`:

```sql
CREATE ORGANELLE loaded FROM CHECKPOINT 'checkpoints/oql_names.ckpt';
```

Then run inference. Expect: identical next-token distributions to the just-trained organelle (within `float` precision). This proves the save/load cycle is faithful.

**Phase 6 — Section 3 writeup**, mirroring E08's and E09's structure.

### 1.4 Pre-registered targets (locked)

| ID | Target | Floor (skip-rule trigger) |
|---|---|---|
| **T1** | TRAIN clause parses cleanly in OQL grammar with all 6 optional sub-clauses (`WITH ROLE` / `STEPS` / `LR` / `BATCH_SIZE` / `SAVE` / `SEED`) | Parse fails on any documented sub-clause |
| **T2** | TRAIN executes a names-corpus training run end-to-end without errors | Crash, hang, or unhandled error |
| **T3** | Loss-curve fidelity vs `names_demo` baseline: `|oql_loss - c_loss| / c_loss ≤ 0.10` at steps 100, 500, 1000, 2500, 5000 with identical seed | Delta > 0.25 at any measurement step |
| **T4** | Saved checkpoint round-trips: `CREATE ORGANELLE FROM CHECKPOINT '<saved>'` produces inference outputs within float precision of the just-trained organelle | Outputs diverge beyond `1e-5` per-logit |
| **T5** | Zero new VM opcodes (E08 T3-lock preserved) | `git diff main -- src/microgpt_vm.{h,c}` ≠ 0 lines |
| **T6** | E07/E08/E09 tests all pass unchanged (17/17 ctest including 22+ in `microgpt_oql_tests`) | Any regression |
| **T7** | Verb-discipline lock holds: TRAIN remains one verb; `CORPUS` is a new *object* via the existing `CREATE` verb; the +6/-4 surface lock from E07 is intact | A 7th top-level verb is needed |
| **T8** | OQL LOC for a minimal "train + save + load + infer" script ≤ 30 lines | > 50 lines |

The headline survives if **T1, T2, T3, T4, T5, T6, T7 all pass**. T8 is a usability backstop.

### 1.5 Skip rules

- **If T5 trips** (a new VM opcode is needed): STOP. Adding an opcode would warrant its own pre-reg. Document the gap.
- **If T3 trips above 0.25** (loss-curve divergence too large): the binding is dropping gradients or applying optimizer steps differently. Investigate before claiming any PASS.
- **If T7 trips** (a 7th verb is genuinely needed): STOP. E07's verb-discipline lock was the most important design call; do not break it for convenience.
- **If T4 trips** (round-trip diverges): checkpoint format is incompatible across the train/load boundary. Investigate — this should be solvable since the underlying engine has a single canonical format.

### 1.6 Falsification risk: Medium

| Risk | Likelihood | Mitigation |
|---|---|---|
| `TrainWorker` API doesn't cleanly expose init/step/save | Medium | Adapter layer in `oql_runtime_train.{c,h}`; do NOT modify `src/microgpt.{h,c}` |
| pthread scheduling causes loss-curve divergence > 10% | Medium | Use single-threaded mode for the smoke test; multi-threaded as a follow-up |
| Compile-time-macro silent failure (E09 §3.4 finding): training in one binary, loading checkpoint in another with different `N_EMBD`/`N_HEAD` | High (this *will* bite) | Reuse E09's `_microgpt_lib_for_defines` mechanism — `oql_names` binary variant with the names-demo's compile-time macros |
| Verb-discipline drift (someone wants `TRAIN ... USING optimizer = ...`) | Low | Hard-lock to existing engine optimizer (Adam); document; defer optimizer-choice DSL to E12+ if needed |
| Memory or pthread leaks under repeated TRAIN invocations | Medium | Add valgrind-clean test for at least 3 sequential TRAIN runs in one process |

### 1.7 What this experiment is NOT testing

- It is **NOT** training organelles to win-rate parity with the C demos on every game. That's E11 (Connect-4 win-rate gap closure, behaviour-side).
- It is **NOT** distributed training, GPU offload, gradient accumulation, learning-rate schedulers beyond the engine default. All deferred to E12+.
- It is **NOT** multi-organelle simultaneous training. Single-organelle TRAIN only.
- It is **NOT** introspection into training state mid-run (no `WATCH` verb or streaming loss). The verb-discipline lock keeps that out.
- It is **NOT** retraining the existing Connect-4 player to fix E09's T2. That's E11's job; orthogonal.

### 1.8 Cross-references

| Topic | Source |
|---|---|
| Parent (OQL grammar with TRAIN stub) | [E07](E07-oql-dsl.md) §1.3.1 |
| Parent (BEHAVIOUR + extern table) | [E08](E08-oql-behaviours.md) |
| Parent (RUN/COMPOSE/FROM CHECKPOINT wired) | [E09](E09-oql-runtime-wiring.md) |
| Engine-side training API | `src/microgpt.{h,c}` (the agent reads to find the canonical `TrainWorker` shape) |
| Compile-time-macro variant pattern (avoids the E09 §3.4 silent failure) | `_microgpt_lib_for_defines` in `CMakeLists.txt` |
| Baseline for loss-curve fidelity | `demos/character-level/names/main.c` |
| Companion experiment running in parallel | [E11](E11-connect4-win-rate-fix.md) — closing E09's T2 |

---

## 2. Initial state

### 2.1 What's currently known

- TRAIN is a stubbed verb (E07): grammar accepts the keyword; the runtime returns `OQL_ERR_NOT_IMPLEMENTED`. The `test_train_stub_still_honest` regression test from E09 covers the stub.
- The engine's training API exists and is well-tested across 11+ demos (every game demo trains inline).
- The names demo is the smallest reference baseline (~4K params, single-second training).
- `_microgpt_lib_for_defines` already supports per-binary compile-time-macro variants; E10 reuses this for the `oql_names` smoke-test binary.
- OQL's runtime registries (E09: `OqlOrganelle`, `OqlPipeline`) are already in place; adding `OqlCorpus` and `OqlTrainSpec` mirrors that structure.

### 2.2 Baselines to beat

| Baseline | Number | TRAIN must |
|---|---|---|
| `names_demo` loss at step 5000, seed 1337 | (TBD — measure in Phase 4) | match within ±10% |
| Checkpoint save/load round-trip | bit-identical (existing C demo behaviour) | per-logit within `1e-5` |
| OQL LOC for "train + save + load + infer" | infinity (impossible today) | ≤ 30 lines |
| Existing ctest pass count | 17/17 | hold at 17/17 |

### 2.3 Dependencies / blockers

- Existing `TrainWorker` API in `src/microgpt.{h,c}` — must be readable; if its shape doesn't match Phase 3's expected lifecycle, the adapter glue in `oql_runtime_train.{c,h}` absorbs the mismatch without modifying the engine.
- Reference to `names_demo` as the loss-curve baseline.
- A trained `c4_player.ckpt` is NOT needed by E10 — that's E09's territory.

### 2.4 What this experiment deliberately does NOT do

- Does NOT add a 7th OQL verb. `CORPUS` is a new object via `CREATE`, the same pattern as `BEHAVIOUR` in E08.
- Does NOT add VM opcodes. T5 hard-locks this.
- Does NOT change the optimizer (stays Adam).
- Does NOT make training distributed.
- Does NOT touch `src/microgpt.{h,c}`. All glue lives in `src/oql_runtime_train.{c,h}` (new) plus `microgpt_oql.{l,y,h,c}` extensions.

---

## 3. Implementation + results

**TODO** — fill on measurement commit. Sections to populate:

- 3.1 Grammar diff: `microgpt_oql.{l,y}` additions for the `TRAIN` clause and `CREATE CORPUS` object
- 3.2 `OqlTrainSpec`, `OqlCorpus`, runtime registry additions
- 3.3 `oql_runtime_train.{c,h}` adapter to `TrainWorker`
- 3.4 Loss-curve fidelity measurement (T3) — per-step delta table
- 3.5 Checkpoint round-trip results (T4) — per-logit divergence summary
- 3.6 VM opcode diff confirmation (T5)
- 3.7 Test regression confirmation (T6)
- 3.8 Verb-lock confirmation (T7)
- 3.9 Smoke-test OQL script (T8)
- 3.10 Per-target verdict matrix

---

## 4. Conclusion

**TODO** — fill on measurement commit when all 8 targets are measured. Sections to populate:

- 4.1 Verdict per T1-T8 (PASS / FAIL / FLOOR-TRIGGER)
- 4.2 Compound benefits realised:
  - The four E07-stubbed verbs are now all wired (TRAIN + the three E09 closed)
  - "Published experiment as a single OQL file" claim becomes literal
  - E01's System C authoring is fully OQL-native
- 4.3 What's NOT done (multi-organelle TRAIN, distributed training, optimizer DSL — deferred to E12+)
- 4.4 If T3 fidelity was tight (< 5%): consider promoting OQL TRAIN as a *replacement* for the C demos' inline training, not just an alternative
- 4.5 Traceability updates (`TRACEABILITY.md`, `ORGANELLE_STATE.md`, `RESEARCH_DISCLOSURE.md`)
