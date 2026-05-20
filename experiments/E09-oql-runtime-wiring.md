# Experiment E09 — Wire OQL `RUN` / `COMPOSE` / `CREATE ORGANELLE FROM CHECKPOINT` end-to-end so `connect4.oql` drives a real game loop

**Status:** 📋 Proposal locked — 2026-05-20.
**Direction:** close the gap E08 left open. E07 shipped the OQL grammar with `TRAIN` / `COMPOSE` / `RUN` / `EVALUATE` returning `OQL_ERR_NOT_IMPLEMENTED`; E08 shipped `BEHAVIOUR` objects with VM TypeScript bodies. E09 wires the runtime so a researcher can type `oql run experiments/connect4.oql` and watch a complete game play out.
**Cost estimate:** ~5-7 weeks (1 wk CREATE ORGANELLE FROM CHECKPOINT + 1 wk COMPOSE storage + 2 wk RUN game-loop dispatch + 1 wk Connect-4 measurement + 1-2 wk replication / writeup).
**Falsification risk:** Medium — depends on whether the existing organelle / pipeline / VM APIs cleanly compose under one orchestrator without forcing structural changes to the engine.

---

## Spear summary

**Point:** After E07 + E08 the OQL operator surface *exists* but nothing actually *runs* — every interesting verb (`RUN`, `COMPOSE`, `TRAIN`, `EVALUATE`) returns `OQL_ERR_NOT_IMPLEMENTED`. `experiments/connect4.oql` parses cleanly and individual behaviours dispatch correctly under a unit-test harness, but the file as a whole does not drive a game. E09 closes that gap by wiring three of the four stubbed verbs (`RUN`, `COMPOSE`, `CREATE ORGANELLE FROM CHECKPOINT`) — deferring `TRAIN` to a follow-up — so the project's "the language *is* the product" thesis becomes testable end-to-end.

**Picture:** A researcher clones the repo, writes a `.oql` file, types `oql run my_experiment.oql`, and sees a real game play out with audit-trail traces, per-move latency, and a final win-rate number. Today: impossible (only `VERIFY` and `AUDIT` execute). After E09: yes, for `connect4.oql`. After E09 + replication: yes, for any of the 11 game families.

**Proof (to be measured):** Connect-4 via OQL holds the C-demo baseline win rate within ±3 pp (target ≥ 85%, floor 80%); end-to-end latency p99 ≤ 50 ms per move; the existing E07 + E08 tests pass unchanged; **`TRAIN` is honestly deferred** with a single test that asserts its stub is preserved.

**Push:** This is the experiment that unlocks **E08's T1 measurement** (which was gated on `RUN` being wired) and indirectly unblocks **E01's System C path** (the OPA system in the head-to-head comparison was always going to be authored as an OQL spec; without `RUN` wired, that authoring is moot).

---

## 1. Proposal

### 1.1 Hypothesis (locked before measurement)

> *Wiring three of OQL's four stubbed verbs — `CREATE ORGANELLE FROM CHECKPOINT`, `COMPOSE … AS @graph …`, and `RUN <pipeline> ON game_loop WITH (opponent = …, games = N) RETURNING (…)` — through the existing engine APIs (`microgpt_load_checkpoint`, `pipeline_parse_text` / `pipeline_verify`, the wiring binary's vote-loop pattern) plus E08's `BEHAVIOUR` dispatch is sufficient to run `experiments/connect4.oql` end-to-end with: (a) Connect-4 win rate within ±3 pp of the C-demo baseline (target ≥ 85%; floor 80%); (b) per-move latency p99 ≤ 50 ms on M2 Max; (c) zero regressions on existing tests; (d) zero new VM opcodes (T3-lock from E08 preserved); (e) `TRAIN` honestly stubbed with a regression test asserting the stub message.*

### 1.2 Why this matters

E08 measured 3 of 8 targets honestly and **deferred 5 with concrete reasons** — chief among them T1 (Connect-4 win rate via OQL+TS). The deferral language is honest but it leaves OQL as a parse-only DSL: parses cleanly, dispatches individual behaviours correctly under unit tests, but does not actually run an experiment. Without `RUN` wired, the "the query language *is* the product" framing from EQL doesn't apply yet — the product is *almost* the query language.

E09 is the single move that closes the gap. Once it lands:

1. **E08's T1 / T5 / T8 become measurable** — game-loop win rate, replication win rates, end-to-end latency.
2. **E01's System C becomes a real OQL spec** — System B (LLM + IR verifier) is already a 50-LOC bridge over `libpipeline_ir`; System C will be a small OQL file once `RUN` is wired.
3. **The reproducibility-by-construction claim from the OQL pitch becomes true** — every future published experiment can ship as a single `.oql` file + a corpus reference + an engine version pin.

### 1.3 Mechanism

**Phase 1 — `CREATE ORGANELLE FROM CHECKPOINT` (1 week).**

Extend `src/microgpt_oql.{h,c}` with:

```c
typedef struct {
    char  name[64];
    char  checkpoint_path[256];
    int   loaded;                /* lazy: load on first RUN reference */
    void *model;                 /* opaque; cast to microgpt model handle */
    /* behaviour bindings populated by `WITH (..._BEHAVIOUR = ...)` */
    OqlBehaviour *input_behaviour;
    OqlBehaviour *output_behaviour;
    OqlBehaviour *validate_behaviour;
    OqlBehaviour *fallback_behaviour;
    OqlBehaviour *score_behaviour;
    OqlBehaviour *cycle_detect_behaviour;
} OqlOrganelle;

typedef struct {
    OqlOrganelle entries[OQL_MAX_ORGANELLES];
    int          count;
} OqlOrganelleTable;
```

The interpreter registers the organelle name and defers checkpoint loading until first reference under `RUN`. Loading binds to `microgpt_load_checkpoint()` from the existing engine.

**Phase 2 — `COMPOSE … AS @graph …` (1 week).**

```c
typedef struct {
    char         name[64];
    Pipeline    *ir;             /* parsed via pipeline_parse_text() */
    /* edge-to-organelle bindings: which OqlOrganelle handles each `call(...)` node */
    OqlOrganelle *call_bindings[PIPELINE_MAX_NODES];
} OqlPipeline;
```

`COMPOSE` parses the inline `@graph…@end` body via `pipeline_parse_text()`, verifies it via `pipeline_verify()`, and resolves each `call(<organelle_name>, …)` node by lookup into the organelle table. Failed lookups produce a parse-time error.

**Phase 3 — `RUN <pipeline> ON game_loop WITH (…) RETURNING (…)` (2 weeks).**

```c
OqlResult oql_run_game_loop(
    OqlPipeline *pipeline,
    const char  *opponent,         /* "random" for v1; future: another organelle */
    int          games,
    OqlMetric    metrics_mask      /* win_rate | p99_latency_ms | audit_coverage */
);
```

The game loop walks the IR in topo order per game. For each `call(organelle, in)` node:

1. Dispatch `INPUT_BEHAVIOUR(in)` via the VM — produces the organelle's input tokens.
2. Run forward inference on the organelle's model — produces a next-token logit distribution.
3. Sample / argmax → token.
4. Dispatch `OUTPUT_BEHAVIOUR(token)` via the VM — produces the move string.
5. Dispatch `VALIDATE_BEHAVIOUR(board, move)` — if false, dispatch `FALLBACK_BEHAVIOUR(board)`.
6. Apply the move to the game state (game-specific C harness — for Connect-4, the same logic the existing `demos/character-level/connect4/main.c` uses, lifted into a reusable `oql_runtime_games.{c,h}`).
7. Opponent moves (random for v1).
8. Repeat until terminal; record outcome.

Returning columns are accumulated across all games and emitted as a final row.

**Phase 4 — Connect-4 measurement (1 week).**

Requires a trained `checkpoints/c4_player.ckpt`. Two pathways:

- **Pathway A (preferred):** existing `connect4_demo` already trains inline. Add a CLI flag `--save-checkpoint=path` to that demo (small surgical change), train once, save. Use the saved checkpoint for OQL.
- **Pathway B (fallback):** if Pathway A's surgical change is non-trivial, ship a small standalone tool `tools/train_c4_player.c` that trains the same architecture and writes a checkpoint. This duplicates ~30 LOC but is reversible.

Once a checkpoint exists, run `oql run experiments/connect4.oql` and measure: win rate over 100 games vs random opponent (matches the C-demo baseline conditions).

**Phase 5 — Replication (deferred to follow-up):** Mastermind, Pentago, 8-puzzle. Not in this run; closes E08's T5.

### 1.4 Pre-registered targets (locked)

| ID | Target | Floor (skip-rule trigger) |
|---|---|---|
| **T1** | `oql run experiments/connect4.oql` completes 100 games without error | Any unhandled error / crash |
| **T2** | Connect-4 win rate via OQL ≥ 85% (matches E08's deferred T1; vs C-demo 88% baseline) | < 80% |
| **T3** | Per-move latency p99 ≤ 50 ms on M2 Max (5× the C demo's order; allows VM dispatch overhead) | > 200 ms |
| **T4** | All existing `test_microgpt_oql` tests pass (E07's VERIFY/AUDIT + E08's BEHAVIOUR dispatch) | Any regression |
| **T5** | All existing `test_microgpt_vm` tests pass; **zero new VM opcodes** (E08's T3-lock preserved) | Any opcode addition |
| **T6** | `TRAIN` returns its existing `OQL_ERR_NOT_IMPLEMENTED` honestly; a new test `test_train_stub_still_honest` asserts this | TRAIN silently no-ops or partially-implements |
| **T7** | OQL+TS line count for Connect-4 stays ≤ 30% of original C demo (re-measure after RUN wiring) | > 50% |
| **T8** | Audit-coverage: every `RUN` produces an audit trace per move (`SELECT * FROM run_traces WHERE pipeline = 'connect4'` returns ≥ games × moves rows) | < 100% |

The headline survives if **T1, T2, T4, T5 all pass**. T3 / T6 / T7 / T8 are usability and discipline backstops.

### 1.5 Skip rules

- **If T1 fails** (RUN crashes / errors): document the API gap. **Do not** ship a partial RUN that papers over the error — fix the underlying API or split the work.
- **If T2 falls below 80%** (win-rate regression vs C demo): the OQL+behaviour indirection is too lossy. Document the per-component latency / accuracy budget; check whether `INPUT_BEHAVIOUR` is failing to feed the model correctly, whether `FALLBACK_BEHAVIOUR` is firing too aggressively, etc.
- **If T5 trips** (new VM opcode required by `RUN`'s dispatch): STOP. Adding a VM opcode breaks E08's hard-lock and warrants its own pre-reg. Document the gap; do not silently expand.
- **If T6 trips** (TRAIN gets partially implemented): STOP. Partial TRAIN that quietly succeeds for some configurations and fails for others is *worse* than a clean stub. If TRAIN-wiring becomes tractable mid-run, split it into a separate experiment (E10).

### 1.6 Falsification risk: Medium

| Risk | Likelihood | Mitigation |
|---|---|---|
| The existing C-demo game loop logic doesn't lift cleanly into a reusable harness | Medium | Phase 4 budget includes the extraction; if extraction is hard, defer to Phase 5 (replication) |
| Checkpoint format incompatibility between training-side and OQL-side load | Low | Same engine, same `microgpt_load_checkpoint`; tested via existing demos that save/load |
| `INPUT_BEHAVIOUR` → forward-inference → `OUTPUT_BEHAVIOUR` chain produces tokens that don't match the C demo's flow | Medium-high | Compare token-by-token against the C demo on a held-out 5-game trace; if divergence, fix the BEHAVIOUR bodies (these are TS, easy to debug) before claiming T2 measurement |
| Game-loop latency exceeds 50 ms p99 | Low-medium | VM dispatch is ~3.7M ops/sec; ~50 ops per move; budget well-met. If exceeded, investigate `pipeline_verify()` calls per move (should be 0 — verifier runs once at COMPOSE time) |
| RUN-side state management bugs (organelle inference state leaking across games) | Medium | Existing `microgpt` engine has KV-cache reset APIs; call them per game. Add a unit test that two consecutive `RUN`s produce identical results when re-seeded |

### 1.7 What this experiment is NOT testing

- It is **NOT** testing whether OQL replaces C for *training* — TRAIN is explicitly deferred to a follow-up (E10).
- It is **NOT** testing whether `RUN` generalises beyond `ON game_loop` — `ON file` / `ON streaming` / `ON kafka` modifiers are future work.
- It is **NOT** testing checkpoint format portability — same engine, same format.
- It is **NOT** measuring "researcher onboarding time" — that was E08's T6, gated on humans.
- It is **NOT** an attempt to make the wiring binary faster — the engine's hot path is unchanged.

### 1.8 Cross-references

| Topic | Source |
|---|---|
| Parent (OQL grammar) | [E07](E07-oql-dsl.md) |
| Parent (BEHAVIOUR objects + VM extern table) | [E08](E08-oql-behaviours.md) |
| The Connect-4 OQL spec being driven | [`experiments/connect4.oql`](connect4.oql) |
| The C demo being matched | [`demos/character-level/connect4/main.c`](../demos/character-level/connect4/main.c) |
| Existing engine APIs being bound | `microgpt_load_checkpoint`, `microgpt_forward_inference`, KV-cache reset in `src/microgpt.h` |
| Pipeline IR substrate already merged via E02 | [`libs/pipeline_ir/include/pipeline_ir/pipeline_ir.h`](../libs/pipeline_ir/include/pipeline_ir/pipeline_ir.h) |
| VM-side TS dialect for behaviour dispatch | `src/microgpt_vm.{l,y,h,c}`, `src/microgpt_vm_natives.{h,c}` |
| Behaviour catalogue mapping six concerns | [`docs/research/BEHAVIOUR_CATALOGUE.md`](../docs/research/BEHAVIOUR_CATALOGUE.md) |
| Verb-discipline lock that must hold | [E07](E07-oql-dsl.md) §1.3.1 (+6 / -4) |
| E08's deferred T1 that this closes | [E08](E08-oql-behaviours.md) §3.4 |

---

## 2. Initial state

### 2.1 What's currently known

- OQL parses `connect4.oql` cleanly (E08, `tests/test_microgpt_oql.c::test_e08_connect4_behaviours`).
- Each of the four Connect-4 behaviours dispatches correctly under a hand-built game state (E08).
- `VERIFY GRAPH` and `AUDIT` execute end-to-end (E07).
- `TRAIN` / `COMPOSE` / `RUN` / `EVALUATE` return `OQL_ERR_NOT_IMPLEMENTED` (E07, intentional honest stubs).
- The C demo Connect-4 baseline is 88% win rate vs random over 100 games.
- The existing wiring binary (`demos/wiring_organelle/main.c`) has a vote-loop pattern that's the closest existing analog to what `RUN` needs.
- KV-cache reset between games is a known engine concern; the existing demos handle it inline.

### 2.2 Baselines to beat

| Baseline | Number | OQL must |
|---|---|---|
| Connect-4 win rate (C demo) | 88% | hold within ±3 pp; floor 80% |
| Connect-4 per-move latency (C demo, M2 Max) | ~5-10 ms | ≤ 50 ms (5× allowance for VM dispatch + IR walk) |
| C demo source-line cost | ~500 LOC | already at 98 lines for `connect4.oql` (E08 T2); E09 must not push this past 150 |

### 2.3 Dependencies / blockers

- **A trained `checkpoints/c4_player.ckpt`** — does not exist on main. See Phase 4 pathway choice.
- **A trained `checkpoints/c4_planner.ckpt`** — same.
- **Engine API surface** — `microgpt_load_checkpoint`, forward inference, KV-cache reset all exist; some adapter glue likely needed for the OQL runtime's organelle lifecycle.
- **Pipeline IR call-node resolution** — the IR's `call(<name>, …)` node-text grammar already exists; resolution from name → `OqlOrganelle*` is new but straightforward.

### 2.4 What this experiment deliberately does NOT do

- Does NOT wire `TRAIN`. Skip-rule T6 hard-locks this.
- Does NOT add a new VM opcode. T5-lock preserves E08's discipline.
- Does NOT redesign the engine's forward-inference API. Adapter layer only.
- Does NOT introduce a new build dep. C99 + Flex/Bison fallback + libc/libm only.
- Does NOT touch the existing C demo. The C demo stays as the baseline / fallback authoring path; OQL is the new path that researchers can choose.
- Does NOT replicate to Mastermind / Pentago / 8-puzzle. That's E08 T5, deferred to a follow-up after E09 measures Connect-4.

---

## 3. Implementation + results

**TODO** — fill on measurement commit. Sections to populate:

- 3.1 `CREATE ORGANELLE FROM CHECKPOINT` integration: OqlOrganelle struct, organelle table, lazy-load semantics
- 3.2 `COMPOSE … AS @graph …` integration: OqlPipeline struct, call-node resolution, parse-time verification
- 3.3 `RUN <pipeline> ON game_loop WITH (…) RETURNING (…)` integration: game loop, behaviour dispatch order, opponent driver, metric accumulation
- 3.4 Connect-4 measurement: Pathway A or B for checkpoint creation; final win rate over 100 games; per-move latency distribution
- 3.5 Audit-trace coverage measurement (T8)
- 3.6 VM opcode diff confirmation (T5)
- 3.7 TRAIN stub regression test (T6)
- 3.8 Per-target verdict matrix (T1-T8)

---

## 4. Conclusion

**TODO** — fill on measurement commit, when ALL eight targets are measured. Sections to populate:

- 4.1 Verdict per T1-T8 (PASS / FAIL / FLOOR-TRIGGER)
- 4.2 Headline outcome — does `oql run` actually drive a game?
- 4.3 Compound benefits realised:
  - E08's T1 (Connect-4 win rate via OQL+TS) now MEASURED, not deferred
  - E01's System C authoring cost collapsed to a small OQL file
  - The "single OQL file reproduces a published experiment" claim becomes true
- 4.4 What's NOT done — TRAIN, EVALUATE-on-file, replication across other games
- 4.5 Next experiment (E10): wire TRAIN, with locked targets around training-loop fidelity vs C-demo baseline
- 4.6 Traceability updates (`TRACEABILITY.md`, `ORGANELLE_STATE.md`, `RESEARCH_DISCLOSURE.md`)
