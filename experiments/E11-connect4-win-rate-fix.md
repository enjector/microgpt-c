# Experiment E11 — Close E09's T2 win-rate gap: fix the INPUT_BEHAVIOUR prompt protocol so OQL Connect-4 matches the C demo's 88% baseline

**Status:** 📋 Proposal locked — 2026-05-20.
**Direction:** close E09's PARTIAL T2 measurement (51% win rate vs 85% target). The wiring is correct end-to-end — the gap is a prompt-protocol mismatch at the `INPUT_BEHAVIOUR` step.
**Cost estimate:** ~2-3 weeks (1 wk diagnosis + 1 wk implementation + 1 wk re-measurement + writeup).
**Falsification risk:** Low-medium — the C-demo baseline is well-defined and reproducible; the diagnosis is narrowly scoped to the prompt-construction logic.

---

## Spear summary

**Point:** E09 demonstrated that OQL can drive a 100-game Connect-4 loop end-to-end (T1 PASS) but the measured win rate was **51%** against the C demo's **88%**. E09's Section 3.4 named two closure paths: fix `parse_c4_board` to reproduce the C demo's corpus-encoded prompt protocol, OR add a `model_sample_from_logits(top_k, temperature)` extern to the VM. E11 picks one and ships it.

**Picture:** The OQL `parse_c4_board` BEHAVIOUR currently produces a board representation that's *correct* (cells encoded as 1/-1/0) but *different* from what the C demo's `c4_player` was trained against. The C demo trained on a specific corpus-encoded prompt format; OQL gets the wiring right but feeds the model a representation it never saw during training. The win rate gap is the cost of that representation mismatch.

**Proof (to be measured):** post-fix Connect-4 win rate via OQL `oql_c4` binary ≥ 85% (matches E08's deferred T1 / E09's locked T2); existing tests pass; no engine surface changes; behaviour LOC stays bounded.

**Push:** This is the experiment that converts E09's PARTIAL into a PASS — the headline measurement for "OQL produces results competitive with hand-coded C demos."

---

## 1. Proposal

### 1.1 Hypothesis (locked before measurement)

> *Closing E09's Connect-4 T2 win-rate gap (51% measured vs ≥ 85% target) is achievable by one of two pathways, each with a falsifiable target: **Pathway A (behaviour-side):** modify `parse_c4_board` in `experiments/connect4.oql` to reproduce the C demo's corpus-encoded prompt-construction logic (board string → exact training-time prompt format), with **zero engine surface changes and zero new VM opcodes**, achieves Connect-4 win rate ≥ 85% vs random opponent over 100 games on the existing `c4_player.ckpt`. **Pathway B (extern-side):** add a single new VM extern `model_sample_from_logits(top_k, temperature)` (NOT a new VM opcode — an extern table entry in `src/microgpt_vm_natives.{h,c}`) so the behaviour author can directly control sampling, also achieves Connect-4 win rate ≥ 85% on the same conditions.*

The agent picks one pathway (preferred: A, smaller surface) and ships the chosen one. If A fails to clear 85%, the agent honestly reports and either falls back to B or stops with a clean partial.

### 1.2 Why this matters

E09 was a structural win — it proved OQL can *drive* a game loop. But its headline measurement (T2 PARTIAL at 51%) is the thing reviewers, the project's own README, and any potential adopters will look at first. A 51% win rate against a random opponent is *barely above coin-flip* — even though the wiring is correct, the *advertised number* doesn't yet justify the architecture's existence on Connect-4.

After E11:
1. **E09's PARTIAL becomes PASS.** The headline number lifts from 51% → ≥ 85%.
2. **E08's deferred T1 closes** (transitively — it was waiting on E09's T2).
3. **The "OQL is competitive with hand-coded C" claim becomes measurable, not aspirational.**
4. **E10's TRAIN wiring** (running in parallel) becomes more useful — `oql run my_experiment.oql` produces results comparable to the C demos rather than degraded versions.

### 1.3 Mechanism

**Phase 1 — Diagnosis (~3-5 days).**

Read in parallel:

- `demos/character-level/connect4/main.c` — find the exact prompt construction (look for `snprintf`-like board-to-string formatting, the corpus generation in `gen_connect4_corpus()`-style functions, and `tokenize`-style boundaries).
- `experiments/connect4.oql` — the current `parse_c4_board` BEHAVIOUR. Identify what it produces vs what the C demo would feed.
- The Connect-4 training corpus generation logic — *this is the ground truth* the trained model expects.

Produce a side-by-side comparison: for a fixed board state, the byte-exact prompt the C demo produces vs the byte-exact tokens the OQL behaviour produces. The diff is the closure target.

**Phase 2 — Pick a pathway (~1 day).**

Decision rule:
- **Pathway A (behaviour-side fix) if** the C demo's prompt protocol can be reproduced in ≤ 80 lines of TS in the existing OQL behaviour grammar without new extern calls. Preferred.
- **Pathway B (extern-side) if** A requires VM-side support that doesn't exist (e.g. the C demo uses a custom tokenizer state that's not exposed). Adds ONE extern (`model_sample_from_logits` OR equivalent) to `src/microgpt_vm_natives.{h,c}`. **NO new VM opcodes** — extern table entry only, mirroring the E08 discipline.

If both pathways are tractable, prefer A — smaller surface, no engine surface change.

**Phase 3 — Implement chosen pathway (~1 week).**

For Pathway A: update `parse_c4_board` body in `experiments/connect4.oql`. The TS source string in the OQL file is what gets compiled by the VM at COMPOSE time. Optionally also update `format_c4_move` if the C demo's output decode differs.

For Pathway B: add the extern to `microgpt_vm_natives.{h,c}`; add a TS test resource exercising it; update `parse_c4_board` (or add a new behaviour) to use it.

**Phase 4 — Re-measure (~1 week).**

Run `oql run experiments/connect4.oql` end-to-end via the existing `./build/oql_c4` binary, against the existing `checkpoints/c4_player.ckpt`. Measure:

1. Win rate over 100 games vs random opponent (same conditions as the C demo's 88% baseline and E09's 51% measurement).
2. Per-move latency p99 (must remain ≤ 50 ms, matching E09's T3).
3. Token-level trace divergence vs C demo on first 5 moves of 10 fixed-seed games.

**Phase 5 — Section 3 writeup**, mirroring E09's structure with explicit before/after numbers.

### 1.4 Pre-registered targets (locked)

| ID | Target | Floor (skip-rule trigger) |
|---|---|---|
| **T1** | Connect-4 win rate via OQL ≥ 85% over 100 games vs random opponent (matches E09's locked T2) | < 80% |
| **T2** | All existing tests pass; no regressions on E07/E08/E09 | Any regression |
| **T3** | Latency p99 per move ≤ 50 ms (E09 baseline held) | > 100 ms |
| **T4** | Token-level trace divergence vs C demo: similarity ≥ 80% on first 5 moves of 10 fixed-seed games | < 60% similarity |
| **T5** | Zero new VM opcodes (E08 hard-lock preserved) | Any new opcode |
| **T6** | If Pathway A: zero new VM externs. If Pathway B: exactly ONE new extern with a clear name and a unit test | More than one new extern, or extern added without test |
| **T7** | OQL behaviour LOC bounded: `parse_c4_board` body ≤ 80 lines TS for Pathway A; ≤ 50 lines for Pathway B | > 100 lines either pathway |

The headline survives if **T1, T2, T5 all pass**. T3/T4 are quality backstops. T6/T7 are discipline floors.

### 1.5 Skip rules

- **If T1 falls below 80%** (floor): the chosen pathway is insufficient. Document the gap precisely; if Pathway A was chosen, try Pathway B; if both fail, the win-rate gap is deeper than a prompt-protocol issue and a different experiment is needed.
- **If T5 trips** (new VM opcode required): STOP. The E08 hard-lock is non-negotiable. Adding an opcode warrants its own pre-reg.
- **If T6 trips on Pathway B** (more than one extern): the pathway is over-scoped. Investigate; a multi-extern fix suggests a different abstraction (e.g. a sampling-strategy object) that warrants its own experiment.
- **If T2 trips** (regression): diagnose; do not weaken any existing test to pass.

### 1.6 Falsification risk: Low-medium

| Risk | Likelihood | Mitigation |
|---|---|---|
| C demo's prompt protocol is too coupled to the C runtime to express cleanly in OQL+TS | Medium | Pathway B fallback exists; agent documents the specific coupling honestly |
| The training corpus itself encodes information that no behaviour can reproduce at inference time (e.g. game history, opponent info beyond board state) | Low-medium | Phase 1 diagnosis surfaces this; if true, the gap is *architectural*, not behaviour-side, and a different experiment is needed |
| Win rate hits 78% (close to but below 85%) | Medium | Reports honestly; consider whether sampling temperature or top-k tuning closes the last few points; if not, PARTIAL is the honest verdict |
| Latency regression from extra TS work in `parse_c4_board` | Low | Behaviour bodies dispatch at VM speed (3.7-5.8M ops/sec); 80 LOC of board parsing is microseconds |
| Pathway B requires touching `src/microgpt.{h,c}` to expose softmax/logits | Low | If true, document and switch to Pathway A even if A is harder; engine surface must stay frozen |

### 1.7 What this experiment is NOT testing

- It is **NOT** retraining `c4_player.ckpt`. Uses the existing trained checkpoint from E09. Retraining belongs to E10 (running in parallel).
- It is **NOT** replicating to Mastermind / Pentago / 8-puzzle. That's E08's deferred T5, separate experiment.
- It is **NOT** improving Connect-4 win rate *above* the C demo's 88%. Parity is the headline; surpassing is a different experiment.
- It is **NOT** changing the wiring loop, the cycle detector, or the OPA kanban. These were validated in E09's T1 PASS.
- It is **NOT** adding new OQL verbs or new object types. Pure behaviour-body or extern-table work.

### 1.8 Cross-references

| Topic | Source |
|---|---|
| The PARTIAL being closed | [E09](E09-oql-runtime-wiring.md) §3 T2 measurement (51%) |
| The C demo baseline | `demos/character-level/connect4/main.c` (88% win rate vs random) |
| The behaviour grammar being extended | [E08](E08-oql-behaviours.md) — `CREATE BEHAVIOUR ... AS VM \`...\`` |
| The extern-table mechanism (Pathway B) | `src/microgpt_vm_natives.{h,c}` |
| The OQL spec being modified | `experiments/connect4.oql` |
| The runtime that re-measures | `./build/oql_c4` from E09 |
| The verb-discipline lock that must hold | [E07](E07-oql-dsl.md) §1.3.1 |
| The opcode-discipline lock that must hold | [E08](E08-oql-behaviours.md) T3 |
| Companion experiment running in parallel | [E10](E10-oql-train-wiring.md) — wiring TRAIN |

---

## 2. Initial state

### 2.1 What's currently known

- E09's measurement: 51% Connect-4 win rate via OQL+TS over 100 games vs random opponent. The wiring is correct end-to-end (T1 PASS); the gap is at the prompt/sampling layer.
- E09 §3.4 finding: compile-time-macro silent failure mode if checkpoint and runtime macros mismatch. The `oql_c4` binary variant (built against Connect-4-dims `microgpt_lib`) is the canonical runtime for this measurement.
- C demo baseline: 88% win rate against random over 100 games on M2 Max.
- Existing `parse_c4_board` BEHAVIOUR body: produces correct cell encoding (1/-1/0) but not necessarily the C demo's exact prompt format.
- VM dispatch rate: 3.7-5.8M ops/sec — even an 80-line behaviour body is microseconds per call.

### 2.2 Baselines to beat

| Baseline | Number | OQL post-E11 must |
|---|---|---|
| C demo Connect-4 win rate | 88% | match within ±3 pp (target ≥ 85%; floor 80%) |
| C demo latency p99 | ~5-10 ms | ≤ 50 ms (E09 baseline held) |
| Existing test pass count | 17/17 ctest, 22 in oql tests | hold |
| Number of VM opcodes | (current) | hold (T5 lock) |

### 2.3 Dependencies / blockers

- `c4_player.ckpt` exists on main from E09 (Pathway A). E11 uses it as-is.
- `oql_c4` binary already built; E11 runs against it.
- Pathway A requires reading the C demo's prompt construction carefully — diagnostic-heavy phase.
- Pathway B requires adding a single extern table entry; precedent: E08 added 8 externs in `microgpt_vm_natives.{h,c}`.

### 2.4 What this experiment deliberately does NOT do

- Does NOT modify `src/microgpt.{h,c}`. Engine surface frozen.
- Does NOT add VM opcodes (T5 hard-lock).
- Does NOT add OQL verbs (E07 +6/-4 lock).
- Does NOT touch the existing C demo. The C demo stays as the baseline target.
- Does NOT retrain any checkpoint. (That's E10.)
- Does NOT touch other game demos.

---

## 3. Implementation + results

**TODO** — fill on measurement commit. Sections to populate:

- 3.1 Diagnostic: byte-exact comparison of C demo prompt vs current OQL behaviour output for a fixed board state
- 3.2 Pathway chosen (A or B) with justification
- 3.3 Implementation diff: `experiments/connect4.oql` behaviour body changes (Pathway A) and/or extern table addition (Pathway B)
- 3.4 Win-rate measurement: 100 games vs random; before/after delta
- 3.5 Latency measurement: per-move p99 (held within E09's budget)
- 3.6 Token-level trace divergence: per-game first-5-moves similarity vs C demo
- 3.7 VM opcode diff confirmation (T5)
- 3.8 Per-target verdict matrix

---

## 4. Conclusion

**TODO** — fill on measurement commit when all 7 targets are measured. Sections to populate:

- 4.1 Verdict per T1-T7 (PASS / FAIL / FLOOR-TRIGGER)
- 4.2 Compound benefits:
  - E09's T2 PARTIAL → PASS (headline closure)
  - E08's deferred T1 closes transitively
  - The "OQL competitive with C demos" claim becomes measurable, not aspirational
- 4.3 What's NOT done: replication across other games (E08's deferred T5)
- 4.4 If Pathway A succeeded: the architectural lesson is "behaviour-side fidelity is enough; engine surface stays frozen." If Pathway B was needed: the architectural lesson is "behaviour authors need direct sampling control; the extern table is the right place for it."
- 4.5 Traceability updates (`TRACEABILITY.md`, `ORGANELLE_STATE.md`, `RESEARCH_DISCLOSURE.md`)
