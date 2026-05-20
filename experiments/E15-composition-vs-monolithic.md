# Experiment E15 — Does coordination beat capacity at equal budget? Composition-vs-monolithic on hard-search problems, with corpus generation via a new OQL `FROM ORACLE` source

**Status:** 📋 Proposal locked — 2026-05-20.
**Direction:** the project's deepest unmeasured claim — *"coordination is the intelligence"* — has been *asserted* and *internally validated* (the architecture works reliably) but never *compared to a same-budget monolithic alternative on the same task with the same corpus*. E15 closes that gap on hard-search problems where the search depth exceeds what a single forward pass can hold.
**Cost estimate:** ~4-6 weeks (1 wk task framing + corpus generator + 1 wk monolithic baseline + 1 wk OPA composition + 1 wk evaluation harness + 1 wk writeup; 1 wk slack for the deepest experiment we've run).
**Falsification risk:** **HIGH — and that's the point.** This is the experiment that either validates or falsifies OPA's founding thesis against the right control.

---

## 0. The input/output search-space taxonomy — what is OPA actually for?

Before any pre-reg targets, we need to be honest about *what problems OPA is supposed to solve well* and what makes that distinct from "any transformer can solve this." The project has eight measured experiments of substrate work but has never written down the **criteria a problem must satisfy** for OPA's architecture to plausibly add value over a same-budget monolithic transformer.

This is the missing framing. Here it is:

### 0.1 Five criteria a problem must satisfy for OPA to plausibly outperform

A problem is in OPA's natural zone iff **ALL FIVE** hold:

| Criterion | Why it matters for OPA |
|---|---|
| **C1. The output is a structured object** (graph, program, move sequence, proof, plan), not a single token or a free-form text completion | OPA's coordination produces output structure piece-by-piece; if the output is unstructured, a single transformer forward pass has no compositional advantage to lose |
| **C2. The output is deterministically verifiable** — compiler, theorem prover, game-rule simulator, type checker | OPA's "Judge" is a verifier; without one, the architecture's filter step is decorative |
| **C3. The search/reasoning depth exceeds what a single forward pass can hold** — long move sequences, multi-step planning, iterative refinement | If a single forward pass can emit the whole answer, capacity wins over coordination by construction |
| **C4. State must accumulate across steps** — visited positions, blocked moves, candidate refinements | OPA's `OpaKanban` exists to externalize this state; if the problem is stateless single-shot, kanban is dead weight |
| **C5. Failure modes are detectable and recoverable** — cycle detection, dead-end backtracking, candidate re-ranking | OPA's deterministic infrastructure (cycle detector, re-ranking, fallback) does the heavy lifting; if failure is silent, the infrastructure adds nothing |

### 0.2 Tasks in OPA's natural zone (all 5 criteria satisfied)

- Search puzzles (8-puzzle, 15-puzzle, Klotski, Sokoban)
- Two-player perfect-information games requiring planning (Connect-4, Othello, Pentago)
- Code synthesis with compilation feedback
- Multi-hop QA with explicit retrieval
- Constraint satisfaction (SAT, Sudoku, scheduling)
- Theorem-proof skeleton generation with proof checker
- Typed-graph generation (the wiring task; SQL; ToolBench tool calls)
- Multi-stage clinical/financial decisioning with regulatory verification

### 0.3 Tasks OUTSIDE OPA's natural zone (at least one criterion missing)

- Sentiment classification (no structured output, no search depth)
- Single-fact retrieval (no state accumulation, no search depth)
- Free-form text generation (no verifier)
- Numerical regression on continuous inputs (no structured output)
- Image classification (no structured output, no verifier)
- Code completion without execution (no verifier)

### 0.4 What this means for E15's choice of task

E15 picks a task that satisfies **all 5 criteria with high confidence**, so the architectural comparison is meaningful. The chosen pair:

- **Klotski** — 80-100 move solutions on hard positions; ~10⁸ reachable states; deterministic move-application; cycle detection in solver is mandatory; existing OPA demo at 62% solve rate
- **15-puzzle** — 22-median / 80-worst-case move solutions; ~10¹³ reachable states; deterministic; existing OPA infrastructure scales from 8-puzzle (which already exists at 90% solve)

Both satisfy C1-C5 with no ambiguity. Both have existing OPA infrastructure to leverage. Neither is so easy that both architectures hit-ceiling-and-tie (the 11-game suite documents that 15-puzzle and Klotski are harder than the other 9 games — which is why this is the right hardness tier for a discriminating experiment).

### 0.5 What this means for what would falsify OPA's value claim

A monolithic 1M-param transformer trained on the same (state, optimal-move-sequence) corpus, with the same training compute, on the same evaluation harness, performing **as well as or better than** the OPA composition would falsify the "coordination is the intelligence" thesis at this scale and task class. Per the project's pre-registration discipline, that result is **more interesting than confirmation** — it tells the field where OPA's distinctive value actually lives (audit, latency, determinism, edge — not capacity efficiency).

This is the first experiment the project has run that can falsify its founding claim against the right control. That's the headline.

---

## 1. Proposal

### 1.1 Hypothesis (locked before measurement)

> *On hard-search puzzle problems satisfying all five criteria from §0.1 (Klotski + 15-puzzle), an OPA composition of three role-specialised organelles (planner → player → judge with `OpaKanban` working memory and cycle detection), each ~300K parameters (total ~900K), trained on a deterministic-oracle-generated corpus expressed via a new OQL `FROM ORACLE` source clause, achieves solve rate ≥ **15 percentage points higher** than a single monolithic transformer of the same ~900K total parameter budget trained on the same corpus with the same compute. The ≥15pp margin is the locked falsification threshold: smaller margins (5-15pp) are "weakly validated"; <5pp is "thesis falsified at this scale"; monolithic-wins is "thesis contradicted, reframe required."*

### 1.2 Why this matters

Eight prior experiments built the substrate; none measured **OPA vs equal-budget monolithic on the same task**. Every game-demo result is *vs random opponents*. Every wiring number is *internal ceiling against itself*. Every LLM follow-up (E12, E13) was *neutral or falsified*. The project's founding claim — that tiny specialists coordinated by a Judge outperform monolithic capacity — has never been measured against the comparator that would test it.

Per `experiments/README.md`'s strategic context:

> *Every architectural claim the project makes — coordination-is-intelligence, audit-trail-native, edge-deployable, composable — needs at least one experiment that compares OPA to a credible alternative on a metric that matters. E15 is the first such measurement.*

The Klotski + 15-puzzle pair is chosen specifically because both satisfy all 5 criteria from §0.1 and both have *real headroom* (the existing OPA demos are at 62% / 90% — not saturated, leaving room to measure differences). They're also tractable: existing engine, existing game-loop infrastructure, deterministic oracle (A*/BFS) for corpus generation.

### 1.3 Mechanism

#### 1.3.1 New OQL SOURCE clause — `CREATE CORPUS … FROM ORACLE`

The corpus-generation step has been an *unspoken* part of every prior experiment — each demo had its own `generate_corpus.py` script. E15 lifts this into OQL as a first-class SOURCE clause:

```sql
CREATE CORPUS klotski_optimal
  FROM ORACLE 'tools/klotski_a_star.c'
  WITH (
    count = 10000,
    difficulty = 'mixed',         -- 30% easy / 50% medium / 20% hard
    seed = 1337,
    cache = '.oql_oracle_cache/'
  )
  PROMPT 'klotski_state -> optimal move sequence (A* shortest path)';

CREATE CORPUS puzzle15_optimal
  FROM ORACLE 'tools/puzzle15_a_star.c'
  WITH (count = 10000, difficulty = 'mixed', seed = 1337)
  PROMPT 'puzzle15_state -> optimal move sequence';
```

The `FROM ORACLE` clause invokes a deterministic C solver and produces (state, move-sequence) pairs. **No new top-level OQL verb** (the +6/-4 lock holds); ORACLE is a new SOURCE alongside FILE (E10), LLM (E12), and the future LLM_SOURCE (E14).

This makes corpus generation **expressible in the same .oql file as training and evaluation** — closing what the user correctly identified as a gap: *"this feels artificial without corpus generation in the loop."*

#### 1.3.2 Monolithic baseline

A single ~900K-parameter transformer trained on the same corpus. Same architecture family as the OPA student organelles (so the comparison is fair); same training compute (same step count, same batch size); same evaluation harness.

```sql
CREATE ORGANELLE klotski_mono WITH (n_embd = 96, n_head = 6, n_layer = 4, block_size = 256);
TRAIN klotski_mono ON klotski_optimal STEPS 50000 LR 1e-3 SAVE 'checkpoints/klotski_mono.ckpt';
```

#### 1.3.3 OPA composition

Three role-specialised organelles totalling the same ~900K params:

```sql
CREATE ORGANELLE klotski_planner WITH (n_embd = 48, n_head = 4, n_layer = 4, block_size = 128);  -- ~300K
CREATE ORGANELLE klotski_player  WITH (n_embd = 48, n_head = 4, n_layer = 4, block_size = 128);  -- ~300K
CREATE ORGANELLE klotski_judge   WITH (n_embd = 48, n_head = 4, n_layer = 4, block_size = 128);  -- ~300K

TRAIN klotski_planner ON klotski_optimal WITH ROLE planner STEPS 50000 LR 1e-3 SAVE '...';
TRAIN klotski_player  ON klotski_optimal WITH ROLE player  STEPS 50000 LR 1e-3 SAVE '...';
TRAIN klotski_judge   ON klotski_optimal WITH ROLE judge   STEPS 50000 LR 1e-3 SAVE '...';

CREATE PIPELINE klotski_opa AS COMPOSE @graph
  state    = read_state()
  plan     = call(klotski_planner, state)
  move     = call(klotski_player, plan)
  verdict  = call(klotski_judge, move)
  out      = verdict
@end;
```

The composition uses the existing `OpaKanban` working memory + cycle detector + fallback (from E08/E09's game-loop runtime). **Total compute budget is identical** (3 organelles × 50000 steps = 150000 organelle-step-equivalents; monolithic at 50000 steps with 3× model capacity has the same training-FLOP budget by construction). The agent must verify this equivalence in Section 3.

#### 1.3.4 Evaluation harness

Held-out test set of **500 puzzle positions** for each task, generated by the oracle but **never included in training corpora** (standing leakage audit verifies). For each position:

1. **Monolithic system** — single forward pass produces a move sequence; deterministic verifier checks against goal state; mark solved/unsolved.
2. **OPA system** — `oql_run_klotski` drives the pipeline with the same time budget; deterministic verifier marks solved/unsolved.

Headline metric: **solve rate** (% of held-out positions where the system reaches the goal within 200 moves).

Secondary metrics: per-position solution length (lower = better), latency p99 per attempt, audit-trace coverage.

#### 1.3.5 Phase order

| Phase | Work | Cost |
|---|---|---|
| 1 | OQL grammar: `FROM ORACLE` source clause | 3-4 days |
| 2 | `tools/klotski_a_star.c` + `tools/puzzle15_a_star.c` — deterministic oracles | 3-4 days |
| 3 | Corpus generation — 10k positions each, via `FROM ORACLE` | 1 day (compute-bound) |
| 4 | Monolithic baseline training | 1 day |
| 5 | OPA composition training (3 organelles) | 1 day |
| 6 | Held-out generation + leakage audit | 1 day |
| 7 | Evaluation runs (500 positions × 2 systems × 2 tasks) | 2-3 days |
| 8 | Section 3 writeup with full verdict matrix and falsification interpretation | 3-4 days |

### 1.4 Pre-registered targets (locked)

| ID | Target | Floor (skip-rule trigger) |
|---|---|---|
| **T1** | OQL grammar accepts `CREATE CORPUS … FROM ORACLE …` | Parse failure |
| **T2** | Corpus generation produces 10k valid (state, optimal-solution) pairs for each task, with **zero leakage** against held-out (standing audit gate) | Leakage detected → tighten held-out generation; do NOT relax audit |
| **T3** | Monolithic baseline solve rate measured on 500 held-out positions for each task | Run fails to complete |
| **T4** | OPA composition solve rate measured on the same 500 positions | Run fails to complete |
| **T5** | **OPA composition solve rate − Monolithic solve rate ≥ 15 percentage points** on Klotski **AND** on 15-puzzle | < 5pp on either task = thesis falsified at this scale |
| **T6** | Total training compute is equivalent within ±10% (verified by step count × parameter count × batch size) | > ±20% imbalance |
| **T7** | Per-task held-out leakage audit passes (zero Jaccard ≥ 0.7 against training corpus) | Any leakage |
| **T8** | All existing tests pass; engine surface frozen; +6/-4 verb lock holds; zero new VM opcodes | Any regression |

The headline result is judged on **T5**.

### 1.5 Outcome ladder (the four corners)

| Margin (OPA − Monolithic) on both tasks | Interpretation |
|---|---|
| ≥ 15pp on both | **Thesis validated.** Coordination beats capacity at equal budget on hard-search problems. Publishable as the project's flagship value demonstration. |
| 5-15pp on both | **Weakly validated.** Composition helps but not decisively. Re-frame value claim toward audit/latency/edge rather than capacity efficiency. |
| 0-5pp on either | **Thesis not supported at this scale.** Coordination doesn't beat capacity on these tasks. Re-think where OPA's distinctive value lives. Still publishable — the most important result the project could produce. |
| Monolithic wins on either | **Thesis contradicted.** Capacity efficiency favours monolithic at this scale. Substantial reframe required: OPA's value is in audit/edge/composability/explainability, not raw task performance. |

All four are scientifically informative. The pre-registration discipline benefits because the result is meaningful regardless of direction.

### 1.6 Skip rules

- **If T5 trips below 5pp** on either task: STOP. Document the thesis-falsified outcome honestly. Do NOT add more training compute, more parameters, or more organelles to try to close the gap — that would be retroactive rationalisation.
- **If T6 trips** (compute imbalance > ±20%): STOP. The comparison isn't fair. Re-budget and re-run; do NOT report results from an unfair comparison.
- **If T7 trips** (leakage): STOP. Regenerate the held-out with stricter disjointness; do NOT relax the audit.
- **If T8 trips** (engine surface change / new opcode / new verb): STOP. These locks are non-negotiable.

### 1.7 Falsification risk: HIGH (intentional)

| Risk | Likelihood | Why this is good |
|---|---|---|
| Monolithic beats OPA at equal budget | Medium-high | If true, this is the **most valuable single result the project could produce** — it forces an honest re-framing of where OPA's value actually lives |
| OPA wins by only 5-10pp | Medium | The weakly-validated outcome; informative but not headline-strong |
| Both architectures hit the ceiling and tie at high accuracy | Low (Klotski is genuinely hard) | If it happens, harder positions needed; document and re-run |
| Oracle is too slow to generate 10k positions | Low | A* with good heuristic is fast; budget allows fallback to 5k |
| Compute equivalence is hard to enforce precisely | Medium | Step count × param count × batch size is a clean proxy; T6 allows ±10% |
| The "role-specialised training" (planner / player / judge) is harder than expected to set up cleanly | Medium | Existing 11-game demos all do this; pattern is established |

### 1.8 What this experiment is NOT testing

- It is **NOT** testing whether OPA beats LLMs. That's E01 (gated on API budget). E15 tests OPA vs equal-budget monolithic — the architectural comparison that's never been done.
- It is **NOT** testing every game in the 11-demo suite. Klotski + 15-puzzle are picked because they satisfy §0.1's criteria with high confidence and have real headroom.
- It is **NOT** testing whether bigger OPA beats bigger monolithic. The 900K-param budget is fixed; scaling-curve work is separate.
- It is **NOT** testing audit-trail value, edge-deployment value, or latency value. Those are different value axes — E15 tests **task accuracy at equal budget**, the one axis the project has never measured.
- It is **NOT** lifting any existing experiment's falsification. E12's 0/20 stands. E13's neutral band stands. E15 is orthogonal.

### 1.9 Cross-references

| Topic | Source |
|---|---|
| The architectural claim being tested | `RESEARCH_INTELLIGENCE.md` (random-baseline check) + `ORGANELLE_STATE.md` (thesis statement) |
| The pre-reg parent | This document is the response to `experiments/README.md` §"What we have NOT shown" |
| The OQL substrate being extended | [E07](E07-oql-dsl.md) (grammar) + [E10](E10-oql-train-wiring.md) (CORPUS) + [E12](E12-llm-wiring-corpus.md) (FROM LLM precedent) |
| Existing Klotski + 8-puzzle demos | `demos/character-level/{klotski,puzzle8,reddonkey}/` |
| Engine surface that must stay frozen | `src/microgpt.{c,h}` + `src/microgpt_vm.*` (E07-E14 cumulative lock) |
| Verb-discipline lock that must hold | [E07](E07-oql-dsl.md) §1.3.1 (+6/-4 surface) |
| The taxonomy this experiment instantiates | §0 of this document |

---

## 2. Initial state

### 2.1 What's currently known

- Klotski OPA demo: 62% solve rate vs random opponent (`RESEARCH_ORGANELLE_GAMES.md`)
- 8-puzzle OPA demo: 90% solve rate (close to ceiling; less informative for a discriminating experiment)
- 15-puzzle: no existing demo; needs new infrastructure (but inherits 8-puzzle's pattern directly)
- No monolithic-baseline comparison exists for any of the 11 games
- The architectural claim ("coordination is the intelligence") has been asserted continuously since 2026-02 but never measured against equal-budget monolithic

### 2.2 Baselines to beat

| Baseline | Number | E15 measures |
|---|---|---|
| Klotski OPA solve rate vs random opponent | 62% | E15 measures vs deterministic oracle solutions, NOT random opponent — different baseline |
| 8-puzzle OPA solve rate vs random | 90% | Excluded from E15 (too easy; near ceiling) |
| Monolithic same-budget on either task | UNKNOWN — **this is what E15 measures** | T3 |
| OPA solve rate on 15-puzzle | UNKNOWN — **also what E15 measures** | T4 |

### 2.3 Dependencies / blockers

- **OQL `FROM ORACLE` source** — new, this experiment builds it
- **`tools/klotski_a_star.c`** — new deterministic solver
- **`tools/puzzle15_a_star.c`** — new
- **Monolithic baseline harness** — needs the same evaluation pipeline as OPA, just with a single model
- **E10's `TRAIN`** — used for both monolithic and OPA training (already on main)
- **`libpipeline_ir`** — used for the composition pipeline (already on main)
- **LM Studio not needed** — E15 is purely deterministic; no LLM in the loop

### 2.4 What this experiment deliberately does NOT do

- Does NOT make a runtime LLM dependency (no LLM at all in E15)
- Does NOT change the engine surface
- Does NOT add VM opcodes or top-level verbs
- Does NOT introduce new build deps
- Does NOT touch the 11-game suite's other demos
- Does NOT scale beyond the 900K-param budget (scaling is a separate experiment)
- Does NOT compare across multiple compute budgets (single-budget comparison only; multi-budget is a follow-up)

---

## 3. Implementation + results

**TODO** — fill on measurement commit. Sections to populate:

- 3.1 OQL grammar extension: `CREATE CORPUS … FROM ORACLE …` parse tests
- 3.2 `tools/klotski_a_star.c` + `tools/puzzle15_a_star.c` — algorithm + sanity checks
- 3.3 Corpus generation stats (10k per task; difficulty distribution; oracle wall-clock)
- 3.4 Held-out generation + leakage audit (T7)
- 3.5 Monolithic baseline training run + solve rate on Klotski (T3a)
- 3.6 Monolithic baseline solve rate on 15-puzzle (T3b)
- 3.7 OPA composition training + Klotski solve rate (T4a)
- 3.8 OPA 15-puzzle solve rate (T4b)
- 3.9 Compute-equivalence audit (T6) — step count × params × batch
- 3.10 Headline comparison table: margin on each task
- 3.11 Per-position solution-length distribution (secondary metric)
- 3.12 Latency p99 + audit-coverage (secondary metrics)
- 3.13 Per-target verdict matrix (T1-T8)

---

## 4. Conclusion

**TODO** — fill on measurement commit when ALL 8 targets are measured. Sections to populate:

- 4.1 Verdict per T1-T8
- 4.2 Headline outcome — which of the four §1.5 corners did T5 land in?
- 4.3 What this means for the project's value claim:
  - If ≥ 15pp: **thesis validated**; OPA has its first measured value demonstration against the right control
  - If 5-15pp: weakly validated; reframe toward audit/latency/edge
  - If < 5pp or monolithic wins: **thesis falsified or contradicted**; substantial repositioning required
- 4.4 What this means for the taxonomy in §0:
  - Does the 5-criteria checklist actually predict OPA's success?
  - Are any criteria load-bearing more than others? (e.g. is C3 — search depth — the critical one?)
- 4.5 What's NOT done: scaling-curve comparison; multi-budget comparison; comparison to LLMs (E01); comparison to RAG-augmented monolithic; replication to other §0.2 task classes
- 4.6 Next experiments suggested:
  - If thesis validated: extend to multi-budget scaling curve (E16)
  - If falsified: pivot E15-equivalents toward audit-value demonstrations
  - Either way: rewrite the project's headline claims in `ORGANELLE_STATE.md` to match the measured truth
- 4.7 Traceability updates (`TRACEABILITY.md`, `ORGANELLE_STATE.md`, `RESEARCH_DISCLOSURE.md`)
