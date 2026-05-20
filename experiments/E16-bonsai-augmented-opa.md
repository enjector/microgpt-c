# Experiment E16 — The real composition test: port Bonsai from the sibling repo, integrate into OPA, redo E15's comparison with the heterogeneous architecture

**Status:** 📋 Proposal locked — 2026-05-20.
**Direction:** E15 tested a strawman of OPA — a transformer-only composition with no Bonsai, no Nucleus, no Membrane Gate. The **real** OPA architecture (shipped in `~/dev/projects/microgpt-c/`) is heterogeneous: transformers for pattern recognition + Bonsai decision-tree organelles for deterministic classification/safety/audit + Nucleus for weighted aggregation. E16 ports Bonsai across, integrates it via OQL, and reruns E15's comparison with the architecture the project's claims have always been *about*.
**Cost estimate:** ~5-7 weeks (2 wk Bonsai + Nucleus + lifecycle port, 1 wk OQL integration, 1 wk training/eval harness extension, 1 wk measurement, 1-2 wk writeup).
**Falsification risk:** Medium-low — Bonsai's 8.6 ns deterministic classification + 100% audit-trail property is a real architectural advantage on classification-shaped problems. The honest risk is that for search-puzzle problems specifically, the transformer is still the bottleneck and Bonsai's contribution is marginal.

---

## 0. Why E15 didn't actually test the project's thesis — and why E16 does

E15 measured *"OPA composition vs monolithic at equal budget on hard search."* But the OPA composition E15 used was **transformer-only**: three role-specialised transformers (planner / player / judge) trained on oracle-optimal corpora, coordinated via `OpaKanban` + cycle detector + fallback. That composition did **not** include:

| OPA pillar | What it does | E15 used it? |
|---|---|---|
| **Bonsai** — deterministic decision-tree engine | 8.6 ns classification, 116M predictions/sec, 100% audit trail, calibrated probabilities, zero-malloc inference | ❌ No |
| **Membrane Gate** (`bonsai_membrane_gate`) | Energy-aware: 90% CPU savings during steady-state noise; entropy-based attention | ❌ No |
| **Signal Decay** (`bonsai_signal_decay`) | Prevents frozen logic; neurotransmitter-like reabsorption | ❌ No |
| **Lateral Inhibition** (`organelle_coordinate`) | Specialist organelles silence generalists during crises | ❌ No |
| **Nucleus** (`microgpt_nucleus.{h,c}`) | Weighted matrix vote across multiple Bonsai trees | ❌ No |
| **Transformer organelles** | Pattern recognition, sequence generation | ✅ Yes (the only thing) |

The sibling repo `~/dev/projects/microgpt-c/` ships all six pillars as production C99 code with worked demos (markets regime gate, flash-crash backtest, sector rotation, predictive 1-day forecast, trading agent with `opa_self_heal`). The projects.github repo (this one) has only the transformer pillar plus the OQL substrate from E07-E14.

**E15's verdict — "T5 FAIL on transformer-only composition" — is correct but narrow.** It did not falsify "coordination is the intelligence" because the real OPA's coordination is *between heterogeneous organelle classes*, not between three copies of the same architecture. E16 is the experiment that actually tests the thesis the project has always claimed.

This is the simple solution: **use the architecture that was already built, not a strawman of it.**

---

## 1. Proposal

### 1.1 Hypothesis (locked before measurement)

> *Porting Bonsai + Nucleus + Organelle Lifecycle from `~/dev/projects/microgpt-c/src/` into this repo, integrating Bonsai into the OPA composition as the deterministic classification / safety-gating / audit-trail layer (with the transformer organelle reduced to ~300K params for pattern-recognition only), and re-running E15's evaluation harness produces solve rate ≥ 15 percentage points higher than the same monolithic ~900K transformer baseline on Klotski **AND** ≥ 5pp higher on 15-puzzle (the latter relaxed because both architectures hit a 15-puzzle scale floor in E15; meaningful margin at this scale is unlikely). The Klotski ≥15pp margin is the locked headline; the 15-puzzle ≥5pp is a softer secondary target.*

### 1.2 Why this matters

E15's T5 falsification was correctly recorded under the pre-reg discipline, but the *interpretation* I almost shipped into `ORGANELLE_STATE.md` and the project README (— *"coordination is the intelligence is falsified, reframe toward audit/edge"* —) was an overreach because the real OPA architecture wasn't under test. E16 corrects this by:

1. **Testing the heterogeneous architecture the project's narrative has always claimed.** Bonsai handles "is this move legal?" (8.6 ns), "which candidate should fire?" (calibrated probability), "show me the audit trail" (100% by construction). The transformer is freed to handle only the hard pattern recognition.
2. **Producing a measurement that can actually falsify the broader thesis.** If E16 also fails to deliver ≥15pp margin, *that's* the result that justifies the reframe. If E16 passes, E15's narrow result is preserved (transformer-only doesn't win) and the broader thesis (heterogeneous OPA wins) is validated.
3. **Closing a major sibling-repo divergence.** The two repos have been drifting; E16 is the move that brings Bonsai into the OQL substrate.

### 1.3 Mechanism

#### 1.3.1 Phase 1 — Port Bonsai + Nucleus + Lifecycle (2 weeks)

Copy three modules from `~/dev/projects/microgpt-c/src/` into this repo's `src/`:

| Sibling repo file | This repo's destination | Notes |
|---|---|---|
| `microgpt_bonsai.{h,c}` | `src/microgpt_bonsai.{h,c}` | Zero-alloc fixed-point decision-tree engine. Self-contained C99. |
| `microgpt_organelle_lifecycle.{h,c}` | `src/microgpt_organelle_lifecycle.{h,c}` | Membrane gate, signal decay, lateral inhibition. Depends on bonsai. |
| `microgpt_nucleus.{h,c}` | `src/microgpt_nucleus.{h,c}` | Weighted multi-organelle voting. Depends on bonsai. |

Copy paired tests:
- `tests/test_microgpt_bonsai.c`
- `tests/test_microgpt_organelle_lifecycle.c`
- `tests/test_microgpt_nucleus.c`

**Hard constraint (T8 preservation):** the port adds NEW files; it does NOT modify `src/microgpt.{c,h}` or `src/microgpt_vm.*`. Engine surface stays frozen across E07-E16.

#### 1.3.2 Phase 2 — OQL integration: new `BONSAI` object type (1 week)

Extend `src/microgpt_oql.{l,y}` with:

```sql
CREATE BONSAI <name>
  FROM FILE '<path>'                    -- load pre-trained tree
  [WITH (max_depth = N, max_features = M, max_classes = K)];

CREATE BONSAI <name>
  TRAIN FROM CORPUS <corpus_name>       -- train new tree from existing CORPUS
  [WITH (max_depth = N, ...)];
```

And extend `CREATE ORGANELLE` to bind Bonsai trees:

```sql
CREATE ORGANELLE klotski_player
  FROM CHECKPOINT 'checkpoints/...'
  WITH (
    INPUT_BEHAVIOUR    = parse_c4_board,
    OUTPUT_BEHAVIOUR   = format_c4_move,
    SAFETY_GATE        = klotski_legality_bonsai,    -- NEW: Bonsai-driven safety
    REGIME_CLASSIFIER  = klotski_state_bonsai,        -- NEW: Bonsai-driven gating
    MEMBRANE_GATE      = ENABLED                      -- NEW: entropy-based attention
  );
```

**`BONSAI` is a new first-class object type via the existing `CREATE` verb** — exactly mirroring `BEHAVIOUR` (E08), `CORPUS` (E10), `LLM_SOURCE` (E14 planned). **The +6/-4 verb lock from E07 holds — no 7th top-level verb.**

#### 1.3.3 Phase 3 — Heterogeneous composition for Klotski + 15-puzzle (1 week)

```sql
-- Train one Bonsai tree per role
CREATE CORPUS klotski_optimal FROM FILE 'data/E15-klotski/train.tsv';

CREATE BONSAI klotski_legality
  TRAIN FROM CORPUS klotski_optimal
  WITH (max_depth = 8, max_features = 32, max_classes = 2);   -- binary: legal/illegal

CREATE BONSAI klotski_move_classifier
  TRAIN FROM CORPUS klotski_optimal
  WITH (max_depth = 8, max_features = 32, max_classes = 16);  -- which-block-which-direction

-- One transformer for pattern recognition (300K params; ~3x smaller than monolithic)
CREATE ORGANELLE klotski_pattern WITH (
  FROM CHECKPOINT 'checkpoints/klotski_pattern_e16.ckpt',
  SAFETY_GATE = klotski_legality,
  REGIME_CLASSIFIER = klotski_move_classifier,
  MEMBRANE_GATE = ENABLED
);

CREATE PIPELINE klotski_opa_bonsai AS COMPOSE @graph
  state    = read_state()
  in       = parse_board(state)
  cand     = call(klotski_pattern, in)        -- transformer proposes
  legal    = bonsai_check(klotski_legality, cand)   -- Bonsai validates (8.6 ns)
  picked   = bonsai_classify(klotski_move_classifier, in)   -- Bonsai also generates a candidate
  out      = pick_best(cand, picked, legal)   -- nucleus aggregation
@end;
```

The architectural shape:
- **Bonsai trees** handle classification (which move?), safety (is move legal?), and gating (is this state interesting enough to invoke the transformer?).
- **The single 300K transformer** handles pattern recognition only — what the proposed move's *quality* is given the board state, where Bonsai's coarse classification needs refinement.
- **Nucleus** votes between the transformer's proposal and Bonsai's classification, weighted by Bonsai's calibrated confidence.

#### 1.3.4 Phase 4 — Re-run E15's evaluation harness (1 week)

Same held-out sets (`data/E15-klotski/heldout_large.tsv` — 113 positions; `data/E15-puzzle15/heldout_large.tsv` — 948 positions). Same 200-move budget per attempt. Same deterministic verifier.

Measure:
- Monolithic 900K transformer (reuse from E15 — `checkpoints/klotski_mono_e15.ckpt` + `puzzle15_mono_e15.ckpt`)
- OPA-Bonsai composition (new from this experiment)

**Direct comparison to E15's results in `results/`.** New per-position CSVs: `results/klotski_opa_bonsai_eval.csv`, `results/puzzle15_opa_bonsai_eval.csv`.

#### 1.3.5 Phase 5 — Section 3 + Section 4 writeup (1-2 weeks)

Per-target verdict matrix. **Headline comparison:**

| System | Klotski solve rate | 15-puzzle solve rate |
|---|---|---|
| Monolithic 900K (E15 reused) | 64.6% | 0.1% |
| Transformer-only OPA (E15 reused) | 69.9% (+5.3pp) | 0.7% (+0.6pp) |
| **OPA-Bonsai (E16, new)** | **TBD** | **TBD** |

Three-way comparison gives a clean architectural story: *did Bonsai earn its keep?*

### 1.4 Pre-registered targets (locked)

| ID | Target | Floor (skip-rule trigger) |
|---|---|---|
| **T1** | Bonsai + Nucleus + Lifecycle ported; all sibling-repo tests pass on this repo's build | Any porting failure |
| **T2** | OQL `CREATE BONSAI` parses; `CREATE ORGANELLE WITH (SAFETY_GATE = ..., REGIME_CLASSIFIER = ..., MEMBRANE_GATE = ENABLED)` parses | Parse failure |
| **T3** | OPA-Bonsai composition trains on Klotski + 15-puzzle (corpora reused from `data/E15-*/`) | Training crashes |
| **T4** | OPA-Bonsai solve rate measured on `data/E15-klotski/heldout_large.tsv` (113 positions) and `data/E15-puzzle15/heldout_large.tsv` (948 positions) | Run fails |
| **T5a** | **OPA-Bonsai − Monolithic ≥ 15 pp on Klotski** (the headline; matches E15's locked threshold) | < 5pp = thesis falsified at this scale |
| **T5b** | OPA-Bonsai − Monolithic ≥ 5 pp on 15-puzzle (relaxed because E15 showed floor effect at this scale; honest target) | < 0pp (i.e. Bonsai actively hurts) |
| **T6** | Transformer-parameter equivalence: monolithic 900K vs OPA-Bonsai's transformer organelle 300K (note: Bonsai trees are ~10 KB each, treated as architectural addition, not "params" in the same sense) | T6 framing breaks |
| **T7** | Engine surface frozen, +6/-4 verb lock holds (`BONSAI` is a CREATE object, not a 7th verb), zero new VM opcodes, zero new build deps | Any lock broken |
| **T8** | Audit-trail coverage: every Bonsai decision produces a `bonsai_decision_path` log; 100% by construction; verifiable in `results/` | Any decision missing audit trail |

Headline survives if **T1, T2, T3, T4, T5a, T7 all pass**. T5b is a softer secondary; T6/T8 are discipline floors.

### 1.5 Outcome ladder (mirrors E15 §1.5 but with the right architecture)

| Klotski margin | 15-puzzle margin | Interpretation |
|---|---|---|
| ≥ 15pp | any | **Thesis VALIDATED.** Bonsai-augmented OPA is the architecture the project's claims have always been about; the right comparison was finally run. |
| 5-15pp | any | **Weakly validated.** Bonsai helps but not decisively; the architectural advantage is real but modest at this scale. |
| 0-5pp | any | **Thesis NOT supported even with Bonsai.** Forces a deeper reframe than E15's narrow falsification. Reframe should target audit/edge/explainability axes, not capacity. |
| Monolithic wins | any | **Thesis contradicted on architecture.** Most informative outcome. OPA's distinctive value is *not* task-accuracy efficiency at any scale tested. |

All four are scientifically informative under the pre-reg discipline.

### 1.6 Skip rules

- **If T5a < 5pp** on Klotski: STOP. Document the deeper falsification honestly. Do NOT add more Bonsai trees, scale up the transformer, or add unmeasured pillars (signal decay / lateral inhibition tuning) retroactively. The §1.5 corner the result lands in is locked.
- **If T1 fails (port doesn't build)**: STOP, document as a porting blocker (mechanical issue, not architectural). Re-attempt with reduced scope (e.g., port only Bonsai, defer Nucleus + Lifecycle).
- **If T7 trips** (engine surface change / new VM opcode / 7th verb / new build dep): STOP. These locks have held across nine experiments; do not break them for E16.
- **If 15-puzzle 0.1% / 0.7% floor effect persists for OPA-Bonsai too** (both at ~0%): T5b honestly NOT-MEASURABLE; report as a scale issue, not a Bonsai issue.

### 1.7 Falsification risk: Medium-low (intentional)

| Risk | Likelihood | Why this is good or bad |
|---|---|---|
| Bonsai actually does deliver +15pp on Klotski | Medium-high | Classification at 8.6 ns + audit trail is genuinely a different architectural class than transformer-only |
| Bonsai is neutral or slightly negative | Medium | Would mean Klotski's pattern requires more than classification; transformer was already doing the right thing |
| 15-puzzle stays at floor for both architectures | High | Expected per E15 §1.5 — neither architecture has enough capacity at this scale |
| Port fails because sibling repo has dependencies this repo doesn't have | Low | Bonsai is self-contained C99; same precedent as the other ports we've done |
| OQL `CREATE BONSAI` integration introduces new VM opcodes | Low | Bonsai inference is direct C calls, not VM-dispatched; no opcode pressure |
| Audit-trail coverage breaks under high-volume eval | Low | Bonsai's 100% audit is by construction; the test just verifies it |

### 1.8 What this experiment is NOT testing

- It is **NOT** changing the corpora. Reuses `data/E15-klotski/train.tsv` + held-outs.
- It is **NOT** changing the monolithic baseline. Reuses `checkpoints/klotski_mono_e15.ckpt`.
- It is **NOT** scaling up the transformer side. 300K cap holds per the heterogeneous design.
- It is **NOT** retesting E15's narrow result. E15's T5 FAIL on transformer-only composition stays. E16 tests a *different* architecture.
- It is **NOT** moving Bonsai to runtime in any other context — the port is into design-time + inference-time of OPA pipelines only.
- It is **NOT** exhausting the OPA architecture's pillars. Signal Decay, Lateral Inhibition, Nucleus weighted voting all get implemented but minimal use in this experiment; full integration tests are separate experiments (E17+).

### 1.9 Cross-references

| Topic | Source |
|---|---|
| Why E15 was a narrow test, not the full thesis | `~/dev/projects/microgpt-c/docs/research/RESEARCH_ORGANELLE_ARCHITECTURE.md` + `RESEARCH_ORGANELLE_ARCHITECTURE_EDGE.md` |
| The Bonsai engine being ported | `~/dev/projects/microgpt-c/src/microgpt_bonsai.{h,c}` + paired tests |
| The Lifecycle module (membrane gate, decay, inhibition) | `~/dev/projects/microgpt-c/src/microgpt_organelle_lifecycle.{h,c}` |
| The Nucleus module (weighted voting) | `~/dev/projects/microgpt-c/src/microgpt_nucleus.{h,c}` |
| Existing integration hooks already exist | `~/dev/projects/microgpt-c/src/microgpt_organelle.h` lines 61, 68 — `opa_bonsai_safety_gate()`, `opa_self_heal()` already designed |
| Bonsai's benchmarked properties | `~/dev/projects/microgpt-c/docs/research/RESEARCH_BONSAI.md` — 98% Iris accuracy, 8.6ns inference, 116M ops/sec, 100% audit |
| Worked Bonsai demos in sibling repo | `~/dev/projects/microgpt-c/demos/bonsai/` + `demos/opa/*` (regime gate, flash crash, sector rotation, trading agent, predictive 1-day forecast) |
| The transformer-only result being augmented | [E15](E15-composition-vs-monolithic.md) §3.5-3.14 |
| The §0 input/output search-space taxonomy (still applies) | [E15](E15-composition-vs-monolithic.md) §0 |
| The corpora E16 reuses | `data/E15-klotski/`, `data/E15-puzzle15/` |
| The monolithic baseline E16 reuses | `checkpoints/klotski_mono_e15.ckpt`, `checkpoints/puzzle15_mono_e15.ckpt` |
| Hard-locks that must hold | [E07](E07-oql-dsl.md), [E08](E08-oql-behaviours.md) |

---

## 2. Initial state

### 2.1 What's currently known

- **The full OPA architecture exists** as production C99 code in `~/dev/projects/microgpt-c/`. Bonsai engine, lifecycle, nucleus, integration hooks — all there. Including paired tests + 11+ demos.
- **This repo (projects.github) is missing the heterogeneous half.** Only transformer organelles + OQL substrate from E07-E14.
- **E15 measured the transformer-only subset.** T5 FAILED — Klotski +5.3pp; 15-puzzle floor effect. **Locked verdict, narrowly applicable.**
- **Bonsai's claimed properties are benchmarked in the sibling repo:**
  - 98.0% Iris accuracy (training set)
  - 8.6 ns per sample inference
  - 116M predictions/sec
  - 100% audit-trail coverage by construction
  - Zero-malloc inference
  - 208 µs training per tree
- **Existing demos prove the architecture works in production:** market regime gate, flash crash detection, sector rotation, OPA trading agent with `opa_self_heal`, predictive 1-day forecast, predictive 15-min forecast.

### 2.2 Baselines to beat

| Baseline | Number | E16 must |
|---|---|---|
| Monolithic 900K (E15) on Klotski | 64.6% | OPA-Bonsai exceeds by ≥ 15pp (target ≥ 79.6%) |
| Monolithic 900K on 15-puzzle | 0.1% | OPA-Bonsai exceeds by ≥ 5pp (target ≥ 5.1%) — relaxed for floor effect |
| Transformer-only OPA on Klotski | 69.9% | OPA-Bonsai exceeds; would show Bonsai earns its keep |
| Transformer-only OPA on 15-puzzle | 0.7% | OPA-Bonsai exceeds |
| Engine surface diff | 0 lines | hold (T7) |
| OQL verb count | 6 | hold (T7) |

### 2.3 Dependencies / blockers

- **Sibling repo path** (`~/dev/projects/microgpt-c/`) must remain readable for the port phase.
- **Bonsai is self-contained C99 with no external deps** — porting is mechanical.
- **E15's corpora + checkpoints are already on main** (committed to `data/` and `checkpoints/` in the prior runs).
- **OQL grammar extension pattern is established** (E08 BEHAVIOUR, E10 CORPUS, E12 FROM LLM, E15 FROM ORACLE) — `CREATE BONSAI` mirrors these.

### 2.4 What this experiment deliberately does NOT do

- Does NOT make Bonsai a runtime LLM dependency. Bonsai is pure C99, sub-10ns at inference.
- Does NOT add VM opcodes (Bonsai inference is direct C calls; T7).
- Does NOT add a 7th OQL verb (`BONSAI` is a `CREATE` object; T7).
- Does NOT modify the engine (`src/microgpt.{c,h}`, `src/microgpt_vm.*` stay 0-line diff cumulative across E07-E16).
- Does NOT lift E15's narrow falsification — the transformer-only result stays.
- Does NOT exhaust all architectural pillars. Signal Decay, Lateral Inhibition, Nucleus get ported but minimal-use in E16; deeper integration is E17+ scope.

---

## 3. Implementation + results

**TODO** — fill on measurement commit. Sections to populate:

- 3.1 Bonsai + Nucleus + Lifecycle port: file moves, build integration, test pass matrix (T1)
- 3.2 OQL grammar extension for `CREATE BONSAI` + `WITH (SAFETY_GATE = ..., REGIME_CLASSIFIER = ..., MEMBRANE_GATE = ENABLED)` (T2)
- 3.3 Bonsai trees trained on Klotski + 15-puzzle corpora (T3)
- 3.4 OPA-Bonsai composition trained: transformer organelle + Bonsai trees + Nucleus binding
- 3.5 Klotski solve rate measured (T4, T5a)
- 3.6 15-puzzle solve rate measured (T4, T5b)
- 3.7 Per-task three-way comparison: Monolithic / Transformer-only OPA / OPA-Bonsai
- 3.8 Compute-equivalence + transformer-parameter accounting (T6)
- 3.9 Engine-surface-frozen + verb-lock confirmation (T7)
- 3.10 Audit-trail coverage verification (T8)
- 3.11 Bonsai decision-path examples (3-5 sample decisions with full audit chain)
- 3.12 Per-target verdict matrix

---

## 4. Conclusion

**TODO** — fill on measurement commit when ALL 8 targets are measured. Sections to populate:

- 4.1 Verdict per T1-T8
- 4.2 Headline outcome — which of the §1.5 four corners did T5a land in?
- 4.3 What this means for the project's value claim:
  - If ≥ 15pp on Klotski: **thesis VALIDATED with the right architecture.** The "coordination is the intelligence" claim is supported when coordination is between heterogeneous organelle classes, not three copies of the same architecture.
  - If 5-15pp: weakly validated; Bonsai helps but not decisively.
  - If < 5pp: **deeper falsification than E15.** Forces a real reframe — OPA's value lives in audit/edge/explainability, not capacity.
  - Monolithic wins: most informative outcome. Reframe heavily.
- 4.4 What this means for E15:
  - E15's T5 FAIL on transformer-only composition is **preserved verbatim**. E16 doesn't re-litigate E15.
  - The narrow finding from E15 + the broad finding from E16 together tell the architectural story: composition by itself doesn't win; heterogeneous composition (with Bonsai) might.
- 4.5 What this means for the project README + ORGANELLE_STATE.md:
  - If E16 PASS: do NOT rewrite the headline claims; the original "coordination is the intelligence" framing is validated by the right experiment.
  - If E16 FAIL: a genuine reframe is justified — but it's now grounded in two experiments (transformer-only + Bonsai-augmented both failing), which is much stronger evidence than E15 alone.
- 4.6 What's NOT done:
  - Full architectural exercise (signal decay tuning, lateral inhibition tuning, nucleus weighted voting variants) — E17+ scope
  - Replication across the 11-game suite — E18 (use OPA-Bonsai on Mastermind, Othello, Pentago, …)
  - Markets demos — sibling repo's domain; out of scope here
- 4.7 Next experiments suggested:
  - **E17:** signal decay + lateral inhibition tuning on the same Klotski/15-puzzle test
  - **E18:** replicate OPA-Bonsai across the 11-game suite vs equal-budget monolithic
  - **E19:** apply OPA-Bonsai to the audit-mandated medical-guideline task from E06 (the audit-trail value finally measured against a domain that requires it)
- 4.8 Traceability updates (`TRACEABILITY.md`, `ORGANELLE_STATE.md`, `RESEARCH_DISCLOSURE.md`, sibling-repo cross-references)
