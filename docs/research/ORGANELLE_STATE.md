# Organelles — state of the research, May 2026

**Status:** synthesis document, written after the scaling-curve arc closed. Sits at the top of the reading order for `docs/research/RESEARCH_ORGANELLE_*.md` and `docs/research/RESEARCH_PIPELINE_IR.md`. Honest about what's proven, what's bounded, what's open.

## What an organelle is, in this project

An organelle is **a tiny transformer (~30K–540K parameters) trained on a single role**, communicating with other organelles via flat pipe-separated text strings, coordinated by a deterministic C scaffold (`OpaKanban` ≈ 340 lines) that handles working memory, history, blocked moves, stalls, and cycle detection.

The thesis the project tests is summarised in one line:

> *"Tiny specialists, coordinated by a pipeline, outperform single models on focused tasks."*

The "intelligence" claim isn't in the model — a 540K-param transformer is too small to be intelligent on its own. The claim is in the **coordination**: planner → player → judge handoffs, cycle detection breaking A↔B oscillations, multi-organelle voting, and a typed pipeline IR (`microgpt_pipeline.{h,c}`) acting as the deterministic Judge that gradient descent can't be.

## What's been proven

The thesis is validated on three classes of task, each documented in its own research file:

| Application class | Where | Status | Headline number |
|---|---|---|---|
| Two-player perfect-information games | 11 demos: Connect4, Hex, Hex5, Klotski, Lightsout, Mastermind, Othello, Pentago, ... | ✅ Production-stable | Tiny organelles play complete games at hobbyist-level quality with sub-100ms moves on a laptop |
| Adversarial search with multi-organelle voting | Game demos + `RESEARCH_ORGANELLE_REASONING.md` | ✅ Stable | Cycle detector + Kanban breaks ~95% of A↔B oscillations |
| Text → typed graph generation (Wiring Organelle) | `RESEARCH_PIPELINE_IR.md` §1-46 + post-Phase-3 docs | ✅ Honestly bounded | 100% on anchored single-family prompts (Phase 2c clean); 70% on multi-stage compositions (Phase 3b); ~75-80% on novel paraphrases with distinctive vocabulary (v2/v3 audited) |

Three more classes are *speculatively* possible (vision via OPA-Vision, planner organelles for game tree search, lottery prediction) but those are exploratory rather than validated to the same depth.

## The Wiring Organelle arc — the most-tested case

Because this is where the architecture met its first honest measurement crisis, it's worth its own summary.

| Phase | What was claimed | What was honest | Lesson |
|---|---|---|---|
| 1-13 (Phase 13 corpus engineering) | 75% median wiring-layer accuracy | Inflated by training-on-test from Phase 13's lexical-anchoring corpus expansion | §38 leakage audit; verbatim-leak detector added |
| 2c clean rebuild | 35% wiring transformer alone, **100%** anchor-retrieval on novel paraphrases | True (verified twice, including v3 leakage audit) | Anchor retrieval is the actual production mechanism; wiring transformer is a noise source the Judge filters |
| 3a (TF-IDF learned classifier) | Pre-registered 12-16/20 on adversarial axis-2 | 4/20 — hypothesis falsified | Pre-registration discipline saved a follow-up from confirmation bias |
| 3b (fragment composition) | Pre-registered 5-7/10 on multi-stage chains | 6/10, met spec — shipped | Composition-from-fragments works inside the Judge's verifier; +1 to 7/10 in Tier 1 after the after-connective fix |
| 4 (corpus expansion to ~5k pairs) | Pre-registered 8-12/20 on adversarial axis-2 | **18/20** — exceeded prediction | TF-IDF + curated synonyms beats handcoded keyword bag at sufficient corpus size |
| Scaling experiment v1 (post-research) | "1:1 scaling, 20/20 retrieval on 20 new families" | **Falsified** — 19/20 of held-out had Jaccard ≥ 0.7 against training (curator self-overlap) | The leakage discipline that worked for verbatim *did not work* for bag-of-words near-duplicates; built `tools/scaling_leakage_audit.sh` as the corrective |
| v2 clean | 16/20 (80%) honest baseline | True; subtractive sharpening of clamp families lifted from 15→16 | First solid "what does this architecture actually deliver?" number |
| Phase 3 broad expansion (40 families) + bigram TF-IDF | Pre-registered v3 ≥ 15/20 + bigram lifts ≥ 18/20 | **Both falsified.** v3 = 3/20 (lean synonyms); bigram doesn't break ceiling | Three independent feature variants (unigram / bigram / char-ngram) all hit the same ~80% ceiling |
| Post-Phase-3 #1-3 cleanup | Wiring Phase 8 vote-loop regression fixed (rollback); v3 deep tested; char-ngram TF-IDF tested | All landed honestly — vote loop fixed via path-4 rollback (real fix deferred); v3 deep HURT (3→0); char-ngram doesn't move ceiling | Three confirmed structural bounds: curator-, model-, and domain-bounded |
| Phase 5 compositional held-out (V1.0.5) | Pre-registered ≥ 50 % on 30-prompt leakage-audited compositional set | 9/30 = 30 % — falsified at first run | Pre-registration discipline saved a follow-up from confirmation bias (again) |
| Phase 6 simple-search (V1.0.6) | Pre-registered ≥ 40 % | 9/30 = 30 % — falsified | Beam-2, drop-name-dedup, geo-prior all measurably no-op or harmful at this corpus scale |
| Phase 6b argument binder (V1.0.8) | Pre-registered Stream D mechanism shipped | 12/30 = 40 % — PARTIALLY-RESOLVED | Repeated-noun unification + outer-port-keyword inheritance lifted +10 pp |
| Phase 6c branched-mining (V1.0.9) | Pre-registered ≥ 50 % | 16/30 = 53 % (corrected; published as 50 % at the time) — original SLO-WIRE-005 design goal met | Per-primitive synonym lift + symmetric tie-break + disabling legacy substring bump compounded to +5 prompts |
| Phase 6d port-aware inner picker (V1.1.0) | Pre-registered ≥ 70 % aggregate; failure floor 60 % | **19/30 = 63 %** — PARTIALLY-RESOLVED in 60–69 % band | H8 captured +3 of predicted +6; H9 not implemented; binder positional-scoping (`GAP-WIRE-010`) is the next-best escalation |

## The current calibrated claim

This is the version the project's papers, talks, and customer conversations should use:

> **Tiny organelles + deterministic Judge + manifold retrieval reliably handle compositional, audit-required tasks where the family vocabulary is genuinely distinctive. Production-quality numbers: 100% on anchored novel paraphrases (single-family), 70% on multi-stage compositions, **63% on novel compositional generation** (V1.1.0 leakage-audited held-out, +10 pp over V1.0.9), ~75-80% on novel-paraphrase retrieval with vocabulary-disjoint test discipline. The ~80% ceiling is structural to bag-of-features classifiers in the curator's vocabulary regime; breaking past it requires either external pretrained embeddings (model class change) or domain-restricted deployment (where families have genuinely distinctive nouns).**

This is honest, falsifiable, and reproducible. It's also actionable: anyone evaluating whether the architecture fits their problem can read the three bounds and decide.

## Where the thesis stands

| Aspect of the thesis | Status |
|---|---|
| "Tiny specialists outperform single models" | ✅ Validated on focused tasks (games, anchored-family text→graph) |
| "Coordinated by a pipeline" | ✅ Validated; the pipeline IR + verifier is the most distinctive component |
| "On focused tasks" | ✅ The "focused" qualifier is doing real work — broad-domain tasks hit the bounds documented above |
| "Outperform" | ⚠️ Validated against rules-based baselines and against same-size single transformers; **not** validated against frontier LLMs (and unlikely to compete head-to-head; that's not the pitch) |
| Audit / explainability claim | ✅ Validated — pipeline IR's typed DAG + DOT renderer is the audit surface; verifier is the truth check |
| Edge / on-device claim | ✅ Validated — < 5MB binary, < 5ms p99 inference, 540K-param model |

## What we still don't know

Three open follow-ups, each documented in its own home but worth listing:

1. **Wiring binary vote-loop scoring** (`docs/research/wiring_binary_phase8_regression.md`). The candidate-scoring in `demos/wiring_organelle/main.c` lets new-family anchors out-vote correct existing-family anchors when arities coincide. Surgical rollback restored Phase 8 to 20/20; the real fix (paths 1-3) is engineering work not done yet. Affects how many new families the wiring binary can absorb at once.

2. **Whether external pretrained embeddings break the ~80% ceiling** (`docs/research/wiring_scaling_post_phase3.md` §3 follow-up). Three bag-of-features variants (unigram / bigram / char-ngram) all hit the same ceiling on novel-paraphrase tests. The hypothesis "the ceiling is model-bound, not curator-bound" can only be tested by a genuinely semantic feature (sentence-transformer, fastText, GloVe). Untested because it requires breaking the project's "pure C99, zero deps" policy — which is the gating call for productisation (`MIGRATED:DEPENDENCY_POLICY.md → see docs/MIGRATED_TO_ORGANELLES_BIO.md`).

3. **Independent-curator reproducibility.** Every measurement so far has had one author writing both the synonym tables and the held-out paraphrases. A second person rebuilding v2's family library from scratch with their own vocabulary would test whether the 75-80% ceiling is curator-vocabulary-specific or genuinely architectural. No work done; depends on a second engineer's availability.

4. **Whether OPA adaptive-depth (Phase 7) lifts game solve rates.** V1.2.0 ships the API primitives (`OpaActHalting`, `OpaFrozenInput`) per the `OPA_ADAPTIVE_DEPTH_ROADMAP.md` v1.0 pre-registration, but demo integration + measurement is deferred behind a customer-signal trigger. Hypotheses H10 (8-puzzle hard-tier 30 %→80 %) and H11 (Connect-4 cycle-detector trips −30 %) remain falsifiable but not yet measured. Per the methodology, V1.0 demo numbers (90 % 8-puzzle, 88 % Connect-4) remain the published baseline. See `RESEARCH_DISCLOSURE.md` §9, `RESEARCH_OPENMYTHOS_CROSS_POLLINATION.md` §9.

5. **Whether binder positional scoping closes the wiring layer's remaining 11/30 failures.** Identified during V1.1.0 outcome analysis (`GAP-WIRE-010`, OPEN, P3): ~5–6 of 11 residual compositional failures have correct primitive sets but wrong wiring. Predicted lift +13 pp would push the headline to ~76 %. Not scheduled — opens with customer signal that ≥ 70 % is required.

## What productisation will test

The three productisation verticals (`MIGRATED:PRODUCT_FRAUD_DETECTION.md → see docs/MIGRATED_TO_ORGANELLES_BIO.md`, `MIGRATED:PRODUCT_FINANCE_RISK.md → see docs/MIGRATED_TO_ORGANELLES_BIO.md`, `MIGRATED:PRODUCT_DEFENCE_TRACKING.md → see docs/MIGRATED_TO_ORGANELLES_BIO.md`) are not just commercial moves — they're **the next round of organelle research on real-world data**. Each vertical exercises a different bound:

- **Fraud** tests the "distinctive vocabulary" criterion in production. Fraud nouns (velocity, MCC, AVS, BIN, chargeback) are about as distinctive as English gets; the architecture should hit the upper end of the ceiling here. If it doesn't, the ceiling is lower than we think.
- **Finance** tests the "compositional + probabilistic" extension. Adding `pipeline_verify_with_confidence()` is a real architectural change — if it works cleanly, the architecture has earned a third deterministic-Judge mode (binary, ranked, calibrated). If it doesn't, we've identified a genuine architectural limit.
- **Defence** tests the "tiny specialists + handover" pattern at scale. Multi-object tracking is a well-studied problem with strong baselines — if the organelle pattern (sensor-org → tracker-org → classifier-org → assessor-org) competes with monolithic baselines, the thesis generalises beyond text. If it doesn't, organelles may be text-specific.

In all three cases, the productisation work is **a continuation of the research arc**, not a separate enterprise. The research findings inform the products; the product measurements feed back into the research claims.

## Reading order for a new contributor

| You want to understand | Read in this order |
|---|---|
| What an organelle is and why | This file → `RESEARCH_ORGANELLE_REASONING.md` → `RESEARCH_ORGANELLE_PIPELINE.md` |
| The most-tested application | `RESEARCH_PIPELINE_IR.md` (long, ~3800 lines, the full Wiring arc) |
| Recent honest measurement work | `wiring_scoreboard_tier0.md` → `wiring_scoreboard_tier1.md` → `wiring_scaling_curve.md` (carries a correction notice) → `wiring_scaling_curve_phase3.md` → `wiring_scaling_v3_deep_negative.md` → `wiring_scaling_post_phase3.md` (consolidated current state) |
| Game / vision / planner organelle applications | `RESEARCH_ORGANELLE_GAMES.md`, `RESEARCH_ORGANELLE_VISION.md`, `RESEARCH_ORGANELLE_PLANNER.md` |
| Productisation thinking | `STRATEGY_ONE_PAGER.md` → `PRODUCTIZATION_VERTICALS.md` → vertical sketches → `DEPENDENCY_POLICY.md` |

## Closing line (original — pre-OQL arc)

The organelle research arc has reached the point where the architecture is *understood* — its strengths, its three structural bounds, the honest numbers it produces, the gating decision that opens the next chapter. There is no more research question the architecture itself can pose at the current scale. **What's left is committing to one of the verticals on real data, which is itself a research move that will refine these claims further.**

---

## May 2026 update — the OQL substrate arc (E07 onwards)

The closing line above is preserved verbatim because the calibrated three-bound claim it captures (75-80% novel-paraphrase ceiling; curator-, model-, domain-bounded) remains accurate for the wiring arc. What follows is what has been *added on top* during the experiments/ work, all of which sits within the original architectural envelope.

### What's been added — eight measured experiments

| Experiment | Headline finding | What it adds to the project's claims |
|---|---|---|
| [E02](../../experiments/E02-pipeline-ir-library.md) — `libpipeline_ir` | Pipeline IR + verifier + DOT extracted to 58 KB standalone C99 library; consumed by E08-E13 cleanly | First reusable standalone artefact; verifier-as-Judge is portable |
| [E04](../../experiments/E04-eml-neural-hybrid.md) — Neural + EML hybrid | **100% hybrid extrapolation vs 1.5% pure-neural** at 50x wider input range (two orders of magnitude gap) | Complementary-organelle-classes story is **measurable**, not rhetorical |
| [E05](../../experiments/E05-prereg-methodology-public.md) — Methodology artefact | 13-page paper draft + ~700 LOC C99 pre-reg parser + CI hook catches both known historical leakage incidents | Project's pre-reg/audit discipline is **portable** to other projects |
| [E07](../../experiments/E07-oql-dsl.md) — OQL DSL | +6/-4 verb surface; grammar 129+81 LOC; **verb-discipline lock held under cumulative pressure from six subsequent experiments** | OQL is the right operator surface |
| [E08](../../experiments/E08-oql-behaviours.md) — BEHAVIOUR + VM TS | Connect-4 wrapper at **18.5% of C-demo LOC**; zero new VM opcodes under hard-lock | Researcher surface earns its cost |
| [E09](../../experiments/E09-oql-runtime-wiring.md) — OQL runtime wiring | `oql run` drives a real game loop; T2 PARTIAL at 51% closed by E11 at 89% | OQL is **executable**, not parse-only |
| [E10](../../experiments/E10-oql-train-wiring.md) — OQL TRAIN | **Bit-identical loss curves** vs C-demo at every measurement step (0.0000 relative delta) | OQL is **self-sufficient** — corpus -> trained organelle -> running pipeline in one .oql file |
| [E11](../../experiments/E11-connect4-win-rate-fix.md) — Connect-4 win rate | **51% -> 89%** (+38pp, +1pp above C-demo's 88% baseline); one new extern, zero new opcodes | "OQL competitive with hand-coded C" is **measured** |
| [E13](../../experiments/E13-llm-game-distillation.md) — LLM distillation | T1 = 89% in the §4.2 *neutral band* — teacher (Qwen 3.6 35B) wins 88.4% itself, so cannot lift student beyond that ceiling | Saturated-distillation regime — first publishable negative result about LLM-distillation-into-tiny-specialists |

### Standing architectural locks (held under cumulative pressure)

- **+6/-4 OQL verb surface** — E07 → E13 added six new object types or SOURCE clauses; ZERO new top-level verbs
- **Zero new VM opcodes** — E08 → E13 added 9 externs in a separate file (`microgpt_vm_natives.{h,c}`); `src/microgpt_vm.{c,h,l,y}` has 0-line diff cumulative
- **Engine surface frozen** — `src/microgpt.{c,h}` has 0-line diff across the entire OQL arc
- **Zero new build deps** — Flex/Bison fallback + curl only; same as the pre-OQL baseline

### Two key cross-experiment architectural findings

1. **Compile-time-macro silent failure mode (E09 §3.4).** Loading a checkpoint trained with different `N_EMBD`/`N_HEAD`/`BLOCK_SIZE` macros silently fails because the macros constant-fold into matmul loops. Resolved by `_microgpt_lib_for_defines` binary variant pattern. **Honoured by every downstream experiment** (E10 `oql_names`, E11 `oql_c4`, E13 same).

2. **Hidden coupling only surfaces when separation is attempted.** Two examples: corpus-coupled native references (E02 — `ref_bmi_clamped` exists because the wiring corpus says "BMI clamped"); STACK_PUSH-only `return` opcode (E08 — only visible because the agent tried to use early-exit and worked around it). Both became **named follow-ups** rather than silent expansions.

### What's still pre-registered but not measured

- [E01](../../experiments/E01-llm-head-to-head.md) — head-to-head vs frontier LLM (Anthropic API budget gated)
- [E03](../../experiments/E03-independent-curator-reproducibility.md) — independent-curator reproducibility (human curator gated; **E12 is the LLM-curator proxy** running in parallel)
- [E06](../../experiments/E06-medical-guideline-graphs.md) — medical guideline -> typed treatment graph (clinician reviewer gated)
- [E12](../../experiments/E12-llm-wiring-corpus.md) — LLM-as-corpus-source for wiring (E03 from a different curator type)
- E14 (named, not drafted) — `LLM_SOURCE` first-class object unifying E12's `FROM LLM` source and E13's procedural distillation tool

### The new closing line

The architecture is now **executable end-to-end through one declarative dialect**, with eight cross-validating measurements that the discipline holds: verb-surface lock, opcode lock, engine-surface freeze. The wiring-arc's calibrated three-bound ceiling above is unchanged; what's changed is that everything *around* that ceiling — corpus generation, organelle training, pipeline composition, audit emission — is now expressible in one .oql file. **The pre-OQL closing line said "what's left is productisation"; the OQL substrate arc has produced the operator surface productisation would need.**
