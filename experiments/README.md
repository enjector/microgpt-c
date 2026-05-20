# Experiments

Pre-registered research experiments for evolving MicroGPT-C beyond the calibrated three-bound plateau documented in [`docs/research/ORGANELLE_STATE.md`](../docs/research/ORGANELLE_STATE.md).

---

## Why this folder exists — the strategic context

### What this project actually is, today

A C99 transformer engine (~460K params/organelle ceiling) wrapped in a deterministic orchestration substrate: `OpaKanban` (working memory + cycle detection), a typed Pipeline IR with verifier + DOT renderer, and a manifold-retrieval front end. The architectural thesis — **coordination, not capacity, is the intelligence** — is validated on three classes (11 games, multi-organelle voting, NL → typed graph), with a calibrated **three-bound ceiling** of:

- **~75–80 %** on novel paraphrases (curator-, model-, domain-bounded)
- **63 %** on novel compositional generation (V1.1.0, leakage-audited held-out)
- **100 %** on anchored single-family prompts (Phase 2c clean)
- **91 % / 90 % / 88 %** on the lead games (Pentago / 8-puzzle / Connect-4)

The thing that distinguishes this project from 90 % of small-model research is the **pre-registration + leakage-audit discipline**. The Phase 13 verbatim-leak, the v1 Jaccard-near-duplicate incident, V1.0.5/V1.0.6/V1.1.0 falsifications-against-pre-registered-targets — that methodology is rarer in ML than the actual numbers it produces.

### Where the project really is, vs where it claims

Three structural bounds are conceded, and three follow-ups are *named but not done*: external pretrained embeddings (gated by the zero-dep policy), independent-curator reproducibility (the architecture's own hardest falsification), wiring-binary vote-loop full fix. V1.2.0 shipped `OpaActHalting` + `OpaFrozenInput` as **API only** — implementation commit with no measurement, gated on a customer signal that has not arrived. Productisation forked to a private companion repo on 2026-05-01, leaving the public repo as **research-only**. So: the architecture is *understood*, the directions catalogue exists ([`RESEARCH_OPA_DIRECTIONS.md`](../docs/research/RESEARCH_OPA_DIRECTIONS.md) enumerates 16 experiments), but the public arc has reached a quiet plateau.

### Five genuine research contributions (under-claimed in the repo today)

1. **A general-purpose typed-DAG verifier as a "Judge substrate."** Pipeline IR + parse → tolerant-parse → repair → verify is exactly the deterministic-verifier scaffold that 2026 neurosymbolic / agent-grounding research keeps reinventing per-domain. It is buried inside OPA but separable.
2. **Adaptive compute lifted from per-token to per-prompt across an external pipeline.** `OpaActHalting` is novel as a category — most ACT work lives inside one transformer.
3. **Two organelle classes that are genuinely complementary.** Neural (pattern matching, soft retrieval) + EML (frozen symbolic, IEEE-754 deterministic, exact extrapolation on shallow elementary closed forms). That's a clean neuro-symbolic decomposition primitive.
4. **A reproducible "honest-claim" methodology** — `tools/scaling_leakage_audit.sh`, the `RESEARCH_DISCLOSURE.md` register, pre-reg-then-measure two-commit pattern. As a methodology paper this is publishable on its own.
5. **The architectural duality observation** — OpenMythos's recurrent-block (one model iterating internally) vs OPA (many tiny models iterating externally) is a publishable framing.

### Where to evolve, ordered by leverage

Each of these directions has a dedicated experiment doc in this folder:

1. **Reposition as neurosymbolic verifier-gated agents, and run ONE head-to-head vs a frontier LLM.** Take the wiring-NL-to-typed-graph task (or SQL on BIRD, or tool-call JSON on ToolBench — pick a public benchmark with ground truth and no curator self-loop). Compare: zero-shot LLM, LLM + Pipeline IR verifier as post-hoc Judge, OPA. The interesting metric is not raw accuracy but **audit-trail-passing accuracy with bounded compute**. This intersects directly with what DeepMind / OpenAI / Anthropic are working on. → **[E01](E01-llm-head-to-head.md)**
2. **Promote Pipeline IR + verifier to a standalone library.** Right now it's a side-product of OPA. As `libpipeline_ir` (C99, typed DAG, verifier, repair, DOT) it could accept graphs emitted by *any* generator — including frontier LLMs. That dramatically widens the audience, and the "deterministic post-hoc Judge for LLM tool calls" framing is timely. → **[E02](E02-pipeline-ir-library.md)**
3. **Do the independent-curator reproducibility experiment** ([§2.3 of `RESEARCH_OPA_DIRECTIONS.md`](../docs/research/RESEARCH_OPA_DIRECTIONS.md)). It's the test you've named as the hardest falsification of your own claims and haven't run. If it survives, the three-bound ceiling becomes an *architectural* finding (publishable). If it falsifies, the result tells the field something genuinely new about curator-vocabulary specificity in bag-of-features retrieval — also publishable, arguably more so. → **[E03](E03-independent-curator-reproducibility.md)**
4. **Build one worked end-to-end neuro-symbolic demo with EML.** Today EML ships as a depth-2 noise-robust demo; bringing back a public-data prediction task (e.g. a small physics or epidemiology dataset where the underlying law is genuinely a shallow elementary form) with neural-planner → EML-predictor → IR-verifier would demonstrate the *complementary-organelle-classes* story end-to-end. Right now neural and EML live in separate doc paragraphs. → **[E04](E04-eml-neural-hybrid.md)**
5. **Open the pre-registration methodology as a public artefact.** Experiment 7.2 (pre-reg database) and 7.1 (auto-audit on commit) cost 3 weeks combined, and the output — a tool other ML projects can adopt to leakage-audit their held-out sets and track pre-reg outcomes — has reach well beyond MicroGPT-C. Reviewers love this kind of side contribution. → **[E05](E05-prereg-methodology-public.md)**
6. **Pick a real-world public-data application that exercises the three bounds.** The migrated verticals (fraud, finance risk, defence tracking) were the obvious choice but went private. A research-friendly substitute: **medical guideline → typed-treatment-graph** (audit-mandated; distinctive vocabulary; public datasets like SNOMED/UMLS available). It hits the architecture's strongest claim (audit + tiny + composable) on a domain where reviewers can verify the work. → **[E06](E06-medical-guideline-graphs.md)**
7. **Elevate the operator surface to a SQL-shaped DSL — OQL (Organelle Query Language).** Inspired by [EQL](../../EnX/EnX-Research-Prototypes/aerospike.github/cpp/enx-db/book-eql.v7/The_Expressive_Power_of_EQL.md)'s "the query language is the product" thesis. SQL + 6 verbs (`TRAIN`, `COMPOSE`, `RUN`, `EVALUATE`, `VERIFY`, `AUDIT`) - 4 verbs (`CREATE TRIGGER`, `CREATE FUNCTION`, `DECLARE CURSOR`, `SAVEPOINT`). Re-uses the existing Flex/Bison VM infrastructure. Compounds with E05 — pre-registration becomes a first-class `CREATE EXPERIMENT … WITH TARGETS …` statement instead of prose. Each of E01-E06 rewrites to ≤ 50 lines of OQL. → **[E07](E07-oql-dsl.md)**
8. **High-level researcher surface — `BEHAVIOUR` objects in OQL whose body is the VM's TypeScript dialect.** Researchers stop writing C wrappers for organelles; they write TS functions (`function eval(board: string): string { ... }`) that bind to engine primitives via `declare function` externs. Connect-4's ~500 LOC C demo collapses to ~80 lines of OQL + 4 small TS bodies. Zero new VM opcodes (extern table only). Worked target: Connect-4 win rate held within ±3 pp, then replicated to 3 more games. → **[E08](E08-oql-behaviours.md)**
9. **Make OQL actually run things — wire `RUN` / `COMPOSE` / `CREATE ORGANELLE FROM CHECKPOINT` so `connect4.oql` drives a real game loop.** Closes E08's deferred T1 measurement and unblocks E01's System C authoring. `TRAIN` honestly deferred to E10. → **[E09](E09-oql-runtime-wiring.md)**
10. **Wire OQL `TRAIN` — the last `OQL_ERR_NOT_IMPLEMENTED` stub.** OQL scripts go from "load a checkpoint and run" to "train + save + load + run" in one file. Loss-curve fidelity locked at ±10% vs the C-demo equivalent. → **[E10](E10-oql-train-wiring.md)**
11. **Close E09's Connect-4 win-rate gap (51% → ≥85%).** The wiring is correct; the gap is a prompt-protocol mismatch at `INPUT_BEHAVIOUR`. Pathway A (behaviour-side fix) preferred; Pathway B (one new VM extern, zero new opcodes) as fallback. → **[E11](E11-connect4-win-rate-fix.md)**
12. **LLM-as-corpus-source for the wiring organelle.** Local LM Studio + Pipeline IR verifier filter + standing leakage audit. Tests whether the curator-bound ceiling (`INV-WIRE-061`) is architectural or curator-vocabulary-specific — answers E03's open question via a different curator type. Adds one new SOURCE clause (`CREATE CORPUS … FROM LLM …`); zero new build deps beyond curl. → **[E12](E12-llm-wiring-corpus.md)**
13. **LLM distillation into a Connect-4 player organelle.** Ports `RESEARCH_OPA_DIRECTIONS.md` §5.1 (Experiment 4.1) to the two-commit pre-reg shape. Local LM Studio plays N games; student organelle (≤460K params) trains on (board, move) pairs; target ≥93% (vs 88% C-demo baseline). All three possible outcomes (≥93% / 88-92% / <88%) are publishable. → **[E13](E13-llm-game-distillation.md)**
14. **Unify E12 + E13's parallel LLM bridges under one OQL `LLM_SOURCE` object + shared `tools/llm_endpoint.{c,h}` transport.** Adds one new first-class object type via the existing `CREATE` verb (no 7th verb). Targets ≤600 LOC combined (current ~1200), bit-identical reproduction of both E12 and E13 outputs via cache replay, plus a third worked `paraphrase` mode demonstrating the dispatch pattern absorbs creation/game-play/transformation shapes cleanly. → **[E14](E14-oql-llm-source-unification.md)**

### What we explicitly will NOT prioritise

For honest-disclosure discipline (matching [`RESEARCH_OPA_DIRECTIONS.md`](../docs/research/RESEARCH_OPA_DIRECTIONS.md) §10):

- **Scaling individual organelles past ~1M params** — contradicts the thesis (`INV-WIRE-041`).
- **More games** — the 11-demo set is sufficient evidence; adding game #12 doesn't change any reviewer's mind.
- **More wiring-organelle corpus engineering** — you've documented the ceiling; pushing it without a model-class change is diminishing returns and risks another leakage incident.
- **Productisation moves on this repo** — that work has moved to `organelles.bio`. Keep the public repo focused on the architectural research.

### The one-sentence reframe

> *"A composable, audit-trail-native, verifier-gated substrate for neurosymbolic systems — with a pre-registration methodology rigorous enough to publish, two complementary organelle classes (neural + symbolic EML), and an external-iteration architectural dual to recurrent-depth transformers."*

That framing is what the repo already supports; it just isn't the framing it leads with. Today it leads with "tiny composable LEGO blocks for the edge," which underplays the real contribution. The six experiments below are the moves that make the reframe earnable.

---

## Methodology — the two-commit pattern

Each experiment follows the project's **pre-register-then-measure** discipline established in [`docs/research/RESEARCH_PIPELINE_IR.md`](../docs/research/RESEARCH_PIPELINE_IR.md) §40/§42/§45:

1. `feat(research): pre-registered Experiment <ID> — <name>` — implementation/scaffold with locked hypothesis + targets + floors + skip rules, **no measurement output**.
2. `research(<area>): Experiment <ID> measurement vs pre-reg targets` — measurement output + verdict (PASS / PARTIAL / FALSIFIED).

## Document structure

Every experiment doc is a single markdown file with four sections:

| Section | When written | What goes in |
|---|---|---|
| **1. Proposal** | Pre-reg commit | Hypothesis (locked), mechanism, pre-reg targets, floors, skip rules, cost, falsification risk, what it's NOT testing, cross-refs |
| **2. Initial state** | Pre-reg commit | Baselines to beat, what's currently known, dependencies, blockers |
| **3. Implementation + results** | Measurement commit | What was built, raw numbers, links to artefacts |
| **4. Conclusion** | Measurement commit | Verdict against each pre-reg target, lessons, next moves, traceability updates |

## The fourteen experiments

| ID | Title | Direction | Cost | Falsification risk |
|---|---|---|---|---|
| [E01](E01-llm-head-to-head.md) | Head-to-head vs frontier LLM on a public typed-graph benchmark | Position OPA as a neurosymbolic verifier-gated agent | ~6-8 wk | Medium |
| [E02](E02-pipeline-ir-library.md) | Promote Pipeline IR + verifier to a standalone C99 library | Widen the audience for the project's most distinctive component | ~3-4 wk | Low |
| [E03](E03-independent-curator-reproducibility.md) | Independent-curator reproducibility of the v2 anchor library | Architecture's hardest unrun falsification test | ~4-6 wk (+ 2nd person) | **High** |
| [E04](E04-eml-neural-hybrid.md) | End-to-end neural + EML hybrid on a public physics dataset | Worked example of complementary organelle classes | ~4-5 wk | Medium |
| [E05](E05-prereg-methodology-public.md) | Open the pre-reg + leakage-audit methodology as a public artefact | Methodology paper + tooling — reach beyond MicroGPT-C | ~3 wk | Low |
| [E06](E06-medical-guideline-graphs.md) | Medical guideline → typed treatment graph on public data | Real-world public-data application exercising the three bounds | ~8-10 wk | Medium-high |
| [E07](E07-oql-dsl.md) | OQL — a SQL-shaped DSL for organelles, pipelines, behaviours, and experiments | Replace 3 surfaces (C demos + scripts + markdown specs) with one declarative dialect | ~6-8 wk | Medium |
| [E08](E08-oql-behaviours.md) | VM TypeScript dialect as the body of OQL `BEHAVIOUR` objects | High-level researcher surface — Connect-4 ~500 LOC C → ~80 lines OQL+TS, zero new VM opcodes | ~4-5 wk | Medium |
| [E09](E09-oql-runtime-wiring.md) | Wire OQL `RUN` / `COMPOSE` / `CREATE ORGANELLE FROM CHECKPOINT` end-to-end | Make `oql run connect4.oql` actually drive a game loop; closes E08's T1 | ~5-7 wk | Medium |
| [E10](E10-oql-train-wiring.md) | Wire OQL `TRAIN` so scripts can train organelles from scratch | Last `OQL_ERR_NOT_IMPLEMENTED` stub closes; loss-curve fidelity ±10% vs C-demo | ~3-5 wk | Medium |
| [E11](E11-connect4-win-rate-fix.md) | Close E09's Connect-4 win-rate gap (51% → ≥85%) | Behaviour-side prompt-protocol fix (Pathway A) or single new VM extern (Pathway B) | ~2-3 wk | Low-medium |
| [E12](E12-llm-wiring-corpus.md) | LLM-as-corpus-source for wiring (NL → typed-graph), via local LM Studio + verifier filter | Tests curator-bound ceiling from a different curator type; answers E03 indirectly | ~2-3 wk | Medium |
| [E13](E13-llm-game-distillation.md) | LLM distillation into Connect-4 player organelle, via local LM Studio | Ports `RESEARCH_OPA_DIRECTIONS.md` §5.1; ≥93% Connect-4 target, all outcomes publishable | ~2-4 wk | Medium |
| [E14](E14-oql-llm-source-unification.md) | Unify E12 + E13's LLM bridges under one `LLM_SOURCE` OQL object + shared transport | One declarative surface for all design-time LLM use; targets >50% LOC reduction + bit-identical reproductions + new paraphrase mode | ~2-3 wk | Medium |

## Status legend

- 📋 **Proposal locked** — pre-reg section frozen; awaiting work.
- 🔬 **In flight** — implementation underway; results pending.
- ✅ **Measured: PASS** — pre-reg targets met or exceeded.
- ⚠️ **Measured: PARTIAL** — some targets met, some floors held.
- ❌ **Measured: FALSIFIED** — pre-reg targets not met; falsification recorded honestly.
- 🚫 **Cancelled** — skip rule triggered before completion.

## Current status

All eight: 📋 **Proposal locked** (2026-05-20).

Measured (worktree-branch agent runs, all merged into main):
- E02 ✅ merged at `a02a71d` — 5/6 targets PASS; T6 deferred on API key
- E04 ✅ merged at `05932dc` — 7/7 targets PASS at σ=0.05
- E05 ✅ merged at `6aba1c8` — 6/6 targets PASS; methodology paper draft 13 pages
- E07 ✅ merged at `83e5673` — verb-discipline lock held; 5 PASS, 1 PARTIAL, 2 deferred
- E08 ✅ merged at `e9b8620` — 3 PASS (T2 18.5% LOC, T3 zero new opcodes, T4 +4 tests); 5 honestly deferred
- E09 ✅ merged at `d4eb478` — 6 PASS; T2 PARTIAL (51% win-rate) and T8 PARTIAL — both closed by E11/follow-ups below
- E10 ✅ merged at `4824900` — **all 8 targets PASS; T3 and T4 bit-identical to C-demo baseline** (0.0000 relative delta at every loss-curve sample step, 0.000000e+00 per-logit on round-trip)
- E11 ✅ merged at `d6593aa` — **6 PASS, T1 = 89%** (+38pp vs 51% E09 baseline, +1pp parity with 88% C-demo baseline); T4 PARTIAL (RNG-path artefact, documented); Pathway B chosen (one new extern `c4_model_propose_column`)
- E12 ✅ merged at `de9ea6d` + follow-up — **T4 MEASURED at 0/20 (FALSIFIED)** via agent-built `tools/e12_eval_v2.c`. Lands in §1.2 "<11/20" corner: literal reading favours human-curator-is-load-bearing claim. Honest framing (§3.9): falsifies the **combined** hypothesis (LLM curator + 100 examples + 489K char-level model), NOT LLM-as-curator in isolation. Corpus-size confound (100 vs 400 examples) named as principal limitation. **Path to a meaningful T4 follow-up named in §4:** scale corpus to 10k+ examples (12-15h overnight) or substitute a non-thinking model — one-command rerun. Key Qwen3 finding: thinking-model output lands in `reasoning_content`, not `content`, when max_tokens hits
- E13 ✅ merged at `c7f030d` — 7 PASS, 2 PARTIAL. **T1 = 89% in §4.2 neutral band**. LLM teacher itself wins 88.4% vs random → saturated-distillation regime → tiny-specialist thesis robust but unboosted. E13d (OQL TRAIN vocab-mismatch bug) named as follow-up

Awaiting external inputs:
- E01 — needs Anthropic API budget (~$100)
- E03 — needs independent human curator
- E06 — needs clinician reviewer

## Cross-references

- [`docs/research/ORGANELLE_STATE.md`](../docs/research/ORGANELLE_STATE.md) — synthesis: where the architecture is today
- [`docs/research/RESEARCH_OPA_DIRECTIONS.md`](../docs/research/RESEARCH_OPA_DIRECTIONS.md) — full 16-experiment catalogue; the six here are the high-leverage subset
- [`docs/research/RESEARCH_OPENMYTHOS_CROSS_POLLINATION.md`](../docs/research/RESEARCH_OPENMYTHOS_CROSS_POLLINATION.md) — three pre-registered cross-pollination experiments (some already graduated in V1.2.0)
- [`docs/engineering/CLEAN_ROOM_IMPLEMENTATION/RESEARCH_DISCLOSURE.md`](../docs/engineering/CLEAN_ROOM_IMPLEMENTATION/RESEARCH_DISCLOSURE.md) — regulator-friendly disclosure register; outcomes land here too
- [`tools/scaling_leakage_audit.sh`](../tools/scaling_leakage_audit.sh) — standing audit infrastructure
