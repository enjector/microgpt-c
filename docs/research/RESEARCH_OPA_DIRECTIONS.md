# OPA research directions — modifications, enhancements, and pre-registered experiments

> Catalogue of research directions for the Organelle Pipeline Architecture itself, organised by where they bite. Independent of (but cross-references) the OpenMythos cross-pollination set in `RESEARCH_OPENMYTHOS_CROSS_POLLINATION.md`. Each direction names concrete experiments with locked targets, skip rules, and abandon-triggers per the project's pre-register-then-measure discipline.

**Status:** **Catalogue, not commitment.** Not all of these will ship; the prioritisation matrix in §9 names which are the highest-leverage and which are flagged as research-only. Seven directions, sixteen experiments. Each experiment is independently sized; none assume any other has run.

**Scope:** OPA — the multi-organelle Kanban + Pipeline IR + verifier + retrieval stack. Excludes pure-transformer research (which lives in `RESEARCH_DEEPSEEK_V4_*.md` for the V4-port stack and `RESEARCH_OPENMYTHOS_CROSS_POLLINATION.md` for looped-block ideas) unless the experiment is OPA-coordination-shaped.

---

## 1. Where OPA stands today (load-bearing summary)

| Aspect | Calibrated state |
|---|---|
| Multi-organelle game pipelines | 11 demos production-stable; Pentago 91 %, Connect-4 88 %, 8-puzzle 90 %, etc. |
| Wiring NL → graph | 100 % anchor (single-family); 70 % multi-stage composition; ~75-80 % novel-paraphrase retrieval (calibrated three-bound ceiling) |
| Compositional search (V1.0.7c Phase 6c) | 50 % correct on 30-prompt held-out — the original SLO-WIRE-005 design target |
| Cross-organelle wire format | Pipe-separated text strings per `FS_organelle_wire.md` (audit-trail-explicit; gradient-non-differentiable across organelles) |
| Verifier output type | Binary pass/fail (`PIPE_OK` / `PIPE_ERR_*`) |
| KV memory | TurboQuant 4-bit + RotorQuant + MSA covering edge deployment |
| Curator-side surface | Manual editing of anchor/fragment/synonym tables (~30 min/family per `chapter 19`) |
| Standing protections | `tools/scaling_leakage_audit.sh` + `INV-WIRE-060/061/062` + `RESEARCH_DISCLOSURE.md` |

Three structural bounds per `ORGANELLE_STATE.md` §"Three structural bounds":
- **Curator-bounded** — synonym tables limit retrievability of unseen vocabulary.
- **Model-bounded** — bag-of-features classifiers (any n-gram variant) hit the same ceiling on novel-paraphrase tests.
- **Domain-bounded** — distinctive-noun domains hit the upper bound; generic-vocabulary domains drop to ~15 %.

Open follow-ups documented but not done: external-embedding integration; independent-curator reproducibility; the wiring binary's full vote-loop fix (V1.0.7 used the surgical-rollback path).

---

## 2. Direction 1 — Inter-organelle protocol

The pipe-separated text format `KEY=val|KEY2=val2` (`FS_organelle_wire.md`) is the canonical Kanban handoff today. It is audit-explicit but discretisation-bound; alternatives carry tradeoffs that need explicit measurement.

### 2.1 Experiment 1.1 — Latent-tensor handoffs (cross-link)

Pre-registered as Experiment D in `RESEARCH_OPENMYTHOS_CROSS_POLLINATION.md` §2.3. Lifts continuous latent state across organelle boundaries, ADR-gated for vertical adoption due to audit-trail tradeoff. **Cost:** ~4 weeks. **Falsification risk:** medium. See the linked note for full pre-reg.

### 2.2 Experiment 1.2 — Typed-tensor handoffs (audit-preserving alternative to 1.1)

**Hypothesis (locked):** A *structured* latent representation that pairs a continuous tensor with a typed schema (rendered alongside the value as a Pipeline IR DOT trace) preserves audit while enabling some of the gradient-flow benefit. Concrete form: each emit is `(tensor, type_schema, structural_summary)` where `tensor` is the continuous payload, `type_schema` is the Pipeline IR type signature the recipient organelle must consume, and `structural_summary` is a deterministic projection of the tensor into a finite vocabulary auditable like the existing pipe-string. The audit trace renders both the structural summary (for regulators) and the tensor shape/type (for cross-checks); the consumer organelle has access to the tensor for differentiable processing.

**Pre-registered targets:** ≥ 1.5 pp lift over text-string baseline on at least 2 of 3 games (8-puzzle, Connect-4, Mastermind), with **100 % audit-trace coverage retained** (no audit-floor compromise needed, unlike Experiment 1.1).

**Cost:** ~6 weeks. Higher than 1.1 because the typed-schema design is genuinely novel — there's no off-the-shelf template. Falsification risk: medium-high (the structural-summary projection may discretise the gradient enough to lose the differentiability benefit).

**Skip rule:** if 1.2 lifts game solve rates by less than 1 pp, the typed-tensor approach is documented as a falsified middle-ground. Either fall back to text-strings (default) or go to pure latent (Experiment 1.1, ADR-gated). No silent middle-ground.

### 2.3 Experiment 1.3 — Vote-aggregation strategies beyond best-of-N

**Hypothesis (locked):** Replacing the wiring binary's best-of-16 self-consistency with one of three principled alternatives lifts the calibrated 75-80 % ceiling on at least one held-out test:

1. **Probabilistic aggregation** — each candidate carries a posterior; the Judge picks by maximum-marginal-likelihood under the verifier's typing constraints rather than majority vote.
2. **Learned ranker organelle** — a small (~30K param) Judge organelle trained on (candidate, ground-truth-correct) pairs from the existing audit logs; this is a Judge that *learns* to pick rather than verify.
3. **Geodesic ensemble** — candidates are embedded in the geodesic state space and the Judge picks by Mahalanobis distance to the prompt's predicted family centroid.

**Pre-registered targets:** any one of (1)/(2)/(3) lifts v2 retrieval from 16/20 to ≥ 18/20 OR Phase 3b composition from 7/10 to ≥ 9/10, *with the bag-of-features audit invariant maintained* (`INV-WIRE-060`).

**Cost:** ~3 weeks per variant, ~9 weeks for all three. **Falsification risk:** medium for (1) and (3), high for (2) — the learned ranker has a real chance of just memorising the audit logs and failing to generalise. Per-variant skip rules locked in advance.

---

## 3. Direction 2 — Curator-side automation

Per the calibrated three-bound claim, the curator is the binding constraint on the architecture's distinctive value. Reducing curator labour without losing audit explicitness is high-leverage.

### 3.1 Experiment 2.1 — Curator-as-organelle (active learning loop)

**Hypothesis (locked):** A small (~50K param) "curator" organelle trained on the existing `wiring_anchor_graphs.c` / `wiring_fragments.c` corpus can *propose* new anchor entries from observed production prompts that fail anchor retrieval. Human-in-the-loop accepts/rejects each proposal; accepted proposals enter the anchor library after passing the standing leakage audit.

**Pre-registered targets:**
- On a held-out batch of 50 production-style prompts that the current 20-family library misses, the curator organelle proposes ≥ 35 anchor candidates that a human curator accepts as valid (≥ 70 % proposal acceptance rate).
- Latency: per-proposal generation ≤ 2 seconds on the reference machine.
- The standing leakage audit MUST still gate every proposed anchor before it enters the library — automation does not bypass the audit.

**Cost:** ~5 weeks. New `demos/curator_organelle/` plus modifications to the OpaKanban to support a "review queue" stage.

**Skip rule:** if proposal acceptance is < 50 %, the curator-as-organelle approach is falsified at our scale and the curator's hand stays the canonical mechanism. Documented as a research-only result.

### 3.2 Experiment 2.2 — External-vocabulary ingestion (WordNet / ConceptNet / domain glossaries)

**Hypothesis (locked):** Augmenting `corpus_expand.c` synonym tables from WordNet (or a domain glossary like the 200-term fraud-vocabulary list in `docs/PRODUCT_FRAUD_DETECTION.md` migrated to organelles.bio) lifts TF-IDF retrieval on adversarial axis-2 by ≥ 3/20.

**Pre-registered targets:** TF-IDF adversarial axis-2 ≥ 18/20 → ≥ 19/20; v2 retrieval ≥ 16/20 (no regression); audit invariant `INV-WIRE-061` (distinctive-noun bound) unaffected.

**Cost:** ~2 weeks. Requires either embedding a WordNet C library (Category B dependency per organelles.bio's `DEPENDENCY_POLICY.md`) or vendoring a static text dump.

**Skip rule:** if the lift is below 1 pp, the external vocabulary expansion is falsified — synonym distinctiveness, not vocabulary size, is what binds (consistent with `RESEARCH_DISCLOSURE.md` §7).

### 3.3 Experiment 2.3 — Independent-curator reproducibility (already-named follow-up)

**Hypothesis (locked):** A second, independent curator (different person, no access to the existing anchor/synonym tables) rebuilding v2's 20-family library from scratch achieves a TF-IDF retrieval rate within ±5 pp of the current 16/20.

**Pre-registered targets:** independent curator's library scores 14-18/20 on the v2 held-out — i.e. the calibrated 75-80 % ceiling is robust to curator identity.

**Cost:** ~4-6 weeks of an independent curator's time. Bottleneck is finding the second person, not engineering.

**Skip rule (and the hardest one to honour):** if the independent curator's library scores < 11/20 (more than 5 pp below current), the calibrated ceiling is *curator-vocabulary-specific*, not architectural — a much weaker claim than what we currently report. This would require a substantial rewrite of `RESEARCH_DISCLOSURE.md` §7 and `INV-WIRE-061`. Pre-registered honestly so we can't quietly retreat from the result.

---

## 4. Direction 3 — Probabilistic Judge (ties to productisation)

Already pre-registered in the productisation roadmap (organelles.bio's `PRODUCT_FINANCE_RISK.md`) as the architectural change required for finance regime/risk. Listing here so the research-side dependency is visible.

### 4.1 Experiment 3.1 — `pipeline_verify_with_confidence` extension

**Hypothesis (locked):** Extending the verifier from binary pass/fail to (pass, confidence ∈ [0, 1], evidence list) preserves all existing test passes (51/51 unit tests stay green) AND enables a new class of probabilistic Judge consumers downstream. Critical for finance vertical (regime classifier outputs are distributions); useful for fraud (graded suspicion levels); useful for defence (target identification confidence).

**Pre-registered targets:** all 51 existing pipeline IR unit tests pass under the extended API; new `PipelineVerifyResult` struct documented in `BS_pipeline_ir.md` with `INV-PIPE-NEW`; one worked example (a probabilistic sentiment classifier) demonstrates the extension end-to-end.

**Cost:** ~6-8 weeks. This is genuine new mechanism, not just plumbing — the calibration story alone (Brier score, evidence attribution) is a research project.

**Skip rule:** if the extension breaks any existing test, the probabilistic verifier is shipped as a *parallel* API (`pipeline_verify_v2`) leaving the binary version untouched. If the calibration cannot reach Brier ≤ 0.15 on held-out periods, the calibration story is research-only and the verifier ships as "uncalibrated probabilistic" with a flag.

---

## 5. Direction 4 — Cross-organelle learning

OPA today trains each organelle independently. Experiments here test whether organelles can learn *from each other*.

### 5.1 Experiment 4.1 — Distillation from a frontier LLM into role-specialised organelles

**Hypothesis (locked):** Distilling a frontier LLM (Claude / GPT / Gemini, accessed via API at training time only — not at inference time) into role-specialised organelles for a single game (Connect-4 as the worked example) lifts the win rate from the current 88 % to ≥ 93 %.

**Pre-registered targets:** Connect-4 win rate ≥ 93 %; Mastermind solve rate ≥ 84 %; both organelles still ≤ 460K params and ≤ 5 ms p99 inference (no model-size growth).

**Cost:** ~6 weeks. The LLM API access is a build-time-only dependency under organelles.bio's Category B (the runtime is unaffected). Falsification risk: medium — distillation may not transfer a 100B-param model's structure into a 460K-param student, especially for adversarial-game reasoning.

**Skip rule:** if win rate is below current baseline, distillation is falsified at our scale and the result is documented as "OPA's tiny-specialist thesis is robust to LLM distillation attempts." If the lift is meaningful but only by re-training to ≥ 5M params, the result still contradicts the tiny-specialists thesis and the technique is documented as research-only.

### 5.2 Experiment 4.2 — Federated training between independently-trained organelles

**Hypothesis (locked):** Two organelles trained on different game datasets (e.g. one trained on Connect-4 only, one on Othello only) can have their weights combined via parameter averaging or model soup to produce a cross-game organelle that performs at ≥ 80 % of single-game baseline on both games.

**Pre-registered targets:** soup organelle ≥ 70 % win rate on Connect-4 (baseline 88 %); ≥ 54 % win rate on Othello (baseline 67 %).

**Cost:** ~3 weeks. `model_soup_average` already exists in the engine; this is mostly experiment-design work.

**Skip rule:** if soup organelle is < 50 % on either game, model averaging is falsified for cross-game transfer in OPA. If it works, opens up "soup as a curation primitive" — production deployments could combine vertical-specialist organelles to cover overlapping use cases.

---

## 6. Direction 5 — Pipeline IR extensions

The Pipeline IR is the load-bearing primitive. Two safe extensions are worth pre-registering.

### 6.1 Experiment 5.1 — Conditional / branching graphs

**Hypothesis (locked):** Adding `if/else` typed-condition nodes to the Pipeline IR (current grammar is strict DAG, no conditionals) extends what can be expressed without breaking any existing graph. Worked example: a fraud rule that says "if MCC ∈ {gas pumps}, run gas-skim-pattern check, else skip" — currently expressed clumsily by always running both branches and selecting by post-hoc filter.

**Pre-registered targets:**
- All 51 existing pipeline tests pass under the extended grammar.
- New `IfNode` type in `BS_pipeline_ir.md` with type-checking that both branches return the same output type.
- DOT renderer produces a readable conditional diamond shape.
- Verifier rejects `IfNode` with mismatched branch types.

**Cost:** ~3 weeks. New `pipeline_ir_text.md` v2 grammar (per `GAP-PIPE-001` already deferred for V2.0).

### 6.2 Experiment 5.2 — Recursive sub-graph references

**Hypothesis (locked):** Allowing `@graph` nodes to reference *other* `@graph` definitions as primitives (modular composition) lets the wiring organelle express deeper nesting without exploding the anchor library size.

**Pre-registered targets:** the 20-family anchor library expressed as ≤ 8 atomic primitives + composition references, with Phase 2c clean retrieval unchanged at 20/20.

**Cost:** ~4 weeks. Touches `microgpt_pipeline.{h,c}` parse + verify + execute paths. Risk: cycle detection has to extend across sub-graph references (currently DAG-only in a single graph).

---

## 7. Direction 6 — Edge-deployment benchmarks (closes a deferred gap)

`GAP-PERF-001` (DEFERRED) tracks the absence of real edge-device benchmarks. This is a measurement direction, not a research direction strictly speaking, but it gates the architecture's edge story.

### 7.1 Experiment 6.1 — Raspberry Pi 4 / 5 reference benchmarks

**Hypothesis (locked):** All 11 game demos plus the wiring organelle plus the manifold TF-IDF demo run on Raspberry Pi 4 and 5 with documented latency / RAM / power numbers, without any code changes (i.e. our edge-deployable claim survives a real edge platform).

**Pre-registered targets (RPi 5):**
- All demos build cleanly with `bootstrap.sh`.
- 8-puzzle solve latency ≤ 5x the M2 Max reference machine.
- Wiring binary `--clean-only` p99 ≤ 50 ms (vs ≤ 5 ms on M2 Max).
- TurboQuant compression maintains the 8x memory reduction at the same encode rate ratio.

**Cost:** ~2 weeks. Most of the cost is hardware procurement + setup; the engineering is automated CI.

**Skip rule:** if any demo fails to build or runs > 10x slower than the reference machine, the edge-deployable claim narrows to "M-class CPUs and equivalents only" — a real and load-bearing change to the strategy story.

### 7.2 Experiment 6.2 — ESP32 micro-deployment of one minimal organelle

**Hypothesis (locked):** A single ~30K-param minimal organelle (Klotski or Lights Out — the smallest in the demo set) can be quantised to INT8, statically linked, and run inference on an ESP32-S3 (~520 KB SRAM) within the chip's RAM budget.

**Pre-registered targets:** binary fits in flash; runtime fits in 400 KB SRAM (leaving headroom); inference per move ≤ 200 ms.

**Cost:** ~6 weeks. INT8 ABI + checkpoint format work (some of which is in `GAP-INT8-001` already DEFERRED). Falsification risk: medium-high — INT8 quantisation may degrade the Klotski 62 % solve rate below 30 %, in which case ESP32 deployment is research-only at this organelle scale.

**Skip rule:** if quantisation degrades solve rate by > 50 %, the ESP32 deployment story narrows to "stem cell foundation only — no useful organelles run there yet."

---

## 8. Direction 7 — Honest-claim infrastructure

The audit + pre-registration discipline is the project's most distinctive trait. Two experiments tighten the discipline itself.

### 8.1 Experiment 7.1 — Auto-audit on every commit

**Hypothesis (locked):** A pre-commit (or CI) hook that runs `tools/scaling_leakage_audit.sh` on any PR touching a held-out test file or a synonym table catches future leakage incidents before the commit lands. Mechanism: the hook fails the commit if Audit B (Jaccard ≥ 0.7) crosses the per-file threshold defined in a new `tools/leakage_audit_thresholds.json`.

**Pre-registered targets:** the hook catches the v1 leakage incident if applied retroactively to commit `5a478bc` (the post-Phase-3 cleanup pass). Zero false positives on the current main branch's audit runs.

**Cost:** ~1 week. CI integration via `.github/workflows/leakage-audit.yml`.

### 8.2 Experiment 7.2 — Pre-registration database

**Hypothesis (locked):** A searchable index of every `RESEARCH_*.md` pre-registration + outcome, machine-extractable from the existing markdown via a small parser, enables (a) dashboard visibility on which pre-regs have unresolved outcomes, (b) auto-cross-references when a new experiment's hypothesis overlaps a previous one, (c) early-warning for would-be-retroactive rationalisations.

**Pre-registered targets:** parser extracts pre-reg blocks from all `RESEARCH_*.md` and `wiring_scaling_*.md` files; produces a summary table with status per pre-reg (pre-registered / measured-pass / measured-fail / cancelled). Tracked in a new `tools/research_status_dashboard.{c,sh}`.

**Cost:** ~2 weeks. Pure C99 markdown parser + simple table output (no dependencies). Falsification risk: low — this is engineering, not science.

---

## 9. Prioritisation matrix

| Experiment | Cost | Falsification risk | Research payoff | Productisation payoff |
|---|---|---|---|---|
| **OpenMythos A — ACT halting** (cross-link) | ~2 wk | Low | Low | **Very high** (sub-1 ms easy-prompt latency, fraud Phase 1) |
| **OpenMythos B — LTI vote-loop** (cross-link) | ~2 wk | Medium | Medium | High (closes `GAP-WIRE-003` properly) |
| **3.1 — Probabilistic verifier** | ~6-8 wk | Medium-low | High | **Very high** (gates finance vertical) |
| **2.3 — Independent-curator reproducibility** | ~4-6 wk (+ 2nd person) | High (genuine falsification possibility) | **Very high** | Medium (calibrates the productisation pitch) |
| **1.1 — Latent inter-organelle handoffs** (= OpenMythos D) | ~4 wk | Medium | High | Conditional (vertical-ADR-gated) |
| **6.1 — RPi 4/5 benchmarks** | ~2 wk | Low | Low | High (edge story validated) |
| **7.1 — Auto-audit on commits** | ~1 wk | Low | Low | High (productisation-grade hygiene) |
| **2.1 — Curator-as-organelle** | ~5 wk | Medium-high | Medium | Medium (cuts curator labour for vertical onboarding) |
| **5.1 — Conditional graphs** | ~3 wk | Low | Medium | Medium (better fraud rule expressiveness) |
| **4.1 — LLM distillation** | ~6 wk | Medium | High | Low (build-time-only dep is OK; research-side use) |
| **2.2 — External vocabulary (WordNet)** | ~2 wk | Medium | Medium | Medium |
| **1.3 — Vote-aggregation strategies** | ~3-9 wk | Medium per variant | High | Medium |
| **OpenMythos C — RDT for planners** | ~3 wk | High | Medium | None |
| **6.2 — ESP32 micro-deployment** | ~6 wk | Medium-high | Medium | Conditional (ESP32-class verticals) |
| **5.2 — Recursive sub-graphs** | ~4 wk | Low | Medium | Low |
| **1.2 — Typed-tensor handoffs** | ~6 wk | Medium-high | Medium | Conditional |
| **4.2 — Federated training (model soup)** | ~3 wk | Medium | Medium | Low |
| **7.2 — Pre-reg database** | ~2 wk | Low | Low | High (productisation-grade hygiene) |

**Recommended top-5 to commit to (in order):**

1. **7.1 — Auto-audit on commits** (1 wk, low risk, productisation hygiene that prevents future v1-style incidents).
2. **OpenMythos A — ACT halting** (2 wk, productisation latency unblock).
3. **6.1 — RPi 4/5 benchmarks** (2 wk, validates edge story before any vertical demo claims it).
4. **3.1 — Probabilistic verifier** (6-8 wk, gates finance vertical entirely; longest cost but highest gate to clear).
5. **2.3 — Independent-curator reproducibility** (4-6 wk, the most important *honest-claim* check the project has not yet done; could falsify the calibrated three-bound claim — which is itself the most valuable possible outcome).

Items 6.2 (ESP32), 4.1 (LLM distillation), 4.2 (federated soup), 1.1/1.2 (latent handoffs), and 5.1/5.2 (Pipeline IR extensions) are *interesting research that is not blocking productisation*. Schedule them when there is engineering capacity beyond the top-5.

---

## 10. What we explicitly will NOT pursue

For honest-disclosure discipline, rejections are documented as carefully as acceptances.

| Direction | Why not |
|---|---|
| Scaling individual organelles past 5M params | Contradicts the tiny-specialists thesis (`INV-WIRE-041`); the architecture's distinctive value is composition, not capacity |
| Replacing the wiring transformer with a frontier LLM as the front-line mechanism | Would convert OPA into "a thin orchestration layer over a cloud LLM" — defeats the audit/edge/composition pitch |
| Adding a chat-style conversation interface to organelles | Out of scope; the architecture is a pipeline, not a conversational agent. Belongs in a separate downstream product, not in OPA core |
| Reinforcement-learning training of organelles on game self-play | Considered and rejected per the earlier RL-for-composition discussion: bottleneck is curator-bounded library size, not policy. RL would only earn its keep once the action space genuinely needs policy optimisation, not before |
| Open-domain natural-language understanding | Per the calibrated three-bound domain ceiling: the architecture is not a general NLU system. Vertical productisation (private companion repo) targets distinctive-noun domains specifically |
| Convolutional / vision primitives in the core engine | Out of scope unless a defence vertical (per the migrated productisation plans) actually commits; sensor adapters belong in vertical-specific repos, not the research core |

---

## 11. Cross-references

| Topic | Source |
|---|---|
| Where the research stands today | `ORGANELLE_STATE.md` |
| Calibrated three-bound retrieval claim | `wiring_scaling_post_phase3.md`, `RESEARCH_DISCLOSURE.md` §7 |
| OpenMythos-side experiments (A/B/C/D) | `RESEARCH_OPENMYTHOS_CROSS_POLLINATION.md` |
| Wiring binary vote-loop bug (Direction-aligned) | `wiring_binary_phase8_regression.md`, `GAP-WIRE-003` in `TRACEABILITY.md` |
| Productisation plans that several experiments here gate | `docs/MIGRATED_TO_ORGANELLES_BIO.md` (companion private repo) |
| Pre-registration discipline this catalogue follows | `RESEARCH_PIPELINE_IR.md` §40, §42, §45 |
| Standing audit infrastructure | `tools/scaling_leakage_audit.sh`, `INV-WIRE-062` |
| Honest-disclosure register (where outcomes land) | `docs/engineering/CLEAN_ROOM_IMPLEMENTATION/RESEARCH_DISCLOSURE.md` |

---

## 12. Status

**Catalogue, not commitment.** Sixteen experiments enumerated, five recommended for the next planning window. None implemented. Each will follow the project's pre-register-then-measure two-commit discipline if it graduates from "catalogue" to "in flight":

1. `feat(research): pre-registered Experiment <X.Y> — <name>` with implementation but no measurement output.
2. `research(<area>): Experiment <X.Y> measurement vs §<Z> pre-reg targets` with the measurement output and verdict.

The catalogue itself is intentionally separate from `ORGANELLE_STATE.md` (which describes *where we are*) — this document describes *where we could go*, with the structural discipline that keeps us honest about which moves earn their cost.

— Pre-registered 2026-05-01.
