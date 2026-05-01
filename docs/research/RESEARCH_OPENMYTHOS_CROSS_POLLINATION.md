# Cross-pollination from OpenMythos to OPA — three pre-registered experiments

> Assessment of which OpenMythos (Recurrent-Depth Transformer) techniques port cleanly to the Organelle Pipeline Architecture, ranked by leverage. Three experiments pre-registered with skip rules; six techniques explicitly **rejected** with reasons. Status: **research note only — no code shipped**. Each pre-registered experiment is an independent ~1-3 week investment that survives or dies on its own pre-registered targets.

**Reference:** [OpenMythos class reference] — `OpenMythos`, `RecurrentBlock`, `LTIInjection`, `ACTHalting`, `LoRAAdapter`, `MoEFFN`, `MLAttention`. Underlying papers: Loop-Think-Generalize (2025), Parcae (2026), Saunshi 2025, Graves 2016, Bae 2024.

**Status:** No flag, no code. Pre-registration only, in the spirit of `RESEARCH_PIPELINE_IR.md` §40 / §45 / §46 discipline. If any of these experiments graduate from "pre-registered" to "implemented", they each get their own dedicated `RESEARCH_OPENMYTHOS_<EXPERIMENT>.md` follow-on per the project's pre-register-then-measure pattern.

---

## 1. Spear Summary

**Point:** OpenMythos's *pipeline-level* ideas (ACT halting; LTI-stable iteration) port cleanly to OPA with high leverage. Its *model-internal* ideas (looped Recurrent-Block; intra-model MoE; MLA) need orders-of-magnitude more model width than our 30K–540K-param organelles have, and conflict with the tiny-specialists thesis the architecture stands on. The cross-pollination is therefore **pipeline-side, not model-side** — and the most useful import is the *adaptive-compute* concept moved up a level from "per token within one model" to "per prompt across the multi-organelle pipeline."

**Picture:** OpenMythos solves *expressiveness* by looping a single transformer block with a LoRA adapter and ACT halting per token position. OPA solves the same problem differently — by composing many tiny specialised organelles coordinated by a deterministic Judge. The two architectures are dual: OpenMythos has one big model with internal iteration; OPA has many small models with external iteration. The techniques that survive the translation are the ones that govern *how iteration unfolds* (halting, stability), not the ones that govern *what gets iterated on* (model width, MoE depth, attention compression — already addressed orthogonally in our stack).

**Proof:** Three pre-registered experiments, listed below. Six rejected techniques, listed in §5 with explicit reasons (not just "not relevant" — *what would have to change for them to apply*).

---

## 2. Pre-registered experiments — locked targets and skip rules

### 2.1 Experiment A — ACT halting at the OPA pipeline level

**Hypothesis (locked before measurement):** Lifting the ACT halting mechanism from "per-token within one transformer" to "per-prompt across the multi-organelle pipeline" delivers ≥ 5× latency reduction on the easy-prompt majority while preserving the calibrated 75-80 % retrieval ceiling on the hard-prompt minority.

**Mechanism:** A single linear head over the current pipeline-stage output produces a per-prompt halting probability. The wiring binary's pipeline (anchor retrieval → fragment composition → wiring transformer → Judge vote) is gated by this probability. Threshold tuned to match `cfg.act_threshold = 0.99` from the OpenMythos default.

| Stage | What halts when threshold crossed |
|---|---|
| Anchor classifier (Geo top-1) | Skip fragment composition + wiring transformer + Judge re-vote — return the anchor candidate directly |
| Fragment composer | Skip wiring transformer + Judge re-vote |
| Wiring transformer | Skip best-of-N additional samples |
| Judge vote | Return final candidate |

**Pre-registered targets:**

| Metric | Target | Floor |
|---|---|---|
| Phase 2c clean held-out latency p50 | ≤ 1 ms (vs current ~5 ms) | 2 ms |
| Phase 2c clean retrieval (correct on all inputs) | ≥ 19/20 (vs current 20/20) | 18/20 |
| Phase 3b composition retrieval | ≥ 6/10 (no regression vs current 7/10) | 6/10 |
| TF-IDF v2 retrieval | ≥ 16/20 (no regression vs current) | 15/20 |

**Pre-registered skip rule:** if retrieval drops below the floor on any metric, ACT-halting is shipped behind a build flag `-DMICROGPT_OPA_ACT_HALT=ON` defaulting OFF. If the floor holds and the latency target is met, the flag becomes default-ON for the wiring binary and gets a `BS_wiring.md` `INV-WIRE-070` invariant.

**What this gates in the larger product story:** Per `RESEARCH_DISCLOSURE.md` §3 and the calibrated three-bound claim, the hard-prompt minority (~20 % of novel paraphrases) is structurally bounded by the bag-of-features ceiling. Adaptive compute lets us *spend* compute only on the hard cases. For the productisation hand-off (in the private companion repo), this is the latency mechanism that makes the architecture viable for sub-5 ms fraud-decision latency in the easy-majority case while preserving the careful Judge stack for the hard-minority case.

**Cost estimate:** ~1 week implementation + 1 week measurement. Single ~30 LoC linear head; modifies `demos/wiring_organelle/main.c` vote loop; no core engine changes.

### 2.2 Experiment B — LTI-stable scoring for the wiring vote loop

**Hypothesis (locked):** Parameterising the wiring binary's candidate-score updates as a contraction map (ρ < 1, OpenMythos-style `A_continuous = -exp(log_A)` then ZOH-discretised) eliminates the Phase 8 vote-loop regression documented in `wiring_binary_phase8_regression.md`, *without* requiring the surgical rollback of new-family anchor entries that V1.0.7 path-4 used.

**Mechanism:** Each candidate's score across re-rank passes follows `s_{t+1} = A · s_t + B · evidence + Δ_judge` where `A` is a per-family scalar parameterised so its continuous-time eigenvalue is always negative; the ZOH discretisation guarantees `A ∈ (0, 1)`. Effect: a candidate's score cannot grow unbounded across re-rank passes regardless of how many evidence injections favour it; the diminishing-returns dynamic is built into the parameterisation rather than tuned via thresholds.

**Pre-registered targets:**

| Metric | Target |
|---|---|
| Phase 8 (correct on all 5 inputs) on `--clean-only` with ALL Phase 1-4 anchor entries restored | ≥ 18/20 (vs the 3/20 regression and the 20/20 surgical-rollback baseline) |
| Phase 2c HEADLINE strict-verified | ≥ 20/20 (must hold) |
| Phase 3b composition correct on all inputs | ≥ 6/10 (must hold) |

**Pre-registered skip rule:** if Phase 8 reaches ≥ 18/20 with the new-family anchors restored, `GAP-WIRE-003` and (the originating issue behind) `GAP-WIRE-007`-class regressions both close. If Phase 8 stalls below 15/20 even with the LTI parameterisation, the contraction-map approach is falsified for our scoring topology and a different fix path (either path 1 candidate-scoring change or path 4 permanent rollback) becomes the chosen direction.

**Cost estimate:** ~2 weeks. Single per-family scalar parameter + ZOH discretisation in the vote loop's score-accumulator; modifies `demos/wiring_organelle/main.c`; touches no core transformer code.

### 2.3 Experiment D — Latent-tensor inter-organelle handoffs (the third level)

**Surfaced by an external review (May 2026)** that pointed out the original cross-pollination assessment under-weighted the *inter-organelle wire format*. The OpenMythos technique that maps here isn't internal to a single transformer — it's the principle of "continuous latent state crossing iteration boundaries." Lifted from "iterations within a model" to "stages within the OPA Kanban", this becomes a candidate for the third highest-leverage move.

**Hypothesis (locked):** Replacing the planner→player→judge text-string handoff (the current `OpaKanban` `KEY=val|KEY2=val2` pipe-string format, `FS_organelle_wire.md`) with a continuous latent-tensor handoff lifts the multi-organelle game-pipeline solve rate on tasks where the discretisation step is documented to lose information.

**Mechanism:**
- Each organelle exposes a *latent emit* alongside its existing text emit. The latent is the post-final-RMSNorm hidden state at the chosen output position (dim=96-128 depending on the organelle).
- Downstream organelle accepts EITHER the legacy text input (parsed via existing pipe-string protocol) OR the latent (projected through a small linear bridge to its embedding dimension) at runtime via the Kanban dispatcher.
- Both modes coexist for measurement: same prompt, both wire formats, A/B comparison.

**Pre-registered targets:**

| Eval | Text-string baseline | Latent target | Floor |
|---|---|---|---|
| 8-puzzle solve rate (current 90 %) | 90 % | ≥ 92 % | ≥ 88 % |
| Connect-4 win rate (current 88 %) | 88 % | ≥ 90 % | ≥ 86 % |
| Mastermind solve rate (current 79 %) | 79 % | ≥ 82 % | ≥ 77 % |
| Pipeline IR DOT-renderer audit coverage on the resulting traces | 100 % structural + 100 % value-trace | 100 % structural + ≥ 70 % value-trace | 100 % structural required |

**Pre-registered skip rule (and the honest tradeoff this experiment must surface):**
1. If latent handoff lifts game solve rates by ≥ 2 pp on at least 2 of 3 games AND the audit floor (100 % structural) holds, the latent handoff is shipped behind a build flag `-DMICROGPT_ORG_LATENT_WIRE=ON` defaulting OFF. **It does not become default-ON until** the productisation verticals (private companion repo) explicitly accept the audit-trail tradeoff via an ADR.
2. If the audit floor is breached (the DOT-renderer cannot reconstruct the structural graph from the latent-only traces), the experiment is **falsified** and the text-string handoff stays as the canonical wire format. Latent emits remain available as an *augmentation* — added to text emits, not replacing them — for any future per-vertical experiment that wants the gradient flow.
3. If solve rates are flat or worse, the experiment is **falsified** and the discretisation-wall framing is itself called into question (the bottleneck would be elsewhere).

**Honest tradeoff disclosure (load-bearing):**
The pipe-string text handoff is **audit infrastructure**, not just a limitation. Pipeline IR's typed DAG + DOT renderer makes pipeline traces regulator-reviewable end-to-end. Latent handoffs would let the *signal* flow continuously and make the cross-organelle pipeline end-to-end differentiable — but at the cost of losing the value-by-value audit trace (the structural graph remains, the per-step values become opaque tensors). For productisation verticals where audit is mandatory (fraud per PCI-DSS, finance per SR 11-7, defence per RAI principles), this is potentially **disqualifying** by default. The flag-default-OFF + ADR-required adoption path is the honest way to ship the technique without breaking the productisation pitch.

This experiment must NOT silently swap audit for gradient flow. The trade has to be made explicit in an ADR per the dependency-policy precedent (for verticals that opt in), with a documented per-vertical decision recorded as either "latent ON" or "text ON" — not a global default change.

**What this experiment is NOT testing:**
- It does NOT test whether latent handoffs lift the calibrated 75-80 % retrieval ceiling. Per `INV-WIRE-060` that ceiling is bag-of-features-bound for the Wiring Organelle's retrieval mechanism, not handoff-bound. Latent handoffs help with gradient flow across organelle boundaries, not with the underlying retrieval-mechanism limit.
- It does NOT bridge categorical reasoning with numerical sensing for OPA's continuous-domain attempts (e.g. financial time-series). That would require a separate experiment with an explicit numerical-input organelle, not a wire-format change.

**Cost estimate:** ~4 weeks. New `microgpt_organelle_latent.{h,c}` ~500 LoC implementing the latent emit/accept paths and the per-organelle linear bridge. Modifies `OpaKanban` dispatcher to choose wire format per organelle pair. Updates `FS_organelle_wire.md` to document both formats normatively. One game demo (8-puzzle, since it's the most-tested) re-trained end-to-end to validate gradient flow. Most cost is in the audit-trace work (proving the DOT renderer can still produce a reviewable artefact under latent handoffs) — without which the experiment cannot ship even on a flag.

### 2.4 Experiment C — Depth extrapolation for game-playing organelles

**Hypothesis (locked):** A planner organelle trained at recurrent-depth `T` can be evaluated at depth `T + k` and achieves higher solve rates on hard game positions than the same organelle at depth `T`. Concretely: an 8-puzzle planner trained at depth-T = 8, evaluated at depth-T = 16, lifts the current 90 % solve rate to ≥ 92 % on the difficulty-stratified hard-position subset.

**Mechanism:** Wrap the existing 8-puzzle planner organelle in a minimal Recurrent-Depth wrapper following the OpenMythos `RecurrentBlock` pattern: freeze the encoded input `e` after a single forward pass, then loop the same transformer block T times with `LTIInjection` + `LoRAAdapter` (rank-4 to fit the 460K-param budget). No ACT halting (depth is fixed for this experiment to isolate the depth-extrapolation signal).

**Pre-registered targets:**

| Eval | Target | Floor |
|---|---|---|
| 8-puzzle solve rate at depth-T (training depth) | ≥ 88 % (no regression vs current 90 %) | 85 % |
| 8-puzzle solve rate at depth-(T+8) on **hard-position subset** | ≥ 92 % | ≥ training-depth rate |
| Mastermind solve rate at depth-(T+8) | ≥ 81 % (vs current 79 %) | ≥ training-depth rate |

**Pre-registered skip rule:** if depth extrapolation lifts solve rate by ≥ 2 pp on either game's hard-position subset, the technique is validated and a follow-on `RESEARCH_OPENMYTHOS_RDT_PLANNER.md` documents the per-game results plus the limit at which depth extrapolation breaks (the "depth where the LTI A approaches 1.0 and the loop becomes a no-op"). If solve rate is flat or worse at depth-(T+k), the depth-extrapolation hypothesis is **falsified at our 460K-param scale** and the technique is documented as research-only (likely needs ≥ 2k dim to express meaningful per-iteration deltas).

**Cost estimate:** ~3 weeks. New `microgpt_recurrent_depth.{h,c}` ~400 LoC implementing `RecurrentBlock` + `LTIInjection` + `LoRAAdapter`. Re-trains 8-puzzle and Mastermind planner organelles. Most of the cost is training-time + difficulty-stratified eval-set construction, not implementation.

---

## 3. The honest scaling consideration

OpenMythos's published config uses `dim=2048, n_heads=16, max_loop_iters=16`. The whole *premise* of the looped-shared-weight architecture is that one block executed L times can substitute for an L-block stack — but only if each block is **wide enough** to express meaningful per-iteration deltas. The Bae et al. (2024) Relaxed Recursive Transformer paper that grounds the LoRA-per-loop adapter shows that *width below ~512 dims* makes the LoRA delta either a no-op (rank too small) or an entire model rewrite (rank approaches dim) — there is no useful intermediate.

Our organelles run at dim=96 to dim=128. Per-iteration LoRA deltas at rank-4 to rank-16 against a dim-96 base are in the "approaches the dim itself" regime — closer to "rebuild the layer" than "small adapter." Two implications:

1. Experiment C (depth extrapolation for planners) is the most *fragile* of the three. Falsification is a real and likely outcome at our scale. Pre-registering the target deliberately above the training-depth baseline + the floor at "no worse than training depth" guards against silent re-tuning.
2. Experiments A and B do not depend on model width at all — they govern the *coordination layer*, where we have plenty of room. Those are the safe high-leverage bets.

---

## 4. Cumulative scope

| Experiment | Cost | Falsification risk | Productisation impact |
|---|---|---|---|
| A — ACT halting at pipeline level | ~2 weeks | Low (well-understood mechanism) | **Direct** — sub-1 ms easy-prompt latency for fraud Phase 1 |
| B — LTI-stable vote-loop scoring | ~2 weeks | Medium (novel application of contraction mapping to discrete scoring) | **Indirect** — closes `GAP-WIRE-003` properly, lets new family anchors land without rollback |
| D — Latent inter-organelle handoffs | ~4 weeks | Medium (gradient-flow gain plausible; audit-trail tradeoff is the binding constraint) | **Conditional** — high impact for verticals that ADR-accept the audit tradeoff; OFF by default |
| C — RDT for game planners | ~3 weeks | High (likely falsified at our 460K-param scale) | None directly; research-side hygiene |

**Recommended order if any of these proceed:** A first (highest leverage, lowest risk, directly enables productisation latency targets), then B (closes the open vote-loop bug at the right architectural level), then D (only if the audit-trail tradeoff is acceptable to a target vertical — ADR-gated adoption), then C only if the project has appetite for a probably-falsifiable research experiment. None should run before the existing open `GAP-WIRE-006` Phase 6c+ work has its own outcome on record.

The broader OPA-side research direction catalogue (independent of OpenMythos cross-pollination) is in [`RESEARCH_OPA_DIRECTIONS.md`](RESEARCH_OPA_DIRECTIONS.md).

---

## 5. Rejected techniques — explicit reasons

For honest-disclosure discipline, the rejections are documented as carefully as the acceptances.

| Technique | Why rejected | What would have to change for adoption |
|---|---|---|
| **MLA (Multi-Latent Attention)** | We already have TurboQuant + RotorQuant + MSA covering KV memory at the project's edge-deployment scale. MLA's ~10-20× cache reduction is measured at production scale (dim ≥ 2048); at dim=96-128 the absolute cache size is already tiny and the latent-reconstruction overhead exceeds the storage saving. | A vertical product needs ≥ 1k-dim transformer in the inference path. Unlikely without changing the tiny-specialists thesis. |
| **GQA (Grouped Query Attention)** | Same reasoning as MLA — KV-cache concerns are downstream of model width, and our width is too small for GQA's group-share trick to benefit measurably. | Same as MLA. |
| **Looped Recurrent-Block in the wiring transformer** | The wiring transformer alone is 35 % on the clean baseline. Most of OPA's value is in the Judge stack, not the transformer (per `INV-WIRE-041`). Looping the transformer might lift it to ~50 %; the calibrated 75-80 % retrieval-mechanism ceiling is bag-of-features-bound (per `INV-WIRE-060`), not transformer-bound. So even a maximally-successful RDT wiring transformer would not move the headline retrieval claim. | The wiring transformer becomes the front-line mechanism (currently the anchor library is). Architecturally incompatible with the calibrated three-bound thesis. |
| **MoE FFN inside one organelle** | Directly contradicts the tiny-specialists thesis. The architecture's distinctive value is multi-organelle coordination (model-level MoE via Kanban), not intra-model MoE. Adopting intra-model MoE would let us shrink the organelle count — at the cost of the explainability + auditability + edge-deployability story that the productisation pitch depends on. | A vertical product chooses to ship one big model + intra-model MoE instead of multi-organelle coordination. Strictly out of scope. |
| **LoRA per-loop adapter (in isolation)** | At rank-4 to rank-16 against dim=96, the LoRA delta approaches the model dimension itself — closer to "rebuild the layer" than "small adapter." Falsified-by-design at our model-width scale. | Adopted as part of Experiment C (RDT for planners), where the rank-vs-dim ratio is acknowledged as the failure-mode under test. |
| **Loop-index sinusoidal embedding** | Conceptually elegant but only valuable when paired with a looped block. Without looping, it's a positional-embedding alternative that buys us nothing over the existing `wpe` table or the optional partial-RoPE port. | Adopted automatically as part of Experiment C if RDT graduates. |

---

## 6. Standing protections inherited from the project's research discipline

Per `docs/engineering/CLEAN_ROOM_IMPLEMENTATION/RESEARCH_DISCLOSURE.md` §4 and `INV-WIRE-062`:

- Any experiment in §2 that produces a held-out result MUST run `tools/scaling_leakage_audit.sh` on its held-out set BEFORE reporting the score. The v1 leakage incident (`RESEARCH_DISCLOSURE.md` §3.1) is the precedent; the audit is non-negotiable.
- Any experiment cancelled per its skip rule MUST be documented in `RESEARCH_DISCLOSURE.md` §2-style cancellation notes, not silently dropped.
- The "what is NOT being claimed" discipline applies: a successful Experiment A does **not** lift the calibrated 75-80 % retrieval ceiling — it improves the *latency* of reaching that ceiling. The headline calibrated claim is unchanged regardless of these experiments' outcomes.

---

## 7. Cross-references

| Topic | Source |
|---|---|
| The calibrated three-bound retrieval claim | `docs/research/ORGANELLE_STATE.md`, `wiring_scaling_post_phase3.md` |
| The wiring binary vote-loop regression that Experiment B targets | `docs/research/wiring_binary_phase8_regression.md`, `GAP-WIRE-003` in `TRACEABILITY.md` |
| The pre-registration discipline this note follows | `docs/research/RESEARCH_PIPELINE_IR.md` §40, §42, §45 |
| The honest-disclosure register that any outcome lands in | `docs/engineering/CLEAN_ROOM_IMPLEMENTATION/RESEARCH_DISCLOSURE.md` |
| The companion private repo that benefits most from Experiment A | `docs/MIGRATED_TO_ORGANELLES_BIO.md` |
| OpenMythos class reference (the source material) | `OpenMythos` / `MythosConfig` / `RecurrentBlock` / `LTIInjection` / `ACTHalting` / `LoRAAdapter` |

---

## 8. Status

**Pre-registration only.** No code. No flag. No measurement.

The next step on any of the three experiments is a `feat(research): pre-registered Experiment A — ACT halting at OPA pipeline level` (or B, or C) commit that lands the implementation **with no measurement output**, followed by a separate `research(openmythos): Experiment A measurement vs §2.1 pre-reg targets` commit that lands the measurement output. The two-commit shape preserves the discipline that pre-registration has to be locked in before the result is known.

— Pre-registered 2026-05-01.
