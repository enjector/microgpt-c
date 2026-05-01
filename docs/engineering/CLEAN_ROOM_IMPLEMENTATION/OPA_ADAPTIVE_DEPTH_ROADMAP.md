# OPA Adaptive-Depth Roadmap

**Document ID:** MGC-PLAN-OPA-AD-001
**Version:** 1.0
**Status:** PRE-REGISTERED (not scheduled — opens on customer signal)
**Date:** 2026-05-01
**Tracks:** new gaps `GAP-OPA-001..004` (each pre-registered below).
**Sibling plan:** `COMPOSITIONAL_GENERATOR_PHASE_6D_PLAN.md`
**Author:** Claude (clean-room corpus author)

---

## Context

The OPA (Organelle Pipeline Architecture) ships in V1.0 with deterministic C scaffolding (`OpaKanban`, `OpaCycleDetector`, valid-move filter, `OpaTrace`) coordinating a small number of trained transformer organelles. The current behaviour is **fixed-depth, fixed-pipeline**: every puzzle gets the same number of replans, every Connect-4 turn invokes the planner once, every game terminates by a deterministic stop condition.

A survey of OpenMythos (Recurrent-Depth Transformer with ACT halting + LTI injection + loop-index embedding + depth-extrapolation training) surfaced four mechanisms that **transfer to OPA at the pipeline-coordination layer** without changing the compile-time architecture or the rebuild-test corpus's normative contract. This document records the four mechanisms as a pre-registered Phase 7 roadmap so they can be picked up cleanly when (if) a customer signal warrants the engineering investment.

The roadmap is **deliberately conservative**: each mechanism is gated by a falsification criterion, has a predicted axis-by-axis lift on existing demos, and preserves the architectural invariants the V1.0 corpus depends on (`pipeline_verify` as Judge, deterministic C scaffolding, `OpaKanban` semantics).

## Cross-cutting non-goals

The roadmap does NOT propose:

- Replacing OPA's coordination layer with a learned recurrent-depth block (would destroy the verifier-as-Judge auditability story).
- Adding learned routers / MoE FFN inside the 540K-param organelles (architecture-level change incompatible with the compile-time-define contract in `BS_core.md`).
- MLA / GQA / KV-cache compression at the organelle level (V1.0 already ships MSA / TurboQuant / RotorQuant; no measurable footprint problem at 540K).
- Replacing the deterministic Judge with an opaque continuous halting (would invalidate the corpus's `pipeline_verify` invariants).

What it DOES propose: four small, additive mechanisms at the OPA scaffolding layer, each implementable in pure C99 + libc + libm with no new dependency.

## Pre-registered mechanisms (priority order)

### Mechanism 1 — `OpaActHalting` per-state halting probability (`GAP-OPA-001`)

**Inspiration:** OpenMythos `ACTHalting` (Graves 2016 ACT) — accumulates a per-position halting probability across iterations; exits when cumulative probability exceeds a threshold.

**OPA translation:** at the pipeline level, accumulate a per-puzzle `p_halt` across replans. The signal source is the planner organelle's confidence (entropy of next-token softmax, or a small learned head if a customer-funded training run is acceptable). When `cumulative_p > opa_act_threshold` (default 0.99), exit the puzzle as "solved" / "abandoned"; when below `opa_act_floor` (default 0.05) after K replans, force a hard fallback.

**Hypothesis (H10):** ACT-driven adaptive replan depth lifts 8-puzzle hard-tier solve rate from 30 % (the current published 90 % overall = 100 % easy + 100 % medium + 70 % hard) to ≥ 80 % hard-tier, **without disturbing easy/medium**. Predicted aggregate lift: 90 % → 93 %.

**Falsification criterion:** if hard-tier lift < +3 % vs the V1.0 fixed-depth baseline, the mechanism is falsified. Phase 7-Mechanism-1 reverts.

**Files:**
- New: `src/microgpt_organelle.c` extension — add `OpaActHalting` struct + `opa_act_init` / `opa_act_observe(p_local)` / `opa_act_should_halt(threshold)` API.
- New: `src/microgpt_organelle.h` extension — public declarations.
- New: `tests/test_microgpt_organelle.c` — three tests (init, accumulation, threshold-cross).
- Modified: at least one game demo (recommended: `c_puzzle8_demo`) integrates `OpaActHalting` and reports `act_halted`/`act_replans` counts in the trace.

**Cost:** ~1 week. ~120 LOC C99. No new training corpus required if entropy-of-softmax is used as the signal; ~500 examples + 500 training steps if a learned head is preferred.

### Mechanism 2 — Frozen-input injection (`GAP-OPA-002`)

**Inspiration:** OpenMythos LTI invariant — the encoded input `e` is captured after the Prelude and **frozen**, then injected at every recurrent loop iteration. This guarantees the hidden state cannot drift from the original input regardless of loop depth.

**OPA translation:** in multi-organelle pipelines, capture the canonical state-string at puzzle start (or at every "fresh state" transition), hash it, and prepend a stable token to every subsequent organelle prompt. Different from the current OPA flow which re-emits the planner state through the planner organelle every replan.

**Hypothesis (H11):** Frozen-input injection reduces `OpaCycleDetector` trip count by ≥ 30 % on Connect-4 deep solves. Reason: cycles arise when the planner drifts away from the original puzzle state under deep replan loops. Freezing the prompt prefix prevents that drift.

**Falsification criterion:** cycle-detector trips unchanged within ±10 % across 100 evaluation games per axis (8-puzzle, Connect-4, Mastermind).

**Files:**
- New: `src/microgpt_organelle.c` — `opa_freeze_input(state_str, &handle)` / `opa_prefix_with_frozen(handle, prompt_buf)` helpers.
- Modified: `demos/character-level/connect4/main.c` — opt into the frozen prefix.
- Modified: `tests/test_microgpt_organelle.c` — round-trip + idempotency tests.

**Cost:** 3 days. ~80 LOC. Fully pure-C99.

### Mechanism 3 — Loop-index step-id token (`GAP-OPA-003`)

**Inspiration:** OpenMythos `loop_index_embedding` — sinusoidal signal injected into the first `loop_dim` channels of the hidden state so the same shared weights behave differently at different loop iterations.

**OPA translation:** prepend a small `STEP|t=N|` token to the planner organelle's prompt at iteration N. The same planner can produce different outputs at iteration 0 vs iteration 5 of the same puzzle, without any new parameters.

**Hypothesis (H12):** the planner produces measurably different next-action distributions at step 0 vs step ≥ 5 of the same puzzle (KL divergence ≥ 0.1 on at least 30 % of replan transitions in 8-puzzle hard tier).

**Falsification criterion:** KL divergence < 0.05 on > 90 % of replan transitions (the planner is ignoring the step token).

**Files:**
- Modified: `demos/character-level/puzzle8_reasoning/main.c` — add the step-id prefix.
- New tiny re-training pass on the planner corpus (or, if customer-signal warrants, a from-scratch corpus that includes step-tokens).

**Cost:** 2 days for the prefix change; +1 week if re-training is required. Falsifiable cleanly via the KL-divergence measurement.

### Mechanism 4 — Depth-extrapolation measurement (`GAP-OPA-004`)

**Inspiration:** OpenMythos's depth-extrapolation property: train at `n_loops=N`, infer at `n_loops=N+k`, observe lift on harder problems.

**OPA translation:** train an organelle pipeline on Connect-4 (7×6 board, depth=N replans) and evaluate on a larger board (8×8, depth=N+k). The looped invariant in our case is the deterministic C scaffolding rather than learned weights, so the question is: **does the existing 460K-param Connect-4 organelle generalise to a board it wasn't trained on, given enough replans?**

**Hypothesis (H13):** open question. This is an evidence-gathering experiment, not a hypothesis with a SLO. Predicted: 8×8 win rate ≥ 40 % (vs random) at depth=2N replans, with a baseline of ≥ 70 % at depth=N. If true: depth-extrapolation transfers to OPA. If false: OPA is fundamentally fixed-depth and additional pipeline cycles don't substitute for board-specific training.

**Falsification criterion:** **none** — this is a measurement, not a hypothesis. Report whatever the 8×8 win rate is.

**Files:**
- Modified: `demos/character-level/connect4/main.c` — accept board-size flag.
- New: an 8×8 reference opponent + scoring rig.

**Cost:** ~1 week. The result feeds into Phase-8 strategy (whether OPA can claim depth-extrapolation as a property).

## Combined-mechanism prediction

Mechanisms 1 + 2 + 3 are independent and additive:

| Mechanism | Predicted lift | Source workload |
|---|---|---|
| H10 ACT halting | +3 pp on 8-puzzle (90 → 93 %) | hard tier, fixed-depth puzzles |
| H11 frozen input | −30 % cycle-detector trips | Connect-4 deep solves |
| H12 step-id token | KL ≥ 0.1 on ≥ 30 % of replans | 8-puzzle hard tier |

Mechanism 4 is an open measurement; no aggregate prediction.

## Disposition logic (per-mechanism, not aggregate)

Same §6.3 framework as Phase 6c. **Each mechanism has its own disposition:**

- **Achieved ≥ pre-registered target:** new `GAP-OPA-NNN` → RESOLVED. The relevant SLO (8-puzzle solve rate, cycle-detector trip count, etc.) is promoted in `NFRD.md`.
- **Within ±50 % of target:** PARTIALLY-RESOLVED with achieved score as new baseline.
- **Below falsification floor:** mechanism reverted; gap stays OPEN with the V1.0 baseline.

Cross-mechanism aggregate: if 0 of 4 confirm, the OpenMythos-→OPA transfer hypothesis is **falsified** at the OPA scaffolding layer; the recurrent-depth idea does not transfer to deterministic-C-coordinated organelle pipelines. Recorded honestly in `RESEARCH_DISCLOSURE.md` §7 (to be added when results land).

If 4 of 4 confirm, the OPA layer becomes **adaptive-depth**, and `BS_organelle.md` §1 is rewritten to document the four-mechanism extension.

## No-regression invariants

- Each mechanism is gated by a compile-time flag (default OFF) so V1.0 behaviour is bit-identical when the flag is off.
- The deterministic Judge property (`pipeline_verify`) is untouched.
- `OpaKanban`, `OpaCycleDetector`, `OpaTrace` semantics in `BS_organelle.md` §2 are unchanged.
- All current game demos (8-puzzle, Connect-4, Mastermind, Sudoku, etc.) MUST hit their published win rates with the new mechanisms enabled. Regression by > 1 prompt fails the no-regression gate.
- ctest 15/15 must persist.

## Standing leakage / methodology discipline

- The OPA mechanisms operate on game-state strings, not on the wiring held-out. The `tools/scaling_leakage_audit.sh` is irrelevant here.
- The honest-disclosure-first rule applies as before: pre-register predicted lifts, do not silently re-tune the targets, record actual results in `RESEARCH_DISCLOSURE.md` §7 when Phase 7 is scheduled.

## What this roadmap deliberately does NOT do

- **Does not change `BS_organelle.md` invariants.** The V1.0 specification of `OpaKanban` / `OpaCycleDetector` is the rebuild-test contract; this plan adds NEW primitives (`OpaActHalting`, `opa_freeze_input`) without rewriting existing ones.
- **Does not propose a Phase 8.** Avoid stacking unfounded escalations; if any of the four mechanisms confirm, the next plan is written from the actual data.
- **Does not reach for OpenMythos's MoE / MLA / GQA / RDT-as-architecture.** Those are full-architecture rewrites, not transferable mechanisms.
- **Does not require new external dependencies.** Aligns with `GAP-DEP-001` — the OPA scaffolding layer remains pure C99 + libc.

## Trigger criteria

Phase 7 is opened when ANY of:

1. A customer asks for higher 8-puzzle hard-tier solve rate (or analogous regulated-domain hard-tier accuracy).
2. A customer asks for stability on deeper game-tree search (Connect-4 12+ ply, Mastermind > 10 guesses).
3. A research opportunity surfaces a clean dataset that exercises depth-extrapolation (Mechanism 4) and the engineering team has spare capacity.

In the absence of all three, this roadmap stays pre-registered but not scheduled. The V1.0 baseline (90 % 8-puzzle, 88 % Connect-4, etc.) is the published number.

## Per-mechanism gap entries (for `TRACEABILITY.md`)

| Gap ID | Severity | Mechanism | Falsification target |
|---|---|---|---|
| `GAP-OPA-001` | P2 | OpaActHalting (Mechanism 1) | hard-tier lift < +3 % falsifies |
| `GAP-OPA-002` | P2 | Frozen-input injection (Mechanism 2) | cycle-detector trips unchanged within ±10 % falsifies |
| `GAP-OPA-003` | P3 | Loop-index step-id token (Mechanism 3) | KL < 0.05 on > 90 % of replans falsifies |
| `GAP-OPA-004` | P3 | Depth-extrapolation measurement (Mechanism 4) | no falsification — measurement only |

## Cross-references

- OpenMythos class reference (the inspiration; external).
- `RESEARCH_DISCLOSURE.md` §3, §4, §5, §6 — the methodology this plan follows.
- `COMPOSITIONAL_GENERATOR_PHASE_6D_PLAN.md` — sibling pre-registered roadmap.
- `BS_organelle.md` §2 — the OPA invariants this roadmap leaves intact.
- `TRACEABILITY.md` — `GAP-OPA-001..004` to be added on Phase-7 scheduling.
- `book.7th/Reversible_Engineering.md` Chapter 6.5 — pre-registration discipline.

## Revision history

| Version | Date | Change |
|---|---|---|
| 1.0 | 2026-05-01 | Initial pre-registration. Authored after surveying OpenMythos's Recurrent-Depth Transformer mechanisms and identifying four (ACT halting, LTI frozen-input, loop-index embedding, depth-extrapolation) as transferable to the OPA scaffolding layer without disturbing the V1.0 rebuild-test corpus. NOT scheduled — opens on customer signal. |
