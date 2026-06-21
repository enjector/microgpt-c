# VibeThinker-3B → ideas for microgpt-c

> Which VibeThinker post-training ideas *survive contact* with this project's prior findings — and how they map onto the organelle stack.

**Status:** Idea 1 (CLR) **measured** — see [`experiments/E18-clr-reliability-reranker.md`](../../../../experiments/E18-clr-reliability-reranker.md): falsified as a lift mechanism, exactly as the §3.4 ceiling pre-stated. The rest remain evaluation notes.

**Reference:** [VibeThinker-3B.pdf](VibeThinker-3B.pdf) · paper summary in [README.md](README.md).

---

## 0. Grounding & caveats (read first)

Per the repository's research-discipline rule (`CLAUDE.md` §"Research discipline"), here is the reading log behind this document:

- **VibeThinker-3B paper** — read **end-to-end by the author** (all 14 pages: abstract, §1 intro + Compression-Coverage hypothesis, §2.1–2.4 methods, §3 evaluation + Tables 1–3, §4 conclusion, references).
- **Pipeline-IR verifier API** (`libs/pipeline_ir/include/pipeline_ir/pipeline_ir.h`) — **author-verified directly**: `pipeline_verify()` runs 8 ordered local checks (unique ids, edge endpoints exist, no dangling inputs, sig-inputs connected, sig-outputs connected once, edge type-match, acyclic, topo-sort), each returning a specific error code and a `pipeline_last_error()` message naming the offending node/edge/port; `pipeline_verify_partial()` returns a `missing` count for incremental construction (lines 28, 294–354).
- **Project research state** (experiment arc E1–E17, the calibrated thesis, prior rejections of RL/curriculum/distillation, existing `organelle_generate_ensemble()` / Model Soup / wiring re-ranker) — **summarised via Explore sub-agents**, *not* read end-to-end by the author. Treat the project-state claims below as **exploration-sourced**; any of them that would gate a real experiment must be re-read in full before a pre-registration is locked.

**Scope rule for this document (per the user's request): survivors only.** Three VibeThinker pillars are deliberately **out of scope** because the project has already tested and closed them:

| Out of scope | Why (exploration-sourced) |
|---|---|
| **MGPO / RL of organelles** | Rejected — `RESEARCH_OPA_DIRECTIONS.md` §10: the bottleneck is curator-bounded library size, not policy; RL earns its keep only once the action space genuinely needs policy optimisation. |
| **Curriculum SFT** (easy→hard ordering) | Rejected — `RESEARCH_OPTIMISATIONS.md`: counterproductive for *pattern-memorisation* organelles, which must learn exact input→output mappings. |
| **Teacher→student LLM distillation** | Falsified at this scale — **E13**: a 35B teacher lifted a 460K student only into the neutral band; saturates at the teacher's own ceiling. |

This document does **not** re-litigate any of those. It also proposes **no rewrite of `ORGANELLE_STATE.md`** and drafts **no locked E1N pre-registration** — every concrete idea below is flagged as needing its own full-arc read + pre-reg before it becomes an experiment.

---

## 1. Spear summary

**Point.** Exactly one VibeThinker idea has real leverage here: **Claim-Level Reliability Assessment (CLR)** — and only because it is a *test-time* refinement layered on infrastructure the project has **already built and validated** (the pipeline-IR verifier + the wiring organelle's best-of-16 vote). Three smaller survivors are cheap re-rank / selection tweaks. One is a framing import, not code.

**Picture.** The wiring organelle already samples 16 candidate graphs and majority-votes. CLR says: don't count votes — *interrogate each candidate's own claims*. A pipeline graph makes 8 checkable claims about itself (its types bind, it's acyclic, its ports connect…). The verifier already answers each one locally. CLR turns those local answers into a nonlinear trust score that **collapses any candidate harbouring a single broken claim**, then aggregates by trust instead of by headcount.

**Proof.**

| Survivor idea | Builds on (already in project) | New surface | Verdict |
|---|---|---|---|
| **CLR reliability-weighted aggregation** (§3) | `pipeline_verify_partial()` + best-of-16 vote | re-ranker only, 0 engine change | ✅ headline |
| **Compression-Coverage lens** (§4) | the project's own calibrated thesis | none (framing) | ✅ corroboration |
| **Brevity tie-breaker** (§5) | the wiring re-ranker | ~5 LOC | ✅ cheap |
| **Pass@K checkpoint selection** (§6) | `organelle_train_soup()` + role specialists | selection criterion only | ✅ cheap |

**Push.** §3 is the only one worth an experiment number. Treat it as a *candidate* E18, with its ceiling pre-stated (§3.4) — re-read the wiring arc (`RESEARCH_PIPELINE_IR.md`) and `RESEARCH_MANIFOLD_LEARNING.md` in full before locking it.

---

## 2. Survivor map

Every VibeThinker mechanism, against what already exists here:

| VibeThinker mechanism | Already in project? | Survivor verdict |
|---|---|---|
| **CLR** test-time scaling | Partial — best-of-16 + verify + plain majority vote (wiring Phase 8) | ✅ **Survives** — CLR adds nonlinear penalty + reliability-weighted aggregation. §3 |
| **Parametric Compression-Coverage Hypothesis** | Conceptually — the calibrated thesis says the ~80% ceiling is coverage-bound | ✅ **Survives** as a shared vocabulary / lens. §4 |
| **Long2Short** brevity preference | No (RL machinery rejected) | ✅ **Survives** *only* as a non-RL deterministic tie-breaker. §5 |
| **Diversity-Exploring Distillation** — Pass@K checkpoint selection | Partial — Model Soup averages seeds; per-role specialists exist | ✅ **Survives** as a selection-criterion swap (not the teacher-student half). §6 |
| MGPO / boundary-weighted RL | No | ⛔ Out of scope (RL rejected, §0) |
| Two-stage curriculum SFT | No | ⛔ Out of scope (curriculum rejected, §0) |
| Teacher→student distillation | Tested | ⛔ Out of scope (E13 falsified, §0) |
| Multi-domain RL, Instruct RL, offline self-distillation (RL-coupled) | No | ⛔ Out of scope (all RL-coupled) |

---

## 3. Idea 1 (headline): CLR over the pipeline-IR verifier

### 3.1 What the paper does

CLR (§3.1, eqs. 5–6) is a weights-frozen test-time scaling method for answer-verifiable tasks:

1. Sample `K = 32` candidate trajectories.
2. Extract `M = 5` **decision-relevant claims** per trajectory.
3. Self-verify each claim to a binary verdict `v_{k,m} ∈ {0,1}`.
4. Score each trajectory **nonlinearly**: `r_k = ((1/M)·Σ_m v_{k,m})^M`.
5. Cluster the final answers by equivalence; select the cluster maximising `Σ r_k`.

The `(·)^M` exponent is the whole point: a trajectory with even one flawed claim is *heavily* penalised, not merely out-voted. CLR "isolates critical logical anchors" rather than re-processing entire verbose traces.

### 3.2 What the project does today

The wiring organelle samples **best-of-16** graphs, runs each through the parse→tolerant→repair cascade, and does **plain majority vote** for self-consistency (CLAUDE.md §"Pipeline IR + Wiring Organelle", Phase 8). Every candidate is one equal vote regardless of *how cleanly* it verifies.

### 3.3 The mapping (why it fits almost exactly)

A generated pipeline graph already **is** a bundle of checkable claims. The verifier (`pipeline_verify()`, author-verified) runs 8 local checks, each naming its offending element. CLR's `v_{k,m}` are these per-check verdicts; `pipeline_verify_partial()` already returns a `missing` count of unsatisfied-but-recoverable elements. So the substrate CLR needs is *present* — only the **scoring + aggregation** are new:

```c
/* Sketch — lives in the wiring re-ranker / a natives file, not in the engine.
   Maps a candidate graph to a CLR-style reliability score in [0,1]. */
static double clr_reliability(Pipeline *cand) {
    int missing = 0;
    int hard_err = pipeline_verify_partial(cand, &missing);   /* PIPE_OK or negative */
    if (hard_err != PIPE_OK) return 0.0;                      /* a broken claim -> trust 0 */

    int total   = pipeline_node_count(cand) + pipeline_edge_count(cand);
    int satisfied = total - missing;
    if (total <= 0) return 0.0;

    double frac = (double)satisfied / (double)total;          /* mean verdict */
    /* nonlinear penalty, mirroring r_k = (mean)^M : one missing claim hurts a lot */
    return pow(frac, (double)total);
}

/* Aggregation: replace equal-weight majority vote with reliability-weighted vote.
   Cluster the 16 candidates by rendered-graph equivalence, sum clr_reliability
   within each cluster, and pick the argmax cluster. */
```

`pow()` is already permitted (`libm` is a core dependency). No new VM opcode, no `src/microgpt.{c,h}` change — consistent with the project's engine-surface and opcode locks (E07–E15).

### 3.4 Falsification risk (pre-stated honestly)

CLR re-ranks; it **cannot rescue a candidate the model never generates**. The project's own diagnosis is that the residual wiring failures are at *generation*, not *ranking* time (`RESEARCH_MANIFOLD_LEARNING.md`: "no re-ranking strategy can recover from a diffuse-prior ceiling… failure at generation, not ranking"). On the leakage-free clean set the wiring transformer alone is ~35%. Therefore CLR's expected lift is **bounded to the cases where ≥1 of the 16 candidates is correct-but-currently-out-voted** — i.e. where a clean-verifying minority loses to a popular-but-broken majority. If that overlap is empty, CLR adds nothing.

A candidate experiment should pre-state: measure on the **leakage-free Phase 2c clean set**, with `--no-anchor` to isolate the transformer; expected lift is small (a few points at most) and is **0 if generation never surfaces a correct candidate**. That outcome would *confirm* the project's generation-bottleneck thesis, not contradict it — making this a low-cost, can't-lose diagnostic regardless of sign.

**This is the only idea here worth an E-number.** Before locking it, re-read `RESEARCH_PIPELINE_IR.md` (wiring arc, Phases 1–15) and `RESEARCH_MANIFOLD_LEARNING.md` in full.

### 3.5 Measured outcome (E18)

After reading the full wiring arc, this was run as **[E18](../../../../experiments/E18-clr-reliability-reranker.md)**. The §3.4 ceiling held — and tightened into a clean falsification. On the 20 leakage-free clean paraphrases, wiring transformer only:

| Selector | Result |
|---|---|
| **Oracle@16** (correct candidate in pool at all?) | **35%** |
| **Majority@16** | **35%** |
| **CLR@16** | **35%** |

`Oracle@16 = Majority@16` ⇒ **zero re-ranking headroom**: the correct answer is absent from the pool for 13/20 clean prompts, so no re-ranker (CLR included) can promote it. And CLR specifically can't discriminate because the IR verifier is *structural* — the failing candidates verify and execute fine (uniform reliability `r=1`), so CLR collapses to majority. The lever is candidate **generation/retrieval**, not scoring: injecting the retrieved anchor candidate raises Oracle 35→100%. **CLR falsified as a lift; the oracle-bound proof is the keeper.**

### 3.6 Where CLR *does* transfer — E19 (games)

E18 left a converse open: does CLR work where a **semantic** (not structural) verifier exists? [E19](../../../../experiments/E19-game-clr-quality-verifier.md) answers yes. On Connect-4, a cheap 1-ply *quality* verifier turns the oracle-first probe around: on 273 critical decisions, **Baseline (ensemble pick) good = 37%, Oracle@16 good = 60% → +23pp re-ranking headroom** — the inverse of the wiring null. The model generates good moves; the legality-only Judge just can't select them. **The unifying lesson across E18+E19: the project's deterministic Judges are all *structural*; CLR (and test-time scaling generally) pays off exactly when you add a cheap *semantic*/quality verifier — and not before.**

---

## 4. Idea 2 (framing): the Compression-Coverage hypothesis as a shared vocabulary

VibeThinker's central claim (§1, §4) splits capability by the *structural form* of its parameter demand:

- **Parameter-dense** — verifiable reasoning (search, constraint satisfaction, error correction, composition). Compresses into a small reusable core.
- **Parameter-expansive** — open-domain knowledge / long-tail facts. Needs broad coverage that scales with raw parameter count.

This is **independent external corroboration** of the project's own calibrated thesis (exploration-sourced from `ORGANELLE_STATE.md`): the organelle stack hits **100% on anchored compositional generation** (the *reasoning* axis) but a structural **~80% bag-of-features ceiling tied to curator-vocabulary coverage** (the *knowledge* axis). Read through VibeThinker's lens, the project's ceiling is not a reasoning failure — it is exactly the parameter-expansive *coverage* axis that the hypothesis predicts a tiny model cannot compress.

What the lens *buys*:

- A naming for **why** the residual failures sit where they do (coverage-bound, not reasoning-bound), consistent with the project's already-documented diagnoses.
- A sharp prediction about ceiling-breakers: they must be **coverage-side** — external pretrained embeddings (already named as gated by the zero-dep policy) or domain-restriction (where families have genuinely distinctive nouns). It predicts that *reasoning-side* tricks (more composition machinery, better search) will **not** move the ~80% number.

**Boundary (discipline):** this is offered as a *lens*, not a rewrite. No edit to `ORGANELLE_STATE.md` is proposed here. If the maintainers find the framing useful, importing it into the state doc is a separate decision that should follow a full re-read of the Markets/wiring arcs, not this excerpt-level note.

---

## 5. Idea 3 (cheap): brevity tie-breaker — the non-RL survivor of Long2Short

Long2Short (§2.2) redistributes RL reward toward *shorter correct* trajectories. The RL machinery is out of scope (§0), but the **preference itself** survives as a deterministic re-rank tie-breaker, no training involved:

> Among best-of-N candidates that verify **equally** (same `clr_reliability`, or all clean), prefer the one with the **fewest nodes / shortest rendered trace**.

This is ~5 lines in the existing wiring re-ranker, and it composes naturally with §3: use `clr_reliability` as the primary sort key and `-graph_size` as the tie-break. Rationale matches the project's audit-native values — a smaller verified graph is cheaper to audit and execute, and a parsimonious solution is less likely to carry an incidental-but-unverified flourish. Pre-statable A/B: does brevity-tie-breaking change the chosen graph on the clean set, and when it does, is the shorter graph at least as correct? (Expected: neutral-to-slightly-positive; the win is audit cost, not accuracy.)

---

## 6. Idea 4 (cheap): Pass@K checkpoint selection for Model Soup / specialists

Diversity-Exploring Distillation (§2.1) has two halves. The teacher→student half is E13-falsified and out of scope (§0). The **transferable half** is its *selection criterion*: VibeThinker picks per-domain specialist checkpoints by **Pass@K** (the checkpoint producing the most *valid distinct solutions*) rather than by lowest validation loss, then merges specialists at the parameter level.

The project already has both ingredients: `organelle_train_soup()` (element-wise averages independently-seeded models) and per-role planner/player/judge specialists. The swap is cheap:

> Select soup ingredients / role-specialist checkpoints by **Pass@K on a probing set** (count valid solutions the verifier accepts) instead of lowest val-loss, then soup/merge exactly as today.

Pre-statable A/B: soup-by-loss vs soup-by-Pass@K on the same seeds, evaluated on the clean held-out set. Cost is a few extra verifier passes per checkpoint; no architecture change. Honest caveat: on memorisation-style organelles (games) "valid solutions" and "low loss" may coincide, so the win — if any — is expected on the *generalisation* organelles (wiring), where solution **diversity** is the thing soup is meant to preserve.

---

## 7. What this document does NOT claim

- It does **not** reopen RL/MGPO, curriculum, or teacher-student distillation — all three are closed here (§0) and nothing above depends on them.
- The four survivors are, respectively: a **test-time** re-rank (§3), a **framing lens** (§4), a **deterministic tie-breaker** (§5), and a **selection-criterion swap** (§6). None requires a new training objective or engine-surface change.
- None of these is an experiment yet. §3 is the only candidate worth an E-number; promoting it requires a full re-read of the wiring arc and a locked pre-registration with the §3.4 ceiling stated up front.
- Project-state assertions here are **exploration-sourced** (§0) except the verifier API, which was author-verified. Challenge any synthesis with *"name what you've read in full"* — the honest answer for project state is "the Explore sub-agent reports, not the underlying arcs end-to-end."

---

## References

- [VibeThinker-3B.pdf](VibeThinker-3B.pdf) — the paper (§2.1–2.4 methods, §3.1 CLR, §1/§4 Compression-Coverage). Summary: [README.md](README.md).
- [`RESEARCH_PIPELINE_IR.md`](../../RESEARCH_PIPELINE_IR.md) — wiring organelle arc (Phases 1–15), best-of-16 + verify-as-judge, leakage audit.
- [`RESEARCH_MANIFOLD_LEARNING.md`](../../RESEARCH_MANIFOLD_LEARNING.md) — generation-not-ranking bottleneck; why re-ranking can't beat a diffuse-prior ceiling (the §3.4 risk).
- [`ORGANELLE_STATE.md`](../../ORGANELLE_STATE.md) — the project's calibrated thesis and the ~80% coverage-bound ceiling (the §4 corroboration target).
- [`experiments/README.md`](../../../../experiments/README.md) — experiment registry; **E13** (distillation falsified), **E15** (composition test) referenced above.
- Pipeline-IR verifier — `libs/pipeline_ir/include/pipeline_ir/pipeline_ir.h` (`pipeline_verify`, `pipeline_verify_partial`).
