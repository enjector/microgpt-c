# E18 — Oracle-gap analysis: can VibeThinker's CLR test-time scaling break the wiring ceiling?

**Status:** ✅ **Measured — prediction confirmed (negative result on CLR; positive on the oracle-bound formalisation).**

**Origin:** [`docs/research/papers/VibeThinker/IDEAS.md`](../docs/research/papers/VibeThinker/IDEAS.md) §3 named Claim-Level Reliability Assessment (CLR) as the one VibeThinker-3B idea worth an experiment. This is that experiment. Suggested as an E18 follow-on in [E15 §4](E15-composition-vs-monolithic.md).

---

## 0. Spear summary

**Point.** VibeThinker-3B's headline test-time-scaling trick — **CLR** (generate K trajectories, self-verify M claims each, weight each trajectory by a nonlinear reliability `r_k = (mean verdict)^M`, aggregate) — was mapped onto the wiring organelle's best-of-16 + verify-as-judge loop. The pre-registered prediction was that it would **not** lift the leakage-free clean-set headline, for a precise reason the full arc (`RESEARCH_PIPELINE_IR.md` §32, §38) implies: the failures are *generation*-side, and the Pipeline-IR verifier is a *structural* judge, not a semantic one. The experiment confirms the prediction and, more usefully, **formalises the generation ceiling with a metric the 17-phase arc never computed: Oracle@N.**

**Picture.** Phase 1a (Vietoris-Rips modal-cluster re-rank) and Phase 1c (geodesic hint + bonus) already showed re-ranking can't break the ceiling, but neither measured the *information-theoretic ceiling for any re-ranker* — i.e. **is a correct candidate ever in the 16-vote pool?** Oracle@N is that ceiling. If `Oracle@N == Majority@N`, then no re-ranker (CLR included) can do anything: the right answer simply isn't there to be promoted.

**Proof.** On the 20 leakage-free Phase 2c paraphrases, wiring transformer only (`--no-anchor`):

| Selector | Result | Meaning |
|---|---|---|
| **Oracle@16** | **7/20 (35%)** | ceiling for *any* re-ranker |
| **Majority@16** | **7/20 (35%)** | pure self-consistency vote |
| **CLR@16** | **7/20 (35%)** | VibeThinker reliability-weighted aggregation |

All three equal. Re-ranking headroom is **zero**. The 35% reproduces the documented clean-set baseline (`RESEARCH_PIPELINE_IR.md` §38), validating the harness.

**Push.** CLR is **falsified as a lift mechanism** at this scale, for a now-quantified reason. The result strengthens the project's own thesis — *the bottleneck is candidate generation, not candidate scoring* — and adds the missing oracle-bound proof. Ship the measurement code (gated behind `--oracle`, zero default-behaviour change); do **not** ship a CLR re-ranker.

---

## 1. Proposal (pre-registered)

### 1.1 Hypothesis (locked)

> On the leakage-free clean set, `Oracle@16 ≈ Majority@16 ≈ CLR@16 ≈ 35%`. The wiring transformer's 35% ceiling is a **generation** ceiling with no re-ranking headroom; VibeThinker's CLR — a re-ranker over a fixed candidate pool — therefore cannot lift it.

### 1.2 Mechanism

CLR (VibeThinker-3B §3.1, eqs. 5–6) is weights-frozen test-time scaling: sample `K` trajectories, extract `M` decision-relevant claims each, self-verify each claim to a binary verdict `v_{k,m}`, score each trajectory `r_k = ((1/M)Σ_m v_{k,m})^M`, cluster answers, pick the cluster maximising `Σ r_k`. Mapped onto the wiring organelle:

- **Trajectories** = the best-of-16 candidate graphs already collected per prompt (`demos/wiring_organelle/main.c`).
- **Claims / verdicts** = the per-candidate execution on the 5 standard input sets. Reliability `r = (valid_results / 5)^5` — the nonlinear `(mean verdict)^M` with `M = 5`.
- **Aggregation** = cluster candidates by identical 5-vector, sum reliability per cluster, pick the argmax (replacing plain majority count).

Three selectors are computed over the **same** candidate pool, purely for measurement (the headline pick is untouched):

| Selector | Definition |
|---|---|
| **Oracle@N** | ∃ candidate whose 5-vector == reference 5-vector (executed on all 5). |
| **Majority@N** | self-consistency vote winner (no anchor/geo/planner bonus); correct iff == reference. |
| **CLR@N** | reliability-weighted cluster argmax; correct iff == reference. |

A **brevity** diagnostic (Long2Short survivor, `IDEAS.md` §5) also records the node-count of the CLR pick vs the smallest correct candidate, as an audit-cost measure.

### 1.3 Pre-registered targets

| ID | Target | Floor / falsification |
|---|---|---|
| **T1** | `Oracle@16` on clean set ≈ 35% (i.e. == documented wiring baseline) | If `Oracle@16 ≫ 50%`, the pool contains correct candidates the vote misses → re-ranking *does* have headroom → reopen the question. |
| **T2** | `Majority@16 == Oracle@16` (vote already saturates the ceiling) | If `Majority@16 ≪ Oracle@16`, a *naive* re-ranker is leaving correct candidates unpicked. |
| **T3** | `CLR@16 ≤ Oracle@16` and `CLR@16` shows **no lift** over `Majority@16` | If `CLR@16 > Majority@16` by ≥2 prompts, CLR captures real headroom → ship it. |
| **T4** | Harness sanity: `correct-on-all-5 == 35% (7/20)` matches `RESEARCH_PIPELINE_IR.md` §38 | If it doesn't reproduce 7/20, the measurement harness is wrong. |
| **T5** | No regression: default-mode (no `--oracle`) behaviour bit-identical | Flag gates every change; the 100% anchor headline must reproduce. |

### 1.4 Why it matters

The 17-phase arc + manifold-retrieval addendum concluded "the failure is at generation, not re-ranking" (`RESEARCH_PIPELINE_IR.md` §32, §44) — but proved it *indirectly* (specific re-rankers failed). Oracle@N is the **direct** proof: it bounds *every possible* re-ranker at once. Confirming `Oracle == Majority` closes the re-ranking question with a single number, and tells the field something clean about why claim-level test-time scaling (a frontier-LLM technique) does not transfer to a tiny structural-verifier regime.

### 1.5 What this is NOT testing

- Not testing whether CLR helps *frontier* models (it does — that's VibeThinker's result). It tests whether it transfers to a 540K structural-verifier organelle.
- Not re-opening RL/MGPO, curriculum, or distillation (out of scope per `IDEAS.md` §0).
- Not changing the headline pick or the anchor-retrieval system. Pure measurement.

### 1.6 Falsification risk

**Low-to-moderate.** The arc's bimodal finding (§22.5: "every executing prompt is solidly 5/5 or 0/5") and Phase 1a's "16/16 unanimous on the wrong answer" (§32) strongly predict `Oracle == Majority`. The genuine risk is on the full-pool (anchor-on) variant, where the injected anchor candidate *could* create an oracle/vote gap — and it does (see §3.3), which is the experiment's most informative surprise.

---

## 2. Initial state

### 2.1 Baselines to beat / reproduce

- Wiring transformer alone on clean set: **35% (7/20)** — `RESEARCH_PIPELINE_IR.md` §38, reproducible via `./wiring_organelle_demo --no-anchor --clean-only`.
- Full anchor-retrieval system on clean set: **100% (20/20)** — §37.
- Prior re-ranking attempts that hit the ceiling: Phase 1a (VR modal cluster, §32) → 70%; Phase 1c (geo hint + bonus, §34) → 70%. Neither measured the oracle bound.

### 2.2 Dependencies

- `demos/wiring_organelle/main.c` best-of-16 vote loop + cached 5-vectors per candidate (Phase 8).
- `wiring_references.c` reference 5-vectors (the ground-truth oracle).
- Cached checkpoints (`build/wiring_organelle{,_2,_3}.ckpt`, `wiring_planner.ckpt`); no retrain.

### 2.3 Locked implementation choices

- New `--oracle` CLI flag. It (a) disables the first-fidelity early-break so the pool is the genuine full 16 votes, and (b) emits the three selectors + brevity. **Every change is gated on the flag**; default behaviour is unchanged.
- `M = 5` claims = the 5 input-set executions (the only per-candidate structural verdict available; the verifier itself gives a single pass/fail already consumed at collection).

---

## 3. Implementation + results

### 3.1 What was built

- `demos/wiring_organelle/main.c`: `--oracle` flag; `count_node_lines()` + `e18_vec_eq()` helpers; a pure-measurement block after the Phase 8 correctness count computing Oracle@16 / Majority@16 / CLR@16 / brevity over the full candidate pool; gated early-break; a printed "E18: VibeThinker CLR study" section. **~120 LOC, zero engine-surface change, zero new build deps.**

Reproduce:
```sh
./wiring_organelle_demo --no-anchor --clean-only --oracle   # wiring pool
./wiring_organelle_demo --clean-only --oracle               # full pool (anchor on)
```

### 3.2 Wiring pool (`--no-anchor --clean-only --oracle`)

| Metric | Result |
|---|---|
| Best-of-16 correct on all 5 (existing headline) | 7/20 (35%) — reproduces §38 ✓ (**T4 PASS**) |
| **Oracle@16** | **7/20 (35%)** |
| **Majority@16** | **7/20 (35%)** |
| **CLR@16** | **7/20 (35%)** |
| Brevity (CLR pick vs smallest correct) | 2.1 vs 2.1 nodes (n=7) |

**All three selectors equal at 35%.** For the 13 failing prompts, *no* candidate among the 16 is correct — re-ranking headroom is exactly zero.

### 3.3 Full pool (`--clean-only --oracle`, anchor retrieval ON)

| Metric | Result |
|---|---|
| **Oracle@16** | **20/20 (100%)** |
| Majority@16 | 16/20 (80%) |
| CLR@16 | 16/20 (80%) |
| System headline (anchor source-trust bonus) | 20/20 (100%) |
| Brevity | 2.5 vs 2.5 nodes (n=20) |

Injecting the **retrieved** anchor candidate raises the oracle 35→100% — what re-ranking structurally cannot do. But generic re-rankers (majority *and* CLR) reach only 80%: the single correct anchor is out-voted by 16 verifying-but-wrong wiring votes, and **CLR can't break the tie because the verifier is structural** — all 16 wiring candidates execute on 5/5 inputs, so every candidate gets reliability `r = 1`, and CLR degenerates exactly to majority. The system reaches 100% only via its **source-trust** signal (the +30/+60 geodesic+planner agreement bonus), i.e. trusting *where a candidate came from*, not *how reliably it verifies*.

### 3.4 The oracle gap, side by side

| Selector | Wiring pool | Full pool (anchor) |
|---|---:|---:|
| Oracle@16 (any-correct ceiling) | 35% | **100%** |
| Majority@16 | 35% | 80% |
| CLR@16 | 35% | 80% |
| Shipped system | 35% | **100%** |

---

## 4. Conclusion

### 4.1 Verdict vs pre-registered targets

| ID | Target | Result | Verdict |
|---|---|---|---|
| T1 | Oracle@16 ≈ 35% | 35% (7/20) | ✅ PASS |
| T2 | Majority@16 == Oracle@16 | both 35% | ✅ PASS |
| T3 | CLR shows no lift over Majority | both 35% (wiring) / both 80% (full) | ✅ PASS — **CLR falsified as a lift** |
| T4 | Harness reproduces 7/20 baseline | 7/20 | ✅ PASS |
| T5 | No default-mode regression | flag-gated; 100% headline reproduces | ✅ PASS |

**All five targets met. CLR is falsified as a lift mechanism for the wiring organelle at this scale.**

### 4.2 What was learned

1. **The wiring ceiling is a generation ceiling, proven by the oracle bound.** `Oracle@16 = Majority@16 = 35%` means the correct answer is absent from the candidate pool for 13/20 clean prompts. This is the *direct* form of the arc's §32/§44 conclusion — no re-ranker, however clever, can promote an answer that was never generated.

2. **CLR doesn't transfer because the IR verifier is structural, not semantic.** VibeThinker's CLR derives signal from *claim verification*. The Pipeline-IR verifier checks types/cycles/connectivity — exactly the dimensions on which the failing candidates are *correct*. They verify and execute fine; they compute the wrong number (`RESEARCH_PIPELINE_IR.md` §21.6). So the reliability score is uniform (`r=1`) precisely where discrimination is needed, and CLR collapses to majority vote.

3. **Retrieval, not re-ranking, is the lever — and the new insight is the residual 20pp.** Even after anchor injection makes the oracle 100%, generic re-ranking leaves 20pp on the table (80% vs 100%). That gap is recoverable only by a *source-trust* signal (which retrieved candidate the classifiers agree on), not by verifier/claim reliability. This sharpens the arc's "deterministic Judge stack does the heavy lifting" claim: the load-bearing test-time signal is *provenance*, not *self-consistency*.

4. **Brevity (Long2Short survivor) is a no-op at this scale.** Correct graphs cluster at 2.1–2.5 nodes with no shorter correct alternative, so the brevity tie-breaker never fires. Audit-cost neutral; revisit only if anchor/composition libraries grow deeper graphs.

5. **Pass@K-soup (Diversity-Exploring Distillation survivor) is already answered.** The pool is a 3-seed ensemble (`ENSEMBLE_SIZE=3`, Phase 17). `Oracle@16 = 35%` over that 3-seed pool means seed diversity adds *no* correct candidates — a fresh, direct confirmation of Phase 17's "failures are correlated across seeds" (§31.3), without a redundant retrain.

### 4.3 Disposition

- **Ship:** the `--oracle` measurement instrument (it's a permanent, zero-cost diagnostic of the generation ceiling for any future candidate source).
- **Do NOT ship:** a CLR re-ranker. Falsified.
- **Headline of record:** unchanged. The system is 100% on clean paraphrases via retrieval + source-trust; the wiring transformer alone is 35%; re-ranking (CLR included) cannot move the latter.

### 4.4 Traceability / next moves

- The genuine lever remains candidate **generation / retrieval** (anchor-table extension, axis-1 of `RESEARCH_PIPELINE_IR.md` §44.3) — labour-bounded, not research-bounded.
- A CLR-*flavoured* idea that *could* help would weight by **source provenance** (geodesic+planner agreement), which the system already does — i.e. the project independently arrived at the part of CLR that transfers, and skipped the part (claim/verifier reliability) that doesn't.
- Cross-refs: [`docs/research/papers/VibeThinker/IDEAS.md`](../docs/research/papers/VibeThinker/IDEAS.md) §3 (origin), [`docs/research/RESEARCH_PIPELINE_IR.md`](../docs/research/RESEARCH_PIPELINE_IR.md) §32/§38/§44 (generation-bottleneck arc), [`docs/research/RESEARCH_MANIFOLD_LEARNING.md`](../docs/research/RESEARCH_MANIFOLD_LEARNING.md) (re-ranking-can't-help-diffuse-prior).
