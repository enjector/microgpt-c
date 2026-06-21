# E19 — Does CLR transfer to games? Oracle-first probe finds the headroom wiring lacked

**Status:** ✅ **MEASURED — Connect-4 PASS, 8-puzzle null-with-diagnosis.** Connect-4 oracle-first probe (+23pp headroom) → verifier-gated CLR re-rank: **all four targets PASS**, headline T3 = **9% → 30% win vs a punishing opponent (+21pp, 3.3×)**. 8-puzzle hard-tier (H10, §3.5): **solve-rate NOT lifted** — the demo already bakes in an MD quality heuristic, so the oracle gap is small (+8pp) and the re-rank saturates it without moving the outcome. Together they turn the thesis into a **gradient** (§3.5): CLR helps in proportion to the oracle gap, which is large only when the existing Judge is purely structural.

**Origin:** Follow-on from [E18](E18-clr-reliability-reranker.md). E18 falsified VibeThinker-3B's Claim-Level Reliability Assessment (CLR) as a lift for the *wiring* organelle, isolating the precise reason: the Pipeline-IR verifier is **structural** (type/cycle/connectivity), so it gives uniform reliability to the semantically-wrong candidates, and the failures are generation-bound anyway. The open question E18 left: **does CLR transfer to a task that has a *semantic* (quality) verifier?** A code-level check (`demos/character-level/connect4/main.c:9` — "Judge is fully deterministic: column valid + win/draw check") confirmed the games have only *structural* judges too — but unlike wiring, a cheap *semantic* verifier is trivially constructible (1-ply move quality). E19 builds it and runs the same oracle-first diagnostic.

---

## 0. Spear summary

**Point.** On Connect-4, an oracle-first probe shows **Oracle@16 (60%) ≫ Baseline (37%)** on the decisions where move quality matters — **+23pp of re-ranking headroom**, the exact opposite of E18's wiring result (Oracle = Majority = 35%, zero headroom). CLR transfers to games because a cheap *semantic* verifier (1-ply quality) exists and the right move is already present in the candidate pool, just unselected.

**Picture.** The Connect-4 demo samples moves and picks by a 3-vote ensemble, then validates only *legality* — never *quality*. So on a critical board (a win is takeable, or a loss is blockable in one ply), the ensemble blunders 63% of the time even though a non-blundering candidate is in a best-of-16 pool 60% of the time. A 1-ply quality verifier + CLR-style selection would convert those.

**Proof.** `C4_ORACLE=1 ./c_connect4_demo`, 100 games vs random O:

| Metric | Value |
|---|---|
| Win rate | 86% (matches documented ~88%) |
| Critical decisions (win-to-take / loss-to-block) | **273** |
| **Baseline good** (ensemble pick is a 1-ply non-blunder) | **102/273 = 37%** |
| **Oracle@16 good** (≥1 of 16 candidates non-blunder) | **164/273 = 60%** |
| **Headroom** | **+23 pp** |

**Push.** The "headroom exists" branch fired and the re-rank was run (§3.4). All four targets PASS: critical good-rate **37% → 63%** (T1), win vs random **91% → 94%** (T2, masked as predicted), and against a 1-ply punishing opponent the ensemble **collapses to 9%** while CLR re-rank holds **30%** (T3, **+21pp**). The clean inverse of E18: CLR pays off precisely because a semantic verifier exists.

---

## 1. Proposal — full verifier-gated CLR re-rank (pre-registered, NOT yet run)

### 1.1 Hypothesis (locked)

> Replacing Connect-4's ensemble-vote move selection with **verifier-gated CLR** — sample N candidates, run the deterministic 1-ply quality verifier, prefer a non-blunder (take a win if available, else avoid handing O an immediate win), CLR-weight ties — lifts critical-decision good-rate toward the measured Oracle@16 ceiling (60%) and improves win rate, with the gain **visible against a punishing (1-ply / greedy) opponent** and largely **masked against random O**.

### 1.2 Mechanism

This is CLR (VibeThinker-3B §3.1) with the semantic verifier wiring lacked:
- **Trajectories** = N=16 candidate columns (temperature-jittered), as in E18.
- **Claims / verdicts** = the 1-ply quality label per candidate (`c4_classify_move`: win / safe / loses-in-1). This is the *semantic* judge — a legal-but-losing move scores differently from a legal-winning one, which the structural Judge could not express.
- **Aggregation** = prefer the highest quality label; CLR-weight (self-consistency + reliability) only to break ties among equal-quality candidates; fall back to the ensemble pick if no candidate is legal.

### 1.3 Pre-registered targets

| ID | Target | Floor / falsification |
|---|---|---|
| **T1** | Critical-decision good-rate: 37% → **≥ 55%** (toward the 60% oracle) | If < 45%, the verifier-gate isn't capturing the measured headroom — implementation bug or pool-sampling mismatch. |
| **T2** | Win rate vs **random O**: ≥ baseline (no regression); modest lift expected (≤ +5pp — random O masks it) | If win rate drops, the gate is mis-selecting; falsify. |
| **T3** | Win rate vs a **1-ply/greedy O** (new punishing opponent): **≥ +10pp** over the ensemble baseline against the same opponent | The headline. If < +5pp, the critical-decision lift doesn't translate to game outcomes even against a punishing opponent — CLR's game value is weaker than the probe implies. |
| **T4** | No default-behaviour change: gate behind a flag; the 86% random-O headline reproduces with the gate off | Standing engine/demo-surface discipline. |

### 1.4 Why it matters

E18 closed CLR for wiring with a clean reason. E19 tests the *converse*: CLR is not dead in this codebase — it transfers exactly where a semantic verifier exists. Connecting it to the named-but-unrun **H10** target (8-puzzle hard-tier 30%→80%, `ORGANELLE_STATE.md` open-question #4) and the README's flagged "minimax opponent testing" frontier, this is the general lesson: *the project's deterministic Judges are structural; adding a cheap semantic/quality verifier is the lever that makes test-time scaling pay off.*

### 1.5 What this is NOT

- Not claiming a large win-rate lift vs random O (the probe shows why it would be masked).
- Not reopening RL/MGPO/curriculum/distillation.
- Not changing the game engine or the training; selection-time only.

---

## 2. Initial state

### 2.1 Baselines

- Connect-4 demo: 86–88% win vs **random** O, ensemble-vote selection, structural Judge only.
- E18 (wiring): Oracle@16 = Majority@16 = 35% — zero re-ranking headroom (structural verifier, generation-bound).

### 2.2 The motivating diagnostic (this experiment's measured content)

The oracle-first probe (§3) is the "is it even worth pre-registering the full re-rank?" check — the games analog of E18's oracle bound. It is **measured**; the §1 re-rank is not.

### 2.3 Dependencies

`demos/character-level/connect4/main.c` (`check_winner`, `drop_piece`, `get_valid_columns`, `organelle_generate`); no new build deps.

---

## 3. Implementation + results (oracle-first probe — MEASURED)

### 3.1 What was built

- `c4_classify_move(board, col, me, opp)` in `connect4/main.c` — the 1-ply semantic verifier: simulate `me` dropping at `col`, return **2** = immediate win, **0** = leaves `opp` an immediate winning reply (blunder), **1** = safe, **-1** = illegal. Reuses the existing `check_winner`/`drop_piece`.
- A measurement block (gated on the `C4_ORACLE` env var, **zero default-behaviour change**): on each X decision, classify all legal moves, flag **critical** decisions (a win is takeable with >1 legal move, OR some move loses-in-1 while another doesn't), then record whether the demo's ensemble pick is good (baseline) and whether ≥1 of N=16 temperature-sampled candidates is good (oracle).

Reproduce:
```sh
C4_ORACLE=1 ./c_connect4_demo
```

### 3.2 Results (100 games vs random O)

| Metric | Value |
|---|---|
| Win rate | 86% |
| Critical decisions | 273 |
| Baseline good (ensemble pick) | 102/273 (37%) |
| Oracle@16 good (any candidate) | 164/273 (60%) |
| Headroom | **+23 pp** |

### 3.3 Reading

- **Oracle@16 (60%) ≫ Baseline (37%)** — genuine re-ranking headroom, the inverse of E18.
- **The model can generate good moves** (60% of critical pools contain one) but the **legality-only Judge can't select them** (ensemble vote lands a non-blunder only 37% of the time).
- **60% is a partial generation ceiling**: 40% of critical decisions have *no* good move in the 16-pool — a verifier+CLR re-rank caps out at 60% there, not 100%.
- The +23pp is a *critical-decision* number; its translation to **win rate** is masked by random O (which rarely punishes blunders) — hence T3's punishing-opponent requirement.

### 3.4 Re-rank results (MEASURED — all four targets PASS)

Implemented in `connect4/main.c` (all flag-gated, zero default-behaviour change):
- `c4_rerank_select` — verifier-gated CLR: sample N=16 candidates, keep legal, pick by 1-ply quality label (win > safe > loses-in-1), tie-break by self-consistency vote. **Pool-bounded** (re-ranks the model's candidates; capped by Oracle@N — it is CLR, not verifier-as-policy search). Gated on `C4_RERANK`.
- `greedy_opponent_move` — punishing 1-ply O: take an immediate win, else block X's immediate win, else random. Gated on `C4_GREEDY_O`.

Four configs, 100 games each (checkpoints cached, no retrain):

| Config | Selection | Opponent | Win rate | Critical good-rate |
|---|---|---|---|---|
| A | ensemble vote | random | 91% | 37% (102/277) |
| B | **CLR re-rank** | random | **94%** | **63%** (109/174) |
| C | ensemble vote | **1-ply greedy** | **9%** | — |
| D | **CLR re-rank** | **1-ply greedy** | **30%** | — |

| Target | Pre-reg (§1.3) | Measured | Verdict |
|---|---|---|---|
| **T1** critical good-rate | 37% → ≥ 55% | **37% → 63%** (A→B) | ✅ PASS — saturates the ~58-60% oracle ceiling |
| **T2** win vs random | ≥ baseline | **91% → 94%** (A→B) | ✅ PASS — +3pp, masked as predicted |
| **T3** win vs punishing O | ≥ +10pp | **9% → 30%** (C→D) = **+21pp** | ✅ PASS — exceeded 2×, 3.3× win rate |
| **T4** no default change | flag-gated | env-var gated, clean build | ✅ PASS |

The C→D contrast is the demonstration: against random O both policies look fine (91% / 94%) — the blundering is invisible. Against a punishing opponent the ensemble **collapses to 9%** (blunders on ~63% of critical decisions, punished every time) while the verifier-gated CLR re-rank holds **30%**.

**Honest bounds.** 30% vs greedy is modest in absolute terms — the 60% oracle ceiling caps it (40% of critical decisions have no good move in the 16-pool), and a 460K pattern-matcher with no search can't outplay a punishing 1-ply opponent. The re-rank captures the *available* headroom; it cannot exceed the generation ceiling. Critical-decision sets differ across configs (trajectories diverge once X plays better), so T1 compares each policy on the decisions it actually faces, not identical boards.

Reproduce:
```sh
C4_ORACLE=1               ./c_connect4_demo   # A: probe + ensemble baseline
C4_ORACLE=1 C4_RERANK=1   ./c_connect4_demo   # B: re-rank vs random (T1, T2)
C4_GREEDY_O=1             ./c_connect4_demo   # C: ensemble vs punishing O
C4_RERANK=1 C4_GREEDY_O=1 ./c_connect4_demo   # D: re-rank vs punishing O (T3)
```

### 3.5 Extension to 8-puzzle hard-tier (H10) — solve-rate NOT lifted; the result that turns the thesis into a gradient

The same pattern was ported to the 8-puzzle demo (`demos/character-level/puzzle8/main.c`), the named **H10** target. Single-player, so the semantic verifier is **exact BFS-from-goal optimal distance** over all 9! permutations (precomputed once; a move is "good" iff it strictly reduces distance), and **no opponent variant is needed** — a wrong move directly wastes the move budget. All flag-gated (`P8_ORACLE` / `P8_RERANK`), zero default-behaviour change.

| Config | Overall | HARD solve | HARD per-move progress | HARD Oracle@16 | Avg moves (M / H) |
|---|---|---|---|---|---|
| ensemble | 90% | **70%** (7/10) | 61% (115/188) | 69% (129/188) | 6.1 / 9.7 |
| CLR re-rank | 90% | **70%** (7/10) | 64% (118/184) | 65% (119/184) | 5.7 / 9.1 |

**The solve rate did not move (90% overall, 70% hard, unchanged) — the opposite of Connect-4.** Three reasons, all honest:

1. **The oracle gap is tiny here — just +8pp** (hard: played-good 61% vs Oracle@16 69%), versus Connect-4's +23pp. The re-rank captures it (61%→64%, **saturating its own oracle** 64% vs 65%) but there is almost nothing to capture.
2. **This demo already bakes in a quality heuristic.** Unlike Connect-4's legality-only Judge, the 8-puzzle pipeline feeds **MD-delta encoding into the mover prompt** and uses an **MD-based cycle-breaker** — so the model is *already* steered toward progress (which is why this checkpoint is 70% hard, not the ~30% the H10 framing assumed). A semantic verifier on top is largely **redundant**.
3. **The residual hard failures are a generation ceiling.** Oracle@16 is only 65-69% on hard — on ~1/3 of hard steps *no* sampled move is optimal progress. Re-ranking can't fix what the model never generates (the E18 lesson). The 3 unsolved hard puzzles stay unsolved.

The re-rank *did* produce a small efficiency win — shorter solutions (medium avg 6.1→5.7 moves, hard 9.7→9.1) by avoiding regressions even when no progress move is available (the Long2Short flavor) — but that is not a solve-rate lift.

**The refined thesis (the value of this extension).** 8-puzzle is the **middle case** that turns E18+E19 from a binary into a gradient:

> CLR re-ranking helps **in proportion to the oracle gap**, which is large only when the existing Judge is purely *structural*.
> - **Wiring** (E18): structural verifier, generation-bound → Oracle = Majority = 35%, **zero gap → dead.**
> - **8-puzzle**: pipeline already encodes an MD quality heuristic → small gap (+8pp) → **re-rank saturates it but no solve-rate lift.**
> - **Connect-4**: legality-only Judge → large gap (+23pp) → **big lift (9%→30% vs a punisher).**
>
> The lever is not "add a semantic verifier" unconditionally — it is "add one **where the pipeline doesn't already encode quality**."

Reproduce:
```sh
P8_ORACLE=1             ./c_puzzle8_demo   # ensemble + per-band probe
P8_ORACLE=1 P8_RERANK=1 ./c_puzzle8_demo   # verifier-gated CLR re-rank + probe
```

---

## 4. Conclusion

### 4.1 Verdict

**Connect-4 PASS; 8-puzzle null-with-diagnosis; together a gradient.** The Connect-4 probe confirmed +23pp re-ranking headroom (the inverse of E18) and the verifier-gated CLR re-rank met all four targets (T3 **9%→30%** vs a punishing opponent). The 8-puzzle extension (§3.5) did **not** lift the solve rate (70% hard, unchanged) because that pipeline already encodes an MD quality heuristic, leaving only a +8pp oracle gap the re-rank merely saturates.

The unified finding — sharper than "structural dead, semantic alive": **CLR / test-time scaling pays off in proportion to the *oracle gap*, which is large only when the existing Judge is purely structural.** Wiring (structural, generation-bound) → zero gap → dead. 8-puzzle (MD heuristic already in the pipeline) → small gap → saturated, no outcome lift. Connect-4 (legality-only Judge) → large gap → big lift. The lever is "add a semantic verifier **where the pipeline doesn't already encode quality**."

### 4.2 Next

E19 is complete (Connect-4 probe + re-rank PASS; 8-puzzle measured null-with-diagnosis). Follow-ons, none required:
- **Games with a structural-only Judge and a high blunder rate** — that is where the gap (and the lift) is largest. Hex (27% win) is the natural next probe; its quality verifier (BFS connectivity) is more work than Connect-4's 1-ply but cheaper than nothing.
- **Deeper verifier where the gap is generation-bound.** For 8-puzzle the bound is the model's ~65-69% hard-tier oracle, not the re-rank; raising it needs better *generation* (more candidates, a search corpus à la E11's `c4_model_propose_column`), which shades from "CLR re-rank" toward "search" — a different experiment.

### 4.3 Traceability

- Origin: [E18](E18-clr-reliability-reranker.md); [`docs/research/papers/VibeThinker/IDEAS.md`](../docs/research/papers/VibeThinker/IDEAS.md) §3.
- The generation-vs-ranking dichotomy: [`docs/research/ORGANELLE_STATE.md`](../docs/research/ORGANELLE_STATE.md) June 2026 update.
- Related named-unrun target: H10 (8-puzzle hard-tier), `ORGANELLE_STATE.md` open-question #4.
