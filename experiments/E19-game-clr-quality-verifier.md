# E19 — Does CLR transfer to games? Oracle-first probe finds the headroom wiring lacked

**Status:** 🔬 **Oracle-first probe MEASURED (+23pp headroom confirmed on Connect-4).** The full verifier-gated CLR re-rank + punishing-opponent measurement is **pre-registered below but not yet run.**

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

**Push.** The "headroom exists" branch fires. The full verifier-gated CLR re-rank (§1) is worth running — with a *punishing* opponent, because random O masks the lift.

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

---

## 4. Conclusion (probe verdict; full re-rank pending)

### 4.1 Probe verdict

**Headroom confirmed (+23pp).** CLR is not dead in this codebase — E18 falsified it for *wiring* (structural verifier, generation-bound), and E19's probe shows it *transfers to games* where a cheap semantic verifier exists and the candidate pool already contains good moves. The general finding: **the project's deterministic Judges are all structural; the lever that makes test-time scaling pay off is adding a cheap semantic/quality verifier.**

### 4.2 Next (the pre-registered §1 re-rank)

Two-commit pattern: this commit lands the probe + pre-registration; a follow-up implements the verifier-gated CLR re-rank + a 1-ply/greedy opponent and measures T1–T4. Expected: critical-decision good-rate 37%→~55-60%, win-rate lift modest vs random O, ≥+10pp vs a punishing O.

### 4.3 Traceability

- Origin: [E18](E18-clr-reliability-reranker.md); [`docs/research/papers/VibeThinker/IDEAS.md`](../docs/research/papers/VibeThinker/IDEAS.md) §3.
- The generation-vs-ranking dichotomy: [`docs/research/ORGANELLE_STATE.md`](../docs/research/ORGANELLE_STATE.md) June 2026 update.
- Related named-unrun target: H10 (8-puzzle hard-tier), `ORGANELLE_STATE.md` open-question #4.
