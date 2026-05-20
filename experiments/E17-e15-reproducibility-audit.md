# Experiment E17 — Reproducibility audit of E15's measurements committed on `main`

**Status:** 📋 Proposal locked — 2026-05-20.
**Direction:** the first experiment drafted under the new `CLAUDE.md` research-discipline section. Audits the existing E15 measurements (the only multi-target measurement currently merged to `main`) for internal consistency between the claimed Section 3 numbers and the artefacts under `results/`, `data/E15-*/`, and the committed verdict matrix. Not a new architectural claim. Not a new measurement against the world. **A measurement of whether E15's existing measurements reproduce from what's committed.**
**Cost estimate:** ~1-2 days (static audit only; ~1 agent run). Optional Phase 6 dynamic re-run adds 4-8 hours of CPU.
**Falsification risk:** **LOW — intentionally.** The audit should mostly confirm. Any discrepancy is the most important finding — it would invalidate (in part or whole) E15's published verdict.

---

## 0. Why this is the first experiment drafted under the new CLAUDE.md discipline

The session that produced E15 + E16 + the proposed "asymmetric safety" reframe was interrupted when the user noticed I had been synthesising strategic conclusions from `head -50` to `head -80` excerpts of research documents I had not read in full. The CLAUDE.md commit `14c6a7a` formalised the corrective rule: **read source documents end-to-end before drawing conclusions, or refuse to draw them.** E17 is the first experiment drafted under that rule. It deliberately:

- **Scopes only to artefacts on `main`** — no claims about the sibling repo `~/dev/projects/microgpt-c/`, no claims about Bonsai integration, no claims about the V14-V24 Markets arc. Those would require document reads E17 does not depend on.
- **Cites only the experiment doc I read in full** (E15's Section 3 lines 272-636 + Section 4 lines 638-861, read on this session's last turn) and the file listings under `results/`, `data/E15-*/`, `experiments/`.
- **Produces a reading log in §1.9** naming exactly what was read in full vs what is inferred.

The audit's value to the project is not just "did E15 reproduce" but "does the project's measurement discipline produce reproducible artefacts at all." If E17 finds discrepancies, the rest of the project's published verdicts (E02 through E13's PASS/FAIL claims) become open to the same question. If E17 finds clean reproduction, the project has a single quantified data point about its measurement integrity.

---

## 1. Proposal

### 1.1 Hypothesis (locked before measurement)

> *Every quantitative claim in E15's Section 3 (the eight pre-registered targets T1-T8, the solve-rate tables in §3.8, the compute-equivalence math in §3.9, the held-out leakage audit in §3.4, and the engine-surface-frozen claim in T8) is reproducible from artefacts currently committed to `main` — specifically the eval CSVs under `results/`, the corpora under `data/E15-*/`, the experiment doc itself, and `git diff` against the pre-E15 baseline. The audit produces zero discrepancies above the noise floor (±0.1 pp on solve rates; ±0.5% on compute-equivalence ratios; exactly 0 on integer row counts and on engine-surface line-diff counts).*

### 1.2 Why this matters

The session that produced E15 included a confident "T5 = FAIL" headline that has not been independently verified by reading the underlying artefacts. The headline conclusion was then used to motivate E16 (a port + replication attempt) and to draft a substantial proposed rewrite of `ORGANELLE_STATE.md` and the project README that I almost committed. Under the new CLAUDE.md discipline, those downstream commits should not happen unless the source claim is independently verifiable. E17 produces that verification — or surfaces the gap.

If E15 reproduces cleanly: the project has a verified baseline measurement to build on. Future architectural experiments (E16 if merged; E18+; any rewrites of synthesis docs) can cite E15's numbers with confidence.

If E15 *does not* reproduce: the project has surfaced a deeper measurement-integrity issue and the eight prior experiments' PASS/FAIL verdicts should be audited under the same protocol before any further architectural claims are made.

### 1.3 Mechanism

Six phases. Phases 1-5 are static (read committed artefacts, do arithmetic). Phase 6 is optional dynamic re-run from seed.

#### 1.3.1 Phase 1 — Eval CSV row-count audit (T1)

The eval CSVs live at:
- `results/klotski_mono_eval.csv`
- `results/klotski_opa_eval.csv`
- `results/puzzle15_mono_eval.csv`
- `results/puzzle15_opa_eval.csv`

E15 §3.4 claims canonical held-out sizes of **113 Klotski positions** (after Jaccard ≥ 0.7 audit; 1887 dropped from 2000-pool) and **948 Puzzle15 positions** (52 dropped from 1000-pool). E15 §3.8 reports solve counts of 73, 79, 1, 7 respectively.

Audit procedure:
1. `wc -l results/klotski_mono_eval.csv` → expect 114 (113 data + 1 header) or 113 (no header) — auditor reports which
2. Same for the other three eval CSVs
3. Cross-check against `data/E15-klotski/heldout_large.tsv` row count (expect 113) and `data/E15-puzzle15/heldout_large.tsv` row count (expect 948)
4. **Discrepancy threshold: zero.** Row counts are integer; any mismatch is reportable.

#### 1.3.2 Phase 2 — Solve-rate recomputation (T2-T5)

From each eval CSV, count rows where `solved == 1` (or whatever the CSV's success column is named — audit must determine schema first). Compute:

```
klotski_mono_solve_rate    = sum(solved) / total
klotski_opa_solve_rate     = ...
puzzle15_mono_solve_rate   = ...
puzzle15_opa_solve_rate    = ...
```

E15 §3.8 claims **64.6% / 69.9% / 0.1% / 0.7%**. E15 §3.10 derives the T5 margins **+5.3pp (Klotski)** and **+0.6pp (Puzzle15)**.

Audit procedure:
1. Recompute the four solve rates from the CSVs.
2. Recompute the two T5 margins.
3. Compare to the claimed values.
4. **Discrepancy threshold: ±0.1 pp on solve rates; ±0.1 pp on derived margins.**

#### 1.3.3 Phase 3 — Compute-equivalence math (T6)

E15 §3.9 claims:
```
compute(mono, klotski)    = 25000 × 469632 × 8 = 9.39 × 10¹⁰
compute(opa,  klotski)    = 3 × 25000 × 157696 × 8 = 9.46 × 10¹⁰
|Δ| / compute(mono)        = 0.74 %

compute(mono, puzzle15)   = 25000 × 471168 × 8 = 9.42 × 10¹⁰
compute(opa,  puzzle15)   = 3 × 25000 × 158720 × 8 = 9.52 × 10¹⁰
|Δ| / compute(mono)        = 1.06 %
```

Audit procedure:
1. Recompute the four `compute = steps × params × batch` products from the literal numbers in §3.5 + §3.6 (steps = 25000, params from the final-training tables, batch = 8).
2. Recompute the two `|Δ| / compute(mono)` ratios.
3. Compare to the claimed 0.74% and 1.06%.
4. **Discrepancy threshold: ±0.5% on the ratios.** Arithmetic is fully reproducible; any mismatch indicates a misreported param count or step count.

Additional cross-check: the agent's training logs (`results/*_train.log`) should contain final step numbers ≥ 25000 and visible loss curves. The auditor reads the tail of each log to verify the step count is what's claimed.

#### 1.3.4 Phase 4 — Held-out leakage audit re-run (T7)

E15 §3.4 claims **zero verbatim leakage and zero Jaccard ≥ 0.7 overlap** between training corpora (`data/E15-klotski/train.tsv`, `data/E15-puzzle15/train.tsv`) and held-out canonical sets (`data/E15-*/heldout_large.tsv`).

Audit procedure:
1. Re-run `tools/scaling_leakage_audit.sh` or equivalent against the four pairs (klotski train↔heldout, puzzle15 train↔heldout — both directions).
2. Compare to the zero-leakage claim.
3. **Discrepancy threshold: zero leakage. Any Jaccard ≥ 0.7 entry is reportable.**

If `tools/scaling_leakage_audit.sh` operates only on wiring-corpus shapes, the audit may need a small `tools/e15_leakage_audit.c` driver instantiated specifically for TSV-shaped state strings. That's acceptable — the auditor flags the implementation gap honestly and produces the result.

#### 1.3.5 Phase 5 — Engine-surface diff (T8)

E15 §3.13 + §3.14 claim `git diff main -- src/microgpt.{c,h} src/microgpt_vm.*` = **0 lines cumulative** across the E07-E15 arc.

Audit procedure:
1. Identify the pre-E15 baseline commit (the commit immediately before `2148ecd E15: grammar: …`).
2. Run `git diff <baseline>..main -- src/microgpt.{c,h} src/microgpt_vm.*`.
3. Compare to the zero-lines claim.
4. **Discrepancy threshold: zero lines.** Any diff is reportable.

Additional cross-check: the same diff command applied to `src/microgpt_vm_natives.{c,h}` (which exists from E08) and to the OQL grammar files (`src/microgpt_oql.{l,y,h,c}`) should show only additive changes — no removals of established surface. Auditor records.

#### 1.3.6 Phase 6 — Optional: dynamic re-run from seed (extends T1-T8)

E15 §4.8 publishes a reproducibility recipe. Phase 6 follows it:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release --parallel 8
./build/e15_generate experiments/E15-corpus.oql
./build/e15_generate experiments/E15-heldout-klotski-large.oql --audit-against build/klotski_optimal.tsv
./build/e15_generate experiments/E15-heldout-puzzle15-large.oql --audit-against build/puzzle15_optimal.tsv
# train all 8 checkpoints (mono + 3-organelle OPA per task)
# re-evaluate
```

If Phase 6 reproduces bit-identical corpora (T2's determinism claim) and the solve rates regenerate within ±2pp of §3.8's claimed values, the dynamic audit confirms the static one. If they diverge, the divergence is the finding.

**Phase 6 is OPTIONAL** because it requires 4-8 hours of CPU on a 12-core machine. If the static audit (Phases 1-5) finds no discrepancies, Phase 6 is a strong but expensive confirmation. If the static audit finds discrepancies, Phase 6 is mandatory to determine whether the discrepancy is in the artefacts or in the harness.

### 1.4 Pre-registered targets (locked)

| ID | Target | Discrepancy threshold (skip-rule trigger) |
|---|---|---|
| **T1** | Eval CSV row counts: klotski 113, puzzle15 948 (matching held-out canonical counts from `data/E15-*/heldout_large.tsv`) | Any non-zero deviation |
| **T2** | Klotski monolithic solve rate reproduces 64.6% from `results/klotski_mono_eval.csv` | > 0.1 pp |
| **T3** | Klotski OPA solve rate reproduces 69.9% from `results/klotski_opa_eval.csv` | > 0.1 pp |
| **T4** | Puzzle15 monolithic solve rate reproduces 0.1% from `results/puzzle15_mono_eval.csv` | > 0.1 pp |
| **T5** | Puzzle15 OPA solve rate reproduces 0.7% from `results/puzzle15_opa_eval.csv` | > 0.1 pp |
| **T6** | T5-margin recomputation: Klotski +5.3 pp / Puzzle15 +0.6 pp (from §3.10) | > 0.1 pp on either |
| **T7** | Compute-equivalence ratios reproduce 0.74% / 1.06% (from §3.9) | > 0.5 % on either |
| **T8** | Held-out leakage audit re-runs at zero Jaccard ≥ 0.7 entries between train and held-out for both tasks | Any non-zero leakage entry |
| **T9** | Engine-surface `git diff` cumulative across E07-E15 = 0 lines on `src/microgpt.{c,h}` + `src/microgpt_vm.*` | Any non-zero diff |
| **T10** | Reading log produced (§1.9): every cited claim names a specific file + line range that was read in full to verify it | Any claim without a citation |

Audit PASSES iff all 10 targets are satisfied within their thresholds.

### 1.5 Outcome ladder

| Outcome | Interpretation | Action |
|---|---|---|
| All 10 PASS | E15's measurements are reproducible from committed artefacts. The project's measurement discipline produces verifiable evidence. | Treat E15's verdict as canonical; downstream rewrites of `ORGANELLE_STATE.md` may proceed (under the same audit discipline applied to each cited claim). |
| 1-3 minor discrepancies (within ~2× threshold) | Minor reporting / rounding errors. Fix the experiment doc; do not invalidate the verdict. | Patch E15's Section 3 to match the audited values. Note the audit in §3.14 + §4.7. |
| ≥ 4 discrepancies OR any single discrepancy > 3× threshold | Material measurement-integrity issue. E15's published verdict is unreliable. | Audit E02-E13 under the same protocol before any architectural claim is made from those experiments. Substantial repositioning of project claims. |
| T8 (leakage) or T9 (engine surface) fails | Critical. The pre-reg discipline's hardest locks broke. | Stop all forward experiments. Audit every prior experiment's leakage + engine-surface claims. |

### 1.6 Skip rules

- **If T1-T9 fail discretely:** report each failure with the specific file, line, and discrepancy. Do NOT smooth over.
- **If T10 fails (any claim in this doc lacks a reading-log citation):** the audit doc itself violates CLAUDE.md. STOP, fix, re-pre-register.
- **If Phase 6 (optional dynamic re-run) is started but cannot complete within the 8-hour budget:** report partial results honestly; do not extrapolate.
- **Do NOT touch `ORGANELLE_STATE.md`, the project README, or any other experiment's Section 4 in response to E17's findings within E17 itself.** E17 produces a measurement; downstream rewrites are separate experiments.

### 1.7 Falsification risk: LOW (intentional)

| Risk | Likelihood | Why this is good or bad |
|---|---|---|
| All claims reproduce cleanly | High (expected) | Confirms E15's verdict + the project's measurement discipline. Worth running because the alternative (silent miscalibration) is unmeasurable otherwise. |
| Minor rounding discrepancies | Medium | Reportable; patches E15's doc; doesn't invalidate the verdict. |
| Material discrepancies on solve rates | Low-medium | Would surface a measurement issue in the eval harness or the eval CSV writing. **Most-valuable possible outcome.** |
| T8 leakage re-audit finds non-zero overlap | Low | Would surface a contamination issue not caught by the original audit. Critical. |
| T9 engine-surface diff non-zero | Low (the discipline has held across nine prior experiments) | Would break a foundational lock. Critical. |
| Phase 6 dynamic re-run produces different numbers than static audit | Medium-low | Would surface a non-determinism issue in the training pipeline. Significant finding. |

### 1.8 What this experiment is NOT testing

- It is **NOT** re-litigating E15's T5 = FAIL verdict. The verdict stands per the pre-reg discipline; E17 audits whether the numbers underpinning the verdict reproduce.
- It is **NOT** auditing E16's measurements. E16 is not merged to `main` (verified via `git log` on 2026-05-20 turn after CLAUDE.md commit). A separate E17b can audit E16's worktree branch if/when it lands.
- It is **NOT** auditing the Bonsai engine claims, the sibling-repo Markets arc, or any document I have not read in full. Per CLAUDE.md, those would require their own document reads.
- It is **NOT** producing a new architectural claim, a strategic reframe, or a recommendation about the project's headline narrative. E17 produces evidence; downstream re-framing is separate experiments.
- It is **NOT** comparing OPA to any external baseline. It only checks E15's internal consistency.

### 1.9 Cross-references + reading log (mandatory per CLAUDE.md)

**What I have read in full to support every claim in this E17 pre-reg:**

| File | Lines read | When | Supports |
|---|---|---|---|
| `experiments/E15-composition-vs-monolithic.md` | §3 (272-636) + §4 (638-861) | Current session, prior turn | All E15 numerical claims cited in §1.3 + §1.4 of this E17 doc |
| `CLAUDE.md` | Research-discipline section authored this session | Commit `14c6a7a` | §0 framing + §1.9 reading-log requirement |
| File listings: `ls results/`, `ls data/E15-{klotski,puzzle15}/`, `ls -la checkpoints/*_e15*` | Full output | Current session | Confirms the four eval CSVs + two heldout TSVs exist on main; confirms `*.ckpt` files are NOT committed (only `*.vocab`); informs Phase 6 optionality |
| `git log` head + branch state | Recent commits + branch list | Current session | Confirms E16 is on `worktree-agent-a62e7f7d8fbf04d2c` only and is excluded from E17's scope |

**What I have NOT read in full and what this E17 does not depend on:**

| Artefact | Status |
|---|---|
| The eval CSV contents themselves (only row counts; the audit reads them) | Not read by me; the audit *will* read them |
| The training logs in `results/*_train.log` | Not read by me; the audit *will* read them |
| `tools/e15_eval.c`, `tools/e15_train.c` | Not read by me; the audit *may* read them if discrepancies surface |
| `tools/scaling_leakage_audit.sh` | Not read by me; the audit *will* read it in Phase 4 |
| RESEARCH_BONSAI.md | Read in full this session — irrelevant to E17 (E17 does not audit Bonsai) |
| Sibling-repo Markets-arc docs (V1-V24) | NOT READ; explicitly out of scope per §1.8 |

**Cross-references to other experiments:**

| Topic | Source |
|---|---|
| The experiment being audited | [E15](E15-composition-vs-monolithic.md) §3 + §4 |
| The discipline that mandates this audit | `CLAUDE.md` commit `14c6a7a` — load-bearing research-discipline section |
| The non-merged sibling experiment that this audit deliberately excludes | [E16](E16-bonsai-augmented-opa.md) on main has pre-reg only; agent work is on `worktree-agent-a62e7f7d8fbf04d2c` branch |
| The corpora E17 reads from | `data/E15-klotski/` and `data/E15-puzzle15/` |
| The eval traces E17 reads from | `results/*_eval.csv` and `results/*_eval.log` |
| The provenance hashes E17 cross-checks | `data/E15-*/provenance.txt` |

---

## 2. Initial state

### 2.1 What's currently known on `main`

From the `git log` and `ls` queries run this session:

- **8 eval-pair files in `results/`**: klotski + puzzle15, mono + opa, csv + log. Plus 8 train logs (4 per task × planner/player/judge/mono).
- **2 corpus directories in `data/E15-*/`** with train + heldout + heldout_large + provenance.
- **8 vocab files in `checkpoints/*_e15.vocab`** (mono + 3-organelle × 2 tasks). **Checkpoint `.ckpt` files NOT committed** (`.gitignore`d per the prior session's data-policy decision).
- **The E15 doc at HEAD** has Section 3 measured (lines 272-636) + Section 4 conclusion (lines 638-861).
- **E16 doc at HEAD** has Section 3 still as TODO; the agent's measurements are on `worktree-agent-a62e7f7d8fbf04d2c` only.
- **CLAUDE.md at HEAD** carries the new research-discipline section as of commit `14c6a7a`.

### 2.2 Phase 6 dependency note

Phase 6 (the optional dynamic re-run) requires retraining the 8 checkpoints from seed because `*.ckpt` files are not committed. The reproducibility recipe in E15 §4.8 takes "~15 min on 12-core CPU" for the training. The auditor should budget accordingly. Phase 6 is OPTIONAL — Phases 1-5 alone produce the headline audit result.

### 2.3 Dependencies / blockers

- **Phase 4 leakage audit** may require a small new audit driver if `tools/scaling_leakage_audit.sh` is wiring-corpus-shaped. Auditor flags + implements + documents.
- **No new build deps.** No new VM opcodes (T9 audits this).
- **The CLAUDE.md discipline applies to E17 itself** — every claim in the audit's eventual Section 3 must cite a specific file + line range it was read from.

### 2.4 What this experiment deliberately does NOT do

- Does NOT modify any source code in `src/`.
- Does NOT modify any experiment doc other than `experiments/E17-e15-reproducibility-audit.md` (this file) and its eventual Section 3 + 4 writeup.
- Does NOT touch `ORGANELLE_STATE.md` or the project README (per §1.6 + the CLAUDE.md discipline).
- Does NOT merge or unmerge any worktree branches.
- Does NOT spawn any other agents (E17 is itself an agent run; recursive spawning is out of scope).

---

## 3. Implementation + results

**TODO** — fill on measurement commit. Sections to populate:

- 3.1 Eval CSV row-count audit (T1) — actual counts + comparison
- 3.2 Solve-rate recomputation (T2-T5) — actual percentages + delta to claims
- 3.3 T5 margin recomputation (T6)
- 3.4 Compute-equivalence math audit (T7) — actual ratios + delta to claims
- 3.5 Held-out leakage re-audit (T8) — actual Jaccard scan results
- 3.6 Engine-surface diff (T9) — actual `git diff` output
- 3.7 (optional) Phase 6 dynamic re-run results
- 3.8 Discrepancy summary with file + line citations per the CLAUDE.md reading-log requirement
- 3.9 Per-target verdict matrix (T1-T10)
- 3.10 Reading log: every file read in full during the audit, with line ranges

---

## 4. Conclusion

**TODO** — fill on measurement commit when ALL 10 targets are measured. Sections to populate:

- 4.1 Verdict per T1-T10 (PASS / DISCREPANCY-MINOR / DISCREPANCY-MATERIAL / CRITICAL)
- 4.2 Headline outcome — which of the §1.5 outcome corners did the audit land in?
- 4.3 If all PASS: implications for future architectural experiments (they can cite E15 with confidence).
- 4.4 If discrepancies found: implications for E02-E13's verdicts (the same protocol should be applied to those).
- 4.5 What this means for the new CLAUDE.md discipline: did it produce a cleaner experiment than the pre-discipline ones?
- 4.6 Next experiments suggested:
  - If audit PASSES cleanly: proceed with whichever experiment was queued before E17 (possibly E16's merge decision, or a new architectural follow-up).
  - If audit reveals minor discrepancies: patch E15's Section 3 only; resume.
  - If audit reveals material discrepancies: pre-register E17b (audit E02-E13) before any other experiment.
- 4.7 Traceability updates: `TRACEABILITY.md`, `RESEARCH_DISCLOSURE.md`. Explicitly NOT `ORGANELLE_STATE.md` or the project README (per §1.6).
