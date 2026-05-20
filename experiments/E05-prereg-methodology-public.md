# E05 — Open the pre-registration + leakage-audit methodology as a public artefact

**Status:** PRE-REGISTERED (Section 1, 2) — Section 3 (Implementation + Results) populated by the implementation run.
**Owner:** Research / methodology
**Pre-registration date:** 2026-05-20
**Pairs with:** `RESEARCH_OPA_DIRECTIONS.md` §8.1 (Experiment 7.1 — auto-audit), §8.2 (Experiment 7.2 — pre-reg database). E05 is the **public-artefact** version that bundles both into a single shippable bundle (parser + CI hook + paper).

---

## 1. Hypothesis (locked before measurement)

The project's most distinctive trait is not its architecture but its **discipline around it**: every multi-phase research arc names a falsifiable hypothesis + skip rule **before** measurement, every held-out test set is audited for leakage **before** any number is reported, and every cancelled phase is documented in `RESEARCH_DISCLOSURE.md` rather than silently dropped (see §3.1, §6, §7 of that file).

E05 asserts that this discipline, **as currently practised**, is reproducible by independent readers and machine-auditable:

> **H1 (locked):** The pre-registration discipline is currently practised in a sufficiently uniform format that a pure-C99 parser can extract ≥ 95 % of pre-reg target blocks from `docs/research/RESEARCH_*.md`, `docs/research/wiring_*.md`, and `experiments/E0?.md` without per-file special casing.
>
> **H2 (locked):** The leakage-audit script (`tools/scaling_leakage_audit.sh`) is sufficiently sensitive to catch the two known historical leakage incidents — the v1 1:1 scaling result (Jaccard ≥ 0.7 on 19/20 held-out, commit `5a478bc`) and the Phase 13 verbatim-leak (13/20 held-out appearing verbatim in training) — when applied retroactively.
>
> **H3 (locked):** The methodology is documentable as a self-contained paper draft (≥ 10 markdown pages, ≥ 5 worked examples drawn from this repo) such that an independent reader could adopt it without reading the underlying source.

---

## 2. Pre-registered targets

| ID | Target | Floor (falsification) | Mechanism |
|---|---|---|---|
| **T1** | ≥ 95 % of pre-reg target blocks extracted | < 80 % → standardise pre-reg format in a separate canonicalisation commit; do NOT silently widen the parser | `tools/research_status_dashboard.{c,sh}` walks the corpus, counts `Pre-registered targets:` headers vs blocks the parser actually parses |
| **T2** | ≥ 90 % of extracted blocks correctly classified into {PROPOSAL-LOCKED, PASS, FALSIFIED, PARTIAL, CANCELLED, EXCEEDED} | < 75 % → the classifier's heuristics are insufficient; document the failure modes and revise the rubric, not the labels | Hand-audited against `RESEARCH_DISCLOSURE.md` outcomes |
| **T3** | The leakage-audit hook (`.github/workflows/leakage-audit.yml`) catches the v1 incident at commit `5a478bc` when run retroactively | Hook is silent or under threshold → calibrate thresholds via binary search against the known incident (do NOT loosen `tools/leakage_audit_thresholds.json` defaults) | Run `tools/scaling_leakage_audit.sh pipeline_corpus_scaling_heldout.txt pipeline_corpus_phase4_train.txt`; expect Audit B ≥ 19/20 ≥ 0.7 |
| **T4** | The leakage-audit hook catches the Phase 13 verbatim-leak when run retroactively | Hook is silent → revise threshold logic (do NOT relax) | Run `tools/check_held_out_leakage.sh` against the pre-Phase-2d wiring corpus + held-out file; expect ≥ 13 verbatim matches |
| **T5** | Zero false positives on `main` HEAD audit runs (any held-out file currently in the repo runs clean against its calibrated thresholds in `tools/leakage_audit_thresholds.json`) | Any FP → tighten the matching FP threshold AND document the relaxation in `RESEARCH_DISCLOSURE.md` | Run the hook against every held-out file enumerated in `leakage_audit_thresholds.json` |
| **T6** | Methodology paper draft ≥ 10 markdown pages and ≥ 5 worked examples from this repo | < 10 pages or < 5 worked examples → ship as a draft note, label as such; do NOT inflate | Paper at `docs/research/RESEARCH_METHODOLOGY_PAPER.md`; worked examples sourced from `RESEARCH_DISCLOSURE.md` §3.1 (Phase 13 leak), §2.1 (Phase 3a TF-IDF cancellation), §3 (V1.0.5 Phase 5), §4 (V1.0.6 Phase 6), §8 (V1.1.0 Phase 6d) |

### 2.1 Pre-registered skip rules

- **If T1 < 80 %:** standardise the pre-reg block format in a separate canonicalisation commit; do NOT silently widen the parser to accept inconsistent formats. The parser's job is to certify uniformity, not to paper over divergence.
- **If T2 < 75 %:** document the classification failure modes; revise the rubric, not the labels (the same way `RESEARCH_DISCLOSURE.md` §6 records hypotheses that were "PARTIALLY confirmed" rather than reclassifying CANCELLED → PASS).
- **If T3 or T4 fail:** calibrate thresholds by binary search against the known incidents (`leakage_audit_thresholds.json`); do NOT loosen the default thresholds. The hook's job is to catch the known historical incidents; if it doesn't, the calibration is wrong, not the floor.
- **If T5 fails:** tighten the matching FP threshold AND document the relaxation in `RESEARCH_DISCLOSURE.md`.
- **If T6 fails on page count:** ship the draft labelled as "partial draft" with a `# TODO` index; do NOT inflate with filler.

### 2.2 What this experiment is NOT testing

- Not testing whether the methodology *itself* is sound — that's the cumulative work of `RESEARCH_DISCLOSURE.md` §3.1 (Phase 13), §3 (V1.0.5), §6 (V1.0.6), §8 (V1.1.0).
- Not testing whether the methodology generalises beyond this repo — claims about other projects would require those projects to adopt it. Out of scope.
- Not adding new pre-registration to existing arcs — the parser is read-only against the corpus.
- Not changing the leakage thresholds in `tools/leakage_audit_thresholds.json` — those are pre-registered separately as Experiment 7.1 (`RESEARCH_OPA_DIRECTIONS.md` §8.1).

---

## 3. Implementation + Results

> Section 3 is written **after** Section 1 and Section 2 are committed. Per the
> project's pre-register-then-measure pattern (`RESEARCH_PIPELINE_IR.md` §40 →
> §41 transition; `RESEARCH_OPA_DIRECTIONS.md` §12 "two-commit discipline"),
> this section is populated in a follow-up commit and the §1/§2 content above is
> never retroactively edited.

_To be populated by follow-up commits._

---

## 4. Conclusion

_To be populated only after every pre-registered target in §2 has been measured. Otherwise leave for follow-up._

— Pre-registered 2026-05-20.
