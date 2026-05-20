# Experiment E05 — Open the pre-registration + leakage-audit methodology as a public artefact

**Status:** 📋 Proposal locked — 2026-05-20.
**Direction:** elevate the project's most distinctive process (pre-reg + leakage-audit + two-commit discipline) into a tool + methodology paper with reach beyond MicroGPT-C.
**Cost estimate:** ~3 weeks (1 wk pre-reg parser + 1 wk auto-audit CI hook + 1 wk methodology paper draft).
**Falsification risk:** Low — this is engineering + writing, not science. The "falsification" risk is that the methodology doesn't get adopted externally, which is a multi-month signal not in scope here.

---

## Spear summary

**Point:** MicroGPT-C ships three pieces of methodology infrastructure rare in ML research: (1) **pre-registered hypotheses with locked targets and skip rules**, (2) **leakage audits gated by Jaccard similarity on held-out vs training corpus** (`tools/scaling_leakage_audit.sh`), (3) **the two-commit pre-reg-then-measure pattern** that prevents retroactive rationalisation. These exist in this repo but nowhere else — they should.

**Picture:** Package the three together. A methodology paper documents the pattern with worked examples (Phase 3a TF-IDF falsification, V1.0.5/V1.0.6 partial-resolutions, the Phase 13 verbatim leak incident → audit infrastructure). A pre-reg parser auto-extracts all `Pre-registered targets` blocks from `RESEARCH_*.md` and `experiments/*.md` into a status dashboard. An auto-audit CI hook fails commits that touch held-out sets without passing `scaling_leakage_audit.sh`. Together: a portable, opinionated honesty-substrate for ML research.

**Proof (to be measured):** parser extracts ≥ 95% of pre-reg blocks in existing files; auto-audit hook catches the v1 leakage incident on retroactive application to commit `5a478bc`; the methodology paper is submission-ready to a venue.

**Push:** Reviewers love methodology contributions because they generalise. The substantive research claims in this repo can be challenged on details; the methodology is unambiguous good practice.

---

## 1. Proposal

### 1.1 Hypothesis (locked before measurement)

> *The pre-registration + leakage-audit + two-commit pattern that MicroGPT-C has accumulated over the wiring-organelle arc is (a) machine-extractable from the existing markdown documentation, (b) machine-enforceable via a pre-commit / CI hook on any project that adopts it, and (c) publishable as a methodology paper with concrete worked examples from this repo.*

This experiment combines two prior pre-reg entries from [`RESEARCH_OPA_DIRECTIONS.md`](../docs/research/RESEARCH_OPA_DIRECTIONS.md):

- **§7.1** (Auto-audit on every commit, ~1 wk)
- **§7.2** (Pre-registration database, ~2 wk)

…plus a new third component: a **standalone methodology paper** that frames both as reusable artefacts.

### 1.2 Why this matters

Pre-registration in ML papers is borderline non-existent. Reviewer-2 "but did you peek at the test set?" goes unanswered on most papers because the methodology isn't in place to answer it. The repo has built the methodology *by accident* (the Phase 13 verbatim leak forced it) and now has standing audit infrastructure that 99% of ML research lacks.

Three distinct audiences benefit:

| Audience | What they get |
|---|---|
| **Methodology-curious ML researchers** | A worked methodology paper they can cite and adapt |
| **Their reviewers** | A reference for "how should this paper have measured itself?" |
| **Engineering teams shipping ML to regulated domains** | A drop-in CI tool for "we did not train on the test set" assurance |

### 1.3 Mechanism

**Phase 1 — Pre-reg parser (1 week).** Build `tools/research_status_dashboard.{c,sh}` per `RESEARCH_OPA_DIRECTIONS.md` §7.2.

Input: every `RESEARCH_*.md`, `wiring_*.md`, and `experiments/*.md` file.
Output: a structured table:

```
| File | Pre-reg ID | Hypothesis (one line) | Targets locked | Status |
|---|---|---|---|---|
| RESEARCH_PIPELINE_IR.md | Phase 3a | TF-IDF on adversarial axis-2 ≥ 12-16/20 | 4 | FALSIFIED (§41) |
| RESEARCH_PIPELINE_IR.md | Phase 3b | Composition ≥ 5-7/10 on multi-stage | 3 | PASS (§43) |
| RESEARCH_PIPELINE_IR.md | Phase 4 | TF-IDF expanded corpus ≥ 8-12/20 | 2 | EXCEEDED (§46) |
| RESEARCH_OPENMYTHOS_CROSS_POLLINATION.md | A | ACT halting latency ≥ 5x | 4 | PARTIAL — API shipped V1.2.0, demo deferred |
| experiments/E01-llm-head-to-head.md | E01 | Headline 4-tuple inequality | 8 | PROPOSAL-LOCKED |
...
```

Parser:
- Pure C99 + small `bash` glue (zero deps, matches project policy).
- Regex/state-machine extraction of `Pre-registered targets (locked)` blocks and `Pre-registered skip rule` blocks.
- Cross-reference resolution: when one pre-reg references another, link them.
- Output formats: markdown (for `STATUS_DASHBOARD.md`), JSON (for tooling), HTML (for static-site rendering).

**Phase 2 — Auto-audit CI hook (1 week).** Make `tools/scaling_leakage_audit.sh` a first-class CI gate.

```yaml
# .github/workflows/leakage-audit.yml
on:
  pull_request:
    paths:
      - '**/wiring_*.{c,h,txt}'
      - '**/corpus*.{c,h,txt}'
      - '**/pipeline_corpus_held_out*.txt'
      - 'src/wiring_anchor_graphs.{c,h}'
jobs:
  audit:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: tools/scaling_leakage_audit.sh
      - run: |
          if [ $? -ne 0 ]; then
            echo "::error::Leakage audit failed. See tools/leakage_audit_thresholds.json."
            exit 1
          fi
```

**Validation:** retroactively apply the hook to commit `5a478bc` (the post-Phase-3 cleanup that introduced the v1 leakage incident). The hook must fail the commit. If it doesn't, the audit thresholds are too loose and need calibrating.

Also retroactively test on the Phase 13 verbatim-leak commit — different leakage mode (verbatim, not Jaccard), so the hook needs an Audit-A (exact match) path as well as Audit-B (Jaccard).

**Phase 3 — Methodology paper draft (1 week).** Single ~12-page draft, target venue: NeurIPS / ICML methodology track, or a CS-research-methods venue like *Patterns* or *Quantitative Science Studies*.

Section sketch:

1. **Introduction.** ML's reproducibility crisis is well-documented. Pre-registration in psychology / clinical trials is mature; in ML it is rare. The MicroGPT-C project has accidentally accumulated the pieces; this paper makes them portable.
2. **The two-commit pattern.** `feat(research): pre-registered Experiment X` then `research(area): Experiment X measurement vs pre-reg targets`. Concrete examples from `git log`: the Phase 3a TF-IDF case (pre-reg + falsification both committed honestly).
3. **The locked-target structure.** Hypothesis + targets + floors + skip rules. Why each matters. Worked examples of skip-rules saving experiments from confirmation bias.
4. **The leakage audit.** Audit-A (verbatim) and Audit-B (Jaccard ≥ 0.7). Worked examples of two leakage incidents in this repo and how the audit caught them retroactively.
5. **The pre-reg parser + dashboard.** Tooling for the methodology to scale beyond individual papers.
6. **Limitations and counterexamples.** When pre-reg is not the right move (genuinely exploratory work). The discipline-tax tradeoff. Honesty about the project's *own* falsifications (V1.0.5, V1.0.6, V1.1.0 partial-resolution) as evidence the methodology works.
7. **Adoption recipe.** A 1-page checklist for "add this to your project."

### 1.4 Pre-registered targets (locked)

| ID | Target | Floor (skip-rule trigger) |
|---|---|---|
| **T1** | Parser extracts ≥ 95% of pre-reg blocks in existing `RESEARCH_*.md`, `wiring_*.md`, `experiments/*.md` | < 80% (= parser bug) |
| **T2** | Parser correctly classifies status (PRE-REGISTERED / PASS / FALSIFIED / PARTIAL / CANCELLED) for ≥ 90% of extracted pre-regs | < 75% |
| **T3** | Auto-audit hook catches v1 leakage incident on retroactive application to commit `5a478bc` | Hook passes the bad commit |
| **T4** | Auto-audit hook catches Phase 13 verbatim leak on retroactive application | Hook passes |
| **T5** | Zero false positives on main branch over the last 50 commits | ≥ 1 false positive (= thresholds need calibration) |
| **T6** | Methodology paper draft ≥ 10 pages, with ≥ 5 worked examples from this repo, ≥ 1 from another project (or explicitly noted as future work) | < 8 pages or < 3 worked examples |

### 1.5 Skip rules

- If T1 < 80% (parser misses too many blocks): standardise the pre-reg block format across repo (introduce a YAML front-matter section per pre-reg) and re-run extraction.
- If T3 or T4 fails (hook doesn't catch known incidents): the thresholds are too loose. Calibrate by binary search against the v1 incident's actual Jaccard scores; ship the recalibrated thresholds.
- If T5 ≥ 1 false positive: investigate each; either fix the audit script or document the false-positive class as a known limitation.

### 1.6 Falsification risk: Low

| Risk | Likelihood | Mitigation |
|---|---|---|
| Pre-reg blocks are inconsistently formatted across files (T1 low) | Medium | Tolerant parser + one-shot canonicalisation pass on existing files |
| Audit thresholds are too loose to catch the historical incident | Low | Thresholds were derived from that incident; should catch it by construction |
| Methodology paper rejected at first-choice venue | High (any submission risk) | Multiple venues; the artefact (parser + hook) is the value either way |
| External adoption is zero | Medium (long-tail signal) | Out of scope — this experiment measures the artefact, not adoption |

### 1.7 What this experiment is NOT testing

- It is **not** measuring whether other projects adopt the methodology. That is a 12-month signal.
- It is **not** a venue-acceptance experiment. The paper draft is the deliverable; whether it gets in is a separate process.
- It is **not** about the *content* of MicroGPT-C's research claims — it's about the *process* by which those claims were made and corrected.
- It is **not** a replacement for `RESEARCH_DISCLOSURE.md`. That document holds the final disclosure of outcomes; this experiment holds the *tools* by which those outcomes get tracked.

### 1.8 Cross-references

| Topic | Source |
|---|---|
| Pre-reg origin §7.1 | [`RESEARCH_OPA_DIRECTIONS.md`](../docs/research/RESEARCH_OPA_DIRECTIONS.md) §7.1 — auto-audit on commits |
| Pre-reg origin §7.2 | [`RESEARCH_OPA_DIRECTIONS.md`](../docs/research/RESEARCH_OPA_DIRECTIONS.md) §7.2 — pre-registration database |
| Worked example: Phase 3a falsification | [`RESEARCH_PIPELINE_IR.md`](../docs/research/RESEARCH_PIPELINE_IR.md) §40 (pre-reg) + §41 (result) |
| Worked example: Phase 13 verbatim leak | [`RESEARCH_PIPELINE_IR.md`](../docs/research/RESEARCH_PIPELINE_IR.md) §38 (incident report) |
| Worked example: V1.0.5 falsification | [`ORGANELLE_STATE.md`](../docs/research/ORGANELLE_STATE.md) Phase 5 row |
| Existing audit | [`tools/scaling_leakage_audit.sh`](../tools/scaling_leakage_audit.sh) |
| Existing disclosure register | [`RESEARCH_DISCLOSURE.md`](../docs/engineering/CLEAN_ROOM_IMPLEMENTATION/RESEARCH_DISCLOSURE.md) |

---

## 2. Initial state

### 2.1 What's currently known

- ~30 `RESEARCH_*.md` files in `docs/research/` containing pre-reg blocks of varying formats.
- 16 enumerated experiments in `RESEARCH_OPA_DIRECTIONS.md`, of which 2 graduated to V1.2.0 implementation.
- Standing audit script `tools/scaling_leakage_audit.sh` with Audit-A (verbatim) + Audit-B (Jaccard ≥ 0.7) modes.
- `RESEARCH_DISCLOSURE.md` has 9+ logged outcomes (PASS / PARTIAL / FALSIFIED variants).
- No external project has adopted the pattern (zero baseline).

### 2.2 Dependencies / blockers

- Pre-reg block format is *similar* but not strictly *uniform* across files. Phase 1 may require a canonicalisation PR to normalise.
- Methodology paper depends on having at least 5 worked examples; the project has ≥ 8, comfortably above target.
- Venue choice / submission cycle is out of band; the artefact is what this experiment measures.

### 2.3 Baselines

| Baseline | State |
|---|---|
| Pre-reg parser | Does not exist; build it |
| Auto-audit CI hook | Audit script exists but not CI-integrated; integrate it |
| Methodology paper | Does not exist; draft it |
| External adopters | 0 (not in scope for this experiment) |

---

## 3. Implementation + results

**TODO** — fill on measurement commit. Sections to populate:

- 3.1 Parser commit + extraction stats (T1, T2)
- 3.2 CI hook commit + retroactive test results (T3, T4, T5)
- 3.3 Methodology paper draft (link to PDF or markdown source)
- 3.4 Status dashboard artefact (rendered `STATUS_DASHBOARD.md`)
- 3.5 Reproduction instructions

---

## 4. Conclusion

**TODO** — fill on measurement commit. Sections to populate:

- 4.1 Verdict per T1-T6 (PASS / FAIL)
- 4.2 Lessons (especially: which pre-reg blocks were hardest to parse? what does that say about format design?)
- 4.3 Format-canonicalisation PR — needed or not?
- 4.4 Next moves: submit paper; cross-post the parser + hook as a standalone Hacker News / r/MachineLearning announcement; consider standardising as a small open-source project (`mlpre-reg/`)
- 4.5 Traceability updates: `TRACEABILITY.md`, `ORGANELLE_STATE.md`
