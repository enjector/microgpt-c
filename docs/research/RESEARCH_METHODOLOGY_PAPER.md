# A pre-registration + leakage-audit methodology for small-model ML research

**Status:** Draft. Pre-registered as Experiment E05 §3.3 in
`experiments/E05-prereg-methodology-public.md` and as Experiment 7.2 in
`docs/research/RESEARCH_OPA_DIRECTIONS.md` §8.2.

**Authors:** Ajay Soni (project lead), with the audit + pre-reg discipline
contributed iteratively across the project's research log (see references).

**Date:** 2026-05-20

---

## 1. Preamble + scope

This paper documents a methodology that the `microgpt-c` project has used,
iteratively refined, and tested over a 10-month research arc. The
methodology has three pillars:

1. **Pre-registration** of every multi-phase research hypothesis with
   falsifiable targets, floors, and skip rules — committed to a public
   `RESEARCH_*.md` document **before** any measurement runs.
2. **Leakage auditing** of every held-out test set against the
   training corpus — three deterministic awk-and-bash checks (verbatim,
   bag-of-words Jaccard, lexical anchoring) producing a numeric verdict
   before any retrieval or classification number is reported.
3. **Honest disclosure** of every pre-registered phase that was
   subsequently cancelled, falsified, or partially-resolved — collected
   in a single regulator-friendly register
   (`docs/engineering/CLEAN_ROOM_IMPLEMENTATION/RESEARCH_DISCLOSURE.md`)
   rather than scattered across the development log.

The methodology was developed *because* the project's claims survived a
real measurement crisis (the "Phase 13 verbatim leak" — see Worked
example 2 below), and the discipline that emerged from that crisis is
what kept subsequent claims honest. The artefact set that supports the
discipline is small enough to lift into another project in a day:

- **Pre-reg parser**: ~700 LOC C99, walks the corpus, emits a status
  dashboard. Zero deps.
- **Leakage-audit script**: ~200 LOC bash + awk. Zero deps.
- **CI hook**: a 60-line GitHub Actions workflow that fires on every PR
  touching the corpus or audit infrastructure.
- **Threshold ratchet**: a JSON file with per-file calibrated bounds.
  Loosening requires a `RESEARCH_DISCLOSURE.md` entry; tightening is
  always free.

The scope of this paper is the methodology, not the architecture it
governs. The architecture (a "tiny specialists + deterministic Judge"
organelle pipeline; see `docs/research/ORGANELLE_STATE.md`) is described
elsewhere. The methodology applies to any research project where:

- Tasks are evaluated by accuracy / coverage / latency on a held-out
  test set.
- The held-out set is constructed (in whole or in part) by the same team
  that wrote the training corpus.
- Reproducibility, audit-defensibility, or honest claim-making is a
  goal.

It does NOT apply to:

- Pure inference benchmarks against an externally-curated test set
  (where leakage is bounded by the curator's diligence, not the team's).
- Single-shot research outputs with no iteration (the discipline has
  zero marginal value if there's only one phase).
- Projects where the team has unlimited budget to retest with fresh
  data per iteration (where leakage is mitigated structurally rather
  than via auditing).

## 2. The problem the methodology solves

In early 2026, this project published a result — "1:1 scaling of the
TF-IDF retrieval mechanism: 20/20 on a fresh held-out family library".
The result was internally consistent: the audit script we had at the
time (verbatim matches only) reported 0 leaks; the model genuinely
returned 20/20.

A second-pass audit by a different mechanism — bag-of-words Jaccard
overlap against the training corpus — surfaced a different reality.
**19 of the 20 held-out prompts shared Jaccard ≥ 0.7 with at least one
training prompt; 2 of those 20 were word-shuffled clones at Jaccard =
1.0.** The verbatim check had passed because no string appeared
character-identical; the model had memorised the near-duplicate.

The honest correction, on a vocabulary-disjoint v2 held-out, was
**15/20 (75 %)** — a 25-point drop. The pre-registered threshold of ≥
18/20 was falsified.

This was the V1.0 inflection point: the methodology was forced to
treat leakage as a first-class measurement-discipline problem, not an
afterthought. The result was the three pillars above. The Phase-13
verbatim leak (a separate incident at the wiring transformer's training
corpus — see Worked example 2) had a similar shape but was caught
earlier in the lifecycle and was less load-bearing on a public claim.
Together the two incidents drove the methodology to its current form.

## 3. The three pillars

### 3.1 Pre-registration

Every multi-phase research arc names a falsifiable hypothesis, a
pre-registered target with a floor, and a skip rule — **before**
measurement runs. The format is uniform enough that a ~700-LOC C99
parser extracts and classifies the corpus. The canonical block looks
like this (from `RESEARCH_OPA_DIRECTIONS.md` §8.1):

```markdown
### 8.1 Experiment 7.1 — Auto-audit on every commit

**Hypothesis (locked):** A pre-commit (or CI) hook that runs
`tools/scaling_leakage_audit.sh` on any PR touching a held-out test
file or a synonym table catches future leakage incidents before the
commit lands. Mechanism: the hook fails the commit if Audit B (Jaccard
≥ 0.7) crosses the per-file threshold defined in a new
`tools/leakage_audit_thresholds.json`.

**Pre-registered targets:** the hook catches the v1 leakage incident
if applied retroactively to commit `5a478bc` (the post-Phase-3 cleanup
pass). Zero false positives on the current main branch's audit runs.

**Cost:** ~1 week. CI integration via
`.github/workflows/leakage-audit.yml`.
```

The discipline is enforced by two-commit convention:

1. The pre-registration is committed (`feat(research): pre-registered
   Experiment 7.1 — ...`) with the hypothesis, targets, floor, and
   skip rule named but NO measurement.
2. The measurement is committed separately (`research(area): Experiment
   7.1 measurement vs §X pre-reg targets`) with the outcome reported
   verbatim against the pre-registered prediction.

If the pre-registration commit gets retroactively edited to match the
outcome, git history shows it — and the methodology's audit trail is
broken at that point. The discipline is socially enforced; the
infrastructure makes the breach visible.

Worked example 1 below shows what happens when the hypothesis is
falsified: the skip rule fires, the phase is cancelled, and the
disclosure record is updated.

### 3.2 Leakage audit

Three deterministic checks run against every held-out test set before
any retrieval/classification number is reported:

- **Audit A — verbatim**: does the held-out prompt appear character-
  identically in the training corpus? Catches the easy class of leaks.
- **Audit B — Jaccard near-duplicate**: bag-of-words Jaccard ≥ 0.7
  against any training prompt. Catches word-shuffled clones that
  Audit A misses (this is the check that surfaced the v1 incident).
- **Audit C — lexical anchoring**: for each held-out prompt, what
  fraction of its content words appear in **only** its own family's
  training? High anchoring inflates the apparent retrieval rate even
  without literal duplication.

The script is `tools/scaling_leakage_audit.sh` — ~200 LOC of bash +
awk, zero deps. It runs at build time (via the wiring organelle's
POST_BUILD step) and at CI time (via
`.github/workflows/leakage-audit.yml`). Thresholds are per-file and
live in `tools/leakage_audit_thresholds.json`. Loosening a threshold
requires a `RESEARCH_DISCLOSURE.md` note citing the reality (the
threshold is meant to track HEAD; relaxations are honest calibration,
not weakening of the contract).

### 3.3 Honest disclosure

Every pre-registered phase that is subsequently cancelled, falsified,
or partially-resolved is recorded in
`docs/engineering/CLEAN_ROOM_IMPLEMENTATION/RESEARCH_DISCLOSURE.md`
under a §-numbered cancellation entry. The register is the single
read-once document a regulator, customer security team, or independent
reviewer can use to know which claims have been retracted, which were
never made, and which were validated under stronger conditions than
originally promised.

The discipline is that an outcome cannot be elided. Cancellation is
not failure — it is the methodology working. Falsification is the
*intended* output of a pre-registered hypothesis, not an aberration to
hide. Partial resolution is published verbatim against the pre-reg
target, not retroactively narrowed.

This pillar's value compounds with the other two: the parser (3.1)
provides a dashboard view of every pre-reg, the audit (3.2) keeps
held-out tests trustworthy, and the disclosure (3.3) maintains the
documented audit trail.

## 4. Worked example 1 — Phase 3a TF-IDF cancellation

**Source:** `docs/engineering/CLEAN_ROOM_IMPLEMENTATION/RESEARCH_DISCLOSURE.md` §2.1,
`docs/research/RESEARCH_PIPELINE_IR.md` §40 (pre-reg) and §41 (outcome).

**Pre-registration (RESEARCH_PIPELINE_IR.md §40.2, written before measurement):**
A learned EKAN-Network classifier should outperform the handcoded
keyword-bag baseline on the adversarial axis-2 stress test. The
pre-registered target was 12-16/20 (vs handcoded 2-5/20). A pre-
registered skip rule was named:

> *If the simplest learned encoder underperforms the handcoded baseline
> by more than 4 points on adversarial axis-2, the more complex encoder
> (EKAN-Network) shall be cancelled.*

**Decision before measurement:** Run Phase 3a-lite (TF-IDF centroid
classifier, ~150 LOC, ~30 min) first — the simplest possible learned
encoder. If it shows benefit, escalate to Phase 3a-full (EKAN-Network).
If it misses by ≥ 4 points on the 12-16/20 prediction, cancel Phase
3a-full.

**Outcome (RESEARCH_PIPELINE_IR.md §41.2):** TF-IDF centroid scored
**4/20 on the adversarial axis-2 set** — 8 points below the lower
bound of the prediction interval. The §40.7 skip rule fired:

> Phase 3a-lite scored 4/20. The skip condition fires. Phase 3a-full
> (EKAN-Network classifier) is **cancelled** at this scale.

**Cancellation transparency (RESEARCH_DISCLOSURE.md §2.1):**

> **Decision:** **CANCELLED** per the skip rule. The 4/20 result was 8
> points below the lower bound of the prediction interval.
>
> **Implication:** No learned encoder beats the handcoded keyword bag
> *at the 408-example corpus scale*. Phase 4 (corpus expansion)
> reopened the question and answered it positively under more data.

**Why this is load-bearing for the methodology:** Without the pre-
registered skip rule, the team would have spent another 1-2 days
training the EKAN-Network classifier, getting a similarly poor result,
and *interpreting* the poor result as "EKAN-Network is bad" rather than
"the corpus is too small for any learned encoder". The pre-registered
skip rule pre-committed the right interpretation; the post-hoc analysis
matches the pre-hoc framing exactly.

**Parser output:** The pre-reg parser (`tools/research_status_dashboard.c`)
classifies this block as **CANCELLED** (block #22 in `STATUS_DASHBOARD.md`),
sourced from `RESEARCH_DISCLOSURE.md:18`.

## 5. Worked example 2 — Phase 13 verbatim leak

**Source:** `docs/research/RESEARCH_PIPELINE_IR.md` §38 (audit),
`RESEARCH_DISCLOSURE.md` §3.1 (restatement).

**The original claim:** 75 % median / 80 % peak on the wiring layer
over a 20-prompt held-out set, after a 17-phase corpus-engineering
arc (Phases 8 → 9 → 10 → 11 → 12 → 13 → 15).

**The audit (Phase 2d, §38):** 13 of the 20 original held-out prompts
appeared **verbatim** in the wiring training corpus. The leakage was
introduced by Phase 13's lexical-anchoring expansion — paraphrases that
Phase 13 had *explicitly added to training* to lift the model's
familiarity with the held-out vocabulary.

**Restated honest headlines (RESEARCH_DISCLOSURE.md §3.1):**

> - Anchor-retrieval mechanism on the leakage-free Phase 2c paraphrases:
>   **100 % (20/20)**.
> - Wiring transformer alone on the same clean set: **35 % (7/20)**.
> - The 35 → 75 % lift attributed to the 17-phase corpus engineering
>   was largely the model memorising prompts that Phase 13 had
>   explicitly added to training.

**The standing protection that emerged:** `tools/check_held_out_leakage.sh`
(verbatim guard) and the broader `tools/scaling_leakage_audit.sh`
(verbatim + Jaccard + lexical anchoring). The hook now fires at build
time and at CI time. Retroactive validation (E05 T4) reports 30
verbatim matches against the current wiring training corpus — even
stronger than the original §38 audit's 13 (because the check covers
train + val + planner, not just train).

**Why this is load-bearing for the methodology:** Phase 13's intent was
benign — *give the model more vocabulary coverage* — but it accidentally
trained-on-test. Without a leakage audit, the headline number was
indistinguishable from real generalisation. The audit script's bag-of-
words check is what made the leak visible; pre-registration alone
would not have caught it.

**Parser output:** The §3.1 restatement is captured as a non-pre-reg
section (no marker) and therefore not in the dashboard, but the
audit script that catches it (`tools/scaling_leakage_audit.sh`) and
the CI hook (`tools/run_leakage_audit_ci.sh`) ARE pre-registered
artefacts (E05 §3.2 / Exp 7.1) tracked in the dashboard.

## 6. Worked example 3 — V1.0.5 Phase 5 first falsification

**Source:** `RESEARCH_DISCLOSURE.md` §3 (Phase 5 compositional generator
pre-reg + outcome).

**Pre-registration (§3.2, written before measurement):**

> | Mode | Target | Definition |
> | Default ranking | ≥ 50 % (15/30) verified-and-correct on the
>   leakage-audited compositional set. | "Verified" = `pipeline_verify`
>   returned `PIPE_OK`. "Correct" = the `pipeline_execute_vm` numeric
>   output matches the reference for at least 3 of the 5 input sets. |
> | No-regression | 100 % (20/20) on Phase 2c clean. | Honest baseline.|
>
> **Skip rule:** If the achieved score on Axis 1 + Axis 2 falls below
> 5/20 (25 %) under the default ranker, Stream C is
> `PARTIALLY-RESOLVED` rather than `RESOLVED`; the actual achieved
> score is recorded as the new SLO ...

**Outcome (§3.5):**

> Verified rate: 100 %. **Correctness rate: 30 %** — below the §3.2
> pre-registered target of 50 %.

**Disposition (§3.3 skip rule fired):**

> Axis 1 + Axis 2 combined = 6/20 = 30 %, **above** the 5/20 (25 %) floor.
> Therefore: **`GAP-WIRE-005` is PARTIALLY-RESOLVED, not RESOLVED.**

**Honest analysis (§3.5):** 21 of the 30 verified graphs produced
numerically-wrong answers; the failures clustered into "wrong primitive
ordering" (≈ 10), "synonym mismatch" (≈ 5), "type-mismatch or arity-
mismatch in the input mapping" (≈ 6). The pre-registered baseline of
30 % became the V1.0.5 SLO; the design goal of 50 % stayed open as a
gap to track for Phase 6.

**Why this is load-bearing for the methodology:** The original claim
of "50 % composition coverage" would have been a respectable headline.
The methodology forced the falsification *and* the honest restatement
*and* the gap registration in one atomic disclosure. The next phase
(Phase 6) inherited the new baseline as its starting point, not the
inflated target.

**Parser output:** Block #25 in `STATUS_DASHBOARD.md`, classified as
**PARTIAL** (sourced from `RESEARCH_DISCLOSURE.md:86`).

## 7. Worked example 4 — V1.0.6 Phase 6 hypothesis falsified

**Source:** `RESEARCH_DISCLOSURE.md` §4 (Phase 6 simple-search
falsification).

**Pre-registration (§4.1, three hypotheses in falsifiable form):**

> - **H1 — beam widening.** Replacing the greedy beam=1 outer-pick with
>   a beam-2 search lifts axis-1 wrong-outer-ordering failures.
>   Predicted axis-1 lift: ≥ +2 prompts.
> - **H2 — drop name-dedup pass.** ... Predicted axis-1 + axis-2 lift:
>   ≥ +2 prompts.
> - **H3 — geo-classifier tie-break.** ... Predicted axis-2 lift: ≥ +1
>   prompt.

Aggregate target: ≥ 15/30 (50 %). Failure target: < 12/30 (40 %).

**Outcome (§4.5):**

> | Axis | V1.0.5 | V1.0.6 | Δ |
> | 1 (novel pair)        | 2/10 | 3/10 | +1 |
> | 2 (synonym stress)    | 4/10 | 4/10 |  0 |
> | 3 (outer transform)   | 3/10 | 2/10 | -1 |
> | **Total correct**     | **9/30** | **9/30** | **0** |
>
> **Aggregate target was ≥ 50 % (15/30); achieved 30 % (9/30). Failure
> target was < 40 % (12/30); achieved 30 %.** The simple-search
> hypothesis is **falsified** under the §4.3 disposition logic.

**Honest analysis (§4.5):**

> Per §4.3:
> - 9/30 = 30 % is **below** the 40 % failure threshold.
> - `GAP-WIRE-006` remains **OPEN**. The V1.0.5 30 % baseline persists.
> - Phase 6b (learned ranker / beam widening to ≥ 4 / external semantic
>   embeddings) is **deferred indefinitely**, opened only on customer
>   signal.

**Why this is load-bearing for the methodology:** All three hypotheses
were falsified or no-op. The honest restatement is "the simple-search
hypothesis is falsified at this corpus scale"; the next-phase plan
(Phase 6b) is gated on a customer signal rather than continued
speculative tuning. The disclosure transparently says the changes
"don't regress correctness in the aggregate, they expose a structural
failure mode" — keeping the negative result as informative, not
embarrassing.

**Parser output:** Block #26 in `STATUS_DASHBOARD.md`, classified as
**PARTIAL** (the gap stayed open, but the V1.0.5 baseline persisted).

## 8. Worked example 5 — V1.1.0 Phase 6d partially-resolved

**Source:** `RESEARCH_DISCLOSURE.md` §8 (Phase 6d pre-reg + outcome).

**Pre-registration (§8.2, two hypotheses):**

> - **H8 — Per-port noun-aware inner picker.** Predicted lift on
>   Pattern A (12 prompts): +6/12 → axis-1 +3, axis-2 +1, axis-3 +2
>   (aggregate +6).
> - **H9 — Depth-2 inner recursion.** Predicted lift on Pattern B (3
>   prompts): +2/3.

Aggregate target: ≥ 21/30 (70 %). Failure floor: < 18/30 (60 %).

**Outcome (§8.7):**

> H9 (depth-2 inner recursion) was **not implemented** in V1.1.0.
> [...]
> | Axis | V1.0.9 | V1.1.0 | Δ |
> | Axis 1 (novel pair)     | 6/10 | 7/10 | +1 |
> | Axis 2 (synonym stress) | 7/10 | 7/10 |  0 |
> | Axis 3 (outer transform)| 3/10 | 5/10 | +2 |
> | **Total**               | **16/30 (53 %)** | **19/30 (63 %)** | **+3 (+10pp)** |
>
> **Disposition (per §8.4):** 63 % is in the **60–69 % PARTIALLY-
> RESOLVED band**. H8 partially confirmed. `GAP-WIRE-005`,
> `GAP-WIRE-006`, `GAP-WIRE-009` all remain PARTIALLY-RESOLVED. New SLO
> baseline: 63 %. Phase 6e (H9 + binder positional scoping) not opened.

**Honest analysis of the 11 residual failures (§8.7):**

The disclosure includes a per-prompt table of all 11 wrong answers,
categorising them into "dedup over-engaged", "wrong replacement after
dedup", "binder issues" (correct primitive set, wrong wiring),
"symmetric-keep mis-fired", and "needs depth-3 (H9 territory)". This
categorisation is exactly the kind of granular post-hoc analysis that
the methodology *encourages*: the failure modes are informative; the
next phase's plan is grounded in them.

**Why this is load-bearing for the methodology:** Even at PARTIALLY-
RESOLVED, the methodology produces a publishable artefact:
- The pre-registration interval (≥ 70 % / < 60 %) put a number on the
  PARTIAL band before measurement.
- The disposition logic (§8.4) maps the achieved 63 % to PARTIAL
  unambiguously.
- The 11 wrong-answer breakdown gives the next phase concrete fixes
  to attempt (or, equivalently, customer-signal triggers).

**Parser output:** Block #29 in `STATUS_DASHBOARD.md`, classified as
**PARTIAL** (sourced from `RESEARCH_DISCLOSURE.md:449`).

The corresponding `RESEARCH_PIPELINE_IR.md` §47 is block #21, also
classified PARTIAL.

## 9. The tooling

### 9.1 Pre-reg parser (`tools/research_status_dashboard.{c,sh}`)

- **Language:** C99 + minimal bash wrapper. Zero deps beyond `libc`.
- **Size:** ~700 LOC C99, ~70 LOC bash.
- **Inputs:** All `docs/research/RESEARCH_*.md`,
  `docs/research/wiring_*.md`, `ORGANELLE_STATE.md`,
  `RESEARCH_DISCLOSURE.md`, and `experiments/E0?-*.md`.
- **Outputs:** `STATUS_DASHBOARD.md` (markdown table) and
  `STATUS_DASHBOARD.json` (machine-readable sidecar).
- **Classification:** PROPOSAL-LOCKED / PASS / FALSIFIED / PARTIAL /
  CANCELLED / EXCEEDED.
- **Block-detection model:** Markdown section (## or ###) whose body
  contains at least one canonical pre-reg marker. Markers are listed
  in the parser's header comment.
- **Outcome detection:** A two-pass rollup associates sibling
  Outcome / Disposition / Hypothesis-review sections with their
  parent pre-reg block before classification.
- **Legacy-format handling:** Non-canonical pre-reg blocks (e.g. the
  `H_main / H_alt` heading in `wiring_scaling_curve.md`) are surfaced
  as PARSER-MISS records — not silently absorbed.

Measurement (E05 §3.1): 31 + 1 PARSER-MISS = 32 blocks extracted from
12 files; 30/31 correctly classified (hand-audit). T1 = 96.8 %, T2 =
96.8 %.

### 9.2 Leakage-audit script (`tools/scaling_leakage_audit.sh`)

- **Language:** Bash + awk. Zero deps.
- **Size:** ~200 LOC.
- **Audits:**
  - **A** verbatim — strict string match against the training corpus.
  - **B** Jaccard ≥ 0.7 near-duplicate — bag-of-words overlap.
  - **C** lexical anchoring ≥ 50 % — fraction of content words
    appearing in ONLY the held-out's own family's training data.
- **Inputs:** held-out test file + training corpus file (both in
  `build/`).
- **Outputs:** stdout report with per-prompt verdict + summary counts.
- **Standing protection:** runs at build time (via the wiring
  organelle's POST_BUILD step) and at CI time (via the GitHub Actions
  hook).

### 9.3 CI hook (`.github/workflows/leakage-audit.yml`)

- **Triggers:** PRs touching any wiring corpus, synonym table, audit
  script, threshold JSON, or CI workflow itself.
- **Runner:** `ubuntu-latest`.
- **Steps:** Checkout → build cmake → generate Phase-4 corpus → run
  `tools/run_leakage_audit_ci.sh` → upload log on failure.
- **Failure contract:** non-zero exit fails the PR.

### 9.4 Threshold ratchet (`tools/leakage_audit_thresholds.json`)

- **Form:** per-file `max_jaccard_07_count` / `max_jaccard_10_count` /
  `max_lexical_anchors_50pct_count` thresholds + a `default` entry.
- **Discipline:** loosening (raising a threshold) requires a
  `RESEARCH_DISCLOSURE.md` entry citing the relaxation. Tightening
  (lowering) is always free.
- **Defaults:** `max_jaccard_07_count: 0`, `max_jaccard_10_count: 0`,
  `max_lexical_anchors_50pct_count: 2`. Most held-out files are at or
  near these defaults; a few have grandfathered higher values
  documented as historical artefacts.

### 9.5 Honest-disclosure register (`RESEARCH_DISCLOSURE.md`)

The single document a regulator can read in 5 minutes to know which
claims have been retracted, which were never made, and which were
validated under stronger conditions than originally promised.
Append-only by social convention; never edited retroactively.

## 10. Cost (and what we couldn't afford)

### Cost

| Artefact | LOC / effort |
|---|---|
| Pre-reg parser | ~700 LOC C99, ~1 day to write, ~½ day to tune |
| Leakage audit | ~200 LOC bash+awk, ~½ day per audit |
| CI hook | ~60 LOC YAML, ~½ day to integrate |
| Threshold JSON | ~60 lines, ~½ day to calibrate |
| Methodology paper (this file) | ~600 lines markdown, ~1 day to draft |

Total: ~3.5 person-days for the tooling, plus the cumulative cost of
writing every pre-reg before measurement (estimated at ~10-15 % of
research effort).

### What we couldn't afford

- **Continuous pre-reg enforcement.** A robot that *prevents* a commit
  unless it includes a pre-reg block first. The discipline is
  socially enforced; git history shows breaches but doesn't prevent
  them. The dashboard helps spotlight gaps.
- **Independent-curator reproducibility.** Every measurement so far
  has had one author writing both the synonym tables and the
  held-out paraphrases. A second-curator rebuild would test whether
  the calibrated 75–80 % retrieval ceiling is curator-specific. This
  is pre-registered as Experiment 2.3 in `RESEARCH_OPA_DIRECTIONS.md`
  §3.3; not yet executed.
- **Adversarial threshold validation.** The thresholds JSON's
  ratchet discipline depends on honest curator behaviour. A
  deliberate attacker could quietly raise a threshold under
  plausible-looking commentary; the `RESEARCH_DISCLOSURE.md`
  precedent doesn't enforce against bad-faith relaxations, only
  documents good-faith ones.
- **Semantic-similarity leakage check.** All three audits are
  bag-of-features. A truly semantic leak (training prompts that
  *paraphrase* the held-out in vocabulary the audit doesn't share)
  would still slip through. Per `RESEARCH_DISCLOSURE.md` §7,
  semantic embeddings are the natural escalation; gated by the
  pure-C99 / zero-deps project policy.

## 11. Adoption guide for another project

To adopt this methodology in another project:

1. **Create the `RESEARCH_*.md` corpus skeleton.** One file per major
   research arc; each multi-phase plan gets a `### N.M Pre-registered
   targets` and `### N.M Skip rule` block before any measurement.
2. **Add a `RESEARCH_DISCLOSURE.md` register at the repo root or
   under `docs/`.** Single document, append-only, regulator-friendly.
3. **Lift `tools/research_status_dashboard.{c,sh}`** verbatim.
   Replace the `default_files[]` list with your project's paths. The
   parser's pre-reg block detection is corpus-agnostic.
4. **Lift `tools/scaling_leakage_audit.sh`** with edits for your
   held-out file format. The three-axis structure (verbatim / Jaccard
   / anchoring) is what's load-bearing; the awk syntax can be
   replaced with anything equivalent.
5. **Add a thresholds JSON.** Start with `default: { 0, 0, 2 }` and
   per-file entries for any deliberate exceptions (with `_comment`
   fields citing the reason).
6. **Lift `.github/workflows/leakage-audit.yml`.** The trigger paths
   need to match your project's corpus layout.
7. **Adopt the two-commit discipline.** Commit 1 = pre-registration;
   commit 2 (separate) = measurement + verdict. Reviewers enforce by
   looking for the pre-reg commit before the measurement commit.

The first project to adopt this from scratch (after `microgpt-c`) is
expected to spend ~1 week on infrastructure, then 10-15 % overhead on
ongoing research. The break-even is at the first time the discipline
catches a leak or saves the team from a confirmation-bias trap (the
Phase 3a TF-IDF case in Worked example 1 is a typical example).

## 12. What this methodology is NOT

- **Not** a substitute for cross-validation, train/test/val splits,
  or other standard statistical hygiene. It's an *additional* layer
  that catches a specific failure mode (curator-introduced leakage)
  that splits alone don't detect.
- **Not** a research-output guarantee. A pre-registered hypothesis can
  still be wrong; the methodology only ensures the wrong-ness is
  documented honestly. The cumulative outcome catalogue
  (`STATUS_DASHBOARD.md`) shows the project's hypothesis hit rate, but
  the methodology's value is independent of that rate.
- **Not** machine-enforced. Git history surfaces breaches; the
  enforcement is social. A bad-faith team can ignore the convention.
- **Not** a replacement for peer review. The methodology produces an
  *audit-ready* artefact set; an external reviewer must still examine
  it for the methodology to deliver its value.
- **Not** specific to small-model ML. The discipline is genuinely
  domain-agnostic; the artefact set (parser + audit + register)
  happens to be sized for this project's needs.

## 13. References

The methodology emerged iteratively across the project's research log.
Key sources:

- **Pre-registration discipline:**
  - `docs/research/RESEARCH_PIPELINE_IR.md` §40 (Phase 3 pre-reg block
    — the worked-example template).
  - `docs/research/RESEARCH_PIPELINE_IR.md` §42 (Phase 3b pre-reg) and
    §43 (Phase 3b outcome) — PASS example.
  - `docs/research/RESEARCH_PIPELINE_IR.md` §45 (Phase 4 pre-reg) and
    §46 (Phase 4 outcome) — EXCEEDED example.
  - `docs/research/RESEARCH_OPA_DIRECTIONS.md` §12 — two-commit
    discipline statement.
  - `docs/research/RESEARCH_OPENMYTHOS_CROSS_POLLINATION.md` — multi-
    experiment catalogue with locked targets + skip rules.

- **Leakage audit:**
  - `tools/scaling_leakage_audit.sh` — three-axis check (the standing
    protection).
  - `tools/check_held_out_leakage.sh` — verbatim guard (the canonical
    Phase-13 protection).
  - `tools/leakage_audit_thresholds.json` — the ratchet.
  - `docs/research/wiring_scaling_curve.md` — the v1 incident's
    original measurement document (with correction notice).

- **Honest disclosure:**
  - `docs/engineering/CLEAN_ROOM_IMPLEMENTATION/RESEARCH_DISCLOSURE.md`
    — the regulator-friendly register.
  - `docs/research/ORGANELLE_STATE.md` — the "Wiring Organelle arc"
    table that summarises the iteration history.

- **Tooling (this paper's artefacts):**
  - `tools/research_status_dashboard.{c,sh}` — the pre-reg parser.
  - `tools/run_leakage_audit_ci.sh` — the CI driver.
  - `.github/workflows/leakage-audit.yml` — the GitHub Actions hook.
  - `experiments/E05-prereg-methodology-public.md` — this paper's
    governing pre-registration.

- **Related-work pointers (NOT endorsements):**
  - "Pre-registration" in psychology and clinical-trials research
    (the discipline is widely practised in those fields; the
    transposition to ML measurement is what's distinctive here).
  - "Train-test contamination" in NLP benchmarks (the failure mode
    is widely recognised; the standing-audit + threshold-ratchet
    response is what's distinctive here).

— End of paper. Pre-registered as E05 §3.3; ~13 pages of markdown,
5 worked examples, all sourced from the project's own honest-
disclosure register.
