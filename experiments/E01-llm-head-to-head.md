# Experiment E01 — Head-to-head vs frontier LLM on a public typed-graph benchmark

**Status:** 📋 Proposal locked — 2026-05-20.
**Direction (per [`RESEARCH_OPA_DIRECTIONS.md`](../docs/research/RESEARCH_OPA_DIRECTIONS.md)):** repositioning OPA as a neurosymbolic verifier-gated substrate; the missing public-baseline comparison.
**Cost estimate:** ~6-8 weeks (4 wk benchmark wrangling + leakage-audit + 2-4 wk measurement + writeup).
**Falsification risk:** Medium (LLM may match OPA on audit-trail-passing-accuracy, collapsing the (accuracy, audit, latency, determinism) tuple).

---

## Spear summary

**Point:** A frontier LLM almost certainly out-accuracies OPA on raw natural-language→typed-graph generation, but OPA's contribution is not raw accuracy — it is the *audit-trail-passing*, *deterministic*, *bounded-compute* quadruple. The honest claim is testable only by running both systems on the same public benchmark with the same Pipeline IR verifier judging both.

**Picture:** Two systems answer the same 500 prompts. The LLM emits free-form `@graph...@end`; OPA emits via the wiring pipeline. The same `pipeline_verify()` judges both. We report four numbers per system: (raw accuracy, audit-trail-coverage, latency p50/p99, determinism index across 5 reruns).

**Proof (to be measured):** pre-registered targets in §1.4. Falsification = LLM dominates all four metrics — in which case the architecture's positioning needs rework.

**Push:** This is the missing public-benchmark comparison reviewers will ask for the first time the project pitches at a conference.

---

## 1. Proposal

### 1.1 Hypothesis (locked before measurement)

On a public typed-graph generation benchmark (target: ToolBench tool-call subset, or BIRD-mini SQL, or a curated Hugging Face datasets/typed-graph derivative — to be locked at §2.4), the following four-tuple inequality holds:

| Metric | Zero-shot LLM | LLM + Pipeline IR verifier | Pure OPA |
|---|---|---|---|
| Raw accuracy (exact graph match) | X | ≥ X (verifier may filter but not fix) | < X by ≤ 20 pp |
| Audit-trail coverage (verifier-passing graphs) | < 70% | 100% (by construction) | 100% |
| Latency p50 | ≥ 800 ms (network + LLM) | ≥ 800 ms | ≤ 5 ms |
| Determinism index (5 reruns, same prompt, exact-match rate) | < 100% | < 100% (LLM stochasticity) | 100% (modulo `srand` seed) |

The honest claim OPA can make if the inequalities hold: *"On distinctive-vocabulary typed-graph tasks, OPA delivers 100× lower latency and full determinism at ≤ 20 pp accuracy cost, with an audit trail that the LLM cannot provide — and can also serve as a post-hoc verifier on LLM output to lift its audit-trail coverage from <70% to 100%."*

### 1.2 Why this matters

The project's strongest framing in 2026 is **neurosymbolic verifier-gated agents** — exactly the category the frontier labs are converging on (DeepMind AlphaProof, OpenAI o-series, Anthropic safety/constitutional work). Each frontier system hand-writes its verifier per domain. OPA has built a *general-purpose typed-DAG verifier substrate* that can judge any system's output, not just its own. This experiment surfaces that asymmetry as a measurement.

Without this measurement, the project's claims sit in a vacuum — reviewers will assume "tiny model loses to LLM" until shown otherwise on the metrics that actually matter.

### 1.3 Mechanism

**Three systems under test, same benchmark prompts:**

1. **System A — Zero-shot LLM.** Prompt the LLM (Claude Sonnet 4.6 by default — see §2.3) with `(natural language prompt, IR grammar reference, 5 in-context examples)` and ask it to emit `@graph...@end` text. No post-processing.
2. **System B — LLM + IR verifier (post-hoc Judge).** As System A, but the emitted graph is parsed by `pipeline_parse_text_tolerant()` → `pipeline_repair()` → `pipeline_verify()`. Failing graphs are flagged "verifier-rejected" (counted as wrong); passing graphs go through the existing OPA reference-set check for accuracy.
3. **System C — Pure OPA.** The existing wiring binary (`demos/wiring_organelle/main.c`) with `--clean-only` flag, no LLM in the loop.

All three systems are judged by the **same** evaluation harness: prompt → expected primitives set → exact-match accuracy + audit-pass + latency + determinism.

Compute & cost budget per system per prompt: System A/B charged at LLM-API list price (recorded as a fifth metric); System C runs on the M2 Max reference machine with `MICROGPT_BLAS=ON`.

### 1.4 Pre-registered targets (locked)

Let X = System A raw accuracy on the locked benchmark. The hypothesis is judged on four pre-registered comparisons:

| Comparison | Target | Floor (skip-rule trigger) |
|---|---|---|
| **C1: System C raw accuracy ≥ X − 20 pp** | ≥ X − 20 pp | < X − 35 pp |
| **C2: System B audit coverage = 100%** | 100% (by construction; checks the harness, not the LLM) | < 99% (= harness bug) |
| **C3: System C audit coverage = 100%** | 100% | < 99% (= harness bug) |
| **C4: System A audit coverage < 70%** | < 70% (= LLM emits enough invalid graphs that the IR verifier is doing real work) | ≥ 95% (= LLM emits valid IR almost always; verifier-as-judge is redundant for this benchmark) |
| **C5: System C latency p50 ≤ 5 ms** | ≤ 5 ms | > 50 ms |
| **C6: System A latency p50 ≥ 800 ms** | ≥ 800 ms (network-dominated; tested) | < 100 ms (= LLM is local/cached) |
| **C7: System C determinism = 100% across 5 reruns** | 100% (`srand` seeded) | < 100% (= bug in seeding) |
| **C8: System A determinism < 100% across 5 reruns** | < 100% (= LLM stochasticity is measured) | 100% (= LLM is deterministic mode and the determinism story is weaker) |

The headline claim survives if **C1, C3, C5, C7 all pass** (OPA's quadruple holds) AND **C4 OR C8 passes** (the LLM has at least one weakness OPA addresses).

### 1.5 Skip rules

- If C1 falls below floor (System C accuracy < X − 35 pp): OPA accuracy gap is too large for the audit/latency story to compensate. Result: reposition the claim as "audit-trail post-hoc verifier" only (i.e. drop System C from the narrative; ship only System B as the contribution).
- If C4 fails above floor (LLM emits valid IR ≥ 95%): the IR-as-verifier-of-LLM-output story is weak for *this benchmark*. The IR verifier's value transfers to harder typed-output benchmarks; rerun on a more adversarial set before final claim.
- If audit harness reports < 99% coverage on B or C: stop and fix the harness; do not interpret the accuracy number.

### 1.6 Falsification risk: Medium

| Risk | Likelihood | Mitigation |
|---|---|---|
| LLM emits valid IR ≥ 95% (C4 fails) | Medium | Pick a benchmark with intentionally tight typing (multi-stage, multi-port composition) |
| OPA accuracy gap > 35 pp (C1 fails floor) | Medium-high on out-of-vocab axes — but `--clean-only` should keep it within band on the calibrated retrieval set | Run on Phase 2c clean equivalent first; widen only if floor holds |
| LLM API pricing makes the experiment too expensive | Low (Claude Sonnet 4.6 at list price for 500 prompts × 3 systems × 5 reruns ≈ $30-100) | Cap budget at $500; if exceeded, halve sample size |
| Benchmark licensing precludes redistribution | Medium | Pre-clear before locking benchmark in §2.4 |

### 1.7 What this experiment is NOT testing

- It is **not** testing whether OPA replaces LLMs. The architecture's thesis (`INV-WIRE-041`) is that tiny specialists + verifier substrate complement LLMs, not replace them.
- It is **not** testing whether OPA wins on accuracy. It is expected to lose by up to 20 pp; the contribution is the other three corners of the quadruple.
- It is **not** testing whether the calibrated 75-80% retrieval ceiling moves. That bound is a property of the bag-of-features retrieval mechanism (`INV-WIRE-060`) and is unaffected by what an LLM does on the same benchmark.
- It is **not** testing whether System B beats System C — both should hit 100% audit coverage by construction; their accuracy difference is the LLM's residual contribution after the verifier filter, not an OPA quality signal.

### 1.8 Cross-references

| Topic | Source |
|---|---|
| Why this experiment is on the recommended top-5 | [`RESEARCH_OPA_DIRECTIONS.md`](../docs/research/RESEARCH_OPA_DIRECTIONS.md) §9 (rank #1 framing-impact, though not in original top-5 list because it didn't have a single direction entry) |
| Why LLM-as-replacement was rejected | [`RESEARCH_OPA_DIRECTIONS.md`](../docs/research/RESEARCH_OPA_DIRECTIONS.md) §10 ("Replacing the wiring transformer with a frontier LLM as the front-line mechanism") — head-to-head as *baseline* is a different question |
| Calibrated retrieval claim being measured | [`ORGANELLE_STATE.md`](../docs/research/ORGANELLE_STATE.md) §"The current calibrated claim" |
| Wiring binary the test exercises | [`demos/wiring_organelle/main.c`](../demos/wiring_organelle/main.c) |
| Verifier-as-Judge mechanism | [`src/microgpt_pipeline.c`](../src/microgpt_pipeline.c) `pipeline_verify()` |

---

## 2. Initial state

### 2.1 What's currently known

- OPA Phase 2c clean retrieval: 20/20 (100%) on novel paraphrases in single-family anchored prompts.
- OPA Phase 3b composition: 7/10 on multi-stage compositional prompts.
- OPA V1.1.0 novel compositional generation: 19/30 (63%) leakage-audited held-out.
- OPA latency p50 on M2 Max: ≤ 5 ms (un-measured under load; needs benchmarking step).
- Frontier LLM zero-shot on similar tasks: 70-95% raw accuracy in published reports, but no published numbers on `@graph...@end` IR grammar specifically.

### 2.2 Baselines to beat / match

| Baseline | Source | OPA must |
|---|---|---|
| LLM raw accuracy on locked benchmark | To be measured in System A | match within 20 pp |
| 100% audit coverage | OPA's own headline claim | hold (no regression) |
| 5 ms p50 latency | OPA's own headline claim | hold |
| 100% determinism with seeded RNG | OPA's own headline claim | hold |

### 2.3 LLM choice (locked)

- **Primary:** Claude Sonnet 4.6 (`claude-sonnet-4-6`) via Anthropic API. Reason: most-capable model in the project's natural reach (per `CLAUDE.md` defaults), good IR-grammar generalisation reported.
- **Secondary (sanity check):** GPT-4o-mini (cheap second data point to verify the trend isn't model-specific).
- **NOT used:** Local quantised models. The point is *frontier* comparison; a 7B-param local model is a different experiment.

### 2.4 Benchmark choice — TO BE LOCKED before measurement commit

Three candidate benchmarks; pick one in the pre-reg commit, do not retroactively reshape:

| Candidate | Pros | Cons |
|---|---|---|
| **ToolBench tool-call subset (~500 prompts)** | Public, typed JSON output, distinctive vocabulary | JSON ≠ `@graph...@end`; needs grammar bridge |
| **BIRD-mini SQL (~200 prompts)** | Public, typed SQL output, hard-graded ground truth | SQL ≠ graph; needs typed-graph adapter |
| **Curated typed-graph subset from existing held-out** | Native to OPA's grammar | Smaller (~50 prompts); risk of curator-self-overlap → must pass [`tools/scaling_leakage_audit.sh`](../tools/scaling_leakage_audit.sh) gate before locking |

**Decision rule:** prefer the one with the largest *public* ground-truth set that admits an IR adapter ≤ 200 LOC. Lock by writing the choice into §2.4 of this doc in the pre-reg commit.

### 2.5 Dependencies / blockers

- Anthropic API key (~$100 budget cap).
- Public benchmark licensing cleared.
- Standing leakage audit run on the chosen benchmark vs the wiring corpus — must pass `tools/scaling_leakage_audit.sh` Audit-B (Jaccard < 0.7) before any measurement.

---

## 3. Implementation + results

**TODO** — fill on measurement commit. Sections to populate:

- 3.1 What was built (scripts, benchmark adapter, harness)
- 3.2 Raw results table (X for each of 8 comparisons C1-C8)
- 3.3 Cost log (API spend, machine time)
- 3.4 Audit-trail artefacts (links to verifier output for each system)
- 3.5 Reproduction instructions

---

## 4. Conclusion

**TODO** — fill on measurement commit. Sections to populate:

- 4.1 Verdict per pre-reg target (PASS / FAIL / FLOOR-TRIGGER per C1-C8)
- 4.2 Headline claim — whether it survives (§1.4 last line)
- 4.3 What we learned (especially: was the (accuracy, audit, latency, determinism) tuple framing the right framing?)
- 4.4 Next moves (E02 — promote Pipeline IR to standalone library — is the natural follow-on if C2 measurement holds)
- 4.5 Traceability updates (`TRACEABILITY.md`, `RESEARCH_DISCLOSURE.md`, `ORGANELLE_STATE.md`)
