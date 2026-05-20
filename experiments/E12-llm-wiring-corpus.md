# Experiment E12 — LLM-as-corpus-source for the wiring organelle (NL → typed-graph), via local LM Studio + Pipeline IR verifier filter

**Status:** 📋 Proposal locked — 2026-05-20.
**Direction:** add `CREATE CORPUS … FROM LLM …` as a new SOURCE clause in OQL; use the local LM Studio endpoint to generate (NL prompt, `@graph…@end`) pairs; filter through `libpipeline_ir.verify()` + the standing leakage audit; train a new wiring organelle on the LLM-generated corpus; measure against the v2 sealed held-out.
**Cost estimate:** ~2-3 weeks (1 wk OQL grammar + LLM bridge + cache + 1 wk corpus generation + verifier loop + 1 wk train + measure + writeup).
**Falsification risk:** Medium — depends on whether a 35B-class local LLM can produce structurally valid typed graphs at high enough yield, and whether the resulting corpus quality matches the current human-curated v2 library.

---

## Spear summary

**Point:** The wiring organelle's calibrated 75-80% ceiling has been documented as curator-bounded (`INV-WIRE-061`). E03 was supposed to falsify or confirm that bound by recruiting an independent human curator — but that requires 3-5 weeks of a second person's time and has never been run. **An LLM curator is the cheap, on-demand version of the same falsification test.** If a local LLM, gated by `pipeline_verify()` + leakage audit, can train a wiring organelle to ≥75% on the v2 sealed held-out, the curator bound is *not* the architectural ceiling — the architecture, the deterministic infrastructure (Judge + audit), and the LLM together can curate.

**Picture:** OQL extends by one new SOURCE clause: `CREATE CORPUS X FROM LLM 'qwen/qwen3.6-35b-a3b'@http://127.0.0.1:1234 PROMPT '…' VERIFY_VIA pipeline_ir AUDIT_AGAINST v2_heldout`. The LLM emits candidates; the verifier rejects ill-typed; the leakage audit rejects accidentally-leaked. Survivors land in the corpus. A new wiring organelle trains on it and is evaluated against the same sealed held-out the human curator's v2 library was scored against.

**Proof (to be measured):** ≥95% of LLM emissions pass `pipeline_verify` (the LLM produces structurally valid graphs); zero leakage-audit failures (the LLM doesn't memorise the held-out set through training-data accident); trained organelle reaches **≥75% on v2 sealed held-out** (matches the human curator's 16/20 = 80% — i.e. the LLM as second-independent-curator hits the same ceiling); engine surface frozen; zero new build deps beyond curl; ≤4 hours wall-clock for 10k examples.

**Push:** This is the experiment that gives a falsifiable answer to E03's open question (*is the calibrated ceiling curator-vocabulary-specific or architectural?*) without needing to recruit a human curator. If E12 PASSES, the ceiling is architectural-and-tooling-bound. If E12 FAILS, the ceiling really is curator-bound — and that's the more interesting result.

---

## 1. Proposal

### 1.1 Hypothesis (locked before measurement)

> *Adding `CREATE CORPUS <name> FROM LLM '<model>'@<endpoint> PROMPT '<prompt>' [WITH (count=N, seed=S, cache=PATH)] VERIFY_VIA pipeline_ir [AUDIT_AGAINST <held_out_name>];` as a new OQL SOURCE clause, using the local LM Studio endpoint at `http://127.0.0.1:1234` with model `qwen/qwen3.6-35b-a3b`, is sufficient to generate a 10k-example wiring corpus such that: (a) ≥95% of LLM emissions pass `pipeline_verify()`; (b) zero generated prompts trigger the standing leakage audit against the v2 sealed held-out; (c) a new wiring organelle trained on this LLM-generated corpus achieves ≥75% on the v2 sealed held-out (within ±5pp of the human curator's 16/20 = 80%); (d) zero new build deps beyond curl; (e) engine surface frozen.*

### 1.2 Why this matters

**E03 hypothesis was:** *"an independent human curator's library scores 14-18/20 on the v2 held-out (within ±5pp of current 16/20)"*. E03 has never run because finding an independent curator costs 3-5 weeks of someone else's time.

**E12 tests the same hypothesis with an LLM curator.** A 35B-class local LLM, gated by the existing deterministic infrastructure (verifier + leakage audit), is effectively a second-independent-curator that's available on demand. The four-corners interpretation:

| Outcome | Architectural meaning |
|---|---|
| LLM corpus → organelle scores 14-18/20 | Ceiling is **architectural-and-tooling-bound**, NOT curator-vocabulary-specific. Big win for the project's claim. |
| LLM corpus → organelle scores >18/20 | LLM curates *better* than the human. Either the LLM has seen the held-out (leakage; audit must catch this), OR the architecture has untapped headroom under better curation. Investigate. |
| LLM corpus → organelle scores 11-13/20 | LLM is a *worse* curator than the human. The ceiling is partly curator-skill — interesting and partly weakens the claim. |
| LLM corpus → organelle scores <11/20 | LLM is a *substantially worse* curator. The human curator's structural understanding matters in a way the LLM doesn't reproduce — answers E03's deepest question and strengthens the "human curator is load-bearing" claim. |

**All four are valuable.** Pre-registration discipline benefits because the result is informative regardless of direction. This is *much* better than E03's "find a human curator" framing because E12 produces a falsifiable answer in ~3 weeks, not 3+ months of recruitment.

### 1.3 Mechanism

#### 1.3.1 OQL grammar extension

Extend `microgpt_oql.{l,y}` with the new SOURCE clause:

```sql
CREATE CORPUS <name>
  FROM LLM '<model_id>'[@<endpoint_url>]
  PROMPT '<prompt_text>'
  [WITH (count = <int>, seed = <int>, cache = '<path>', max_retries = <int>)]
  [VERIFY_VIA pipeline_ir]
  [AUDIT_AGAINST <held_out_corpus_name> WITH (mode = jaccard, threshold = <float>)]
;
```

`FROM LLM` is a new **SOURCE**, not a new verb. The +6/-4 verb lock from E07 holds; CORPUS still uses `CREATE` (it was already a CREATE object in E10).

Defaults: `endpoint = http://127.0.0.1:1234`, `count = 1000`, `seed = 1337`, `cache = '.oql_llm_cache/'`, `max_retries = 5`.

#### 1.3.2 LLM bridge in `tools/llm_corpus_source.{c,h}` (NEW file)

Pure C99 + curl (subprocess). No new build dep — curl is universally available; the project's CI workflows already use it.

```c
typedef struct {
    const char *model_id;
    const char *endpoint_url;        /* http://127.0.0.1:1234 */
    const char *cache_dir;
    int         seed;
    int         max_retries;
} LlmSource;

/* Returns 0 on success, nonzero on irrecoverable error.
 * On success, `out` receives a heap-allocated string the caller frees.
 * Cache key: hash of (prompt + seed + model_id). Replay returns identical output. */
int llm_emit(const LlmSource *src,
             const char *prompt,
             char **out, size_t *out_len);
```

Transport: `curl -X POST <endpoint>/v1/chat/completions` with OpenAI-compatible JSON payload. Parse the `choices[0].message.content` field with a small handwritten JSON-extract function (≤ 50 LOC; no new deps).

#### 1.3.3 Verifier + leakage-audit loop

For each candidate from `llm_emit`:

1. **Parse:** parse the emitted text as `{"prompt": "...", "graph": "@graph ... @end"}` JSON line.
2. **Verify:** run `pipeline_parse_text_tolerant()` → `pipeline_repair()` → `pipeline_verify()` from `libpipeline_ir`. Reject on failure.
3. **Audit:** check the emitted prompt's Jaccard similarity against the held-out (if `AUDIT_AGAINST` clause provided). Reject if ≥ threshold.
4. **Append** survivor to the in-memory corpus. Continue until `count` survivors reached or `max_retries × count` total attempts.

Cache every LLM response under the cache directory keyed by `sha256(prompt + seed + model_id)`. Subsequent runs with same (prompt, seed, model) replay from cache — bit-identical corpora across re-runs (T6).

#### 1.3.4 Train + evaluate

After corpus generation:

```sql
CREATE ORGANELLE wiring_llm_v3 WITH (...);
TRAIN wiring_llm_v3 ON <llm_generated_corpus> STEPS 20000 LR 1e-3 SAVE 'checkpoints/wiring_llm_v3.ckpt';
```

Then evaluate against the existing v2 sealed held-out using the existing wiring evaluation harness (`./build/wiring_organelle_demo --clean-only` or `EVALUATE` if E10's TRAIN is wired enough to integrate).

#### 1.3.5 Phase order

| Phase | Work | Cost |
|---|---|---|
| 1 | OQL grammar + LLM bridge + cache | 3-4 days |
| 2 | Verifier + leakage-audit filter loop | 2-3 days |
| 3 | Generate 10k corpus | 2 hours (overnight, walk-away) |
| 4 | Train wiring organelle on it | 1-2 hours |
| 5 | Evaluate against v2 sealed held-out | 1 hour |
| 6 | Section 3 writeup | 2-3 days |

### 1.4 Pre-registered targets (locked)

| ID | Target | Floor (skip-rule trigger) |
|---|---|---|
| **T1** | `CREATE CORPUS … FROM LLM …` parses cleanly; new SOURCE clause integrates with existing OQL grammar | Parse failure |
| **T2** | ≥ 95% of LLM emissions pass `pipeline_verify()` after `pipeline_repair()` | < 80% (= LLM drifts wildly off-spec; prompt engineering needed) |
| **T3** | Zero generated corpus prompts trigger leakage audit against v2 sealed held-out (Jaccard ≥ 0.7) | ≥ 1 prompt fails audit (= LLM has memorised or accidentally retrieved held-out content) |
| **T4** | Trained wiring organelle on LLM-generated corpus achieves ≥ 75% on v2 sealed held-out (matches human curator's 16/20 = 80% within ±5pp) | < 65% |
| **T5** | Engine surface frozen: `git diff main -- src/microgpt.{c,h} src/microgpt_vm.{c,h,l,y}` = 0 lines | Any change |
| **T6** | Determinism: re-running with same (seed, model, prompts) produces bit-identical corpus (via cache) | Non-deterministic output |
| **T7** | Cost ≤ 0 USD (local model); wall-clock ≤ 4 hours for 10k examples on the user's hardware | > 8 hours |
| **T8** | Zero new build deps beyond curl (already universally available) | Any new dep |

The headline result is judged on **T4**. T1, T2, T5, T6, T8 are infrastructure floors. T3 is the leakage protection. T7 is the practical viability check.

### 1.5 Skip rules

- **If T2 < 80%** (LLM drifts wildly off-spec): the prompt template needs revision before measurement is meaningful. STOP, document the prompt engineering needed, do not relax the verifier filter.
- **If T3 ≥ 1** (leakage): STOP. Investigate — the LLM may have seen the held-out set during its own training. Tighten the held-out (different paraphrases) before re-running; do NOT relax the audit threshold.
- **If T4 < 65%** (organelle quality far below human-curator baseline): the LLM-as-curator falsification is **interesting** — document honestly. Per the four-corners interpretation in §1.2, this is one of the four meaningful outcomes.
- **If T5 trips** (engine surface change): STOP. The LLM is build-time-only; do not let any LLM-related code touch the engine.
- **If T8 trips** (new build dep): STOP. Investigate; the project's discipline is to keep design-time tooling at curl-or-less.

### 1.6 Falsification risk: Medium

| Risk | Likelihood | Mitigation |
|---|---|---|
| LM Studio endpoint is not running at evaluation time | Low (user-controlled) | Agent must report endpoint health as a precondition in Section 3.1 |
| Qwen 3.6 35B doesn't know the IR grammar well enough — T2 fails | Medium | Phase 1 includes a 5-prompt smoke test; iterate the prompt template until T2 ≥ 80% on the smoke set before generating 10k |
| Verifier filter rejects so much that the survivor rate is < 1% — wall-clock blows out | Medium | T7 is the budget. If exceeded, halve the requested count to 5k and document |
| LLM accidentally produces held-out paraphrases — T3 fails | Medium | Audit-B with threshold 0.7 is the standing guard; tighten to 0.5 if needed |
| Trained organelle fails to converge on LLM-generated corpus | Low | E10's loss-curve smoke test pattern catches this; reuse |
| Wall-clock on user's hardware is unknown | Medium | Agent measures and reports; if > 4hr, scale down |

### 1.7 What this experiment is NOT testing

- It is **NOT** testing whether the LLM beats the human curator on absolute win rate. The headline is **parity within ±5pp**, not improvement.
- It is **NOT** distillation-style training (LLM as teacher; student mimics LLM behaviour). E12 is corpus-curation only — the LLM produces *examples*, not *labels for examples the human already wrote*.
- It is **NOT** an LLM-replaces-architecture experiment. The LLM is a build-time tool; the runtime organelle is still the tiny transformer.
- It is **NOT** trying to make LM Studio a runtime dependency. The trained organelle never calls the LLM.
- It is **NOT** testing whether E03's human-curator experiment is now redundant. E12 and E03 are complementary measurements; if E12 PASSES, E03 becomes lower priority but still informative (humans and LLMs may curate differently).

### 1.8 Cross-references

| Topic | Source |
|---|---|
| The open question being closed | [E03](E03-independent-curator-reproducibility.md) — independent-curator reproducibility, never run |
| The OQL substrate being extended | [E07](E07-oql-dsl.md) (grammar) + [E10](E10-oql-train-wiring.md) (CORPUS object) |
| The verifier filter | [E02](E02-pipeline-ir-library.md) — `libpipeline_ir` |
| The audit infrastructure | [E05](E05-prereg-methodology-public.md) — `tools/scaling_leakage_audit.sh` + CI hook |
| The held-out set being scored against | `demos/wiring_organelle/pipeline_corpus_scaling_heldout_v2.txt` |
| The human-curator baseline being matched | 16/20 (80%) on v2 sealed held-out, per `ORGANELLE_STATE.md` |
| Pre-existing LLM-distillation pre-reg (game-specific, complementary to E12) | [E13](E13-llm-game-distillation.md) — running in parallel |
| OPA directions catalogue entry | `RESEARCH_OPA_DIRECTIONS.md` §5.1 (LLM distillation; E12 is the corpus-curation cousin) |
| Policy line E12 must NOT cross | `RESEARCH_OPA_DIRECTIONS.md` §10 — LLM as runtime is forbidden; build-time only |

---

## 2. Initial state

### 2.1 What's currently known

- Human-curator (Ajay) v2 wiring corpus: 16/20 (80%) on v2 sealed held-out.
- v2 sealed held-out exists at `demos/wiring_organelle/pipeline_corpus_scaling_heldout_v2.txt`.
- Standing leakage audit (`tools/scaling_leakage_audit.sh`) handles Audit-A (verbatim) + Audit-B (Jaccard ≥ 0.7).
- `libpipeline_ir` shipped via E02; verifier accessible from anywhere via `<pipeline_ir/pipeline_ir.h>`.
- OQL has `CREATE CORPUS … FROM FILE …` from E10; extending to `FROM LLM …` is one new SOURCE clause.
- E03 never run; this is its proxy via a different curator-source.

### 2.2 Baselines to beat (or match)

| Baseline | Number | E12 must |
|---|---|---|
| Human curator v2 score | 16/20 = 80% | match within ±5pp (target ≥ 75%; floor 65%) |
| Number of new build deps (current) | 0 | hold at 0 (curl already universal) |
| Wall-clock for human curator to produce v2 | 3-5 weeks | reduce to ≤ 4 hours (T7) |
| Engine code changes | 0 lines | hold at 0 (T5) |

### 2.3 Dependencies / blockers

- **LM Studio running at `http://127.0.0.1:1234`** with `qwen/qwen3.6-35b-a3b` loaded — user-provided, must be live at agent runtime. Agent's first step is a curl health-check.
- **curl available** — universal; no install needed.
- **`libpipeline_ir`** — on main from E02.
- **`scaling_leakage_audit.sh`** — on main from E05.
- **Wiring training engine** — on main; agent uses the existing `wiring_organelle_demo` or `TRAIN` (E10) for the training run.

### 2.4 What this experiment deliberately does NOT do

- Does NOT make the LLM a runtime dependency. Trained organelle has no `curl` calls at inference.
- Does NOT add VM opcodes (E08 hard-lock).
- Does NOT add OQL verbs (E07 +6/-4 lock). `FROM LLM` is a SOURCE clause inside the existing `CREATE CORPUS`.
- Does NOT touch the engine (`src/microgpt.{h,c}`, `src/microgpt_vm.*`).
- Does NOT replace the human-curator v2 corpus on main. The LLM corpus lives as a separate corpus file; the human's v2 stays as the baseline.

---

## 3. Implementation + results

**TODO** — fill on measurement commit. Sections to populate:

- 3.1 LM Studio health-check + endpoint confirmation
- 3.2 OQL grammar extension diff
- 3.3 `tools/llm_corpus_source.{c,h}` — bridge + cache + verifier/audit loop
- 3.4 Prompt-template iterations (Phase 1 smoke test; document the iterations honestly)
- 3.5 Corpus generation stats (T2 yield, T3 leakage count, T6 cache hit rate, T7 wall-clock)
- 3.6 Trained organelle measurement on v2 sealed held-out (T4)
- 3.7 Engine-surface-frozen confirmation (T5: 0-line diff)
- 3.8 Per-target verdict matrix
- 3.9 Four-corners interpretation of the T4 result per §1.2

---

## 4. Conclusion

**TODO** — fill on measurement commit when ALL 8 targets are measured. Sections to populate:

- 4.1 Verdict per T1-T8
- 4.2 Headline outcome — which of the four corners (§1.2) did T4 land in?
- 4.3 What this says about the curator bound (`INV-WIRE-061`):
  - If T4 ≥ 75%: ceiling is architectural-and-tooling-bound; weakens the "human curator is essential" claim
  - If T4 < 65%: ceiling is partly curator-skill-bound; strengthens the claim
- 4.4 What this says about E03 (the unrun human-curator experiment):
  - If T4 ≥ 75%: E03 becomes lower priority but still informative on within-human variance
  - If T4 < 65%: E03 becomes more urgent — we need to know if a human curator hits the LLM's lower bound or stays at 80%
- 4.5 Compound benefits realised:
  - OQL gains a new SOURCE (`FROM LLM`); applies generally to any corpus class
  - The verifier-filter pattern is reusable for any LLM-source experiment
  - Future experiments (E14+) can author corpora in a single `.oql` file with the LLM as the generator
- 4.6 Traceability updates (`TRACEABILITY.md`, `ORGANELLE_STATE.md`, `RESEARCH_DISCLOSURE.md` §X recording the curator-bound result)
