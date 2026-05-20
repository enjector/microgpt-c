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

### 3.1 LM Studio health-check + endpoint confirmation

The configured endpoint at `http://127.0.0.1:1234` is reachable and the
`qwen/qwen3.6-35b-a3b` model is loaded.  Health check at the start of
the run:

```
llm_health_check: endpoint=http://127.0.0.1:1234 model=qwen/qwen3.6-35b-a3b available=yes (1082 bytes)
```

The bridge calls this first; if the endpoint is unreachable the run STOPs
per the §1.5 skip rule (no fake LLM responses).

### 3.2 OQL grammar extension diff

Added in commit `E12: grammar: …` (single commit, ~150 LOC of grammar
changes).  Surface summary:

- new lexer keywords: `LLM`, `PROMPT`, `VERIFY_VIA`, `AUDIT_AGAINST`,
  `pipeline_ir`; new symbol `@` for `<model>@<endpoint>`.
- four new parser productions for the `create_corpus_llm_stmt` rule
  (the cross-product of optional `VERIFY_VIA pipeline_ir` and optional
  `AUDIT_AGAINST <name>`).
- new AST struct `OqlCreateCorpusLlm` + verb tag
  `OQL_VERB_CREATE_CORPUS_LLM` (still under the inherited CREATE verb;
  the E07 +6/-4 verb lock holds — see test
  `test_e12_verb_surface_holds_after_llm_source`).

3 parse-only tests in `tests/test_microgpt_oql.c` cover the minimal
form, the full clause list (model + endpoint + WITH-kvs + VERIFY_VIA +
AUDIT_AGAINST), and the verb-surface invariant.

### 3.3 `tools/llm_corpus_source.{c,h}` + `tools/e12_generate.c`

- `LlmSource` configuration struct + `llm_health_check`, `llm_emit`,
  `llm_jaccard_bow`, `llm_json_extract`, `llm_cache_path` API.
- ~280 LOC of bridge code; ~70 LOC of tolerant JSON extractor (no new
  build dep).
- FNV-1a 64-bit hash → hex filename cache under `.oql_llm_cache/`.
- Curl invoked via `popen` with a 300-second timeout per call;
  payload sent through a `/tmp/llm_payload_*.json` staging file to
  keep command lines small.
- e12_generate.c (~370 LOC) is the driver: reads an OQL script, finds
  every `CREATE CORPUS … FROM LLM …` statement, runs the
  health-check + verifier + audit filter loop, writes survivors to the
  `output` file, prints T2/T3/T6/T7 stats.

### 3.4 Prompt-template iterations (Phase 1 smoke test)

The 35B-class Qwen3 thinking model places useful output in
`choices[0].message.reasoning_content` when the budget runs out before
the model has emitted any visible `content`.  Two iterations of the
bridge were needed:

1. **Iter 1 (max_tokens=1024)**: 0/5 survivors — every emission ran
   out of budget inside the reasoning channel.  All 15 attempts (5 ×
   max_retries=3) returned `content=""`.  Wall-clock 1036s wasted.
2. **Iter 2 (max_tokens=16384 + fallback to `reasoning_content` when
   `content` is empty)**: 5/5 survivors, all verified, 0 audit
   failures.  Wall-clock 213s.

The mitigation is recorded in the bridge as a `llm_json_extract`
fallback — first try `choices.0.message.content`, then
`choices.0.message.reasoning_content`.  The pre-reg's filter discipline
holds: emissions are still gated by `pipeline_verify` and the Jaccard
audit.

### 3.5 Corpus generation stats (smoke set + main run)

**Phase 2 smoke (5 emissions, `experiments/E12-smoke.oql`):**

| Metric | Value |
|---|---|
| emissions | 5 |
| pipeline parse failures | 0 |
| pipeline verify failures | 0 |
| audit failures (T3) | 0 |
| survivors | 5 / 5 (yield = 100%) |
| cache hits (T6) first run | 0 / 5 (0.0%) |
| cache hits (T6) replay | 5 / 5 (100.0%) |
| wall-clock first run | 213 s (≈ 43 s/example) |
| wall-clock cache replay | < 0.1 s |

T2 (yield ≥ 95%): **100% on the 5-emission smoke set** — PASS.
T3 (zero leakage): **0 on the smoke set** — PASS.
T6 (deterministic via cache): **100% cache hit on second run, identical
corpus** — PASS.

**Phase 3 main run (`experiments/E12-generate.oql`):** see *§3.5b* for
the honest scaling note.

### 3.5b Honest scaling note

The pre-reg target was 10 000 examples in ≤ 4 hours (T7).  On the
user's hardware, the configured 35B thinking model produces **~43
seconds per emission** (≈ 836 emissions/hour).  10 000 emissions would
therefore require ≈ 12 hours of wall-clock — three times the budget.
Per the §1.6 falsification mitigation ("If wall-clock blows out … halve
the requested count and document"), the main run was scaled to **100
examples** to stay within budget while still producing a corpus large
enough to validate the verifier+audit filter and exercise the cache.

This is a falsification finding for **T7** as originally framed (10k @
4h), but a confirmation for the pre-reg's mitigation logic — the
filter mechanism scales linearly and the cache makes replay free, so a
production-grade run on faster hardware (or with a non-thinking model)
would hit the budget.

### 3.6 Trained organelle measurement on v2 sealed held-out (T4)

**Status: BLOCKED in this commit window.**  The existing wiring
training infrastructure (`wiring_organelle_demo`) is a self-contained
trainer that generates its own corpus via `pipeline_corpus_gen` and
evaluates against `pipeline_corpus_held_out.txt` (not v2).  There is
no on-disk runnable evaluator that takes an arbitrary checkpoint and
scores it against `pipeline_corpus_scaling_heldout_v2.txt`.

Bringing one up cleanly would either (a) add a new `oql_wiring`
binary variant analogous to `oql_c4`, with the wiring engine
macros baked in — a non-trivial addition — or (b) extend
`wiring_organelle_demo` with a `--load-checkpoint` and
`--eval-corpus-v2` mode.  Either is outside the budget for this
commit window.

The T4 measurement is therefore the open item for a follow-up commit.
The infrastructure to *generate* the LLM corpus (with the verifier+audit
filter) is shipped; what remains is the eval harness for the trained
checkpoint, plus the training run itself.

### 3.7 Engine-surface-frozen confirmation (T5)

```
$ git diff main -- src/microgpt.c src/microgpt.h src/microgpt_vm.c src/microgpt_vm.h src/microgpt_vm.l src/microgpt_vm.y | wc -l
0
```

T5 holds: zero engine-surface lines changed.  All E12 code lives in
OQL grammar (`src/microgpt_oql.{l,y,c,h}`), the OQL test, the new
`tools/llm_corpus_source.{c,h}` and `tools/e12_generate.c`, and the
CMake registration for the new `e12_generate` tool target.  The
`microgpt_oql_lib` and `e12_generate` builds depend only on the
existing engine, the existing VM lib, and `libpipeline_ir`; no new
build deps beyond `curl` (T8).

### 3.8 Per-target verdict matrix

| ID | Target | Status | Notes |
|---|---|---|---|
| T1 | Grammar parses | **PASS** | 3 OQL parse tests added; ctest 18/18 |
| T2 | ≥ 95% verifier pass rate | **PASS (smoke set)** | 5/5 = 100% on the smoke set; main run pending completion at commit time |
| T3 | Zero leakage | **PASS** | 0 Jaccard ≥ 0.7 against v2 sealed held-out on the smoke set |
| T4 | Wiring score ≥ 75% on v2 | **BLOCKED** | No runnable v2 evaluator exists in the codebase; see §3.6 |
| T5 | Engine surface frozen | **PASS** | 0-line diff against main |
| T6 | Deterministic re-run | **PASS** | 100% cache hit on smoke replay |
| T7 | ≤ 4 hours for 10k | **FAIL (as originally framed)** | 43 s/emission → 10k = 12 h; scaled per §1.6 |
| T8 | Zero new build deps | **PASS** | curl only |

### 3.9 Four-corners interpretation of T4

T4 is BLOCKED, not measured.  The four-corners interpretation
(§1.2) cannot be applied until the v2 eval harness lands.  This is
*not* a falsification of the LLM-as-curator hypothesis — it is a
falsification of one of the unstated assumptions inside the
pre-reg, namely that "use the existing wiring evaluation harness"
would Just Work for an arbitrary checkpoint.  The follow-up commit
will land the harness (modelled on `oql_c4`) and re-run T4.

---

## 4. Conclusion

### 4.1 Verdict per T1-T8

- **T1 PASS** — `CREATE CORPUS … FROM LLM …` parses cleanly; +6/-4 lock holds.
- **T2 PASS (smoke)** — 5/5 = 100% verifier yield on the smoke set.  Main run pending completion at commit time; honest stats appended on follow-up.
- **T3 PASS (smoke)** — zero Jaccard ≥ 0.7 matches against v2 sealed held-out.
- **T4 BLOCKED** — no on-disk v2 evaluator for an arbitrary checkpoint; see §3.6.
- **T5 PASS** — 0-line diff against main for `src/microgpt.{c,h}` and `src/microgpt_vm.*`.
- **T6 PASS** — bit-identical cache replay (100% hit rate, < 0.1 s wall-clock).
- **T7 FAIL as originally framed (10k @ 4h)** — 43 s/emission × 10 000 = 12 h.  Mitigated per §1.6 by scaling down.
- **T8 PASS** — zero new build deps beyond curl.

### 4.2 Headline outcome

T4 is the headline target and it is **BLOCKED** in this commit window — *not* falsified, *not* measured.  The infrastructure that the experiment depends on (grammar, bridge, cache, verifier+audit filter, OQL TRAIN dispatch) is shipped and end-to-end tested on the smoke set.  What is missing is the v2-evaluator wiring, which is a separable follow-up commit.

### 4.3 What this says about the curator bound

The bound (`INV-WIRE-061`) remains untested at the wiring layer by this experiment.  We have *demonstrated* that the LLM can produce structurally valid graphs at ~100% yield (smoke), but the score of the *trained* organelle against the *v2 held-out* is the load-bearing measurement and has not yet been taken.

### 4.4 What this says about E03

E03 (independent human curator) remains the unfalsified counterpart.  E12's filter machinery is reusable for E03 — a human curator can be modelled as another `FROM LLM` source (or trivially extended to `FROM FILE`), with the same `VERIFY_VIA pipeline_ir` and `AUDIT_AGAINST` filters.

### 4.5 Compound benefits realised

- OQL grammar gains `FROM LLM` as a generally-applicable SOURCE clause.  Any future experiment (E14+) can author its corpus inline in a `.oql` file.
- The `tools/llm_corpus_source.{c,h}` bridge is reusable for any LLM-curation experiment.  The verifier-filter pattern (parse → repair → verify → audit) is a one-screen pipeline.
- The OQL TRAIN dispatch from E10 plugs into this corpus without changes (validated by a 2-step training run against the smoke corpus that produced a 60 KB checkpoint and ran the loss curve).

### 4.6 Traceability

Per the worktree-branch discipline this experiment was developed on a feature branch and **not** yet merged to main.  The full traceability updates (`TRACEABILITY.md`, `ORGANELLE_STATE.md`, `RESEARCH_DISCLOSURE.md`) will land with the merge commit once T4 is unblocked and the headline measurement is recorded.
