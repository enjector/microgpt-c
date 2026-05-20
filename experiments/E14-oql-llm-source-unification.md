# Experiment E14 — Unify E12 and E13's LLM access under a single `LLM_SOURCE` OQL object + shared `tools/llm_endpoint.{c,h}` transport

**Status:** 📋 Proposal locked — 2026-05-20.
**Direction:** consolidate the parallel LLM-bridge work from [E12](E12-llm-wiring-corpus.md) and [E13](E13-llm-game-distillation.md) into one OQL surface (`CREATE LLM_SOURCE ...`) backed by one transport layer (`tools/llm_endpoint.{c,h}`). Mode-specific logic (corpus_emit / game_player / paraphrase) becomes a typed configuration, not a separate code path.
**Cost estimate:** ~2-3 weeks (1 wk grammar + LLM_SOURCE object + endpoint refactor; 1 wk E12/E13 bit-identical reproduction; 1 wk paraphrase mode + writeup).
**Falsification risk:** Medium — depends on whether three mode shapes (creation, game-play, transformation) absorb cleanly under one dispatch pattern without forcing a fourth abstraction layer.

---

## Spear summary

**Point:** E12 and E13 both shipped LLM-access code against the same local LM Studio endpoint with the same Qwen 3.6 35B model — but as two parallel implementations (`tools/llm_corpus_source.{c,h}` ~500 LOC, `tools/llm_game_player.{c,h}` ~700 LOC) with substantial duplication (curl + JSON payload + cache + reasoning_content fallback + retry loop). The user flagged this mid-flight ("could we have one OQL surface for LLMs?"); E14 ships that surface plus the underlying consolidation.

**Picture:** OQL extends by **one new first-class object type** (`LLM_SOURCE`, created via the existing `CREATE` verb — no 7th verb; the +6/-4 lock from E07 holds). The two existing corpus-generation specs (`E12-generate.oql`, `c4_distill` workflow) rewrite to use the unified surface and produce **bit-identical output** to the current E12/E13 corpora. A third worked mode (`paraphrase`) demonstrates the unification absorbs a new shape without further duplication.

**Proof (to be measured):** E12's smoke set reproduces bit-identically under `FROM LLM_SOURCE qwen WITH (mode = corpus_emit, …)`; E13's distillation corpus reproduces bit-identically under `mode = game_player`; combined LOC of `tools/llm_endpoint.{c,h}` + refactored `llm_corpus_source.{c,h}` + refactored `llm_game_player.{c,h}` is ≤ **600 lines** (down from ~1200); engine surface frozen; +6/-4 verb lock holds.

**Push:** This is the experiment that completes the OQL substrate's claim to be a **single declarative surface** for design-time LLM use. Without E14, the project has two LLM bridges that happen to use the same endpoint. With E14, the project has one `LLM_SOURCE` object that researchers compose into any corpus-generation, game-playing, or transformation workflow declaratively.

---

## 1. Proposal

### 1.1 Hypothesis (locked before measurement)

> *Adding `CREATE LLM_SOURCE <name> FROM '<model>' AT '<endpoint>' WITH (...)` as a new first-class OQL object type — created via the existing `CREATE` verb, with no new top-level verb — and refactoring `tools/llm_corpus_source.{c,h}` + `tools/llm_game_player.{c,h}` to share a new `tools/llm_endpoint.{c,h}` transport layer is sufficient to: (a) reproduce E12's smoke corpus bit-identically under the unified surface; (b) reproduce E13's distillation corpus bit-identically under the unified surface; (c) implement a third mode (`paraphrase`) demonstrating the dispatch absorbs a new shape; (d) cut combined LOC from ~1200 to ≤ 600 (>50% reduction); (e) hold the +6/-4 verb lock, the zero-new-VM-opcode lock, and the engine-surface-frozen lock from prior experiments.*

### 1.2 Why this matters

The user's question — *"could LLM use be controlled via OQL?"* — was correct on first principles and is now validated by the fact that **two parallel agent runs built essentially the same transport twice**. E12 and E13 each independently solved:
- curl subprocess invocation
- JSON payload construction with OpenAI-compatible chat-completions shape
- Response parsing (including the Qwen3 `reasoning_content` fallback the E12 agent discovered)
- Cache key derivation + cache hit/miss accounting
- Retry loop on parse failure
- Endpoint URL + model ID configuration

That's the duplication. The unique-to-each-experiment parts are:
- E12: parse JSON-line `{"prompt": ..., "graph": ...}` + verify via `libpipeline_ir` + Jaccard audit
- E13: parse single-digit move + validate legality + game-rule fallback

The first set is the **shared transport** (lives in `llm_endpoint.{c,h}`). The second set is the **mode-specific filter** (stays in `llm_corpus_source` and `llm_game_player`, but slimmed down). The OQL surface exposes the combined capability declaratively.

Beyond consolidation, E14 also **opens a new design move**: future modes (paraphrase, adversarial, spec-to-corpus, critic) become small additions to the dispatch table, not parallel C tools. That's the same compounding pattern E08's `BEHAVIOUR` had — a new object type that absorbs many future capabilities without changing the verb surface.

### 1.3 Mechanism

#### 1.3.1 OQL grammar extension

Extend `microgpt_oql.{l,y}` with the new object type:

```sql
CREATE LLM_SOURCE <name>
  FROM '<model_id>'
  AT '<endpoint_url>'
  WITH (
    temperature = <float>,        -- default 0.2
    max_tokens = <int>,           -- default 16384 (per E12's thinking-model finding)
    seed = <int>,                 -- default 1337
    cache = '<path>',             -- default '.oql_llm_cache/'
    max_retries = <int>           -- default 5
  );
```

And extend existing `CREATE CORPUS … FROM LLM …` to accept `LLM_SOURCE <name>`:

```sql
-- The current E12 form (still supported as a sugar for the unified form):
CREATE CORPUS X FROM LLM 'qwen/...' AT '...' PROMPT '...' WITH (...);

-- The new unified form:
CREATE LLM_SOURCE qwen FROM 'qwen/qwen3.6-35b-a3b' AT 'http://127.0.0.1:1234';

CREATE CORPUS wiring_v3 FROM LLM_SOURCE qwen
  WITH (mode = corpus_emit, count = 10000, prompt = '...')
  VERIFY_VIA pipeline_ir
  AUDIT_AGAINST wiring_v2_heldout;

CREATE CORPUS c4_distill FROM LLM_SOURCE qwen
  WITH (mode = game_player, game = connect4, games = 1000, only_winning = true);

CREATE CORPUS wiring_v3_aug FROM LLM_SOURCE qwen
  WITH (mode = paraphrase, base_corpus = wiring_v2, multiplier = 5);
```

**`LLM_SOURCE` is a new object type, not a new verb.** The +6/-4 verb lock holds (same pattern as E08's `BEHAVIOUR` and E10's `CORPUS`).

#### 1.3.2 Shared transport — `tools/llm_endpoint.{c,h}` (NEW file, ~300 LOC budget)

```c
typedef struct {
    const char *model_id;
    const char *endpoint_url;          /* http://127.0.0.1:1234 */
    const char *cache_dir;
    int         seed;
    int         max_retries;
    float       temperature;
    int         max_tokens;             /* default 16384 per E12 finding */
} LlmEndpoint;

/* Returns 0 on success, nonzero on irrecoverable error.
 * Handles the Qwen3 thinking-model trap automatically: tries
 * choices[0].message.content first, falls back to .reasoning_content.
 * Cache key: hash of (prompt + seed + model_id + temperature + max_tokens). */
int llm_endpoint_emit(const LlmEndpoint *ep,
                      const char *prompt,
                      char **out, size_t *out_len,
                      int *cache_hit);  /* optional: caller can pass NULL */

/* Cache stats — used in measurement output */
void llm_endpoint_get_stats(const LlmEndpoint *ep,
                            int *hits, int *misses, double *wall_clock_seconds);
```

Houses: curl subprocess + JSON payload construction + tolerant JSON extractor (including reasoning_content fallback) + cache + retry loop. **Everything E12 and E13 currently duplicate.**

#### 1.3.3 Mode adapters (refactored, slimmer)

`tools/llm_corpus_source.{c,h}` shrinks to **mode-specific filtering** only:

```c
/* corpus_emit mode: parse JSON-line, verify graph, Jaccard audit */
int llm_mode_corpus_emit(const LlmEndpoint *ep,
                         const char *prompt,
                         const Pipeline *verifier_ctx,
                         const char *audit_corpus_path, float audit_threshold,
                         char **out_prompt, char **out_graph);

/* paraphrase mode: read base corpus entry, ask LLM to rephrase */
int llm_mode_paraphrase(const LlmEndpoint *ep,
                        const char *base_prompt,
                        const char *base_graph_unchanged,
                        char **out_paraphrased_prompt);
```

`tools/llm_game_player.{c,h}` shrinks to the game-player parser + legality check:

```c
/* game_player mode: parse single-digit, validate legality */
int llm_mode_game_player(const LlmEndpoint *ep,
                         const char *board_string,
                         int (*legal_check)(const char *board, int move),
                         int *out_move);
```

**Estimated LOC**: `llm_endpoint` ~300 + `llm_corpus_source` (refactored) ~150 + `llm_game_player` (refactored) ~100 = **~550 total** (target ≤ 600; current combined ~1200).

#### 1.3.4 OQL runtime dispatch

OQL runtime resolves `FROM LLM_SOURCE <name>` to the registered endpoint, then dispatches based on the `mode = …` clause to the appropriate `llm_mode_*` adapter. The dispatch table is the OQL object-type table extended with `OQL_OBJECT_LLM_SOURCE`. Mirrors E08's BEHAVIOUR / E10's CORPUS / E09's ORGANELLE patterns.

#### 1.3.5 Bit-identical reproduction tests

| Test | Pre-E14 baseline | E14 result must |
|---|---|---|
| E12 smoke (5 prompts) | Cache hash for each emission stored in build/.oql_llm_cache | reproduce same 5 emissions byte-for-byte |
| E13 distillation (~5652 pairs) | `data/c4_distill_corpus.txt` on main | reproduce same lines byte-for-byte |
| Paraphrase mode (new) | N/A — first worked example | output non-empty, parseable, distinct from base prompt, base_graph_unchanged respected |

#### 1.3.6 Phase order

| Phase | Work | Cost |
|---|---|---|
| 1 | Grammar (`CREATE LLM_SOURCE`) + AST + runtime registry | 2-3 days |
| 2 | `tools/llm_endpoint.{c,h}` extraction from E12 + E13's shared code | 3-4 days |
| 3 | Refactor `llm_corpus_source.{c,h}` to use `llm_endpoint`; E12 bit-identical test | 2-3 days |
| 4 | Refactor `llm_game_player.{c,h}` to use `llm_endpoint`; E13 bit-identical test | 2-3 days |
| 5 | Implement `paraphrase` mode | 3-4 days |
| 6 | Section 3 writeup; LOC + bit-identical measurements | 2-3 days |

### 1.4 Pre-registered targets (locked)

| ID | Target | Floor (skip-rule trigger) |
|---|---|---|
| **T1** | `CREATE LLM_SOURCE … FROM '…' AT '…' WITH (…);` parses cleanly; OQL grammar accepts the new object type | Parse failure |
| **T2** | E12's 5-prompt smoke corpus reproduces **bit-identically** under the unified surface | Any divergence beyond LLM nondeterminism (cache replay should be deterministic) |
| **T3** | E13's distillation corpus reproduces **bit-identically** under the unified surface (cache replay) | Any divergence |
| **T4** | Combined LOC for `llm_endpoint.{c,h}` + refactored `llm_corpus_source.{c,h}` + refactored `llm_game_player.{c,h}` ≤ **600 lines** (current combined ~1200) | > 800 |
| **T5** | Engine surface frozen: `git diff main -- src/microgpt.{c,h} src/microgpt_vm.*` = 0 lines | Any change |
| **T6** | +6/-4 OQL verb lock holds: `CREATE LLM_SOURCE` is a CREATE object, not a 7th verb | A 7th top-level verb is added |
| **T7** | Zero new build deps beyond curl | Any new dep |
| **T8** | `paraphrase` mode demonstrated as the third worked example: parses, runs, produces non-trivial output on a 5-prompt test | Mode fails to dispatch or produces empty output |
| **T9** | All existing tests pass: `test_microgpt_oql` + `test_microgpt_oql_train` + ctest 18/18 | Any regression |

The headline survives if **T1, T2, T3, T4, T5, T6, T9 all pass**. T7/T8 are usability backstops.

### 1.5 Skip rules

- **If T2 or T3 trips** (bit-identical reproduction fails): the unification has *lost* something. Diagnose; do NOT ship until both reproductions are exact (cache replay must be deterministic by construction).
- **If T4 trips above 800 LOC** (consolidation not happening): the unification isn't actually consolidating — it's just adding a thin layer. Falsifies the design; pre-reg explicitly anticipates this.
- **If T5 or T6 trip** (engine surface or verb lock): STOP. These are the project's longest-held architectural locks; they must hold.
- **If T8 trips** (paraphrase mode fails): the dispatch pattern doesn't absorb three different mode shapes cleanly. Document the design limit; consider whether a fourth abstraction layer is needed. Do NOT add a third bridge file as a workaround.

### 1.6 Falsification risk: Medium

| Risk | Likelihood | Mitigation |
|---|---|---|
| The three mode shapes (creation, game-play, transformation) don't share enough to consolidate cleanly under one dispatch pattern | Medium | T4 LOC budget is the falsification gate; the design either consolidates or it doesn't |
| Bit-identical reproduction of E12/E13 fails due to subtle cache-key differences | Medium | Cache key includes (prompt + seed + model_id + temperature + max_tokens) — must match what E12/E13 used; pre-reg names this explicitly |
| Paraphrase mode requires a different prompt template than corpus_emit and the dispatch doesn't accommodate cleanly | Medium-low | Each mode owns its prompt template; the shared part is the transport |
| `LLM_SOURCE` object type accidentally drifts the verb count (e.g. someone adds `START LLM_SOURCE` or similar) | Low | T6 hard-locks this |
| LM Studio endpoint unreachable at measurement time | Low (user-controlled) | Agent must health-check; T2/T3 use cache replay so endpoint isn't actually needed for bit-identical reproduction |

### 1.7 What this experiment is NOT testing

- It is **NOT** lifting E12's T4 (corpus-size confound). The 0/20 falsification stands; E14 doesn't change the corpus scale.
- It is **NOT** lifting E13's T1 (saturated-distillation regime). The 89% neutral-band result stands; E14 doesn't change the teacher.
- It is **NOT** adding modes beyond `corpus_emit` / `game_player` / `paraphrase`. Future modes (adversarial, spec-to-corpus, critic) are explicit E15+ scope.
- It is **NOT** making `LLM_SOURCE` work at runtime. The LLM stays design-time only — the trained organelles never call it.
- It is **NOT** consolidating to a single binary. `e12_generate`, `c4_distill_corpus_gen`, and the future paraphrase driver stay as separate tools — they just share `llm_endpoint`.

### 1.8 Cross-references

| Topic | Source |
|---|---|
| Origin question | User message after E11: *"could we have one OQL surface for LLMs?"* |
| Parent A (corpus generator pattern) | [E12](E12-llm-wiring-corpus.md) |
| Parent B (game player pattern) | [E13](E13-llm-game-distillation.md) |
| Verb-discipline lock that must hold | [E07](E07-oql-dsl.md) §1.3.1 |
| New-object-via-CREATE precedent | [E08](E08-oql-behaviours.md) (`BEHAVIOUR`), [E10](E10-oql-train-wiring.md) (`CORPUS`) |
| OQL runtime registry pattern to mirror | [E09](E09-oql-runtime-wiring.md) (`OqlOrganelle`, `OqlPipeline`) |
| Cross-experiment finding driving this | [`ORGANELLE_STATE.md`](../docs/research/ORGANELLE_STATE.md) §"Three key cross-experiment architectural findings" #3 |
| The Qwen3 thinking-model trap | E12 §3.4 — must be encapsulated in `llm_endpoint` |
| Pre-reg origin (E07 §4.6 mentioned this) | E07 §4.6 next-moves |

---

## 2. Initial state

### 2.1 What's currently known

- E12 shipped `tools/llm_corpus_source.{c,h}` (~500 LOC) — curl + cache + JSON + Jaccard + tolerant JSON extractor with `reasoning_content` fallback.
- E13 shipped `tools/llm_game_player.{c,h}` (~700 LOC) — curl + cache + JSON + single-digit parse + legal-move check.
- **Shared code between the two** (the consolidation target): curl invocation, JSON payload, response parsing (incl. reasoning_content), cache key derivation, cache hit/miss, retry loop, endpoint config.
- Both use the same LM Studio endpoint and same Qwen 3.6 35B model.
- Cache files from E12 (100 emissions) and E13 (5652 emissions) are on disk in their respective worktree build directories.

### 2.2 Baselines to beat

| Baseline | Current | E14 must |
|---|---|---|
| Combined LOC | ~1200 | ≤ 600 (>50% reduction) |
| LLM bridges that exist | 2 parallel implementations | 1 shared transport + 3 mode adapters |
| OQL syntactic surface for LLM use | Inconsistent: `FROM LLM …` (E12) + procedural tools (E13) | Unified: `CREATE LLM_SOURCE` then `FROM LLM_SOURCE` |
| Number of new top-level OQL verbs | 0 (E07 lock) | 0 (T6 lock) |
| Engine surface diff | 0 lines (E07-E13 cumulative) | 0 lines |
| New build deps | curl only | curl only |

### 2.3 Dependencies / blockers

- **E12 + E13 merged on main** ✅ — both consolidations available
- **LM Studio at `http://127.0.0.1:1234`** with `qwen/qwen3.6-35b-a3b` — only needed for *new* LLM calls; bit-identical reproductions (T2, T3) use cache replay
- **Existing cache files** — E12's `build/.oql_llm_cache/` and E13's `build/data/c4_distill_cache/` need to be reachable for replay (path may need normalisation under the unified surface)
- **`libpipeline_ir`** — already on main from E02; `llm_mode_corpus_emit` consumes it for verification

### 2.4 What this experiment deliberately does NOT do

- Does NOT make `LLM_SOURCE` callable at runtime. Trained organelles still have no `curl` calls.
- Does NOT add VM opcodes or OQL verbs.
- Does NOT change the engine.
- Does NOT lift E12's T4 falsification or E13's neutral-band outcome.
- Does NOT introduce new bridge files. Three modes + one transport is the architectural budget.
- Does NOT touch other experiment territories (E01-E11 docs, wiring corpus, game demos).

---

## 3. Implementation + results

**TODO** — fill on measurement commit. Sections to populate:

- 3.1 OQL grammar extension: `CREATE LLM_SOURCE` parsing + AST + runtime registry
- 3.2 `tools/llm_endpoint.{c,h}` extraction — what moved, what stayed
- 3.3 `tools/llm_corpus_source.{c,h}` refactor diff (E12's logic now minus the transport)
- 3.4 `tools/llm_game_player.{c,h}` refactor diff (E13's logic now minus the transport)
- 3.5 E12 smoke-corpus bit-identical reproduction (T2) — hash table or byte-comparison summary
- 3.6 E13 distillation-corpus bit-identical reproduction (T3) — same shape
- 3.7 `paraphrase` mode implementation + 5-prompt smoke test (T8)
- 3.8 LOC measurement (T4) — `wc -l` against the three files
- 3.9 Verb-lock confirmation (T6) — grep against grammar
- 3.10 Engine-surface-frozen confirmation (T5)
- 3.11 Test regression confirmation (T9)
- 3.12 Per-target verdict matrix

---

## 4. Conclusion

**TODO** — fill on measurement commit when ALL 9 targets are measured. Sections to populate:

- 4.1 Verdict per T1-T9 (PASS / FAIL / FLOOR-TRIGGER)
- 4.2 Headline outcome: did the unification consolidate cleanly?
- 4.3 LOC reduction achieved — actual vs target
- 4.4 Bit-identical reproduction integrity — what (if anything) couldn't reproduce
- 4.5 Paraphrase-mode lessons: did the dispatch pattern absorb a third shape cleanly?
- 4.6 Compound benefits realised:
  - One declarative surface for design-time LLM use across the project
  - Future modes (adversarial, spec-to-corpus, critic) become small dispatch-table additions
  - The "every subsequent experiment uses OQL" pattern from E07's Section 4 extends to LLM-driven workflows
- 4.7 What's NOT done: scaled-corpus re-runs that would lift E12's T4 (orthogonal); a teacher that exceeds the student baseline that would lift E13's T1 (orthogonal); modes beyond the three shipped
- 4.8 Next experiments suggested:
  - **E15** — `paraphrase` mode at scale to test whether semantic-paraphrase augmentation lifts wiring v2's 16/20 ceiling
  - **E16** — `adversarial` mode that probes a target organelle's failure modes
  - **E17** — `spec-to-corpus` mode that reads a BS_/TDD_/FS_ doc and emits a corpus per the spec (compounds with E05's methodology)
- 4.9 Traceability updates (`TRACEABILITY.md`, `ORGANELLE_STATE.md`, `RESEARCH_DISCLOSURE.md`)
