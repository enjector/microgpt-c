# TDD_wiring — Technical Design Document (Wiring Organelle + Anchor Retrieval)

**Document ID:** TDD-WIRE-001
**Version:** 1.0
**Status:** DRAFT
**Paired BS:** `BS_wiring.md`
**Sources:** `demos/wiring_organelle/`, `demos/manifold_classifier/`, `tools/pipeline_corpus_gen.c`, `tools/corpus_expand.c`, `tools/scaling_leakage_audit.sh`. Research log: `docs/research/RESEARCH_PIPELINE_IR.md`.

## 1. Overview

The **Wiring Organelle** is a 540K-parameter word-level transformer trained to emit pipeline-IR `@graph` text from natural-language prompts. It is augmented by:

- A **best-of-N + verify-as-Judge** loop that re-ranks N candidate graphs by `pipeline_verify` success and domain heuristics.
- **Native primitive registration** via `wiring_natives.{h,c}` (≥ 40 primitives).
- A **reference-answer suite** (`wiring_references.{h,c}`).
- A **planner organelle** (Phase 15) that emits a re-ranking hint based on the prompt.
- A **manifold-retrieval anchor table**: 20 canonical `@graph` entries indexed by domain keywords, looked up via either a 20-D Geodesic top-1 classifier (`wiring_geo_classifier.{h,c}`, Phase 2c) or a TF-IDF centroid classifier (Phase 4 — `manifold_tfidf_demo`).
- A **composition fallback** (`wiring_compose_for_prompt`, Phase 3b) that chains top-2/3 fragments by output→input linkage when a multi-stage prompt has no matching anchor.

The honest, leakage-audited headlines (per `RESEARCH_PIPELINE_IR.md` §38, §41, §43, §46):

- Anchor-retrieval mechanism on the leakage-free Phase 2c paraphrases — **100 % (20/20)**.
- Wiring transformer alone on the same clean set — **35 % (7/20)**.
- Phase 3b composition multi-stage prompts — **60 % (6/10)**.
- Phase 4 TF-IDF on adversarial axis-2 — **90 % (18/20)**, with 100 % (20/20) no-regression.

## 2. Architecture

```
                NL prompt
                    │
         ┌──────────┼─────────────────────────────────────┐
         ▼                                                ▼
   anchor classifier (Geodesic / TF-IDF)         planner organelle (Phase 15)
         │                                                │
         ▼                                                ▼
   anchor table lookup                                hint string
         │                                                │
         ▼                                                │
   candidate@graph (or NULL)                              │
                                                          │
                  best-of-N ──── wiring transformer ◄─────┘
                  candidates                               
                          │
                          ▼
              pipeline_parse_text_tolerant
                          │
                          ▼
                 pipeline_repair (drop dead fragments)
                          │
                          ▼
                 pipeline_verify (Judge)
                          │
                          ▼
                  ranking + selection
                          │
            ┌─────────────┴─────────────┐
            ▼                           ▼
   anchor candidate (if good)   wiring_compose_for_prompt
                                (multi-stage fallback)
                                              │
                                              ▼
                                final @graph (verified)
```

## 3. Data flow

`wiring_organelle_demo` runs the prompt through, in order:

1. Anchor classifier picks the best anchor; if it matches the prompt with high confidence, the anchor's @graph is the leading candidate (the **anchor-retrieval mechanism**).
2. The wiring transformer generates N candidate @graphs (typically N=16) at low temperature with prefix-cache sharing.
3. Each candidate is parsed (tolerant), repaired, and verified.
4. Candidates that verify are re-ranked by:
   - Match against the planner organelle's hint (Phase 15).
   - Composition fragment match for multi-stage prompts (Phase 3b).
   - A "fidelity-trumps" gate that gives a +1000 score boost when the composition has the expected primitive set and no anchor does.
5. The top-ranked verified candidate is returned. If none verify, return an empty graph and a non-zero status code.

## 4. Key data structures

### 4.1 Anchor library

```c
typedef struct {
  const char *family_name;
  const char *keyword_set[K];
  const char *graph_text;
} WiringAnchor;
```

20 entries shipping in `wiring_anchor_graphs.{h,c}`. The keyword sets are used by the TF-IDF classifier; the Geodesic classifier additionally uses a hand-coded "embedder" that maps each entry to a unique slot in 20-D state space.

### 4.2 Geodesic classifier slot table

`GEO_DIMS = 40` (this fork; the sibling fraud-detection code uses 12). Each anchor family is assigned a unique axis. The classifier embeds a prompt by tallying keyword hits along each axis, then runs `geo_compute_tension` (or Euclidean fallback) to find the axis of minimum geodesic distance to the origin → top-1 anchor.

### 4.3 TF-IDF classifier

A standard centroid classifier:
- Term frequency vectors of all training prompts in a family.
- Centroid per family.
- Inverse document frequency weighting across the corpus.
- Cosine similarity to centroids; argmax → predicted family.

The corpus-expand pipeline (`tools/corpus_expand.c`) generates 4,102 prompts deterministically from per-family synonym tables and sentence templates. The classifier trained on this expanded corpus achieves the Phase 4 SLO.

## 5. Algorithms

### 5.1 Best-of-N generation with prefix cache

The prompt is tokenised once and processed through the model with a single prefix forward pass. The KV cache is then **copied** (`kv_cache_copy`) per vote, and N votes diverge from the same starting state with temperature jitter. This is the same prefix-cache-sharing trick documented in `RESEARCH_SSD.md` and gives a 1.9×–5.7× speedup vs naive independent ensemble.

### 5.2 Verify-as-Judge

Each candidate goes through:
- `pipeline_parse_text_tolerant(src)` — accept three named repairs.
- `pipeline_repair(p, &report)` — drop dead fragments.
- `pipeline_verify(p)` — full check; failure disqualifies the candidate.

Re-ranking signals (in addition to verify pass/fail):
- Edit distance to the planner's hint @graph_name.
- Match between the candidate's primitive set and the prompt's distinctive nouns (per the `RESEARCH_WIRING_ORGANELLE_PAPER.md` "Phase 15 graded family-match" rubric).
- Fidelity-trumps: +1000 to a composition that has the expected primitive set when no anchor does (Phase 3b §43).

### 5.3 Composition fallback (Phase 3b)

`wiring_compose_for_prompt(prompt, ...)`:

1. Decompose existing anchors into 15 reusable fragments (`wiring_fragments.{h,c}`).
2. Pick top-2 / top-3 fragments by keyword hits in the prompt.
3. Chain them by output→input linkage (the type system enforces compatibility).

Achieves 60 % (6/10) on the multi-stage composition test set without disturbing the single-anchor 100 % headline (the fidelity-trumps gate is what arbitrates between the two paths).

## 6. CLI flag matrix

The demo exposes flags that reproduce the leakage-audited honest baselines:

| Flag | What runs | Reproduces |
|---|---|---|
| `--clean-only` | Skip the first 20 leaked prompts; evaluate on Phase 2c clean only | Honest restated headlines |
| `--no-anchor` | Disable anchor classifier; wiring transformer alone | 35 % wiring-only baseline |
| `--composition` | Evaluate against `pipeline_corpus_composition_test.txt` | Phase 3b §43 6/10 |
| `--no-composition` | Disable composition fallback | Confirms anchor-retrieval headline is unaffected |

The Phase 4 reproduction is via `manifold_tfidf_demo`:
```
./corpus_expand pipeline_corpus_phase4_train.txt 42
./manifold_tfidf_demo pipeline_corpus_adversarial.txt pipeline_corpus_phase4_train.txt
```

## 7. Concurrency model

The wiring demo runs single-threaded: training is one-shot, inference uses ensemble vote with shared prefix cache (no thread-level parallelism beyond what the core engine's optional `MICROGPT_HEAD_PARALLEL` provides). The anchor / TF-IDF classifiers are read-only after construction.

## 8. Trade-offs considered

| Decision | Chosen | Rejected | Rationale |
|---|---|---|---|
| Generation vs retrieval | Hybrid: anchor first, generate as fallback | Pure autoregressive generation | Generation alone hit 35 % on the clean set; retrieval-first achieves 100 %. |
| Classifier | Geodesic (Phase 2c) + TF-IDF (Phase 4) | Learned encoder (EKAN-Network, Phase 3a-full) | Phase 3a falsified the learned encoder hypothesis at the 408-example corpus scale (4/20 vs predicted 12–16/20). Per `§40.7` skip rule, simpler classifiers were preferred. |
| Corpus expansion | Synthetic synonym + template generator | Hand-written 5K paraphrases | Deterministic, regenerable, leakage-checkable. The `tools/scaling_leakage_audit.sh` script enforces non-overlap with held-out. |
| Composition signal | Fidelity-trumps gate (+1000) | Soft re-ranking | Boundary between "anchor matches" and "composition needed" was unstable under continuous scores; the +1000 gate enforces a hard tier ordering. |
| Held-out reporting | Always cite Phase 2c clean (leakage-free) | Cite the original 20-prompt set | The original set had 13/20 verbatim leaks (introduced by Phase 13's lexical-anchoring expansion). Honest reporting requires the audited set. |

## 9. Known limitations

- The 540K-param transformer alone is **not** the load-bearing piece — the anchor table is. A clean-room rebuild that re-implements only the transformer will achieve ≈ 35 % on the clean set, not 100 %.
- The anchor table is curated; scaling to "100s of families across all domains" requires external semantic embeddings or a much larger anchor library (per Post-Phase-3 #3 in `STRATEGY_ONE_PAGER.md`).
- The known regression in the wiring binary's vote loop is rolled back surgically; the proper fix is documented as future work in the strategy one-pager.
- The composition fallback handles two-stage and three-stage chains only; deeper chains are tested but not optimised.

## 10. References

- `docs/research/RESEARCH_PIPELINE_IR.md` — full 17-phase development log + leakage audit.
- `docs/research/RESEARCH_WIRING_ORGANELLE_PAPER.md` — paper v2.0.
- `docs/research/RESEARCH_MANIFOLD_LEARNING.md` — manifold-retrieval composition.
- `docs/research/wiring_scaling_post_phase3.md` — honest scaling-curve closure.

## 6. Type-directed compositional search (V1.0.4)

A new mechanism (Stream B of the compositional generator fix) synthesises pipelines from the primitive manifest without invoking the wiring transformer or anchor table.

### 6.1 Inputs / outputs

- `wiring_compositional_search(prompt, &report)` — returns a verified `Pipeline *` or NULL.
- `wiring_compositional_search_render(prompt, &pipeline_out, &report)` — renders the verified graph as `@graph` text.

### 6.2 Algorithm (V1.0.4 — greedy beam=1)

1. Lower-case the prompt and normalise hyphens to spaces.
2. Score every primitive in `wiring_primitive_manifest` by counting case-insensitive whole-word keyword hits (with a simple plural-`s` tolerance).
3. Pick the highest-scoring primitive whose output type matches `PIPE_T_INT` (the V1 default) as the **outer** node.
4. For each input port of the outer, search the manifest (excluding the outer's name) for the highest-scoring primitive whose output type matches that port's input type — this is the **inner** node for that port. If no inner has a positive score, the input becomes a fresh signature input.
5. De-duplicate inners that the search picked for multiple ports — keep the one on the highest-scoring port; the others fall back to signature inputs.
6. Build the graph: signature inputs first, then inner nodes (each with its full input arity bound to fresh signature inputs), then the outer node connected to inners' outputs and any remaining signature inputs. Connect the outer's output to a single signature output `y`.
7. Run `pipeline_verify(p)`. Return the verified pipeline or NULL on any failure.

### 6.3 Reuse

- `pipeline_create / _add_node / _connect / _connect_signature_in / _connect_signature_out / _set_signature` for graph construction.
- `pipeline_verify` as the deterministic Judge.
- `pipeline_render_text` for the text emission convenience entry point.
- `wiring_primitive_manifest` for the primitive catalogue.

### 6.4 Limitations

- Beam width 1 — the greedy heuristic favours manifest-order ties.
- Inner recursion depth 1 — graphs are at most 3 nodes (one outer + two inners) in practice; deeper compositions require the existing anchor / fragment paths.
- Output type fixed at `PIPE_T_INT`; FLOAT pipelines need a small extension to thread `desired_output_type` through the search.

A future revision (Phase 6) widens the beam, adds depth >1 recursion, and lets the planner organelle re-rank across mechanisms (1)–(4).

## 11. Revision history

| Version | Date | Change |
|---|---|---|
| 1.0 | 2026-04-30 | Initial extraction. |
| 1.0.4 | 2026-04-30 | Added §6 — type-directed compositional search. |
