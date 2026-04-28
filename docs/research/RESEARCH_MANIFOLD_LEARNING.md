# Manifold-Learning Composition: A Research Sketch

*Replacing the Wiring Organelle's token-level retrieval with continuous-topology composition built on EnX's `ekan` + `geodesic` + `vr` engines, bootstrapped via chemistry/biology-style structural priors.*

— *MicroGPT-C research note, April 2026 — author: Ajay Soni, Enjector Software Ltd.*

---

## 0. Why this document exists

The Wiring Organelle research arc (`RESEARCH_WIRING_ORGANELLE_PAPER.md`, 17 phases) reaches a characterised structural ceiling at **80% peak / 75% median correct on natural-English tool composition**. Five lever classes — capacity scaling, corpus paraphrasing, structural-diversity templates, multi-organelle re-ranking, multi-seed ensembling — all converge to this band. The Phase 17 diagnostic finding closes the arc:

> *Failures are correlated across model seeds. Different RNG seeds roll different wrong primitive choices for the persistent failure prompts (e.g. #17 fibonacci+factorial+adding rolls* `subtract`*,* `multiply`*,* `min`*,* `fib alone`*,* `fact alone` *— but never converges on* `add`*). The right interpretation has no preferred mass in the model's learned distribution, regardless of seed.*

The architecture is **tight against its design**. Pushing past the ceiling requires moving composition out of *token-level retrieval* into *geometric proximity on a learned manifold*. This document explores the feasibility of that pivot using three engines already shipped in the EnX-cpp project: **`ekan`** (Entangled Kolmogorov-Arnold Network — parametric surface representation), **`geodesic`** (Riemannian solver — geodesic shortest-path on a metric tensor), and **`vr`** (Vietoris-Rips persistent cohomology — topological-feature validator).

The thesis: **EKAN parameterises the manifold; Geodesic measures distance on it; VR validates its topology**. Together they form the substrate for composition-as-geometry, replacing the Wiring Organelle's softmax-over-graph-names with nearest-neighbour-on-a-learned-surface. The deterministic-infrastructure thesis from the Wiring Organelle work (IR + verifier + repair + executor stack) is preserved unchanged — only the candidate-generation front end gets replaced.

This is research-grade scope. The document is a feasibility sketch: what to build, how the existing engines fit, what the bootstrap corpus looks like (chemistry/biology offers a clean template), and what the headline metric should measure. Not an implementation. Not a commitment. A scoped pivot from a saturated ceiling to a categorically different approach.

---

## 1. The diagnosis the Wiring Organelle leaves us with

After 17 phases the Wiring Organelle's failure modes resolve into one diagnostic statement:

> *The model is a retrieval engine. When the held-out prompt sits in a region where the corpus has competing valid retrievals, the model picks one stochastically. The deterministic infrastructure downstream (parser, repair, verifier, executor) cannot recover the* intended *one without a preferred-mass signal — which the architecture cannot provide.*

**Bimodal failures across phases**: every executing graph is either correct on all 5 input sets (5/5) or wrong on all 5 (0/5). No noisy intermediate. The model commits to a topology and then commits to all the right primitives or all the wrong ones consistently. This is a *clustering* signal: outputs collapse into well-separated modes per prompt.

**Correlated failures across seeds**: Phase 17 trained 3 wiring organelles with different RNG seeds, distributing 16 votes round-robin across them. Surface metrics (well-formed, parsed) hit 100% — *some seed always succeeds at structural form*. But correctness held at 70%: the prompts that fail correctness fail across all three seeds. The mode collapse is *seed-invariant*.

The two findings together imply a **manifold structure to the failure modes**:

1. The output space has discrete clusters per prompt (bimodal pattern).
2. The clusters are seed-invariant, meaning they reflect the corpus's geometry, not the model's training noise.
3. Token-level statistical retrieval samples from the union of clusters proportional to corpus mass; when multiple clusters have similar mass, the model picks uniformly.

A statistical learner cannot solve this. A *geometric* learner — one that represents the cluster structure as a continuous surface and resolves ambiguity by spatial proximity to a query embedding — can.

---

## 2. The three EnX engines that make this tractable

The EnX-cpp project ships three engines that, when composed, form exactly the substrate we need. All three are header-first C++17, with public APIs documented and tested.

### 2.1 EKAN — Entangled Kolmogorov-Arnold Network engine

`/Users/user/dev/projects/EnX-cpp/engines/ekan/include/enx/ekan/ekan_engine.hpp`

**What it computes.** EKAN is a learnable parametric activation function engine. Each "edge" in a layer carries either a Fourier-basis function (sinusoidal coefficients `a₀, a₁, b₁, …`) or a B-spline (cubic spline with knots and coefficients), gated by a residual `α` and mixed through an "entanglement" matrix `W`. Unlike a ReLU MLP — which represents *whether* each input matters — EKAN represents *how* each input bends the output. The result is a smooth, differentiable parametric surface whose shape is learned per-edge, not per-neuron.

**Public API surface (the parts that matter for composition):**

```cpp
EntangledKAN(topology, grid_size=5, basis_type);  // parametric surface
predict(input);                                    // forward pass
train(inputs, targets, epochs, lr);                // backprop (MSE)
explain(input);                                    // per-edge attribution
sensitivity(input, epsilon);                        // detect dead zones
save(ostream); load(istream);                      // checkpoints
```

**Why it fits manifold composition.** EKAN is *designed* for parametric surface representation. The spline coefficients directly parameterise smooth curves; the entanglement matrix mixes them into a higher-dimensional surface. We can train EKAN on `(prompt_embedding, graph_embedding)` pairs to learn a *smooth latent surface* on which graph topologies are points and prompts project onto neighbourhoods.

Performance: 5,500-5,900 tokens/sec inference, 3.80 ms forward at 12→64→12 with Fourier basis. Adequate for inference-time composition without GPU.

### 2.2 Geodesic — Riemannian shortest-path solver

`/Users/user/dev/projects/EnX-cpp/engines/geodesic/include/enx/geodesic/geodesic_engine.hpp`

**What it computes.** Given a metric tensor field `G(x)` (which can come from EKAN's learned surface), Geodesic solves the geodesic ODE via 4th-order Runge-Kutta and returns the shortest-path *length* between two points on the manifold. It also reports the path itself (a sequence of intermediate positions), the path-length-vs-Euclidean ratio (curvature), and an optional "gauge work" term for external biasing.

**Public API:**

```cpp
GeodesicSolver<DIMS>(steps, epsilon);
compute_tension(metric_field, deviation, gauge, gauge_weight);  // shortest-path on manifold
compute_euclidean(deviation, gauge, gauge_weight);              // flat-space fallback
compute_batch(metric_field, deviations, gauge, gauge_weight);   // vectorised
christoffel(metric_field, x, k, i, j);                          // inspect curvature
```

Returns a `GeodesicResult<D>` with `tension` (path length), `gauge_work`, `total_risk`, `final_position`, `final_velocity`, `steps_taken`.

**Why it fits.** This is the *distance metric* on the manifold. Given a query prompt projected to a surface point and a candidate graph topology projected to another surface point, Geodesic returns the geodesic distance. **Composition becomes nearest-neighbour retrieval on the learned manifold instead of softmax over a finite graph-name vocabulary.** The "preferred mass" the Wiring Organelle lacks is replaced by *spatial proximity in metric space*.

Performance: 33 ns flat-space (>30M TPS), 1.13 μs RK4-cached at 12D (~880K TPS). The O(D⁵) Christoffel-symbol cost caps practical use at ≤16D — but EKAN can learn an embedding *into* a sub-16D space.

### 2.3 VR — Vietoris-Rips persistent cohomology

`/Users/user/dev/projects/EnX-cpp/engines/vr/include/enx/vr/vr_engine.hpp`

**What it computes.** VR detects *topological features* (connected components β₀, loops β₁, voids β₂) in a point cloud via persistent homology. It builds a Vietoris-Rips flag complex at increasing radii, runs F₂ cohomology reduction, and emits persistent Betti numbers.

**Public API:**

```cpp
VietorisRips<DIMS, MAX_PTS>(max_radius, max_dim);
compute(points, min_persistence);            // returns PersistenceDiagram
betti_numbers(points, at_radius, ...);       // [β₀, β₁, β₂]
```

**Why it fits.** VR is the *topology validator*. After EKAN+Geodesic produce a candidate graph topology for a query, VR can verify the *structural correctness* of the embedding itself: if our manifold is supposed to be a torus (β₁=2, β₂=1), VR confirms it. If it's a DAG, β₁ should be 0. **VR is the second-order Judge** — it doesn't pick candidates but verifies that the manifold's geometry matches the expected topology of valid compositions. In Wiring-Organelle terms, it's analogous to `pipeline_verify()`: a deterministic check that the geometric output is structurally sound.

Performance: 33.9 μs for 8-point 12D persistence, 3,440× faster than GUDHI reference. Practical up to ~30-50 points per query — fine for individual graph validation.

---

## 3. The composition pipeline (proposed)

```
natural-English prompt
        ↓
[1] PROMPT EMBEDDING        — encode prompt as a 12-16D vector
                              (could reuse the existing word-level
                              tokenizer + a small projector, or a
                              dedicated contrastive embedder)
        ↓
[2] EKAN MANIFOLD            — query the learned parametric surface,
    PROJECTION                 returning a metric tensor G(x) at
                              the projected point
        ↓
[3] GEODESIC NEAREST-K       — for each candidate graph anchor in the
                              manifold's vocabulary (~30 template
                              families × ~5 op variants ≈ 150 anchors),
                              compute geodesic distance from the query
                              point. Take top-K (K=8 candidates).
        ↓
[4] VR TOPOLOGY VALIDATION   — verify the local topology of the K
                              candidates matches the expected
                              composition structure (e.g. β₀=K
                              connected components, β₁=0 if no
                              expected cycles).
        ↓
[5] CANDIDATE → @graph        — render each surviving manifold-anchor
                              back to its canonical Pipeline IR text
                              form (lookup by anchor ID; the manifold
                              is parameterised to encode a graph for
                              every point).
        ↓
[6] EXISTING PIPELINE         — strict parse → tolerant parse →
    INFRASTRUCTURE              repair → verify (Wiring Organelle's
                                deterministic stack, unchanged)
        ↓
[7] EXECUTE                  — pipeline_execute() with native C
                              dispatch (unchanged)
        ↓
numeric answer  →  compared against canonical reference on 5 input sets
```

**Key invariant**: the entire downstream half (steps 5-7) is *byte-identical* to the v2.0 Wiring Organelle infrastructure. Steps 1-4 replace the wiring + planner organelles with a geometric composition module. The deterministic-infrastructure thesis is preserved.

---

## 4. The training/learning recipe

The wiring organelle's training was supervised on `(prompt, @graph_text)` pairs. Manifold-learning composition needs three different learnings:

### 4.1 Anchor learning (offline, one-time)

Each known graph topology becomes an *anchor point* on the EKAN manifold:

```
T = { tpl_chain, tpl_fib_fact_op, tpl_distance_midpoint,
      seed_compound_interest, seed_bmi_classified, … }
```

For ~30 template families × ~5 op variants ≈ 150 distinct anchors. Each anchor gets:
- A canonical `@graph` string (the existing corpus already has these)
- An anchor coordinate `a_t ∈ ℝ¹²` on the manifold (initialised randomly, refined by training)

### 4.2 Embedder training (the contrastive objective)

For each `(prompt_p, anchor_t)` pair from the existing 408-example corpus, train an embedding network `E(prompt) → ℝ¹²` so that:

- `geodesic_distance(E(prompt_p), a_t) → small` for the correct anchor
- `geodesic_distance(E(prompt_p), a_t') → large` for incorrect anchors (negative samples)

This is **contrastive learning on the manifold**: positives pull toward their anchor, negatives push away, but distance is measured by Geodesic (not by raw Euclidean), so the loss respects the EKAN-learned surface curvature.

EKAN itself is trained jointly: its parametric surface adjusts so that anchors at clearly distinct compositions are far apart geodesically, while anchors at semantically similar compositions (e.g. `fib_fact_op_add` and `fib_fact_op_multiply`) cluster nearby (separated only along the "op-choice" axis).

### 4.3 Manifold geometry priors (the bootstrap)

The corpus alone has only 408 examples. To learn a 12D parametric surface that generalises, we need *structural priors* — predictable relationships between anchors that the manifold must respect. Two domains offer clean templates:

#### 4.3.1 Chemistry: molecule embeddings as composition prior

Molecules are tool compositions in disguise. A molecular structure has:

- **Atoms** (analogous to primitives like `add`, `bmi`, `compound`)
- **Bonds** (analogous to edges connecting node outputs to node inputs)
- **Functional groups** (analogous to template families like `tpl_compound_then`)
- **Reactions** (analogous to graph composition: combining sub-graphs into wholes)

Chemistry has well-known *geometric* structure: SMILES strings → fingerprints → manifold (e.g. ChEMBL's chemical-space-embedding work). The manifold has well-defined *similarity* (Tanimoto, MCS) and *substitution* (one functional group swappable for another at known geodesic distance).

We can bootstrap by:
1. Treating each `tpl_*` family as a "scaffold" (analogous to a chemical scaffold).
2. Treating op variants within a family as "substituents" (analogous to functional-group substitution).
3. Pre-training EKAN on a synthetic chemistry-like corpus where scaffold-distance and substituent-distance are known by construction.
4. Fine-tuning on the 408 (prompt, graph) pairs so prompts project to the right scaffold + substituent.

This gives the manifold a *correct geometry* from a much larger corpus before the small Wiring corpus refines it.

#### 4.3.2 Biology: pathway composition prior

Metabolic pathways and gene-regulatory networks are *real* graph compositions in nature. KEGG and Reactome have ~10,000 enzyme→reaction→product graphs that decompose into:

- **Enzymes** (primitives)
- **Substrates / products** (input/output ports)
- **Pathways** (template families)
- **Cross-pathway compositions** (multi-organelle equivalents)

A pre-trained "biology-pathway manifold" (each pathway → a coordinate, similar pathways nearby, distinct pathways far) gives the same prior structure as chemistry but with a richer compositional grammar (cycles, feedback loops — directly testable with VR's β₁ validation).

#### 4.3.3 Why bootstrap from a different domain?

The Wiring Organelle's mode collapse on prompts like #17 ("fibonacci of n combined with factorial of n by adding") happens because the corpus has no structural prior to disambiguate between the 5 ops in `tpl_fib_fact_op`. A chemistry-bootstrapped manifold *already knows* that operator substitution is a one-dimensional axis of variation (like a methyl→ethyl substitution in a scaffold). The "right" interpretation of #17 then becomes a small directed nudge along that axis — the kind of resolution that geometric proximity supports natively but token-level statistical learning cannot.

### 4.4 Inference

At inference time:

1. Compute `q = E(prompt)` — project prompt to the manifold (tiny EKAN forward pass, ~ms).
2. For each anchor `a_t`, compute `d_t = geodesic_distance(q, a_t)` via Geodesic's RK4 (1.13 μs each at 12D × 150 anchors ≈ 0.2 ms total).
3. Take the K=8 nearest anchors.
4. Run VR on the K candidates' embeddings to verify expected topology (33.9 μs).
5. For each surviving candidate, look up the canonical `@graph` text and feed to Pipeline IR's existing parse/repair/verify/execute stack.
6. Self-consistency vote across the 8 results' 5-input output vectors (Wiring Organelle Phase 8 logic, unchanged).

Total latency: <1 ms per query for steps 1-4, dominated by parse/verify/execute downstream. Vastly faster than the current best-of-16 transformer sampling.

---

## 5. Why this should beat the 75% ceiling

The Wiring Organelle's failures all share one structural feature: **competing valid retrievals with no preferred-mass signal**. The candidates exist in the model's learned distribution; the model just can't tell which one the prompt asked for.

Manifold-learning composition resolves this in three ways:

1. **Geodesic distance is single-valued.** A query embedding has *exactly one* nearest anchor. There is no uniform-random sampling over equally-likely candidates because there is no softmax — there is only metric proximity.

2. **The learned surface encodes corpus geometry.** During training, anchors that the corpus pulls in different directions end up at *different geodesic neighbourhoods*, not the same softmax bucket. The "preferred mass" the Wiring Organelle lacks is replaced by *spatial separation*.

3. **VR catches manifold corruption.** If the embedding network drifts to put two distinct compositions on top of each other (collapsing the metric), VR's β₀ count would fall below the expected number of clusters — flagging the failure mode at training time, before deployment.

The bimodal-failure pattern from Wiring Organelle Phase 8 becomes a *feature*: the manifold learning is *built around* the assumption that valid compositions cluster discretely. We're working with that geometry, not against it.

### Predicted headline lift

The Wiring Organelle's 5 persistent failures:

| # | Prompt | Wiring failure | Manifold-composition fix |
|---|---|---|---|
| 1 | "body mass index … limit it inside" | mode collapse — diffuse prior | distinct anchor for `seed_bmi_classified`; "limit it inside" projects to its neighbourhood |
| 2 | "interest gained on an investment when principal compounds…" | mode collapse | similar anchor for `seed_compound_interest`; "interest gained" projects there |
| 3 | "weighted combination of three measurements…" | reference mismatch (model emits `multiply→add→divide`; reference expects `multiply→add→percentage`) | distinct anchors for the two interpretations; query projects to whichever the embedder was trained for |
| 6 | "take home pay from gross income…" | primitive drift to percentage-style graph | distinct anchor for `seed_net_pay`; embedder learns "take home pay" → that anchor |
| 17 | "fibonacci of n combined with factorial of n by adding" | uniform-random over 5 op variants in `tpl_fib_fact_op` | the 5 op-variants are 5 distinct anchor points along a one-dimensional substituent axis; "by adding" projects to the `add` neighbourhood by geodesic proximity |

If all 5 close, the headline lifts from **75% median** (Wiring Organelle) to **100% on this 20-prompt held-out**. Realistically, prompt-embedder training will have its own variance, so the ceiling is probably **~90%** with the variance band shifted up rather than collapsed entirely.

---

## 6. Build feasibility

The three EnX engines are header-first C++17 with stable APIs. Wiring them into MicroGPT-C requires:

### 6.1 Source code

- A new module `src/microgpt_manifold.{h,c}` (or `.cpp` if we accept C++17 in this module — TBD with the user; the rest of MicroGPT-C is C99) that wraps:
  - EKAN for the parametric surface
  - Geodesic for distance computation
  - VR for topology validation
  - An anchor table (`AnchorRegistry`) mapping anchor IDs ↔ canonical `@graph` strings
  - An embedder network (could itself be EKAN, or a small dedicated network)

- A new tool `tools/pipeline_manifold_corpus_gen.c` analogous to the existing `pipeline_corpus_gen.c` — emits `(prompt, anchor_id, graph_text)` triples for training the embedder.

- A new demo `demos/manifold_organelle/main.c` that:
  - Trains the embedder + EKAN jointly on the 408 corpus pairs
  - Optionally bootstraps from a chemistry/biology pretraining corpus
  - At eval, runs the manifold composition pipeline end-to-end
  - Reports the same metrics the Wiring Organelle does (well-formed, parsed, verified, executed, correct on all 5 inputs)

### 6.2 Build system

EnX-cpp is a sibling project, not a submodule. Two integration choices:

- **Vendor**: copy the relevant headers into `vendor/enx/` in MicroGPT-C. Pinned to a specific commit. Header-only, no build complications. Recommended for a research sketch.
- **Submodule**: add `EnX-cpp` as a Git submodule under `external/enx/`. Cleaner long-term but couples the build systems.

CMakeLists.txt adds an `add_demo(NAME manifold_organelle ...)` block analogous to `wiring_organelle_demo`.

### 6.3 Performance budget

- EKAN forward pass: 3.80 ms per query (Fourier basis, 12→64→12).
- Geodesic batch over 150 anchors: ~0.2 ms.
- VR validation on 8 candidates: 33.9 μs.
- Pipeline IR parse + repair + verify + execute: <1 ms.
- **Total: ~5 ms per query** — vastly faster than the Wiring Organelle's 16 transformer samples.

Training time is the unknown. EKAN training is currently 816 tokens/sec. A 408-pair corpus × 1000 epochs = ~500K examples ÷ 816 = ~10 min. Plus contrastive negative sampling (~3× the cost). **Estimated 30-60 min wall-clock training**, broadly comparable to the Wiring Organelle's 50-min total.

### 6.4 Estimated effort

| Task | Effort |
|---|---|
| Vendor EnX-cpp headers, wire into CMakeLists | 1 day |
| Anchor registry + corpus generator | 1 day |
| Embedder network + contrastive training loop | 3-5 days |
| EKAN+Geodesic+VR composition pipeline | 2-3 days |
| Demo + held-out eval + metrics reporting | 1-2 days |
| Chemistry-bootstrap corpus + pretraining | 5-7 days (if pursued) |
| Documentation + writeup | 2-3 days |

**Total: 2-3 weeks single-developer effort** for the Phase 1 (no chemistry bootstrap), 4-6 weeks with the chemistry bootstrap. That's a meaningful but bounded research investment, well-scoped because all three EnX engines already exist and are tested.

---

## 7. Open questions

1. **C99 vs C++17 mixing.** EnX-cpp is C++17. MicroGPT-C is strict C99. If we vendor EnX-cpp into the manifold module, that module is C++17. The IR + verifier + executor stack remains C99. Whether this two-language split is acceptable is a project-policy decision.

2. **Embedder architecture.** Option A: reuse the existing word-level transformer as a feature extractor, then project to 12D via a small MLP. Option B: a dedicated EKAN-as-embedder, end-to-end on the manifold. Option B is cleaner but more research-novel; A is faster to prototype.

3. **Chemistry-bootstrap dataset.** ChEMBL has ~2M molecules with computed fingerprints. RDKit can compute Tanimoto distances and scaffold decompositions. The bootstrap corpus would need ~10k-100k molecule pairs to teach the manifold structure. Sourcing and licensing of the chemistry data is a real cost.

4. **Anchor count scaling.** 150 anchors are tractable for nearest-neighbour search. For real tool libraries (e.g. all of `w_vm_functions.txt` ≈ 192 primitives × multiple compositions each), we'd need 1000-10000 anchors. Geodesic's O(D⁵) Christoffel cost rules out high-dim, but k-d-tree / FAISS-like indexing in 12D handles 100k+ points in <1ms — we'd need to add an index layer.

5. **VR's role in training vs inference.** Phase 1 uses VR only at inference (topology validation). It could also drive training: a regulariser that penalises embeddings whose Betti numbers deviate from expected values at expected radii. This makes the training objective topology-aware. Powerful but expensive.

6. **Failure mode of the new pipeline.** The Wiring Organelle's bimodal failure has a clean replacement. But the new pipeline has *its own* failure mode: if the embedder projects a query to the wrong neighbourhood, no nearest-neighbour retrieval can recover. Phase 2's research would need to characterise these failures.

---

## 8. Bootstrapping strategy: why chemistry first

If we're going to invest in a manifold-learning composition module, the bootstrap matters more than the architecture. The Wiring Organelle's lesson was that 408 examples are enough for token-level retrieval to saturate but *not enough for compositional generalisation*. Manifold learning is even more data-hungry — geometric structure needs many examples to be inferred.

Chemistry is the right bootstrap for three reasons:

1. **Massive corpus, ground-truth geometry.** ChEMBL: 2M+ molecules. Each molecule has computed fingerprints, so similarity is *known by construction*. We can train EKAN on (molecule_a, molecule_b, tanimoto_distance) triples and verify that the learned manifold's geodesic distances match Tanimoto.

2. **Compositional structure is explicit.** SMILES → atoms → bonds → functional groups → scaffolds → reactions. The hierarchy maps directly to (primitives → ports → templates → composed graphs → execution). The learned manifold has the *right shape* for tool composition because chemistry already lives on a manifold of that shape.

3. **VR finds known features.** Aromatic rings (β₁=1), fused bicyclic systems (β₁=2), polyhedral cages (β₂≥1) are *known* topological features of molecular graphs. Training VR's expected-feature thresholds against ChEMBL gives us a calibrated topology-validation regime *before* we apply it to Pipeline IR graphs.

Biology (KEGG/Reactome) is the second-order bootstrap: pathway compositions add cycles and feedback loops, exercising compositional patterns chemistry alone doesn't have. Run chemistry first to get the geometry; layer biology on top to get the composition grammar.

---

## 9. Where this fits in MicroGPT-C's research arc

This isn't a "next step" of the Wiring Organelle — it's a *categorically different* approach that the Wiring Organelle's ceiling diagnosis points at. The relationship:

| Layer | Wiring Organelle (v2.0) | Manifold-Learning Composition (research) |
|---|---|---|
| Front end | 540K-param wiring transformer + 540K-param planner, best-of-16 sampling | EKAN parametric surface + embedder network + Geodesic nearest-K + VR topology validator |
| Candidate set | 16 sampled `@graph` strings | Top-K=8 anchor points retrieved by geodesic proximity |
| Re-ranking | Self-consistency vote + planner-family-bonus | Self-consistency vote on candidate execution outputs (unchanged) |
| Downstream | Parse → repair → verify → execute | **Identical**: parse → repair → verify → execute |
| Headline ceiling | 75-80% correct on natural-English | **Predicted 85-95%** if the bootstrap is sufficient |
| Training data | 408 (prompt, graph) pairs | Same 408 pairs + chemistry-bootstrap pretraining (~10-100k molecule pairs) |
| Compute footprint | ~50 min training, ~1 min inference / 20 prompts | ~30-60 min training (incl. pretraining), ~100 ms inference / 20 prompts |
| Dependencies | Pure C99, libc + libm | C99 core + C++17 manifold module (EnX-cpp headers) |

The deterministic-infrastructure thesis ("organelles retrieve; pipelines compose; the Judge guarantees correctness") **strengthens** here: composition itself moves from finite-corpus retrieval (Wiring Organelle) to continuous-manifold geometry (Manifold-Learning Composition). The Judge (verifier) is unchanged. The *retrieval mechanism* changes, but the architecture's commitment to deterministic verification of every emitted graph is preserved.

---

## 10. Recommendation

**Build a Phase-1 prototype** (without chemistry bootstrap, single-developer, 2-3 weeks):

1. Vendor EnX-cpp's `ekan/include`, `geodesic/include`, `vr/include` into `vendor/enx/` of MicroGPT-C.
2. Write `src/microgpt_manifold.{h,cpp}` implementing the composition pipeline.
3. Reuse the existing 408-example Wiring Organelle corpus; emit `(prompt, anchor_id)` pairs.
4. Train the embedder + EKAN jointly with contrastive loss using Geodesic distances.
5. Eval on the existing 20 held-out NL prompts. Measure the same metrics.

**If Phase 1 lifts the headline above 80% without the chemistry bootstrap**, that's a strong signal — the bottleneck was the architecture, not the corpus, and manifold composition resolved it. Phase 2 then adds chemistry-bootstrap training to push higher.

**If Phase 1 is no better than 75%**, the bottleneck is something deeper than retrieval-vs-geometry, and we need the chemistry bootstrap before drawing any further conclusion. That escalates effort to 4-6 weeks.

Either outcome is publishable — a positive result validates the manifold-composition thesis; a null result tightens the diagnosis from "structural ceiling at 75% for token-level retrieval" to "structural ceiling at 75% for any architecture trained on 408 examples without external geometric priors", which is a stronger claim about the data regime.

This document is a feasibility sketch. Implementation, if pursued, is a separate project.

---

## 11. References within this repository

- `docs/research/RESEARCH_WIRING_ORGANELLE_PAPER.md` — the standalone paper for the v2.0 Wiring Organelle, including the §16 manifold-learning forward pointer that this document expands.
- `docs/research/RESEARCH_PIPELINE_IR.md` — the 31-section development log with all 17 phases, 5 documented negative results, and the variance characterisation that diagnoses the structural ceiling.
- `src/microgpt_pipeline.{h,c}` — the IR + verifier + repair + executor stack that manifold composition would reuse unchanged.
- `tools/pipeline_corpus_gen.c` — the corpus generator producing the 408-example training set; would extend to also emit `(prompt, anchor_id)` triples.

## 12. References to EnX-cpp

- `engines/ekan/include/enx/ekan/ekan_engine.hpp` — Entangled KAN engine, lines 1142-1210 for main API surface.
- `engines/geodesic/include/enx/geodesic/geodesic_engine.hpp` — Riemannian solver, lines 434-640 for `GeodesicSolver`.
- `engines/vr/include/enx/vr/vr_engine.hpp` — Vietoris-Rips persistent cohomology, lines 190-256 for `VietorisRips`.
- `engines/ekan/README.md`, `engines/geodesic/README.md`, `engines/vr/README.md` — public-facing engine docs.

---

*This is a research sketch, not an implementation plan. The Wiring Organelle is shipped at v2.0; manifold-learning composition is the next research direction the v2.0 ceiling characterisation points at. Whether to pursue it is a research-program decision.*

— Ajay Soni, Enjector Software Ltd. April 2026.
