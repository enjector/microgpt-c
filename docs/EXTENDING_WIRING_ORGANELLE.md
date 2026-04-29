# Extending the Wiring Organelle

*A practical guide for the engineering-mode work that follows the closed research arc. The why is in `docs/research/RESEARCH_PIPELINE_IR.md` and `RESEARCH_WIRING_ORGANELLE_PAPER.md`; this is the how.*

---

## What you can extend

The Wiring Organelle has four working retrieval mechanisms layered on top of the same deterministic Judge stack (Pipeline IR + verifier + repair + executor). Each mechanism has its own extension surface:

| Mechanism | Headline | Extension surface | File |
|---|---|---|---|
| **Anchor retrieval** | 100% on novel paraphrases | Add a canonical `@graph` DAG + a unique 20D-slot keyword bag entry | `wiring_anchor_graphs.c`, `wiring_geo_classifier.c` |
| **Fragment composition** | 60% on multi-stage chains | Add a fragment entry (primitive + arg names + keyword bag) | `wiring_fragments.c` |
| **TF-IDF on Phase 4 corpus** | 90% on adversarial paraphrases | Add synonyms to a family's per-concept-group synonym list | `tools/corpus_expand.c` |
| **Wiring transformer** | 35% on novel English | (don't extend — its 35% ceiling is documented through 17 phases) | `demos/wiring_organelle/main.c` |

**Axis 1 (novel families)** is the only genuinely-open boundary. Every other axis is "soft-closed" within the curator's coverage.

---

## Recipe 1 — Add a new reference family

A new family is the largest extension unit. Walk-through example: adding a `geometric_mean` family (`geo_mean(a, b, c) = cube_root(a * b * c)`).

### Step 1.1 — Pick a family name and its reference function

Edit `demos/wiring_organelle/wiring_references.c`:

```c
/* New family: cube_root(a * b * c) — geometric mean of three values.
 * S[0]=a, S[1]=b, S[2]=c */
DEF_REF(geometric_mean) {
    int64_t prod = S[0] * S[1] * S[2];
    /* Cube root via integer Newton's method (or use a native if added). */
    int64_t x = (int64_t)cbrt((double)prod);
    return x;
}
```

Register it in the `references[]` table at the bottom of the file:

```c
    {"geometric_mean", ref_geometric_mean},
```

### Step 1.2 — Add the canonical anchor `@graph`

Edit `demos/wiring_organelle/wiring_anchor_graphs.c`. Append an entry to `ANCHORS[]`:

```c
{ "geometric_mean",
  "@graph geometric_mean\n"
  "  : in a -> int\n"
  "  : in b -> int\n"
  "  : in c -> int\n"
  "  : out y -> int\n"
  "  | p1 = multiply(x: <a>, y: <b>) :: x:int, y:int -> out:int\n"
  "  | p2 = multiply(x: p1.out, y: <c>) :: x:int, y:int -> out:int\n"
  "  | r = cube_root(x: p2.out) :: x:int -> out:int\n"
  "  y <- r.out\n"
  "@end\n"
},
```

Make sure every primitive used (`multiply`, `cube_root`, …) is in `wiring_natives.c`'s registry. If `cube_root` isn't, add it as a native first (~5 lines).

### Step 1.3 — Add the family to the geodesic classifier

Edit `demos/wiring_organelle/wiring_geo_classifier.c`. Append to `FAMILIES[]`:

```c
{ "geometric_mean", 20, { "geometric", "mean", "cube", "root", "third-root", NULL } },
```

The number `20` is the unique slot — bump `GEO_DIMS` in `src/microgpt_geodesic.h` if you exceed the existing axis count. The keyword bag should be tight: 3–8 keywords, lexically distinct from every other family's bag.

### Step 1.4 — Mirror the classifier table in the diagnostic demo

Edit `demos/manifold_classifier/main.c`'s `FAMILIES[]` table to match. Keep the two in sync; otherwise the Phase 1b diagnostic and the wiring eval disagree on what families exist.

### Step 1.5 — Add a held-out test prompt for the new family

Edit `demos/wiring_organelle/pipeline_corpus_held_out.txt` (or a new test file):

```
# EXPECTED: multiply cube_root
# REFERENCE: geometric_mean
// the geometric mean of three values
---
```

### Step 1.6 — Verify

```bash
cmake --build build --target wiring_organelle_demo manifold_classifier_demo
./build/manifold_classifier_demo  # confirm new family classifies
./build/wiring_organelle_demo --clean-only
```

Expected: the new family appears in the classifier output; the wiring eval anchor pick-rate stays 100% on the existing prompts and includes the new prompt at the same rate.

### Step 1.7 — Add fragment entries (optional, if the family decomposes)

If `geometric_mean` is the cube-root composed with multiply, it can also serve as a fragment for richer compositions like "the geometric mean clamped between lo and hi". Add to `wiring_fragments.c`:

```c
{ "geometric_mean_step",
  { "geometric mean", "geo-mean", "cube-rooted product", NULL },
  "_geo_mean_3",  /* composer special-case (multiply chain + cube_root) */
  { "a", "b", "c" }, 3, 0 },
```

Then add a special-case emission in `emit_fragment()` for `_geo_mean_3` (~15 lines, mirroring the existing `_fib_fact_mul` special case).

### Step 1.8 — Extend the Phase 4 synonym table

Edit `tools/corpus_expand.c`. Append a new `Family` entry to `FAMILIES[]`:

```c
{ "geometric_mean",
  {
    { "geometric mean", "geo-mean", "cube-rooted product", "third-root of product", NULL },
    { "three values", "three numbers", "trio of inputs", NULL },
    { NULL },
    { NULL },
  },
  {
    "%0% of %1%",
    "the %0% taken from %1%",
    NULL
  } },
```

Regenerate the expanded corpus:

```bash
./build/corpus_expand build/pipeline_corpus_phase4_train.txt 42
./build/manifold_tfidf_demo build/pipeline_corpus_held_out.txt build/pipeline_corpus_phase4_train.txt
```

Confirm the new family classifies at the TF-IDF layer.

### Step 1.9 — Run the leakage check

```bash
./tools/check_held_out_leakage.sh build/
```

If your synonym tables accidentally produced the held-out test prompt verbatim, the check will flag it and you should either rephrase the test prompt or trim the synonym list.

### Time budget for adding a family

- Steps 1.1–1.6 (single-anchor capability): **~30 minutes**
- Step 1.7 (composition support): **+10 minutes**
- Step 1.8 (Phase 4 synonyms for adversarial robustness): **+10 minutes per concept group**

Linear in the number of families you want to support.

---

## Recipe 2 — Add a fragment to `wiring_fragments.c`

A fragment is a sub-DAG (typically 1–3 primitives) that can be chained with others to form multi-stage compositions. Walk-through: adding `square_step`.

### Step 2.1 — Pick a primitive and arg layout

`square(x)` takes one input. The fragment table entry:

```c
{
  "square_step",
  { "squared", "to-the-power-of-two", "second-power", "raised", NULL },
  "square",
  { "x" }, 1, 0
},
```

The fields:
- `name`: fragment ID
- `keywords`: NULL-terminated list, lexically distinct from other fragments' bags
- `primitive`: the underscore-prefixed form for special-case emissions, or the bare native name for general-case emission
- `arg_names`: ordered argument names (matches the primitive's signature)
- `n_args`: arg count
- `chain_arg_idx`: which arg (always 0) accepts the predecessor's output when chained

### Step 2.2 — Test with a multi-stage prompt

Edit `demos/wiring_organelle/pipeline_corpus_composition.txt`:

```
# EXPECTED: distance_1d square
# REFERENCE: squared_distance
// the distance between two readings squared
---
```

And add `r_squared_distance` to `wiring_references.c`:

```c
DEF_REF(squared_distance) {
    int64_t d = r_distance_1d(S[0], S[1]);
    return d * d;
}
```

### Step 2.3 — Verify

```bash
./build/wiring_organelle_demo --composition
```

Expected: composition pick-rate increments by 1; the new prompt classifies correctly.

### Time budget for adding a fragment: **~10 minutes**.

---

## Recipe 3 — Extend the synonym table only

If the family already exists but a synonym is missing (a real-world prompt uses a word the curator didn't pre-register), the cheapest extension is adding to `tools/corpus_expand.c`'s synonym list for that family:

```c
{ "bmi_clamped",
  {
    { "bmi", "body mass index", "Quetelet index", "Quetelet ratio", "mass-height index", "BMI score", "BMI value", "BMI",
      /* NEW */ "obesity index", "weight-stature ratio",
      NULL },
    /* ... */
  },
  /* ... */
}
```

Regenerate + re-eval. **~5 minutes per added synonym.** Effect: the TF-IDF Phase 4 classifier now recognises that synonym, lifting adversarial axis-2 robustness for prompts using it.

---

## Recipe 4 — Run the eval matrix

Quick reference for the four CLI modes (already in `RESEARCH_PIPELINE_IR.md` §44.6 + §46.6):

```bash
# Anchor retrieval, novel paraphrases (the headline 100% claim):
./wiring_organelle_demo --clean-only

# Multi-stage composition:
./wiring_organelle_demo --composition

# Wiring transformer alone (true generalisation baseline):
./wiring_organelle_demo --no-anchor --clean-only

# Phase 4 TF-IDF on adversarial axis-2 (the 90% claim):
./corpus_expand build/pipeline_corpus_phase4_train.txt 42
./manifold_tfidf_demo build/pipeline_corpus_adversarial.txt build/pipeline_corpus_phase4_train.txt
```

Run all four to validate that no regression is introduced by your change.

---

## Recipe 5 — Catch leakage at build time

Before committing changes that touch `tools/pipeline_corpus_gen.c` or `tools/corpus_expand.c`:

```bash
./tools/check_held_out_leakage.sh build/
```

The script greps every held-out prompt against the training files. The script exits 0 by default (the documented Phase 13 leakage is grandfathered) but flags new contamination loudly. If you want to enforce no-leakage hard, change the final `exit 0` in the script to `exit 1`.

---

## OPA curator mode preview

The architecture is a *general* pipeline-IR + retrieval scaffold; it doesn't have to be specialised to the 20 anchored families. **Curator mode** is the engineering effort to point the same architecture at a specific deployment scenario:

- **Financial calculator domain.** Curate ~50 anchors covering specific tax / loan / investment / yield / depreciation / amortisation calculations + a fragment library for chaining + a synonym table covering bank / fintech vocabulary. Expected coverage on a 200-prompt domain test: ~95%.
- **Embedded ML pipeline domain.** Curate anchors for activation functions, losses, regularisers, normalisers + fragments for stacking + synonyms covering ML jargon. Expected coverage: ~90%.
- **Command-language assistant.** Curate anchors for shell-style operations (find, grep, awk-equivalents) + composition fragments + synonyms covering CLI vocabulary. Expected coverage: ~85%.

Each scenario is a curator-effort investment of 1–4 weeks (depending on scope) and produces a deployable, deterministic, auditable, no-cloud-dependency translator from natural English in that domain to verified pipeline-IR DAGs that execute end-to-end.

The same Recipe 1–5 above applies to each scenario; only the domain content changes.

---

## When NOT to extend

- **Don't extend the wiring transformer** to chase higher % on its own. The 35% true-generalisation ceiling is documented through 17 phases. Adding more wiring-corpus paraphrases reintroduces Phase 13-style training-on-test contamination if not carefully audited.
- **Don't try to make the system handle truly out-of-anchor families** (e.g. arbitrary calculus, symbolic algebra). The architecture is anchor-bounded by design; what you can express is what you've curated.
- **Don't add a learned encoder larger than TF-IDF** unless the curator's synonym table is exhausted and you have a corpus 10×+ the current 4k size. The §41 + §46 result says simpler models match or beat complex ones at this scale.

---

## Reference

- The arc's full development log: `docs/research/RESEARCH_PIPELINE_IR.md` (sections §1–§46 plus §16 closing remark)
- The standalone paper: `docs/research/RESEARCH_WIRING_ORGANELLE_PAPER.md` (v3.5)
- The manifold-learning research note: `docs/research/RESEARCH_MANIFOLD_LEARNING.md`
- The book chapter narrating the engineering transition: `book/12.md` (and the consolidated book file)
- The state-of-the-arc snapshot: `RESEARCH_PIPELINE_IR.md` §44

— Ajay Soni, Enjector Software Ltd. April 2026.
