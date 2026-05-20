# E04 — Neural + EML hybrid (pendulum)

End-to-end neuro-symbolic demo pre-registered in
[`experiments/E04-eml-neural-hybrid.md`](../../../experiments/E04-eml-neural-hybrid.md).

## What this is

A worked composition where two organelle classes — neural (pattern
matching) and EML (shallow elementary symbolic) — cooperate inside a
verified Pipeline IR document on a synthetic pendulum dataset.

```
noisy (L, theta_obs)
      │
      ▼
[classifier:neural]     ~30K-param char-level transformer
      │                  emits 'S' (small-angle) or 'L' (large-angle)
      ▼
[eml_<regime>:eml]      frozen depth-≤-4 EML tree per regime
      │
      ▼
[mux + bounds:judge]    type-routed mux + deterministic bounds check
      │
      ▼
prediction (T, regime, sympy_audit)
```

The graph is parsed via `pipeline_parse_text()` and verified via
`pipeline_verify()`. The classifier is an `Organelle` (existing
`microgpt.h` engine); the EML nodes call `eml_eval()`; the verifier is
a host-side callback.

## Build / run

```bash
cmake --build build --target eml_neural_hybrid_demo
./build/eml_neural_hybrid_demo
```

Two checkpoints are produced on first run (`c_regime_classifier.ckpt`,
`c_pureneural.ckpt`) and reused on subsequent runs — delete them to
retrain. The two corpora (`c_regime_corpus.txt`,
`c_baseline_corpus.txt`) are regenerated deterministically each run
from a pinned LCG seed.

## Placeholder caveat

The EML tree headers (`c_eml_smallangle.h`, `c_eml_largeangle.h`)
currently re-use the **depth-2 paper tree from `eml_organelle/`** as a
stand-in. The actual pendulum-target trees must be trained offline in
the companion EML research repo (`~/dev/research/eml/`) and exported
via `tools/eml_export.py`.

To keep the experiment measurable today, the demo computes the
closed-form pendulum period via `math.h` while still routing the full
classifier → IR → mux → verifier path. Once the offline-trained trees
drop in, set `-DDEMO_USE_REFERENCE_PHYSICS=0` and the float will flow
through the EML evaluator instead.

See the header docstrings for the target depth-≤-4 EML form, and
§3 of the experiment doc for the measurement scaffold and current
numbers.
