# EML Organelle Demo

Pure C99 demonstration of a "frozen organelle" recovered by the EML
symbolic-regression trainer (PyTorch, in the companion EML research repo)
and deployed as a compile-time constant.

The target is the paper's depth-2 EML expression
`y = e − log(exp(input_y) − log(input_x))`. The PyTorch trainer recovers the
*exact* tree from 200 noisy observations (σ=0.1) on 16/16 random seeds (see
the parent research's §9.1 / §9.3.1). After snap, the tree is exported via
`tools/eml_export.py` to `c_eml_tree.h` — a tiny C header containing three
`unsigned char` arrays. This demo loads that header, evaluates the tree on
clean test sets (in-domain and extrapolation), and reports MSE / max
absolute error.

## Build & run

```bash
cmake --build build --target eml_organelle_demo
./build/eml_organelle_demo
```

## Expected output

```
MicroGPT-C  EML Organelle Demo
  Target: y = e - log(exp(input_y) - log(input_x))
  Train: 200 points on [1, 3]^2, sigma=0.10 Gaussian noise
  In-domain test: 1000 clean points on [1, 3]^2
  Extrapolation:  1000 clean points on [0.5, 5.0]^2 (outside training)

                       train MSE   test MSE   test max|err|   extrap MSE   extrap max|err|
  ---------------------------------------------------------------------------------------
  EML organelle       ~1.1e-02    0.000e+00     0.000e+00      0.000e+00      0.000e+00
  LinReg (quadratic)  ~6e-04      1.4e-04       ~5e-2          3.4e-02        ~1.6
```

EML's train MSE is `σ² ≈ 0.01`, the optimal residual for a model that
exactly captures the data-generating relation (the residual is purely
the irreducible noise). Test and extrapolation MSE are 0 — the recovered
tree IS the target.

LinReg with `(1, x, y, xy, x², y²)` features achieves a small in-domain MSE
because the target is smooth, but its max error blows up on the
extrapolation domain.

## Re-export from a fresh trained tree

If the EML trainer is re-run and produces a new snapped `.m` file, regenerate
`c_eml_tree.h` with:

```bash
python tools/eml_export.py \
    --m /path/to/your_run_seedNNN_strategy_<timestamp>.m \
    --name eml_d2_recovered \
    --target-desc "y = e - log(exp(input_y) - log(input_x))" \
    --out demos/character-level/eml_organelle/c_eml_tree.h
```

Pass the *discretized* `.m` file (no `_continuous` in the filename); the
exporter rejects continuous-weight exports.

## When to reach for an EML organelle (and when not to)

**Use it when:**
- The underlying relation is a known-shallow elementary function
  (depth ≤ 4 in EML form: arithmetic mostly excluded — even `x − y` is
  K=83 in compiled EML).
- Inputs and outputs are continuous real-valued.
- Training data is available, may be noisy (the snap acts as a noise
  filter — the parent research's §9 documents up to σ=0.1 robustness on
  shallow targets).
- You need OOD extrapolation that a fitted model can't deliver.

**Don't use it when:**
- The task is categorical / combinatorial — board games, classification,
  symbolic puzzles. Use a neural organelle.
- The relation is deeper than depth 4 in EML form. The trainer's
  random-init success rate drops to ~1 % at depth 5, and even simple
  multiplication is depth 8.

See `docs/research/RESEARCH_EML_ORGANELLE.md` for the full integration
story and the PySR / KAN / linear comparisons documented in the parent
research.
