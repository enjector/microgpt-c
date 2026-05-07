# EML Hybrid OPA Pipeline Demo

**Neuro-symbolic composition** in the OPA style: a tiny neural organelle
handles fuzzy categorical pattern matching, two EML organelles handle
deterministic numeric transforms, and the project's standard Kanban +
cycle-detector scaffolding routes between them.

## What the pipeline does

At each simulated time step, the pipeline observes
`(price, rt, recent_directions)` and produces a structured triple
`(predicted_direction, log_price, discount_factor)`:

```
   stream observation         →     pipeline      →      output
 ────────────────────────         ─────────────         ─────────────
   price        ─────────────────▶  EML logprice  ─────▶  log(p)
   rt           ─────────────────▶  EML compound  ─────▶  exp(rt)
   history (U/D/F tokens) ────▶  neural predictor  ─┐
                                                      ├──▶  Kanban
                              ◀── cycle override  ───┘    Cycle detector
                                                          ─────▶  predicted_dir
```

**Why this split?** Each organelle does what gradient descent (resp. the
discrete snap) is good at:

- The neural organelle (~10 K params, char-level transformer) learns the
  categorical autocorrelation in the direction stream — exactly the
  fuzzy-pattern task that 30 lines of deterministic C couldn't capture.
- The EML organelles compute `log(p)` and `exp(rt)` *exactly* via snapped
  Sheffer trees — no floating-point parameter drift, perfect
  extrapolation, deterministic across platforms. This is what neural
  organelles are wasteful at (see the parent research's §9.3.3
  extrapolation comparison).
- The Kanban + cycle detector are deterministic C scaffolding: they
  catch U↔D oscillations the neural part would otherwise continue
  indefinitely.

This mirrors the project's broader OPA philosophy: same total compute
budget, sharper allocation per task.

## Build & run

```bash
cmake --build build --target eml_hybrid_pipeline_demo
cd build && ./eml_hybrid_pipeline_demo
```

(The demo expects `c_directions.txt` in the working directory; the build
system copies it next to the binary.)

Expected output (deterministic from a pinned RNG seed):

```
Training direction-predictor organelle (resumable from c_directions.ckpt)...

========================================
ORGANELLE: DirPredictor
========================================
corpus: 40 docs | 1000 chars (1.0 KB)
vocab: 5 characters
params: 10224 | steps 600 | lr 0.0100

Predictor ready (vocab=5, 40 documents).

 step  history        true  pred  cyc   price    log(p)        rt  exp(rt)
 ----  -------------  ----  ----  ---  ------  --------   ------  -------
    0  UUUDDF          F     F     no     2.93    1.0740  +0.038   1.039
    1  UUDDFF          F     F     no     2.07    0.7251  +0.346   1.414
    ...
    9  DUUDDD          D     U     no     2.47    0.9060  +0.203   1.225

Direction predictor accuracy: 4/10  (random baseline ≈ 33%)
EML self-checks (math.h reference):
  log(p) tree max abs err over 50 points: 1.246e-07
  exp(x) tree max abs err over 50 points: 3.318e-07
```

The neural-organelle accuracy is modest by design — it's a 10 K-param
model trained on 40 short documents. The point is *composition*, not
the predictor's standalone performance. The EML self-checks confirm
the deterministic transforms recover their targets at float32 epsilon.

## Architecture details

| Organelle | Type | Params | Role |
|---|---|---|---|
| `DirPredictor` | char-level transformer (N_LAYER=2, N_EMBD=24, BLOCK_SIZE=32) | ~10 K | Categorical next-token prediction over `{U, D, F}` |
| `EML logprice` | snapped EML tree (depth 3) | 8 leaves + 7 gates | `log(p)` |
| `EML compound` | snapped EML tree (depth 1) | 2 leaves + 1 gate | `exp(rt)` |
| `OpaKanban` | deterministic C struct | — | Rolling history, blocked-action set |
| `OpaCycleDetector` | deterministic C struct | — | Detects + overrides A↔B oscillations |

The neural organelle's checkpoint (`c_directions.ckpt`) is reused on
subsequent runs; delete it to force a fresh training pass.

## Why this is in microgpt-c

This demo is the proof-of-concept for the broader integration
documented in
[`docs/research/RESEARCH_EML_ORGANELLE.md`](../../../docs/research/RESEARCH_EML_ORGANELLE.md).
The boundary map (`eml_quant_boundary_demo`) shows where the EML
organelle works and where it fails standalone; this hybrid demo shows
how to combine it with neural organelles when neither tool can solve
the full task alone.

It complements rather than competes with existing OPA demos:
- Connect-4 / Pentago / Tic-Tac-Toe etc. are pure neural-organelle
  pipelines (categorical state spaces, no continuous numeric content
  to transform).
- The Wiring Organelle is also pure neural (NL→graph).
- Lottery is a negative control (entropy floor).
- This demo is the first hybrid (neural + EML symbolic) that produces a
  *structured numeric output* from sequential categorical history.

## Honest caveats

- The 4/10 predictor accuracy is consistent with the project's general
  "tiny specialist organelle" philosophy — it's not a benchmark win.
  More training steps, more docs, or a slightly larger N_EMBD would
  push it higher; that wasn't the goal here.
- The synthetic stream is autocorrelated by construction, so the
  predictor is solving an easy task. A real-world stream where the
  next direction depends on richer features would need a different
  architecture (or a feature-engineered prompt).
- The EML organelles are *deployed* here, not trained on the device.
  Training stays in PyTorch in the companion EML research repo
  (`~/dev/research/eml/`); the snapped trees ship as compile-time
  constants.
