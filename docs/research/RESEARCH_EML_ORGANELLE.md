# EML Organelle: Noise-Robust Symbolic Recovery for OPA

**Status:** proof-of-concept landed (May 2026). One frozen depth-2 organelle
ships with the demo (`demos/character-level/eml_organelle/`); full
trainer remains in the companion EML research repo (`~/dev/research/eml/`).

This document explains what an EML organelle is, what it adds to the
Organelle Pipeline Architecture (OPA), what it doesn't, and how to use it.

---

## Origin

The EML Sheffer operator
   `eml(x, y) = exp(x) − ln(y)`
is the subject of Odrzywołek (arXiv:2603.21852, April 2026). The paper
proves that EML together with the constant `1` forms a functionally
complete basis for elementary functions: every standard elementary
function (arithmetic, trig, log, exp, π, e, i) can be written as a
binary tree of identical EML nodes whose leaves are drawn from `{1, x}`.

The companion *EML research repository* (`~/dev/research/eml/`,
`experiments/RESEARCH.md`) contains an independent reproduction and
extension of the paper's results. Most relevant findings for OPA:

| Finding | Source | What it means here |
|---|---|---|
| Compiler verified on 16 elementary primitives | EML repo §2 | A `.m`-format snapped tree from the trainer is a faithful symbolic representation. |
| Discrete-tree SR succeeds at d=2 100 % / d=3 25 % / d=4 28 % | EML repo §3 | The reachable depth ceiling for blind random-init recovery. |
| **Noise-robust exact recovery on shallow targets** | EML repo §9.1, §9.3 | The discrete snap acts as a symbolic denoiser. At d=2, recovery is 16/16 at every σ ∈ {0, 0.001, 0.01, 0.1}. At d=3, it's constant 6/16 across σ. |
| Extrapolation: EML 0 error, PySR/LinReg degrade | EML repo §9.3.3 | When the underlying law is a shallow EML expression, the snapped tree extrapolates perfectly while approximators don't. |
| Master-formula simplex path: 0/16 exact | EML repo §10 | Practitioners should use the discrete-tree trainer, not the simplex master formula. |

The full reproduction is 1 000+ lines of analysis and 17 experiment scripts
in the EML repo. This integration into microgpt-c keeps only what's directly
useful for OPA: the deployment-side evaluator and a working demo.

---

## What an EML organelle is in OPA

An **EML organelle** is a frozen specialist whose forward pass is a fixed
binary tree of `eml(a, b) = exp(a) − log(b)` nodes. After being trained
offline (in the EML research repo's PyTorch trainer) and snapped to a
discrete tree, the organelle is a compile-time constant — three
`unsigned char` arrays in a generated header. Evaluation goes through
`src/microgpt_eml.{h,c}`, ~150 lines of pure C99 + `<math.h>`.

```
                  Offline (PyTorch, EML repo)              Deployment (microgpt-c)
                  ────────────────────────────             ──────────────────────────────
                                                           eml(a, b) = exp(a) − ln(b)
   noisy data ──▶ tree_prototype_torch_v16_final.py ──▶ snapped .m
                                                              │
                                                  tools/eml_export.py
                                                              │
                                                              ▼
                                            demos/.../c_eml_tree.h ──▶  EmlTree literal
                                                                         │
                                                                         ▼
                                                                  microgpt_eml_eval()
                                                                  (deterministic, math.h)
```

The organelle has the properties that fit OPA's "deterministic C
scaffolding" aesthetic:

- **Zero floating-point parameters.** Every choice in the snapped tree is a
  one-hot enum (leaf ∈ {1, x, y}) or a binary gate. No quantisation, no
  drift, no checkpoint to ship.
- **Deterministic across platforms.** `exp` and `log` are IEEE-754
  specified; the same tree gives bit-identical answers on x86, ARM, and
  embedded targets.
- **Trivial deployment.** The snapped tree fits in <1 KB even at depth 4.
  The tree can be unrolled to straight-line C if a particular target wants
  to skip the loop.
- **Interpretable.** The tree decodes to a sympy expression or a
  human-readable formula. After training, you can paste the closed form
  back into Mathematica or a paper.

---

## What it's good for, and what it isn't

Consistent with the EML research's §11 conclusions:

**Good fit for an EML organelle:**

- The underlying relation is a **shallow** elementary closed form
  (depth ≤ 4 in EML compiled form).
- Inputs and outputs are continuous real-valued.
- Training data may be noisy; the snap acts as a symbolic denoiser. The
  parent research showed up to σ = 0.1 robustness on shallow targets.
- The downstream consumer needs **out-of-distribution extrapolation** that
  a fitted approximator can't deliver.
- Your scaffolding wants a deterministic, interpretable C function rather
  than opaque weights.

**Not a fit:**

- **Categorical / combinatorial tasks.** The OPA games (Connect-4, Pentago,
  Sudoku, Tic-Tac-Toe, …) are *not* EML candidates. There is no shallow
  elementary closed form for an optimal Connect-4 move; it's a minimax over
  a game tree. Use neural organelles there. This is the most important
  caveat — the EML organelle does not generalise to OPA's existing games
  in any useful way.
- **Targets deeper than depth 4 in EML form.** Even simple multiplication
  is depth 8 in compiled EML; subtraction `x − y` is K=83. Black-Scholes,
  GBM log-prices, factor models, Sharpe — all unreachable.
- **Real quant finance time series.** SNR < 1 over typical horizons, and
  the relations are too deep regardless. The EML repo's §9.2 documents
  this with a GBM probe.

The two organelle classes (neural and EML) are **complementary, not
competitive**. Almost every OPA task today fits the neural side. EML is a
new tool for the narrow but well-defined problem class above.

---

## Demo

`demos/character-level/eml_organelle/` ships a single working demo that
loads a pre-trained snapped tree (`c_eml_tree.h`, exported from the EML
research repo's depth-2 sweep) and evaluates it against a noisy elementary
target.

```
$ ./build/eml_organelle_demo
MicroGPT-C  EML Organelle Demo
  Target: y = e - log(exp(input_y) - log(input_x))
  Train: 200 points on [1, 3]^2, sigma=0.10 Gaussian noise
  In-domain test: 1000 clean points on [1, 3]^2
  Extrapolation: 1000 clean points on [0.5, 5.0]^2 (outside training)

                       train MSE   test MSE   test max|err|   extrap MSE   extrap max|err|
  ------------------------------------------------------------------------------------
  EML organelle       9.895e-03  1.745e-15    2.384e-07    2.041e-15    2.384e-07
  LinReg (quadratic)  1.006e-02  4.422e-04    7.410e-02    5.529e-02    1.275e+00
```

Key numbers:

- EML train MSE: `9.9e-3` ≈ σ² = 0.01. This is the **optimal** residual
  for any model that exactly captures the data-generating relation; the
  remaining error is purely the irreducible noise.
- EML in-domain test MSE: `1.7e-15` (float32 epsilon squared). The recovered
  tree IS the target function.
- EML extrapolation max error: `2.4e-7` (float32 epsilon). Extrapolation
  outside the training domain `[1, 3]²` to `[0.5, 5.0]²` is exact.
- LinReg extrapolation max error: `1.28`. Five orders of magnitude worse,
  because the polynomial approximation breaks down outside the fit range.

The EML organelle is `188 bytes` of compile-time data (3 `unsigned char`
arrays + struct) plus ~150 lines of evaluator. No checkpoint, no
quantisation, no learning-rate schedule.

---

## How to add a new EML organelle

Workflow when you have a new candidate target with continuous-valued
training data:

### 1. Train in the EML research repo

```bash
# In ~/dev/research/eml
cd experiments
source venv/bin/activate
# (Optional) write a custom target into the trainer's TARGET_FUNCTIONS
# dict; or pick from eml_depth2/eml_depth3/eml_depth4.

# Run the trainer
python /path/to/SymbolicRegressionPackage/EML_toolkit/EmL_training/PyTorch_v16_final/tree_prototype_torch_v16_final.py \
    --target-fn eml_depth2 \
    --depth 2 \
    --init-strategy all \
    --seed0 137 --seeds 8 \
    --save-prefix runs/your_new_organelle \
    --skip-plot
```

The trainer writes `*_metrics_*.json` summarising recovery, plus per-seed
`.m` and `.pt` files. Pick a successful seed (`success: true,
symbol_success: true, snap_mse: 0.0`) — its discretized `.m` file is the
input to the export tool.

### 2. Export to a C99 header

```bash
# In microgpt-c
python tools/eml_export.py \
    --m /path/to/runs/your_new_organelle_runNN_seedXXX_strategy_<ts>.m \
    --name your_organelle_name \
    --target-desc "human-readable description of what this computes" \
    --out demos/character-level/your_demo/c_your_organelle.h
```

The exporter writes a header containing:
- `static const unsigned char your_organelle_name_leaves[N]` (one-hot enum)
- `static const unsigned char your_organelle_name_gates_left[M]`
- `static const unsigned char your_organelle_name_gates_right[M]`
- `static const EmlTree your_organelle_name = { ... };`

### 3. Use in a C99 demo

```c
#include "microgpt_eml.h"
#include "c_your_organelle.h"

scalar_t pred = eml_eval(&your_organelle_name, x, y);
```

That's the entire deployment path.

### 4. Add a CMake entry

```cmake
add_demo(
  NAME   your_eml_demo
  SOURCE demos/character-level/your_demo/main.c
)
target_include_directories(your_eml_demo PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/demos/character-level/your_demo)
```

---

## Files

| Path | Role |
|---|---|
| `src/microgpt_eml.h` | Public API: `EmlTree` struct, `eml_eval`, `eml_eval_batch`, `eml_mse`, `eml_max_abs_err`. |
| `src/microgpt_eml.c` | Implementation (~150 lines, depends only on `<math.h>`). |
| `tools/eml_export.py` | Reads a discretized `.m` file from the trainer, emits a C99 header. |
| `demos/character-level/eml_organelle/main.c` | POC demo: in-domain + extrapolation against LinReg. |
| `demos/character-level/eml_organelle/c_eml_tree.h` | Generated; the snapped depth-2 tree literal. |
| `demos/character-level/eml_organelle/README.md` | Demo run instructions. |
| `tests/test_microgpt_eml.c` | 6 unit tests for the evaluator (exp, ln, eml(x,y), eml_d2, batch, mse). |

Build & test:

```bash
cmake --build build --target eml_organelle_demo test_microgpt_eml
./build/test_microgpt_eml          # 6/6 unit tests
./build/eml_organelle_demo         # 5-row table
ctest --test-dir build -R microgpt_eml
```

---

## Sensor calibration PWJ pipeline (eml_sensor_calibration_demo)

`demos/character-level/eml_sensor_calibration/` is the strongest
real-world fit of the EML organelle into the planner / worker / judge
pattern. The task is photodiode-style log-amp calibration — recovering
`V = log(I)` from noisy current readings — which is exactly the regime
EML SR was designed for in the parent paper's spirit (recovering
elementary physical laws from noisy measurements).

Pipeline:

1. **EML Worker** (snapped depth-3 log tree): computes
   `V_calibrated = log(I_obs)` exactly, deterministically. This is the
   first PWJ Worker in the project that produces a *continuous* output
   rather than a categorical token. The depth-3 log tree is the canonical
   `eml(1, eml(eml(1, x), 1))` Sheffer construction verified by
   `tests/test_microgpt_eml.c::test_eml_ln_x`.

2. **Judge** (pure C): maps `V_calibrated` to `{OK, LO, HI}` via
   threshold comparison. Same pattern as the game demos' is-valid checks,
   adapted for a continuous input.

3. **Planner** (~10 K-param char-level transformer): trained on 30
   synthetic state-action sequences encoding the rule "K consecutive
   OKs → REPORT; LO/HI → CALIBRATE; else MEASURE". Learns the *fuzzy*
   sequencing trigger; the boolean LO/HI logic stays in the Judge where
   it belongs.

The demo also implements the project's standard deterministic safety
override (LO/HI → CALIBRATE regardless of what the Planner emits),
mirroring the connect4 / tictactoe pattern when an invalid move comes
back.

Demo result: 12/14 Judge-OK, 2/14 flagged at the scripted excursions
(I = 0.30 mA → LO; I = 6.50 mA → HI), 2/14 RECALIBRATE. EML Worker
self-check against `math.h` on a 50-point grid: max abs err 1.46e-7
(float32 epsilon).

This demo is the proof point for the broader claim: **EML organelles
are the right tool whenever the Worker slot needs a continuous,
deterministic, exact transform of an elementary closed form** —
sensor calibration, simple physics decay, log-amp circuits, photodiode
output, pH-meter readouts, and similar. The categorical PWJ demos
cannot use EML this way because their Worker output is a token; this
demo introduces a new Worker shape (continuous numeric) for the OPA
pattern.

## Hybrid OPA pipeline (eml_hybrid_pipeline_demo)

`demos/character-level/eml_hybrid_pipeline/` is the first OPA demo to
combine a neural organelle with EML organelles. The pipeline:

1. **Neural organelle** (`DirPredictor`, ~10 K params, char-level
   transformer): predicts the next direction token `∈ {U, D, F}` from
   recent history. Trained on a 40-doc corpus of synthetic autocorrelated
   sequences.
2. **EML organelle 1** (`logprice`, snapped depth-3 tree): deterministic
   `log(p)` transform.
3. **EML organelle 2** (`compound`, snapped depth-1 tree): deterministic
   `exp(rt)` transform.
4. **OpaKanban + OpaCycleDetector**: shared deterministic C scaffolding
   (same modules used in connect4 / tictactoe / puzzle8) that record the
   rolling direction history and break U↔D oscillations.

Per-step output is a structured triple `(predicted_direction, log_price,
discount_factor)` that no single organelle could produce alone:
- The neural organelle can only emit categorical tokens, not log-prices
  or discount factors.
- The EML organelles cannot predict direction from a sequence of
  categorical observations — that's not an elementary closed-form
  relation.
- The Kanban + cycle detector cannot generalise to fuzzy patterns —
  that's gradient descent's job.

This is OPA composition with a sharper compute allocation: the neural
part learns *what gradient descent is good at* (categorical sequence
prediction), and the deterministic scaffolding + EML organelles handle
*what gradient descent is wasteful at* (numeric transforms, oscillation
breaking, exact symbolic recovery).

The demo's predictor accuracy is modest (~40 % vs 33 % random) — that's
not a benchmark win, just a working proof of concept on a tiny corpus.
The EML self-checks at the end verify the snapped trees recover `log(p)`
and `exp(x)` to float32 epsilon (~1e-7) over a 50-point grid.

## Boundary map (eml_quant_boundary_demo)

`demos/character-level/eml_quant_boundary/` ships a 6-case boundary map
showing where the EML organelle works on quant-flavored data and where
it doesn't, in the same diagnostic style as the lottery negative control.

**Positive cases** (EML organelle delivers exact recovery, 3/3 PASS):

| Case | Target | EML depth | Train MSE (noisy) | Clean-test MSE | Extrap MSE |
|---|---|---|---|---|---|
| `compound_factor` | `exp(rt)` (continuous compounding) | 1 | ≈ σ² | 0 | 0 |
| `log_price` | `log(p)` | 3 | ≈ σ² | ~5e-15 | ~5e-15 |
| `depth-2 frontier` | `e − log(exp(y) − log(x))` | 2 | ≈ σ² | ~2e-15 | ~2e-15 |

In every positive case the train MSE equals the additive noise variance
σ² — the optimal residual for a model that exactly captures the data-
generating relation — and the clean-test / extrapolation MSE are at
float-precision squared.

**Negative cases** (EML organelle fails by construction, characterised
not fitted):

| Case | Failure mode | Headline metric | What it means |
|---|---|---|---|
| GBM log-price σ=0.2 | SNR floor | `Var(signal)/Var(noise) ≈ 3.5e-3` | No SR method (EML, PySR, KAN, neural) can recover σ·W(t) |
| log return `log p − log q` | depth wall | `K = 83` RPN, depth ≥ 7 | Trainer's reachable depth is 4 |
| Black-Scholes call | not elementary | Taylor floor MSE ≈ 0.18 | `N(·)` has no finite elementary form |

Together the 6 cases delineate the regime sharply: **shallow elementary +
continuous + noisy + extrapolation needed** is where to reach for an EML
organelle. Everything else — categorical tasks, deep relations, real
quant time series, non-elementary targets — needs a different tool.

## Open follow-ups

These are reasonable next steps if the EML organelle proves useful:

1. **A second EML organelle on a different shallow target.** The current
   demo uses the paper's `eml_d2` synthetic target. A real candidate
   would be a sensor calibration or simple physics relation present in
   one of the project's own data sources.
2. **Decode helper in C99.** The Python `decode_to_symbolic()` helper in
   the trainer turns a snapped tree into a sympy expression. A C99
   equivalent that prints `eml(eml(1, x), 1)`-style strings would be
   useful for organelle introspection in OPA.
3. **Bridge to the Discretisation Wall finding.** The project has an open
   research direction on continuous-valued data (finance time series).
   EML is a natural fit *only* on the narrow "shallow elementary law +
   noise + need for extrapolation" subset of that space; it doesn't solve
   the wall in general but may close one specific subdomain.
4. **A C99 trainer.** Out of scope for this POC. Would need backward pass,
   Adam, complex128 internals, and careful overflow handling. Probably
   not worth doing unless on-device EML training becomes a load-bearing
   requirement.

---

## Pointer to the parent research

For the full critical reproduction of the paper, including:
- compiler verification (Table 4 replication)
- depth-2/3/4 SR sweeps with init-strategy breakdowns
- depth-5 random-init failure mode + manual+noise basin probe
- master-formula plain-Adam vs simplex+harden replication
- cousin-operator probe (EDL, −EML)
- PySR head-to-head benchmark (8 targets, 75 s total)
- noise-robustness probe across depth and σ
- IC / Rank-IC / long-short-Sharpe analogue metrics
- extrapolation comparison (EML vs PySR vs LinReg)

see `~/dev/research/eml/experiments/RESEARCH.md` (1 017 lines, May 2026
v6 with reviewer-tightened §9.3 and §10.3.1 capacity confirmation).
