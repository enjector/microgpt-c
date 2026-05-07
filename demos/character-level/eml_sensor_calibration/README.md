# EML Sensor Calibration Demo

The first OPA demo that puts an EML organelle into the **planner / worker /
judge** pattern with a *continuous-output* Worker — the slot the
categorical game demos (Connect-4, Tic-Tac-Toe, Pentago, …) cannot fill.

## What the pipeline does

A photodiode-style log-amp sensor produces noisy current readings. The
true physical law is `V_calibrated = log(I)` — the canonical log-amp
relation in instrumentation electronics. We want to recover it exactly,
flag out-of-range readings, and let a neural Planner orchestrate when
to MEASURE / REPORT / CALIBRATE.

```
  noisy current I_obs ─▶ [EML Worker]  V = log(I_obs)
                                          (depth-3 EML tree, deterministic)
                            │
                            ▼
                       [Judge]   V → {OK, LO, HI}
                                          (pure C, range comparator)
                            │
                            ▼
   recent state history ─▶ [Planner]   {M, R, C}
                                          (neural organelle, ~10 K params)
                            │
                            ▼
                       deterministic safety override
                       (LO/HI → CALIBRATE regardless)
```

## Why this is a good fit

| Slot | Substrate | What it's good at | Why this slot |
|---|---|---|---|
| Worker | snapped EML log tree | exact deterministic numeric transform | the continuous output that game demos can't produce |
| Judge | pure C | boolean range check | unchanged from the project's standard PWJ pattern |
| Planner | tiny char-level transformer | fuzzy state-history sequencing | learns "should we report yet" from training |

This matches EML's documented sweet spot per the parent research's §9.1
and §9.3.1: **shallow elementary closed form + continuous data + noise
robustness needed**.

The depth-3 log tree is hand-coded in `c_eml_logprice.h` (re-used from
`eml_quant_boundary/`) and verified at machine precision by
`tests/test_microgpt_eml.c::test_eml_ln_x`.

## Build & run

```bash
cmake --build build --target eml_sensor_calibration_demo
cd build && ./eml_sensor_calibration_demo
```

(The build copies `c_calibration_planner.txt` next to the binary; the
demo expects to be run from the `build/` directory.)

## Expected output

```
 step  I_true  I_obs    V_calib  judge  planner   verdict
 ----  ------  ------   -------  -----  -------   --------
    0   1.00  0.936   -0.066   OK    M        MEASURE
    1   1.50  1.534   +0.428   OK    R        REPORT
    2   2.00  1.855   +0.618   OK    R        REPORT
    3   2.50  2.271   +0.820   OK    R        REPORT
    4   0.30  0.293   -1.229   LO    C        RECALIBRATE
    5   1.00  0.937   -0.065   OK    M        MEASURE
    6   1.50  1.429   +0.357   OK    M        MEASURE
    7   2.00  2.044   +0.715   OK    M        MEASURE
    8   6.50  6.463   +1.866   HI    C        RECALIBRATE
    9   1.50  1.522   +0.420   OK    M        MEASURE
   10   1.00  1.012   +0.011   OK    M        MEASURE
   11   1.50  1.431   +0.358   OK    M        MEASURE
   12   2.00  2.041   +0.713   OK    R        REPORT
   13   1.50  1.480   +0.392   OK    M        MEASURE

Summary: 12 OK, 2 flagged (LO/HI), 2 RECALIBRATE.
EML Worker self-check vs math.h on 50 grid points: max abs err 1.464e-07
```

The two scripted excursions (steps 4 and 8: I=0.30 mA → LO,
I=6.50 mA → HI) are correctly flagged by the deterministic Judge and
trigger CALIBRATE via the safety override regardless of what the Planner
emits — exactly the same pattern the game demos use when an invalid
move comes back.

The EML Worker self-check at the end confirms the deployed log tree
recovers `log(I)` to float32 epsilon over a 50-point grid — independent
verification that the snapped tree is functioning as intended.

## What the Planner is actually learning

The 30-document training corpus in `c_calibration_planner.txt` encodes
the rule:
- After a few OK readings → REPORT
- LO or HI → CALIBRATE
- Otherwise → MEASURE

A tiny char-level transformer (~10 K params, 600 training steps) learns
to sequence these tokens reasonably. It doesn't need to learn the
*boolean* LO/HI logic — that's the Judge's job. It learns the *fuzzy*
"have we had enough OK readings to report" trigger.

This is the OPA philosophy in action: each substrate handles what it's
substrate-good at. Gradient descent doesn't waste capacity on the
deterministic LO/HI threshold; the Judge doesn't waste C code on
state-history sequencing; the EML Worker doesn't try to learn anything
because its closed form is already known.

## Honest scope

- The data is **synthetic**. The 14-step current schedule
  (`sample_true_current`) is hand-scripted to exercise OK / LO / HI in
  a known order. A real instrument would feed actual ADC readings.
- The EML Worker is **deployed, not trained on-device**. The depth-3
  log tree was recovered offline by the PyTorch trainer in the parent
  EML research repo and exported as a C99 header. On-device EML
  training is out of scope (would require backward pass + Adam in C99,
  not implemented).
- The Planner is **tiny** (~10 K params, 30 training docs). A real
  deployment would need a larger corpus reflecting actual operational
  patterns, but the architecture would be unchanged.
- The Judge thresholds are **constants**. A more sophisticated demo
  could have the Judge compare against an EML reference value
  (calibration curve) rather than hard limits.

## Why this is the right kind of EML demo

This demo plays directly to the strengths documented in the parent
research:
- **Shallow elementary target** (`log(I)`, depth 3 in EML).
- **Continuous data** with additive Gaussian noise on the log signal.
- **Exact recovery** in the snapped form — no parameter drift, no
  quantisation, no checkpoint to ship.
- **Extrapolation** outside the training range works perfectly because
  the recovered tree IS `log(I)`.

Compare to the boundary demo (`eml_quant_boundary_demo`):
- Cases 1-3 there demonstrate EML organelles standalone.
- This demo demonstrates an EML organelle **in a multi-organelle
  pipeline** that mixes neural pattern-matching, deterministic logic,
  and continuous numeric transform — i.e., it shows EML composing
  cleanly inside the project's existing PWJ architecture.

See `docs/research/RESEARCH_EML_ORGANELLE.md` for the integration
context and the boundary map.
