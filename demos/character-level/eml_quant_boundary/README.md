# EML Quant Boundary Demo

A boundary map of where the EML organelle works on quant-flavored data and
where it doesn't, in the spirit of the project's
[lottery negative control](../lottery/) — failure modes are documented as
evidence of method validity, not hidden.

## Why this exists

The parent EML research (`~/dev/research/eml/experiments/RESEARCH.md`)
established that EML symbolic regression is competitive only in a narrow
regime:
- target is a **shallow** elementary closed form (depth ≤ 4 in EML),
- inputs / outputs are **continuous** real-valued,
- data may be **noisy** (the discrete snap acts as a symbolic denoiser),
- you need **extrapolation** outside the training distribution.

For real-world quant work most of the relevant relations sit *outside*
that regime — even simple log-return is depth-83 in compiled EML, and
realistic GBM has SNR < 1. Rather than ship a quant demo that hides this,
we ship the boundary itself: 3 positive cases that work, 3 negative cases
that diagnostically fail.

## Cases

### Positive (EML organelle delivers exact recovery)

| # | Case | Target | EML depth |
|---|------|--------|-----------|
| 1 | `compound_factor` | `y = exp(rt)` (continuous compounding) | 1 |
| 2 | `log_price` | `y = log(p)` (log-price transform) | 3 |
| 3 | `depth-2 frontier` | `y = e − log(exp(y) − log(x))` | 2 |

For each, training data is generated with additive Gaussian noise; the
snapped EML tree is evaluated on a clean held-out test set (in-domain)
and on an extrapolation set (wider domain).

### Negative (EML organelle out-of-scope, by design)

| # | Case | Failure mode | Headline metric |
|---|------|--------------|-----------------|
| 4 | GBM log-price, σ=0.2 | SNR floor | `Var(signal) / Var(noise) ≪ 1` |
| 5 | log return `log p − log q` | depth wall | `K = 83` (RPN) — depth ≥ 7 needed; trainer ceiling = 4 |
| 6 | Black–Scholes call | not elementary | `N(·)` has no finite elementary form |

For negative cases we don't run the trainer (it would predictably fail).
Instead we generate the data and directly compute the diagnostic metric —
exactly the lottery demo's pattern.

## Build & run

```bash
cmake --build build --target eml_quant_boundary_demo
./build/eml_quant_boundary_demo
```

Expected (representative numbers; trees are deterministic, RNG is pinned):

```
[POSITIVE]
  1 compound_factor  exp(rt)      train 2.5e-3 ≈ σ²    test 0.0e+00    extrap 0.0e+00   PASS
  2 log_price        log(p)       train 4.0e-4 ≈ σ²    test ~5e-15     extrap ~5e-15    PASS
  3 depth-2 frontier elementary   train 2.5e-3 ≈ σ²    test ~2e-15     extrap ~2e-15    PASS

[NEGATIVE]
  4 GBM log-price σ=0.2           SNR ≈ 3.5e-3   (no SR method recovers)
  5 log return                     K = 83          (depth wall vs trainer ceiling = 4)
  6 Black-Scholes call             Taylor floor MSE ≈ 0.18  (N(·) is non-elementary)
```

Train MSE on positive cases equals the noise variance σ² — the optimal
residual for a model that exactly captures the data-generating relation.
Test and extrapolation MSE are 0 to float-precision squared.

## Files

| File | Role |
|---|---|
| `main.c` | 6-case driver, prints unified results table |
| `c_eml_compound.h` | Hand-coded depth-1 tree for `exp(x)` |
| `c_eml_logprice.h` | Hand-coded depth-3 tree for `log(x)` |
| `c_eml_d2_elementary.h` | Hand-coded depth-2 tree for the parent research's `eml_d2` target |

The hand-coded trees are the canonical Sheffer constructions verified by
`tests/test_microgpt_eml.c` — using them avoids spinning up the PyTorch
trainer just to recover something we already know analytically.

## What this demo does *not* claim

- It does **not** claim EML is competitive with PySR or KAN on
  general-purpose symbolic regression. The parent research's §8 showed
  PySR matches or beats EML on EML's own home-turf targets.
- It does **not** claim EML is suitable for real financial time-series
  modelling. Cases 4-6 demonstrate why.
- It **does** claim the EML organelle is a clean, deterministic, exactly-
  interpretable C99 primitive for the narrow set of problems where its
  preconditions hold — and the discrete snap gives noise-robust exact
  recovery + perfect extrapolation that approximator-based fitters
  (LinReg, PySR) cannot match. See the parent research's §9.3 for the
  measurements supporting that claim.

See `docs/research/RESEARCH_EML_ORGANELLE.md` for the full integration
story.
