# Vertical sketch: finance market regime / risk detection

**Status:** working draft, follows from `docs/PRODUCTIZATION_VERTICALS.md`. The "plausible second vertical" — has the right shape, needs more new infrastructure than fraud, slower sales cycle.

## Why this vertical fits the architecture (mostly)

A modern risk-detection / regime-classification product is a graph of typed transformations:

```
prices → returns → rolling covariance → portfolio variance → expected shortfall
                                       → regime distance from each anchor regime
                                       → changepoint score
                                       → risk report DAG (auditable)
```

The pipeline IR encodes this naturally. The geodesic infrastructure was *literally designed* to model state-space distance — modelling "the market is in low-vol-trend regime" as a position in feature space and detecting transitions via geodesic distance to anchor regimes is a direct mapping.

What's harder than fraud:

1. **The verifier needs to return *probabilities*, not pass/fail.** A regime classifier outputs `P(regime = crisis | features) = 0.34`, not `flag/no-flag`.
2. **Time-series primitives don't exist yet.** `wiring_natives.c` has scalar arithmetic. Real risk needs rolling windows, EWMA, GARCH residuals, jump detection, drawdown tracking.
3. **Unsupervised / semi-supervised.** Fraud has labels (chargebacks, customer reports). Regime detection has *historical labels* (crisis periods are obvious in retrospect) but no live ground truth.
4. **Lookahead-bias is everything.** A risk model that subtly cheats on backtests is worse than no model. Engineering rigour around walk-forward + deflated metrics is essential.

## Probabilistic verifier: API change sketch

Today:
```c
int pipeline_verify(const Pipeline *p, char *err_buf, size_t err_size);
/* returns 0 = pass, non-zero = fail */
```

Proposed extension (additive, doesn't break existing callers):
```c
typedef struct {
    int    pass;          /* 0 or 1 */
    double confidence;    /* [0,1] — for probabilistic verifiers */
    int    n_evidence;
    struct {
        const char *primitive_name;
        double      contribution;    /* signed contribution to confidence */
    } evidence[16];
    char   reason[256];
} PipelineVerifyResult;

int pipeline_verify_with_confidence(const Pipeline *p, PipelineVerifyResult *out);
```

For deterministic graphs (fraud rules), `confidence` is always 1.0 when `pass=1`. For probabilistic graphs (regime classifier), `confidence` is the model's posterior. The `evidence` array gives feature-level contributions for explainability.

Implementation requires changes to:
- `pipeline_verify()` callsite plumbing (back-compat by leaving the int-returning version in place)
- The verifier's traversal needs to call per-node `confidence_contribution()` hooks
- Anchor entries need to optionally declare a calibration mode (`CALIBRATED`, `UNCALIBRATED`, `DETERMINISTIC`)

Estimated effort: ~6-8 weeks. Genuinely new mechanism, not just plumbing.

## Time-series primitive library — `wiring_natives_finance.c`

~25 stateful primitives. All take a "series handle" plus current value, return a derived value.

| Category | Primitives |
|---|---|
| **Rolling stats** | `rolling_mean(series, window)`, `rolling_std(series, window)`, `rolling_var(series, window)`, `rolling_min(series, window)`, `rolling_max(series, window)`, `rolling_quantile(series, window, q)` |
| **Smoothers** | `ewma(series, alpha)`, `hp_filter(series, lambda)`, `kalman_filter_1d(series, q, r)` |
| **Returns** | `simple_return(series)`, `log_return(series)`, `excess_return(series, benchmark)` |
| **Volatility** | `realized_vol(series, window)`, `garch_11_residual(series)`, `parkinson_vol(high, low, window)` |
| **Drawdown** | `running_max(series)`, `current_drawdown(series)`, `max_drawdown(series, window)` |
| **Correlation** | `rolling_corr(s1, s2, window)`, `rolling_beta(s1, benchmark, window)` |
| **Jump detection** | `lee_mykland_jump(returns, window, threshold)`, `bipower_var(returns, window)` |
| **Regime features** | `vol_of_vol(series, window)`, `skew_kurt(series, window)`, `acf_lag1(series, window)` |
| **Risk metrics** | `historical_var(returns, alpha, window)`, `expected_shortfall(returns, alpha, window)`, `cornish_fisher_var(returns, alpha)` |

All deterministic, all integer or fixed-point in C99 (use a `decimal` typedef for prices/returns). Each primitive ~30-100 LOC + tests. Total: ~3-4 weeks of implementation.

## Regime taxonomy — 7 named anchor regimes

Curator-defined. Each is a position in the geodesic feature space (slot + jitter, similar to the family classifier).

| # | Regime | Anchor features (rough) | Historical exemplars |
|---|---|---|---|
| 1 | `low_vol_trend` | low realized vol, positive trend strength, low autocorrel | 2017 equity bull, 2014 USD trend |
| 2 | `low_vol_range` | low vol, weak trend, high mean reversion | 2019 USD/EUR ranging |
| 3 | `mid_vol_trend` | moderate vol, clear directional bias | 2021 commodities bull |
| 4 | `mid_vol_choppy` | moderate vol, no trend, intermittent jumps | 2018 H2 equity correction |
| 5 | `high_vol_crisis` | very high vol, large negative drawdown, correlation spike | 2008 GFC, March 2020, Sept 2022 LDI |
| 6 | `vol_compression` | falling vol-of-vol, narrowing ranges, "calm before storm" | early 2020, mid 2021 |
| 7 | `regime_transition` | rising vol-of-vol, breaking correlation patterns, jump cluster | identified by changepoint detector |

Feature space (geodesic dimensions) — ~20 dims:
- Realized vol (5d, 20d, 60d)
- Skew, kurtosis (20d, 60d)
- Drawdown (current, 60d max)
- Correlation matrix L2 distance from rolling avg
- Trend strength (Kendall tau or simple slope)
- Jump intensity (count over 20d)
- Vol-of-vol (20d)
- ... (curator-defined per asset class)

Each regime is a one-hot anchor in geodesic space, with slot index + small jitter. Live classifier embeds current market state and finds nearest anchor.

## Pipeline IR template — worked example for regime classification

```
@graph regime_classify_equity
  : in price_series -> Series<float>
  : out regime_distribution -> Distribution<RegimeLabel>
  | rv5 = realized_vol(series: <price_series>, window: 5)
  | rv20 = realized_vol(series: <price_series>, window: 20)
  | dd = current_drawdown(series: <price_series>)
  | acf = acf_lag1(series: <price_series>, window: 20)
  | trend = trend_strength(series: <price_series>, window: 60)
  | jumps = lee_mykland_jump(series: <price_series>, window: 20, threshold: 4.0)
  | features = pack_features(rv5: rv5.out, rv20: rv20.out, dd: dd.out,
                             acf: acf.out, trend: trend.out, jumps: jumps.out)
  | regime = geodesic_regime_classify(features: features.out)
  regime_distribution <- regime.out
@end
```

The `geodesic_regime_classify` primitive returns a `Distribution` — array of `(regime_label, probability)` pairs summing to 1. The probabilistic verifier passes through the entropy + top-1 confidence as auditable values.

## Backtesting harness — engineering investment

Probably the biggest hidden cost. Needs:

- **Walk-forward harness:** train on (t-N, t), test on (t, t+1), roll. No exception.
- **Lookahead-bias detector:** automated check that no primitive in the pipeline references future timestamps. The pipeline IR's typed graph makes this *enforceable* — add a `temporal:past_only` type on Series inputs and the verifier rejects forward-looking references.
- **Deflated Sharpe + p-hacking penalty:** apply Bailey-Lopez de Prado deflation when reporting backtest performance with > 1 strategy parameter.
- **Regime-aware splits:** can't just K-fold time series. Need block bootstrap or purged cross-validation.

Estimated ~4-6 weeks for a real backtest harness with lookahead-detection enforcement.

## 90-day-ish MVP plan (more like 5 months realistically)

| Weeks | Workstream | Deliverable |
|---|---|---|
| 1-3 | Time-series primitive library | `wiring_natives_finance.c` with 25 primitives + property tests |
| 4-6 | Probabilistic verifier extension | `pipeline_verify_with_confidence()` with calibration support |
| 7-9 | Regime taxonomy + geodesic anchors | 7 regimes, anchor-coords, live classifier |
| 10-12 | Backtest harness with lookahead enforcement | Walk-forward runner; lookahead-detector; deflated reporting |
| 13-15 | Backtest on 10y of S&P 500 + TSY 10Y + EUR/USD | Regime-transition lead-time vs vol-only baseline; explainability score |
| 16-20 | First customer demo / paper / pre-pitch material | Regime dashboard, audit-log explorer, paper draft for SSRN |

**Pass conditions:**
- Walk-forward backtest: regime detector identifies 2008, 2020, 2022 crisis transitions ≤ 5 trading days after a vol-of-vol-only baseline detects them (lead-time competitive)
- Lookahead-bias detector flags any primitive that uses `series[t+k]` for k>0 — zero false negatives in red-team tests
- Calibration: confidence distribution is well-calibrated (Brier score < 0.15 on held-out periods)
- Backtest reproducibility: same seed → identical results across two runs

## Customer profile

Not a tier-1 bank. They have $50M+ committed budgets to vendors like MSCI BarraOne, Bloomberg PORT, RiskMetrics. Decade-long contracts.

The right first customer is one of:
- A mid-tier asset manager ($5-50bn AUM) whose risk officer wants to migrate off Excel + ad-hoc Python
- A multi-strategy hedge fund that needs regime-aware position sizing and resents paying $500k/yr to a vendor
- A regulator (FCA, BaFin, SEC) building internal market-monitoring tools — they care about explainability above all else
- A clearing house or CCP needing real-time risk for margin calls
- A crypto fund (where regime transitions are particularly violent and the incumbents are absent)

The pitch: *"explainable, auditable, on-premise risk that you can extend yourself, at <10% the cost of MSCI / Bloomberg, with backtest-bias enforcement built in"*.

## Differentiation against incumbents

| Incumbent | Their strength | Where this product wins |
|---|---|---|
| MSCI BarraOne | Industry-standard factor risk | On-prem, no per-seat licensing, customer extensions allowed |
| Bloomberg PORT | Data network + tooling integration | Composability of risk pipelines; audit-trail-as-IR |
| Numerix / FINCAD | Derivatives valuation | Lower cost; Python-friendly; clearer audit |
| Internal risk teams (Excel + Python) | Customer ownership | Same ownership + typed verifiable IR + lookahead enforcement |

## Honest gaps for finance

1. **The probabilistic verifier is a research project.** Calibration, evidence-attribution, and the type-system change are non-trivial.
2. **Regulatory approval (SR 11-7 in US, MAR in EU)** of any risk model takes 6-18 months at any regulated bank. This is not skippable.
3. **Backtesting bugs are silent.** A subtly biased backtest looks great until production. Need adversarial review of the harness before any customer sees a number.
4. **Distribution access is hard.** Even with a great product, getting onto a buy-side risk officer's desk requires intro from someone they trust.
5. **Sales cycles are 12-24 months.** Plan revenue accordingly.
6. **The "tiny model" thesis fights the "use a real GBM / NN ensemble" expectation.** Risk teams expect sophisticated models. The pitch needs to make virtue of the smaller compute / better explainability story, not pretend to compete on raw accuracy.

## Decision triggers — when to abandon finance

- If the time-series primitive library takes > 8 weeks (vs 3-4 budget), there's a hidden architectural problem with stateful primitives in the pipeline IR.
- If the probabilistic verifier takes > 12 weeks or breaks the existing pipeline tests, that's a research project disguised as engineering — escalate or descope.
- If backtest of regime detector vs vol-only baseline shows < 1 trading day average lead-time on the three crisis exemplars, the regime taxonomy isn't capturing useful structure; revisit feature design before more engineering.
- If two senior risk officers in customer conversations say "we already have this" or "explainability is solved", the differentiator isn't there — pivot.
