# Vertical sketch: fraud detection

**Status:** working draft, follows from `docs/PRODUCTIZATION_VERTICALS.md`. Aim is to make the abstract "fraud is the cleanest vertical" claim concrete enough to act on.

## Why this vertical can ship in 90 days

The architecture's distinctive bets (composable typed graphs, deterministic verifier, edge deployment, ~75-80% retrieval over distinctive-vocabulary corpora) line up with how a real fraud system actually works:

- A fraud "rule" is literally a typed DAG: `transaction + cardholder profile + history → feature extraction → scoring → flag with reason code`. The pipeline IR encodes this directly.
- Every customer (issuer, processor, acquiring bank) has its own rule library. Today these live as untyped Drools / Python / SAS scripts. Migrating them to a typed graph IR with a verifier is **value the customer pays for** — auditable, edit-by-non-engineer, regulator-explainable.
- Fraud vocabulary is genuinely distinctive: `card-not-present`, `MCC`, `AVS-mismatch`, `velocity`, `chargeback`, `BIN`, `interchange`. The §44.5 finding (75-80% on distinctive-noun domains) holds here, and could plausibly hit 85-90% with deeper domain curation.

## Concrete anchor library — 20 starter families

Each family is a typed pipeline anchor. The classifier-keyword in parentheses is what the geo classifier (or TF-IDF) keys off; the primitive set is what the wiring/anchor mechanism produces.

| # | Family name | Primitive set | Distinctive nouns |
|---|---|---|---|
| 1 | `velocity_spike_24h` | `txn_count_window` → `compare_to_baseline` → `flag` | velocity, spike, 24h |
| 2 | `velocity_spike_1h` | `txn_count_window` → `compare_to_baseline` → `flag` | burst, rapid-fire, 1h |
| 3 | `geo_distance_anomaly` | `haversine` → `compare_to_home` → `flag` | geo, distance, home, far |
| 4 | `mcc_outlier` | `mcc_lookup` → `compare_to_cardholder_profile` → `flag` | MCC, category, outlier |
| 5 | `amount_z_score_anomaly` | `rolling_zscore` → `flag_if_above` | amount, z-score, deviation |
| 6 | `avs_mismatch_cluster` | `avs_check` → `cluster_in_window` → `flag` | AVS, mismatch, address |
| 7 | `cvv_failure_repeat` | `cvv_check` → `count_failures_window` → `flag` | CVV, failure, retry |
| 8 | `bin_attack_pattern` | `bin_lookup` → `sequential_pattern_check` → `flag` | BIN, sequential, enumeration |
| 9 | `cnp_in_high_risk_country` | `country_lookup` → `cnp_flag` → `flag` | card-not-present, CNP, high-risk-country |
| 10 | `merchant_first_seen_high_amount` | `merchant_history_lookup` → `flag_if_first_and_high` | merchant, first-seen, unfamiliar |
| 11 | `card_present_but_chip_disabled` | `terminal_capability_check` → `flag_if_chip_skipped` | terminal, fallback, magstripe |
| 12 | `chargeback_pattern_match` | `chargeback_history_lookup` → `pattern_match` → `flag` | chargeback, dispute, pattern |
| 13 | `dormant_card_sudden_activity` | `last_active_lookup` → `flag_if_dormant_and_active` | dormant, inactive, awakened |
| 14 | `multi_card_same_device` | `device_fingerprint_lookup` → `count_cards_window` → `flag` | device, fingerprint, multi-card |
| 15 | `account_takeover_credential_change` | `recent_credential_change` → `correlate_with_txn` → `flag` | account-takeover, ATO, credential |
| 16 | `synthetic_identity_pattern` | `identity_components_check` → `synthetic_score` → `flag` | synthetic, fabricated, identity |
| 17 | `gas_pump_skim_pattern` | `mcc_check_5541` → `cluster_geo_window` → `flag` | gas-pump, skimming, MCC-5541 |
| 18 | `refund_then_reuse_pattern` | `refund_lookup` → `txn_after_refund_check` → `flag` | refund, reuse, lap |
| 19 | `peer_to_peer_money_mule` | `p2p_recipient_check` → `flag_if_high_recipient_count` | money-mule, P2P, layering |
| 20 | `card_testing_low_value` | `low_value_window_check` → `flag_if_burst_low_value` | card-testing, low-value, probe |

Each anchor is ~10-30 lines of `@graph` syntax (see `wiring_anchor_graphs.c` for the form), with named primitives that need to exist in `wiring_natives_fraud.c`.

## New primitives needed in `wiring_natives_fraud.c`

~25 primitives, all stateful (they carry per-cardholder rolling state):

- **Window aggregates:** `txn_count_window(card_id, hours) → int`, `txn_sum_window(card_id, hours) → int64`, `txn_amount_max_window(card_id, hours) → int64`
- **Profile lookups:** `cardholder_home_geo(card_id) → (lat, lon)`, `cardholder_baseline_velocity(card_id) → int`, `cardholder_baseline_amount(card_id) → int`
- **Geo:** `haversine(lat1, lon1, lat2, lon2) → meters`
- **Statistical:** `rolling_zscore(card_id, value, window) → float`, `rolling_quantile(card_id, value, window, q) → float`
- **Lookup:** `mcc_category(mcc) → string`, `bin_country(bin) → string`, `device_fingerprint_lookup(card_id, device_id) → bool`
- **Pattern:** `sequential_pattern_check(values) → bool`, `cluster_in_window(events, distance_threshold, time_window) → int`
- **Decision:** `flag(severity, reason_code, contributing_features) → flag_record`

Total scope: ~3-4 weeks of implementation + tests. None requires research — these are well-defined operations.

## Schema

Three core types:

```c
typedef struct {
    char     transaction_id[32];
    int64_t  timestamp;
    int64_t  amount_cents;
    char     currency[4];
    char     card_id[32];
    int      mcc;
    char     merchant_id[32];
    double   merchant_lat;
    double   merchant_lon;
    char     entry_mode;        /* 'P'=card-present, 'N'=CNP, 'F'=fallback */
    char     avs_result;
    char     cvv_result;
    /* ... ~40 fields total per real-world EMV spec */
} Transaction;

typedef struct {
    char     card_id[32];
    double   home_lat, home_lon;
    int64_t  baseline_amount_p50, baseline_amount_p95;
    int      baseline_velocity_24h;
    int      account_age_days;
    /* ... rolling state populated by the cardholder-profile worker */
} CardholderProfile;

typedef struct {
    char     transaction_id[32];
    int      severity;           /* 0=clear, 1=watch, 2=hold, 3=block */
    char     reason_code[8];     /* 'V24H', 'GEOA', 'MCCO' ... matches family names */
    char     contributing_features[256];  /* JSON list of (primitive, value) */
    char     anchor_family[32];  /* which family fired */
    /* For audit: */
    char     pipeline_dot[2048]; /* DOT-rendered DAG of the decision */
    int64_t  decision_timestamp;
    char     model_version[16];
} FlagRecord;
```

## Pipeline IR template — worked example

For family #1 (`velocity_spike_24h`):

```
@graph velocity_spike_24h
  : in txn -> Transaction
  : in profile -> CardholderProfile
  : out flag -> FlagRecord
  | count = txn_count_window(card_id: <txn>.card_id, hours: 24)
                :: card_id:str, hours:int -> out:int
  | baseline = cardholder_baseline_velocity(card_id: <txn>.card_id)
                :: card_id:str -> out:int
  | ratio = divide(x: count.out, y: baseline.out)
                :: x:int, y:int -> out:float
  | trip = flag_if_above(value: ratio.out, threshold: 3.0,
                         severity: 2, reason_code: "V24H")
                :: value:float, threshold:float, severity:int, reason_code:str -> out:FlagRecord
  flag <- trip.out
@end
```

The verifier checks: types match, no cycles, all input ports connected, reason_code is in known set. Produces the audit DOT automatically.

## Audit log format

Append-only, write-once, hash-chained:

```
{
  "ts": 1714512000123,
  "transaction_id": "TXN-202604300001",
  "anchor_family": "velocity_spike_24h",
  "decision": "hold",
  "severity": 2,
  "reason_code": "V24H",
  "contributing": [
    {"primitive": "txn_count_window", "args": {"hours": 24}, "value": 47},
    {"primitive": "cardholder_baseline_velocity", "value": 12},
    {"primitive": "ratio", "value": 3.92}
  ],
  "pipeline_dot_sha256": "a3f...",
  "model_version": "fraud-anchors-v1.2",
  "prev_log_hash": "8c1...",
  "this_log_hash": "9f4..."
}
```

The hash chain makes tampering detectable (if any past entry is altered, all subsequent hashes break). This is a 1-month engineering job — not novel cryptography, just careful append-only persistence.

## 90-day MVP plan (refined)

| Weeks | Workstream | Deliverable |
|---|---|---|
| 1-2 | Schema + primitive library scaffolding | `wiring_natives_fraud.c` with 25 primitives + tests; transaction/cardholder/flag types in headers |
| 3-4 | 20-family anchor library | `wiring_anchor_graphs_fraud.c` with 20 entries; held-out test set built BEFORE measurement (see `tools/scaling_leakage_audit.sh`) |
| 5-6 | Streaming adapter | Kafka or HTTP-POST input adapter; pluggable so a customer with FIS / TSYS / Adyen webhooks can swap easily |
| 7-8 | Audit log + persistence | Hash-chained append-only log; SQLite for cardholder profile state; recovery on restart |
| 9-10 | Integration: end-to-end on PaySim | Demo: feed PaySim transactions through the pipeline, produce flags + audit log; measure precision/recall/p99 latency |
| 11-12 | First customer-facing demo | Replace PaySim with a customer's NDA-protected sample dataset; produce a comparison vs their existing rules engine |

**Pass conditions:**
- ≤ 5ms p99 decision latency on a single CPU core (achievable for 540K-param model + ~25 primitives in the hot path)
- ≥ 95% of analyst-reviewed flags have an audit log entry that an analyst can explain in < 30 seconds
- Precision/recall within 5pp of the customer's existing baseline at the same threshold
- Zero hash-chain breaks under deliberate tampering tests

## Customer profile

The right first customer is **NOT** a top-10 bank. They have entrenched vendors (NICE Actimize, FICO Falcon, SAS, Featurespace) and procurement cycles measured in years.

The right first customer is one of:
- A mid-sized payment processor (~$10-50bn / yr volume) with an in-house rules team that's frustrated with their current tooling
- A neobank that's grown past its initial fraud-by-thresholding stage and needs auditable rules for regulators
- A fintech challenger in EMEA where PSD2 strong-customer-authentication is forcing rule rewrites anyway
- A card issuer-as-a-service platform (Marqeta, Galileo, Stripe Issuing) that wants to differentiate on explainability

The pitch isn't *"more accurate than your model"*. It's *"comparable accuracy, with audit trails your regulator already wants and rules your fraud analysts can edit without an engineer"*.

## Differentiation against incumbents

| Incumbent | Their strength | Where this product wins |
|---|---|---|
| FICO Falcon | Mature ML + rich data network | Edge deployment; on-prem; analyst-editable rules; transparent audit |
| NICE Actimize | Broad case-management suite | Lower TCO; sub-5ms decisions; no per-decision SaaS bill |
| Featurespace ARIC | Adaptive behavioural analytics | Deterministic verifiable rules (regulators love); white-box explanations |
| In-house rules engines | Customer owns the IP | Same ownership + composable typed IR + audit infrastructure |

## Honest gaps for fraud

1. **Real fraud datasets are gated.** PaySim/IEEE-CIS are toy. Need a paid POC with a real customer to validate at scale.
2. **State management is genuinely new infrastructure.** The current architecture is stateless per-prompt; cardholder profile windows + rolling stats need a persistence layer.
3. **Tamper-evident logging is not novel cryptography but it's not free either.** Get a security review before pitching to a regulated customer.
4. **Adversarial fraudsters will probe.** Need rate-limit + perturbation defences; today the architecture has none.
5. **The wiring transformer's NL interface is mostly cosmetic for fraud.** Analysts will edit the YAML/JSON anchor library directly; NL → graph is a "demo wow" feature, not a core sale.

## Decision triggers — when to abandon fraud

Pre-register, in the spirit of the scaling-curve work:
- If the first 5 anchored families take > 2 weeks each (vs the 1-week budget), the primitive library design has a hidden problem; pause and refactor before continuing.
- If PaySim end-to-end shows precision/recall worse than a 50-line XGBoost baseline, the architecture isn't carrying its weight; reassess.
- If three serious customer conversations all say "we don't actually need explainability that much, we need recall", pivot to a different vertical or a different angle.
