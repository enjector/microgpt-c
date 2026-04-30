# From Research to Product — three verticals where the architecture earns its keep

**Status:** strategic working document, not a roadmap commitment. Written 2026-04-30 after the scaling-curve arc closed (`docs/research/wiring_scaling_post_phase3.md`). Honest about gaps; aim is to surface trade-offs, not sell a story.

## TL;DR

The microGPT-C architecture's distinctive bets — *tiny specialists coordinated by a deterministic Judge, manifold retrieval, typed pipeline composition, ~30K–540K param models that run on a laptop* — are wrong for general-purpose AI work and right for a specific class of vertical problems: **regulated, on-edge, compositional, audit-trail-required problems where explainability beats accuracy**.

Three verticals fit this shape: **fraud detection**, **finance market regime / risk detection**, and **defence digital-twin object tracking**. Each has different gap profiles. The honest order of "shortest path to a real product" is fraud → finance → defence, with each successive vertical requiring more new infrastructure.

The single largest unblocker across all three is **dropping the "pure C99, zero deps" project policy** (or formalising a thin dependency boundary). Without that, the system can't ingest real-world data sources or interoperate with the systems-of-record each vertical lives in. With it, all three become tractable 6-12 month investments.

## What the research actually built (and what it didn't)

| Reusable component | Verticals it's directly relevant to | Maturity |
|---|---|---|
| **Pipeline IR + verifier** (`src/microgpt_pipeline.{h,c}`) | All three — typed DAG composition with deterministic verifier is the core "compositional explainable orchestration" primitive | Stable, 51/51 unit tests, used by Wiring Organelle and Phase 3b composition |
| **Geodesic metric infrastructure** (`src/microgpt_geodesic.{h,c}`) | Fraud (existing `geo_metric_fraud`), finance (regime distance), defence (trajectory state-space) | 16/16 tests, GEO_DIMS=40, used by anchor classifier |
| **Multi-organelle Kanban orchestration** (`src/microgpt_organelle.{h,c}`) | Any pipeline of specialist roles (planner / player / judge / triage / escalate) | Production-ready in 11 game demos |
| **TF-IDF retrieval + corpus expansion** | Fraud (transaction-description matching), finance (event-type classification) | ~80% ceiling on distinctive-vocabulary domains, ~15% on generic-vocabulary domains |
| **Wiring Organelle (NL → graph)** | Any vertical where users describe a workflow in English and want a typed DAG built deterministically | 75-80% on distinctive-vocabulary novel paraphrases; lower elsewhere |
| **Memory Sparse Attention (MSA)** | Finance (long-history time-series), defence (long trajectory windows) | Production, with rope-aware helpers |
| **TurboQuant / RotorQuant KV compression** | Edge-deployed inference where memory matters | Working but only worthwhile for `N_EMBD ≥ 512` |
| **DeepSeek-V4 port stack** (partial RoPE / attn sink / QK-norm / MSA pool) | Any deployed transformer in the stack | -8.7% PPL combined, free at deployment |

| Genuinely missing | Verticals blocked | Cost to build |
|---|---|---|
| Streaming / real-time data ingestion (Kafka, FIX, NMEA, message-bus) | All three | 1-2 eng-months per stream type |
| Time-series primitives in `wiring_natives.c` (rolling stats, change-point, ARIMA) | Finance, defence | 3-6 weeks |
| Probabilistic / fuzzy verifier outputs (today verifier is binary: pass/fail) | Finance regime detection (which is unsupervised) | 1-2 eng-months |
| External embedding / semantic similarity (per Post-Phase-3 #3 finding) | Any vertical needing wide-domain coverage | 2-4 weeks if a real dep is allowed |
| Audit / compliance log surface (write-once, tamper-evident, regulator-readable) | Fraud, finance | 1 eng-month |
| Sensor adapters (vision, RADAR, ADS-B, AIS) | Defence | 2-4 eng-months per sensor type |

## Productization principles for this architecture

These are the rules a vertical product built on microGPT-C should follow if it wants to keep the architecture's distinctive value:

1. **Compose, don't reach for a bigger model.** The thesis that earned the research credibility is "tiny specialists coordinated by a Judge beat one big model on focused tasks." Throwing GPT-4 at the problem defeats the whole point. Each new capability should be a new specialist with a clear contract, not a finetune of the wiring transformer.
2. **Verify before you act.** Every output that affects the world (a fraud flag, a trade halt, a target track) should pass through `pipeline_verify()` or its domain analogue. Audit trails come for free.
3. **Stay on-device by default.** The architecture is small enough to run on a single laptop or embedded SoC. Cloud-only deployments should be a deliberate choice, not the default.
4. **Make retrieval the first-class citizen.** Phase 4 showed that retrieval over a curated library (with a Judge filtering wrong answers) beats generation. Build the per-vertical anchor table early; treat the wiring transformer as a fallback, not the front line.
5. **Audit your own claims.** The leakage audit (`tools/scaling_leakage_audit.sh`) found that 19/20 of the original "1:1 scaling" result was inflated by curator self-overlap. Whatever metrics a vertical product reports, audit them with the same discipline before they ship.

## Vertical 1: Fraud detection

### Why this vertical fits the architecture

Fraud detection is the cleanest fit for what microGPT-C already is:

- **Compositional rules.** A real fraud system is a graph: `(geo distance from cardholder home > X) AND (velocity over last hour > Y) AND (merchant category code in high-risk set) → flag`. The pipeline IR is exactly this with typed edges and a verifier.
- **Explainability mandatory.** Regulated industry — a `flag` decision must be explainable to an analyst, a regulator, and a chargeback dispute panel. Pipeline IR's typed DAG → DOT renderer is already a compliance-ready audit trail.
- **Edge constraints common.** ATM-side, POS-side, and embedded-card-reader fraud screening must run on hardware with no cloud. A 540K-param wiring organelle + retrieval table fits in <2MB.
- **Distinctive vocabulary.** "Velocity", "card-present-flag", "MCC code", "AVS mismatch" are domain-specific nouns that score well under the §44.5 finding (~80% retrieval ceiling). Fraud is the kind of domain where TF-IDF + curator's synonyms can plausibly hit 85-90% with a properly distinct corpus.
- **`geo_metric_fraud` already exists.** The geodesic infrastructure was scoped with fraud in mind. Real risk-scoring algorithms (Mahalanobis distance from baseline, charge-graph distance, behavioral-profile drift) translate directly.

### What a 90-day MVP looks like

Phase 1 (weeks 1-4): **schema + corpus**
- Define the typed primitives: `transaction(timestamp, amount, mcc, merchant_id, ...)`, `cardholder_profile(home_geo, baseline_velocity, ...)`, `flag(severity, reason_code, contributing_features)`.
- Build a 20-family fraud anchor library (cardholder-velocity-spike, MCC-deviation, geo-distance-anomaly, AVS-mismatch-cluster, chargeback-pattern-match, etc.). This is where the §44.5 finding applies directly — distinctive nouns per family.
- Pre-register held-out paraphrases, run `tools/scaling_leakage_audit.sh` BEFORE any measurement.

Phase 2 (weeks 5-8): **adapter + first deployment**
- Build a Kafka consumer (or whatever the customer's transaction stream is). This is the first real dependency the project takes on.
- Wire the pipeline-IR Judge as a synchronous filter on each transaction. Latency budget: target ≤ 5ms p99 (achievable for a 540K-param model on modern CPU).
- Compliance audit log: every `flag` decision writes the (transaction, predicted family, anchor graph used, verifier result, contributing primitives) tuple to a write-once log.

Phase 3 (weeks 9-12): **measurement + pilot**
- Acquire a labelled fraud dataset (PaySim, IEEE-CIS, or a customer's historical labels under NDA).
- Measure: false-positive rate, true-positive rate, time-to-decision, % of flags an analyst can explain from the audit log alone.
- Compare against the customer's existing rules-based or XGBoost baseline. The architecture's pitch is **not "higher accuracy"** — it's **"comparable accuracy + auditable decisions + edge-deployable + composable rules an analyst can edit"**.

### Honest gaps for fraud

- **Dataset access is the bottleneck.** PaySim is a toy. Real fraud labels are jealously guarded. First customer deal probably starts as a paid POC under NDA.
- **Real-time stream integration is genuine new infrastructure** (Kafka, gRPC, etc.) and breaks the zero-deps policy.
- **The wiring transformer's natural-language interface is a "nice-to-have"**, not a core capability for fraud — analysts will mostly edit the anchor library directly via a UI, not via NL prompts. This means Phase 3b composition (the multi-stage chains) is also lower-priority for fraud.
- **Adversarial fraudsters will read your audit log if leaked.** Tamper-evident logging (Merkle tree, signed entries) is a real engineering investment.

### Honest fit verdict for fraud

**The single best vertical fit.** The architecture's strengths (composable typed graphs, deterministic Judge, edge deployment, explainable retrieval) line up cleanly with the customer's needs (audit, compliance, low latency, on-device, analyst-editable rules). Most of the engineering investment is in adapters and ops, not in research.

## Vertical 2: Finance market regime / risk detection

### Why this vertical partially fits

Finance is a more ambitious application of the same architecture:

- **Composition of risk calculations.** A market-risk pipeline is genuinely a DAG: `prices → returns → covariance matrix → portfolio variance → expected shortfall`. The pipeline IR composes this naturally.
- **Geodesic regime detection.** Market regimes (low-vol, trending, mean-reverting, crisis) are positions in a feature space. Detecting transitions is geodesic distance between the current state and a regime anchor — exactly what the geodesic infrastructure was designed for.
- **MSA for long-horizon context.** A regime-detection model needs to consider months of history compactly. MSA's LRU-paged latent storage is the kind of compact long-context primitive this needs.
- **Explainability mandatory** (basically every financial regulator wants this — MAR, MiFID II, SR 11-7).

### What's harder than fraud

- **Probabilistic, not binary.** A regime detector outputs a *probability distribution* over regimes, not a `flag/no-flag` decision. The pipeline verifier currently returns binary pass/fail. Need to add probabilistic outputs (e.g. `verify_with_confidence` returning a `[0, 1]` belief score).
- **Unsupervised / semi-supervised.** "What regime is the market in *now*" doesn't have ground truth labels at decision time. The architecture's Judge approach (verify against expected primitive set) doesn't directly apply. You'd need:
  - A regime taxonomy (curator-defined, e.g. 5-10 regimes)
  - Labelled historical periods for training the regime classifier
  - A live drift detector (changepoint over the geodesic state)
- **Time-series primitives missing.** `wiring_natives.c` has scalar arithmetic (multiply, divide, percentage) but no rolling-window stats, no exponential moving averages, no GARCH, no jump-detection. Real risk uses these. Building them is mechanical (~3-6 weeks) but it's net-new code.
- **Latency / throughput targets are different.** Fraud is per-transaction (5ms p99). Risk is per-portfolio-snapshot (could be daily, intraday, or per-tick depending on use case). Need to design for the right tier.

### What a 90-day MVP looks like

Phase 1 (weeks 1-6): **time-series primitive library + regime taxonomy**
- Build `wiring_natives_finance.c`: rolling mean/std, EWMA, realized variance, simple jump detector (e.g. Lee-Mykland), correlation rolling, drawdown.
- Define 5-7 named regimes (low-vol-trend, low-vol-range, high-vol-crisis, mean-reverting, momentum). Each is an anchor in the geodesic space with a curated set of primitive outputs.
- Pre-register held-out periods (e.g. "the 2020 March COVID crisis", "the 2022 LDI crisis", "Q1 2023 banking stress") with their expected regime label.

Phase 2 (weeks 7-10): **regime classifier + risk pipeline**
- Map each regime to a geodesic anchor (slot + jitter, similar to family classifier).
- Build a classifier that takes (rolling realized vol, drawdown, correlation, trend strength) → regime distribution.
- Build pipeline DAG templates for typical risk reports (VaR, ES, factor exposures).

Phase 3 (weeks 11-13): **measurement + comparison**
- Backtest on 10 years of one major asset class (S&P 500, TSY 10Y, EUR/USD).
- Measure: regime-transition detection lead-time vs a baseline (e.g. simple vol-based regime), false-transition rate, explainability (can a risk analyst trace why the model went from regime A to regime B from the audit log?).

### Honest gaps for finance

- **The probabilistic-verifier change is non-trivial.** It's a research question in itself, not just engineering.
- **Backtesting infrastructure** (lookahead bias prevention, deflated Sharpe, walk-forward) is its own engineering investment.
- **Customer integration** (Bloomberg, Refinitiv, internal tick stores) is heterogeneous and slow.
- **Regulatory approval** for a risk model takes 6-18 months at a bank, regardless of how good the model is.
- **Sales cycle** for buy-side / sell-side risk software is 12-24 months.

### Honest fit verdict for finance

**Plausible second vertical, with significantly more research-feeling work upfront than fraud.** The architecture has the right shape but needs ~2-3 months of net-new infrastructure (probabilistic verifier, time-series primitives, regime taxonomy) before a useful prototype exists. The sales cycle and regulatory cycle are also longer. **Don't start here unless you have a deep finance partner already.**

## Vertical 3: Defence digital-twin object tracking

### Why this vertical fits the *spirit* but barely the *substance*

Defence is the most ambitious of the three and the least direct fit:

- **Compositional reasoning** — yes, a target-tracking pipeline is a DAG (`sensor reads → fuse → predict → classify → assess threat → recommend action`).
- **Explainable AI is mandatory** — UK MOD's "Defence AI Strategy", US DOD's RAI principles all require documented decision trails.
- **Edge deployment is genuine** — sensor platforms can't always phone home; latency must be sub-100ms; SWaP-C constraints are real.
- **Geodesic state tracking** — modelling object trajectories in N-dim state space (position, velocity, IFF, signature features) and detecting deviations from expected behaviour is mathematically related to the existing geodesic work.

But the gaps are bigger:

- **No vision / sensor / radar primitives.** Defence object tracking starts with raw sensor data: imagery, ADS-B, AIS, radar contacts, RF emissions. None of these are in `wiring_natives.c`. Building a compliant sensor adapter is ~2-4 eng-months *per sensor type*.
- **Multi-object tracking is its own subfield.** Joint Probabilistic Data Association, Multi-Hypothesis Tracking, Random Finite Sets — these are 30 years of mature literature. The architecture doesn't have any of this.
- **Real-time constraints are stricter than fraud.** sub-100ms end-to-end with thousands of contacts is a real performance budget that needs profiling and likely Metal/CUDA acceleration.
- **Procurement, security clearance, IL ratings.** A defence customer can't just download the GitHub repo. Selling here requires UK Defence Cyber accreditation, IL3+ hosting, ITAR or equivalent export-control review for crypto and ML weights, formal V&V evidence, and a years-long sales cycle.

### What a credible defence offering would actually look like

Not a 90-day MVP — a 12-18 month investment with a defence prime as a partner. The realistic shape:

- **Phase 0** (months 1-3): partner identification. Find a UK or US prime with an existing digital-twin programme (DDS, JADC2, Maven, Project Convergence) where compositional / explainable reasoning is missing.
- **Phase 1** (months 4-9): build one sensor adapter (probably AIS — civilian-equivalent maritime tracking — for early dev without classified data). Build the multi-object-tracking primitives. Prove the pipeline-IR + geodesic stack can ingest, fuse, predict, and classify on AIS data with auditable per-track explanations.
- **Phase 2** (months 10-15): partnership-led port to a real defence dataset. Performance optimisation. Security accreditation prep.
- **Phase 3** (months 16-24): formal V&V, accreditation, first paid pilot.

### Honest fit verdict for defence

**Right architecture for the problem class, wrong project shape today.** The composition + verification + audit story is exactly what defence wants for AI, but the engineering distance from microGPT-C-as-it-stands to a deployable system is ~12-18 months and requires a defence prime as partner. **Pick this only if you have a partner already lined up; do not start cold.**

## Cross-vertical engineering investments

Three pieces of work would benefit all three verticals (and any future vertical):

1. **External-embedding integration** (~2-4 weeks): per Post-Phase-3 #3 finding, the bag-of-features TF-IDF ceiling at 75-80% is real. Integrating a small pretrained sentence embedder (sentence-transformers via ONNX, or fastText vectors) gives the architecture genuine semantic similarity for retrieval. **Breaks zero-deps policy.** Critical for fraud (transaction descriptions vary widely), useful for finance, neutral for defence.
2. **Probabilistic verifier output** (~1-2 eng-months): change `pipeline_verify()` to return a confidence score in addition to pass/fail. Critical for finance regime detection, useful for fraud (graded suspicion levels), useful for defence (target identification confidence).
3. **Audit log surface** (~1 eng-month): write-once, tamper-evident, regulator-readable log of every Judge decision. Required for all three verticals.

Pick these in this order. Each is independently shippable and unblocks downstream vertical work.

## Decision framework: which vertical first?

| Criterion | Fraud | Finance | Defence |
|---|---|---|---|
| Time-to-first-prototype | 3 months | 6 months | 12-18 months |
| Net-new infrastructure required | Adapters + audit log | Time-series primitives + probabilistic verifier + adapters | Sensor adapters + multi-object tracking + V&V |
| Sales cycle | 3-6 months | 12-24 months | 24-48 months |
| Architecture fit (1-5) | 5 | 4 | 3 |
| Dataset access | Hard but solvable (paid POC) | Doable (Bloomberg, FRED, customer) | Very hard (clearance, partnership) |
| Regulatory friction | Medium (PCI, SOX) | High (MAR, MiFID, SR 11-7) | Very high (defence accreditation) |
| Differentiation vs incumbents | High (audit + edge) | Medium (incumbents have explainability stories too) | High (most defence AI is opaque) |

**Recommended order: fraud → finance → defence.** Each successive vertical re-uses the cross-cutting investments (#1-3 above) and validates the architecture in progressively harder contexts.

The fraud vertical is the genuine first move because:
- It's the only one where 90 days from now you could have a real customer pilot
- The architecture's strengths (audit, edge, composition) are direct customer pain points
- Failure modes are tolerable — false positives waste analyst time, not lives or capital
- The infrastructure investments needed (adapters, audit log) are vertical-portable

## What this document does NOT commit to

- **Specific revenue projections** — depends entirely on customer access and which prime / bank / processor you can land first.
- **Open-source vs commercial licensing** — the project is currently MIT-flavoured permissive. Commercialisation may need a dual-license scheme (e.g. AGPL + paid commercial), which is a separate strategic call.
- **Hiring** — none of the above is doable as a single-engineer project for more than the first prototype phase. Each vertical past prototype needs at least one domain expert.
- **Anthropic-Claude-API or other LLM dependency** — the project is presently independent of any LLM. Productization may want to selectively integrate an LLM for the natural-language UX layer (prompt → wiring), but that's an explicit decision to make, not a default.

## Honest closing

The microGPT-C architecture is *small* and *composable* and *auditable* in ways that the dominant LLM-based AI products are not. That's a real differentiator in regulated, edge-constrained, compositional verticals. It's also a real handicap in any vertical that needs broad-domain understanding or maximum accuracy on open inputs.

The three verticals named above are exactly the ones where the differentiator dominates the handicap. Fraud is the cleanest, fastest path. Finance is harder but plausible. Defence is right-shape but wrong-shape-today.

The biggest single decision in front of you isn't which vertical — it's **whether to drop the "pure C99, zero deps" research-project policy** to enable any of them. That policy was the right call for research (kept the codebase pedagogically clean, made claims reproducible). For product it's a constraint you'll trip over within the first month of any vertical work. The replacement should be a *thin, deliberate* dependency boundary — name the few external libraries you'll allow (Kafka client / a sensor SDK / an embedding library) and treat them as non-negotiable contracts, not creeping dependency churn.
