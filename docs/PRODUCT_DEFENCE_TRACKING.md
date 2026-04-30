# Vertical sketch: defence digital-twin object tracking

**Status:** working draft, follows from `docs/PRODUCTIZATION_VERTICALS.md`. The most ambitious vertical and the slowest to prove out — right architectural shape, wrong project shape today. Realistic only with a defence prime as a partner from day one.

## Why this vertical fits the architectural *spirit*

Modern defence digital-twin systems (UK Defence Digital's data-fabric, US JADC2 / DDIL pieces, Project Maven, Project Convergence) are converging on a common need: **compose multiple sensor-derived models into auditable decision pipelines that can run on-edge with sub-100ms latency and survive disconnected operations**.

That description is essentially the microGPT-C architecture re-pitched in defence vocabulary:

| Architecture feature | Defence digital-twin need |
|---|---|
| Pipeline IR + typed graph + verifier | Compositional sensor → fusion → classification → threat-assessment DAGs with provable types |
| Geodesic state-space metrics | Object-trajectory state in N-dim space; deviation from expected behaviour via geodesic distance |
| Multi-organelle Kanban | Coordinator → tracker → classifier → escalator pipeline with handoffs |
| Tiny specialists, edge deployable | SWaP-C constrained platforms (UAVs, vehicle-mounted, dismounted infantry) |
| Deterministic Judge + audit | RAI compliance (US DoD), Defence AI Strategy (UK MOD), Article 36 weapons review |
| MSA / TurboQuant | Long trajectory windows compressed for memory-constrained platforms |

## Why the *substance* doesn't yet fit

Three real gaps that aren't bridgeable in 90 days:

### Gap 1: no sensor primitives

`wiring_natives.c` does scalar arithmetic. Defence object tracking starts with raw sensor data:
- **Imagery** (visible, IR, SAR): bounding boxes, segmentation masks, classification logits
- **ADS-B / AIS / IFF**: structured contact reports
- **Radar** (search, track, fire-control): plot extracts, track files
- **RF / SIGINT**: emitter classification, geolocation
- **Acoustic**: bearing-only tracks

Each is a real engineering subdiscipline. Building a compliant adapter for *one* sensor type is ~2-4 engineer-months. A useful product needs at least three.

### Gap 2: no multi-object tracking infrastructure

Multi-object tracking is 30+ years of mature literature: Joint Probabilistic Data Association (JPDA), Multi-Hypothesis Tracking (MHT), Random Finite Sets / Probability Hypothesis Density (PHD/CPHD), Generalized Labelled Multi-Bernoulli (GLMB). Real systems use these.

The architecture has zero of this. Building from scratch is a significant project (3-6 engineer-months for a competent baseline). Re-using an existing C/C++ library (e.g. Stone Soup's C bindings if they exist, or rolling our own GLMB) is faster but adds a dependency.

### Gap 3: deployment realities

A defence customer cannot:
- Download from GitHub
- Use a cloud-only product
- Tolerate unauditable models
- Skip formal V&V, security accreditation, IL3+ hosting, ITAR / export-control review

This isn't a tech gap — it's a business-model and ops gap. Selling defence requires:
- A defence prime or system integrator as a partner from day one (Lockheed, Raytheon, Northrop, BAE, Thales, Leonardo, QinetiQ, ...)
- UK MOD List X / US DSS facility clearance for any classified work
- Cleared engineers (UK SC/DV; US Secret/TS-SCI) on the team
- Formal evidence: DO-178C-equivalent, IEC 62443 for industrial control, AAQR-equivalent for autonomous platforms
- Sales cycles measured in years, not quarters

## The realistic shape: 12-18 month horizon, partner-led

Not a 90-day MVP. This is a multi-phase commitment:

### Phase 0 (months 1-3): partner identification

Find a defence prime or system integrator that:
- Has an active digital-twin / multi-domain-operations / autonomy programme
- Is publicly known to be looking for compositional / explainable AI
- Has a procurement vehicle that allows a small specialist to participate (DASA in UK; SBIR/STTR in US; OTA in DOD)

Candidates worth approaching (publicly known to have relevant programmes):
- UK: Defence Science and Technology Laboratory (Dstl), Defence AI Centre (DAIC), Defence Digital, BAE Systems Digital Intelligence, QinetiQ
- US: AFWERX / SOFWERX, DIU, AFRL, NavSEA innovation arms, primes' AI/autonomy divisions
- Both: NATO Allied Command Transformation's emerging-tech programmes

The right partner provides: domain expertise, security cover, customer access, regulatory navigation. In exchange they want: differentiation, IP rights structure, joint go-to-market.

### Phase 1 (months 4-9): AIS demo as a pre-cleared first proof

Before touching anything classified, build the architecture's first defence-flavoured demo on **civilian AIS** (Automatic Identification System — open maritime tracking data via APIs like aisstream.io, MarineTraffic, or recorded NOAA datasets).

What to build:
- AIS adapter (parse NMEA-style messages, ingest 1000s of contacts/sec)
- Multi-track maintenance (associate AIS messages to tracks; handle gaps; spawn / kill tracks)
- Behavioural anomaly detection: "this vessel changed course outside its normal pattern"; "this AIS-off period followed by reappearance is suspicious"; "this fishing vessel entered a marine protected zone"
- Pipeline IR templates expressing each above as a typed DAG with auditable explanations
- A demo dashboard showing: live tracks, anomaly flags with audit explanations, on-laptop performance numbers

Why AIS:
- **Civilian** — no clearance needed for development
- **Real-world complexity** — millions of vessels, GPS noise, AIS spoofing, course-change clusters
- **Genuinely useful even outside defence** — coastguards, MPA enforcement, port authorities, insurers care
- **Direct architectural rehearsal** for the eventual classified version

### Phase 2 (months 10-15): partnership-led port

With the AIS demo as a tangible artefact, the prime partner can sponsor a port to a real defence dataset behind their security boundary. At this point engineering happens on the partner's accredited infrastructure, not on the open repo. Performance optimisation, MIL-STD interfaces, security accreditation prep.

### Phase 3 (months 16-24): formal V&V, accreditation, first paid pilot

V&V evidence package, security accreditation review, first paid pilot under a Phase II SBIR / DASA Phase 2 / equivalent. Revenue starts here, not before.

## Multi-object tracking primitive set (the engineering core)

For the AIS demo and for any defence offering, build a tracking core in `wiring_natives_tracking.c`:

| Primitive | Purpose |
|---|---|
| `nearest_neighbour_associate(measurements, tracks, gate)` | Simplest data association — works for AIS, weak for radar in clutter |
| `jpda_associate(measurements, tracks, clutter_model)` | Probabilistic data association — handles ambiguous returns |
| `kalman_predict(track, dt)` | Linear motion model prediction |
| `imm_predict(track, dt, models)` | Interacting Multiple Models for manoeuvring targets |
| `track_initiate(measurements_stream, m_of_n_window)` | Spawn tentative tracks; promote on M-of-N detections |
| `track_terminate(track, miss_count_threshold)` | Drop stale tracks |
| `track_classify(track, classifier_model)` | Apply a classifier (transformer or rule-based) to a track's history |
| `behavioural_anomaly_score(track, baseline_model)` | Geodesic distance from track's expected behaviour |
| `geofence_check(track, polygon)` | Has the track entered a forbidden / monitored region? |
| `course_change_detector(track, window, threshold)` | Detect significant heading changes |
| `ais_off_then_on_pattern(vessel_id, gap_threshold)` | Identify suspicious AIS dropouts |
| `convoy_detection(tracks, spacing, parallel_courses)` | Group tracks into convoys / formations |

Total scope: ~6 engineer-months for a competent baseline. Reuses pipeline IR + geodesic infrastructure throughout.

## Pipeline IR template — worked example for AIS anomaly

```
@graph ais_off_then_on_anomaly
  : in vessel_track -> Track<AISContact>
  : out anomaly_flag -> AnomalyRecord
  | gap = ais_off_then_on_pattern(track: <vessel_track>, gap_threshold: 1800)
                :: track:Track, gap_threshold:int -> out:GapEvent
  | reappearance_jump = course_change_detector(track: <vessel_track>,
                                                window: 300, threshold: 30.0)
                :: track:Track, window:int, threshold:float -> out:JumpEvent
  | high_value_zone = geofence_check(track: <vessel_track>,
                                      polygon: HIGH_VALUE_AREAS)
                :: track:Track, polygon:Polygon -> out:bool
  | combined = anomaly_combine(events: [gap.out, reappearance_jump.out],
                                modifiers: [high_value_zone.out])
                :: events:list<Event>, modifiers:list<bool> -> out:AnomalyScore
  | flag = flag_if_above(value: combined.out.score, threshold: 0.8,
                          severity: 2, reason_code: "AIS_OFF_REAP_HVZ")
                :: value:float, threshold:float, severity:int, reason_code:str -> out:AnomalyRecord
  anomaly_flag <- flag.out
@end
```

The verifier checks types, no cycles, all evidence flows into the final flag. The DOT renderer produces an analyst-readable explanation: *"this vessel went AIS-off for 35 min, reappeared with a 47° course change, while inside a high-value-area polygon → anomaly score 0.84 → flag"*.

## Realistic risk register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| No prime partner materialises | Medium | Fatal | Don't start cold — only commit if a partner is in conversation |
| Sensor primitives take 4x estimate | High | Schedule slip | AIS-only first scope reduces this; defer SIGINT/RADAR to Phase 2+ |
| Multi-object-tracking quality insufficient for real customers | Medium | Need to license existing MOT library | Acceptable; allowed dependency in the policy |
| Security accreditation rejected | Medium | Cannot deploy on classified networks | Mitigated by partner-led process, not solo |
| Customer requires LLM in the loop | High | Architecture conflict | Pitch the architecture as a *complement* to LLM (the auditable layer) not a replacement |
| Procurement timeline > runway | High | Cash burn | Plan for 24-month sales cycle in P&L |
| Geopolitical / export control issues | Medium | Limit to allied markets | Five-Eyes-only; accept that EU + Pacific allies need separate paths |

## What this product is *not*

To stay honest about scope:

- **Not a SIGINT / EW system.** Those have specialised tooling (e.g. Pythia, GENESIS) and decades of incumbents.
- **Not a kinetic targeting system.** That's Article 36 territory (UK) / DoDD 3000.09 (US) — much harder ethics/legal review and probably outside reasonable scope for a small specialist.
- **Not a sensor itself.** The product processes *outputs* of sensors others build.
- **Not a fully autonomous platform.** Human-in-the-loop / human-on-the-loop is the honest fit — the architecture's audit trail is the human's interpretation surface.
- **Not a counter-AI / GAN-defence product.** Adversarial ML defence is its own field with different tooling needs.

The narrow product: **compositional, explainable, edge-deployable object-track-anomaly detection that augments human decision-making with auditable reasoning, with maritime AIS as the first proof and migration to multi-sensor classified deployments as the longer-horizon plan.**

## Decision triggers — when to abandon defence

- If no credible partner conversation lands within 3 months of starting Phase 0, the defence sales motion isn't going to happen — abandon and concentrate on fraud/finance.
- If AIS demo takes > 9 months to be customer-presentable, the engineering distance is too large; reassess.
- If the partner conversations consistently say "we want a turnkey end-to-end system, not a primitive library", the architecture's distinctive value is being lost — reframe or abandon.
- If procurement-readiness review (with cleared lawyer) finds export-control / IL-rating obstacles that would take >12 months to clear, redirect to defence-adjacent verticals (border security, port security, MPA enforcement) where fewer such obstacles apply.

## Why bother at all

If fraud and finance are easier first verticals, why mention defence here?

Three reasons:
1. **The architecture's strongest fit theoretically.** Compositional, explainable, edge — these are not "nice to have" in defence; they are increasingly mandated.
2. **The largest TAM and the most defensible margin** of the three verticals, *if* the partner motion works.
3. **Strategic positioning.** A successful defence pilot creates massive optionality for adjacent regulated verticals (critical infrastructure, public safety, border security) that share the audit / edge / compliance pattern but have lower barriers.

The honest order of moves remains: prove fraud first (90 days, customer revenue), prove finance second (6 months, customer revenue), *then* defence — using the audit, edge, and pipeline IR maturity earned in the first two as evidence the partner needs.
