# Dependency boundary policy — moving from research-pure to product-pragmatic

**Status:** working draft, follows from `docs/PRODUCTIZATION_VERTICALS.md`. The single largest gating decision before any vertical ships. Not a research aesthetic — a deliberate engineering boundary.

## What the current policy is

From `CLAUDE.md`:
> **C99 only** in core engine. C11/C23 features are not allowed in `microgpt.{h,c}`.
> **Zero deps** in core. Platform accelerators (Metal, BLAS, etc.) live behind `#ifdef` guards and are gated by CMake options.

This was correct for research:
- Kept the codebase pedagogically clean
- Made every claim reproducible without dependency provenance issues
- Forced the architecture to be small enough to live on a single laptop
- Kept the `bootstrap.sh` → working-binary path under 60 seconds
- Made every CI failure attributable to *our code*, not someone else's transitive dep

It's wrong for product. Within the first month of any vertical work the project trips over:
- No streaming ingestion (Kafka, FIX, NMEA, Kinesis)
- No external embedding for retrieval beyond the §44.5 ceiling
- No SBOM generation for compliance
- No standard observability (metrics, traces, structured logs)
- No HSM / KMS integration for tamper-evident audit logs
- No standard cryptography for the audit log's hash chain
- No JSON / Protobuf serialisation for cross-system interop

Every one of those is needed for fraud. More are needed for finance and defence.

## What replaces it: thin deliberate boundary

Three categories: **allowed**, **conditionally allowed**, **forbidden**. Each with explicit governance rules.

### Category A: allowed (commit to these, treat as platform)

A library qualifies for Category A only if **all** of:
1. Permissive license compatible with the project's commercialisation path (MIT, BSD, Apache 2.0, ISC; not GPL/AGPL unless intentional)
2. Available on every target platform (Linux, macOS, Windows, embedded Linux, common edge SoCs)
3. C99-callable (i.e. has C bindings or is C itself)
4. Has been actively maintained for ≥ 5 years with ≥ 2 active maintainers
5. Used by ≥ 100 production systems we can name
6. Stable ABI or source-code linkable so we control the version
7. Documented and reviewed by the project's tech lead before adoption

Initial Category A list (proposed, not yet committed):

| Library | Purpose | License | Why this and not alternatives |
|---|---|---|---|
| `librdkafka` (or `cppkafka` C bindings) | Streaming ingestion (fraud, finance) | BSD-2 | De-facto Kafka client; stable; widely deployed |
| `OpenSSL` (or `BoringSSL`, or `mbedTLS` for embedded) | Cryptographic primitives for audit logs | Apache-2 / OpenSSL / Apache-2 | Standard; FIPS-validatable variants exist |
| `SQLite` | Local persistent state (cardholder profiles, session state) | Public domain | Single-file embedded; battle-tested |
| `cJSON` (or `jansson`) | JSON serialisation for adapters and APIs | MIT | Tiny, no transitive deps |
| `protobuf-c` | Protobuf for systems integration | BSD-3 | When customers need protobuf wire format |
| `libuv` | Async I/O event loop (if/when needed) | MIT | Single-process async; widely deployed |
| `prometheus-client-c` | Metrics export | Apache-2 | Standard observability surface |
| `OpenTelemetry C SDK` | Distributed traces (when relevant) | Apache-2 | Standard cross-process observability |

**Rule:** every Category A library must be vendored or pinned by exact version + cryptographic hash. Never `apt install` from a moving target.

### Category B: conditionally allowed (project-specific, scoped to one product line)

A library qualifies for Category B if it doesn't meet all Category A criteria but is necessary for one specific vertical:

| Library | Purpose | Restricted to | Notes |
|---|---|---|---|
| `ONNX Runtime` (C API) | External pretrained sentence embeddings (per Post-Phase-3 #3) | Vertical that needs ≥ 80% retrieval | Heavy; use only when measurement justifies |
| `fastText` (C++ but C-callable) | Lightweight word-vector embeddings | Same | ~600MB models; lighter than transformer alternatives |
| `Stone Soup` Python (or its C bindings if extracted) | Multi-object tracking | Defence vertical only | Cross-language; ops cost |
| `Eigen` (C++ header-only, C bindings via wrapper) | Dense linear algebra beyond what BLAS gives | Finance regime classifier | Header-only; no link footprint |
| Specific FIX engines (QuickFIX/C++ etc.) | FIX protocol for finance | Finance vertical only | Vendor lock-in if not careful |
| Sensor SDKs (specific per-vendor) | Defence sensor adapters | Defence vertical, per-deployment | Almost always proprietary; license per-deal |

**Rule:** Category B libraries are scoped to one product line. They never enter the core `microgpt.{h,c}` engine. Each one is documented in its product's `DEPENDENCIES.md` with: license, why this and not alternatives, fallback if it disappears, escape plan.

### Category C: forbidden (drawn explicitly to prevent drift)

Libraries that *would* make the architecture's distinctive value disappear:

| Type | Why forbidden | Exception path |
|---|---|---|
| Cloud-only ML APIs (OpenAI, Anthropic, Cohere, Hugging Face Inference) | Defeats edge / on-prem positioning; unverifiable; latency unbounded | Customer-explicit: a customer pays for cloud-only mode and accepts the trade |
| GPL / AGPL libraries in commercially-distributed code | License contagion | Internal tooling only; never in shipped binaries |
| LLM frameworks that pull in CUDA / ROCm by default in core | Eliminates SoC / embedded targets | Optional CMake flag, off by default |
| Anything > ~10MB binary footprint in core | Breaks edge deployment | Vertical-only opt-in |
| Anything that requires telemetry to home (JetBrains, etc.) | Defence customers will reject; finance customers will reject | Strict ban |
| JavaScript / Python runtimes in the core decision path | Defeats the C99 latency story | Adapter-side allowed; decision path C only |

## Governance — how to actually run this

### Adding a new Category A library

A pull request that adds a Category A dependency MUST include:
1. A `docs/dependencies/<name>.md` justification: license, version pinned, source provenance, alternatives considered, why this won, escape plan
2. A vendoring strategy: either `git subtree`, `git submodule` pinned to a tag, or a CMake `FetchContent` block pinned to a commit SHA
3. A hash check in CI: download is verified against a committed SHA256
4. SBOM update (CycloneDX or SPDX)
5. License-compatibility check via `licensee` or `scancode-toolkit` in CI
6. Tech-lead sign-off (single named human, recorded in PR)

### Updating a Category A version

- Patch version (X.Y.Z → X.Y.Z+1): tech-lead approval; CI re-runs
- Minor version (X.Y → X.Y+1): justification PR, 7-day review window
- Major version (X → X+1): treated as a new dependency adoption (full Category A check)

### Adding a Category B library

Same as Category A but scoped to one product directory (e.g. `products/finance/`). Cannot enter `src/`. Tech-lead + product-lead both sign off.

### Removing a dependency

Easier than adding, but same documentation: why, what replaces it, migration path for existing customers. No silent removals.

## Reproducibility — what stays true after this policy

The research codebase had perfect reproducibility because there were no deps. Product code has graded reproducibility:

1. **Bit-exact** for the pure-C99 core (no change from today)
2. **Hash-pinned** for Category A deps (exact commit SHA + verified SHA256)
3. **Behavioural** for Category B deps (semver guarantees + integration test suite)
4. **Customer-managed** for Category B vendor SDKs (we document what we tested against; customer-side updates are their problem)

The promise to customers is: *"give us the tagged version of this product and the same input, get bit-exact output, modulo any vendor SDK on your side."*

## License compatibility matrix

For each candidate license type, what shipping conditions apply:

| License | Internal use | Embedded in commercial product | Distributed source | Notes |
|---|---|---|---|---|
| MIT / BSD / ISC / Apache-2 | Yes | Yes | Yes | Default green |
| MPL-2.0 | Yes | Yes if separate file | Yes | Per-file copyleft only |
| LGPL-2.1+ | Yes | Yes if dynamically linked | Yes | Static link triggers reciprocity |
| GPL / AGPL | Yes (internal tooling only) | No | No (or if entire product GPL'd) | License contagion |
| OpenSSL | Yes | Yes (1.1.1+ is dual-licensed Apache-2/OpenSSL) | Yes | Older versions need attribution |
| Custom / commercial | Per-license review | Per-license review | Usually no | Always involve legal |

## Specific decisions to make in the first product cycle

These should be settled before any customer demo:

1. **Crypto library**: OpenSSL (mature, FIPS-able, big footprint) vs. mbedTLS (smaller, cleaner, better embedded fit). Recommend mbedTLS for fraud/defence (edge) and OpenSSL for finance (server-side).
2. **Database**: SQLite for state; question is whether to add a streaming-friendly KV (RocksDB? LMDB?) for higher throughput. Recommend deferring to Phase 2 — SQLite suffices through first POC.
3. **Streaming ingestion**: librdkafka is the obvious answer. Confirm by checking the first customer's stack — if they're on Kinesis, switch.
4. **Embedding library** (if/when crossing the §44.5 ceiling): ONNX Runtime + a quantised sentence-transformer. Document the model file separately from code.
5. **Observability**: Prometheus metrics + structured JSON logs as a baseline. OpenTelemetry traces only if customer asks.
6. **Build system extension**: keep CMake. Add Conan or vcpkg only if Category A list grows past ~10. Today: just CMake `FetchContent`.

## What this policy preserves

The thing the research project did right that a product can keep:
- **Single-binary runtime**: no Python interpreter, no JVM, no Node runtime in the decision path
- **Reproducible builds**: pinned versions + hash checks
- **Edge-deployable footprint**: < 50 MB total binary including all Category A
- **Verifiable claims**: every benchmark cites the exact build artefact + git SHA
- **Audit-friendly**: every library has a documented justification a regulator can read

What it loses:
- The "zero dep" badge on the README
- The pedagogical purity of "everything is in this single repo"
- The micro-second-startup-time guarantee (some Category A libs have init costs)

These are acceptable losses for a product. They were correctly preserved for research.

## Decision: when to commit to the policy

Don't adopt the policy abstractly. Adopt it **when the first vertical's first deliverable forces the question**.

Concretely: when the fraud vertical's Phase 1 starts (per `docs/PRODUCT_FRAUD_DETECTION.md`), one of the first PRs will need to add `librdkafka` for transaction streaming. *That* PR is when this policy becomes binding. Before that, the project remains pure C99 + the existing optional accelerators (Metal, BLAS).

The policy's goal isn't to add dependencies — it's to make the addition deliberate, traceable, and reversible.
