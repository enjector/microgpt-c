# AUDITLOG_SPEC — Hash-chained audit log format (V0 stub)

**Document ID:** MGC-FS-AUDITLOG-001
**Version:** 0.1 — STUB (not load-bearing until Phase 1 fraud customer pilot per `docs/PRODUCT_FRAUD_DETECTION.md`)
**Status:** DRAFT — placeholder for the FS-level format spec that fraud Phase 1 will need to ship
**Type:** FS_* (prescriptive; RFC 2119 voice when filled in)
**Trigger:** Open this stub when the first vertical adds an audit-log persistence requirement; until then it sits as a known-gap marker.

---

## RFC 2119

The key words MUST, MUST NOT, REQUIRED, SHALL, SHALL NOT, SHOULD, SHOULD NOT, RECOMMENDED, MAY, and OPTIONAL in this document, when filled in, are to be interpreted as described in RFC 2119.

---

## 1. Scope (placeholder)

Define the byte-level format for the append-only, hash-chained audit log emitted by every Pipeline IR Judge decision in production verticals (fraud, finance, defence). Each entry MUST be append-only, tamper-evident via a hash chain, and parseable without referring to the producing binary.

## 2. Status

This is a stub. The format will be specified when the first vertical (fraud per `docs/PRODUCT_FRAUD_DETECTION.md` Phase 1) commits to a customer-deliverable audit format. Ahead of that, the conceptual fields are documented in `docs/PRODUCT_FRAUD_DETECTION.md` "Audit log format" section as a JSON sketch — that sketch is the input to this FS doc when work begins.

## 3. What this stub is not

- **Not** a binding contract today. No code emits this format; no regulator has reviewed it.
- **Not** the project's structured-logging surface. Operational logs go through Prometheus / OpenTelemetry per `DEPENDENCY_POLICY.md` Category A; the audit log is a separate write-once store with cryptographic integrity guarantees.

## 4. Open questions to resolve when filling in

1. Hash function (SHA-256 vs BLAKE3 vs SHA-3) — gated on the chosen crypto library (mbedTLS for fraud per `DEPENDENCY_POLICY.md`).
2. Entry format (JSON vs CBOR vs custom binary) — gated on regulator readability vs storage footprint trade-off.
3. Chain checkpointing strategy (per-N-entries Merkle tree vs flat chain) — gated on customer's read-pattern (occasional full-chain replay vs frequent point queries).
4. Key management (does the producer sign each entry, or only the chain root?) — gated on the customer's HSM / KMS setup.
5. Entry-count vs time-based rotation policy.
6. Recovery semantics on producer crash mid-write.

## 5. Cross-references

- `docs/PRODUCT_FRAUD_DETECTION.md` "Audit log format" — the conceptual sketch this FS doc will formalise.
- `DEPENDENCY_POLICY.md` Category A (mbedTLS / OpenSSL) — the crypto library candidates.
- `BS_pipeline_ir.md` — the producer side of every audit entry (each Pipeline IR Judge decision).
- New gap to register when work starts: `GAP-AUDIT-001`.

## 6. Revision history

| Version | Date | Change |
|---|---|---|
| 0.1 | 2026-05-01 | Stub. No format defined yet. |
