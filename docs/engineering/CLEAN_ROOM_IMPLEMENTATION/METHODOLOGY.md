# METHODOLOGY — Clean-Room Implementation Process

**Document ID:** MGC-CRI-METHOD
**Version:** 1.0
**Status:** DRAFT
**Last updated:** 2026-04-30
**Source:** Distilled from `book.7th/Reversible_Engineering.md` (Chapters 5–7, Appendices A–B), specialised for MicroGPT-C.

---

## RFC 2119

The key words MUST, MUST NOT, REQUIRED, SHALL, SHALL NOT, SHOULD, SHOULD NOT, RECOMMENDED, MAY, and OPTIONAL in this document are to be interpreted as described in RFC 2119.

---

## 1. Scope

This document defines the rules a clean-room reimplementation team SHALL follow to reproduce the MicroGPT-C platform from the corpus in `docs/engineering/CLEAN_ROOM_IMPLEMENTATION/` without consulting the original source tree. It also defines the rules the corpus MUST follow to remain a valid input to that reimplementation.

In scope: corpus structure, ID discipline, voice rules, gap discipline, the rebuild test.

Out of scope: the project's strategic positioning (see `BRD.md`), security review (deferred to Phase 3 — see `book.7th` Chapter 6), formal models (deferred).

## 2. Invariants of the methodology

These are the load-bearing invariants. A corpus that violates any of them is not a Reversible Engineering corpus regardless of how good the prose is.

| ID | Invariant | Rationale |
|---|---|---|
| INV-RE-1 | Code is the oracle for what the system does. Disagreements between corpus and code are gaps, not silent fixes. | Without this rule, the corpus drifts and the rebuild test stops being meaningful. |
| INV-RE-2 | Every requirement, invariant, error, SLO, and acceptance criterion has a stable ID. | IDs are the only mechanism for cross-document linkage. |
| INV-RE-3 | IDs are assigned once and never reused. Withdrawn IDs stay retired. | Reuse silently invalidates references in tests, threat models, and compliance tables. |
| INV-RE-4 | Every `BS_*.md` document covers exactly one subsystem. Mixing subsystems is forbidden. | Subsystem-scoped contracts are reviewable; system-wide contracts are not. |
| INV-RE-5 | Prescriptive (RFC 2119) and descriptive voices MUST live in separate documents. | Mixed voices dilute both meanings; readers cannot tell what is binding. |
| INV-RE-6 | Every gap has a disposition: `OPEN`, `TRIAGED`, `RESOLVED`, `ACCEPTED`, `DEFERRED`, `BLOCKED`. None is left untyped. | The gap list is the project's honest register; an untyped gap is invisible. |
| INV-RE-7 | A reviewer running the Rebuild Test (§7) MUST NOT have authored the subsystem under test and MUST NOT read its `TDD_*.md` or source. | Knowledge contamination would invalidate the test's signal. |
| INV-RE-8 | Cross-references to source files cite a path *and* a function name. Cross-references to tests cite a path *and* a test case name. | A "see `microgpt.c`" link is not actionable; "see `microgpt.c::forward_inference`" is. |

## 3. Document types and voice rules

The corpus uses six document types. Each has a strict voice rule.

### 3.1 BS_*.md — Behaviour Specs (prescriptive)

A `BS_<subsystem>.md` document SHALL describe what the subsystem promises to its callers. It SHALL NOT describe how the subsystem is implemented. It MUST use RFC 2119 voice. It SHOULD follow the template in Appendix A.1 of `book.7th`.

Required sections: Scope; Type contracts; Operation contracts; Invariants table (INV-IDs); Errors table (ERR-IDs); Concurrency model; Performance SLOs (SLO-IDs); Scenarios (SCN-IDs); Acceptance criteria (ACC-IDs); Cross-references; Revision history.

A `BS_*.md` is the artefact a clean-room reimplementer reads. They MUST be able to write a behaviourally equivalent implementation from the BS alone.

### 3.2 TDD_*.md — Technical Design Docs (descriptive)

A `TDD_<subsystem>.md` document SHALL describe how the subsystem is built: data structures, algorithms, design trade-offs. It SHALL use ordinary descriptive prose. It MUST NOT use RFC 2119 voice. It MUST cite the corresponding `BS_*.md`.

A `TDD_*.md` is the artefact a new contributor reads to onboard. It is *not* a contract: a TDD cannot be used as the input to a clean-room rebuild because it leaks implementation choices that the reimplementer is allowed to differ on.

### 3.3 FS_*.md — Functional / Format Specs (prescriptive)

An `FS_<format>.md` document SHALL specify a wire format, file format, ABI, or error code catalogue at byte level. It MUST use RFC 2119 voice. It MUST contain at least one normative example expressed as a byte sequence.

### 3.4 BRD / FRD / NFRD — Requirements Documents (descriptive + REQ-IDs)

These three documents capture the why (BRD), what (FRD), and how-well (NFRD) of the platform. They use descriptive voice but every distinct requirement carries a stable REQ-, NFR-, or SLO- ID.

### 3.5 TRACEABILITY.md — Master Index (structural)

The traceability matrix maps every ID to its document of definition, source files, tests, and dependent compliance/threat artefacts (when the latter exist). The matrix is load-bearing per INV-RE-2 / INV-RE-3.

### 3.6 Out-of-scope-for-V1.0 types

`TSM_*.md` (threat models), `COMPLIANCE_*.md` (per-framework mappings), `FORMAL_*.md` (TLA+/Alloy/Coq), `FMEA_*.md` (failure-mode analysis) are Phase 3 artefacts deferred for the MicroGPT-C V1.0 corpus. They are tracked as gaps in `TRACEABILITY.md`.

## 4. ID discipline

### 4.1 ID format

| Prefix | Used for | Scope |
|---|---|---|
| `REQ-<SUBSYSTEM>-NNN` | Functional requirement | One subsystem |
| `BREQ-NNN` | Business requirement | Whole platform |
| `NFR-NNN` | Non-functional requirement | Whole platform |
| `INV-<SUBSYSTEM>-NNN` | Invariant | One subsystem |
| `ERR-<SUBSYSTEM>-NNN` | Error code or condition | One subsystem |
| `SLO-<SUBSYSTEM>-NNN` | Performance SLO | One subsystem |
| `ACC-<SUBSYSTEM>-NNN` | Acceptance criterion | One subsystem |
| `SCN-<SUBSYSTEM>-NNN` | Scenario | One subsystem |
| `GAP-<CATEGORY>-NNN` | Gap | Cross-cutting |

`<SUBSYSTEM>` SHALL be one of: `CORE`, `TOK`, `CKPT`, `ORG`, `MSA`, `QUANT`, `PIPE`, `WIRE`, `GEO`, `VR`, `EKAN`, `VM`, `METAL`, `BUILD`. `NNN` is a zero-padded 3-digit number assigned at creation time.

### 4.2 ID rules (enforce in review)

1. An ID MUST be assigned at the moment a requirement is written down. Drafting a requirement without an ID is forbidden.
2. An ID MUST NOT be reused. If a requirement is removed, its ID is marked `WITHDRAWN` in the source document and remains in `TRACEABILITY.md`.
3. An ID MUST be cited in the test that verifies it. Tests without ID citations cannot demonstrate coverage.
4. An ID MUST be cited in any commit that materially implements or modifies the requirement.

## 5. Gap discipline

A gap is anything the corpus knows it does not yet promise or implement. Gaps are recorded in `TRACEABILITY.md` with these dispositions:

| Disposition | Meaning |
|---|---|
| `OPEN` | Newly identified; not yet examined |
| `TRIAGED` | Examined; severity assigned (P0/P1/P2/P3); plan documented; not yet acted on |
| `RESOLVED` | Closed in code or by spec change; cite the closing commit |
| `ACCEPTED` | Known and judged acceptable for a documented reason |
| `DEFERRED` | Postponed to a named future version (e.g., V2.0) |
| `BLOCKED` | Cannot be closed until a named dependency is satisfied |

The matrix MUST be current at all times. A gap with `OPEN` for more than the project's triage cadence (RECOMMENDED: 30 days) is a process failure.

## 6. The clean-room reimplementation procedure

A team rebuilding MicroGPT-C from this corpus SHALL proceed as follows.

### 6.1 Inputs the rebuild team gets

- All `BS_*.md` documents.
- All `FS_*.md` documents.
- `BRD.md`, `FRD.md`, `NFRD.md`.
- `TRACEABILITY.md` summary section (matrix may be consulted, gap list MUST NOT — gaps may bias the rebuild).
- `METHODOLOGY.md` (this file) and `book.7th/Reversible_Engineering.md`.

### 6.2 Inputs the rebuild team does NOT get

- The original source tree (`src/`, `demos/`).
- Any `TDD_*.md` document.
- Any test file (`tests/`).
- Any research note or design discussion (`docs/research/`).
- Any version-control history.

### 6.3 Sequencing

1. Read `BRD.md` to understand goals.
2. Read `FRD.md` and `NFRD.md` to enumerate the surface area.
3. For each subsystem in `FRD.md`, read its `BS_*.md` and the FS-formats it depends on.
4. Implement in the order: core → tokeniser → checkpoint → organelle → MSA → quant → pipeline IR → wiring → geodesic / VR / EKAN → VM → metal.
5. Write tests citing the IDs in the BS documents. Every `ACC-<SUBSYSTEM>-NNN` SHOULD have at least one test.
6. Compare behaviour to the published demo invariants in `BRD.md` (e.g., character-level Shakespeare zero-`<unk>`, lottery entropy floor).

### 6.4 Allowed degrees of freedom

The clean-room implementation MAY differ from the original in:

- Programming language (the original is C99; an idiomatic Rust, Go, or C++ implementation conforming to the BS contracts is a valid rebuild).
- Internal data structures (the BSes specify behaviour, not layout — except where explicitly stated).
- Optimisation strategy (BLAS / Metal / SIMD choices are NFR concerns, not BS contracts).
- Build system (CMake is the original choice; Bazel, Meson, plain Make are valid alternatives).

The clean-room implementation MUST NOT differ in:

- Public API surface (function signatures, types, error codes — these are in BSes / FSes).
- File formats (`FS_checkpoint.md`, `FS_pipeline_ir_text.md`, `FS_vm_bytecode.md`, `FS_organelle_wire.md`).
- RFC 2119 invariants in any BS marked `APPROVED`.

## 7. The Rebuild Test (per `book.7th` Chapter 7 + Appendix B)

The rebuild test is the operational definition of "the corpus is a faithful reflection of the system." It is run per-subsystem.

### 7.1 Roles

- **Reviewer**: a person (or AI session) who has not written any of the subsystem's source. They read the BS only.
- **Validator**: a person who has read both the BS and the source, who scores the reviewer's answers.

### 7.2 Steps

1. Reviewer reads the subsystem's `BS_*.md` and any cross-referenced FSes.
2. Reviewer answers the question set (Appendix B.3 of the methodology):
   - What types does this subsystem expose?
   - What operations does it support? (signatures, preconditions, postconditions)
   - What errors can it produce?
   - What invariants does it preserve?
   - What happens under concurrent access?
   - What performance guarantees does it offer?
3. Validator compares the reviewer's answers to the actual behaviour of the source.
4. Each discrepancy is categorised as: **specification hole** (BS missing content — fix BS), **specification ambiguity** (BS unclear — clarify BS), **legitimate degree of freedom** (BS deliberately leaves this open — note in BS), or **reviewer error** (reviewer misread).

### 7.3 Pass criteria

A subsystem's BS PASSES the rebuild test when:

- Every public type and operation is recoverable from the BS alone.
- Every error code is recoverable from the BS or an FS it cross-references.
- Every invariant tested in the source corresponds to an `INV-<SUBSYSTEM>-NNN` recoverable from the BS.
- Reviewer-vs-source discrepancies are limited to legitimate-degree-of-freedom and reviewer-error categories.

A subsystem's BS that has passed the rebuild test MAY be marked `APPROVED`. Otherwise it remains `DRAFT` or `REVIEW`.

## 8. Versioning rules

- A BS or FS document carries a semantic version. `1.0 → 1.1` for non-breaking additions or clarifications. `1.1 → 2.0` for breaking contract changes.
- A breaking BS change SHALL be accompanied by an entry in `TRACEABILITY.md` documenting the migration path.
- A breaking FS change SHALL include a versioning section showing how a producer/consumer of the previous version is supposed to behave (typically: the format carries an explicit version field; new readers reject unknown old versions or upgrade them; old readers reject unknown new versions).

## 9. AI-assisted authoring rules

This corpus is expected to be authored partly by AI. The rules in `book.7th` Chapter 15 apply. Specifically:

- An AI MAY draft a BS or TDD section; a human MUST verify each invariant against the cited source lines before the document is committed.
- An AI MUST NOT mark a gap as `RESOLVED` without citing a closing commit.
- An AI MUST flag any case where it cannot find a source citation for a claim it is asked to make — fabricated citations are a hard failure.
- An AI session reading this corpus is expected to follow the cross-references, not just the document at hand.

## 10. When this corpus is "done"

The corpus is *not* "done" in any final sense; per `book.7th`, a Reversible Engineering corpus is alive while the system is alive. This corpus's V1.0 milestone is reached when:

- Every subsystem in the table in `README.md` has both a `BS_*.md` and a `TDD_*.md` document at `APPROVED` status, with rebuild test passes recorded in `TRACEABILITY.md`.
- Every public function in `microgpt.h` and the other headers is referenced from at least one `BS_*.md` operation contract.
- Every demo in `demos/` is expressible as a sequence of operations defined in BSes (i.e., the demos are not doing anything secret that escapes the BS surface).
- `TRACEABILITY.md` reports counts of MET / PARTIAL / GAP per pillar.

V1.0 of *this* corpus (the document set committed in this directory) is `DRAFT` — it has been authored in a single archaeological extraction pass and has not yet been independently rebuild-tested.

---

## Cross-references

- `book.7th/Reversible_Engineering.md` Chapter 5 — Phase 2 archaeological extraction.
- `book.7th/Reversible_Engineering.md` Chapter 7 — The Rebuild Test.
- `book.7th/Reversible_Engineering.md` Appendix A — Templates.
- `book.7th/Reversible_Engineering.md` Appendix B — Rebuild Test Checklist.
- `README.md` (this directory) — Corpus index.
- `TRACEABILITY.md` — Master index.

## Revision history

| Version | Date | Change |
|---|---|---|
| 1.0 | 2026-04-30 | Initial extraction. Whole-corpus authoring pass. |
