# ADR-NNN: <one-line decision title>

**Status:** Proposed | Accepted | Superseded by ADR-MMM | Deprecated
**Date:** YYYY-MM-DD
**Decider(s):** <single named human or 2-3 people>
**Tags:** <comma-separated, e.g. `dependency-policy`, `architecture`, `performance`>

---

## Context

What is the situation that requires a decision? Briefly describe the forces in play (technical, business, customer, regulatory). Cross-reference relevant `BS_*.md`, `BRD.md` BREQ-IDs, or `GAP-*` entries that motivated this.

State the **specific question** being decided in one sentence.

## Options considered

For each viable option, give one paragraph:

### Option A: <name>
- Brief description.
- Pros: ...
- Cons: ...
- Cost / risk / reversibility: ...

### Option B: <name>
- ...

### Option C (often "do nothing" / "defer"): <name>
- ...

## Decision

**Chosen: <Option X>.** One sentence saying what was decided.

## Consequences

- **Positive:** what this enables.
- **Negative:** what this constrains or makes harder.
- **Reversibility:** how to back out if the decision proves wrong, and what the trip-wires are.

## Compliance with the methodology

- Is this decision consistent with `BRD.md` § ... ?
- Does it close, open, or modify any `GAP-*` in `TRACEABILITY.md`?
- Does it require a `BS_*.md` invariant change? If so, which?

## Cross-references

- Code change: <commit hash or PR>
- Related ADRs: <ADR-MMM, ADR-PPP>
- Source-of-truth docs: <BS_*.md, FS_*.md>

## Revision history

| Version | Date | Change |
|---|---|---|
| 0.1 | YYYY-MM-DD | Initial draft |

---

## Template usage notes (delete when filling in)

- ADRs are immutable once Accepted. To change a decision, write a new ADR that supersedes the old one and link both ways.
- Keep ADRs short — under 500 words is ideal. If it's longer, the decision is probably actually two decisions.
- File at `docs.engineering/CLEAN_ROOM_IMPLEMENTATION/ADR_NNN_<short_slug>.md`. NNN is a zero-padded 3-digit number assigned at creation time per `METHODOLOGY.md` §4 ID discipline.
- Add the row to `TRACEABILITY.md` §6 ID assignment registry (new column "Highest ADR" if first ADR being added).
- ADRs are descriptive (TDD voice), not prescriptive (BS voice). They record *what was decided and why*; they do not bind future implementers — a future ADR can supersede.
