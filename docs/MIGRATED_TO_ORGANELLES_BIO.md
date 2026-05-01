# Productisation artefacts — moved to the private companion repo

The following documents previously lived in this repo's `docs/` and `docs/engineering/CLEAN_ROOM_IMPLEMENTATION/` directories. They were migrated to the private sibling repo **`organelles.bio`** on 2026-05-01 to maintain a clean separation between **research** (this repo) and **product** (the companion repo).

| Migrated document | Original path in `microgpt-c` |
|---|---|
| `STRATEGY_ONE_PAGER.md` | `docs/STRATEGY_ONE_PAGER.md` |
| `PRODUCTIZATION_VERTICALS.md` | `docs/PRODUCTIZATION_VERTICALS.md` |
| `PRODUCT_FRAUD_DETECTION.md` | `docs/PRODUCT_FRAUD_DETECTION.md` |
| `PRODUCT_FINANCE_RISK.md` | `docs/PRODUCT_FINANCE_RISK.md` |
| `PRODUCT_DEFENCE_TRACKING.md` | `docs/PRODUCT_DEFENCE_TRACKING.md` |
| `DEPENDENCY_POLICY.md` | `docs/DEPENDENCY_POLICY.md` |
| `AUDITLOG_SPEC.md` | `docs/engineering/CLEAN_ROOM_IMPLEMENTATION/AUDITLOG_SPEC.md` |
| `ROADMAP.md` (trigger-conditioned product roadmap) | `docs/engineering/CLEAN_ROOM_IMPLEMENTATION/ROADMAP.md` |
| `ADR_template.md` | `docs/engineering/CLEAN_ROOM_IMPLEMENTATION/ADR_template.md` |
| Older research-era ROADMAP | `ROADMAP.md` (root) |
| Older research-era VALUE_PROPOSITION | `VALUE_PROPOSITION.md` (root) |

These artefacts have access controls reflecting commercial sensitivity. Reach out to the project owner for access to the companion repo.

## What stays in this repo (research)

- All source code (`src/`, `demos/`, `tests/`, `tools/`)
- The full research arc (`docs/research/`) including audit logs and the `wiring_scaling_*` series
- The clean-room rebuild-test corpus (`docs/engineering/CLEAN_ROOM_IMPLEMENTATION/`) — BRD/FRD/NFRD/BS/TDD/FS/METHODOLOGY/TRACEABILITY/RESEARCH_DISCLOSURE/README
- The narrative book (`book/`) — chapter 21 deliberately retained as the research arc's own statement of "what we hand off to product"
- The calibrated honest-claim infrastructure (`tools/scaling_leakage_audit.sh`)

## Dependency direction (one-way)

The product repo depends on this research repo (consumes its tagged releases when code work begins). This research repo does NOT depend on the product repo. Cross-references in research artefacts that point at productisation documents are stub-redirected to this file.
