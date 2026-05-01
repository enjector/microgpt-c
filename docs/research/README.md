# `docs/research/` — index and reading order

This directory holds the research arc for MicroGPT-C: topical research notes (`RESEARCH_*.md`), the Wiring Organelle scaling-curve experiments (`wiring_*`), per-eval audit logs (`audit_logs/`), and the referenced papers (`papers/`). It is the *experimental* layer — descriptive, not normative. The normative source-of-truth for the platform is the rebuild-test corpus at `docs/engineering/CLEAN_ROOM_IMPLEMENTATION/`.

## Start here

**`ORGANELLE_STATE.md`** is the recommended top-of-stack entry point. Written 2026-05-01 after the scaling-curve arc closed, it is the calibrated synthesis: what an organelle is, what's been validated, where the three structural bounds bite, what's still open, and the reading-order pointer table for the rest of this directory.

After that, choose by audience:

| If you are | Read in this order |
|---|---|
| New contributor wanting the working-knowledge map | `ORGANELLE_STATE.md` → `RESEARCH_ORGANELLE_REASONING.md` → `RESEARCH_ORGANELLE_PIPELINE.md` |
| Reviewer auditing the most-tested research arc | `ORGANELLE_STATE.md` → `RESEARCH_PIPELINE_IR.md` (long, ~3800 lines, full Wiring arc) → `wiring_scaling_curve.md` (correction notice) → `wiring_scaling_post_phase3.md` (consolidated current state) |
| Engineer reproducing a calibrated number | `wiring_scoreboard_tier0.md` → `wiring_scoreboard_tier1.md` → `wiring_scaling_curve.md` → `wiring_scaling_curve_phase3.md` → `wiring_scaling_v3_deep_negative.md` → `wiring_scaling_post_phase3.md` |
| Curious about non-wiring applications | `RESEARCH_ORGANELLE_GAMES.md`, `RESEARCH_ORGANELLE_VISION.md`, `RESEARCH_ORGANELLE_PLANNER.md` |
| Looking for cancelled-phase / leakage-audit history | `wiring_scaling_leakage_audit.log` (this dir's `audit_logs/`) and the regulator-friendly distillation in `docs/engineering/CLEAN_ROOM_IMPLEMENTATION/RESEARCH_DISCLOSURE.md` |

## Subdirectory layout

| Path | Contents |
|---|---|
| `RESEARCH_*.md` | Topical research notes (manifold learning, attention mechanisms, intelligence framing, optimisation studies, V4-port stack, organelle applications, etc.) |
| `wiring_*.md`, `wiring_*.log` | The Wiring Organelle scaling arc (~v1 inflated → v2 clean → Phase 2 sharpening → Phase 3 broad expansion + bigram → post-Phase-3 cleanup → Phase 5/6/6b compositional search) |
| `audit_logs/` | Per-eval `*.log` files (leakage audits, scoreboards, classifier runs). Evidence for any cited number. |
| `papers/` | Referenced external papers (DeepSeek_V4, MSA, RotorQuant, TurboQuant, Attention-Residuals) |
| `images/` | Diagrams + infographics referenced from the research notes and the book |

## What lives outside this directory

- **Production-ready normative contracts** for every subsystem: `docs/engineering/CLEAN_ROOM_IMPLEMENTATION/`
- **Honest disclosure register** (cancelled phases, restated headlines): `docs/engineering/CLEAN_ROOM_IMPLEMENTATION/RESEARCH_DISCLOSURE.md`
- **Build-flag documentation**: `docs/BUILD_OPTIONS.md`
- **Extension recipes for the Wiring Organelle**: `docs/EXTENDING_WIRING_ORGANELLE.md`
- **Performance methodology + reference machine**: `docs/testing/PERFORMANCE.md`
- **Narrative book**: `book/MicroGPT-C_Composable_Intelligence_at_the_Edge.{md,pdf}`
- **Productisation strategy + vertical sketches**: migrated to the private companion repo (see `docs/MIGRATED_TO_ORGANELLES_BIO.md`)
