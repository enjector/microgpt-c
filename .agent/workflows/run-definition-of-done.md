---
description: Post-change "definition of done" checklist — ensures all downstream artifacts are updated after any code or experiment change.
---

# Definition of Done

Run this workflow after completing any significant change (core engine, experiment, documentation) to ensure all downstream artifacts are consistent and up to date before committing and pushing.

## 1. Identify What Changed

Determine the scope of the change by inspecting the working tree:

```bash
git status --short
git diff --stat HEAD
```

Classify the change into one or more categories:
- **Core engine** (`src/microgpt.c`, `src/microgpt.h`, `src/microgpt_*.c/.h`)
- **Experiment code** (`experiments/organelles/<game>/main.c`, `generate_corpus.py`)
- **Experiment results** (new corpus files, checkpoint files, win/solve rates)
- **Documentation only** (`docs/`, `README.md`, etc.)

---

## 2. Core Engine Changes → Run Tests

If any files in `src/` were modified:

// turbo
```bash
cd /Users/user/dev/projects/microgpt-c && mkdir -p build && cd build && cmake .. && cmake --build . 2>&1 | tail -20
```

// turbo
```bash
cd /Users/user/dev/projects/microgpt-c/build && ctest --output-on-failure 2>&1 | tail -30
```

- All tests must pass. Fix any failures before proceeding.
- If new public API was added, check that test coverage exists.

---

## 3. Experiment READMEs → Update with Latest Results

For each experiment that was modified or re-run:

1. **Check the experiment's README** at `experiments/organelles/<game>/README.md`
2. **Update the headline result** (e.g., win rate, solve rate, parse error rate)
3. **Update corpus stats** if the corpus changed (number of positions, file size)
4. **Update training stats** if retraining occurred (loss, epochs, training time)

Grep for stale numbers across experiment READMEs:
// turbo
```bash
cd /Users/user/dev/projects/microgpt-c && grep -rn "win\|solve\|% " experiments/organelles/*/README.md | grep -i "rate\|win\|solve"
```

---

## 4. Main Repository Artifacts → Propagate Results

These files reference game results and must stay in sync. Search each for stale numbers:

// turbo
```bash
cd /Users/user/dev/projects/microgpt-c && grep -n "Hex\|Red Donkey\|Pentago\|Othello\|Mastermind\|Klotski\|Sudoku\|Lights Out" README.md ROADMAP.md VALUE_PROPOSITION.md docs/organelles/ORGANELLE_GAMES.md docs/PERFORMANCE.md models/README.md 2>/dev/null
```

Files to check and update:

| File | What to update |
|------|---------------|
| `README.md` | Game leaderboard table (win/solve rates, param counts) |
| `ROADMAP.md` | Milestone entries with actual results |
| `VALUE_PROPOSITION.md` | Game list with percentages |
| `docs/organelles/ORGANELLE_GAMES.md` | Leaderboard table + commentary sections |
| `docs/PERFORMANCE.md` | Training/inference timing data (if changed) |
| `models/README.md` | Checkpoint performance table |
| `experiments/organelles/README.md` | Summary table of all experiments |

**Important:** Do NOT add references to any files under `docs/geometry/` — this directory is private (gitignored via `docs/.gitignore`). Verify no geometry links leak into public files:

// turbo
```bash
cd /Users/user/dev/projects/microgpt-c && grep -rn "geometry/" README.md ROADMAP.md VALUE_PROPOSITION.md docs/organelles/ experiments/organelles/*/README.md models/README.md 2>/dev/null
```

---

## 5. Models → Checkpoints and Training Logs

If an experiment was retrained:

1. **Verify checkpoint files exist** in `models/organelles/`:
// turbo
```bash
ls -la /Users/user/dev/projects/microgpt-c/models/organelles/*_planner.ckpt /Users/user/dev/projects/microgpt-c/models/organelles/*_player.ckpt 2>/dev/null | awk '{print $5, $9}'
```

2. **Verify training logs exist** alongside checkpoints (`.ckpt.log` files)
3. **Update `models/README.md`** if any checkpoint sizes or performance numbers changed

---

## 6. Book → Update If Results Changed

If any game results or major features changed:

1. **Search the book for stale numbers:**
// turbo
```bash
cd /Users/user/dev/projects/microgpt-c && grep -rn "% win\|% solve" docs/book/*.md | grep -v "MicroGPT-C_Composable" | head -20
```

2. **Update Chapter 7** (`docs/book/7.md`) — contains the game leaderboard table
3. **Update Appendix E** (`docs/book/A.md`) — add a changelog entry for the new version
4. **Rebuild the book:**
```bash
cd /Users/user/dev/projects/microgpt-c/docs/book && bash _build.sh
```
   This auto-increments the VERSION file and regenerates the combined markdown + PDF.

---

## 7. GitHub Discussions → Post Update Comment (If Significant)

If the change represents a significant experimental result:

1. **Check existing discussions** at https://github.com/enjector/microgpt-c/discussions
2. **Post a follow-up comment** on the relevant discussion thread
3. **Do NOT reveal methodology from `docs/geometry/`** — frame results in terms of encoding, validation, and corpus improvements

Relevant discussions:
- **#3 "Can Organelles Show Reasoning?"** — for game experiment results, reasoning findings
- **#2 "Can Organelles Show Intelligence?"** — for model learning verification results

---

## 8. Commit and Push

Stage all changed files, excluding anything under `docs/geometry/`:

```bash
cd /Users/user/dev/projects/microgpt-c && git status --short
```

Review the diff, then commit with a descriptive message:

```bash
git add <files>
git commit -m "<type>(<scope>): <description>"
git push
```

Commit message conventions:
- `feat(hex)`: new feature or experiment result
- `fix(engine)`: bug fix in core engine
- `docs(book)`: documentation-only changes
- `refactor(organelle)`: code restructuring without behavior change
- `test(vm)`: test additions or fixes

---

## Quick Reference: Common Stale-Number Grep

Run this one-liner to find ALL percentage-based results across public files:

// turbo
```bash
cd /Users/user/dev/projects/microgpt-c && grep -rn "[0-9]*% \(win\|solve\|exact\)" README.md ROADMAP.md VALUE_PROPOSITION.md docs/organelles/ORGANELLE_GAMES.md models/README.md experiments/organelles/*/README.md docs/book/7.md 2>/dev/null | sort
```
