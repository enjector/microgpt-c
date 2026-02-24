# MicroGPT-C Kanban Board

> Last updated: Feb 24, 2026

## Backlog

| ID | Title | Priority |
|----|-------|----------|
| | | |

## In Progress

| ID | Title | Started |
|----|-------|---------|
| | | |

## Done

| ID | Title | Completed |
|----|-------|-----------|
| K-1 | Hex topology uplift (4%→27%) | Feb 23 |
| K-2 | Red Donkey corpus expansion (12%→19%) | Feb 23 |
| K-3 | Update all public docs with new results | Feb 23 |
| K-4 | Post discussion #3 comment on Hex uplift | Feb 23 |
| K-5 | Add book Appendix E changelog from git history | Feb 23 |
| K-6 | Create `/run-definition-of-done` workflow | Feb 23 |
| K-7 | Create `/use-kanban` workflow | Feb 23 |
| K-8 | Hex 5×5 variant (32% win rate) | Feb 23 |
| K-9 | Hex virtual connections — negative result (prompt overflow) | Feb 24 |
| K-10 | Red Donkey 5×4 (0% solve — board too large) | Feb 24 |
| K-11 | Deeper MCTS for 5×5 Hex (no additional benefit) | Feb 23 |
| K-12 | Transfer learning TTT→Othello (no transfer benefit) | Feb 24 |
| K-13 | Transfer+FT: TTT→Othello +4% over scratch | Feb 24 |

---

## Task Details

### K-1: Hex topology uplift (4%→27%)

- **Point:** Hex win rate jumped 6.75× with zero engine changes — just encoding, filtering, and better training data.
- **Picture:** Like giving a blind chess player a description of which pieces are connected instead of just listing coordinates.
- **Proof:** 27% win rate vs random, parse errors dropped 50%→17%, corpus grew 4,981→13,510.
- **Push:** Try virtual connection templates or smaller board (5×5) for further gains.

**Outcome:** Shipped in `3070522`. BFS connectivity features + topological Judge + MCTS corpus generation. Three interventions stacked: Judge alone = 25%, encoding = 19%, MCTS corpus = 27%.

---

### K-2: Red Donkey corpus expansion (12%→19%)

- **Point:** More training data unlocked a 58% relative improvement with no code changes to the engine.
- **Picture:** Like studying from a textbook that went from 10 pages to 25 — same student, more material.
- **Proof:** 19% solve rate from 523 BFS-solved positions (up from 199).
- **Push:** Limited by 4×3 puzzle state space (~500 unique positions). Try classic 5×4 layout with bounded A*.

**Outcome:** Shipped in `3070522`. Increased scrambles (1000→5000), widened depth (2-12→2-20), raised BFS limit (50K→200K), added multiprocessing.

---

### K-3: Update all public docs with new results

- **Point:** Every public-facing file now reflects the latest Hex 27% and Red Donkey 19% numbers.
- **Picture:** Like updating all the scorecards in a stadium after the final score changes.
- **Proof:** 8 files updated: README, ROADMAP, VALUE_PROPOSITION, ORGANELLE_GAMES, models/README, experiment READMEs. Zero geometry references leaked.
- **Push:** Run `/run-definition-of-done` next time to catch these automatically.

**Outcome:** Shipped in `3070522` and `ea87a68`.

---

### K-4: Post discussion #3 comment on Hex uplift

- **Point:** Community updated on Hex results without revealing the geometric methodology.
- **Picture:** Like announcing a sports result without revealing your training regimen.
- **Proof:** Comment posted at `#discussioncomment-15901718` with full results table, framed as encoding + validation + MCTS.
- **Push:** Monitor for community responses.

**Outcome:** Posted to "Can Organelles Show Reasoning?" discussion #3.

---

### K-5: Add book Appendix E changelog from git history

- **Point:** The book now has a traceable version history mined from real git commits.
- **Picture:** Like adding a "revision history" page to a textbook so readers know what changed between editions.
- **Proof:** 17 commits across 6 version bumps, each with commit hash and date. Book rebuilt as v1.1.1.
- **Push:** Just add a row and run `_build.sh` for future changes.

**Outcome:** Shipped in `15733e3`.

---

### K-6: Create `/run-definition-of-done` workflow

- **Point:** Automated post-change checklist so no downstream artifact is ever forgotten.
- **Picture:** Like a pilot's pre-flight checklist — same steps every time, nothing missed.
- **Proof:** 8-step workflow: tests, experiment READMEs, repo docs, models, book, discussions, commit/push.
- **Push:** Invoke with `/run-definition-of-done` after any significant change.

**Outcome:** Created at `.agent/workflows/run-definition-of-done.md`.

---

### K-7: Create `/use-kanban` workflow

- **Point:** Persistent task tracking across sessions with context preserved in Spear format.
- **Picture:** Like a sticky-note board that follows you between meetings.
- **Proof:** Workflow + board created, seeded with 7 completed tasks from today's session.
- **Push:** Start each session by reading `KANBAN.md` to pick up where we left off.

**Outcome:** Created at `.agent/workflows/use-kanban.md` and `KANBAN.md`.

---

### K-8: Hex 5×5 variant (32% win rate)

- **Point:** Shrinking the board from 7×7 to 5×5 gave an 18.5% relative improvement with zero code changes beyond a grid constant.
- **Picture:** Like switching from a large exam paper to a pocket quiz — same student, same skills, fewer questions to misparse.
- **Proof:** 32% win rate on 5×5 vs 27% on 7×7. Prompt length halved (25 vs 49 cells), state space reduced by ~1000×.
- **Push:** Virtual connection templates (K-9) are the next lever — precomputed bridge patterns would give the model real strategic knowledge beyond local adjacency.

**Outcome:** Built and run in single session. `main.c` parameterized via `HEX_GRID` define. `hex5_demo` CMake target added. 5,523 position corpus generated with MCTS + topo features.

---

### K-11: Deeper MCTS for 5×5 Hex (no additional benefit)

- **Point:** On a 5×5 board, 200 MCTS iterations already converges — doubling to 500 produces identical training data.
- **Picture:** Like studying the same textbook twice — if you memorised it the first time, a second pass adds nothing.
- **Proof:** Identical MD5 hash between 200-iter and 500-iter corpora. Both produce 32% win rate.
- **Push:** Deeper MCTS may still help on 7×7 where the tree is much wider. On 5×5, the bottleneck is elsewhere (model capacity or Judge quality, not corpus quality).

**Outcome:** Tested and confirmed. No separate commit needed — the finding is documented here and in ORGANELLE_GAMES.md.

---

### K-12: Transfer learning TTT→Othello (no transfer benefit)

- **Point:** Internal transformer weights (attention + MLP) trained on TicTacToe don't transfer to Othello without fine-tuning.
- **Picture:** Like transplanting a chess player's intuition into a Go player's body — the muscle memory is for the wrong game.
- **Proof:** SCRATCH=43% win, TRANSFER(TTT→Oth)=32% win, RANDOM(untrained)=33% win. Transfer ≈ Random.
- **Push:** Fine-tuning the transferred model on Othello corpus could show benefit. Also try same-game transfer (e.g., TTT planner→TTT player).

**Outcome:** Added `model_transfer_weights()` to `microgpt.h`/`microgpt.c`. Created `transfer_demo` experiment with CMake target. Three conditions tested over 100 Othello games each. Transfer without fine-tuning provides no advantage over random initialization, confirming that vocab-dependent layers (wte/lm_head) dominate for character-level game models.

---

### K-13: Transfer+FT: TTT→Othello +4% over scratch

- **Point:** Pre-training on TicTacToe then fine-tuning on Othello yields **+4% win rate** over training Othello from scratch.
- **Picture:** Like a musician who learned piano first picks up guitar slightly faster — the finger dexterity transfers even if the instrument is different.
- **Proof:** SCRATCH=30%, TRANSFER+FT=34%, TRANSFER(no FT)=30%, RANDOM=31%. Transfer+FT is the best condition.
- **Push:** Try same-game transfer (K-14: planner→player) where vocab overlap should amplify the effect.

**Outcome:** Added `organelle_train_transfer()` to library (refactored `organelle_train()` into internal helper). Updated `transfer_demo` to 4 conditions. Positive result: internal transformer representations do transfer value when combined with fine-tuning, even across different games.
