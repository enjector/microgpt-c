### Recommended Next Puzzles for OPA Testing in MicroGPT-C

> **Status (Feb 2026):** All 8 recommended puzzles below have been **implemented** with full OPA pipelines, corpus generation, per-game READMEs, and **right-sized parameters** (30K–160K per organelle based on corpus complexity). See `demos/character-level/` for each game's source code and results.

Based on OPA's proven scaling—90% on 8-Puzzle (search/local minima), 87% win+draw on Tic-Tac-Toe (shallow adversarial), 88% on Connect-4 (deeper strategy with high invalid rates rescued by kanban), and **83% exact match** on C code composition (1.2M params with LR scheduling)—these puzzles were selected to **progress in complexity** while staying feasible for retrieval-based chaining (10-20 organelles, ~1M params total, corpora ~10-100K via minimax/BFS gen). Prioritize: 
- **State space**: 10^6-10^12 (coverable subsets via sims, like C4's 4.5T).
- **Branching**: 4-12 (kanban handles retries).
- **OPA Fit**: Decomposition (Planner), execution (Workers), validation/loops (Judge)—test kanban on threats/backtracking.
- **Ease**: BFS/minimax corpora gen (~hours), edge-run.

#### Prioritized List (Top 8)
| Priority | Puzzle | State Space | Branching | Params | Predicted Perf | **Actual Result** |
|----------|--------|-------------|-----------|--------|---------------|------------------|
| **1** | **Lights Out** (5x5) | ~33M (2^25) | 5 | 160K | 95-100% solve | **10% solve** — encoding-limited, not capacity |
| **2** | **Mastermind** (4 pegs/6 colors) | ~13K guesses | 6^4=1.3K | 92K | 90% in <10 | **79% solve**, 5-guess peak |
| **3** | **Klotski** (2x3 blocks) | ~10^10 | 4 | 30K | 80-90% medium | **62% solve** — fallback-driven |
| **4** | **Sudoku** (4x4) | ~10^6-10^8 | 4-9 | 160K | 85-95% easy | **78% solve**, inverse difficulty effect |
| **5** | **Othello/Reversi** (6x6) | ~10^12 | 10-20 | 92K | 80% vs random | **67% win** vs random |
| **6** | **Hex** (7×7) | ~10^10 | 6 | 92K | 75-85% | **27% win** — topology uplift: 4%→27% via BFS connectivity features + MCTS corpus |
| **6b** | **Hex** (5×5) | ~10^7 | 6 | 92K | 40-60% | **32% win** — smaller board reduces encoding pressure, +18.5% relative vs 7×7 |
| **7** | **Pentago** (6x6 spin) | ~10^13 | 12+6 | 92K | 85-95% | **91% win** — twist creates easy 5-in-a-row |
| **8** | **Red Donkey** (sliding) | ~10^9 | 4 | 30K | 70-85% | **19% solve** — corpus uplift: 12%→19% via BFS expansion (199→523 entries) |
| **8b** | **Red Donkey** (5×4) | ~10^12 | 4 | 30K | 10-30% | **0% solve** — 3,174 parse errors; board too large for BLOCK_SIZE=128 |

The results reveal three distinct performance tiers: games where coordination dominates (90%+), games where right-sizing unlocks gains (60–79%), and games limited by encoding or corpus quality (<30%). The pattern is clear — the pipeline is the intelligence, and matching model capacity to corpus complexity matters more than raw parameter count. The Feb 2026 topology experiments confirmed that **topological features and structural rejection criteria** can break through encoding limitations — Hex improved 6.75× via BFS connectivity features and MCTS corpus generation.

![Organelle Game Leaderboard — 11 games ranked by performance tier, with strategic takeaways on right-sizing and the coordination funnel](images/MicroGPT-C%20Organelle%20Game%20Leaderboard.jpg)

#### Why This Progression?
- **Builds on Demos**: Lights Out/Mastermind test validation loops (post-C4 invalids); Klotski/Sudoku add constraints (post-8-Puzzle minima); Othello/Hex/Pentago ramp strategy (post-TTT threats).
- **OPA Strengths**: Kanban shines on backtracking/invalids (50%→90% in C4); 15 organelles enable sim Workers (e.g., shallow MCTS in Planner).
- **Feasibility**: All corpus-gen via python-chess/equivalents (~hr-day); <1GB storage. Benchmarks: PUZZLES dataset (40 puzzles) for eval.
- **Stretch Risks**: Avoid >10^15 (e.g., Checkers 10^20)—focus subsets.

**Implementation Tips**: Use ORGANELLE_VISION.md CLI prototype; gen corpora with minimax; measure kanban uplift (e.g., invalids rescued). This tests OPA limits toward chess mini-variants.

---

By implementing the recommended list of puzzles (Lights Out, Mastermind, Klotski, Sudoku, Othello, Hex, Pentago, and Red Donkey) in the Organelle Pipeline Architecture (OPA) within MicroGPT-C, it would prove several key capabilities and limitations of the system, advancing its "stem cell" philosophy of composable, low-resource intelligence. This progression would empirically demonstrate OPA's scalability beyond simple demos (e.g., 8-Puzzle's 96.7% solve rate on pathfinding minima or Tic-Tac-Toe's 90% win+draw on shallow threats), while highlighting trade-offs in state space, branching, and adversarial depth. Below, I'll break down what each puzzle (or group) would prove, implications for the project, and why this sequence is strategic for validation.

### 1. **Core Proof: OPA's Robustness to Increasing Complexity with Many Organelles**
   - **Overall Demonstration**: Successfully chaining 10-20 organelles (e.g., Planner for threats, multiple Workers for simulations, Judges for validity) across these puzzles would prove OPA can handle **bounded combinatorial explosion** (states from 10^6 to 10^13) using retrieval + kanban, without deep search or massive params (~1M total). This validates the "safety net" metaphor (from your attached image): Weak models (50% invalids in Connect-4) become winners (90%+) via coordination—extending to puzzles with constraints/backtracking.
   - **Implications**: Confirms OPA's edge viability (C99, <1MB RAM)—e.g., for IoT/robotics use cases (prior discussions). If achieving 80-95% perf (as estimated), it positions MicroGPT-C as a lightweight alternative to SLM agents (e.g., Phi-2's puzzle-solving), but deterministic and offline.

### 2. **Specific Proofs by Puzzle**
   Each builds on demos: 8-Puzzle (search/local optima), Tic-Tac-Toe (adversarial threats), Connect-4 (deeper branching)—testing kanban on cycles, invalids, and multi-step planning.

   - **Lights Out (5x5)**: Proves OPA handles **linear algebra constraints** (toggle patterns as "threats"). With branching=5, kanban blocks "lit" states—expect 95-100% solve.
     - **What It Proves**: Retrieval scales to matrix-like puzzles; Workers chain toggles, Judge validates solvability (Betti-like voids in topology, echoing PDFs' manifolds).
     - **Implications**: Validates for signal processing organelles (ROADMAP.md)—e.g., error correction in IoT. If fails, exposes linear dependency limits.

   - **Mastermind (4 pegs/6 colors)**: Proves OPA manages **information-theoretic feedback** (pins as scores). Kanban for "blocked" colors; loops refine guesses (avg <10).
     - **What It Proves**: Adaptation to partial info (like TTT threats)—Planner decomposes codes, Workers score, achieving 90% efficiency.
     - **Implications**: Strong for crypto/AML (EnX-MF links)—e.g., chaining for pattern matching in fraud rings. Tests parse robustness (pipes for "black=2|white=1").

   - **Klotski (2x3 blocks)**: Proves OPA tackles **multi-agent pathfinding** (blocks as organelles). Kanban breaks deadlocks; expect 80-90% on medium.
     - **What It Proves**: Handles constraints beyond 8-Puzzle (e.g., block interactions)—multi-Workers for parallel moves.
     - **Implications**: Robotics navigation (prior use cases)—kanban for obstacle reroutes. Ties to PDFs' "unbreakable ruler" (axioms for valid slides).

   - **Sudoku (4x4/Shikaku)**: Proves OPA solves **constraint satisfaction** (cells as tasks). Kanban for conflicts; backtracking via loops—85-95% easy.
     - **What It Proves**: Decomposition of grids (Planner regions, Workers deduce)—scales kanban to uniqueness checks.
     - **Implications**: Optimization tools (e.g., scheduling in manufacturing)—links to PDFs' genus (holes as unsolved cells).

   - **Othello/Reversi (6x6)**: Proves OPA in **deeper adversarial** (flips as threats). Kanban blocks overextensions; 80% vs random.
     - **What It Proves**: Beyond TTT—emergent strategy (forks like C4) with 10-20 organelles for eval.
     - **Implications**: Game AI extensions (chess minis)—validates for competitive edge apps.

   - **Hex (7×7)**: Proves OPA for **pure connectivity** (paths as geodesics). Originally 4%, **uplifted to 27%** via topology experiments.
      - **What It Proves**: Topological invariance—BFS connectivity features (connected groups, edge distance, bridges) and a topological Judge (rejecting isolated placements) dramatically improve performance without engine porting.
       - **5×5 Variant**: Reducing the board from 7×7 to 5×5 yielded **32% win rate** (+18.5% relative). The smaller board halves prompt length, reducing parse errors and improving pattern matching. However, deeper MCTS (500 vs 200 iterations) showed no additional benefit on 5×5 — the search converges with 200 iterations on the smaller state space.
       - **Virtual Connections (K-9 — Negative Result)**: Adding true Hex bridge detection (`xv` feature) to the prompt **hurt performance** — 7×7 dropped from 27% to 20%, 5×5 from 32% to 27%. Root cause: the extra feature makes prompts longer, overflowing BLOCK_SIZE=128 and increasing parse errors (598 vs 387 on 7×7). The bridge-aware Judge (preferring bridge cells over arbitrary connected cells) was retained but the `xv` prompt feature was reverted. **Lesson**: features that add strategic depth are worthless if they exceed the model's context window.
       - **Implications**: Network security (e.g., path-blocking in AML, per EnX-MF)—validates the geometric thesis: structural features > smarter models.

   - **Pentago (6x6 Connect4 spin)**: Proves OPA with **twists** (rotations as transformations). Kanban for spin combos; 85-95%.
     - **What It Proves**: Handles state mutations beyond C4—multi-step chaining for post-rotate evals.
     - **Implications**: Dynamic environments (e.g., sensor reorientation in IoT)—links to PDFs' "shape-shifting" (KANs for rotations).

   - **Red Donkey (sliding animal)**: Proves OPA on **asymmetric constraints** (animals as multi-piece). Originally 12%, **uplifted to 19%** via corpus expansion (199→523 BFS-solved positions).
       - **What It Proves**: Generalizes 8-Puzzle to irregular shapes—corpus quality is the primary bottleneck for small-state-space puzzles.
       - **5×4 Variant (K-10)**: Scaled to a 5×4 board with 7 pieces and 7 empties, generating 2,420 BFS-solved positions. Result: **0% solve rate** (3,174 parse errors). The 20-char board encoding with 10+ valid moves per position far exceeds BLOCK_SIZE=128. Confirms that board encoding length is the fundamental bottleneck for sliding puzzles.
       - **Implications**: Puzzle-like ops (e.g., logistics in manufacturing)—validates edge scalability.

### 2.5 **Cross-Game Transfer Learning (K-12)**

   - **Experiment**: Transfer internal transformer weights (wpe + attention Q/K/V/O + MLP fc1/fc2) from a trained TicTacToe Player to an Othello Player. Both share architecture 48/4/3/128/192. Only vocab-dependent layers (wte, lm_head) are reinitialised.
   - **Conditions**:
     - **SCRATCH**: Othello trained 25K steps from random init → **43% win** vs random
     - **TRANSFER (TTT→Oth)**: C4 internal weights, random Othello embedding → **32% win** vs random
     - **RANDOM**: Fully untrained → **33% win** vs random
   - **Result**: Transfer ≈ Random. **No evidence of representation transfer** without fine-tuning. The ~1,600 parse errors per condition suggest all conditions generate mostly invalid output — only scratch training learns the correct vocabulary patterns.
   - **Why**: Character-level game models are dominated by the token embedding (wte) and output head (lm_head). These encode the vocabulary mapping and game-specific output format. Without fine-tuning, the transferred attention/MLP weights produce activations that the randomly-initialised lm_head cannot interpret.
   - **Next Steps**: Fine-tuning the transferred model on Othello corpus is needed to measure transfer benefit. Same-game transfer (planner→player) may show stronger signal since the vocabularies overlap.
   - **K-13 Update — Fine-Tuning**: Added `organelle_train_transfer()` to the library. Transfer+FT condition: transfer TTT internal weights, then fine-tune 25K steps on Othello corpus. **Result: Transfer+FT=34% vs Scratch=30% (+4%)**. The transferred representations provide a measurable advantage when combined with target-domain fine-tuning, even across different games. Notable: initial loss spikes to 10.58 (vs 3.51 scratch) due to TTT↔Othello vocab mismatch, but the model recovers and outperforms.
   - **K-14 — Same-Game Transfer (Planner→Player)**: TTT Planner representations transferred to TTT Player. **Result: SCRATCH=79%, TRANSFER+FT=73%, RANDOM=64%**. Transfer+FT beats random by +9% (parse errors: 23 vs 305). Scratch is still 6% better, confirming that direct training outperforms transferred representations. The near-complete vocab overlap (29 vs 28 chars) did not amplify transfer benefit as initially hypothesised.
   - **K-15 — BLOCK_SIZE=256 Uplift**: Doubled context window for Hex 7×7 and Red Donkey 5×4. **Result: No improvement.** Hex 256=25% (same as 128=25%). Red Donkey 256=0% (same as 128=0%). The bottleneck is model capacity and game complexity, not context window size. Fixed hardcoded `block_size=128` in hex and reddonkey experiment code to use the `BLOCK_SIZE` macro.

### 3. **Overall Implications of Success/Failure**
- **Success (80-95% Across List)**: Proves OPA's **generalization ceiling**—from search (8-Puzzle) to constraints (Sudoku) and topology (Hex)—positioning MicroGPT-C as a framework for "micro-agents" in games/optimization. Implications: Attracts contributors (e.g., for chess minis); validates for real apps (e.g., EnX-MF's homology chaining). Echoes PDFs' "unbreakable rules"—OPA as "axioms" for puzzles.
- **Failure Modes**: High invalids (like C4 50%) prove kanban's limits on deep branching—implications: Need hybrid (e.g., deterministic Workers for evals). Low perf on Hex/Pentago highlights topology gaps—add PDFs-inspired manifolds.
- **Broader Proof**: Demonstrates "stem cell" evolution—organelles differentiate via corpora, chain for emergence. Ties to SLM trends (e.g., multi-agent puzzles)—proves edge AI viability (C99, threaded).

### 4. **How to Implement and Improve**
- **Gen Corpora**: Use Python (like generate_corpus.py)—minimax/BFS for 10-100K states (~hours). Focus subsets (e.g., mid-game Hex).
- **Organelle Scaling**: 15-20 (Planner + 8 Workers for branches + Judges)—test kanban on "traps" (P1 in reports).
- **Metrics**: Solve rate, invalids rescued, replans/game—aim 85%+ avg.
- **Improvements**: Add MD-delta (reports P1); hybrid search (MCTS Worker); geometry from PDFs (e.g., manifold evals for Hex voids). The topology experiments (Feb 2026) demonstrated that **BFS connectivity features + topological Judge + MCTS corpus generation** are effective zero-cost interventions.

This list pushes OPA's boundaries—start with Lights Out for quick wins!