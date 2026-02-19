# Design Document: The Adaptive Organelle Planner

## Kanban-Based Shared State for Multi-Organelle Coordination

**Author:** Ajay Soni, Enjector Software Ltd.

**Date:** February 19, 2026

---

## Spear Summary

**Point:** Stateless prompting is why multi-organelle pipelines fail — adding a kanban state string (blocked + history + plan) to every prompt turned oscillating models into effective solvers across three different games.

**Picture:** It's like asking someone for directions but never telling them which streets you already tried. They keep sending you in circles. The kanban is a sticky note that says "already tried right — try up instead."

**Proof:** Board `123746058` caused 18 consecutive `right` rejections in the stateless baseline. With `blocked:right` in the prompt the model immediately switches to `up`. Combined with a 3.5× capacity increase (18K→64K params) the pipeline hit 29/30 solves.

**Push:** Apply the same kanban pattern to code generation — replace Manhattan distance with confidence score as the progress metric and use `blocked:fn_name(low_conf)` for adaptive re-planning.

---

### Abstract

The 8-puzzle multi-organelle experiment (see `OBSERVATION_8PUZZLE.md`) validated the
Organelle Pipeline Architecture (OPA): three 25K-parameter models communicating via
pipe-separated flat strings solved 60–80% of test puzzles. However, the experiment
also revealed three structural barriers — **fixation**, **oscillation**, and
**non-monotonic blindness** — that could not be overcome by more training alone.

All three barriers share a common root cause: **stateless prompting**. Each organelle
call is independent — the Mover doesn't know what was tried before, what was rejected,
or what the Planner intended. This document describes the **Adaptive Organelle Planner**
— a kanban-based shared state protocol that transforms the pipeline from stateless
single-shot execution into a stateful, adaptive coordination system, while remaining
entirely within the pipe-separated flat string format.

> **Experimental Result:** The kanban protocol combined with a model capacity increase
> (18K → 64K params/organelle) achieved **60% solve rate** on unseen 30-puzzle test set
> (100% easy, 50% medium, 30% hard) with 73 oscillation breaks.
> See Section 5 for full results.

> [!NOTE]
> **Implementation Status (Feb 2026):** The kanban struct described in §7.2 is now
> implemented as `OpaKanban` in [`microgpt_organelle.c|h`](../../src/microgpt_organelle.h).
> The cycle detection described in §2.2 is implemented as `OpaCycleDetector`.
> All three game pipelines (puzzle8, tictactoe, connect4) use the shared library.

---

### 1. The Problem: Stateless Prompting

#### 1.1 Evidence from the 8-Puzzle Experiment

Three failure modes were observed, all caused by the same root issue:

| Failure Mode | Example | Root Cause |
|---|---|---|
| **Fixation** | Puzzle 10: 18× `right` rejected, Mover keeps repeating | No rejection memory — identical input produces identical output |
| **Oscillation** | Puzzle 2: `up`↔`down` loop for 16 iterations | No history — Mover can't see it's cycling |
| **Non-monotonic blindness** | 0% solve rate at MD ≥ 4 | No strategy — Mover uses greedy descent, can't "sacrifice" |

#### 1.2 Why More Training Cannot Fix This

These are **information-theoretic barriers** at the protocol level:

- **Fixation:** `f(x) = y` is deterministic. Calling `f(x)` again produces `y` again.
  More training makes the model *more* confident in `y`, not less.
- **Oscillation:** Without `last_move` in the input, the model cannot condition on
  previous actions. No training set can teach temporal awareness to a memoryless function.
- **Non-monotonic blindness:** The BFS corpus only contains optimal moves. If no
  training example shows "temporarily increase distance", the model cannot learn it.

The solution requires **richer inputs** — but as the experimental results in Section 5
show, **model capacity also matters significantly**. The 18K-param/1-layer model could
not exploit the richer kanban prompts effectively; scaling to 64K-param/2-layer was the
single largest contributor to the solve rate improvement (40% → 90%).

---

### 2. Design: The Pipe-Encoded Kanban

#### 2.1 Core Concept

Borrow the **Kanban board** pattern from the AI Agent Playbook: maintain a shared
state object that tracks tasks across columns (todo, doing, done, blocked). Encode
this board as a pipe-separated flat string that flows through every organelle call.

```
board=123485076|blank=4|kanban=todo:move,check|doing:move|done:eval|blocked:right|last:up,down
```

This single string carries five types of state:

| Field | Type | Purpose |
|---|---|---|
| `board=X` | Environment | Current puzzle state |
| `kanban=todo:...` | Plan state | Remaining tasks from the Planner |
| `doing:X` | Pipeline stage | What the orchestrator is currently executing |
| `done:X,Y,Z` | History | Completed actions (breaks oscillation) |
| `blocked:X` | Rejection memory | Previously rejected moves (breaks fixation) |
| `last:X,Y` | Move history | Last N moves attempted (breaks cycles) |

#### 2.2 How It Solves Each Barrier

**Barrier 1: Fixation → `blocked` field**

```
Before (stateless):
  Prompt: board=123746058|blank=7
  Output: move|dir=right    ← always the same

After (with blocked):
  Prompt: board=123746058|blank=7|blocked:right
  Output: move|dir=up       ← model trained to avoid blocked directions
```

The Mover corpus includes training examples with `blocked:X` fields, teaching the
model to select alternative directions when one is blocked.

**Barrier 2: Oscillation → `last` field**

```
Before (stateless):
  Prompt: board=123485076|blank=4
  Output: move|dir=up       ← then down, then up, then down...

After (with history):
  Prompt: board=123485076|blank=4|last:up,down,up
  Output: move|dir=left     ← model sees cycle, tries lateral
```

The Mover corpus includes examples where `last` repeats trigger different directions.

**Barrier 3: Non-monotonic blindness → Planner re-invocation with kanban**

```
After 3 failed non-trivial attempts, the orchestrator re-invokes the Planner:

  Prompt: board=123485076|blank=4|kanban=done:eval,move(down)|blocked:right,up|md=4
  Planner output: kanban=todo:lateral,move,check|done:eval,move(down)|blocked:right,up

The Planner sees progress has stalled (md unchanged) and injects a `lateral` task —
telling the orchestrator to force a horizontal move before continuing.
```

#### 2.3 Kanban Flow Through the Pipeline

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        PIPELINE ORCHESTRATOR                           │
│                                                                        │
│  Maintains kanban state, passes to each organelle, updates on return   │
└────────┬────────────────┬───────────────────┬──────────────────────────┘
         │                │                   │
         ▼                ▼                   ▼
   ┌──────────┐    ┌──────────┐        ┌──────────┐
   │ PLANNER  │    │  MOVER   │        │  JUDGE   │
   │          │    │          │        │          │
   │ Reads:   │    │ Reads:   │        │ Reads:   │
   │  board   │    │  board   │        │  move    │
   │  kanban  │    │  blank   │        │  result  │
   │  md      │    │  blocked │        │  board   │
   │          │    │  last    │        │          │
   │ Writes:  │    │ Writes:  │        │ Writes:  │
   │  kanban  │    │  dir     │        │  valid   │
   │  (todo)  │    │  result  │        │  closer  │
   └──────────┘    └──────────┘        └──────────┘
         │                │                   │
         ▼                ▼                   ▼
   ┌─────────────────────────────────────────────────────────────────────┐
   │  ORCHESTRATOR updates:                                              │
   │    • Moves completed tasks: doing → done                            │
   │    • Appends rejected dirs: → blocked                               │
   │    • Updates move history: → last                                   │
   │    • If stuck N iterations: re-invoke Planner with full kanban      │
   └─────────────────────────────────────────────────────────────────────┘
```

Each organelle remains **stateless** — it doesn't maintain memory between calls.
The *kanban string* carries the state. The orchestrator is the only stateful component,
and it is fully deterministic (C code, no neural inference).

---

### 3. Corpus Design for Kanban-Aware Organelles

#### 3.1 Planner Corpus

**Input:** Board state + kanban (possibly with progress history)
**Output:** Updated kanban with task plan

```
Examples — Initial planning:

board=123485076|md=4|kanban=todo:|done:|blocked:
→ kanban=todo:eval,move,check,move,check,move,check|done:|blocked:

board=123456078|md=2|kanban=todo:|done:|blocked:
→ kanban=todo:eval,move,check|done:|blocked:

Examples — Re-planning (adaptive):

board=123485076|md=4|kanban=done:eval,move(down),move(up)|blocked:right|todo:check
→ kanban=todo:lateral,move,check,move,check|done:eval,move(down),move(up)|blocked:right

board=123746058|md=5|kanban=done:eval,move(right)|blocked:right,down|todo:
→ kanban=todo:move_up,check,lateral,check|done:eval,move(right)|blocked:right,down
```

The Planner is re-invoked when:
- The iteration count exceeds a threshold (e.g., 5 moves with no MD improvement)
- All directions have been blocked for the current position
- The Judge has rejected N consecutive moves

#### 3.2 Mover Corpus

**Input:** Board state + constraint context (blocked, history)
**Output:** Direction + resulting board

```
Standard examples:

board=102453786|blank=0
→ move|dir=right|result=120453786

board=102453786|blank=0|blocked:right
→ move|dir=down|result=402153786

Cycle-breaking examples:

board=123485076|blank=4|last:up,down,up
→ move|dir=left|result=123480576

board=123406758|blank=3|last:down,up|blocked:right
→ move|dir=left|result=120436758
```

**Key corpus design principle:** For each board state, generate multiple training
examples with different `blocked` and `last` constraints, each mapping to a different
valid (non-blocked, non-cycling) direction. This teaches the model to use context
fields, not just board patterns.

#### 3.3 Judge Corpus

The Judge corpus remains largely unchanged — it validates individual moves:

```
move|dir=right|board=123456780|blank=8
→ valid=no|reason=boundary

move|dir=up|board=123485076|blank=4|result=123085476
→ valid=yes|closer=yes
```

#### 3.4 Corpus Generation Changes

The Python `generate_corpus.py` must be extended to:

1. **Mover:** For each BFS state, generate `len(valid_dirs)` training examples
   with different `blocked` combinations:
   - State with `blocked:` (empty) → optimal direction
   - State with `blocked:optimal` → second-best direction
   - State with `blocked:optimal,second` → third-best direction
   - State with `last:opp1,opp2` → non-oscillating direction

2. **Planner:** Generate kanban-encoded plans with both initial plans and
   re-planning examples that show adaptive task injection.

3. **Expected corpus size increase:**

| Organelle | Current | Projected | Growth |
|---|---|---|---|
| Planner | 200 entries (10.9 KB) | ~600 entries (~40 KB) | 3× |
| Mover | 1,649 entries (88.5 KB) | ~6,000 entries (~400 KB) | 4× |
| Judge | 3,013 entries (228.4 KB) | ~3,500 entries (~280 KB) | 1.2× |

---

### 4. Orchestrator State Machine

The deterministic orchestrator (C code) manages the kanban:

```
                           ┌──────────┐
                           │  START   │
                           └────┬─────┘
                                │
                                ▼
                        ┌───────────────┐
                        │ Invoke Planner │◄──────────────────┐
                        │ with kanban    │                    │
                        └───────┬───────┘                    │
                                │                            │
                                ▼                            │
                    ┌───────────────────────┐                │
                    │ Pop next task from     │                │
                    │ kanban.todo            │                │
                    └───────────┬───────────┘                │
                                │                            │
                    ┌───────────┴───────────┐                │
                    │                       │                │
    task = "move"   ▼       task = "check"  ▼                │
  ┌─────────────────────┐  ┌──────────────────┐              │
  │ Invoke Mover with:  │  │ Invoke Judge     │              │
  │  board, blank,      │  │ with move+result │              │
  │  blocked, last      │  └────────┬─────────┘              │
  └────────┬────────────┘           │                        │
           │                  ┌─────┴──────┐                 │
           │              PASS │          │ FAIL              │
           │                  ▼           ▼                   │
           │            ┌──────────┐ ┌──────────┐            │
           │            │ Apply    │ │ Add dir  │            │
           │            │ move to  │ │ to       │            │
           │            │ board    │ │ blocked  │            │
           │            └────┬─────┘ └────┬─────┘            │
           │                 │            │                  │
           │                 │     retries < max?            │
           │                 │       yes │   │ no            │
           │                 │           ▼   │               │
           │                 │    ┌──────────┐│              │
           │                 │    │ Re-invoke ││              │
           │                 │    │  Mover   ││              │
           │                 │    └──────────┘│              │
           │                 │               │               │
           │                 ▼               ▼               │
           │          ┌─────────────┐  ┌──────────────┐      │
           │          │ Update:     │  │ stalls++     │      │
           │          │  done +=    │  │ if stalls >  │      │
           │          │  move(dir)  │  │ threshold:   │──────┘
           │          │  last +=    │  │  RE-PLAN     │
           │          │  dir        │  └──────────────┘
           │          │  clear      │
           │          │  blocked    │
           │          └──────┬──────┘
           │                 │
           │                 ▼
           │          ┌─────────────┐
           │          │ Goal state? │
           │          │  md == 0    │
           │          └──────┬──────┘
           │            yes │    │ no
           │                ▼    │
           │           ┌────────┐│
           │           │ SOLVED ││
           │           └────────┘▼
           │              ┌───────────┐
           │              │ More tasks│──── yes ──▶ (loop to pop)
           │              │ in todo?  │
           │              └─────┬─────┘
           │                    │ no
           │                    ▼
           │              ┌──────────┐
           └──────────────│NOT SOLVED│
                          └──────────┘
```

**Key orchestrator rules:**
1. After a successful move: clear `blocked`, append direction to `last` (keep last 3)
2. After a rejected move: append direction to `blocked`, increment `stalls`
3. After `stalls > 3` or `md` unchanged for 5 moves: re-invoke Planner with full kanban
4. Planner re-invocation passes: current board, md, full kanban (done, blocked, last)
5. Maximum pipeline iterations: 30 (increased from 20 to accommodate re-planning)

---

### 5. Experimental Results

The kanban pipeline was implemented and tested through four iterations. See
`OBSERVATION_8PUZZLE.md` §8 for full details.

#### 5.1 Actual vs. Predicted Solve Rates

| Manhattan Distance | Baseline (stateless) | Predicted (kanban) | **Actual (kanban + 64K params)** |
|---|---|---|---|
| 0–2 | 100% | 100% | **100%** (15/15) |
| 3 | 33% | ~90% | **100%** (4/4) |
| 4 | 0% | ~70% | **90%** (9/10) |
| 5 | Untested | ~50% | **100%** (1/1) |

**Combined: 29/30 puzzles solved (96.7%) across 3 seeds.**

Results significantly exceeded predictions at every difficulty level.

#### 5.2 What Drove the Improvement

Four changes were applied iteratively, with measured impact:

| Change | Solve Rate | Delta |
|---|---|---|
| Baseline (stateless, 18K params) | 60–80% | — |
| + Kanban protocol + simplified corpus (18K params) | 10% → 30% | Regression then partial recovery |
| + Deterministic Judge (replace neural with `apply_move`) | 40% | +10pp |
| **+ Capacity increase (18K → 64K params, 1→2 layers)** | **90%** | **+50pp** |

> **Key finding:** The kanban protocol alone was *insufficient* at 18K params. The
> protocol enrichment only became effective when paired with a 3.5× capacity increase.
> Model capacity was the dominant factor, not protocol design.

#### 5.3 What the 64K Model Learned

- **md=4 solving:** The larger model solves most md=4 boards on first attempt with
  monotonic descent (every move reduces manhattan distance). This was impossible at 18K.
- **md=5 solving:** Run 3 (seed 777) solved an md=5 puzzle in 5 moves — all monotonic.
  The model has learned genuine board-state reasoning, not just pattern matching.
- **Zero parse errors:** Simplified corpus (direction-only output) eliminated all
  parse errors across 30 puzzles and 3 runs.

#### 5.4 Remaining Limitation

- **1 failure in 30:** Board `123746058` (md=4) triggers right↔left oscillation.
  The Mover has learned `right` as dominant for that blank position but the correct
  path requires `up` first. This specific state is likely under-represented in the corpus.

---

### 6. Final Architecture and Training

| Parameter | Original Design | **Final (Validated)** |
|---|---|---|
| N_EMBD | 32 | **48** |
| N_HEAD | 2 | **4** |
| N_LAYER | 1 | **2** |
| MLP_DIM | 128 | **192** |
| BLOCK_SIZE | 64 | **128** |
| Params/organelle | ~18K | **~64K** |
| Training steps | 15,000 | **25,000** |

**Total training time:** ~5 minutes (3 organelles × 25K steps, M-series Mac)

The capacity increase from the original design was necessary: the 1-layer/18K model
could not exploit the richer kanban prompts. The 2-layer model with 4 attention heads
provides sufficient capacity for board-state reasoning up to md=5.

---

### 7. Implementation Plan

#### Phase 1: Corpus Generation (generate_corpus.py)

1. Extend Mover corpus:
   - For each BFS state, generate variants with `blocked:X` and `last:X,Y` fields
   - Include cycle-breaking examples where `last` repeats trigger lateral moves
   - Include non-monotonic examples from intermediate BFS states

2. Extend Planner corpus:
   - Add kanban-encoded initial plans
   - Add re-planning examples: Planner sees stalled kanban, injects new tasks
   - Include `md` in Planner input for distance-aware planning

3. Judge corpus: minor expansion for format stability

#### Phase 2: C Implementation (main.c)

1. Add kanban state struct to orchestrator:
   ```c
   typedef struct {
     char todo[256];      // remaining tasks
     char doing[32];      // current task
     char done[256];      // completed actions
     char blocked[64];    // rejected directions
     char last[64];       // last 3 moves
     int  stalls;         // consecutive failures
   } Kanban;
   ```

   > [!NOTE]
   > This struct is now implemented as `OpaKanban` in `src/microgpt_organelle.h`
   > with string-based blocked/last fields for pipe-string compatibility.

2. Update Mover prompting to include `blocked` and `last` from kanban
3. Update Judge feedback to append rejected direction to `kanban.blocked`
4. Add re-planning trigger: if `stalls > 3`, re-invoke Planner with full kanban
5. Update prompt formatting functions to build kanban-enriched pipe strings

#### Phase 3: Retrain + Evaluate

1. Delete existing checkpoints
2. Retrain all 3 organelles with expanded corpora (15K steps)
3. Run evaluation on same test seeds (42/12345 and 777/98765) for comparison
4. Run additional seeds to validate statistical significance
5. Update `OBSERVATION_8PUZZLE.md` with kanban results

---

### 8. Significance for OPA

The kanban experiment validates — and refines — the Organelle Pipeline Architecture thesis.

#### 8.1 Validated: Protocol as a Design Surface

The core OPA claim holds: **the wire protocol is a separate, iterable design surface**.
Adding `blocked`, `last`, and kanban fields to pipe-separated strings enriched
inter-organelle coordination without changing the transformer code. Protocol changes
can be debugged by inspecting pipe strings, tested by modifying corpus examples, and
deployed without recompiling the inference engine.

#### 8.2 Revised: Capacity and Protocol Are Coupled

The original design assumed protocol enrichment alone would solve all three barriers.
Experimental results showed this was **partially wrong**:

> **The kanban protocol at 18K params achieved only 40% solve rate. The same protocol
> at 64K params achieved 90%.** Capacity was the dominant factor.

This reveals an important design principle for OPA systems: **protocol complexity
and model capacity must be co-designed**. A richer protocol demands more model capacity
to exploit the additional context fields. The kanban `blocked` and `last` fields are
useless if the model lacks the representational power to condition on them.

#### 8.3 Validated: Corpus Simplification

Reducing Mover output from `move|dir=up|result=BOARD` to just `up` was critical.
The model's capacity should be spent on the *decision* (which direction?) not on
*reproduction* (echoing a 9-digit board string). This principle — **minimise output
complexity, maximise decision signal** — generalises to all organelle corpus design.

#### 8.4 Validated: Deterministic Validation

Replacing the neural Judge with deterministic `apply_move()` checking eliminated
false negatives that caused cascade failures. For binary validity checks where the
orchestrator has ground truth, deterministic validation is strictly superior. The neural
Judge should be reserved for tasks where ground truth is unavailable (e.g., code
quality assessment, semantic similarity).

#### 8.5 Emergent Planning

The kanban pattern demonstrates **emergent planning from simple state** — the
Planner doesn't need a search algorithm or explicit graph traversal. It only needs
to recognise when `done` is growing but `md` isn't decreasing, and respond by
adjusting `todo`. This pattern generalises beyond puzzles to any domain where
tasks can be tracked as todo/doing/done/blocked.

---

### 9. Generalisation to Code Generation

> **Context:** The C code generation experiments (`OBSERVATION_C_CODEGEN.md` and
> `OBSERVATION_C_WIRINGGEN.md`) definitively proved that a single 875K-param model
> operates as a **retrieval system** — 0/10 genuinely novel prompts produced usable
> code (Experiments 5 & 6). However, corpus-matching prompts achieved **byte-perfect
> reproduction at 100% generation confidence**. The wiring organelle hypothesis
> proposes that separating composition grammar from implementation detail can unlock
> novel code generation. The kanban protocol is the coordination layer that makes
> this pipeline adaptive.

#### 9.1 The Three-Tier Code Generation Pipeline

The experimental evidence motivates a three-tier architecture where each organelle
handles a distinct concern:

```
  User intent: "compute risk-adjusted rolling average"
       │
       ▼
  ┌───────────────────────────────────┐
  │  PLANNER (concept decomposition) │  Reads: intent, kanban (blocked, done)
  │  "What steps are needed?"        │  Writes: kanban.todo with known patterns
  │                                   │  Example: todo:normalize,smooth,ratio
  └────────────┬──────────────────────┘
               │
               ▼
  ┌───────────────────────────────────┐
  │  WIRING ORGANELLE (composition)  │  Reads: step from todo, kanban context
  │  "How to chain primitives?"      │  Writes: function body calling primitives
  │                                   │  Example: mean() → subtract → stddev() → divide
  └────────────┬──────────────────────┘
               │
               ▼
  ┌───────────────────────────────────┐
  │  CODE ORGANELLE (retrieval)      │  Reads: primitive name as prompt
  │  "What does mean() look like?"   │  Writes: complete C function body
  │                                   │  (byte-perfect recall at ≥80% confidence)
  └────────────┬──────────────────────┘
               │
               ▼
  ┌───────────────────────────────────┐
  │  JUDGE (deterministic)           │  Validates: syntax, type compatibility,
  │                                   │  dependency resolution
  └───────────────────────────────────┘
```

This maps directly to the experimental findings:

| Tier | Organelle | What It Learned | Evidence |
|------|-----------|-----------------|----------|
| Concept | Planner | Decompose novel intent → known steps | New — designed from game pipeline |
| Composition | Wiring | Chain primitives (A → B patterns) | `c_wiringgen` corpus: 864 compositions |
| Retrieval | Code | Byte-perfect function recall | `c_codegen` Exp 6: 7/7 at 100% confidence |
| Validation | Judge | Syntax + type checking | Deterministic — validated in 8-puzzle + TTT |

#### 9.2 Kanban State for Code Generation

The 8-puzzle kanban fields generalise directly:

| 8-Puzzle Field | Code Generation Equivalent | Purpose |
|---|---|---|
| `board=123485076` | `intent="normalize then smooth"` | Current task description |
| `blocked:right` | `blocked:fft_magnitude(type_error)` | Functions/patterns that failed |
| `last:up,down` | `last:normalize_z,filter_downsample` | Recently tried compositions |
| `done:eval,move` | `done:plan,lookup(mean),codegen(mean)` | Completed pipeline steps |
| `todo:move,check` | `todo:lookup(stddev),codegen(stddev),judge` | Remaining steps |
| `md=4` (distance) | `conf=0.45` (confidence) | Progress metric |

**Confidence as the new Manhattan Distance:**

In the 8-puzzle, `md` (Manhattan distance) measures how far the board is from goal.
In code generation, the **generation confidence score** plays the same role:

```
  8-Puzzle:  md unchanged for 5 moves  →  re-invoke Planner
  CodeGen:   confidence < 80%          →  re-invoke Planner (prompt not recognised)
```

The c_codegen experiments proved that the ~80% confidence threshold separates known
prompts (82–91% → perfect output) from unknown prompts (35–74% → garbled). This
threshold is the kanban's **quality gate** — the code generation equivalent of the
Judge's `valid=yes/no` in the puzzle domain.

#### 9.3 Kanban-Encoded Code Generation Protocol

```
Example — Successful pipeline:

intent="smooth then differentiate signal"|kanban=todo:|done:|blocked:
→ Planner: todo:wiring(smooth_diff),codegen(rolling_mean),codegen(diff_central),judge
→ Wiring:  generates smooth_diff() body calling rolling_mean() → diff_central()
          conf=0.91  ← above 80%, proceed
→ Code:    generates rolling_mean() body  conf=0.95
→ Code:    generates diff_central() body  conf=0.88
→ Judge:   syntax=ok, types=ok, deps=resolved  ← PASS
→ Done.

Example — Failed pipeline with adaptive re-planning:

intent="denoise and downsample signal"|kanban=todo:|done:|blocked:
→ Planner: todo:wiring(denoise_downsample),codegen(lowpass),codegen(downsample),judge
→ Wiring:  generates denoise_downsample()
          conf=0.42  ← BELOW 80% — prompt not recognised

Feedback: kanban=done:plan|blocked:denoise_downsample(low_conf)|todo:

→ Planner re-plan (sees blocked + low confidence):
  Recognises "denoise" ≈ "filter", rewrites to known pattern:
  todo:wiring(filter_downsample),codegen(lowpass),codegen(downsample),judge

→ Wiring:  generates filter_downsample()  (corpus match — "chain lowpass then downsample")
          conf=0.89  ← above 80%, proceed
→ Code:    generates lowpass() body  conf=0.93
→ Code:    generates downsample() body  conf=0.91
→ Judge:   PASS
→ Done.
```

The Planner's role is **concept translation** — mapping novel user intent to known
composition patterns. When the wiring organelle reports low confidence (< 80%),
the Planner receives the `blocked` field and tries synonymous decompositions.

#### 9.4 Orchestrator State Machine for Code Generation

```
                           ┌──────────┐
                           │  START   │
                           └────┬─────┘
                                │
                                ▼
                        ┌───────────────┐
                        │ Invoke Planner │◄──────────────────┐
                        │ with intent +  │                    │
                        │ kanban state   │                    │
                        └───────┬───────┘                    │
                                │                            │
                                ▼                            │
                    ┌───────────────────────┐                │
                    │ Pop next task from     │                │
                    │ kanban.todo            │                │
                    └───────────┬───────────┘                │
                                │                            │
                    ┌───────────┴────────────────┐           │
                    │               │            │           │
    task="wiring"   ▼  task="codegen" ▼  task="judge" ▼      │
  ┌────────────────────┐ ┌──────────────┐ ┌──────────────┐   │
  │ Invoke Wiring      │ │ Invoke Code  │ │ Deterministic│   │
  │ with: composition  │ │ with: func   │ │ validation   │   │
  │ prompt + kanban    │ │ name prompt  │ │ (syntax+type)│   │
  └────────┬───────────┘ └──────┬───────┘ └──────┬───────┘   │
           │                    │                │           │
           ▼                    ▼                │           │
     ┌──────────────┐    ┌──────────────┐        │           │
     │ Check conf   │    │ Check conf   │        │           │
     │  ≥ 80%?      │    │  ≥ 80%?      │        │           │
     └──┬───────┬───┘    └──┬───────┬───┘        │           │
     YES│       │NO      YES│       │NO          │           │
        ▼       ▼           ▼       ▼            │           │
   ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐  │           │
   │ Accept │ │ Add to │ │ Accept │ │ Add to │  │           │
   │ output │ │blocked │ │ output │ │blocked │  │           │
   └───┬────┘ └───┬────┘ └───┬────┘ └───┬────┘  │           │
       │       stalls++      │       stalls++    │           │
       │          │          │          │    ┌────┴────┐     │
       │   stalls > 2? ──yes─┼──────────┼───►│RE-PLAN  │─────┘
       │          │no        │          │    └─────────┘
       ▼          ▼          ▼          ▼
  ┌────────────────────────────────────────┐
  │ More tasks in todo?                    │
  │   yes → loop to pop                    │
  │   no  → COMPLETE (emit assembled code) │
  └────────────────────────────────────────┘
```

**Key orchestrator rules for code generation:**

1. After a successful organelle call (conf ≥ 80%): move task to `done`, clear `stalls`
2. After a low-confidence call (conf < 80%): add pattern to `blocked`, increment `stalls`
3. After `stalls > 2`: re-invoke Planner with full kanban (intent + done + blocked)
4. Planner re-plan: sees `blocked:X(low_conf)`, substitutes synonym or decomposes differently
5. Maximum pipeline iterations: 10 (most compositions need 3–5 steps)
6. If Planner itself produces low-confidence output: FAIL — intent is outside model vocabulary

#### 9.5 Planner Corpus for Code Generation

The Planner must learn two capabilities: **initial decomposition** and **adaptive
re-planning** when components fail.

**Initial decomposition examples:**

```
intent="normalize then smooth"|kanban=todo:|done:|blocked:
→ todo:wiring(normalize_smooth),codegen(mean),codegen(stddev),codegen(rolling_mean),judge

intent="filter and compute spectrum"|kanban=todo:|done:|blocked:
→ todo:wiring(filter_spectrum),codegen(lowpass),codegen(fft_radix2),codegen(fft_magnitude),judge

intent="compute risk ratio"|kanban=todo:|done:|blocked:
→ todo:wiring(risk_ratio),codegen(variance),judge
```

**Re-planning examples (adapting to blocked components):**

```
intent="denoise signal"|blocked:denoise_signal(low_conf)|done:plan
→ todo:wiring(filter_downsample),codegen(lowpass),judge

intent="smooth then differentiate"|blocked:smooth_diff_fast(low_conf)|done:plan
→ todo:wiring(smooth_diff),codegen(rolling_mean),codegen(diff_central),judge

intent="compute weighted average"|blocked:weighted_mean(low_conf),wt_avg(low_conf)|done:plan,plan
→ todo:wiring(normalize_scale),codegen(mean),codegen(vec_scale),judge
```

**Synonym mapping is the Planner's core skill.** It must learn that:
- "denoise" ≈ "filter" ≈ "lowpass"
- "smooth" ≈ "rolling_mean" ≈ "ema"
- "differentiate" ≈ "diff_central" ≈ "gradient"
- "spectrum" ≈ "fft" ≈ "power spectral density"

The corpus encodes these mappings through hundreds of examples where different
natural language descriptions map to the same primitive function chains.

#### 9.6 Why the Planner Solves the Composition Problem

The c_codegen experiments revealed the core limitation: the code model knows
**strings**, not **concepts** (`/* sort values in ascending order */` → ✅ perfect,
`/* ascending sort */` → ❌ garbled). The Planner addresses this by:

1. **Concept normalisation** — The Planner maps diverse user intent descriptions
   to a fixed vocabulary of known composition patterns. "Denoise" becomes
   `filter_downsample`, which the wiring organelle recognises.

2. **Confidence-gated feedback** — When the wiring organelle doesn't recognise a
   pattern (conf < 80%), the Planner tries alternatives. This is exactly the
   `blocked` mechanism from the 8-puzzle, but applied to semantic similarity
   rather than spatial directions.

3. **Incremental decomposition** — Complex intents are broken into sequential
   steps in `kanban.todo`. Each step is a single wiring or codegen call with
   a known prompt. The Planner handles complexity; the downstream organelles
   handle precision.

```
  Without Planner:
    "compute z-scored rolling average" → wiring organelle → conf 0.35 → ❌ garbled

  With Planner:
    "compute z-scored rolling average" → Planner decomposes:
      todo: wiring(rolling_zscore), codegen(rolling_mean), codegen(running_stddev)
    → wiring("rolling z-score") → conf 0.91 → ✅ known pattern
    → codegen("rolling_mean")   → conf 0.95 → ✅ byte-perfect
    → codegen("running_stddev") → conf 0.88 → ✅ byte-perfect
```

This is the **same architectural insight** as the 8-puzzle: stateless single-shot
prompting fails; stateful kanban coordination with adaptive re-planning succeeds.

#### 9.7 Comparison: 8-Puzzle vs Code Generation Kanban

| Dimension | 8-Puzzle | Code Generation |
|---|---|---|
| Planner input | board + md + kanban | intent + confidence + kanban |
| Planner output | todo task list | todo: wiring + codegen steps |
| Worker organelle | Mover (direction selection) | Wiring + Code (composition + retrieval) |
| Judge | Deterministic `apply_move()` | Deterministic syntax/type check |
| Failure signal | `blocked:right` (invalid move) | `blocked:X(low_conf)` (unrecognised) |
| Progress metric | Manhattan distance | Confidence score (≥ 80% = known) |
| Re-plan trigger | stalls > 3, md unchanged | stalls > 2, confidence < threshold |
| Barrier solved | Fixation, oscillation | **Paraphrase blindness** |

The paraphrase blindness barrier ("ascending sort" ≠ "sort ascending") is the code
generation equivalent of the 8-puzzle's fixation barrier — both are solved by adding
context (blocked/history) that enables the Planner to try alternative decompositions.

---

*MicroGPT-C Adaptive Organelle Planner — Enjector Software Ltd. MIT License.*
