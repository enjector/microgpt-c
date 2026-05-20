# BEHAVIOUR Catalogue — Per-Demo Wrapper Concerns

**Source experiment:** [E08](../../experiments/E08-oql-behaviours.md) Phase 1.
**Goal:** classify every wrapper function across the 11 game demos under the
six BEHAVIOUR concerns defined in E08 §1.3.1; identify the union of engine
primitives the externs will need.

The catalogue is the falsifier for E08's central claim that "every demo
implements some subset of these six concerns." If a demo has a wrapper that
fits none of the six, the catalogue *says so* — and the experiment STOPs per
the §1.3.1 falsification rule.

---

## 1. The six concerns (recap)

| ID | BEHAVIOUR concern | Where it lives in C today |
|----|---|---|
| **I** | `INPUT_BEHAVIOUR`     | Parse board / prompt → model-ready tokens |
| **O** | `OUTPUT_BEHAVIOUR`    | Format model's next-token → domain move |
| **V** | `VALIDATE_BEHAVIOUR`  | Test move legality in current state |
| **F** | `FALLBACK_BEHAVIOUR`  | Pick a fallback move when stuck |
| **S** | `SCORE_BEHAVIOUR`     | Rank multiple candidates (best-of-N) |
| **C** | `CYCLE_DETECT_BEHAVIOUR` | Detect A↔B oscillation across recent moves |

Two more orthogonal concerns surfaced during the walk; they are *not*
behaviours in the E08 sense (they're loops, not transformations), but are
listed here so the gap is visible:

- **Game-loop driver** — outer `while` over moves, opponent step, terminal-state check. Belongs in the C harness or in a future OQL `RUN PIPELINE ON game_loop` verb; not a BEHAVIOUR.
- **Per-demo training** — `organelle_train(...)` calls. Already covered by OQL `TRAIN` (E07 §1.3.1). Not a BEHAVIOUR.

---

## 2. Per-demo classification

Each row is one named C function from the demo's `main.c`. Concern column maps
the function to one of the six tags above (`-` = not a BEHAVIOUR concern; it's
glue / harness / pure helper).

### 2.1 `demos/character-level/connect4/main.c` (529 LOC)

| Function | Concern | Notes |
|---|---|---|
| `cell_idx(r,c)` | - | trivial coord helper |
| `board_to_str` | I | snprintf the board into a flat string for the prompt |
| `get_valid_columns` | V | enumerate legal column drops |
| `drop_piece` | (judge) | terminal-state machinery, not BEHAVIOUR |
| `count_pieces` | (judge) | terminal-state machinery |
| `check_winner` | (judge) | terminal-state machinery |
| `is_draw` | (judge) | terminal-state machinery |
| `print_board` | - | diagnostic only |
| `random_opponent_move` | F (also opponent) | doubles as random fallback when entropy spikes |
| inline `kb.stalls / kb.blocked` | C | kanban-bounded cycle detection, calls `opa_kanban_*` |

Wrapper concerns hit in connect4: **I, V, F, C.** No SCORE (single-candidate
generate). OUTPUT is one-token (char `'0'..'6'`), so `format_c4_move` is the
trivial identity wrapper named in the worked example for symmetry.

### 2.2 `demos/character-level/othello/main.c` (324 LOC)

| Function | Concern | Notes |
|---|---|---|
| `cell(r,c)` | - | coord helper |
| `board_to_str` | I | 64-char flat string |
| `init_board` | - | board state init |
| `get_flips` | V | helper for legality (which discs flip) |
| `get_valid_moves` | V | enumerate legal `(r,c)` for player |
| `apply_move` | (judge) | terminal-state machinery |
| `count_pieces` | (judge) | terminal-state machinery |

Wrapper concerns hit in othello: **I, V, F (via `opa_valid_fallback`), C
(kanban).** Same shape as connect4 with 2-axis (r,c) moves instead of 1-axis
column drop.

### 2.3 `demos/character-level/pentago/main.c` (296 LOC)

| Function | Concern | Notes |
|---|---|---|
| `cell(r,c)` | - | coord helper |
| `board_to_str` | I | 36-char flat string |
| `rotate_quad` | (judge) | board state mutation (rule-specific) |
| `check_five` | (judge) | terminal-state |
| `get_empties` | V | legality |

Wrapper concerns hit in pentago: **I, V, F, C.** `OUTPUT_BEHAVIOUR` is
composite (cell + quadrant + direction) — three tokens, not one. This is a
*non-trivial* OUTPUT example.

### 2.4 `demos/character-level/mastermind/main.c` (374 LOC)

| Function | Concern | Notes |
|---|---|---|
| `compute_score` | S | black/white peg scoring; this IS a SCORE_BEHAVIOUR |
| `is_valid_guess` | V | check guess length & alphabet |
| `generate_secret` | - | game setup |

Wrapper concerns hit in mastermind: **I (guess parse), V, S (the only one in the
catalogue with a real SCORE concern — peg counting against a held secret).**
Mastermind is the deduction game — F/C have minimal role.

### 2.5 `demos/character-level/puzzle8/main.c` (599 LOC)

| Function | Concern | Notes |
|---|---|---|
| `manhattan_distance` | S | heuristic score for best-of-N |
| `find_blank` | V | helper for legality |
| `get_valid_dirs` | V | enumerate up/down/left/right legality |
| `apply_move(dir)` | (judge) | state mutation |
| `is_goal` | (judge) | terminal-state |
| `board_to_str` | I | flat string |
| `md_delta_str` | I | append "md=Δ" for the player prompt |
| `dir_to_id` | O | parse direction char from model output |
| `scramble_puzzle` | - | game setup |
| `scramble_to_target_md` | - | game setup |

Wrapper concerns hit in puzzle8: **I, O, V, F, S, C.** Most concerns lit. This
is the canonical demo for cross-concern stress.

### 2.6 `demos/character-level/tictactoe/main.c` (400 LOC)

Concerns: **I, V, F, C.** Smaller 3×3 board, single-axis output. Same shape as
connect4.

### 2.7 `demos/character-level/sudoku/main.c` (410 LOC)

| Function | Concern | Notes |
|---|---|---|
| `is_valid_placement` | V | row/col/box constraint |
| `is_solved` | (judge) | terminal-state |
| `solve_sudoku` | - | (only used in puzzle generation pipeline) |
| `generate_solved_grid` | - | game setup |
| `generate_puzzle` | - | game setup |
| `count_empty` | I | for prompt context |
| `board_to_str` | I | 81-char flat string |

Wrapper concerns hit in sudoku: **I, V, F.** No CYCLE_DETECT (one-shot fill,
not bidirectional play). No SCORE (constraint-prop, not best-of-N).

### 2.8 `demos/character-level/hex/main.c` (545 LOC)

| Function | Concern | Notes |
|---|---|---|
| `get_empties` | V | enumerate legal cells |
| `count_groups` | S | strategic heuristic |
| `shortest_edge_distance` | S | strategic heuristic |
| `count_bridges` | S | strategic heuristic |
| `count_virtual_connections` | S | strategic heuristic |
| `is_bridge_cell` | - | predicate for the above scoring helpers |
| `check_connection` | (judge) | terminal-state (win-by-connection) |

Wrapper concerns hit in hex: **I, V, F, S, C.** Hex's SCORE concern is the
richest in the catalogue — four orthogonal score signals.

### 2.9 `demos/character-level/lightsout/main.c` (408 LOC)

| Function | Concern | Notes |
|---|---|---|
| `count_lit` | I | for prompt context (also terminal-check) |
| `toggle_cell` | (judge) | state mutation |
| `is_solved` | (judge) | terminal-state |
| `generate_puzzle` | - | game setup |

Wrapper concerns hit in lightsout: **I, V, F, C.** Output is one cell index;
trivial.

### 2.10 `demos/character-level/klotski/main.c` (331 LOC)

| Function | Concern | Notes |
|---|---|---|
| `dir_delta` | - | direction-char → (dr,dc) helper |
| `can_move(block,dir)` | V | legality |
| `apply_move(block,dir)` | (judge) | state mutation |
| `is_goal` | (judge) | terminal-state |
| `get_valid_moves` | V | enumerate all (block,dir) pairs |
| `generate_puzzle` | - | game setup |

Wrapper concerns hit in klotski: **I, O (block+dir composite), V, F, C.**

### 2.11 `demos/character-level/reddonkey/main.c` (355 LOC)

Same shape as klotski (it IS a variant of klotski with a fixed starting
position). Concerns: **I, O, V, F, C.**

---

## 3. Concern coverage matrix

| Demo | I | O | V | F | S | C |
|---|:-:|:-:|:-:|:-:|:-:|:-:|
| connect4    | ✓ | ✓ | ✓ | ✓ |   | ✓ |
| othello     | ✓ |   | ✓ | ✓ |   | ✓ |
| pentago     | ✓ | ✓ | ✓ | ✓ |   | ✓ |
| mastermind  | ✓ | ✓ | ✓ |   | ✓ |   |
| puzzle8     | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| tictactoe   | ✓ |   | ✓ | ✓ |   | ✓ |
| sudoku      | ✓ |   | ✓ | ✓ |   |   |
| hex         | ✓ |   | ✓ | ✓ | ✓ | ✓ |
| lightsout   | ✓ |   | ✓ | ✓ |   | ✓ |
| klotski     | ✓ | ✓ | ✓ | ✓ |   | ✓ |
| reddonkey   | ✓ | ✓ | ✓ | ✓ |   | ✓ |
| **Coverage** | 11/11 | 6/11 | 11/11 | 10/11 | 3/11 | 9/11 |

**Findings.**
- **`INPUT_BEHAVIOUR` and `VALIDATE_BEHAVIOUR` are universal.** Every game
  parses a board and tests legality — these are the load-bearing two.
- **`FALLBACK_BEHAVIOUR`** is present in 10/11; only mastermind lacks it
  (deduction games don't have a "the model is stuck, pick something legal"
  path; the model either guesses or doesn't).
- **`CYCLE_DETECT_BEHAVIOUR`** is present in 9/11; sudoku and mastermind
  don't oscillate (one-shot fill / monotonic guess sequence).
- **`OUTPUT_BEHAVIOUR`** is *composite* in 6/11 cases (multi-token moves) and
  *identity* in 5/11 (single-token moves where the token IS the move). The
  trivial identity form is still a real wrapper — it asserts the output is in
  the expected lexical class.
- **`SCORE_BEHAVIOUR`** is rarest (3/11). It's a real concern in puzzle8 (MD
  heuristic), hex (strategic scoring), and mastermind (peg counting). The
  other 8 demos use "first legal output wins" — no best-of-N today.

**No wrapper function in any of the 11 demos fell *outside* the six
concerns.** The six-concern taxonomy is empirically complete at the catalogue
scale. (Some functions are not BEHAVIOURs — `dir_delta`, `cell_idx`, etc. —
but they are pure helpers, not coordination layers; they belong inside the
relevant BEHAVIOUR body, not as their own behaviour.)

---

## 4. Engine primitives the externs will need

The VM's `vm_engine_register_fn` ABI is `double(int argc, const double *argv)`
— a number-only interface. String I/O between OQL/VM and the C host is
modeled via *integer handle IDs* into a side table owned by the host. The
catalogue below lists every primitive the four Connect-4 behaviours and the
broader replication set would need; the Phase 2 extern table extension
materialises them as registered natives.

### 4.1 Connect-4 worked example (Phase 4 in-scope)

| `declare function` signature | Backing C call | Notes |
|---|---|---|
| `c4_legal_columns(board_handle: number, out_mask: number): number` | `get_valid_columns` | returns a 7-bit mask of legal columns |
| `c4_column_is_legal(board_handle: number, col: number): boolean` | mask + col check | gate for VALIDATE |
| `c4_parse_token(tok_handle: number): number` | single-digit parse | the OUTPUT body — token "3" → 3 |
| `c4_centre_col(): number` | constant 3 | preferred FALLBACK column |
| `c4_last_entropy(): number` | model.last_logit_entropy | FALLBACK gate |
| `c4_token_handle(c: number): number` | char → handle | for tests that hand-build a token |
| `c4_board_handle_from_str(s_handle: number): number` | char-array → board id | for tests that hand-build a board |
| `str_intern(literal_id: number): number` | side-table id → cstr | mirrors the wiring_natives precedent |

### 4.2 Cross-demo union (Phase 5 — DEFERRED)

A Phase 5 replication across mastermind/pentago/puzzle8 would add:

| Primitive | Used by |
|---|---|
| `kanban_add_blocked(action_id)` | all CYCLE_DETECT |
| `kanban_clear_blocked()` | all CYCLE_DETECT |
| `kanban_is_blocked(action_id)` | all CYCLE_DETECT |
| `kanban_last_history(k)` | CYCLE_DETECT lookback |
| `cycle_detect_record(id)` | OpaCycleDetector wrapper |
| `cycle_detect_check(id)` | OpaCycleDetector wrapper |
| `manhattan_dist(board_handle)` | puzzle8 SCORE |
| `mastermind_peg_score(guess_handle, secret_handle, *black, *white)` | mastermind SCORE |
| `random_legal_pick(valid_mask, rng_seed)` | universal FALLBACK |
| `model_next_token_handle()` | universal OUTPUT |
| `model_last_entropy()` | universal FALLBACK gate |

The Phase 5 union is **~12 extra externs on top of Phase 4's 8**. None
require a new VM opcode — they are all callable through
`vm_engine_register_fn`'s existing `opCALL_EXT_METHOD` dispatch.

### 4.3 Zero-opcode confirmation

Every primitive above maps to:
1. one C function with the VM's existing `vm_native_fn` ABI, and
2. registration via the existing `vm_engine_register_fn` table.

No primitive requires:
- new bytecode (would need `opcode` enum extension in `vm_instruction_opcode_t`)
- new TS surface (existing `declare function` covers it)
- new verifier pass (existing 6-pass verifier handles `opCALL_EXT_METHOD`)
- new runtime stack instruction (existing `vm_module_runtime_run` handles dispatch)

**T3's zero-new-opcodes lock holds at the catalogue level.** Phase 2 will
confirm this in code.

---

## 5. What this catalogue is *not*

- It is not a proof that the rewrite *will* hit 88% Connect-4 win-rate. That
  is T1 and is measured in Phase 4.
- It is not a claim that the cross-demo extern union is small. It's ~20
  primitives; the Phase 5 LOC ratio is not measured here.
- It does not commit to Phase 5. Per the run brief, Phase 5 is deferred.
