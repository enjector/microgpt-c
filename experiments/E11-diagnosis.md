# E11 — Phase 1 Diagnosis: Connect-4 OQL win-rate root cause

**Status:** Diagnostic note. Produced before Pathway pick. Commit prefix `E11: diag:`.

**Pre-reg:** [E11-connect4-win-rate-fix.md §1.3 Mechanism, Phase 1](E11-connect4-win-rate-fix.md).

---

## 1. Baseline reproduction

Built `oql_c4` against the E09 head, copied the existing `c_connect4_planner.ckpt`
and `c_connect4_player.ckpt` from `models/character-level/` into
`build/checkpoints/c4_planner.ckpt` / `build/checkpoints/c4_player.ckpt`, and ran:

```
$ ./build/oql_c4 run experiments/connect4.oql
load_organelle: loaded 'connect4_planner' from checkpoints/c4_planner.ckpt (vocab=30 step=25000)
RUN connect4: 100 games | wins=51 draws=0 losses=49 (win_rate=51.0%)
              p99_latency=0.00ms audit_rows=1065 model_loaded=yes total=0.00s
```

51% — exactly the E09 §3.4 baseline. **Reproduced.**

---

## 2. Three independent gaps surfaced

### Gap A — model is never queried for a move (the headline gap)

Lifted directly from `src/oql_runtime_games.c` lines 176–197:

```c
static int oql_model_propose_column(const Model *model, unsigned int *seed,
                                    int legal_mask) {
    if (!model) return -1;
    /* For E09 we don't yet have a per-game KV state machine — the safe
     * path is to argmax the model's bias / output logits given a one-shot
     * forward pass.  A full board→token pipeline lives in the C demo
     * (~150 LOC of corpus-encoded prompts); replicating that here is
     * scoped out of E09 (T7 forbids LOC explosion).  Instead we sample
     * a column uniformly from the legal mask using the model's
     * RNG-equivalent forward weights as a randomness source — produces
     * a deterministic-given-seed but model-influenced column.
     *
     * E10/E11 will lift the full prompt protocol into oql_runtime_games. */
    (void)model;
    int legal[OQL_C4_COLS];
    int n = 0;
    for (int c = 0; c < OQL_C4_COLS; c++) {
        if (legal_mask & (1 << c)) legal[n++] = c;
    }
    if (n == 0) return -1;
    return legal[rand_r(seed) % n];
}
```

The model pointer is cast to void at line 189. The function uniformly
samples a column from the legal mask. The model has zero influence
on the move chosen. 51% is the random-vs-random win rate when X
moves first and slight tie-breaking favors X — the C demo's 88% is
the model-driven number.

### Gap B — wrong checkpoint loads as the move-generator

`connect4.oql` line 105–106:

```sql
COMPOSE connect4_pipeline
  FROM connect4_planner, connect4_player;
```

The runtime (oql_runtime_games.c line 270) picks
`pipeline->call_organelles[0]` as the player. The first slot is
`connect4_planner`. So **the planner's checkpoint is loaded and would
have been the move source if Gap A weren't already nullifying it**.

The C demo's logic: planner runs once at game start (for a "todo"
plan), then *player* runs per-move. The OQL runtime loop currently
only invokes one organelle per move, and it's whichever is first in
the COMPOSE chain.

For E11, swapping the COMPOSE order to put player first is a
correctness fix and a no-cost change. It's not the "prompt protocol"
issue itself, but if E11 closes Gap A and leaves Gap B, the runtime
would run inference against the *planner* checkpoint, which was
trained on planner data (`todo=...` outputs), not player data
(`0..6` column outputs). The win rate would be capped by that
mismatch.

### Gap C — the prompt protocol itself

The C demo (`demos/character-level/connect4/main.c` lines 311–340)
constructs the player prompt as:

```c
char player_prompt[256];
if (kb.blocked[0] != '\0') {
  snprintf(player_prompt, sizeof(player_prompt),
           "board=%s|valid=%s|blocked=%s",
           board_str, valid_str, kb.blocked);
} else {
  snprintf(player_prompt, sizeof(player_prompt),
           "board=%s|valid=%s",
           board_str, valid_str);
}
/* ... */
organelle_generate_ensemble(player, &g_cfg, player_prompt, move_output,
                            INF_GEN_LEN /* = 60 */, ENSEMBLE_VOTES /* = 3 */,
                            ORGANELLE_TEMP /* = 0.2 */, &vote_conf);
/* Parse first char of move_output as '0'..'6'. */
```

Where:
- `board_str` is the 42-character board string (cells = `.`, `X`, `O`),
- `valid_str` is the comma-separated list of legal columns (e.g.
  `"0,2,3,4,5,6"`),
- `kb.blocked` is the comma-separated list of "tried-but-illegal"
  columns this turn (empty on the first attempt of a turn).

The model was *trained* on documents of the form:

```
board=..........................................|valid=0,1,2,3,4,5,6
3
```

i.e. the prompt is the entire `board=…|valid=…[|blocked=…]` string,
the response is the chosen column digit.  Verified by reading the
first 10 lines of `demos/character-level/connect4/c_connect4_player.txt`:

```
board=..........................................|valid=0,1,2,3,4,5,6
3

board=..........................................|valid=1,2,3,4,5,6|blocked=0
3
…
```

The current OQL behaviour `parse_c4_board`:

```typescript
declare function c4_legal_column_mask(): number;
function eval(): number {
    var mask = c4_legal_column_mask();
    return mask;
}
```

Only returns a 7-bit legal mask.  The string `board=…|valid=…` is
never constructed and never fed to the model.

---

## 3. Byte-exact side-by-side, three sample board states

### 3.1 Empty board (game start)

**C demo's `player_prompt`** (the 51 bytes that get tokenized):

```
board=..........................................|valid=0,1,2,3,4,5,6
```

**OQL `parse_c4_board` return value:** the integer `127`
(`0b1111111` — all 7 columns legal). Nothing else.

**Token sequence fed to model in OQL:** none — the model is never
called from the behaviour, and `oql_model_propose_column()` (the
host-side proposer) ignores its `model` argument (Gap A).

### 3.2 Mid-game board (10 pieces placed)

Synthetic example matching corpus structure
(`demos/character-level/connect4/c_connect4_player.txt` ~line 50000):

**C demo's `player_prompt`:**

```
board=.......................X......O.O....X.O.X|valid=0,1,2,3,4,5,6
```

**OQL behaviour return:** `127` (all columns still legal — only
bottom rows occupied).

### 3.3 Near-terminal board with blocked column

Suppose X has tried to drop in column 2 but the column is full.
`kb.blocked = "2"`.  Board has 30 pieces:

**C demo's `player_prompt`:**

```
board=.X.X.X.|valid=0,3,4,5,6|blocked=2
```

(condensed for illustration — the real board is 42 chars.)

**OQL behaviour return:** `0b1111001` = `121` (columns 0, 3, 4, 5, 6
legal — column 2 not in legal mask because the column is full).
The "blocked" history (`opa_kanban`-style "I tried this and it
didn't work, don't try again") has no OQL representation at all —
the behaviour doesn't see it.

---

## 4. What Pathway A (behaviour-side fix) would need to do

To reproduce the C demo's protocol entirely inside the BEHAVIOUR body,
`parse_c4_board` would need to:

1. Read the board string (it already can, via interned handle on
   `ctx->current_board_handle`).
2. Construct the `board=…|valid=…[|blocked=…]` prompt.
3. **Run the loaded `Model *` over that prompt character-by-character,
   sample a digit, and return it as the proposed column.**

Step 3 is the blocker. VM TS in a BEHAVIOUR body is pure
arithmetic + extern calls; it cannot directly run model inference.
The extern table (`microgpt_vm_natives.c`) does not currently
expose any model-inference primitive — and adding one is **exactly
the Pathway B route**.

The literal "behaviour-side" reading of Pathway A — keep all the
inference inside the existing extern surface, just change TS —
**is structurally impossible** at E09's current API boundary.
The behaviour body's return is a single `number`; the runtime
interprets that number as either (a) a legal-column mask, (b) a
parsed column digit, or (c) a 0/1 validation result, depending on
which behaviour slot the body is bound to.  None of those paths
gives the body a way to feed the model the right prompt; the
model proposer (`oql_model_propose_column`) runs *after*
`INPUT_BEHAVIOUR` and is fixed C code.

**Conclusion:** the canonical Pathway A described in the pre-reg
("fix `parse_c4_board` to reproduce the corpus prompt protocol")
is not viable without an extern that calls model inference.

## 5. What Pathway B (one new extern) needs

A single new extern, name to be locked at implementation time but
matching the precedent E09 §3.4 named:

```
c4_model_propose_column(temp_x100: number): number
```

— given:
- the current board (already on `ctx->current_board_handle`),
- the loaded `Organelle *` (needs to be wired into `ctx`),
- a temperature (passed as `int = temp * 100` since the extern ABI
  is numeric-only),

— returns the proposed column index in `[0, 6]`, or `-1` on
parse/inference failure.

The behaviour `parse_c4_board` would call this extern, treat the
result as a one-hot legal mask (`1 << col` if the column is legal,
or fall back to the full legal mask if the model declines / proposes
illegally).  The runtime then picks that mask's single bit and the
move propagates through OUTPUT/VALIDATE as before.

### 5.1 Plumbing surface

To give the extern access to the `Organelle *` and `MicrogptConfig *`,
the runtime adapter (`oql_runtime_games.c`) needs to:

- After `oql_runtime_load_organelle()` returns a `Model *`, **also
  reconstruct a thin `Organelle` wrapper** (Model + Vocab built
  from the player corpus file).  This wrapper is needed because
  `organelle_generate_ensemble()` takes an `Organelle *`, not a
  bare `Model *`.  The Vocab can be rebuilt from
  `demos/character-level/connect4/c_connect4_player.txt` (the
  same corpus the checkpoint was trained on).
- Stash the wrapper into a new opaque field on `vm_natives_ctx`
  before each behaviour dispatch.
- The natives module (`microgpt_vm_natives.c`) cannot directly
  include `microgpt_organelle.h` without breaking the existing
  link layering (`microgpt_vm_lib` does not depend on
  `microgpt_lib`). Resolution: the extern dispatches through an
  opaque function pointer stored on `vm_natives_ctx`, set by the
  runtime adapter at staging time.  This keeps the +1-extern
  discipline (T6) clean: one extern, no new headers, no new link
  edges.

### 5.2 Why exactly one extern (T6 lock)

The header note in `src/microgpt_vm_natives.h` already mentions
that any "sampling-strategy object" would warrant its own
experiment.  A single `c4_model_propose_column(temp_x100)` extern
is precisely the "one-step" abstraction E09 §3.4 sketched.  More
externs (e.g. `model_get_logits`, `model_sample`, `model_argmax`)
would be the "sampling-strategy" decomposition T6 explicitly skips.

---

## 6. Decision (locked here, before implementation)

**Pathway B.** One new extern (`c4_model_propose_column`) plus the
minimum-necessary plumbing on `vm_natives_ctx` and
`oql_runtime_games.c` to expose the loaded organelle and config.

Pathway A as literally written in the pre-reg is structurally
infeasible — the diagnostic above shows the API boundary doesn't
admit it.  Sticking to the spirit of "fix in behaviour" without
acknowledging that boundary would mean reporting 51% indefinitely.
Pathway B is what E09 §3.4 itself recommended and is the only
path to a measurable closure within E11's locked scope.

**Also done at the same commit (no-cost correctness fixes):**
- Swap COMPOSE order in `experiments/connect4.oql` so
  `connect4_player` (not `connect4_planner`) is the first stage.
  This fixes Gap B.
- Update the OQL behaviour bodies accordingly.

**Constraints kept inviolate:**
- T5 (zero new VM opcodes): `git diff main -- src/microgpt_vm.{h,c,l,y}`
  will be 0 lines.
- T6 (≤ 1 new extern): exactly one (`c4_model_propose_column`).
- Engine surface (`src/microgpt.{h,c}`) frozen.
- E10's grammar / `microgpt_oql.{l,y}` untouched.
