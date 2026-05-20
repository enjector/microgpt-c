# Experiment E08 — VM TypeScript dialect as the body of OQL `BEHAVIOUR` objects: a high-level researcher surface for organelle wrappers

**Status:** 📋 Proposal locked — 2026-05-20.
**Direction:** complete the E07 OQL surface by lifting per-demo C wrappers into first-class `BEHAVIOUR` objects whose body is the existing VM TypeScript dialect. Researchers stop writing C; they write TS functions that bind to engine primitives via `declare function` externs.
**Cost estimate:** ~4-5 weeks (1 wk extern-table extension + 1 wk OQL `CREATE BEHAVIOUR` integration + 1 wk Connect-4 worked example + 1 wk replication across 3 more games + 1 wk measurement and writeup).
**Falsification risk:** Medium — depends on whether the existing VM dialect's expressivity covers the wrapper concerns across all 11 games without forcing new opcodes.

---

## Spear summary

**Point:** Every game demo today is 200-800 lines of bespoke C that does the same six things: parse the board, format the move, validate legality, fall back when the model is stuck, score candidates, detect cycles. That is *exactly* what the VM was built for — and the VM speaks **TypeScript**, not stack-based bytecode. Lift those concerns into named OQL `BEHAVIOUR` objects whose body is TS, bind them to organelles via `CREATE ORGANELLE WITH (..._BEHAVIOUR = ...)`, and the per-demo authorship cost collapses to ≤ 30% of today.

**Picture:** Today, `demos/character-level/connect4/main.c` is ~500 lines of C. Tomorrow: `experiments/connect4.oql` is ~30 lines of OQL + 4 small TS behaviour bodies (10-20 lines each). Same engine, same win rate (88% baseline), same audit trail — but a researcher who knows TypeScript can author a new game in an afternoon instead of a week.

**Proof (to be measured):** Connect-4 win rate ≥ 85% (vs the 88% C-coded baseline; locked floor 80%); OQL+TS line count ≤ 30% of the original C demo; **zero new VM opcodes** (extern table extensions only); a first-time researcher writes a new game's `INPUT_BEHAVIOUR` in ≤ 30 min on a stopwatch.

**Push:** This is the move that turns OQL from "a parser with two wired verbs" (E07's honest stopping point) into "the language researchers actually author experiments in." Without it, E07 is a finished grammar with nothing to drive; with it, E07 becomes the substrate.

---

## 1. Proposal

### 1.1 Hypothesis (locked before measurement)

> *A `BEHAVIOUR` object in OQL with a body in the existing VM TypeScript dialect can encapsulate every organelle wrapper concern currently hand-coded in C across the 11 game demos, with: (a) the demo's win/solve rate held within ±3 pp of the C-coded baseline (Connect-4: ≥ 85%, floor 80%); (b) total OQL + TS line count ≤ 30% of the original C demo; (c) **zero new VM opcodes** added — only extern table extensions; (d) all existing `test_microgpt_vm` tests continue to pass; (e) a new-to-the-project researcher writes their first `INPUT_BEHAVIOUR` for a new game in ≤ 30 min.*

### 1.2 Why this matters

E07 shipped the OQL grammar + parser + two wired verbs (`VERIFY`, `AUDIT`) and explicitly stubbed `TRAIN` / `COMPOSE` / `RUN` / `EVALUATE` with `OQL_ERR_NOT_IMPLEMENTED` until the binding surface to organelles existed. **That binding surface is `BEHAVIOUR`**. Without it, OQL is a parse-only DSL with nothing to drive; the four stubbed verbs cannot be wired because they have no behaviour layer to invoke.

There's also a deeper claim: the project's narrative leads with *"deterministic C scaffolding (~340 LOC) does what gradient descent can't"* (per `CLAUDE.md`'s Organelle Pipeline section). Today that scaffolding is **hidden in 11 separate C demos**. Lifting it into named, queryable, versioned `BEHAVIOUR` objects makes the architecture's most distinctive claim *literally inspectable* — you can `SELECT name, lines, used_by FROM behaviours` and read the deterministic substrate that the narrative says is the source of intelligence.

For external adoption, this is the difference between **"a research repo with documented experiments"** and **"a research substrate that other people can author experiments on without learning C99."**

### 1.3 Mechanism

#### 1.3.1 The six wrapper concerns

A walk of `demos/character-level/{connect4,othello,pentago,hex,tictactoe,mastermind,sudoku,lightsout,klotski,reddonkey,puzzle8}/main.c` shows every demo implements some subset of these six concerns:

| Wrapper concern | What it does today (in C) | TS extern dependencies |
|---|---|---|
| **`INPUT_BEHAVIOUR`** | Parse a board / prompt string into the model's input tokens | `tokenize`, `model_block_size` |
| **`OUTPUT_BEHAVIOUR`** | Format the model's next-token output into a domain move | `model_next_token`, `top_k_logits` |
| **`VALIDATE_BEHAVIOUR`** | Test whether the proposed move is legal in the current state | (game-specific; pure-TS predicate) |
| **`FALLBACK_BEHAVIOUR`** | Pick a fallback move when the model is stuck (high entropy or repeat) | `last_entropy`, `legal_moves`, `rand_pick` |
| **`SCORE_BEHAVIOUR`** | Rank multiple candidates from the model (best-of-N) | `verify`, `pipeline_repair` (already extern-able via E02's `libpipeline_ir`) |
| **`CYCLE_DETECT_BEHAVIOUR`** | Detect A↔B oscillation across recent moves | `kanban_history`, `move_equals` |

These six concerns are stable across all 11 demos. A `BEHAVIOUR` registry that holds them once and binds them to organelles via OQL collapses the per-demo cost.

#### 1.3.2 Worked example — Connect-4 in OQL + TS

```sql
-- experiments/connect4.oql

CREATE BEHAVIOUR parse_c4_board AS VM `
    declare function split(s: string, sep: string): string[];

    function eval(board: string): number[] {
        var rows = split(board, " | ");
        var cells: number[] = [];
        var r = 0;
        while (r < rows.length) {
            var row_cells = split(rows[r], "");
            var c = 0;
            while (c < row_cells.length) {
                if (row_cells[c] == "X") cells.push(1);
                else if (row_cells[c] == "O") cells.push(-1);
                else cells.push(0);
                c = c + 1;
            }
            r = r + 1;
        }
        return cells;
    }
`;

CREATE BEHAVIOUR format_c4_move AS VM `
    declare function model_next_token(): string;

    function eval(): string {
        var tok = model_next_token();
        -- expect a single digit 0-6
        return tok;
    }
`;

CREATE BEHAVIOUR c4_move_is_legal AS VM `
    declare function legal_columns(board: string): number[];
    declare function parseInt(s: string): number;

    function eval(board: string, move: string): boolean {
        var col = parseInt(move);
        if (col < 0) return false;
        if (col > 6) return false;
        var cols = legal_columns(board);
        var i = 0;
        while (i < cols.length) {
            if (cols[i] == col) return true;
            i = i + 1;
        }
        return false;
    }
`;

CREATE BEHAVIOUR c4_fallback_when_stuck AS VM `
    declare function legal_columns(board: string): number[];
    declare function last_entropy(): number;

    function eval(board: string): string {
        if (last_entropy() > 0.8) {
            var cols = legal_columns(board);
            -- prefer centre column 3 if legal
            var i = 0;
            while (i < cols.length) {
                if (cols[i] == 3) return "3";
                i = i + 1;
            }
            -- else pick lowest legal column
            if (cols.length > 0) {
                return "" + cols[0];
            }
        }
        return "";  -- defer to model
    }
`;

CREATE ORGANELLE connect4_player
  FROM CHECKPOINT 'checkpoints/c4_player.ckpt'
  WITH (
    INPUT_BEHAVIOUR    = parse_c4_board,
    OUTPUT_BEHAVIOUR   = format_c4_move,
    VALIDATE_BEHAVIOUR = c4_move_is_legal,
    FALLBACK_BEHAVIOUR = c4_fallback_when_stuck
  );

CREATE ORGANELLE connect4_planner
  FROM CHECKPOINT 'checkpoints/c4_planner.ckpt'
  WITH (
    INPUT_BEHAVIOUR  = parse_c4_board,
    OUTPUT_BEHAVIOUR = format_c4_move
  );

CREATE PIPELINE connect4 AS
  COMPOSE @graph
    in       = read_board()
    planned  = call(connect4_planner, in)
    move     = call(connect4_player, planned)
    out      = move
  @end;

RUN connect4 ON game_loop
  WITH (opponent = random, games = 100)
  RETURNING (win_rate, p99_latency_ms, audit_coverage);
```

That's ~80 lines of OQL + TS replacing ~500 lines of C. The four behaviours are ~10-20 lines each; each can be reused across Connect-4, Othello, Hex, Pentago wherever board parsing or fallback semantics are shared.

#### 1.3.3 Implementation phases

| Phase | Work | Cost |
|---|---|---|
| **1. Catalogue** | Read all 11 demo main.c files; categorise every wrapper function under the six concerns; identify the union of engine primitives the behaviours need to call. Output: `BEHAVIOUR_CATALOGUE.md` enumerating the extern table to be built. | 1 wk |
| **2. VM extern table extension** | Add the catalogue's primitives to `src/microgpt_vm.{h,c}` as `declare function`-callable externs. **Zero new opcodes** — only new C functions registered in the dispatch table. Each new extern gated by an existing `test_microgpt_vm` case. | 1 wk |
| **3. OQL `CREATE BEHAVIOUR` + `WITH (..._BEHAVIOUR = ...)` integration** | Extend `src/microgpt_oql.{l,y}` grammar with `BEHAVIOUR` object type; the `AS VM \`...\`` body string is compiled in-line by calling the existing VM parser. `CREATE ORGANELLE` gains the `WITH (..._BEHAVIOUR = ...)` clause. Behaviour bindings stored in the OQL runtime context. | 1 wk |
| **4. Connect-4 worked example** | Rewrite `demos/character-level/connect4/main.c` as `experiments/connect4.oql` (per §1.3.2). Run 100 evaluation games vs random opponent. Compare win rate to baseline 88%. | 1 wk |
| **5. Replication** | If Phase 4 holds, rewrite 3 more demos: Mastermind (78%), Pentago (91%), 8-puzzle (90%). Each tests a different wrapper-concern emphasis (Mastermind: deduction; Pentago: large board; 8-puzzle: search). | 1 wk |
| **6. Section 3 writeup** | Fill in measurements per T1-T6; document any extern table gaps; document the per-game LOC reduction. | embedded |

### 1.4 Pre-registered targets (locked)

| ID | Target | Floor (skip-rule trigger) |
|---|---|---|
| **T1** | Connect-4 win rate via OQL+TS ≥ 85% (vs 88% C baseline) | < 80% |
| **T2** | OQL + TS total line count for the Connect-4 rewrite ≤ 30% of original `connect4/main.c` | > 50% |
| **T3** | **Zero new VM opcodes added.** Only extern table extensions to `microgpt_vm.{h,c}`. | Any new opcode added |
| **T4** | All existing `test_microgpt_vm` tests pass unchanged | Any regression |
| **T5** | Replication: 3 more demos rewritten (Mastermind, Pentago, 8-puzzle), each holding within ±3 pp of C-baseline win/solve rate | Any of the three drops > 10 pp |
| **T6** | A researcher new to the project writes a new game's `INPUT_BEHAVIOUR` in ≤ 30 min (measured on ≥ 2 people) | Median > 60 min |
| **T7** | Audit-trace coverage: every behaviour invocation logged with input/output, behaviour name, and execution time; readable via `SELECT * FROM behaviour_invocations WHERE pipeline = 'connect4'` | < 100% (= telemetry bug) |
| **T8** | VM dispatch overhead: per-move behaviour-execution time ≤ 1 ms p99 on M2 Max | > 10 ms |

The headline survives if **T1, T2, T3, T4 all pass** AND **T5 passes for at least 2 of 3 replications**. T6 and T8 are usability/performance backstops.

### 1.5 Skip rules

- **If T3 trips** (a new VM opcode genuinely needed): STOP and document the gap in Section 3. Adding an opcode is a deliberate VM-extension proposal that warrants its own pre-reg. Do not silently expand the VM ISA.
- **If T1 falls below 80%** (Connect-4 baseline regression): the abstraction is too lossy for game-playing wrappers. Document; do not retroactively weaken the floor.
- **If T2 exceeds 50% LOC ratio**: the OQL+TS surface is not actually cheaper than C for this concern class. Falsifies the headline benefit — document honestly.
- **If T5 trips on 2 of 3 replications**: the Connect-4 result didn't generalise; the wrapper-concern categorisation was too narrow. Document the per-game divergence; do not paper over.

### 1.6 Falsification risk: Medium

| Risk | Likelihood | Mitigation |
|---|---|---|
| Some wrapper concerns need imperative features the VM TS dialect doesn't support (closures, exceptions, regex) | Medium | Phase 1's catalogue surfaces this *before* any code is written; if a feature is genuinely needed, T3 trips and the experiment honestly stops |
| VM dispatch overhead too high on the per-move hot path | Low | `bench_microgpt_vm` shows 3.7-5.8M ops/sec; organelle wrappers do ~10-50 ops per move; budget is well-met |
| The 6-pass VM verifier rejects valid behaviour bodies | Low-medium | Existing `tests/resources/tests/vm/` is the test corpus; extend it before extending the externs |
| Win-rate loss from indirection through VM (extra latency on the fallback path) | Low | Fallback only fires when entropy > threshold; small percentage of moves |
| Researchers find TS-via-backtick-string awkward inside OQL files | Medium | Consider syntactic alternative: `CREATE BEHAVIOUR foo FROM FILE 'foo.behaviour.ts'` — defer to Section 3 if friction shows up |

### 1.7 What this experiment is NOT testing

- It is **not** testing whether OQL replaces C99 in the engine. The engine stays C99; behaviours wrap organelles, not the engine itself.
- It is **not** testing whether the VM should become Turing-complete. It already is; TS-with-functions-and-loops is plenty for wrapper concerns.
- It is **not** testing whether the architecture's intelligence claim survives the rewrite. The claim is "coordination is the intelligence"; the coordination (`OpaKanban`, cycle detection, planner→player→judge) is preserved structurally in the pipeline graph. Behaviours are the *interface layer*, not the coordination layer.
- It is **not** competing with `BPF` / `eBPF` / `Lua` / other embedded scripting choices. The VM exists; it speaks TS; reusing it is the cheapest move.

### 1.8 Cross-references

| Topic | Source |
|---|---|
| Parent experiment (OQL grammar + parser) | [E07](E07-oql-dsl.md) |
| The VM's TypeScript dialect being reused | `src/microgpt_vm.{l,y,h,c}` + `tests/resources/tests/vm/{function,declare_function,compiler}/*.ts` |
| Why imperative bodies belong in the VM, not in OQL | [E07](E07-oql-dsl.md) §1.3.1 (the `-CREATE FUNCTION` omission rationale) |
| Pipeline IR library that behaviours can call | [E02](E02-pipeline-ir-library.md) |
| Calibrated win/solve rates being preserved | [`docs/research/RESEARCH_ORGANELLE_GAMES.md`](../docs/research/RESEARCH_ORGANELLE_GAMES.md) |
| C demos being targeted for rewrite | `demos/character-level/{connect4,othello,pentago,mastermind,puzzle8}/main.c` |
| Project's wrapper-as-intelligence-substrate claim | `CLAUDE.md` "Organelle pipeline pattern"; `docs/research/RESEARCH_INTELLIGENCE.md` |

---

## 2. Initial state

### 2.1 What's currently known

- The VM today (`src/microgpt_vm.{l,y,h,c}` + pre-generated parsers + `tests/resources/tests/vm/`) parses a TypeScript subset: typed `function`s with `number` / `boolean` / `string` / array / object return types; `var` declarations; `if`/`else`; `while`; object literals and arrays (including nested); `declare function` externs.
- VM throughput: 3.7-5.8M ops/sec single-threaded per `bench_microgpt_vm`.
- 6-pass bytecode verifier catches stack underruns, type mismatches, branch validity, etc.
- 11 game demos each hand-code the same six wrapper concerns in C; no shared library exists.
- E07 OQL parser ships with `CREATE` as a first-class verb (E07 §1.3); extending it with a `BEHAVIOUR` object type is *zero new verbs*, just a new keyword in the existing `CREATE` production.
- Behaviours from the 11 demos that look obviously shareable across games: `parse_grid_board`, `random_valid_move`, `legal_moves_filter`, `entropy_gated_fallback`, `cycle_aware_move_history`.

### 2.2 Baselines to beat

| Baseline | Number | OQL+TS must |
|---|---|---|
| Connect-4 win rate (C demo) | 88% | hold within ±3 pp; floor 80% |
| Connect-4 source-line cost | ~500 LOC C | ≤ 150 LOC (OQL + TS) |
| Per-game authorship time (new game, by someone fluent in C) | ~1 week | ~1 afternoon by someone fluent in TS |
| Wrapper-concern reuse across demos | 0% (each demo reimplements) | ≥ 60% of behaviours reusable across ≥ 2 games |

### 2.3 Dependencies / blockers

- **E07 already merged** ✅ — the OQL parser exists.
- **Existing VM** ✅ — parser + bytecode + runtime + verifier all present.
- **Engine primitives the externs will need** — most already exist as C symbols (`microgpt_forward_inference`, `microgpt_sample_token`, ...); some need a small adapter layer to fit the VM's extern ABI. Catalogue them in Phase 1.
- **Connect-4 baseline reproducibility** — confirm the 88% win rate still holds on current main before starting Phase 4. Re-run the existing C demo with the published RNG seed.

### 2.4 What this experiment deliberately does NOT do

- It does NOT migrate the engine to TypeScript. The engine stays C99; behaviours are a TS *surface*.
- It does NOT add OQL verbs. The +6 / -4 verb lock holds (`CREATE BEHAVIOUR` reuses the existing `CREATE`).
- It does NOT introduce a new build dependency. The VM + OQL Flex/Bison fallback already covers TS compilation.
- It does NOT touch the trained organelle checkpoints. Same weights; new wrapper layer.
- It does NOT redesign the VM's TS dialect. Extending the *extern table* is enough; if it isn't, T3 trips and we stop.

---

## 3. Implementation + results

**TODO** — fill on measurement commit. Sections to populate:

- 3.1 Wrapper concern catalogue across 11 demos (`BEHAVIOUR_CATALOGUE.md`)
- 3.2 Extern table extension: which engine primitives became `declare function` externs, with VM test coverage per addition
- 3.3 OQL grammar extension: `CREATE BEHAVIOUR` + `WITH (..._BEHAVIOUR = ...)` patches to `microgpt_oql.{l,y}`
- 3.4 Connect-4 rewrite: `experiments/connect4.oql` + behaviour files; LOC ratio for T2; win rate for T1
- 3.5 Replication: Mastermind, Pentago, 8-puzzle rewrites and their measured rates for T5
- 3.6 Researcher onboarding measurement for T6 (≥ 2 people, recorded times)
- 3.7 Audit trace artefacts demonstrating T7 coverage
- 3.8 VM dispatch overhead measurement for T8
- 3.9 Opcode-extension log — confirming T3's zero-new-opcodes lock (or documenting any necessary additions and the consequent skip-rule activation)

---

## 4. Conclusion

**TODO** — fill on measurement commit. Sections to populate:

- 4.1 Verdict per T1-T8 (PASS / FAIL / FLOOR-TRIGGER)
- 4.2 Headline outcome — does the high-level researcher surface earn its cost?
- 4.3 Wrapper-concern reuse analysis — how much of the cross-game library actually got shared?
- 4.4 Extern table assessment — which additions were obvious, which were forced
- 4.5 If T3 trips: what opcode would have closed the gap, and is it worth a separate pre-reg?
- 4.6 Next moves: extend across remaining 7 games; publish `experiments/connect4.oql` as the canonical worked-example; consider standalone behaviour catalogue as a research artefact
- 4.7 Traceability updates (`TRACEABILITY.md`, `ORGANELLE_STATE.md`, `RESEARCH_DISCLOSURE.md`)
