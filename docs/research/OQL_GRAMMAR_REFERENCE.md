# OQL — Grammar Reference (one page)

A SQL-shaped DSL for organelles, pipelines, behaviours, and pre-registered
experiments. Six verbs, deliberately. Inspired by EQL.

See `experiments/E07-oql-dsl.md` for the pre-registration + targets, and
`src/microgpt_oql.{l,y,h,c}` for the Flex/Bison implementation.

---

## Verb surface (HARD-LOCKED — see E07 §1.3.1)

**Added (+6):** `TRAIN`, `COMPOSE`, `RUN`, `EVALUATE`, `VERIFY`, `AUDIT`.

**Omitted (-4):** `CREATE TRIGGER`, `CREATE FUNCTION`, `DECLARE CURSOR`, `SAVEPOINT`.

A 7th verb voids the experiment. Adding a verb means dropping one.

---

## Keywords (reserved)

```
TRAIN  COMPOSE  RUN  EVALUATE  VERIFY  AUDIT
CREATE  BEHAVIOUR  ORGANELLE  CHECKPOINT  VM       (Experiment E08)
ON  WITH  AGAINST  USING  WHERE  AS  AT  FROM  INTO  OF  BY
GRAPH  CORPUS  PIPELINE  MODEL  REPORT  METRIC  THRESHOLDS
SEED
```

`CREATE`, `BEHAVIOUR`, `ORGANELLE`, `CHECKPOINT`, `VM` are added by E08 to
support `CREATE BEHAVIOUR ... AS VM` and `CREATE ORGANELLE ... WITH (...)`.
None of these tokens are listed in the +6 verb surface — `CREATE` is
inherited from SQL (not added), and the object-type tokens are sub-keywords
of the `CREATE` production.

Identifiers are `[A-Za-z_][A-Za-z0-9_]*`. Strings are single-quoted with
backslash escapes. Numbers are decimal integers or floats. Inline graph blocks
use the pipeline IR's existing `@graph ... @end` form (consumed as an opaque
string token by OQL and parsed by `pipeline_parse_text` at execution time).
Comments start with `--` and run to end of line.

---

## Productions (informal EBNF, every production once)

```
script        ::= ( stmt ';' )*

stmt          ::= train_stmt
              |   compose_stmt
              |   run_stmt
              |   evaluate_stmt
              |   verify_stmt
              |   audit_stmt
              |   create_stmt                       -- E08

train_stmt    ::= 'TRAIN' name
                  ( 'ON' source )?
                  ( 'WITH' kv_list )?

compose_stmt  ::= 'COMPOSE' name
                  'FROM' name_list
                  ( 'WITH' kv_list )?

run_stmt      ::= 'RUN' name
                  ( 'WITH' kv_list )?

evaluate_stmt ::= 'EVALUATE' name
                  'AGAINST' source
                  ( 'USING' 'METRIC' name )?
                  ( 'REPORT' 'AS' STRING )?

verify_stmt   ::= 'VERIFY' verify_subject
                  ( 'WHERE' predicate )?

verify_subject::= 'GRAPH' graph_block
              |   STRING                   -- e.g. path to a report or pipeline file
              |   name                     -- a named pipeline registered at runtime

audit_stmt    ::= 'AUDIT' source
                  'AGAINST' source
                  ( 'USING' 'THRESHOLDS' STRING )?
                  ( 'REPORT' 'AS' STRING )?

source        ::= 'CORPUS' STRING          -- file path
              |   STRING                   -- bare path
              |   name                     -- named artefact

name_list     ::= name ( ',' name )*
kv_list       ::= kv ( ',' kv )*
kv            ::= name '=' value
value         ::= STRING | NUMBER | name

predicate     ::= name op value
op            ::= '<' | '<=' | '=' | '!=' | '>=' | '>'

graph_block   ::= '@graph' ... '@end'      -- opaque, parsed by pipeline_parse_text

-- E08 additions: CREATE BEHAVIOUR / CREATE ORGANELLE
create_stmt   ::= create_behaviour_stmt
              |   create_organelle_stmt

create_behaviour_stmt
              ::= 'CREATE' 'BEHAVIOUR' name 'AS' 'VM' vm_body
vm_body       ::= '`' ... '`'              -- backtick-delimited TS source,
                                           -- parsed at execute time by the
                                           -- existing microgpt_vm parser

create_organelle_stmt
              ::= 'CREATE' 'ORGANELLE' name
                  ( 'FROM' 'CHECKPOINT' STRING )?
                  ( 'WITH' '(' binding_list ')' )?
binding_list  ::= binding ( ',' binding )*
binding       ::= name '=' name            -- e.g. INPUT_BEHAVIOUR = parse_c4
```

---

## Examples — one per verb

```sql
-- TRAIN: fit a wiring organelle
TRAIN wiring_v13
  ON CORPUS 'pipeline_corpus_phase13.txt'
  WITH STEPS = 2000, LR = 0.001, N_LAYER = 6;

-- COMPOSE: wire two organelles into a planner→player chain
COMPOSE planner_player FROM planner_v1, player_v3
  WITH MEMORY = 'kanban';

-- RUN: execute a named experiment with a seed
RUN E01_baseline WITH SEED = 42;

-- EVALUATE: score a model against a held-out corpus
EVALUATE wiring_v13
  AGAINST 'pipeline_corpus_held_out.txt'
  USING METRIC fidelity
  REPORT AS 'reports/E01_baseline.json';

-- VERIFY: verify an inline pipeline IR graph
VERIFY GRAPH @graph demo
| s = stage_a()
| t = stage_b(s.out)
@end;

-- AUDIT: run a leakage audit between two corpora
AUDIT 'pipeline_corpus_phase13.txt'
  AGAINST 'pipeline_corpus_held_out.txt'
  USING THRESHOLDS 'tools/leakage_audit_thresholds.json'
  REPORT AS 'reports/E01_leakage.json';

-- CREATE BEHAVIOUR: define an OQL BEHAVIOUR with a VM-coded body (E08)
CREATE BEHAVIOUR parse_c4_board AS VM `
    declare function c4_legal_column_mask(): number;
    function eval(): number {
        var m = c4_legal_column_mask();
        return m;
    }
`;

-- CREATE ORGANELLE: instantiate an organelle and bind behaviours (E08)
CREATE ORGANELLE connect4_player
  FROM CHECKPOINT 'checkpoints/c4_player.ckpt'
  WITH (
    INPUT_BEHAVIOUR    = parse_c4_board,
    OUTPUT_BEHAVIOUR   = format_c4_move,
    VALIDATE_BEHAVIOUR = c4_move_is_legal,
    FALLBACK_BEHAVIOUR = c4_fallback_when_stuck
  );
```

---

## Implementation status (this commit)

| Verb               | Parses? | Executes?                                              |
|--------------------|---------|--------------------------------------------------------|
| `TRAIN`            | yes     | stub — emits "implementation pending" error            |
| `COMPOSE`          | yes     | stub — emits "implementation pending" error            |
| `RUN`              | yes     | stub — emits "implementation pending" error            |
| `EVALUATE`         | yes     | stub — emits "implementation pending" error            |
| `VERIFY`           | yes     | `VERIFY GRAPH @graph...@end` wired to `pipeline_verify` |
| `AUDIT`            | yes     | shells out to `tools/scaling_leakage_audit.sh`         |
| `CREATE BEHAVIOUR` | yes     | parse-only — VM-compile is harness-driven (E08 Phase 4) |
| `CREATE ORGANELLE` | yes     | parse-only — binding-resolution is harness-driven       |

Parsing is the contract; execution is incremental. The follow-up commits land
each stub one at a time.
