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
ON  WITH  AGAINST  USING  WHERE  AS  AT  FROM  INTO  OF  BY
GRAPH  CORPUS  PIPELINE  ORGANELLE  MODEL  REPORT  METRIC  THRESHOLDS
SEED
```

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
```

---

## Implementation status (this commit)

| Verb       | Parses? | Executes?                                              |
|------------|---------|--------------------------------------------------------|
| `TRAIN`    | yes     | stub — emits "implementation pending" error            |
| `COMPOSE`  | yes     | stub — emits "implementation pending" error            |
| `RUN`      | yes     | stub — emits "implementation pending" error            |
| `EVALUATE` | yes     | stub — emits "implementation pending" error            |
| `VERIFY`   | yes     | `VERIFY GRAPH @graph...@end` wired to `pipeline_verify` |
| `AUDIT`    | yes     | shells out to `tools/scaling_leakage_audit.sh`         |

Parsing is the contract; execution is incremental. The follow-up commits land
each stub one at a time.
