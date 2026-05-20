/*
 * microgpt_oql.h  —  OQL (Organelle Query Language) front-end (public header)
 *
 * Copyright (c) 2026 Ajay Soni.  MIT License.
 *
 * OQL is a small SQL-shaped DSL with a hard-locked +6 / -4 verb surface
 * (TRAIN, COMPOSE, RUN, EVALUATE, VERIFY, AUDIT). See
 * docs/research/OQL_GRAMMAR_REFERENCE.md for the full grammar and
 * experiments/E07-oql-dsl.md for the pre-registration.
 *
 * The header exposes:
 *   - the AST types one verb at a time (one struct per verb),
 *   - oql_parse(source) -> OqlScript* (a Flex/Bison front-end),
 *   - oql_execute(script) -> oql_status (a thin interpreter that dispatches
 *     verified verbs to existing C primitives and stubs the rest with a
 *     clear "implementation pending in follow-up commit" message).
 *
 * Construction and execution are separate: oql_parse() never executes; it
 * only builds an AST. Tests can therefore exercise the parser without any
 * side effects (no shell-out, no filesystem writes).
 */

#ifndef MICROGPT_OQL_H
#define MICROGPT_OQL_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================
 *  Status codes
 * ============================================================ */

typedef enum {
    OQL_OK = 0,
    OQL_ERR_PARSE = -1,         /* syntax error in source */
    OQL_ERR_NOT_IMPLEMENTED = -2, /* verb parsed but interpreter stub */
    OQL_ERR_RUNTIME = -3,       /* dispatch into C primitive failed */
    OQL_ERR_OOM = -4
} oql_status;

/* ============================================================
 *  Locked verb tags (+6)
 * ============================================================ */

typedef enum {
    OQL_VERB_TRAIN = 1,
    OQL_VERB_COMPOSE,
    OQL_VERB_RUN,
    OQL_VERB_EVALUATE,
    OQL_VERB_VERIFY,
    OQL_VERB_AUDIT,
    /* CREATE is inherited from SQL (not part of the +6 added verbs); it
     * carries an object-type subtag.  See experiments/E08-oql-behaviours.md
     * §1.3.2 — verb count remains +6 / -4 by reusing CREATE for BEHAVIOUR
     * and ORGANELLE object types. */
    OQL_VERB_CREATE_BEHAVIOUR,
    OQL_VERB_CREATE_ORGANELLE
} OqlVerb;

/* ============================================================
 *  Key-value list (used by WITH clauses)
 * ============================================================ */

typedef struct OqlKV {
    char *key;          /* identifier, owned */
    char *value;        /* string / number lexeme, owned */
    struct OqlKV *next;
} OqlKV;

/* Append a kv to the list (returns new head). Takes ownership of strings. */
OqlKV *oql_kv_append(OqlKV *head, char *key, char *value);
/* Look up a key; returns the value string (not owned by caller) or NULL. */
const char *oql_kv_get(const OqlKV *head, const char *key);
void oql_kv_free(OqlKV *head);

/* Name list (used by COMPOSE ... FROM a, b, c). */
typedef struct OqlNameList {
    char *name;                     /* owned */
    struct OqlNameList *next;
} OqlNameList;

OqlNameList *oql_namelist_append(OqlNameList *head, char *name);
void oql_namelist_free(OqlNameList *head);

/* ============================================================
 *  Source descriptor (for TRAIN ON, EVALUATE AGAINST, AUDIT ...)
 *
 *  A source is one of:
 *    - CORPUS '<path>'  -> kind = OQL_SRC_CORPUS, value = path
 *    - '<bare-path>'    -> kind = OQL_SRC_PATH,   value = path
 *    - <name>           -> kind = OQL_SRC_NAME,   value = name
 * ============================================================ */

typedef enum {
    OQL_SRC_CORPUS = 1,
    OQL_SRC_PATH,
    OQL_SRC_NAME
} OqlSourceKind;

typedef struct {
    OqlSourceKind kind;
    char *value;        /* owned */
} OqlSource;

void oql_source_free(OqlSource *s);

/* ============================================================
 *  Predicate (used by VERIFY ... WHERE)
 * ============================================================ */

typedef enum {
    OQL_OP_LT = 1, OQL_OP_LE, OQL_OP_EQ, OQL_OP_NE, OQL_OP_GE, OQL_OP_GT
} OqlOp;

typedef struct {
    char *lhs;          /* identifier, owned */
    OqlOp op;
    char *rhs;          /* literal lexeme, owned */
} OqlPredicate;

void oql_predicate_free(OqlPredicate *p);

/* ============================================================
 *  Per-verb statement payloads (one struct per verb — discipline visible)
 * ============================================================ */

typedef struct {
    char *target;       /* model / organelle name, owned */
    OqlSource on_src;   /* ON CORPUS ... (kind=0 if absent) */
    OqlKV  *with_kv;    /* WITH ... (NULL if absent) */
} OqlTrain;

typedef struct {
    char *target;       /* composed-pipeline name, owned */
    OqlNameList *from;  /* FROM a, b, c (required) */
    OqlKV *with_kv;     /* WITH ... (NULL if absent) */
} OqlCompose;

typedef struct {
    char *target;       /* experiment / harness name, owned */
    OqlKV *with_kv;     /* WITH ... (NULL if absent) */
} OqlRun;

typedef struct {
    char *target;       /* model / pipeline name, owned */
    OqlSource against_src; /* AGAINST ... (required) */
    char *metric;       /* USING METRIC ... (NULL if absent) */
    char *report_path;  /* REPORT AS '...'  (NULL if absent) */
} OqlEvaluate;

typedef enum {
    OQL_VS_GRAPH = 1,   /* VERIFY GRAPH @graph...@end */
    OQL_VS_PATH,        /* VERIFY '<path>' */
    OQL_VS_NAME         /* VERIFY <name> */
} OqlVerifySubjectKind;

typedef struct {
    OqlVerifySubjectKind kind;
    char *subject;      /* inline graph text, or path, or name; owned */
    OqlPredicate *where;/* optional WHERE; NULL if absent */
} OqlVerify;

typedef struct {
    OqlSource a;        /* first corpus / source (required) */
    OqlSource b;        /* AGAINST source (required) */
    char *thresholds;   /* USING THRESHOLDS '...' (NULL if absent) */
    char *report_path;  /* REPORT AS '...' (NULL if absent) */
} OqlAudit;

/* CREATE BEHAVIOUR <name> AS VM `<body>`;
 * The body is the literal VM-TypeScript source between backticks; the
 * interpreter compiles it via vm_module_compile when the statement
 * executes and stores the resulting module in the runtime context's
 * behaviour registry. */
typedef struct {
    char *name;         /* behaviour identifier, owned */
    char *vm_body;      /* TS source (unescaped), owned */
} OqlCreateBehaviour;

/* CREATE ORGANELLE <name> FROM CHECKPOINT '<path>' [WITH (kv, kv, ...)];
 * The WITH-list carries the behaviour-binding kvs (INPUT_BEHAVIOUR = name,
 * OUTPUT_BEHAVIOUR = name, ...).  Validated at execute time. */
typedef struct {
    char *name;         /* organelle identifier, owned */
    char *checkpoint;   /* path string, owned (NULL allowed for in-memory) */
    OqlKV *bindings;    /* OqlKV list of *_BEHAVIOUR=name and friends */
} OqlCreateOrganelle;

/* ============================================================
 *  Statement union
 * ============================================================ */

typedef struct OqlStmt {
    OqlVerb verb;
    union {
        OqlTrain    train;
        OqlCompose  compose;
        OqlRun      run;
        OqlEvaluate evaluate;
        OqlVerify   verify;
        OqlAudit    audit;
        OqlCreateBehaviour  create_behaviour;
        OqlCreateOrganelle  create_organelle;
    } u;
    struct OqlStmt *next;
} OqlStmt;

/* ============================================================
 *  Script (top-level container)
 * ============================================================ */

typedef struct OqlScript {
    OqlStmt *head;      /* linked list of statements in source order */
    char *error;        /* parse error message (NULL on success); owned */
    int   error_line;
} OqlScript;

OqlScript *oql_script_create(void);
void       oql_script_free(OqlScript *s);
size_t     oql_script_count(const OqlScript *s);

/* ============================================================
 *  Parser API
 * ============================================================ */

/* Parse OQL source text into an AST. Returns a NEW OqlScript (caller owns).
 * On parse failure, the script's `error` field is populated and `head` is
 * NULL. Never returns NULL itself (allocation failure aside). */
OqlScript *oql_parse(const char *source);

/* ============================================================
 *  Interpreter API
 *
 *  oql_execute() walks the AST in source order and dispatches each verb to
 *  its handler:
 *     VERIFY GRAPH ...  -> microgpt_pipeline (pipeline_parse_text + verify)
 *     AUDIT a AGAINST b -> shell-out to tools/scaling_leakage_audit.sh
 *     TRAIN/COMPOSE/RUN/EVALUATE -> stub returning OQL_ERR_NOT_IMPLEMENTED
 *
 *  On first failing statement, execution halts and the index of the failing
 *  statement is written to *failed_idx (1-based; 0 if success).
 *  Caller-supplied `out` is used for human-readable progress / error
 *  messages (may be NULL to suppress).
 * ============================================================ */

#include <stdio.h>

oql_status oql_execute(const OqlScript *script, FILE *out, int *failed_idx);

/* ============================================================
 *  Runtime registries (E09 — Phase 1 / Phase 2)
 *
 *  When `oql_execute` runs a script that contains CREATE ORGANELLE,
 *  CREATE BEHAVIOUR, or COMPOSE statements, those statements populate a
 *  runtime context — a small in-memory registry of organelles, behaviours,
 *  and pipelines that subsequent statements (notably RUN) can resolve
 *  against.
 *
 *  Organelle storage is lazy: CREATE ORGANELLE only records the checkpoint
 *  path and behaviour bindings. The first RUN that references the
 *  organelle triggers checkpoint_load().
 *
 *  Behaviours are stored as raw VM body strings (not compiled until first
 *  dispatch). The VM compile step is deferred to allow E09's runtime to
 *  link against microgpt_vm_lib while keeping pure-parsing tests free of
 *  the VM dependency.
 *
 *  Pipeline storage holds a parsed Pipeline IR per COMPOSE; call-node
 *  resolution against the organelle table happens at COMPOSE time.
 * ============================================================ */

#define OQL_MAX_ORGANELLES 16
#define OQL_MAX_BEHAVIOURS 64
#define OQL_MAX_PIPELINES  16
#define OQL_MAX_PIPELINE_CALLS 32

/* Forward-declared so the public header doesn't pull in microgpt.h or
 * pipeline_ir.h. `Model` and `Pipeline` are tagged structs in their own
 * headers (typedef'd to themselves), so we use the struct tag form here
 * to avoid C11 typedef-redefinition diagnostics when the consumer
 * includes both microgpt_oql.h and microgpt.h / pipeline_ir.h.
 * `MicrogptConfig` and `vm_module` are typedefs of anonymous structs;
 * we expose them via opaque `void *` and cast at the implementation
 * boundary. */
struct Model;
struct Pipeline;

/* A registered BEHAVIOUR. Body is owned; compiled module is lazily built
 * on first dispatch. `module` is an opaque pointer to a `vm_module`
 * (typedef of an anonymous struct in microgpt_vm.h). */
typedef struct OqlBehaviourEntry {
    char       *name;        /* identifier, owned */
    char       *vm_body;     /* TS source, owned */
    void       *module;      /* lazily compiled vm_module*; freed by oql_runtime_dispose */
} OqlBehaviourEntry;

/* A registered ORGANELLE with lazy checkpoint load. */
typedef struct OqlOrganelle {
    char  name[64];
    char  checkpoint_path[256];
    int   loaded;             /* lazy: load on first RUN reference */
    struct Model *model;      /* checkpoint-loaded; freed by oql_runtime_dispose */
    /* Behaviour bindings (names; resolved against behaviour table at RUN time). */
    char  input_behaviour[64];
    char  output_behaviour[64];
    char  validate_behaviour[64];
    char  fallback_behaviour[64];
    char  score_behaviour[64];
    char  cycle_detect_behaviour[64];
} OqlOrganelle;

/* A registered COMPOSE pipeline.  `ir` is a parsed Pipeline IR (or NULL
 * if COMPOSE's FROM clause didn't include an inline @graph block — in
 * that case `call_organelles` lists the named organelles in source
 * order so RUN can sequence them as a linear chain). */
typedef struct OqlPipeline {
    char      name[64];
    struct Pipeline *ir;          /* parsed/verified pipeline IR, NULL if linear */
    /* Linear-chain fallback: ordered list of organelle names (for the
     * FROM a, b, c style without an inline @graph). */
    char      call_organelles[OQL_MAX_PIPELINE_CALLS][64];
    int       n_calls;
} OqlPipeline;

typedef struct OqlRuntime {
    OqlBehaviourEntry behaviours[OQL_MAX_BEHAVIOURS];
    int               n_behaviours;
    OqlOrganelle      organelles[OQL_MAX_ORGANELLES];
    int               n_organelles;
    OqlPipeline       pipelines[OQL_MAX_PIPELINES];
    int               n_pipelines;
    /* Configuration used when calling checkpoint_load (opaque pointer to
     * MicrogptConfig).  If NULL, the runtime falls back to a default
     * config built from the compile-time macros at OQL-lib build time. */
    const void *cfg;
    /* Metrics accumulated by oql_run_game_loop. */
    int    last_games_played;
    int    last_wins;
    int    last_draws;
    int    last_losses;
    double last_p99_ms;
    double last_total_seconds;
    int    last_audit_rows;
} OqlRuntime;

/* Lifecycle. */
void oql_runtime_init(OqlRuntime *rt);
void oql_runtime_dispose(OqlRuntime *rt);

/* Lookup helpers (returns NULL if not found). */
OqlBehaviourEntry *oql_runtime_find_behaviour(OqlRuntime *rt, const char *name);
OqlOrganelle      *oql_runtime_find_organelle(OqlRuntime *rt, const char *name);
OqlPipeline       *oql_runtime_find_pipeline (OqlRuntime *rt, const char *name);

/* Execute a script against a runtime context.  Identical to oql_execute,
 * except CREATE BEHAVIOUR / CREATE ORGANELLE / COMPOSE register into the
 * runtime, and RUN can resolve organelle / pipeline / behaviour names. */
oql_status oql_execute_with_runtime(const OqlScript *script,
                                    OqlRuntime *rt,
                                    FILE *out, int *failed_idx);

/* Lazy checkpoint-load helper.  Idempotent: returns the already-loaded
 * model if `organelle->loaded` is non-zero. Returns NULL on load failure. */
struct Model *oql_runtime_load_organelle(OqlRuntime *rt, OqlOrganelle *organelle,
                                         FILE *out);

/* ============================================================
 *  Parser-internal handle (exposed only for the Flex/Bison plumbing).
 *  Production code should not touch this directly.
 * ============================================================ */

typedef struct oql_parser {
    const char *source;
    size_t source_len;
    size_t source_index;
    int    line_number;
    OqlScript *script;
} oql_parser;

/* Lexer's character-fetch callback (called by Flex's YY_INPUT). */
int oql_parser_char_fetch_next(oql_parser *p);

/* Bison error callback. */
void oql_parser_error(oql_parser *parser, const char *msg);

#ifdef __cplusplus
}
#endif

#endif /* MICROGPT_OQL_H */
