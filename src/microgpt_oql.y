/*
 * microgpt_oql.y  —  Bison grammar for OQL (Organelle Query Language).
 *
 * Copyright (c) 2026 Ajay Soni.  MIT License.
 *
 * +6 / -4 verb surface (HARD-LOCKED). The top-level `stmt` production has
 * exactly six alternatives — adding a seventh voids Experiment E07.
 * Grammar produces an AST (see microgpt_oql.h) without side effects.
 */

%define api.prefix {oql_parser_}

%{
    #include "microgpt_oql.h"
    #include <stdio.h>
    #include <stdlib.h>
    #include <string.h>

    extern int oql_parser_lex(oql_parser *parser);
    extern void oql_parser_error(oql_parser *parser, const char *msg);

    /* AST helpers — defined in microgpt_oql.c as inlines for the grammar. */
    OqlStmt *oql_y_train(char *name, OqlSource on, OqlKV *with);
    OqlStmt *oql_y_compose(char *name, OqlNameList *from, OqlKV *with);
    OqlStmt *oql_y_run(char *name, OqlKV *with);
    OqlStmt *oql_y_evaluate(char *name, OqlSource against, char *metric, char *report);
    OqlStmt *oql_y_verify(OqlVerifySubjectKind k, char *subject, OqlPredicate *where);
    OqlStmt *oql_y_audit(OqlSource a, OqlSource b, char *thr, char *report);
    OqlKV   *oql_y_kv(char *key, char *val);
    OqlKV   *oql_y_kv_concat(OqlKV *head, OqlKV *tail);
    OqlNameList *oql_y_name(char *n);
    OqlNameList *oql_y_name_concat(OqlNameList *head, OqlNameList *tail);
    OqlPredicate *oql_y_pred(char *lhs, OqlOp op, char *rhs);
    void oql_y_append(oql_parser *p, OqlStmt *s);
%}

%parse-param { oql_parser *parser }
%lex-param   { oql_parser *parser }

%union {
    char *str;
    int   op;
    OqlKV *kv;
    OqlNameList *names;
    OqlSource src;
    OqlPredicate *pred;
    OqlStmt *stmt;
}

%token T_TRAIN T_COMPOSE T_RUN T_EVALUATE T_VERIFY T_AUDIT
%token T_ON T_WITH T_AGAINST T_USING T_WHERE T_AS T_FROM
%token T_GRAPH T_CORPUS T_REPORT T_METRIC T_THRESHOLDS
%token T_LT T_LE T_EQ T_NE T_GE T_GT
%token <str> T_IDENT T_STRING T_NUMBER T_GRAPH_BLOCK

%type <stmt>  stmt train_stmt compose_stmt run_stmt evaluate_stmt verify_stmt audit_stmt
%type <kv>    opt_with kv_list kv
%type <names> name_list
%type <src>   source opt_on
%type <pred>  opt_where predicate
%type <str>   value opt_metric opt_report opt_thresholds
%type <op>    op

%start script
%%

script : /* empty */ | script stmt ';' { oql_y_append(parser, $2); } ;

stmt
    : train_stmt    { $$ = $1; }
    | compose_stmt  { $$ = $1; }
    | run_stmt      { $$ = $1; }
    | evaluate_stmt { $$ = $1; }
    | verify_stmt   { $$ = $1; }
    | audit_stmt    { $$ = $1; }
    ;

/* ── TRAIN ───────────────────────────────────────────────────────── */
train_stmt
    : T_TRAIN T_IDENT opt_on opt_with                 { $$ = oql_y_train($2, $3, $4); }
    ;
opt_on
    : /* empty */                                     { OqlSource z = {0, NULL}; $$ = z; }
    | T_ON source                                     { $$ = $2; }
    ;

/* ── COMPOSE ─────────────────────────────────────────────────────── */
compose_stmt
    : T_COMPOSE T_IDENT T_FROM name_list opt_with     { $$ = oql_y_compose($2, $4, $5); }
    ;

/* ── RUN ─────────────────────────────────────────────────────────── */
run_stmt
    : T_RUN T_IDENT opt_with                          { $$ = oql_y_run($2, $3); }
    ;

/* ── EVALUATE ────────────────────────────────────────────────────── */
evaluate_stmt
    : T_EVALUATE T_IDENT T_AGAINST source opt_metric opt_report
      { $$ = oql_y_evaluate($2, $4, $5, $6); }
    ;
opt_metric
    : /* empty */                                     { $$ = NULL; }
    | T_USING T_METRIC T_IDENT                        { $$ = $3; }
    ;
opt_report
    : /* empty */                                     { $$ = NULL; }
    | T_REPORT T_AS T_STRING                          { $$ = $3; }
    ;

/* ── VERIFY ──────────────────────────────────────────────────────── */
verify_stmt
    : T_VERIFY T_GRAPH T_GRAPH_BLOCK opt_where        { $$ = oql_y_verify(OQL_VS_GRAPH, $3, $4); }
    | T_VERIFY T_STRING opt_where                     { $$ = oql_y_verify(OQL_VS_PATH,  $2, $3); }
    | T_VERIFY T_IDENT opt_where                      { $$ = oql_y_verify(OQL_VS_NAME,  $2, $3); }
    ;
opt_where
    : /* empty */                                     { $$ = NULL; }
    | T_WHERE predicate                               { $$ = $2; }
    ;
predicate
    : T_IDENT op value                                { $$ = oql_y_pred($1, (OqlOp)$2, $3); }
    ;
op
    : T_LT { $$ = OQL_OP_LT; } | T_LE { $$ = OQL_OP_LE; }
    | T_EQ { $$ = OQL_OP_EQ; } | T_NE { $$ = OQL_OP_NE; }
    | T_GE { $$ = OQL_OP_GE; } | T_GT { $$ = OQL_OP_GT; }
    ;

/* ── AUDIT ───────────────────────────────────────────────────────── */
audit_stmt
    : T_AUDIT source T_AGAINST source opt_thresholds opt_report
      { $$ = oql_y_audit($2, $4, $5, $6); }
    ;
opt_thresholds
    : /* empty */                                     { $$ = NULL; }
    | T_USING T_THRESHOLDS T_STRING                   { $$ = $3; }
    ;

/* ── Shared ──────────────────────────────────────────────────────── */
opt_with
    : /* empty */                                     { $$ = NULL; }
    | T_WITH kv_list                                  { $$ = $2; }
    ;
kv_list
    : kv                                              { $$ = $1; }
    | kv_list ',' kv                                  { $$ = oql_y_kv_concat($1, $3); }
    ;
kv
    : T_IDENT T_EQ value                              { $$ = oql_y_kv($1, $3); }
    ;
value
    : T_STRING { $$ = $1; } | T_NUMBER { $$ = $1; } | T_IDENT { $$ = $1; }
    ;
source
    : T_CORPUS T_STRING                               { OqlSource s = {OQL_SRC_CORPUS, $2}; $$ = s; }
    | T_STRING                                        { OqlSource s = {OQL_SRC_PATH,   $1}; $$ = s; }
    | T_IDENT                                         { OqlSource s = {OQL_SRC_NAME,   $1}; $$ = s; }
    ;
name_list
    : T_IDENT                                         { $$ = oql_y_name($1); }
    | name_list ',' T_IDENT                           { $$ = oql_y_name_concat($1, oql_y_name($3)); }
    ;
%%
