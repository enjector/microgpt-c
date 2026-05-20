/*
 * microgpt_oql.c  —  OQL (Organelle Query Language) implementation.
 *
 * Copyright (c) 2026 Ajay Soni.  MIT License.
 *
 * Contains:
 *   - AST allocator/free helpers (OqlKV, OqlNameList, OqlSource,
 *     OqlPredicate, OqlStmt, OqlScript),
 *   - Flex/Bison driver (oql_parse),
 *   - Interpreter (oql_execute) — VERIFY GRAPH is wired to pipeline_verify;
 *     AUDIT shells out to tools/scaling_leakage_audit.sh; the four
 *     remaining verbs return OQL_ERR_NOT_IMPLEMENTED with a clear message.
 */

#include "microgpt_oql.h"
#include "microgpt_pipeline.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ============================================================
 *  Bison action helpers — declared in microgpt_oql.y, defined here so the
 *  grammar stays under T1's 200-LOC budget. Each is a thin AST constructor.
 * ============================================================ */

static OqlStmt *oql_stmt_alloc(OqlVerb v) {
    OqlStmt *s = (OqlStmt *)calloc(1, sizeof(OqlStmt));
    if (s) s->verb = v;
    return s;
}

OqlStmt *oql_y_train(char *name, OqlSource on, OqlKV *with) {
    OqlStmt *s = oql_stmt_alloc(OQL_VERB_TRAIN);
    s->u.train.target = name; s->u.train.on_src = on; s->u.train.with_kv = with;
    return s;
}
OqlStmt *oql_y_compose(char *name, OqlNameList *from, OqlKV *with) {
    OqlStmt *s = oql_stmt_alloc(OQL_VERB_COMPOSE);
    s->u.compose.target = name; s->u.compose.from = from; s->u.compose.with_kv = with;
    return s;
}
OqlStmt *oql_y_run(char *name, OqlKV *with) {
    OqlStmt *s = oql_stmt_alloc(OQL_VERB_RUN);
    s->u.run.target = name; s->u.run.with_kv = with;
    return s;
}
OqlStmt *oql_y_evaluate(char *name, OqlSource against, char *metric, char *report) {
    OqlStmt *s = oql_stmt_alloc(OQL_VERB_EVALUATE);
    s->u.evaluate.target = name; s->u.evaluate.against_src = against;
    s->u.evaluate.metric = metric; s->u.evaluate.report_path = report;
    return s;
}
OqlStmt *oql_y_verify(OqlVerifySubjectKind k, char *subject, OqlPredicate *where) {
    OqlStmt *s = oql_stmt_alloc(OQL_VERB_VERIFY);
    s->u.verify.kind = k; s->u.verify.subject = subject; s->u.verify.where = where;
    return s;
}
OqlStmt *oql_y_audit(OqlSource a, OqlSource b, char *thr, char *report) {
    OqlStmt *s = oql_stmt_alloc(OQL_VERB_AUDIT);
    s->u.audit.a = a; s->u.audit.b = b;
    s->u.audit.thresholds = thr; s->u.audit.report_path = report;
    return s;
}
OqlStmt *oql_y_create_behaviour(char *name, char *vm_body) {
    OqlStmt *s = oql_stmt_alloc(OQL_VERB_CREATE_BEHAVIOUR);
    s->u.create_behaviour.name = name;
    s->u.create_behaviour.vm_body = vm_body;
    return s;
}
OqlStmt *oql_y_create_organelle(char *name, char *ckpt, OqlKV *bindings) {
    OqlStmt *s = oql_stmt_alloc(OQL_VERB_CREATE_ORGANELLE);
    s->u.create_organelle.name = name;
    s->u.create_organelle.checkpoint = ckpt;
    s->u.create_organelle.bindings = bindings;
    return s;
}
OqlKV *oql_y_kv(char *key, char *val) {
    OqlKV *k = (OqlKV *)calloc(1, sizeof(OqlKV));
    if (k) { k->key = key; k->value = val; }
    return k;
}
OqlKV *oql_y_kv_concat(OqlKV *head, OqlKV *tail) {
    if (!head) return tail;
    OqlKV *t = head; while (t->next) t = t->next; t->next = tail;
    return head;
}
OqlNameList *oql_y_name(char *n) {
    OqlNameList *l = (OqlNameList *)calloc(1, sizeof(OqlNameList));
    if (l) l->name = n;
    return l;
}
OqlNameList *oql_y_name_concat(OqlNameList *head, OqlNameList *tail) {
    if (!head) return tail;
    OqlNameList *t = head; while (t->next) t = t->next; t->next = tail;
    return head;
}
OqlPredicate *oql_y_pred(char *lhs, OqlOp op, char *rhs) {
    OqlPredicate *p = (OqlPredicate *)calloc(1, sizeof(OqlPredicate));
    if (p) { p->lhs = lhs; p->op = op; p->rhs = rhs; }
    return p;
}
void oql_y_append(oql_parser *p, OqlStmt *s) {
    if (!p || !p->script || !s) return;
    if (!p->script->head) { p->script->head = s; return; }
    OqlStmt *t = p->script->head;
    while (t->next) t = t->next;
    t->next = s;
}

/* ============================================================
 *  Bison/Flex glue
 * ============================================================ */

extern int  oql_parser_parse(oql_parser *parser);

/* Used by Flex's YY_INPUT macro. Single-threaded; see microgpt_vm.c for
 * the same trade-off. */
oql_parser *_oql_current_parser = NULL;

int oql_parser_char_fetch_next(oql_parser *p) {
    if (!p) return 0;
    if (p->source_index < p->source_len) {
        return (int)(unsigned char)p->source[p->source_index++];
    }
    return 0;
}

void oql_parser_error(oql_parser *parser, const char *msg) {
    if (!parser || !parser->script) return;
    if (parser->script->error) free(parser->script->error);
    /* Be conservative — bound the copy. */
    size_t n = msg ? strlen(msg) : 0;
    if (n > 512) n = 512;
    char *copy = (char *)malloc(n + 64);
    if (!copy) return;
    snprintf(copy, n + 64, "parse error at line %d: %.*s",
             parser->line_number, (int)n, msg ? msg : "(null)");
    parser->script->error = copy;
    parser->script->error_line = parser->line_number;
}

/* ============================================================
 *  Helpers
 * ============================================================ */

static char *oql_dup(const char *s) {
    if (!s) return NULL;
    size_t n = strlen(s) + 1;
    char *d = (char *)malloc(n);
    if (d) memcpy(d, s, n);
    return d;
}

/* ============================================================
 *  OqlKV
 * ============================================================ */

OqlKV *oql_kv_append(OqlKV *head, char *key, char *value) {
    OqlKV *k = (OqlKV *)calloc(1, sizeof(OqlKV));
    if (!k) return head;
    k->key = key;
    k->value = value;
    if (!head) return k;
    OqlKV *t = head;
    while (t->next) t = t->next;
    t->next = k;
    return head;
}

const char *oql_kv_get(const OqlKV *head, const char *key) {
    if (!key) return NULL;
    for (const OqlKV *k = head; k; k = k->next) {
        if (k->key && strcmp(k->key, key) == 0) return k->value;
    }
    return NULL;
}

void oql_kv_free(OqlKV *head) {
    while (head) {
        OqlKV *n = head->next;
        free(head->key);
        free(head->value);
        free(head);
        head = n;
    }
}

/* ============================================================
 *  OqlNameList
 * ============================================================ */

OqlNameList *oql_namelist_append(OqlNameList *head, char *name) {
    OqlNameList *n = (OqlNameList *)calloc(1, sizeof(OqlNameList));
    if (!n) return head;
    n->name = name;
    if (!head) return n;
    OqlNameList *t = head;
    while (t->next) t = t->next;
    t->next = n;
    return head;
}

void oql_namelist_free(OqlNameList *head) {
    while (head) {
        OqlNameList *n = head->next;
        free(head->name);
        free(head);
        head = n;
    }
}

/* ============================================================
 *  OqlSource / OqlPredicate
 * ============================================================ */

void oql_source_free(OqlSource *s) {
    if (!s) return;
    free(s->value);
    s->value = NULL;
    s->kind = 0;
}

void oql_predicate_free(OqlPredicate *p) {
    if (!p) return;
    free(p->lhs);
    free(p->rhs);
    free(p);
}

/* ============================================================
 *  OqlScript
 * ============================================================ */

OqlScript *oql_script_create(void) {
    return (OqlScript *)calloc(1, sizeof(OqlScript));
}

static void oql_stmt_free_inner(OqlStmt *s) {
    if (!s) return;
    switch (s->verb) {
    case OQL_VERB_TRAIN:
        free(s->u.train.target);
        oql_source_free(&s->u.train.on_src);
        oql_kv_free(s->u.train.with_kv);
        break;
    case OQL_VERB_COMPOSE:
        free(s->u.compose.target);
        oql_namelist_free(s->u.compose.from);
        oql_kv_free(s->u.compose.with_kv);
        break;
    case OQL_VERB_RUN:
        free(s->u.run.target);
        oql_kv_free(s->u.run.with_kv);
        break;
    case OQL_VERB_EVALUATE:
        free(s->u.evaluate.target);
        oql_source_free(&s->u.evaluate.against_src);
        free(s->u.evaluate.metric);
        free(s->u.evaluate.report_path);
        break;
    case OQL_VERB_VERIFY:
        free(s->u.verify.subject);
        oql_predicate_free(s->u.verify.where);
        break;
    case OQL_VERB_AUDIT:
        oql_source_free(&s->u.audit.a);
        oql_source_free(&s->u.audit.b);
        free(s->u.audit.thresholds);
        free(s->u.audit.report_path);
        break;
    case OQL_VERB_CREATE_BEHAVIOUR:
        free(s->u.create_behaviour.name);
        free(s->u.create_behaviour.vm_body);
        break;
    case OQL_VERB_CREATE_ORGANELLE:
        free(s->u.create_organelle.name);
        free(s->u.create_organelle.checkpoint);
        oql_kv_free(s->u.create_organelle.bindings);
        break;
    }
}

void oql_script_free(OqlScript *s) {
    if (!s) return;
    OqlStmt *cur = s->head;
    while (cur) {
        OqlStmt *next = cur->next;
        oql_stmt_free_inner(cur);
        free(cur);
        cur = next;
    }
    free(s->error);
    free(s);
}

size_t oql_script_count(const OqlScript *s) {
    if (!s) return 0;
    size_t n = 0;
    for (const OqlStmt *c = s->head; c; c = c->next) n++;
    return n;
}

/* ============================================================
 *  Parser entry point
 * ============================================================ */

OqlScript *oql_parse(const char *source) {
    OqlScript *script = oql_script_create();
    if (!script) return NULL;
    if (!source) {
        script->error = oql_dup("oql_parse: NULL source");
        return script;
    }

    oql_parser parser;
    memset(&parser, 0, sizeof(parser));
    parser.source = source;
    parser.source_len = strlen(source);
    parser.source_index = 0;
    parser.line_number = 1;
    parser.script = script;

    _oql_current_parser = &parser;

    int rc = oql_parser_parse(&parser);
    if (rc != 0 && !script->error) {
        script->error = oql_dup("parse failed");
    }
    _oql_current_parser = NULL;
    return script;
}

/* ============================================================
 *  Verb dispatch — VERIFY GRAPH (wired to pipeline_verify)
 * ============================================================ */

static oql_status oql_exec_verify(const OqlStmt *s, FILE *out) {
    if (!s) return OQL_ERR_RUNTIME;
    if (s->u.verify.kind != OQL_VS_GRAPH) {
        if (out) fprintf(out,
            "VERIFY: only `VERIFY GRAPH @graph...@end` is wired in this commit; "
            "subject form %d is implementation pending.\n", (int)s->u.verify.kind);
        return OQL_ERR_NOT_IMPLEMENTED;
    }
    if (!s->u.verify.subject) {
        if (out) fprintf(out, "VERIFY GRAPH: empty graph block\n");
        return OQL_ERR_RUNTIME;
    }
    Pipeline *p = pipeline_parse_text(s->u.verify.subject);
    if (!p) {
        if (out) fprintf(out,
            "VERIFY GRAPH: pipeline_parse_text failed: %s\n",
            pipeline_last_error());
        return OQL_ERR_RUNTIME;
    }
    int rc = pipeline_verify(p);
    if (out) {
        if (rc == 0) {
            fprintf(out, "VERIFY GRAPH: OK (%zu nodes)\n", p->n_nodes);
        } else {
            fprintf(out, "VERIFY GRAPH: FAILED (code %d): %s\n",
                    rc, pipeline_last_error());
        }
    }
    pipeline_free(p);
    return (rc == 0) ? OQL_OK : OQL_ERR_RUNTIME;
}

/* ============================================================
 *  Verb dispatch — AUDIT (shells out to scaling_leakage_audit.sh)
 *
 *  Locating the script: try CWD first, then the worktree-relative
 *  tools/ path. Quote the source paths to keep the shell happy.
 * ============================================================ */

static const char *oql_src_path(const OqlSource *s) {
    if (!s || !s->value) return NULL;
    return s->value;
}

static oql_status oql_exec_audit(const OqlStmt *s, FILE *out) {
    const char *a = oql_src_path(&s->u.audit.a);
    const char *b = oql_src_path(&s->u.audit.b);
    if (!a || !b) {
        if (out) fprintf(out, "AUDIT: both source paths required\n");
        return OQL_ERR_RUNTIME;
    }

    /* Choose the first existing audit script path. */
    const char *candidates[] = {
        "tools/scaling_leakage_audit.sh",
        "../tools/scaling_leakage_audit.sh",
        NULL
    };
    const char *script_path = NULL;
    for (int i = 0; candidates[i]; ++i) {
        FILE *f = fopen(candidates[i], "r");
        if (f) { fclose(f); script_path = candidates[i]; break; }
    }
    if (!script_path) {
        if (out) fprintf(out,
            "AUDIT: tools/scaling_leakage_audit.sh not found in cwd or ../tools — "
            "skipping shell-out (parser succeeded, executor declined).\n");
        return OQL_ERR_NOT_IMPLEMENTED;
    }

    /* Build a defensive shell command. */
    char cmd[1024];
    snprintf(cmd, sizeof(cmd),
             "sh \"%s\" \"%s\" \"%s\"%s%s%s%s",
             script_path, a, b,
             s->u.audit.thresholds ? " " : "",
             s->u.audit.thresholds ? s->u.audit.thresholds : "",
             s->u.audit.report_path ? " " : "",
             s->u.audit.report_path ? s->u.audit.report_path : "");
    if (out) fprintf(out, "AUDIT: invoking %s\n", cmd);
    int rc = system(cmd);
    if (rc != 0) {
        if (out) fprintf(out, "AUDIT: script returned %d\n", rc);
        return OQL_ERR_RUNTIME;
    }
    return OQL_OK;
}

/* ============================================================
 *  Verb dispatch — TRAIN / COMPOSE / RUN / EVALUATE (stubs)
 *
 *  All four are honest about scope: the grammar parses cleanly, but the
 *  interpreter declines with OQL_ERR_NOT_IMPLEMENTED. Follow-up commits
 *  wire each one to its underlying primitive (e.g. TrainWorker for
 *  TRAIN, the pipeline-compose path for COMPOSE).
 * ============================================================ */

static oql_status oql_exec_pending(const char *verb, FILE *out) {
    if (out) {
        fprintf(out, "%s: implementation pending in follow-up commit.\n", verb);
    }
    return OQL_ERR_NOT_IMPLEMENTED;
}

/* ============================================================
 *  Top-level executor
 * ============================================================ */

oql_status oql_execute(const OqlScript *script, FILE *out, int *failed_idx) {
    if (!script) return OQL_ERR_RUNTIME;
    if (failed_idx) *failed_idx = 0;
    if (script->error) {
        if (out) fprintf(out, "oql_execute: parse error: %s\n", script->error);
        return OQL_ERR_PARSE;
    }
    int idx = 0;
    for (const OqlStmt *s = script->head; s; s = s->next) {
        idx++;
        oql_status st = OQL_OK;
        switch (s->verb) {
        case OQL_VERB_TRAIN:    st = oql_exec_pending("TRAIN",    out); break;
        case OQL_VERB_COMPOSE:  st = oql_exec_pending("COMPOSE",  out); break;
        case OQL_VERB_RUN:      st = oql_exec_pending("RUN",      out); break;
        case OQL_VERB_EVALUATE: st = oql_exec_pending("EVALUATE", out); break;
        case OQL_VERB_VERIFY:   st = oql_exec_verify(s, out); break;
        case OQL_VERB_AUDIT:    st = oql_exec_audit (s, out); break;
        case OQL_VERB_CREATE_BEHAVIOUR:
            /* Parsing is the contract here.  The interpreter's behaviour
             * registry + VM compile step lives in the test harness (see
             * tests/test_microgpt_oql.c::test_e08_connect4_behaviours);
             * the OQL interpreter alone does not know about VM modules
             * and would pull a heavyweight dep — left as a follow-up. */
            if (out) fprintf(out,
                "CREATE BEHAVIOUR %s: parsed (vm body %zu bytes); "
                "compile step is harness-driven — see "
                "tests/test_microgpt_oql.c\n",
                s->u.create_behaviour.name ? s->u.create_behaviour.name : "?",
                s->u.create_behaviour.vm_body
                    ? strlen(s->u.create_behaviour.vm_body) : 0);
            st = OQL_OK;
            break;
        case OQL_VERB_CREATE_ORGANELLE: {
            int n_bindings = 0;
            for (const OqlKV *k = s->u.create_organelle.bindings; k; k = k->next) n_bindings++;
            if (out) fprintf(out,
                "CREATE ORGANELLE %s: parsed (%d bindings)\n",
                s->u.create_organelle.name ? s->u.create_organelle.name : "?",
                n_bindings);
            st = OQL_OK;
            break;
        }
        }
        if (st != OQL_OK) {
            if (failed_idx) *failed_idx = idx;
            return st;
        }
    }
    return OQL_OK;
}
