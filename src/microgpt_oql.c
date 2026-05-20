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
#include "microgpt.h"        /* Model, MicrogptConfig, checkpoint_load */
#include "microgpt_vm.h"     /* vm_module, vm_module_dispose */

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
 *  E09 — Runtime registry (organelle / behaviour / pipeline tables)
 * ============================================================ */

void oql_runtime_init(OqlRuntime *rt) {
    if (!rt) return;
    memset(rt, 0, sizeof(*rt));
}

void oql_runtime_dispose(OqlRuntime *rt) {
    if (!rt) return;
    /* Free behaviour entries. */
    for (int i = 0; i < rt->n_behaviours; i++) {
        free(rt->behaviours[i].name);
        free(rt->behaviours[i].vm_body);
        if (rt->behaviours[i].module) {
            vm_module_dispose((vm_module *)rt->behaviours[i].module);
            rt->behaviours[i].module = NULL;
        }
    }
    rt->n_behaviours = 0;

    /* Free organelle models (only those actually loaded). */
    for (int i = 0; i < rt->n_organelles; i++) {
        if (rt->organelles[i].loaded && rt->organelles[i].model) {
            model_free(rt->organelles[i].model);
        }
        rt->organelles[i].model = NULL;
        rt->organelles[i].loaded = 0;
    }
    rt->n_organelles = 0;

    /* Free parsed pipeline IRs. */
    for (int i = 0; i < rt->n_pipelines; i++) {
        if (rt->pipelines[i].ir) {
            pipeline_free(rt->pipelines[i].ir);
            rt->pipelines[i].ir = NULL;
        }
    }
    rt->n_pipelines = 0;

    rt->cfg = NULL;
}

OqlBehaviourEntry *oql_runtime_find_behaviour(OqlRuntime *rt, const char *name) {
    if (!rt || !name) return NULL;
    for (int i = 0; i < rt->n_behaviours; i++) {
        if (rt->behaviours[i].name &&
            strcmp(rt->behaviours[i].name, name) == 0) {
            return &rt->behaviours[i];
        }
    }
    return NULL;
}

OqlOrganelle *oql_runtime_find_organelle(OqlRuntime *rt, const char *name) {
    if (!rt || !name) return NULL;
    for (int i = 0; i < rt->n_organelles; i++) {
        if (strcmp(rt->organelles[i].name, name) == 0) {
            return &rt->organelles[i];
        }
    }
    return NULL;
}

OqlPipeline *oql_runtime_find_pipeline(OqlRuntime *rt, const char *name) {
    if (!rt || !name) return NULL;
    for (int i = 0; i < rt->n_pipelines; i++) {
        if (strcmp(rt->pipelines[i].name, name) == 0) {
            return &rt->pipelines[i];
        }
    }
    return NULL;
}

/* Register a behaviour (called from CREATE BEHAVIOUR dispatch). */
static oql_status oql_runtime_register_behaviour(OqlRuntime *rt,
                                                 const char *name,
                                                 const char *vm_body,
                                                 FILE *out) {
    if (!rt || !name) return OQL_ERR_RUNTIME;
    if (oql_runtime_find_behaviour(rt, name)) {
        if (out) fprintf(out, "CREATE BEHAVIOUR: duplicate name '%s'\n", name);
        return OQL_ERR_RUNTIME;
    }
    if (rt->n_behaviours >= OQL_MAX_BEHAVIOURS) {
        if (out) fprintf(out, "CREATE BEHAVIOUR: registry full (%d max)\n",
                         OQL_MAX_BEHAVIOURS);
        return OQL_ERR_RUNTIME;
    }
    OqlBehaviourEntry *e = &rt->behaviours[rt->n_behaviours++];
    e->name = oql_dup(name);
    e->vm_body = vm_body ? oql_dup(vm_body) : NULL;
    e->module = NULL;
    return OQL_OK;
}

/* Register an organelle (CREATE ORGANELLE).  Lazy: no checkpoint load. */
static oql_status oql_runtime_register_organelle(OqlRuntime *rt,
                                                 const OqlCreateOrganelle *co,
                                                 FILE *out) {
    if (!rt || !co || !co->name) return OQL_ERR_RUNTIME;
    if (oql_runtime_find_organelle(rt, co->name)) {
        if (out) fprintf(out, "CREATE ORGANELLE: duplicate name '%s'\n", co->name);
        return OQL_ERR_RUNTIME;
    }
    if (rt->n_organelles >= OQL_MAX_ORGANELLES) {
        if (out) fprintf(out, "CREATE ORGANELLE: registry full (%d max)\n",
                         OQL_MAX_ORGANELLES);
        return OQL_ERR_RUNTIME;
    }
    OqlOrganelle *o = &rt->organelles[rt->n_organelles++];
    memset(o, 0, sizeof(*o));
    strncpy(o->name, co->name, sizeof(o->name) - 1);
    if (co->checkpoint) {
        strncpy(o->checkpoint_path, co->checkpoint,
                sizeof(o->checkpoint_path) - 1);
    }
    /* Resolve behaviour-binding kvs by name (strings only — not pointers). */
    const char *bn;
    if ((bn = oql_kv_get(co->bindings, "INPUT_BEHAVIOUR"))) {
        strncpy(o->input_behaviour, bn, sizeof(o->input_behaviour) - 1);
    }
    if ((bn = oql_kv_get(co->bindings, "OUTPUT_BEHAVIOUR"))) {
        strncpy(o->output_behaviour, bn, sizeof(o->output_behaviour) - 1);
    }
    if ((bn = oql_kv_get(co->bindings, "VALIDATE_BEHAVIOUR"))) {
        strncpy(o->validate_behaviour, bn, sizeof(o->validate_behaviour) - 1);
    }
    if ((bn = oql_kv_get(co->bindings, "FALLBACK_BEHAVIOUR"))) {
        strncpy(o->fallback_behaviour, bn, sizeof(o->fallback_behaviour) - 1);
    }
    if ((bn = oql_kv_get(co->bindings, "SCORE_BEHAVIOUR"))) {
        strncpy(o->score_behaviour, bn, sizeof(o->score_behaviour) - 1);
    }
    if ((bn = oql_kv_get(co->bindings, "CYCLE_DETECT_BEHAVIOUR"))) {
        strncpy(o->cycle_detect_behaviour, bn,
                sizeof(o->cycle_detect_behaviour) - 1);
    }
    return OQL_OK;
}

/* Default config used when caller didn't provide one.  Built from the
 * compile-time macros that the OQL lib was compiled against; matches
 * what the connect4 demo emits when saving its player checkpoint. */
static MicrogptConfig oql_default_cfg(void) {
    MicrogptConfig cfg = microgpt_default_config();
    cfg.n_embd = N_EMBD;
    cfg.n_head = N_HEAD;
    cfg.mlp_dim = MLP_DIM;
    cfg.n_layer = N_LAYER;
    cfg.block_size = BLOCK_SIZE;
    return cfg;
}

Model *oql_runtime_load_organelle(OqlRuntime *rt, OqlOrganelle *organelle,
                                  FILE *out) {
    if (!rt || !organelle) return NULL;
    if (organelle->loaded) return organelle->model;
    if (!organelle->checkpoint_path[0]) {
        if (out) fprintf(out,
            "load_organelle: organelle '%s' has no checkpoint path\n",
            organelle->name);
        return NULL;
    }
    MicrogptConfig local_cfg = oql_default_cfg();
    const MicrogptConfig *cfg = rt->cfg ? (const MicrogptConfig *)rt->cfg
                                        : &local_cfg;

    /* Peek at the checkpoint header to learn its actual vocab size.
     * The microgpt checkpoint format is: [int step][size_t vocab][weights].
     * checkpoint_load() rejects loads where the caller-supplied vocab_size
     * doesn't match the header — so we read it ourselves first. */
    FILE *peek = fopen(organelle->checkpoint_path, "rb");
    if (!peek) {
        if (out) fprintf(out,
            "load_organelle: cannot open '%s' (organelle '%s')\n",
            organelle->checkpoint_path, organelle->name);
        return NULL;
    }
    int header_step = 0;
    size_t header_vocab = 0;
    if (fread(&header_step, sizeof(int), 1, peek) != 1 ||
        fread(&header_vocab, sizeof(size_t), 1, peek) != 1) {
        if (out) fprintf(out,
            "load_organelle: failed to read header from '%s'\n",
            organelle->checkpoint_path);
        fclose(peek);
        return NULL;
    }
    fclose(peek);

    /* Adam buffers — model_num_params is vocab × N_EMBD × 2 (wte+lm_head)
     * plus block_size × N_EMBD (wpe) plus per-layer matmul slabs.  For
     * checkpoint_load we just need buffers large enough; allocate from
     * the actual param-count formula. */
    const size_t ne = (size_t)cfg->n_embd;
    const size_t bs_ = (size_t)cfg->block_size;
    const size_t md_ = (size_t)cfg->mlp_dim;
    const int nl_  = cfg->n_layer;
    size_t scratch_n = header_vocab * ne * 2   /* wte + lm_head */
                     + bs_ * ne                /* wpe */
                     + (size_t)nl_ * (4 * ne * ne + 2 * md_ * ne)
                     + 1024;                   /* slack */
    scalar_t *m = (scalar_t *)calloc(scratch_n, sizeof(scalar_t));
    scalar_t *v = (scalar_t *)calloc(scratch_n, sizeof(scalar_t));
    if (!m || !v) {
        if (out) fprintf(out, "load_organelle: OOM allocating Adam scratch\n");
        free(m); free(v);
        return NULL;
    }
    int step_out = 0;
    Model *model = checkpoint_load(organelle->checkpoint_path,
                                   header_vocab, cfg, m, v, &step_out);
    free(m); free(v);
    if (!model) {
        if (out) fprintf(out,
            "load_organelle: checkpoint_load('%s') failed for organelle '%s' "
            "(header_vocab=%zu, cfg dims n_embd=%d n_head=%d n_layer=%d block=%d mlp=%d)\n",
            organelle->checkpoint_path, organelle->name,
            header_vocab, cfg->n_embd, cfg->n_head, cfg->n_layer,
            cfg->block_size, cfg->mlp_dim);
        return NULL;
    }
    organelle->model = model;
    organelle->loaded = 1;
    if (out) fprintf(out,
        "load_organelle: loaded '%s' from %s (vocab=%zu step=%d)\n",
        organelle->name, organelle->checkpoint_path, header_vocab, step_out);
    return model;
}

/* ============================================================
 *  E09 — COMPOSE: parse @graph body from WITH kv, register as pipeline
 * ============================================================ */

static oql_status oql_runtime_register_pipeline(OqlRuntime *rt,
                                                const OqlCompose *co,
                                                FILE *out) {
    if (!rt || !co || !co->target) return OQL_ERR_RUNTIME;
    if (oql_runtime_find_pipeline(rt, co->target)) {
        if (out) fprintf(out, "COMPOSE: duplicate pipeline name '%s'\n",
                         co->target);
        return OQL_ERR_RUNTIME;
    }
    if (rt->n_pipelines >= OQL_MAX_PIPELINES) {
        if (out) fprintf(out, "COMPOSE: registry full (%d max)\n",
                         OQL_MAX_PIPELINES);
        return OQL_ERR_RUNTIME;
    }
    OqlPipeline *p = &rt->pipelines[rt->n_pipelines++];
    memset(p, 0, sizeof(*p));
    strncpy(p->name, co->target, sizeof(p->name) - 1);

    /* Three composition forms supported:
     *  (1) WITH (GRAPH = '<inline @graph block>') — parsed via
     *      pipeline_parse_text; the resulting IR is stored in p->ir
     *      and each call(...) node's primitive is resolved against the
     *      organelle table to populate call_organelles[].
     *  (2) WITH (PIPELINE = 'path/to/graph.txt') — slurps the file and
     *      parses identically to (1).
     *  (3) Plain FROM a, b, c with no GRAPH kv — linear chain across
     *      the named organelles in source order.
     */
    const char *graph_text = oql_kv_get(co->with_kv, "GRAPH");
    const char *graph_path = oql_kv_get(co->with_kv, "PIPELINE");
    char *slurped = NULL;
    if (!graph_text && graph_path) {
        FILE *f = fopen(graph_path, "rb");
        if (f) {
            fseek(f, 0, SEEK_END);
            long n = ftell(f);
            fseek(f, 0, SEEK_SET);
            slurped = (char *)malloc((size_t)n + 1);
            if (slurped) {
                fread(slurped, 1, (size_t)n, f);
                slurped[n] = '\0';
                graph_text = slurped;
            }
            fclose(f);
        }
    }

    if (graph_text) {
        Pipeline *ir = pipeline_parse_text(graph_text);
        free(slurped);
        if (!ir) {
            if (out) fprintf(out,
                "COMPOSE: pipeline_parse_text failed: %s\n",
                pipeline_last_error());
            rt->n_pipelines--;
            return OQL_ERR_RUNTIME;
        }
        if (pipeline_verify(ir) != 0) {
            if (out) fprintf(out,
                "COMPOSE: pipeline_verify failed: %s\n",
                pipeline_last_error());
            pipeline_free(ir);
            rt->n_pipelines--;
            return OQL_ERR_RUNTIME;
        }
        /* Resolve each call-node's primitive against the organelle table.
         * We use the node's "primitive" string as the organelle name. */
        for (size_t ni = 0; ni < ir->n_nodes; ni++) {
            const char *prim = ir->nodes[ni]->primitive;
            if (!prim) continue;
            if (!oql_runtime_find_organelle(rt, prim)) {
                if (out) fprintf(out,
                    "COMPOSE '%s': node '%s' references unknown organelle '%s'\n",
                    co->target, ir->nodes[ni]->id, prim);
                pipeline_free(ir);
                rt->n_pipelines--;
                return OQL_ERR_RUNTIME;
            }
            if (p->n_calls < OQL_MAX_PIPELINE_CALLS) {
                strncpy(p->call_organelles[p->n_calls], prim,
                        sizeof(p->call_organelles[0]) - 1);
                p->n_calls++;
            }
        }
        p->ir = ir;
        if (out) fprintf(out,
            "COMPOSE: registered pipeline '%s' with IR (%zu nodes, %d calls)\n",
            co->target, ir->n_nodes, p->n_calls);
    } else {
        /* Linear-chain fallback: walk co->from in order, validate each. */
        for (const OqlNameList *n = co->from; n; n = n->next) {
            if (!oql_runtime_find_organelle(rt, n->name)) {
                if (out) fprintf(out,
                    "COMPOSE '%s': unknown organelle '%s' in FROM list\n",
                    co->target, n->name ? n->name : "(null)");
                rt->n_pipelines--;
                return OQL_ERR_RUNTIME;
            }
            if (p->n_calls < OQL_MAX_PIPELINE_CALLS) {
                strncpy(p->call_organelles[p->n_calls], n->name,
                        sizeof(p->call_organelles[0]) - 1);
                p->n_calls++;
            }
        }
        if (out) fprintf(out,
            "COMPOSE: registered linear pipeline '%s' (%d organelles)\n",
            co->target, p->n_calls);
    }
    return OQL_OK;
}

/* ============================================================
 *  E09 — RUN dispatch.
 *
 *  Recognised WITH kvs (case-sensitive):
 *      MODE       = game_loop          (default; only mode wired in this commit)
 *      OPPONENT   = random             (only opponent wired in this commit)
 *      GAMES      = <integer>          (default 1)
 *      SEED       = <integer>          (default 42)
 *      GAME       = connect4           (selects the game-specific harness;
 *                                       only 'connect4' is wired in this commit)
 *
 *  Returns:
 *      OQL_OK on completed run (regardless of win rate);
 *      OQL_ERR_NOT_IMPLEMENTED for non-game_loop modes or unknown games;
 *      OQL_ERR_RUNTIME if dispatch fails (e.g. pipeline not found,
 *      checkpoint load fails).
 * ============================================================ */

/* Forward decl — concrete implementation lives in oql_runtime_games.c so
 * the game-specific harness can be unit-tested independently. */
oql_status oql_run_game_loop(OqlRuntime *rt,
                             OqlPipeline *pipeline,
                             const char *opponent,
                             int games,
                             unsigned int seed,
                             const char *game,
                             FILE *out);

static oql_status oql_exec_run_runtime(OqlRuntime *rt, const OqlStmt *s,
                                       FILE *out) {
    if (!rt || !s) return OQL_ERR_RUNTIME;
    const char *pipe_name = s->u.run.target;
    OqlPipeline *p = oql_runtime_find_pipeline(rt, pipe_name);
    if (!p) {
        /* RUN may target a single organelle directly when no COMPOSE
         * was issued — register an implicit one-stage pipeline. */
        OqlOrganelle *o = oql_runtime_find_organelle(rt, pipe_name);
        if (!o) {
            if (out) fprintf(out,
                "RUN: unknown pipeline / organelle '%s'\n", pipe_name);
            return OQL_ERR_RUNTIME;
        }
        if (rt->n_pipelines >= OQL_MAX_PIPELINES) {
            if (out) fprintf(out, "RUN: pipeline table full\n");
            return OQL_ERR_RUNTIME;
        }
        p = &rt->pipelines[rt->n_pipelines++];
        memset(p, 0, sizeof(*p));
        strncpy(p->name, pipe_name, sizeof(p->name) - 1);
        strncpy(p->call_organelles[0], o->name, sizeof(p->call_organelles[0]) - 1);
        p->n_calls = 1;
    }

    const char *mode     = oql_kv_get(s->u.run.with_kv, "MODE");
    const char *opponent = oql_kv_get(s->u.run.with_kv, "OPPONENT");
    const char *games_s  = oql_kv_get(s->u.run.with_kv, "GAMES");
    const char *seed_s   = oql_kv_get(s->u.run.with_kv, "SEED");
    const char *game     = oql_kv_get(s->u.run.with_kv, "GAME");
    int games = games_s ? atoi(games_s) : 1;
    unsigned int seed = seed_s ? (unsigned int)atoi(seed_s) : 42u;
    if (games < 1) games = 1;
    if (!opponent) opponent = "random";
    if (!game) game = "connect4";
    if (mode && strcmp(mode, "game_loop") != 0) {
        if (out) fprintf(out, "RUN: only MODE=game_loop is wired (got '%s')\n", mode);
        return OQL_ERR_NOT_IMPLEMENTED;
    }
    return oql_run_game_loop(rt, p, opponent, games, seed, game, out);
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

/* Internal core executor — handles both legacy oql_execute (no runtime)
 * and the E09 oql_execute_with_runtime (registry-backed). */
static oql_status oql_execute_core(const OqlScript *script, OqlRuntime *rt,
                                   FILE *out, int *failed_idx) {
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
        case OQL_VERB_TRAIN:
            /* E09 T6 hard-lock: TRAIN stays a stub. */
            st = oql_exec_pending("TRAIN", out);
            break;
        case OQL_VERB_COMPOSE:
            if (rt) {
                st = oql_runtime_register_pipeline(rt, &s->u.compose, out);
            } else {
                st = oql_exec_pending("COMPOSE", out);
            }
            break;
        case OQL_VERB_RUN:
            if (rt) {
                st = oql_exec_run_runtime(rt, s, out);
            } else {
                st = oql_exec_pending("RUN", out);
            }
            break;
        case OQL_VERB_EVALUATE:
            st = oql_exec_pending("EVALUATE", out);
            break;
        case OQL_VERB_VERIFY:   st = oql_exec_verify(s, out); break;
        case OQL_VERB_AUDIT:    st = oql_exec_audit (s, out); break;
        case OQL_VERB_CREATE_BEHAVIOUR:
            if (rt) {
                st = oql_runtime_register_behaviour(rt,
                    s->u.create_behaviour.name,
                    s->u.create_behaviour.vm_body, out);
                if (st == OQL_OK && out) {
                    fprintf(out,
                        "CREATE BEHAVIOUR %s: registered (%zu bytes)\n",
                        s->u.create_behaviour.name ? s->u.create_behaviour.name : "?",
                        s->u.create_behaviour.vm_body
                            ? strlen(s->u.create_behaviour.vm_body) : 0);
                }
            } else {
                /* Legacy oql_execute path — keep the original message so
                 * existing test harnesses (E08 worked example) still see
                 * "parsed (vm body ... bytes)" exactly. */
                if (out) fprintf(out,
                    "CREATE BEHAVIOUR %s: parsed (vm body %zu bytes); "
                    "compile step is harness-driven — see "
                    "tests/test_microgpt_oql.c\n",
                    s->u.create_behaviour.name ? s->u.create_behaviour.name : "?",
                    s->u.create_behaviour.vm_body
                        ? strlen(s->u.create_behaviour.vm_body) : 0);
                st = OQL_OK;
            }
            break;
        case OQL_VERB_CREATE_ORGANELLE:
            if (rt) {
                st = oql_runtime_register_organelle(rt, &s->u.create_organelle, out);
                if (st == OQL_OK && out) {
                    int n_bindings = 0;
                    for (const OqlKV *k = s->u.create_organelle.bindings; k; k = k->next) n_bindings++;
                    fprintf(out,
                        "CREATE ORGANELLE %s: registered (%d bindings, lazy load)\n",
                        s->u.create_organelle.name ? s->u.create_organelle.name : "?",
                        n_bindings);
                }
            } else {
                int n_bindings = 0;
                for (const OqlKV *k = s->u.create_organelle.bindings; k; k = k->next) n_bindings++;
                if (out) fprintf(out,
                    "CREATE ORGANELLE %s: parsed (%d bindings)\n",
                    s->u.create_organelle.name ? s->u.create_organelle.name : "?",
                    n_bindings);
                st = OQL_OK;
            }
            break;
        }
        if (st != OQL_OK) {
            if (failed_idx) *failed_idx = idx;
            return st;
        }
    }
    return OQL_OK;
}

oql_status oql_execute(const OqlScript *script, FILE *out, int *failed_idx) {
    return oql_execute_core(script, NULL, out, failed_idx);
}

oql_status oql_execute_with_runtime(const OqlScript *script,
                                    OqlRuntime *rt,
                                    FILE *out, int *failed_idx) {
    return oql_execute_core(script, rt, out, failed_idx);
}
