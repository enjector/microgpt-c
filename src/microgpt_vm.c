/*
 * microgpt_vm.c  —  MicroGPT-C Virtual Machine Engine (implementation)
 *
 * Copyright (c) 2026 Ajay Soni.  MIT License.
 *
 * Implements all non-inline functions declared in microgpt_vm.h:
 *   - Engine API (vm_engine_*)
 *   - Verb dispatch layer (verb_register, verb_compile, verb_exec)
 *   - Module / function lifecycle
 *   - Code generator and deferred code fragments
 *   - Post-processing passes (jump tables, type decoration, variable interning)
 *   - Parser interface (Flex/Bison integration)
 *   - Runtime execution (opcode interpreter loop)
 *   - Expression evaluator (vm_eval)
 *   - Legacy vm_queue-based runtime
 */

#include "microgpt_vm.h"

/* ═══════════════════════════════════════════════════════════════════════════
 *  Engine API  —  high-level C interface (vm_engine)
 * ═══════════════════════════════════════════════════════════════════════════
 */

#include <math.h>
#include <stdio.h>
#include <string.h>

/* ── Internal state ─────────────────────────────────────────────────────── */

/* Simple dynamic array for native functions */
typedef struct {
  char *name;
  vm_native_fn fn;
} native_fn_entry;

struct vm_engine_t {
  vm_module *module;
  vm_module_runtime *runtime;
  char last_error[1024];
  double result_number;
  char result_string[4096];
  int result_bool;

  native_fn_entry *native_fns;
  size_t native_fns_count;
  size_t native_fns_cap;
};

/* ── Native callback dispatcher ─────────────────────────────────────────── */

static void _ext_method_dispatch(vm_module_runtime *runtime,
                                 vm_function *function) {
  /* The vm_engine pointer is stored in the module's verb_context slot */
  vm_engine *e = (struct vm_engine_t *)runtime->module->verb_context_;

  vm_native_fn target_fn = NULL;
  for (size_t i = 0; i < e->native_fns_count; ++i) {
    if (strcmp(e->native_fns[i].name, function->name) == 0) {
      target_fn = e->native_fns[i].fn;
      break;
    }
  }

  if (!target_fn) {
    fprintf(stderr, "[vm_engine] Unknown native function: %s\n",
            function->name);
    vm_module_runtime_stack_push_number(runtime, 0.0);
    return;
  }

  /* Pop arguments from the runtime stack into a local array */
  double args[32];
  int argc = 0;
  vm_variable *var = NULL;

  /* Pop arguments until stack empty or buffer full */
  while (vm_module_runtime_stack_pop(runtime, &var) == VM_OK && var &&
         argc < 32) {
    args[argc++] = var->value.number;
  }

  double ret = target_fn(argc, args);
  vm_module_runtime_stack_push_number(runtime, ret);
}

/* ── Lifecycle ──────────────────────────────────────────────────────────── */

vm_engine *vm_engine_create(void) {
  vm_engine *e = (struct vm_engine_t *)calloc(1, sizeof(vm_engine));
  if (!e)
    return NULL;
  e->module = NULL;
  e->runtime = NULL;
  e->last_error[0] = '\0';
  e->result_number = 0.0;
  e->result_string[0] = '\0';
  e->result_bool = 0;

  e->native_fns_cap = 8;
  e->native_fns =
      (native_fn_entry *)malloc(e->native_fns_cap * sizeof(native_fn_entry));
  e->native_fns_count = 0;

  return e;
}

void vm_engine_dispose(vm_engine *e) {
  if (!e)
    return;
  if (e->runtime)
    vm_module_runtime_dispose(e->runtime);
  if (e->module)
    vm_module_dispose(e->module);

  for (size_t i = 0; i < e->native_fns_count; ++i) {
    free(e->native_fns[i].name);
  }
  free(e->native_fns);
  free(e);
}

/* ── Registration ───────────────────────────────────────────────────────── */

void vm_engine_register_fn(vm_engine *e, const char *name, vm_native_fn fn) {
  if (!e || !name || !fn)
    return;

  /* Update existing */
  for (size_t i = 0; i < e->native_fns_count; ++i) {
    if (strcmp(e->native_fns[i].name, name) == 0) {
      e->native_fns[i].fn = fn;
      return;
    }
  }

  /* Append new */
  if (e->native_fns_count == e->native_fns_cap) {
    e->native_fns_cap *= 2;
    e->native_fns = (native_fn_entry *)realloc(
        e->native_fns, e->native_fns_cap * sizeof(native_fn_entry));
  }

  e->native_fns[e->native_fns_count].name = vm_string_clone(name);
  e->native_fns[e->native_fns_count].fn = fn;
  e->native_fns_count++;
}

/* ── Loading ────────────────────────────────────────────────────────────── */

int vm_engine_load(vm_engine *e, const char *source) {
  if (!e || !source)
    return -1;

  /* Clean up previous state */
  if (e->runtime) {
    vm_module_runtime_dispose(e->runtime);
    e->runtime = NULL;
  }
  if (e->module) {
    vm_module_dispose(e->module);
    e->module = NULL;
  }
  e->last_error[0] = '\0';

  /* Parse into module, passing the engine as verb_context for native dispatch
   */
  vm_module *mod = NULL;
  if (vm_module_parser_generate((verb_context *)e, source, &mod) != VM_OK ||
      !mod) {
    snprintf(e->last_error, sizeof(e->last_error),
             "Parse failed — empty module");
    return -1;
  }

  if (vm_list_count(mod->errors) > 0) {
    /* Collect errors into last_error */
    char buf[64];
    snprintf(e->last_error, sizeof(e->last_error), "Parse errors:");
    vm_list_foreach_of(mod->errors, vm_module_error *, err) {
      snprintf(buf, sizeof(buf), " [L%zu:%zu]", err->source_line_number,
               err->source_line_column);
      strncat(e->last_error, buf,
              sizeof(e->last_error) - strlen(e->last_error) - 1);
      strncat(e->last_error, err->message,
              sizeof(e->last_error) - strlen(e->last_error) - 1);
    }
    vm_module_dispose(mod);
    return -1;
  }

  e->module = mod;
  e->runtime = vm_module_runtime_create(mod);
  if (!e->runtime) {
    snprintf(e->last_error, sizeof(e->last_error), "Failed to create runtime");
    vm_module_dispose(mod);
    e->module = NULL;
    return -1;
  }

  /* Wire up native function dispatcher */
  vm_module_runtime_set_call_ext_method_callback(e->runtime,
                                                 _ext_method_dispatch);

  return 0;
}

/* ── Execution ──────────────────────────────────────────────────────────── */

int vm_engine_run(vm_engine *e, const char *fn_name) {
  if (!e || !e->module || !e->runtime) {
    if (e)
      snprintf(e->last_error, sizeof(e->last_error), "No module loaded");
    return -1;
  }

  vm_function *fn = vm_module_fetch_function(e->module, fn_name);
  if (!fn) {
    snprintf(e->last_error, sizeof(e->last_error), "Function not found: %s",
             fn_name);
    return -1;
  }

  vm_module_runtime_clear(e->runtime);
  fn->instruction_pointer = 0;

  if (!vm_module_runtime_run(e->runtime, fn)) {
    snprintf(e->last_error, sizeof(e->last_error),
             "Runtime error in function: %s", fn_name);
    return -1;
  }

  /* Retrieve return value */
  vm_variable *ret = NULL;
  if (vm_module_runtime_stack_pop(e->runtime, &ret) == VM_OK && ret) {
    e->result_number = ret->value.number;
    e->result_bool = ret->value.boolean ? 1 : 0;
    if (ret->value.string) {
      strncpy(e->result_string, ret->value.string,
              sizeof(e->result_string) - 1);
    } else {
      e->result_string[0] = '\0';
    }
  }

  return 0;
}

/* ── Results ────────────────────────────────────────────────────────────── */

double vm_engine_result_number(const vm_engine *e) {
  return e ? e->result_number : 0.0;
}
int vm_engine_result_bool(const vm_engine *e) { return e ? e->result_bool : 0; }
const char *vm_engine_result_string(const vm_engine *e) {
  return e ? e->result_string : "";
}
const char *vm_engine_last_error(const vm_engine *e) {
  return e ? e->last_error : "";
}

/* ── Diagnostics ────────────────────────────────────────────────────────── */

void vm_engine_dump_il(const vm_engine *e) {
  if (!e || !e->module)
    return;
  char *il = vm_module_to_string(e->module);
  if (il) {
    puts(il);
    vm_string_free(il);
  }
}

/* ═══════════════════════════════════════════════════════════════════════════
 *  Verb System  —  DSL command dispatch layer
 *
 *  A simple command-dispatch framework: callers register "verbs" (named
 *  commands with typed parameters), then parse + execute natural-language
 *  sentences against the registry.
 *
 *  verb_register     — binds a name + parameter spec to a C callback.
 *  verb_compile      — parses a sentence, matches the longest verb, and
 *                      extracts arguments into a vm_map.
 *  verb_exec         — compile + dispatch in one step.
 *  verb_unregister   — removes a verb registration.
 * ═══════════════════════════════════════════════════════════════════════════
 */

void verb_definition_dispose(verb_definition *def) {
  if (!def)
    return;
  free((void *)def->name);
  free((void *)def->definition_params);
  if (def->param_list) {
    vm_list_dispose_items(def->param_list);
    vm_list_free(def->param_list);
  }
  free(def);
}

void verb_context_dispose(verb_context *ctx) {
  if (!ctx)
    return;
  if (ctx->verb_definition_map) {
    vm_map_foreach_of(ctx->verb_definition_map, _vname, verb_definition *,
                      _vdef) {
      verb_definition_dispose(_vdef);
    }
    vm_map_free(ctx->verb_definition_map);
  }
  VM_FREE(ctx);
}

int verb_register(verb_context *ctx, const char *name, const char *params,
                  verb_function fn, void *fctx) {
  if (!ctx || !name || !fn)
    return RESULT_CORE_VERB_ERROR_ARGUMENT_NULL;

  /* Check for duplicates */
  if (verb_find(ctx, name) != NULL)
    return RESULT_CORE_VERB_ERROR_ALREADY_REGISTERED;

  /* Parse parameter names from the space-separated string */
  vm_list *verb_param_list = vm_list_create();
  vm_string_buffer *usage_buf = vm_string_buffer_create_empty();

  if (params && strcmp(params, "") != 0) {
    const size_t plen = strlen(params);
    char *buf = (char *)calloc(1, plen + 1);
    char *buf_idx = buf;
    buf[0] = '\0';

    const char *p = params;
    char c = 0;
    int is_delim = 0, is_end = 0;

    do {
      c = *p++;
      is_delim = (c == ' ');
      is_end = (c == '\0');

      if (is_delim || is_end) {
        *buf_idx = '\0';
        buf_idx = buf;
        vm_list_add(verb_param_list, vm_string_clone(buf));

        /* Build usage string: <param1> <param2> ... */
        if (vm_list_count(verb_param_list) > 1)
          vm_string_buffer_append(usage_buf, " ");
        vm_string_buffer_append_format(usage_buf, "<%s>", buf);
      } else {
        *buf_idx++ = c;
      }
    } while (!is_end);

    VM_FREE(buf);
  }

  /* Create the verb definition */
  verb_definition *def = VM_NEW(verb_definition);
  def->name = vm_string_clone(name);
  def->definition_params = vm_string_buffer_free_not_data(usage_buf);
  def->name_length = strlen(name);
  def->param_list = verb_param_list;
  def->params_count = vm_list_count(verb_param_list);
  def->function = fn;
  def->fcontext = fctx;
  def->next_verb_definition = NULL;

  vm_map_set(ctx->verb_definition_map, name, def);
  return VM_OK;
}

int verb_unregister(verb_context *ctx, const char *name) {
  if (!ctx || !name)
    return RESULT_CORE_VERB_ERROR_ARGUMENT_NULL;

  verb_definition *def = verb_find(ctx, name);
  if (!def)
    return RESULT_CORE_VERB_ERROR_NO_MATCH;

  verb_definition_dispose(def);
  vm_map_remove(ctx->verb_definition_map, name);
  return VM_OK;
}

/* ── verb_compile: natural-language sentence parser ────────────────────────
 *
 * Longest-match verb lookup, followed by structured argument extraction.
 * Handles: bare words, single/double quoted strings, JSON arrays/objects,
 * and XML fragments with proper nesting.
 */

#define VERB_MAX_EXIT_TOKENS 100

verb_compiled *verb_compile(verb_context *ctx, const char *sentence,
                            void *context) {
  verb_compiled *compiled = VM_NEW(verb_compiled);
  memset(compiled, 0, sizeof(verb_compiled));
  compiled->context = context;

  if (!ctx) {
    compiled->vm_result = RESULT_CORE_VERB_ERROR_ARGUMENT_NULL;
    return compiled;
  }

  if (!sentence || strlen(sentence) == 0) {
    compiled->vm_result = RESULT_CORE_VERB_ERROR_ARGUMENT_NULL;
    return compiled;
  }

  const size_t sentence_length = strlen(sentence);

  /* ── Longest-match verb lookup ── */
  verb_definition *best = NULL;
  size_t best_len = 0;

  vm_map_foreach_of(ctx->verb_definition_map, _vk, verb_definition *, vd) {
    if (vd->name_length <= sentence_length) {
      if (strncmp(vd->name, sentence, vd->name_length) == 0) {
        if (vd->name_length > best_len) {
          char nc = sentence[vd->name_length];
          if (nc == ' ' || nc == '\0') {
            best = vd;
            best_len = vd->name_length;
          }
        }
      }
    }
  }

  if (!best) {
    compiled->vm_result = RESULT_CORE_VERB_ERROR_NO_MATCH;
    return compiled;
  }

  /* Check: verb matched but params expected and none provided */
  bool missing_vals = (sentence_length == best_len && best->params_count > 0);
  if (missing_vals) {
    compiled->vm_result = RESULT_CORE_VERB_ERROR_INCORRECT_USAGE;
    compiled->verb_definition_ = best;
    return compiled;
  }

  /* ── Parse arguments ── */
  bool parse_error = false;
  char *sent_clone = vm_string_clone(sentence);
  char *sent_idx = sent_clone + best_len + 1; /* skip verb + space */

  vm_map *arg_list = vm_map_create();
  vm_map_set(arg_list, "verb_context", ctx);

  char *param_start = sent_idx;

  typedef enum {
    VP_UNKNOWN,
    VP_DEFAULT,
    VP_JSON_OBJ,
    VP_JSON_ARR,
    VP_STR_SQ,
    VP_STR_DQ,
    VP_XML
  } vp_type;

  /* Iterate over expected parameter names */
  vm_list_foreach_of(best->param_list, char *, pname) {
    size_t exit_depth = 0;
    char exit_tok = 0;

    const size_t xml_max = VERB_MAX_EXIT_TOKENS;
    char *xml_tag_starts[VERB_MAX_EXIT_TOKENS];
    char *xml_tag_ends[VERB_MAX_EXIT_TOKENS];
    bool xml_tag_short[VERB_MAX_EXIT_TOKENS];
    memset(xml_tag_starts, 0, sizeof(xml_tag_starts));
    memset(xml_tag_ends, 0, sizeof(xml_tag_ends));
    memset(xml_tag_short, 0, sizeof(xml_tag_short));

    vp_type ptype = VP_UNKNOWN;
    bool param_done = false;
    bool is_delim = false;
    bool is_eol = false;
    bool in_sq = false, in_dq = false;
    char c = 0;

    do {
      const char c_prev = c;
      c = *sent_idx;
      const char c_next = *(sent_idx + 1);

      is_eol = (c == '\0');
      const bool is_esc = (c_prev == '\\');

      if (!is_eol) {
        if (ptype == VP_UNKNOWN) {
          switch (c) {
          case '\'':
            exit_tok = '\'';
            ptype = VP_STR_SQ;
            break;
          case '"':
            exit_tok = '"';
            ptype = VP_STR_DQ;
            break;
          case '[':
            exit_tok = ']';
            ptype = VP_JSON_ARR;
            break;
          case '{':
            exit_tok = '}';
            ptype = VP_JSON_OBJ;
            break;
          case '<':
            exit_tok = '<';
            ptype = VP_XML;
            xml_tag_starts[exit_depth] = sent_idx + 1;
            break;
          default:
            exit_tok = ' ';
            ptype = VP_DEFAULT;
            break;
          }
        } else {
          switch (ptype) {
          case VP_STR_SQ:
            param_done = !is_esc && c == '\'';
            break;
          case VP_STR_DQ:
            param_done = !is_esc && c == '"';
            break;
          case VP_JSON_ARR:
            switch (c) {
            case '\'':
              if (!is_esc && !in_dq)
                in_sq = !in_sq;
              break;
            case '"':
              if (!is_esc && !in_sq)
                in_dq = !in_dq;
              break;
            case '[':
              if (!in_sq && !in_dq)
                exit_depth++;
              break;
            case ']':
              if (!in_sq && !in_dq) {
                if (exit_depth > 0)
                  exit_depth--;
                else
                  param_done = (exit_tok == c && exit_depth == 0);
              }
              break;
            }
            break;
          case VP_JSON_OBJ:
            switch (c) {
            case '\'':
              if (!is_esc && !in_dq)
                in_sq = !in_sq;
              break;
            case '"':
              if (!is_esc && !in_sq)
                in_dq = !in_dq;
              break;
            case '{':
              if (!in_sq && !in_dq)
                exit_depth++;
              break;
            case '}':
              if (!in_sq && !in_dq) {
                if (exit_depth > 0)
                  exit_depth--;
                else
                  param_done = (exit_tok == c && exit_depth == 0);
              }
              break;
            }
            break;
          case VP_XML:
            if (!in_sq && !in_dq) {
              if (c == exit_tok && exit_tok == '<' && c_next != '/') {
                if (exit_depth >= xml_max) {
                  parse_error = true;
                  break;
                }
                xml_tag_starts[++exit_depth] = sent_idx + 1;
              } else if (c == exit_tok && exit_tok == '<' && c_next == '/') {
                exit_tok = '>';
                xml_tag_ends[exit_depth] = sent_idx + 2;
              } else if (c == '/' && c_next == '>') {
                xml_tag_short[exit_depth] = true;
                exit_tok = '>';
              } else if (c == exit_tok && exit_tok == '>') {
                bool tag_short = xml_tag_short[exit_depth];
                const char *tag_start = xml_tag_starts[exit_depth];
                const char *tag_end = xml_tag_ends[exit_depth];
                size_t tag_len = (size_t)(sent_idx - tag_end);

                if (!tag_short && (!tag_start || !tag_end ||
                                   strncmp(tag_start, tag_end, tag_len) != 0)) {
                  parse_error = true;
                  break;
                }

                if (exit_depth > 0)
                  exit_depth--;
                else
                  param_done = (exit_tok == c && exit_depth == 0);
                exit_tok = '<';
              }
            }
            break;
          case VP_DEFAULT:
            is_delim = (c == ' ');
            break;
          default:
            break;
          }
        }
      }

      if (param_done || is_delim || is_eol) {
        /* Depth didn't unwind? */
        if (is_eol && exit_depth > 0) {
          parse_error = true;
          break;
        }
        /* Structured param not terminated? */
        if ((ptype == VP_STR_SQ || ptype == VP_STR_DQ || ptype == VP_JSON_ARR ||
             ptype == VP_JSON_OBJ || ptype == VP_XML) &&
            !param_done) {
          parse_error = true;
          break;
        }

        char *param_end = sent_idx;

        if (!is_eol) {
          if (is_delim)
            *param_end = '\0';
          if (param_done) {
            param_end++;
            *param_end = '\0';
            if (c_next != '\0')
              sent_idx++;
          }
        }

        /* Strip quotes from string params */
        if (ptype == VP_STR_SQ || ptype == VP_STR_DQ) {
          param_start++;
          param_end--;
          *param_end = '\0';
        }

        vm_map_set(arg_list, pname, param_start);

        if (is_eol)
          break;

        if (c_next != '\0')
          sent_idx++;

        param_start = sent_idx;
        break; /* next parameter */
      }

      sent_idx++;
    } while (!is_eol && !parse_error);

    if (is_eol || parse_error)
      break;
  }

  /* Handle parse errors */
  if (parse_error) {
    vm_map_free(arg_list);
    vm_string_free(sent_clone);
    compiled->vm_result = RESULT_CORE_VERB_ERROR_ARGUMENT_PARSING;
    compiled->verb_definition_ = best;
    return compiled;
  }

  /* Check parameter count (subtract 1 for the verb_context entry) */
  const size_t expected = best->params_count;
  const size_t actual = arg_list->count > 0 ? arg_list->count - 1 : 0;
  missing_vals = (expected != actual);

  if (missing_vals) {
    vm_map_free(arg_list);
    vm_string_free(sent_clone);
    compiled->vm_result = RESULT_CORE_VERB_ERROR_INCORRECT_USAGE;
    compiled->verb_definition_ = best;
    return compiled;
  }

  compiled->vm_result = VM_OK;
  compiled->verb_definition_ = best;
  compiled->verb_arg_list = arg_list;
  compiled->sentence_values = sent_clone;
  return compiled;
}

void verb_compile_dispose(verb_compiled *compiled) {
  if (!compiled)
    return;
  if (compiled->sentence_values)
    vm_string_free(compiled->sentence_values);
  if (compiled->verb_arg_list)
    vm_map_free(compiled->verb_arg_list);
  VM_FREE(compiled);
}

int verb_exec(verb_context *ctx, const char *sentence, char **response) {
  if (!ctx || !sentence || !response)
    return RESULT_CORE_VERB_ERROR_ARGUMENT_NULL;

  char *result_response = NULL;

  verb_compiled *compiled = verb_compile(ctx, sentence, NULL);
  int r = compiled->vm_result;

  if (r == RESULT_CORE_VERB_ERROR_INCORRECT_USAGE &&
      compiled->verb_definition_) {
    /* Build usage message: "Usage: verb_name <p1> <p2>" */
    char buf[VM_SENTENCE_MESSAGE_MAX_SIZE];
    snprintf(buf, sizeof(buf), "Usage: %s %s", compiled->verb_definition_->name,
             compiled->verb_definition_->definition_params
                 ? compiled->verb_definition_->definition_params
                 : "");
    result_response = vm_string_clone(buf);
  } else if (r == VM_OK && compiled->verb_definition_) {
    result_response = compiled->verb_definition_->function(
        ctx, compiled->verb_arg_list, compiled->verb_definition_->fcontext);

    if (result_response && strncmp(result_response, "ERROR:", 6) == 0) {
      r = RESULT_CORE_VERB_ERROR;
    }
  }

  verb_compile_dispose(compiled);
  *response = result_response;
  return r;
}

const char *verb_result(int code) {
  switch (code) {
  case VM_OK:
    return "VM_OK";
  case RESULT_CORE_VERB_ERROR:
    return "RESULT_CORE_VERB_ERROR";
  case RESULT_CORE_VERB_ERROR_ALREADY_REGISTERED:
    return "RESULT_CORE_VERB_ERROR_ALREADY_REGISTERED";
  case RESULT_CORE_VERB_ERROR_NO_MATCH:
    return "RESULT_CORE_VERB_ERROR_NO_MATCH";
  case RESULT_CORE_VERB_ERROR_ARGUMENT_NULL:
    return "RESULT_CORE_VERB_ERROR_ARGUMENT_NULL";
  case RESULT_CORE_VERB_ERROR_INCORRECT_USAGE:
    return "RESULT_CORE_VERB_ERROR_INCORRECT_USAGE";
  case RESULT_CORE_VERB_ERROR_ARGUMENT_PARSING:
    return "RESULT_CORE_VERB_ERROR_ARGUMENT_PARSING";
  default:
    return "<UNKNOWN>";
  }
}

/* ═══════════════════════════════════════════════════════════════════════════
 *  Module  —  compilation unit containing functions
 * ═══════════════════════════════════════════════════════════════════════════
 */

/**
 * Allocate and initialise a new compilation unit.
 *
 * @param verb_context  Optional verb dispatch table for native functions.
 *                      May be NULL if no native calls are registered.
 * @return Heap-allocated module.  Caller must call vm_module_dispose().
 */
vm_module *vm_module_create(verb_context *verb_context) {
  vm_module *module = VM_NEW(vm_module);
  module->verb_context_ = verb_context;
  module->functions = vm_map_create();
  module->errors = vm_list_create();
  module->functions_list = vm_list_create();
  return module;
}

/** Free a module and all resources it owns (functions, errors, etc). */
void vm_module_dispose(vm_module *module) {
  VM_ASSERT_NOT_NULL(module);

  vm_map_foreach_of(module->functions, name, vm_function *, function) {
    vm_function_dispose(function);
  }
  vm_map_free(module->functions);

  vm_list_foreach_of(module->errors, vm_module_error *, error) {
    vm_module_error_dispose(error);
  }
  vm_list_free(module->errors);
  vm_list_free(module->functions_list);

  VM_FREE(module);
}

/**
 * Look up a function by name.
 *
 * @return Pointer to the function, or NULL if not found.
 */
vm_function *vm_module_fetch_function(vm_module *module, const char *name) {
  VM_ASSERT_NOT_NULL(module);
  VM_ASSERT_NOT_NULL(name);

  vm_function *function = NULL;
  vm_map_item *item = vm_map_get_item(module->functions, name);

  if (item) {
    function = (vm_function *)item->value;
  }

  return function;
}

/**
 * Render the entire module's IL as a heap-allocated string.
 * Includes instruction dumps for every function plus any error messages.
 * Caller owns the returned string — free with vm_string_free().
 */
char *vm_module_to_string(vm_module *module) {
  VM_ASSERT_NOT_NULL(module);

  vm_string_buffer *buffer = vm_string_buffer_create_empty();

  vm_list_foreach_of(module->functions_list, vm_function *, function) {
    vm_string_buffer_append(
        buffer,
        "\n// "
        "------------------------------------------------------------\n");
    vm_string_buffer_append_format(buffer, "// %s(", function->name);
    bool more_params = false;
    vm_list_foreach_of(function->parameters, char *, parameter) {
      if (more_params) {
        vm_string_buffer_append(buffer, ", ");
      }

      vm_string_buffer_append_format(buffer, "%s", parameter);
      more_params = true;
    }
    vm_string_buffer_append(
        buffer,
        ")\n// "
        "------------------------------------------------------------\n");

    size_t pos = 0;

    vm_list_foreach_of(function->instructions, vm_instruction *, instruction) {
      if (instruction->opcode == opCOMMENT) {
        vm_string_buffer_append(buffer, "\n");
      }

      char *instruction_dump = vm_instruction_to_string(instruction);
      vm_string_buffer_append_format(buffer, "%3zu)\t%s\n", pos++,
                                     instruction_dump);
      vm_string_free(instruction_dump);
    }
  }

  if (vm_list_count(module->errors)) {
    vm_string_buffer_append(buffer,
                            "\n// ** ERRORS "
                            "************************************************"
                            "******************\n");
    vm_list_foreach_of(module->errors, vm_module_error *, error) {
      vm_string_buffer_append_format(buffer, "- Line %d: %s\n",
                                     error->source_line_number, error->message);
    }
  }

  return vm_string_buffer_free_not_data(buffer);
}

/** Record a parse/compile error with source location for later reporting. */
void vm_module_error_add(vm_module *module, size_t source_line_number,
                         size_t source_line_column, char *message) {
  VM_ASSERT_NOT_NULL(module);
  VM_ASSERT_NOT_NULL(module->errors);

  vm_module_error *verify_result_error = VM_NEW(vm_module_error);
  verify_result_error->source_line_number = source_line_number;
  verify_result_error->source_line_column = source_line_column;
  verify_result_error->message = vm_string_clone(message);
  vm_list_add(module->errors, verify_result_error);
}

/** Free a single module error record and its message string. */
void vm_module_error_dispose(vm_module_error *error) {
  VM_ASSERT_NOT_NULL(error);

  if (error->message) {
    vm_string_free(error->message);
  }

  VM_FREE(error);
}

#include "microgpt_vm_parser.tab.h"

/**
 * Compile source code into a module (thin wrapper around parser_generate).
 *
 * @param verb_context_  Optional verb dispatch table for native calls.
 * @param source         Null-terminated source string.
 * @param out_module     On success, receives a heap-allocated vm_module.
 * @return VM_OK on success.
 */
vm_result vm_module_compile(verb_context *verb_context_, const char *source,
                            vm_module **out_module) {
  VM_ASSERT_NOT_NULL(source);
  VM_ASSERT_NOT_NULL(out_module);

  return vm_module_parser_generate(verb_context_, source, out_module);
}

void _vm_module_generator_function_emit(vm_module_generator *generator,
                                        vm_instruction_opcode opcode,
                                        char *param1, char *param2,
                                        char *param3);

/**
 * Create a new code generator bound to the given module.
 * The generator maintains the "current function" context and manages
 * deferred code fragment queues for nested constructs (if/else, loops).
 */
vm_module_generator *vm_module_generator_create(vm_module *module) {
  vm_module_generator *generator = VM_NEW(vm_module_generator);
  generator->vm_module_parser_state_current_module = module;
  generator->vm_module_parser_state_current_function = NULL;
  generator->code_fragments = vm_queue_create();
  generator->completed_code_fragments = vm_queue_create();
  return generator;
}

/** Free the generator and its code fragment queues. */
void vm_module_generator_dispose(vm_module_generator *generator) {
  VM_ASSERT_NOT_NULL(generator);

  if (generator->code_fragments) {
    vm_queue_clear(generator->code_fragments);
    vm_queue_free(generator->code_fragments);
  }

  if (generator->completed_code_fragments) {
    vm_queue_clear(generator->completed_code_fragments);
    vm_queue_free(generator->completed_code_fragments);
  }

  VM_FREE(generator);
}

/**
 * Begin a new function definition.  Creates the vm_function, registers it
 * in the module, sets it as the current codegen context, and emits the
 * entry-point LABEL instruction.
 *
 * @return The new function, or NULL if a function with this name exists.
 */
vm_function *vm_module_generator_function_begin(vm_module_generator *generator,
                                                char *name) {
  VM_ASSERT_NOT_NULL(generator);
  VM_ASSERT_NOT_NULL(name);

  vm_module *module = generator->vm_module_parser_state_current_module;
  VM_ASSERT_NOT_NULL(module);
  VM_ASSERT_NOT_NULL(module->functions);

  if (vm_map_contains(module->functions, name)) {
    vm_log_error("Function with the name %s already defined", name);
    return NULL;
  }

  /* Create function and add to module's lookup map + ordered list */
  vm_function *function = vm_function_create(name);
  vm_map_set(generator->vm_module_parser_state_current_module->functions, name,
             function);
  vm_list_add(generator->vm_module_parser_state_current_module->functions_list,
              function);

  /* Set as current context and emit the entry label */
  generator->vm_module_parser_state_current_function = function;
  _vm_module_generator_function_emit(generator, opLABEL, function->name, NULL,
                                     NULL);
  return function;
}

/**
 * Finalise the current function definition.
 * Appends an implicit RETURN if the last instruction is not already one,
 * then switches the codegen context back to "main".
 */
void vm_module_generator_function_end(vm_module_generator *generator) {
  VM_ASSERT_NOT_NULL(generator);
  VM_ASSERT_NOT_NULL(generator->vm_module_parser_state_current_function);

  vm_function *function = generator->vm_module_parser_state_current_function;
  VM_ASSERT_NOT_NULL(function);

  /* Append implicit RETURN if missing */
  vm_instruction *last_instruction =
      vm_list_last_of(function->instructions, vm_instruction);

  if (last_instruction && last_instruction->opcode != opRETURN) {
    _vm_module_generator_function_emit(generator, opRETURN, NULL, NULL, NULL);
  }

  /* Switch codegen context back to main */
  generator->vm_module_parser_state_current_function =
      (vm_function *)vm_map_get_value(
          generator->vm_module_parser_state_current_module->functions, "main");
}

/** Register a typed parameter for the current function being compiled. */
void vm_module_generator_function_parameter(vm_module_generator *generator,
                                            char *name, char *type) {
  VM_ASSERT_NOT_NULL(generator);
  VM_ASSERT_NOT_NULL(name);
  VM_ASSERT_NOT_NULL(type);

  vm_function *function = generator->vm_module_parser_state_current_function;
  VM_ASSERT_NOT_NULL(function);

  vm_function_parameter_add(function, name, type);
}

/**
 * Set the declared return type for a named function.
 * Called by the parser when it sees `: type` after the parameter list.
 */
void vm_module_generator_function_return_type_set(
    vm_module_generator *generator, char *name, char *type) {
  VM_ASSERT_NOT_NULL(generator);
  VM_ASSERT_NOT_NULL(type);

  vm_module *module = generator->vm_module_parser_state_current_module;
  VM_ASSERT_NOT_NULL(module);
  VM_ASSERT_NOT_NULL(module->functions);

  vm_map_item *function_item = vm_map_get_item(module->functions, name);
  if (function_item == NULL) {
    vm_log_error("Function with the name %s doesn't exist", name);
    return;
  }

  vm_function *function = generator->vm_module_parser_state_current_function;
  VM_ASSERT_NOT_NULL(function);

  function->return_type_class =
      vm_variable_param_type_name_to_param_type_class(type);
  function->return_type = vm_string_clone(type);
}

/**
 * Get the declared return type string for a named function.
 * Falls back to verb_context lookup (returns "string" for native verbs).
 *
 * @return Type string (e.g. "number", "string") or NULL if not found.
 */
char *
vm_module_generator_function_return_type_get(vm_module_generator *generator,
                                             char *name) {
  VM_ASSERT_NOT_NULL(generator);
  VM_ASSERT_NOT_NULL(name);

  vm_module *module = generator->vm_module_parser_state_current_module;
  VM_ASSERT_NOT_NULL(module);
  VM_ASSERT_NOT_NULL(module->functions);

  vm_map_item *function_item = vm_map_get_item(module->functions, name);
  if (function_item == NULL && !module->verb_context_) {
    vm_log_error("Function with the name %s doesn't exist", name);
    return NULL;
  } else if (module->verb_context_) {
    if (verb_exists(module->verb_context_, name)) {
      return "string";
    }

    /* Try matching verb name with spaces (underscores → spaces) */
    char *name_with_spaces = vm_string_replace(name, "_", " ");
    bool exists = verb_exists(module->verb_context_, name_with_spaces);
    vm_string_free(name_with_spaces);
    if (exists) {
      vm_string_replace_inplace(name, '_', ' ');
      return "string";
    } else {
      vm_log_error("Verb with the name %s doesn't exist", name);
      return NULL;
    }
  }

  vm_function *function = (vm_function *)function_item->value;
  VM_ASSERT_NOT_NULL(function);

  return function->return_type;
}

/**
 * Internal: emit a single bytecode instruction into the current function.
 * Creates a vm_instruction and appends it to the function's instruction list.
 */
void _vm_module_generator_function_emit(vm_module_generator *generator,
                                        vm_instruction_opcode opcode,
                                        char *param1, char *param2,
                                        char *param3) {
  VM_ASSERT_NOT_NULL(generator);

  vm_function *function = generator->vm_module_parser_state_current_function;
  VM_ASSERT_NOT_NULL(function);

  vm_instruction *instruction = vm_instruction_create_with_meta(
      opcode, param1, param2, param3,
      generator->meta_state_input_line_number_previous);
  vm_list_add(function->instructions, instruction);
}

/**
 * Emit a bytecode instruction with source-level metadata.
 * When meta_source is provided, also emits a COMMENT instruction containing
 * the source text for IL readability / debugging.
 */
void vm_module_generator_function_emit_with_meta(
    vm_module_generator *generator, vm_instruction_opcode opcode, char *param1,
    char *param2, char *param3, char *meta_source, size_t meta_line_number,
    size_t meta_source_index) {
  VM_ASSERT_NOT_NULL(generator);

  vm_function *function = generator->vm_module_parser_state_current_function;
  VM_ASSERT_NOT_NULL(function);

  /* Extract the source fragment that corresponds to this instruction */
  if (meta_source) {
    char buffer[1024];
    memset(buffer, 0, sizeof(buffer));

    // Extract what we have just parsed
    char *start =
        (char *)&meta_source[generator->meta_state_input_index_previous];
    int len = meta_source_index - generator->meta_state_input_index_previous;
    char *end = start + len;
    size_t meta_source_len = vm_string_length(meta_source);
    char *meta_source_end = meta_source + meta_source_len;

    generator->meta_state_input_line_number_previous = meta_line_number;

    bool skip_until_open_brace = false;

    while (end < meta_source_end) {
      len = (size_t)(end - start);

      if (len == 0) {
        break;
      }

      // Trim spaces at the beginning
      if (!skip_until_open_brace && *start == ' ') {
        start++;
        continue;
      }

      // Type def like?
      // :decimal {     var total = amount * rate;
      if (skip_until_open_brace && *start != '{') {
        start++;
        continue;
      }

      skip_until_open_brace = false;

      if (*start == ':') {
        start++;
        skip_until_open_brace = true;
        continue;
      }

      if (*start == ';' || *start == '}' || *start == '{' || *start == '\n') {
        start++;
        continue;
      }

      if (*end != '\n') {
        end++;
        continue;
      }

      vm_string_copy_length(buffer, start, len);

      // Replace all /n inside with spaces
      int i = len;
      char *buffer_scan_index = buffer;

      while (i-- > 0) {
        if (*buffer_scan_index == '\n') {
          *buffer_scan_index = 0;
          break;
        }

        buffer_scan_index++;
      }

      _vm_module_generator_function_emit(
          generator, opCOMMENT,
          (char *)vm_module_generator_symbol_track(
              generator, vm_int_to_string(meta_line_number)),
          (char *)vm_module_generator_symbol_track(generator,
                                                   vm_string_clone(buffer)),
          NULL);
      generator->meta_state_input_line_number_previous = meta_line_number;

      break;
    }
  }

  _vm_module_generator_function_emit(generator, opcode, param1, param2, param3);

  generator->meta_state_input_index_previous = meta_source_index;
}

/** Emit a bytecode instruction without source metadata (convenience wrapper).
 */
void vm_module_generator_function_emit(vm_module_generator *generator,
                                       vm_instruction_opcode opcode,
                                       char *param1, char *param2,
                                       char *param3) {
  vm_module_generator_function_emit_with_meta(generator, opcode, param1, param2,
                                              param3, NULL, 0, 0);
}
/**
 * Emit a COMMENT instruction with source metadata.
 * Skipped if codegen is outside any function definition.
 */
void vm_module_generator_function_emit_comment_with_meta(
    vm_module_generator *generator, char *message, char *meta_source,
    size_t meta_line_number, size_t meta_source_index) {
  if (!generator->vm_module_parser_state_current_function) {
    return;
  }

  _vm_module_generator_function_emit(
      generator, opCOMMENT,
      (char *)vm_module_generator_symbol_track(
          generator, vm_int_to_string(meta_line_number)),
      (char *)vm_module_generator_symbol_track(generator,
                                               vm_string_clone(message)),
      NULL);

  generator->meta_state_input_line_number_previous = meta_line_number;
  generator->meta_state_input_index_previous = meta_source_index;
}

/** Emit a COMMENT instruction without source metadata (convenience wrapper). */
void vm_module_generator_function_emit_comment(vm_module_generator *generator,
                                               char *message) {
  vm_module_generator_function_emit_comment_with_meta(generator, message, NULL,
                                                      0, 0);
}

/**
 * Allocate a new compiler temporary register ("r$0", "r$1", ...).
 * Register names are tracked per-function for later pre-allocation.
 * @return Heap-allocated register name string.
 */
char *vm_module_generator_tmp_register_create(vm_module_generator *generator) {
  VM_ASSERT_NOT_NULL(generator);

  vm_function *function = generator->vm_module_parser_state_current_function;
  VM_ASSERT_NOT_NULL(function);

  const size_t count = vm_list_count(function->register_names);
  char tmp[10];
  snprintf(tmp, sizeof(tmp), "r$%zu", count);

  char *var = vm_string_clone(tmp);
  vm_list_add(function->register_names, var);

  return var;
}

/** Reset register allocation counter for the current function (no-op stub). */
void vm_module_generator_tmp_registers_reset(vm_module_generator *generator) {
  VM_ASSERT_NOT_NULL(generator);
}

/**
 * Create a new jump label ("L0", "L1", ...) for the current function.
 * Label names are tracked per-function for jump resolution.
 * @return Heap-allocated label name string.
 */
char *vm_module_generator_tmp_label_create(vm_module_generator *generator) {
  VM_ASSERT_NOT_NULL(generator);

  vm_function *function = generator->vm_module_parser_state_current_function;
  VM_ASSERT_NOT_NULL(function);

  const size_t count = vm_list_count(function->label_names);
  char tmp[10];
  VM_SPRINTF_S(tmp, sizeof(tmp), "L%zu", count);

  char *label = vm_string_clone(tmp);
  vm_list_add(function->label_names, label);

  return label;
}

/** Push a label name onto the tracking stack (used for nested if/while/for). */
void vm_module_generator_tracking_labels_push(vm_module_generator *generator,
                                              char *label) {
  VM_ASSERT_NOT_NULL(generator);

  vm_function *function = generator->vm_module_parser_state_current_function;
  VM_ASSERT_NOT_NULL(function);

  vm_queue_push(function->tracking_labels, label);
}

/** Pop and return the innermost label from the tracking stack. */
char *vm_module_generator_tracking_labels_pop(vm_module_generator *generator) {
  VM_ASSERT_NOT_NULL(generator);

  vm_function *function = generator->vm_module_parser_state_current_function;
  VM_ASSERT_NOT_NULL(function);
  VM_ASSERT(!vm_queue_is_empty(function->tracking_labels));

  return (char *)vm_queue_pop(function->tracking_labels);
}

/** Clear all labels from the tracking stack (used during generator teardown).
 */
void vm_module_generator_tracking_labels_clear(vm_module_generator *generator) {
  VM_ASSERT_NOT_NULL(generator);

  vm_function *function = generator->vm_module_parser_state_current_function;
  VM_ASSERT_NOT_NULL(function);

  vm_list_dispose_items(function->label_names);
  vm_queue_dispose_items(function->tracking_labels);
}

/**
 * Begin capturing instructions into a deferred code fragment.
 * Used for FOR loops: the increment expression must be emitted after
 * the loop body, but is parsed before it.  Pushes the current function
 * context onto a stack and creates a temporary capture function.
 */
void vm_module_generator_defer_push_begin(vm_module_generator *generator) {
  VM_ASSERT_NOT_NULL(generator);

  vm_function *current_code_fragment =
      generator->vm_module_parser_state_current_function;
  VM_ASSERT_NOT_NULL(current_code_fragment);

  /* Save the current context */
  vm_queue_push(generator->code_fragments, current_code_fragment);

  /* Create a temporary function to capture deferred instructions */
  char fragment_id[128];
  VM_SPRINTF_S(fragment_id, sizeof(fragment_id), "_deferred_code_fragment_L%zu",
               generator->meta_state_input_index_previous);
  vm_function *new_code_fragment =
      vm_function_create(vm_string_clone(fragment_id));
  generator->vm_module_parser_state_current_function = new_code_fragment;
}

/**
 * Stop capturing deferred instructions and stash the fragment.
 * Moves the completed fragment to the completed vm_queue and resumes
 * the previous codegen context.
 */
void vm_module_generator_defer_push_end(vm_module_generator *generator) {
  VM_ASSERT_NOT_NULL(generator);

  vm_function *current_code_fragment =
      generator->vm_module_parser_state_current_function;
  VM_ASSERT_NOT_NULL(current_code_fragment);

  if (vm_queue_is_empty(generator->code_fragments)) {
    VM_ASSERT(false);
    return;
  }

  /* Stash completed fragment */
  vm_queue_push(generator->completed_code_fragments, current_code_fragment);

  /* Resume previous context */
  vm_function *previous_code_fragment =
      (vm_function *)vm_queue_pop(generator->code_fragments);
  generator->vm_module_parser_state_current_function = previous_code_fragment;
}

/**
 * Flush the most recently completed deferred code fragment into
 * the current function.  Merges all instructions, variables, constants,
 * registers, symbols, labels, and type traits, then disposes the fragment.
 */
void vm_module_generator_defer_pop(vm_module_generator *generator) {
  VM_ASSERT_NOT_NULL(generator);

  if (vm_queue_is_empty(generator->completed_code_fragments)) {
    VM_ASSERT(false);
    return;
  }

  /* Merge all collections from the fragment into the current function */
  vm_function *last_completed_code_fragment =
      (vm_function *)vm_queue_pop(generator->completed_code_fragments);
  vm_list_merge(
      generator->vm_module_parser_state_current_function->instructions,
      last_completed_code_fragment->instructions);
  vm_list_merge(generator->vm_module_parser_state_current_function->registers,
                last_completed_code_fragment->registers);
  vm_map_merge(generator->vm_module_parser_state_current_function->variables,
               last_completed_code_fragment->variables);
  vm_map_merge(generator->vm_module_parser_state_current_function->constants,
               last_completed_code_fragment->constants);
  vm_map_merge(generator->vm_module_parser_state_current_function->trait_types,
               last_completed_code_fragment->trait_types);
  vm_list_merge(generator->vm_module_parser_state_current_function->symbols,
                last_completed_code_fragment->symbols);
  vm_queue_merge(
      generator->vm_module_parser_state_current_function->tracking_labels,
      last_completed_code_fragment->tracking_labels);
  vm_list_merge(generator->vm_module_parser_state_current_function->label_names,
                last_completed_code_fragment->label_names);
  vm_list_merge(
      generator->vm_module_parser_state_current_function->register_names,
      last_completed_code_fragment->register_names);
  vm_map_merge(generator->vm_module_parser_state_current_function->trait_types,
               last_completed_code_fragment->trait_types);
  vm_list_merge(generator->vm_module_parser_state_current_function->registers,
                last_completed_code_fragment->registers);
  vm_map_merge(generator->vm_module_parser_state_current_function->constants,
               last_completed_code_fragment->constants);
  vm_map_merge(generator->vm_module_parser_state_current_function->variables,
               last_completed_code_fragment->variables);

  /* Clean up the disposable fragment */
  vm_list_clear(last_completed_code_fragment->instructions);
  vm_list_clear(last_completed_code_fragment->symbols);
  vm_queue_clear(last_completed_code_fragment->tracking_labels);
  vm_list_clear(last_completed_code_fragment->label_names);
  vm_list_clear(last_completed_code_fragment->register_names);
  vm_map_clear(last_completed_code_fragment->trait_types);
  vm_list_clear(last_completed_code_fragment->registers);
  vm_map_clear(last_completed_code_fragment->constants);
  vm_map_clear(last_completed_code_fragment->variables);

  vm_function_dispose(last_completed_code_fragment);
}

/**
 * Track a symbol string's lifetime so it is freed with the function.
 * If the symbol is already tracked, returns it as-is (deduplication).
 */
const char *vm_module_generator_symbol_track(vm_module_generator *generator,
                                             const char *symbol) {
  vm_function *function = generator->vm_module_parser_state_current_function;
  VM_ASSERT_NOT_NULL(function);

  if (!vm_list_contains(function->symbols, (void *)symbol)) {
    vm_list_add(function->symbols, (void *)symbol);
  }

  return symbol;
}

/**
 * Associate a type trait (type string + is_constant flag) with a symbol.
 * If the symbol already has a trait, logs a warning on type mismatch.
 */
void vm_module_generator_trait_type_set(vm_module_generator *generator,
                                        char *symbol, char *type,
                                        bool is_constant) {
  VM_ASSERT_NOT_NULL(generator);
  VM_ASSERT_NOT_NULL(symbol);

  if (!type) {
    vm_log_warn("Attempt to set symbol %s type to NULL", symbol);
    return;
  }

  vm_module *module = generator->vm_module_parser_state_current_module;
  VM_ASSERT_NOT_NULL(module);

  vm_function *function = generator->vm_module_parser_state_current_function;
  VM_ASSERT_NOT_NULL(function);

  /* Check for existing trait and VM_ASSERT type consistency */
  if (vm_map_contains(function->trait_types, symbol)) {
    vm_type_trait *type_trait =
        (vm_type_trait *)vm_map_get_value(function->trait_types, symbol);

    if (!vm_string_equals(type_trait->type, type)) {
      char error[1024];
      snprintf(error, sizeof(error),
               "Cannot assign value of type %s to %s which is already "
               "of type %s",
               type, symbol, type_trait->type);
      vm_log_warn(error, NULL);
    }

    return;
  }

  vm_type_trait *type_trait = vm_type_trait_create(type, is_constant);
  vm_map_set(function->trait_types, symbol, type_trait);
}

/**
 * Look up the type trait for a symbol and return its type string.
 * Logs a warning if the symbol has no trait registered.
 *
 * @return Type string (e.g. "number") or NULL if not found.
 */
char *vm_module_generator_trait_type_get(vm_module_generator *generator,
                                         char *symbol) {
  VM_ASSERT_NOT_NULL(generator);
  VM_ASSERT_NOT_NULL(symbol);

  vm_module *module = generator->vm_module_parser_state_current_module;
  VM_ASSERT_NOT_NULL(module);

  vm_function *function = generator->vm_module_parser_state_current_function;
  VM_ASSERT_NOT_NULL(function);

  if (!vm_map_contains(function->trait_types, symbol)) {
    char error[1024];
    snprintf(error, sizeof(error), "Type trait for %s not found", symbol);
    vm_log_warn(error, NULL);
    return NULL;
  }

  vm_type_trait *type_trait =
      (vm_type_trait *)vm_map_get_value(function->trait_types, symbol);
  return type_trait->type;
}

/* convert.h removed */

/**
 * Post-processing pass 1: Build jump tables.
 * Scans each function for LABEL instructions, maps their positions,
 * then annotates JUMP/JUMP_IF_FALSE/JUMP_IF_TRUE instructions with
 * pre-computed target positions for O(1) dispatch at runtime.
 */
void _vm_module_generator_post_processing_calculate_jump_table(
    vm_module *module) {
  VM_ASSERT_NOT_NULL(module);
  VM_ASSERT_NOT_NULL(module->functions);

  vm_map_foreach_of(module->functions, name, vm_function *, function) {
    // We now need to create a jump table from any labels
    vm_map *jump_table = vm_map_create();
    size_t pos = 0;
    vm_list_foreach_of(function->instructions, vm_instruction *,
                       instruction_label_search) {
      if (instruction_label_search->opcode == opLABEL) {
        vm_map_set(jump_table, instruction_label_search->param1,
                   (void *)pos); // TODO: hold scalar?
      }

      pos++;
    }

    // Annotate any jumps with label with jump table positions
    vm_list_foreach_of(function->instructions, vm_instruction *, instruction) {
      if (instruction->opcode == opJUMP ||
          instruction->opcode == opJUMP_IF_FALSE ||
          instruction->opcode == opJUMP_IF_TRUE) {
        // Find label in jump table
        if (!vm_map_contains(jump_table, instruction->param3)) {
          // TODO: this is fatal and we need to bomb out
          vm_log_error("Jump label doesn't exist: %s", instruction->param3);
        } else {
          // Get the jump table location
          const size_t address =
              (size_t)vm_map_get_value(jump_table, instruction->param3);
          instruction->opJUMP_jump_to_instruction_pos =
              address == 0 ? -1 : address;
        }
      }

      pos++;
    }

    vm_map_free(jump_table);
  }
}

/**
 * Post-processing pass 2: Decorate instruction parameters with type traits.
 * For each instruction param, looks up the symbol in the function's
 * trait_types map and copies type class, is_constant, is_register, and
 * register_id onto the instruction for fast runtime dispatch.
 */
void _vm_post_processing_instruction_param_decorate_type_traits(
    vm_module *module) {
  VM_ASSERT_NOT_NULL(module);
  VM_ASSERT_NOT_NULL(module->functions);

  vm_map_foreach_of(module->functions, name, vm_function *, function) {
    // Add type traits to params
    vm_list_foreach_of(function->instructions, vm_instruction *, instruction) {
      if (instruction->param1 &&
          vm_map_contains(function->trait_types, instruction->param1)) {
        vm_type_trait *type_trait = (vm_type_trait *)vm_map_get_value(
            function->trait_types, instruction->param1);
        instruction->param_type_1 = type_trait->type;
        instruction->param_is_constant_1 = type_trait->is_constant;
        instruction->param_is_register_1 = !type_trait->is_constant &&
                                           instruction->param1[0] == 'r' &&
                                           instruction->param1[1] == '$';
        instruction->param_register_id_1 =
            !instruction->param_is_register_1
                ? 0
                : vm_string_to_int(&instruction->param1[2]);
      }

      if (instruction->param2 &&
          vm_map_contains(function->trait_types, instruction->param2)) {
        vm_type_trait *type_trait = (vm_type_trait *)vm_map_get_value(
            function->trait_types, instruction->param2);
        instruction->param_type_2 = type_trait->type;
        instruction->param_is_constant_2 = type_trait->is_constant;
        instruction->param_is_register_2 = !type_trait->is_constant &&
                                           instruction->param2[0] == 'r' &&
                                           instruction->param2[1] == '$';
        instruction->param_register_id_2 =
            !instruction->param_is_register_2
                ? 0
                : vm_string_to_int(&instruction->param2[2]);
      }

      if (instruction->param3 &&
          vm_map_contains(function->trait_types, instruction->param3)) {
        vm_type_trait *type_trait = (vm_type_trait *)vm_map_get_value(
            function->trait_types, instruction->param3);
        instruction->param_type_3 = type_trait->type;
        instruction->param_is_constant_3 = type_trait->is_constant;
        instruction->param_is_register_3 = !type_trait->is_constant &&
                                           instruction->param3[0] == 'r' &&
                                           instruction->param3[1] == '$';
        instruction->param_register_id_3 =
            !instruction->param_is_register_3
                ? 0
                : vm_string_to_int(&instruction->param3[2]);
      }
    }
  }
}

/**
 * Post-processing pass 3: Resolve function call return types.
 * Scans all instructions — when a param type matches a function name,
 * replaces it with that function's declared return type.  Also classifies
 * each param_type into its enum (ptcNUMBER, ptcSTRING, etc.) for the runtime.
 */
void _vm_post_processing_function_return_decorate_type_traits(
    vm_module *module) {
  VM_ASSERT_NOT_NULL(module);
  VM_ASSERT_NOT_NULL(module->functions);

  vm_map_foreach_of(module->functions, name, vm_function *, function) {
    vm_map_foreach_of(module->functions, scanned_name, vm_function *,
                      scanned_function) {
      vm_list_foreach_of(scanned_function->instructions, vm_instruction *,
                         scanned_instruction) {
        // Look for types which are this function's name, replace with
        // this functions type name
        if (scanned_instruction->param_type_1 &&
            vm_string_equals(scanned_instruction->param_type_1,
                             function->name)) {
          scanned_instruction->param_type_1 = function->return_type;
        }

        if (scanned_instruction->param_type_2 &&
            vm_string_equals(scanned_instruction->param_type_2,
                             function->name)) {
          scanned_instruction->param_type_2 = function->return_type;
        }

        if (scanned_instruction->param_type_3 &&
            vm_string_equals(scanned_instruction->param_type_3,
                             function->name)) {
          scanned_instruction->param_type_3 = function->return_type;
        }

        // If function's name, is in the call method, then put save the
        // return type
        if (scanned_instruction->opcode == opCALL_METHOD &&
            !scanned_instruction->param_type_1 &&
            vm_string_equals(scanned_instruction->param1, function->name)) {
          scanned_instruction->param_type_1 = function->return_type;
        }

        scanned_instruction->param_type_class_1 =
            !scanned_instruction->param_type_1
                ? ptcNONE
                : vm_variable_param_type_name_to_param_type_class(
                      scanned_instruction->param_type_1);
        scanned_instruction->param_type_class_2 =
            !scanned_instruction->param_type_2
                ? ptcNONE
                : vm_variable_param_type_name_to_param_type_class(
                      scanned_instruction->param_type_2);
        scanned_instruction->param_type_class_3 =
            !scanned_instruction->param_type_3
                ? ptcNONE
                : vm_variable_param_type_name_to_param_type_class(
                      scanned_instruction->param_type_3);
      }
    }
  }
}

/**
 * Resolve a single instruction parameter to a vm_variable.
 * Handles three cases:
 *   - Register (r$N):  fetches from the pre-allocated register array.
 *   - Constant (literal): lazy-creates in the function's constants map.
 *   - Variable (named):  lazy-creates in the function's variables map.
 * The resolved variable pointer is written to *out_var.
 *
 * @return true on success, false if assignment/lookup failed.
 */
bool vm_variable_fetch_by_instruction_param(
    vm_function *function, vm_instruction_opcode opcode, char *param,
    char *param_type, vm_param_type_class param_type_class,
    bool param_is_constant, bool param_is_register, size_t param_register_id,
    vm_variable **out_var) {
  VM_ASSERT_NOT_NULL(function);
  VM_ASSERT_NOT_NULL(out_var);

  if (opcode != opLABEL && opcode != opCOMMENT && opcode != opCALL_METHOD &&
      opcode != opCALL_OBJ_METHOD && opcode != opJUMP) {
    if (param && param_type) {
      if (param_is_register) {
        // Fetch register value
        VM_ASSERT_NOT_NULL(function->registers);
        vm_variable *reg = (vm_variable *)vm_list_get_value(function->registers,
                                                            param_register_id);

        if (reg->type_class == ptcNONE) {
          if (!vm_variable_value_assign_constant(param_type_class, param,
                                                 &reg->value)) {
            vm_log_error("Failed to assign constant", NULL);
            return false;
          }

          reg->type_class = param_type_class;
          reg->is_register = true;
        }

        *out_var = reg;
      } else if (param_is_constant) {
        VM_ASSERT_NOT_NULL(function->constants);
        vm_variable *var;

        if (vm_map_contains(function->constants, param)) {
          var = (vm_variable *)vm_map_get_value(function->constants, param);
        } else {
#ifdef VM_ENABLE_VAR_NAME
          var = vm_variable_create(param);
#else
          var = vm_variable_create();
#endif
          var->type_class = param_type_class;
          var->is_constant = true;

          if (!vm_variable_value_assign_constant(var->type_class, param,
                                                 &var->value)) {
            vm_log_error("Failed to assign variable", NULL);
            return false;
          }

          vm_map_set(function->constants, param, var);
        }

        *out_var = var;

      } else {
        // Fetch from variable
        VM_ASSERT_NOT_NULL(function->variables);
        vm_variable *var;

        if (vm_map_contains(function->variables, param)) {
          var = (vm_variable *)vm_map_get_value(function->variables, param);
        } else {
#ifdef VM_ENABLE_VAR_NAME
          var = vm_variable_create(param);
#else
          var = vm_variable_create();
#endif
          var->type_class = param_type_class;
          var->value.string = NULL;
          var->is_constant = false;
          var->is_register = false;

          vm_map_set(function->variables, param, var);
          vm_list_add(function->variables_list, vm_string_clone(param));
        }

        *out_var = var;
        /* #ifdef _DEBUG_TRACE
                        vm_string_buffer* sb =
        vm_string_buffer_create_empty(); vm_variable_to_vm_string_buffer(sb,
        param, var); printf("Recall %s\n", vm_string_buffer_get(sb));
                        vm_string_buffer_free(sb);
        #endif*/
      }
    }
  }

  return true;
}

/**
 * Post-processing pass 4 (per-function): Pre-allocate registers and
 * resolve all instruction params to vm_variable pointers.
 * After this pass, each instruction's var1/var2/var3 fields are set,
 * avoiding map lookups at runtime.
 */
bool _vm_post_processing_precalculate_reg_var_const_fn(vm_function *function) {
  // TODO: set on instructions the var/reg/const references
  // Pre gen registers
  size_t register_count = vm_list_count(function->register_names);

  while (register_count-- > 0) {
#ifdef VM_ENABLE_VAR_NAME
    char name[100]; // TODO: check
    snprintf(name, sizeof(name), "r%zu", vm_list_count(function->registers));
    vm_list_add(function->registers, vm_variable_create(name));
#else
    vm_list_add(function->registers, vm_variable_create());
#endif
  }

  vm_list_foreach_of(function->instructions, vm_instruction *, instruction) {
    vm_param_type_class var_type_class_1 = ptcNONE;
    vm_variable_value *var_value_1 = NULL;

    vm_param_type_class var_type_class_2 = ptcNONE;
    vm_variable_value *var_value_2 = NULL;

    vm_param_type_class var_type_class_3 = ptcNONE;
    vm_variable_value *var_value_3 = NULL;

    // TODO: precompute
    if (!vm_variable_fetch_by_instruction_param(
            function, instruction->opcode, instruction->param1,
            instruction->param_type_1, instruction->param_type_class_1,
            instruction->param_is_constant_1, instruction->param_is_register_1,
            instruction->param_register_id_1, &instruction->var1)) {
      vm_log_error("Unable to fetch var for param 1", NULL);
      return false;
    }

    if (!vm_variable_fetch_by_instruction_param(
            function, instruction->opcode, instruction->param2,
            instruction->param_type_2, instruction->param_type_class_2,
            instruction->param_is_constant_2, instruction->param_is_register_2,
            instruction->param_register_id_2, &instruction->var2)) {
      vm_log_error("Unable to fetch var for param 2", NULL);
      return false;
    }

    if (!vm_variable_fetch_by_instruction_param(
            function, instruction->opcode, instruction->param3,
            instruction->param_type_3, instruction->param_type_class_3,
            instruction->param_is_constant_3, instruction->param_is_register_3,
            instruction->param_register_id_3, &instruction->var3)) {
      vm_log_error("Unable to fetch var for param 3", NULL);
      return false;
    }
  }

  return true;
}

/** Post-processing pass 4: apply register/variable pre-calculation to all
 * functions. */
void _vm_post_processing_precalculate_reg_var_const(vm_module *module) {
  VM_ASSERT_NOT_NULL(module);
  VM_ASSERT_NOT_NULL(module->functions);

  vm_map_foreach_of(module->functions, name, vm_function *, function) {
    _vm_post_processing_precalculate_reg_var_const_fn(function);
  }
}

/**
 * Post-processing pass 5: Pre-resolve CALL_METHOD targets.
 * Stores a direct pointer to the target vm_function (or verb_definition)
 * on each CALL_METHOD instruction for O(1) dispatch at runtime.
 */
void _vm_post_processing_call_method_pre_calc(vm_module *module) {
  VM_ASSERT_NOT_NULL(module);
  VM_ASSERT_NOT_NULL(module->functions);

  vm_map_foreach_of(module->functions, name, vm_function *, function) {
    vm_list_foreach_of(function->instructions, vm_instruction *, instruction) {
      if (instruction->opcode == opCALL_METHOD) {
        char *function_name = instruction->param1;
        if (function_name == NULL) {
          vm_log_error("Function cannot be null", NULL);
          return;
        }

        // Find the function
        bool found_function = false;
        if (vm_map_contains(module->functions, function_name)) {
          instruction->call_method_function =
              (vm_function *)vm_map_get_value(module->functions, function_name);
          found_function = true;
        } else if (!module->verb_context_) {
          vm_log_error("Unable to find the function %s(...)", function_name);
          return;
        }

        if (!found_function) {
          verb_definition *verb_definition_ =
              verb_find(module->verb_context_, function_name);
          if (verb_definition_) {
            instruction->call_method_verb = verb_definition_;
          } else {
            vm_log_error("Unable to find the verb %s(...)", function_name);
            return;
          }
        }
      }
    }
  }
}

/**
 * Run all five post-processing passes in order:
 *   1. Jump table calculation
 *   2. Instruction param type decoration
 *   3. Function return type propagation
 *   4. Register/variable/constant pre-allocation
 *   5. CALL_METHOD target pre-resolution
 */
void vm_module_generator_post_processing_process(vm_module *module) {
  VM_ASSERT_NOT_NULL(module);

  _vm_module_generator_post_processing_calculate_jump_table(module);
  _vm_post_processing_instruction_param_decorate_type_traits(module);
  _vm_post_processing_function_return_decorate_type_traits(module);
  _vm_post_processing_precalculate_reg_var_const(module);
  _vm_post_processing_call_method_pre_calc(module);
}
/**
 * Verifier check 1: Ensure no instruction parameter is marked as both
 * constant and register (mutually exclusive flags).
 */
void _vm_module_generator_verifier_check_instruction_params(vm_module *module) {
  VM_ASSERT_NOT_NULL(module);
  VM_ASSERT_NOT_NULL(module->functions);

  char error[1024];

  vm_list_foreach_of(module->functions_list, vm_function *, function) {
    vm_list_foreach_of(function->instructions, vm_instruction *, instruction) {
      if (instruction->param1 && instruction->param_is_constant_1 &&
          instruction->param_is_register_1) {
        snprintf(error, sizeof(error),
                 "param1 cannot be both constant and register type class");
        vm_module_error_add(module, instruction->source_line, 0, error);
      }

      if (instruction->param2 && instruction->param_is_constant_2 &&
          instruction->param_is_register_2) {
        snprintf(error, sizeof(error),
                 "param2 cannot be both constant and register type class");
        vm_module_error_add(module, instruction->source_line, 0, error);
      }

      if (instruction->param3 && instruction->param_is_constant_3 &&
          instruction->param_is_register_3) {
        snprintf(error, sizeof(error),
                 "param3 cannot be both constant and register type class");
        vm_module_error_add(module, instruction->source_line, 0, error);
      }
    }
  }
}

/**
 * Verifier check 2: Validate function call argument counts and types.
 * For each CALL_METHOD instruction, walks backwards to find the
 * preceding STACK_PUSH instructions and compares them against the
 * target function's declared parameter list.
 */
void _vm_module_generator_verifier_check_function_call_params(
    vm_module *module) {
  VM_ASSERT_NOT_NULL(module);
  VM_ASSERT_NOT_NULL(module->functions);

  vm_list_foreach_of(module->functions_list, vm_function *, function) {
    vm_list_foreach_of(module->functions_list, vm_function *,
                       scanned_function) {
      // How many parameters are there into this function?
      const size_t expected_param_count = vm_list_count(function->parameters);

      // No need to look at the args for 0 param functions
      if (expected_param_count == 0) {
        continue;
      }

      vm_list_item *scanned_instructions =
          vm_list_enumerable(scanned_function->instructions);
      const size_t scanned_instructions_count =
          vm_list_count(scanned_function->instructions);

      int scanned_instructions_index = scanned_instructions_count;

      while (scanned_instructions_index-- > 0) {
        vm_instruction *scanned_instruction =
            (vm_instruction *)scanned_instructions[scanned_instructions_index]
                .value;

        if (!scanned_instruction)
          continue;

        if (scanned_instruction->opcode == opCALL_METHOD &&
            vm_string_equals(scanned_instruction->param1, function->name)) {
          // Now fetch the type traits for the arguments being pushed in
          vm_list *scanned_type_args = vm_list_create();

          while (scanned_instructions_index-- > 0) {
            scanned_instruction =
                (vm_instruction *)
                    scanned_instructions[scanned_instructions_index]
                        .value;

            if (!scanned_instruction)
              continue;

            // Is the type on the param set already?
            if (scanned_instruction->opcode == opSTACK_PUSH) {
              vm_list_add(scanned_type_args, scanned_instruction->param_type_1);
            } else {
              break;
            }
          }

          ///////////////////////////////////////////////////////////////////
          // Check arg count
          ///////////////////////////////////////////////////////////////////

          // Enough parameters being passed in?
          if (vm_list_count(scanned_type_args) != expected_param_count) {
            char error[1024];
            snprintf(error, sizeof(error),
                     "Incorrect number of arguments to fn %s: expected %zd, "
                     "but got %zd",
                     function->name, expected_param_count,
                     vm_list_count(scanned_type_args));
            vm_module_error_add(module, scanned_instruction->source_line, 0,
                                error);
          } else {
            // Check params against declared function params
            for (size_t i = 0; i < expected_param_count; i++) {
              vm_list_item *expected =
                  vm_list_get_item(function->parameters, i);
              char *expected_type = expected->type;
              char *expected_name = (char *)expected->value;
              char *actual_type =
                  (char *)vm_list_get_value(scanned_type_args, i);

              if (!actual_type) {
                char error[1024];
                snprintf(error, sizeof(error),
                         "Serious error, parameter arg %zd (%s) of fn %s "
                         "should have a type declared",
                         i + 1, expected_name, function->name);
                vm_module_error_add(module, scanned_instruction->source_line, 0,
                                    error);
                break;
              }

              if (expected_type &&
                  !vm_string_equals(expected_type, actual_type)) {
                char error[1024];
                snprintf(error, sizeof(error),
                         "Incorrect parameter arg %zd (%s) to fn %s in %s: "
                         "expected %s but got %s",
                         i + 1, expected_name, function->name,
                         scanned_function->name, expected_type, actual_type);
                vm_module_error_add(module, scanned_instruction->source_line, 0,
                                    error);
                break;
              }
            }
          }

          vm_list_free(scanned_type_args);
        }
      }
    }
  }
}

/**
 * Verifier check 3: Ensure every CALL_METHOD / CALL_OBJ_METHOD target
 * references an existing function or verb.  Reports an error if the
 * callee was not resolved during post-processing.
 */
void _vm_module_generator_verifier_check_call_function_declared(
    vm_module *module) {
  VM_ASSERT_NOT_NULL(module);
  VM_ASSERT_NOT_NULL(module->functions);

  vm_list_foreach_of(module->functions_list, vm_function *, function) {
    vm_list_foreach_of(function->instructions, vm_instruction *, instruction) {
      if ((instruction->opcode == opCALL_METHOD ||
           instruction->opcode == opCALL_OBJ_METHOD) &&
          instruction->param_type_1 == NULL) {
        if (!instruction->call_method_function &&
            !instruction->call_method_verb) {
          char error[1024];
          snprintf(error, sizeof(error),
                   "Function/Verb %s(...) does not exist in function %s(...)",
                   instruction->param1, function->name);
          vm_module_error_add(module, instruction->source_line, 0, error);
        }
      }
    }
  }
}

/**
 * Verifier check 4: Ensure every variable referenced in an instruction
 * has a declared type trait.  Undeclared variables generate an error.
 */
void _vm_module_generator_verifier_check_call_var_declared(vm_module *module) {
  VM_ASSERT_NOT_NULL(module);
  VM_ASSERT_NOT_NULL(module->functions);

  char error[1024];

  vm_list_foreach_of(module->functions_list, vm_function *, function) {
    vm_list_foreach_of(function->instructions, vm_instruction *, instruction) {
      if (instruction->opcode != opLABEL && instruction->opcode != opCOMMENT &&
          instruction->opcode != opCALL_METHOD &&
          instruction->opcode != opCALL_OBJ_METHOD &&
          instruction->opcode != opJUMP) {
        if (instruction->param1 && !instruction->param_type_1) {
          snprintf(error, sizeof(error),
                   "Undeclared variable %s used in function %s(...)",
                   instruction->param1, function->name);
          vm_module_error_add(module, instruction->source_line, 0, error);
        }

        if (instruction->param2 && !instruction->param_type_2) {
          snprintf(error, sizeof(error),
                   "Undeclared variable %s used in function %s(...)",
                   instruction->param2, function->name);
          vm_module_error_add(module, instruction->source_line, 0, error);
        }

        if (instruction->opcode != opJUMP_IF_FALSE &&
            instruction->opcode != opJUMP_IF_TRUE) {
          if (instruction->param3 && !instruction->param_type_3) {
            snprintf(error, sizeof(error),
                     "Undeclared variable %s used in function %s(...)",
                     instruction->param3, function->name);
            vm_module_error_add(module, instruction->source_line, 0, error);
          }
        }
      }
    }
  }
}

/**
 * Verifier check 5: Validate that the last STACK_PUSH before RETURN
 * matches the function's declared return type.
 */
void _vm_module_generator_verifier_check_function_return_value_type(
    vm_module *module) {
  VM_ASSERT_NOT_NULL(module);
  VM_ASSERT_NOT_NULL(module->functions);

  vm_list_foreach_of(module->functions_list, vm_function *, function) {
    vm_list_item *instructions = vm_list_enumerable(function->instructions);
    const size_t instructions_count = vm_list_count(function->instructions);

    if (instructions_count < 2) {
      continue;
    }

    size_t return_instruction_index = instructions_count - 1;
    size_t stack_push_instruction_index = instructions_count - 2;
    vm_instruction *return_instruction =
        (vm_instruction *)instructions[return_instruction_index].value;

    if (return_instruction->opcode != opRETURN) {
      vm_log_error("Expected RETURN opcode", NULL);
      continue;
    }

    // Are we returning any value? If so check
    vm_instruction *stack_push_instruction =
        (vm_instruction *)instructions[stack_push_instruction_index].value;

    if (stack_push_instruction->opcode == opSTACK_PUSH) {
      char *expected_type = NULL;

      // Check the return types
      if (function->return_type == NULL &&
          stack_push_instruction->param_type_1 &&
          !vm_string_equals(stack_push_instruction->param_type_1, "void")) {
        expected_type = "void";
      } else if (!vm_string_equals(function->return_type,
                                   stack_push_instruction->param_type_1)) {
        expected_type = function->return_type;
      }

      if (expected_type) {
        char error[1024];
        snprintf(error, sizeof(error),
                 "Wrong type %s:%s being returned from function %s(...):%s",
                 stack_push_instruction->param1,
                 stack_push_instruction->param_type_1, function->name,
                 expected_type);
        vm_module_error_add(module, stack_push_instruction->source_line, 0,
                            error);
      }
    }
  }
}

/**
 * Verifier check 6: After a CALL_METHOD, if the return value is popped
 * and assigned to a variable, VM_ASSERT the return type matches the
 * variable's declared type.
 */
void _vm_module_generator_verifier_check_function_return_type_value_assignment_after_call(
    vm_module *module) {
  VM_ASSERT_NOT_NULL(module);
  VM_ASSERT_NOT_NULL(module->functions);

  vm_list_foreach_of(module->functions_list, vm_function *, function) {
    vm_list_item *instructions = vm_list_enumerable(function->instructions);
    const size_t instructions_count = vm_list_count(function->instructions);

    for (size_t instructions_index = 0; instructions_index < instructions_count;
         instructions_index++) {
      vm_instruction *call_method_instruction =
          (vm_instruction *)instructions[instructions_index].value;

      // If there's a function call then check the return type
      if (call_method_instruction->opcode == opCALL_METHOD ||
          call_method_instruction->opcode == opCALL_OBJ_METHOD) {
        // Now check if we pop the stack and set the value
        if (instructions_index + 2 >= instructions_count) {
          continue;
        }

        vm_instruction *stack_pop_instruction =
            (vm_instruction *)instructions[instructions_index + 1].value;

        if (!stack_pop_instruction ||
            stack_pop_instruction->opcode != opSTACK_POP) {
          continue;
        }

        vm_instruction *set_var_instruction =
            (vm_instruction *)instructions[instructions_index + 2].value;

        if (!set_var_instruction || set_var_instruction->opcode != opSET_VAR) {
          continue;
        }

        // Check assignment
        if (!vm_string_equals(call_method_instruction->param_type_1,
                              set_var_instruction->param_type_3)) {
          char error[1024];
          snprintf(error, sizeof(error),
                   "Return of function %s(...):%s cannot be assigned to "
                   "variable %s:%s",
                   call_method_instruction->param1,
                   call_method_instruction->param_type_1,
                   set_var_instruction->param3,
                   set_var_instruction->param_type_3);
          vm_module_error_add(module, call_method_instruction->source_line, 0,
                              error);
        }
      }
    }
  }
}

/**
 * Run all six verifier checks (static analysis) on the compiled module.
 * Errors are accumulated into module->errors.
 */
void vm_module_generator_verifier_VM_ASSERT_NOT_NULL(vm_module *module) {
  _vm_module_generator_verifier_check_instruction_params(module);
  _vm_module_generator_verifier_check_function_call_params(module);
  _vm_module_generator_verifier_check_call_function_declared(module);
  _vm_module_generator_verifier_check_call_var_declared(module);
  _vm_module_generator_verifier_check_function_return_value_type(module);
  _vm_module_generator_verifier_check_function_return_type_value_assignment_after_call(
      module);
}

#include "microgpt_vm_parser.tab.h"

extern VM_MODULE_PARSER_STYPE vm_module_parser_lval;

#define ERROR_BUFFER_SIZE 1024

#define VM_OK 0
#define VM_UNKNOWN -1
#define RESULT_CORE_VM_SYNTAX_ERROR -2
#define RESULT_CORE_VM_OUT_OF_MEMORY -3

// int vm_module_parser_char_fetch_next(vm_module_parser* parser);

// vm_module_parser* vm_module_parser_create(const char* source);

/**
 * Global current-parser pointer used by Flex's YY_INPUT macro.
 * WARNING: makes the parser non-reentrant (see ISS-1 in ISSUES.md).
 */
vm_module_parser *_vm_ctx_current_parser;

extern int vm_module_parser_parse(vm_module_parser *parser);
extern void vm_module_parser_set_in(FILE *in_str);

/**
 * Top-level entry point: parse source text into a compiled vm_module.
 *
 * Pipeline: parse → post-process (5 passes) → VM_ASSERT (6 checks).
 * On success, *out_module receives a fully resolved module ready for
 * vm_module_runtime_run().  On failure, module->errors is populated.
 *
 * @param verb_context_  Optional native function dispatch table.
 * @param source         Null-terminated source string.
 * @param out_module     Receives the compiled module on success.
 * @return VM_OK (0) on success, negative error code on failure.
 */
vm_result vm_module_parser_generate(verb_context *verb_context_,
                                    const char *source,
                                    vm_module **out_module) {
  VM_ASSERT_NOT_NULL(source);
  VM_ASSERT_NOT_NULL(out_module);

  // Init
  vm_module_parser *parser = VM_NEW(vm_module_parser);
  parser->vm_module_parser_state_input = source;
  parser->vm_module_parser_state_input_index = 0;
  parser->vm_module_parser_state_input_len = vm_string_length(source);
  parser->vm_module_parser_state_input_line_number = 1;

  vm_module *module = vm_module_create(verb_context_);
  parser->generator = vm_module_generator_create(module);

  vm_module_parser_set_in(0);

  // Used by Flex
  _vm_ctx_current_parser = parser;

  int vm_result = vm_module_parser_parse(parser);

  if (vm_list_count(module->errors) == 0) {
    vm_module_generator_post_processing_process(module);
    vm_module_generator_verifier_VM_ASSERT_NOT_NULL(module);
  }

  vm_module_generator_dispose(parser->generator);
  VM_FREE(parser);

  switch (vm_result) {
  case 0:
    // The value returned by yyparse is 0 if parsing was successful
    // (return is due to end-of-input).
    *out_module = module;
    return VM_OK;

  case 1:

    // The value is 1 if parsing failed because of invalid input, i.e.,
    // input that contains a syntax error or that causes YYABORT to be
    // invoked.
    return RESULT_CORE_VM_SYNTAX_ERROR;

  case 2:

    // The value is 2 if parsing failed due to memory exhaustion.
    return RESULT_CORE_VM_OUT_OF_MEMORY;

  case 3:
    return RESULT_CORE_VM_SYNTAX_ERROR;

  default:
    return VM_UNKNOWN;
  }
}

/** Flex wrap callback — returns 1 (no further input segments). */
int vm_module_parser_wrap() { return 1; }

/**
 * Read the next character from the in-memory source buffer.
 * Called by Flex's YY_INPUT macro during lexical analysis.
 * Returns 0 (EOF) when the entire source has been consumed.
 */
int vm_module_parser_char_fetch_next(vm_module_parser *parser) {
  VM_ASSERT_NOT_NULL(parser);

  int c = 0;

  if (parser->vm_module_parser_state_input_index <
      parser->vm_module_parser_state_input_len) {
    c = (int)parser->vm_module_parser_state_input
            [parser->vm_module_parser_state_input_index++];
  }

  return c;
}

/**
 * Bison error callback: records a parse error with the offending source
 * snippet for later reporting through module->errors.
 */
void vm_module_parser_error(vm_module_parser *parser, char *error) {
  VM_ASSERT_NOT_NULL(parser);
  VM_ASSERT_NOT_NULL(error);

  vm_module *module = parser->generator->vm_module_parser_state_current_module;
  VM_ASSERT_NOT_NULL(module);

  // Show the offending line
  const char *input = parser->vm_module_parser_state_input;
  const size_t input_len = parser->vm_module_parser_state_input_len;
  const size_t error_pos = parser->vm_module_parser_state_input_index;
  size_t error_start = error_pos;
  size_t error_end = (error_pos + 10) < input_len ? error_pos + 10 : error_pos;

  // Limit the error line to the following
  char buffer[ERROR_BUFFER_SIZE + 1];
  VM_ASSERT_NOT_NULL(error_end > error_start);
  size_t error_len = error_end - error_start;

  if (error_len > ERROR_BUFFER_SIZE) {
    // Shrink both sides
    const size_t buffer_overflow_len = error_len - ERROR_BUFFER_SIZE;
    const size_t adjustment = buffer_overflow_len / 2;
    error_start += adjustment;
    error_end -= adjustment;
    error_len = ERROR_BUFFER_SIZE;
  }

  vm_string_copy_length(buffer, &input[error_start], error_len);
  buffer[error_len] = 0;

  char error2[1024];
  snprintf(error2, sizeof(error2), "\n>>> script: %s @ %lu:%zu\n    %s", error,
           (unsigned long)parser->vm_module_parser_state_input_line_number,
           parser->vm_module_parser_state_input_line_column, buffer);
  vm_module_error_add(module, parser->vm_module_parser_state_input_line_number,
                      0, error2);
  vm_log_error(error2, NULL);

  char error3[1024];
  snprintf(error3, sizeof(error3), "   %s\n", buffer);
  vm_module_error_add(module, parser->vm_module_parser_state_input_line_number,
                      0, error);

  vm_log_error(error3, NULL);
}

// #define _DEBUG_TRACE

/* convert.h removed */

#define VM_OK 0
#define VM_UNKNOWN -1
// Macros moved to vm_module_runtime.h

/**
 * Create a runtime execution context for a compiled module.
 * The runtime manages the evaluation stack and execution state.
 */
vm_module_runtime *vm_module_runtime_create(vm_module *module) {
  VM_ASSERT_NOT_NULL(module);

  vm_module_runtime *runtime = VM_NEW(vm_module_runtime);
  runtime->module = module;
  //    runtime->stack = vm_queue_create();

  return runtime;
}

/** Dispose a runtime and all its resources (stack, errors, function state).
 */
void vm_module_runtime_dispose(vm_module_runtime *runtime) {
  VM_ASSERT_NOT_NULL(runtime);

  vm_module_runtime_clear(runtime);

  // Reset errors
  vm_list_foreach_of(runtime->module->errors, vm_module_error *, error) {
    vm_module_error_dispose(error);
  }

  // Reset functions
  vm_map_foreach_of(runtime->module->functions, name, vm_function *, function) {
    vm_function_clear(function);
  }
  VM_FREE(runtime);
}

/**
 * Reset the runtime to a clean state without deallocating it.
 * Frees stack variables, clears errors, and resets function state.
 * Allows re-execution of the same compiled module.
 */
void vm_module_runtime_clear(vm_module_runtime *runtime) {
  VM_ASSERT_NOT_NULL(runtime);

  // Clear stack
  size_t i = runtime->stack_size;

  while (i-- > 0) {
    vm_variable *variable = runtime->stack[i];
    VM_ASSERT_NOT_NULL(variable);
    vm_variable_dispose(variable);
  }

  runtime->stack_size = 0;

  // Reset errors
  vm_list_foreach_of(runtime->module->errors, vm_module_error *, error) {
    if (error->message) {
      vm_string_free(error->message);
    }

    VM_FREE(error);
  }
  vm_list_clear(runtime->module->errors);

  // Reset functions
  vm_map_foreach_of(runtime->module->functions, name, vm_function *, function) {
    vm_function_clear(function);
  }
}

/** Push a boolean value onto the runtime evaluation stack. */
void vm_module_runtime_stack_push_boolean(vm_module_runtime *runtime,
                                          bool value) {
  VM_ASSERT_NOT_NULL(runtime);
  VM_ASSERT(runtime->stack_size != VM_MAX_STACK);

#ifdef VM_ENABLE_VAR_NAME
  char name[100]; // TODO: check
  snprintf(name, sizeof(name), "b%d", runtime->stack_size);
  vm_variable *stack_push_var = vm_variable_create(name);
#else
  vm_variable *stack_push_var = vm_variable_create();
#endif

  stack_push_var->type_class = ptcBOOLEAN;
  stack_push_var->value.boolean = value;

  runtime->stack[runtime->stack_size++] = stack_push_var;
}

/** Push a number (double) value onto the runtime evaluation stack. */
void vm_module_runtime_stack_push_number(vm_module_runtime *runtime,
                                         double value) {
  VM_ASSERT_NOT_NULL(runtime);
  VM_ASSERT(runtime->stack_size != VM_MAX_STACK);

#ifdef VM_ENABLE_VAR_NAME
  char name[100]; // TODO: check
  snprintf(name, sizeof(name), "n%d", runtime->stack_size);
  vm_variable *stack_push_var = vm_variable_create(name);
#else
  vm_variable *stack_push_var = vm_variable_create();
#endif

  stack_push_var->type_class = ptcNUMBER;
  stack_push_var->value.number = value;

  runtime->stack[runtime->stack_size++] = stack_push_var;
}

/** Push a string value onto the runtime evaluation stack (cloned). */
void vm_module_runtime_stack_push_string(vm_module_runtime *runtime,
                                         const char *value) {
  VM_ASSERT_NOT_NULL(runtime);
  VM_ASSERT(runtime->stack_size != VM_MAX_STACK);

#ifdef VM_ENABLE_VAR_NAME
  char name[100]; // TODO: check
  snprintf(name, sizeof(name), "s%d", runtime->stack_size);
  vm_variable *stack_push_var = vm_variable_create(name);
#else
  vm_variable *stack_push_var = vm_variable_create();
#endif

  stack_push_var->type_class = ptcSTRING;
  stack_push_var->value.string = value ? vm_string_clone(value) : NULL;

  runtime->stack[runtime->stack_size++] = stack_push_var;
}

/** Push an opaque pointer onto the runtime evaluation stack. */
void vm_module_runtime_stack_push_other(vm_module_runtime *runtime,
                                        void *other) {
  VM_ASSERT_NOT_NULL(runtime);
  VM_ASSERT(runtime->stack_size != VM_MAX_STACK);

#ifdef VM_ENABLE_VAR_NAME
  char name[100]; // TODO: check
  snprintf(name, sizeof(name), "o%d", runtime->stack_size);
  vm_variable *stack_push_var = vm_variable_create(name);
#else
  vm_variable *stack_push_var = vm_variable_create();
#endif

  stack_push_var->type_class = ptcOTHER;
  stack_push_var->value.other = other;

  runtime->stack[runtime->stack_size++] = stack_push_var;
}

/**
 * Pop the top value from the runtime evaluation stack.
 * @param out_variable  Receives the popped variable.  Caller owns it.
 * @return VM_OK or RESULT_CORE_VM_EMPTY_STACK if stack is empty.
 */
vm_result vm_module_runtime_stack_pop(vm_module_runtime *runtime,
                                      vm_variable **out_variable) {
  VM_ASSERT_NOT_NULL(runtime);
  VM_ASSERT_NOT_NULL(out_variable);

  *out_variable = NULL;

  if (runtime->stack_size == 0) {
    return RESULT_CORE_VM_EMPTY_STACK;
  }

  *out_variable = runtime->stack[--runtime->stack_size];

  return VM_OK;
}

#define op(expected_var_type_class_1, expected_var_type_class_2,               \
           expected_var_type_class_3, operation)                               \
  if (var_type_class_1 == expected_var_type_class_1 &&                         \
      var_type_class_2 == expected_var_type_class_2 &&                         \
      var_type_class_3 == expected_var_type_class_3) {                         \
    operation did_process_opcode = true;                                       \
  }

#define op_ignore() did_process_opcode = true;

#define op_completed() did_process_opcode = true;

#define op_set_error(message)                                                  \
  error = vm_string_clone(message);                                            \
  did_error = true;

/**
 * Debug helper: dump an instruction with its resolved variable values.
 * Only used when _DEBUG_TRACE is defined.
 */
void vm_instruction_with_var_to_string(vm_function *function,
                                       vm_instruction *instruction,
                                       vm_param_type_class var_type_class_1,
                                       vm_param_type_class var_type_class_2,
                                       vm_param_type_class var_type_class_3) {
  vm_string_buffer *buffer = vm_string_buffer_create_empty();
  char *instruction_dump = vm_instruction_to_string(instruction);
  vm_string_buffer_append_format(buffer, "%3d)\t",
                                 function->instruction_pointer);
  vm_string_buffer_append(buffer, instruction_dump);
  vm_string_buffer_append(buffer, "|");

  if (var_type_class_1 != ptcNONE) {
    vm_variable_to_vm_string_buffer(buffer, instruction->param1,
                                    instruction->var1);
  }

  vm_string_buffer_append(buffer, "\t|");

  if (var_type_class_2 != ptcNONE) {
    vm_variable_to_vm_string_buffer(buffer, instruction->param2,
                                    instruction->var2);
  }

  vm_string_buffer_append(buffer, "\t|");

  if (var_type_class_3 != ptcNONE) {
    vm_variable_to_vm_string_buffer(buffer, instruction->param3,
                                    instruction->var3);
  }

  puts(buffer->data);

  vm_string_buffer_free(buffer);
  vm_string_free(instruction_dump);
}

/** Register a callback for CALL_EXT_METHOD opcodes (native function calls).
 */
void vm_module_runtime_set_call_ext_method_callback(
    vm_module_runtime *runtime, vm_call_ext_method_callback callback) {
  VM_ASSERT_NOT_NULL(runtime);
  VM_ASSERT_NOT_NULL(callback);

  runtime->call_ext_method_callback = callback;
}

/**
 * Execute a compiled function using the dispatch loop.
 * Iterates through the instruction array, dispatching each opcode via
 * a switch with type-specialised handlers.  This is the core of the VM.
 *
 * WARNING: This function is ~700 lines long (see ISS-7 in ISSUES.md).
 *
 * @param runtime   Execution context (stack, ext callbacks).
 * @param function  The function to execute (instruction_pointer is reset).
 * @return VM_OK on success, negative error code on failure.
 */
vm_result vm_module_runtime_run(vm_module_runtime *runtime,
                                vm_function *function) {
  VM_ASSERT_NOT_NULL(runtime);
  VM_ASSERT_NOT_NULL(function);

  char *error = NULL;

#ifdef _DEBUG_TRACE
  printf("\n--> %s <----------------------------\n", function->name);
#endif
  vm_list_item *instructions = vm_list_enumerable(function->instructions);
  const size_t instructions_count = vm_list_count(function->instructions);

  while (function->instruction_pointer < instructions_count) {
    vm_result r = RESULT_CORE_VM_RUNTIME_ERROR;

    vm_instruction *instruction =
        (vm_instruction *)instructions[function->instruction_pointer].value;

    const vm_param_type_class var_type_class_1 =
        !instruction->var1 ? ptcNONE : instruction->var1->type_class;
    vm_variable_value *var_value_1 =
        !instruction->var1 ? NULL : &instruction->var1->value;

    const vm_param_type_class var_type_class_2 =
        !instruction->var2 ? ptcNONE : instruction->var2->type_class;
    vm_variable_value *var_value_2 =
        !instruction->var2 ? NULL : &instruction->var2->value;

    const vm_param_type_class var_type_class_3 =
        !instruction->var3 ? ptcNONE : instruction->var3->type_class;
    vm_variable_value *var_value_3 =
        !instruction->var3 ? NULL : &instruction->var3->value;

#ifdef _DEBUG_TRACE
    char *instruction_dump = vm_instruction_to_string(instruction);
    puts(instruction_dump);
    vm_string_free(instruction_dump);

    if (instruction->opcode == opCALL_METHOD ||
        instruction->opcode == opRETURN) {
      vm_instruction_with_var_to_string(function, instruction, var_type_class_1,
                                        var_type_class_2, var_type_class_3);
    }

#endif

    bool did_error = false;
    bool did_process_opcode = false;

    vm_variable *stack_push_var = NULL;
    vm_variable *stack_pop_var = NULL;
    char *buffer_string = NULL;
    char *buffer_number = NULL;
    size_t len = 0;
#ifdef VM_ENABLE_VAR_NAME
    char name[100]; // TODO: check
#endif

    switch (instruction->opcode) {
    /******************************************************************
     * opNOP: no operation
     * param1:
     * param2:
     * param3:
     */
    case opNOP:
      op_ignore();
      break;

    /******************************************************************
     * opINC: increment a variable by one
     * param1: input A
     * param2:
     * param3:
     */
    case opINC:
      op(ptcNUMBER, ptcNONE, ptcNONE, { var_value_1->number++; });
      break;

    /******************************************************************
     * opDEC: decrement a variable by one
     * param1: input A
     * param2:
     * param3:
     */
    case opDEC:
      op(ptcNUMBER, ptcNONE, ptcNONE, { var_value_1->number--; });
      break;

    /******************************************************************
     * opADD: add two numbers, decimals, strings and other strings with these
     * param1: input A
     * param2: input B
     * param3: vm_result
     */
    case opADD:
      op(ptcNUMBER, ptcNUMBER, ptcNUMBER,
         { var_value_3->number = var_value_1->number + var_value_2->number; });

      op(ptcSTRING, ptcSTRING, ptcSTRING, {
        len = vm_string_length(var_value_1->string) +
              vm_string_length(var_value_2->string);
        buffer_string = (char *)malloc(len + 1);
        buffer_string[0] = '\0';
        strcat(buffer_string, var_value_1->string);
        strcat(buffer_string, var_value_2->string);
        var_value_3->string = buffer_string;
      });

      op(ptcSTRING, ptcNUMBER, ptcSTRING, {
        buffer_number = vm_int_to_string(var_value_2->number);
        len = vm_string_length(var_value_1->string) +
              vm_string_length(buffer_number);
        buffer_string = (char *)malloc(len + 1);
        buffer_string[0] = '\0';
        strcat(buffer_string, var_value_1->string);
        strcat(buffer_string, buffer_number);
        var_value_3->string = buffer_string;
        vm_string_free(buffer_number);
      });

      break;

    /******************************************************************
     * opMUL: multiply two number or decimals
     * param1: input A
     * param2: input B
     * param3: vm_result
     */
    case opMUL:
      op(ptcNUMBER, ptcNUMBER, ptcNUMBER, {
        var_value_3->number = var_value_1->number * var_value_2->number;
      }) break;

    /******************************************************************
     * opDIV: divide two number or decimals
     * param1: input A
     * param2: input B
     * param3: vm_result
     */
    case opDIV:
      op(ptcNUMBER, ptcNUMBER, ptcNUMBER, {
        // TODO: loose precision
        var_value_3->number = var_value_1->number / var_value_2->number;
      });
      break;

    /******************************************************************
     * opEXP: exponent
     * param1: input A
     * param2: input B
     * param3: vm_result
     */
    case opEXP:
      op(ptcNUMBER, ptcNONE, ptcNUMBER,
         { var_value_3->number = exp(var_value_1->number); });
      break;

    /******************************************************************
     * opEXP: power to
     * param1: input A
     * param2: input B
     * param3: vm_result
     */
    case opPOW:
      op(ptcNUMBER, ptcNUMBER, ptcNUMBER, {
        var_value_3->number = pow(var_value_1->number, var_value_2->number);
      });
      break;

    /******************************************************************
     * opSUB: subtract two number or decimals
     * param1: input A
     * param2: input B
     * param3: vm_result
     */
    case opSUB:
      op(ptcNUMBER, ptcNUMBER, ptcNUMBER,
         { var_value_3->number = var_value_1->number - var_value_2->number; });
      break;

    /******************************************************************
     * opNEG: exit function
     * param1: input
     * param2:
     * param3: vm_result
     */
    case opNEG:
      op(ptcNUMBER, ptcNONE, ptcNUMBER,
         { var_value_3->number = var_value_1->number * -1; });
      break;

    /******************************************************************
     * opNOT: reverse a boolean
     * param1: value
     * param2:
     * param3: vm_result
     */
    case opNOT:
      op(ptcBOOLEAN, ptcNONE, ptcBOOLEAN,
         { var_value_3->boolean = !var_value_1->boolean; });

      break;

    /******************************************************************
     * opSET_VAR: exit function
     * param1:
     * param2:
     * param3:
     */
    case opSET_VAR:

    // TODO: check vars
    // Use create ...

    /******************************************************************
     * opCREATE_SET_VAR: exit function
     * param1:
     * param2:
     * param3:
     */
    case opCREATE_SET_VAR:
      op(ptcNUMBER, ptcNONE, ptcNUMBER,
         { var_value_3->number = var_value_1->number; });
      op(ptcBOOLEAN, ptcNONE, ptcBOOLEAN,
         { var_value_3->boolean = var_value_1->boolean; });
      op(ptcSTRING, ptcNONE, ptcSTRING, {
        if (var_value_3->string)
          vm_string_free(var_value_3->string);
        var_value_3->string = vm_string_clone(var_value_1->string);
      });

      break;

    /******************************************************************
     * opGET_OBJ_VAR: exit function
     * param1:
     * param2:
     * param3:
     */
    case opGET_OBJ_VAR:
      break;

    /******************************************************************
     * opSET_OBJ_VAR: exit function
     * param1:
     * param2:
     * param3:
     */
    case opSET_OBJ_VAR:
      break;

    /******************************************************************
     * opSTACK_POP: exit function
     * param1:
     * param2:
     * param3:
     */
    case opSTACK_POP:
      if (runtime->stack_size == 0) {
        op_set_error("Tried to STACK_POP on empty stack");
      } else {
        stack_pop_var = runtime->stack[--runtime->stack_size];
        VM_ASSERT_NOT_NULL(stack_pop_var);

        // Assign to target var
        if (!vm_variable_value_assign(stack_pop_var, var_type_class_3,
                                      var_value_3)) {
          printf(
              "ERROR assigning popped stack value of %s to var %s\n",
              vm_variable_param_type_class_to_string(stack_pop_var->type_class),
              vm_variable_param_type_class_to_string(var_type_class_3));
          op_set_error("Unable to assigned popped value from stack");
        } else {
          did_process_opcode = true;
        }

        vm_variable_dispose(stack_pop_var);
        //                    VM_FREE(stack_pop_var);
      }

      break;

    /******************************************************************
     * opSTACK_PUSH: exit function
     * param1: value to push onto stack
     * param2:
     * param3:
     */
    case opSTACK_PUSH:
#ifdef VM_ENABLE_VAR_NAME
      snprintf(name, sizeof(name), "s%d", runtime->stack_size);
      stack_push_var = vm_variable_create(name);
#else
      stack_push_var = vm_variable_create();
#endif
      stack_push_var->type_class = var_type_class_1;

      if (var_type_class_1 == ptcSTRING) {
        stack_push_var->value.string = vm_string_clone(var_value_1->string);
      } else {
        stack_push_var->value = *var_value_1;
      }

      VM_ASSERT(runtime->stack_size != VM_MAX_STACK);
      runtime->stack[runtime->stack_size++] = stack_push_var;

      did_process_opcode = true;
      break;

    /******************************************************************
     * opCALL_METHOD: call a function, recursively
     * param1: method name
     * param2:
     * param3:
     */
    case opCALL_METHOD:
      if (!instruction->call_method_function &&
          !instruction->call_method_verb) {
        vm_log_error("Function/Verb doesn't exist %s", instruction->param1);
        r = RESULT_CORE_VM_RUNTIME_METHOD_DOESNT_EXIST;
      } else if (instruction->call_method_function) {
        if (instruction->call_method_function->is_executing) {
          vm_log_error(
              "Cannot call method (function) %s recursively (not supported)",
              instruction->param1);
          r = RESULT_CORE_VM_RUNTIME_RECURSION_NOT_SUPPORTED;
        } else {
          instruction->call_method_function->is_executing = true;
          instruction->call_method_function->instruction_pointer = 0;

          r = vm_module_runtime_run(runtime, instruction->call_method_function);

          if (r == VM_OK) {
            instruction->call_method_function->is_executing = false;
            op_completed();
          }
        }
      } else if (instruction->call_method_verb) {
        verb_definition *verb_definition_ = instruction->call_method_verb;

        // Do we have enough params on stack?
        if (runtime->stack_size < verb_definition_->params_count) {
          char err_msg[1024];
          snprintf(err_msg, sizeof(err_msg),
                   "Cannot call method (verb) %s not enough params (%zu/%zu)",
                   instruction->param1, runtime->stack_size,
                   verb_definition_->params_count);
          vm_log_error(err_msg, NULL);
          r = RESULT_CORE_VM_RUNTIME_INCORRECT_PARAMS;
        } else {
          vm_map *verb_arg_list = vm_map_create();
          //                        vm_map_set( verb_arg_list, "verb_context",
          //                        runtime->module->verb_context_ );

          r = VM_OK;
          vm_list_foreach_of(verb_definition_->param_list, char *,
                             verb_param_name) {
            // size_t verb_params_count = verb_definition_->params_count;
            // while ( verb_params_count-- > 0 && r == VM_OK ) {
            vm_variable *out_variable;
            r = vm_module_runtime_stack_pop(runtime, &out_variable);
            if (r != VM_OK) {
              break;
            }

            vm_map_set(verb_arg_list, verb_param_name,
                       vm_string_clone(out_variable->value.string));
            vm_variable_dispose(out_variable);
          }

          if (r == VM_OK) {
            char *result_response = verb_definition_->function(
                runtime->module->verb_context_, verb_arg_list,
                verb_definition_->fcontext);
            if (result_response) {
              if (!strncmp(result_response, "ERROR:", 6)) {
                r = RESULT_CORE_VERB_ERROR;
              } else {
                vm_module_runtime_stack_push_string(runtime, result_response);
              }

              vm_string_free(result_response);
            }
            r = VM_OK;
            op_completed();
          }

          vm_map_free(verb_arg_list);
        }
      }

      break;

    /******************************************************************
     * opCALL_OBJ_METHOD:
     * param1:
     * param2:
     * param3:
     */
    case opCALL_OBJ_METHOD:
      break;

    /******************************************************************
     * opCALL_EXT_METHOD: external function
     * param1:
     * param2:
     * param3:
     */
    case opCALL_EXT_METHOD:
      if (!runtime->call_ext_method_callback) {
        vm_log_error("call ext method callback not set", NULL);
        r = RESULT_CORE_VM_RUNTIME_CALL_EXT_METHOD_CALLBACK_NOT_SET;
      } else {
        runtime->call_ext_method_callback(runtime, function);
        op_completed();
      }

      break;

    /******************************************************************
     * opRETURN: exit function
     * param1:
     * param2:
     * param3:
     */
    case opRETURN:
      op_ignore();
      break;

    /******************************************************************
     * opCONDITION_GTE: if A >= B
     * param1: input A
     * param2: input B
     * param3: vm_result
     */
    case opCONDITION_GTE:
      op(ptcNUMBER, ptcNUMBER, ptcBOOLEAN, {
        var_value_3->boolean = var_value_1->number >= var_value_2->number;
      });

      break;

    /******************************************************************
     * opCONDITION_LTE: if A <= B
     * param1: input A
     * param2: input B
     * param3: vm_result
     */
    case opCONDITION_LTE:
      op(ptcNUMBER, ptcNUMBER, ptcBOOLEAN, {
        var_value_3->boolean = var_value_1->number <= var_value_2->number;
      });
      break;

    /******************************************************************
     * opCONDITION_GT: if A > B
     * param1: input A
     * param2: input B
     * param3: vm_result
     */
    case opCONDITION_GT:
      op(ptcNUMBER, ptcNUMBER, ptcBOOLEAN,
         { var_value_3->boolean = var_value_1->number > var_value_2->number; });
      break;

    /******************************************************************
     * opCONDITION_LT: if A < B
     * param1: input A
     * param2: input B
     * param3: vm_result
     */
    case opCONDITION_LT:
      op(ptcNUMBER, ptcNUMBER, ptcBOOLEAN,
         { var_value_3->boolean = var_value_1->number < var_value_2->number; });
      break;

    /******************************************************************
     * opCONDITION_NE: if A != B
     * param1: input A
     * param2: input B
     * param3: vm_result
     */
    case opCONDITION_NE:
      op(ptcSTRING, ptcSTRING, ptcBOOLEAN, {
        var_value_3->boolean =
            !vm_string_equals(var_value_1->string, var_value_2->string);
      });
      op(ptcNUMBER, ptcNUMBER, ptcBOOLEAN, {
        var_value_3->boolean = var_value_1->number != var_value_2->number;
      });
      op(ptcBOOLEAN, ptcBOOLEAN, ptcBOOLEAN, {
        var_value_3->boolean = var_value_1->boolean != var_value_2->boolean;
      });

      break;

    /******************************************************************
     * opCONDITION_EQ: if A == B
     * param1: input A
     * param2: input B
     * param3: vm_result
     */
    case opCONDITION_EQ:
      op(ptcSTRING, ptcSTRING, ptcBOOLEAN, {
        var_value_3->boolean =
            vm_string_equals(var_value_1->string, var_value_2->string);
      });
      op(ptcNUMBER, ptcNUMBER, ptcBOOLEAN, {
        var_value_3->boolean = var_value_1->number == var_value_2->number;
      });
      op(ptcBOOLEAN, ptcBOOLEAN, ptcBOOLEAN, {
        var_value_3->boolean = var_value_1->boolean == var_value_2->boolean;
      });

      break;

    /******************************************************************
     * opCONDITION_TRUE: if A is true
     * param1: input A
     * param2:
     * param3: vm_result
     */
    case opCONDITION_TRUE:
      op(ptcBOOLEAN, ptcNONE, ptcBOOLEAN,
         { var_value_3->boolean = var_value_1->boolean; });

      break;

    /******************************************************************
     * opCONDITIONAL_AND: if (A && B)
     * param1: input A
     * param2: input B
     * param3: vm_result
     */
    case opCONDITIONAL_AND:
      op(ptcBOOLEAN, ptcBOOLEAN, ptcBOOLEAN, {
        var_value_3->boolean = var_value_1->boolean && var_value_2->boolean;
      });
      break;

    /******************************************************************
     * opCONDITIONAL_OR: if (A || B)
     * param1: input A
     * param2: input B
     * param3: vm_result
     */
    case opCONDITIONAL_OR:
      op(ptcBOOLEAN, ptcBOOLEAN, ptcBOOLEAN, {
        var_value_3->boolean = var_value_1->boolean || var_value_2->boolean;
      });
      break;

    /******************************************************************
     * opJUMP_IF_TRUE: jump to instruction if A is true
     * param1: input A
     * param2:
     * param3: jump to index
     */
    case opJUMP_IF_TRUE:
      break;

    /******************************************************************
     * opJUMP_IF_FALSE: jump to instruction if A is false
     * param1: input A
     * param2:
     * param3: jump to index
     */
    case opJUMP_IF_FALSE:
      op(ptcBOOLEAN, ptcNONE, ptcNONE, {
        if (!var_value_1->boolean) {
          if (instruction->opJUMP_jump_to_instruction_pos == 0) {
            vm_log_error("Jump index cannot be 0", NULL);
            r = RESULT_CORE_VM_RUNTIME_JUMP_INDEX_INVALID;
          } else {
            function->instruction_pointer =
                instruction->opJUMP_jump_to_instruction_pos;
            continue;
          }
        }
      });

      break;

    /******************************************************************
     * opJUMP: jump to instruction
     * param1:
     * param2:
     * param3: jump to index
     */
    case opJUMP:
      op(ptcNONE, ptcNONE, ptcNONE, {
        if (instruction->opJUMP_jump_to_instruction_pos == 0) {
          vm_log_error("Jump index cannot be 0", NULL);
          r = RESULT_CORE_VM_RUNTIME_JUMP_INDEX_INVALID;
        } else {
          function->instruction_pointer =
              instruction->opJUMP_jump_to_instruction_pos;
          continue;
        }
      });

      break;

    /******************************************************************
     * opLABEL: label of a function or loop/condition jump point
     * param1:
     * param2:
     * param3:
     */
    case opLABEL:
      op_ignore();
      break;

    /******************************************************************
     * opYIELD:
     * param1:
     * param2:
     * param3:
     */
    case opYIELD:
      break;

    /******************************************************************
     * opXPATH:
     * param1:
     * param2:
     * param3:
     */
    case opXPATH:
      break;

    /******************************************************************
     * opJSON:
     * param1:
     * param2:
     * param3:
     */
    case opJSON:
      break;

    /******************************************************************
     * opCOMMENT: comment
     * param1:
     * param2:
     * param3:
     */
    case opCOMMENT:
      op_ignore();
      break;

    default:
      snprintf(error, sizeof(error), "Unsupported VM opcode %s (%d)",
               vm_instruction_opcode_to_string(instruction->opcode),
               instruction->opcode);
      did_error = true;
      break;
    }

    // TODO: move this out so any return false will show error also pass
    // runtime error message
    if (did_error || !did_process_opcode) {
      char *instruction_dump = vm_instruction_to_string(instruction);

      if (did_error) {
#ifdef _DEBUG_TRACE
        printf("%s:\n%3d) %s", error, function->instruction_pointer,
               instruction_dump);
#endif
        char err_msg[1024];
        snprintf(err_msg, sizeof(err_msg), "%s:\n%3zu) %s", error,
                 function->instruction_pointer, instruction_dump);
        vm_log_error(err_msg, NULL);

      } else if (!did_process_opcode) {
#ifdef _DEBUG_TRACE
        printf("No handler to process instruction:\n%3d) %s",
               function->instruction_pointer, instruction_dump);
#endif
        char err_msg[1024];
        snprintf(err_msg, sizeof(err_msg),
                 "No handler to process instruction:\n%3zu) %s",
                 function->instruction_pointer, instruction_dump);
        vm_log_error(err_msg, NULL);
        r = r != RESULT_CORE_VM_RUNTIME_ERROR ? r
                                              : RESULT_CORE_VM_RUNTIME_ERROR;
      }

      vm_string_free(instruction_dump);
      return r;
    }

#ifdef _DEBUG_TRACE

    if (instruction->opcode != opCALL_METHOD &&
        instruction->opcode != opRETURN) {
      vm_instruction_with_var_to_string(function, instruction, var_type_class_1,
                                        var_type_class_2, var_type_class_3);
    }

#endif

    function->instruction_pointer++;
  }

  return VM_OK;
}

/* ═══════════════════════════════════════════════════════════════════════════
 *  Type Traits  —  compile-time type metadata for symbols
 * ═══════════════════════════════════════════════════════════════════════════
 */

/** Create a new type trait with the given type string and constant flag. */
vm_type_trait *vm_type_trait_create(char *type, bool is_constant) {
  VM_ASSERT_NOT_NULL(type);

  vm_type_trait *type_trait = VM_NEW(vm_type_trait);
  type_trait->type = vm_string_clone(type);
  type_trait->is_constant = is_constant;

  return type_trait;
}

/** Free a type trait and its cloned type string. */
void vm_type_trait_dispose(vm_type_trait *type_trait) {
  VM_ASSERT_NOT_NULL(type_trait);

  if (type_trait->type) {
    vm_string_free(type_trait->type);
  }

  VM_FREE(type_trait);
}

/* ═══════════════════════════════════════════════════════════════════════════
 *  Functions  —  named units of compiled bytecode
 * ═══════════════════════════════════════════════════════════════════════════
 */

/**
 * Allocate and initialise a new function.
 * Sets up all internal collections (instructions, variables, constants,
 * registers, symbols, labels, parameters) and tracking state.
 * @param name  Already heap-allocated name string — ownership is transferred.
 */
vm_function *vm_function_create(char *name) {
  VM_ASSERT_NOT_NULL(name);

  vm_function *function = VM_NEW(vm_function);
  function->name = name; // name is already text cloned
  function->parameters = vm_list_create();
  function->instructions = vm_list_create();
  function->tracking_labels = vm_queue_create();
  function->register_names = vm_list_create();
  function->label_names = vm_list_create();
  function->symbols = vm_list_create();
  function->trait_types = vm_map_create();
  function->return_type_class = ptcNONE;
  function->return_type = NULL;

  function->variables = vm_map_create();
  function->constants = vm_map_create();
  function->variables_list = vm_list_create();
  function->registers = vm_list_create();

  function->is_executing = false;
  function->instruction_pointer = 0;

  return function;
}

/**
 * Free a function and all its sub-resources.
 */
void vm_function_dispose(vm_function *function) {
  VM_ASSERT_NOT_NULL(function);
  VM_ASSERT_NOT_NULL(function->name);
  VM_ASSERT_NOT_NULL(function->instructions);

  if (function->return_type) {
    vm_string_free(function->return_type);
  }

  vm_string_free(function->name);

  vm_list_foreach_of(function->instructions, vm_instruction *, instruction) {
    vm_instruction_free(instruction);
  }
  vm_list_free(function->instructions);

  vm_map_foreach_of(function->trait_types, trait_type_name, vm_type_trait *,
                    type_trait) {
    vm_type_trait_dispose(type_trait);
  }
  vm_map_free(function->trait_types);

  vm_list_foreach_of(function->registers, vm_variable *, reg) {
    vm_variable_dispose(reg);
  }
  vm_list_free(function->registers);

  vm_list_dispose_items(function->register_names);
  vm_list_free(function->register_names);

  vm_map_foreach_of(function->constants, constant_name, vm_variable *,
                    constant) {
    // We don't free constant values — this comes from symbols, we only have a
    // reference here
    if (constant->type_class == ptcSTRING) {
      constant->value.string = NULL;
    }

    vm_variable_dispose(constant);
  }
  vm_map_free(function->constants);

  vm_map_foreach_of(function->variables, variable_name, vm_variable *, var) {
    vm_variable_dispose(var);
  }
  vm_map_free(function->variables);

  vm_list_dispose_items(function->variables_list);
  vm_list_free(function->variables_list);

  vm_list_dispose_items(function->parameters);
  vm_list_free(function->parameters);

  vm_list_dispose_items(function->label_names);
  vm_list_free(function->label_names);
  vm_queue_free(function->tracking_labels);

  vm_list_dispose_items(function->symbols);
  vm_list_free(function->symbols);

  VM_FREE(function);
}

/**
 * Reset a function's runtime state without deallocating it.
 * Clears registers and non-constant variables for re-execution.
 */
void vm_function_clear(vm_function *function) {
  VM_ASSERT_NOT_NULL(function);
  VM_ASSERT_NOT_NULL(function->name);
  VM_ASSERT_NOT_NULL(function->instructions);

  function->is_executing = false;
  function->instruction_pointer = 0;

  vm_list_foreach_of(function->registers, vm_variable *, reg) {
    //        vm_variable_dispose(reg);

    vm_variable_clear(reg);
  }
  //    vm_list_clear(function->registers);

  vm_map_foreach_of(function->variables, name, vm_variable *, var) {
    //      vm_variable_dispose(var);
    if (!var->is_constant && !var->is_register) {
      vm_variable_clear(var);
    }
  }
  //    vm_map_clear(function->variables);
}

/** Add a typed parameter to a function's parameter list. */
void vm_function_parameter_add(vm_function *function, char *name, char *type) {
  VM_ASSERT_NOT_NULL(function);
  VM_ASSERT_NOT_NULL(name);
  VM_ASSERT_NOT_NULL(type);

  vm_list_add(function->parameters, vm_string_clone(name));
  vm_list_item *item = (vm_list_item *)vm_list_get_item(
      function->parameters, vm_list_count(function->parameters) - 1);
  if (item) {
    item->type = vm_string_clone(type);
  }
}

/** Render all function-scoped variables as a heap-allocated string. */
char *vm_function_variables_to_string(vm_function *function) {
  VM_ASSERT_NOT_NULL(function);

  vm_string_buffer *buffer = vm_string_buffer_create_empty();

  vm_list_foreach_of(function->variables_list, char *, name) {
    vm_variable *var =
        (vm_variable *)vm_map_get_value(function->variables, name);
    if (!var)
      continue;
    vm_variable_to_vm_string_buffer(buffer, name, var);
    vm_string_buffer_append(buffer, "\n");
  }
  return vm_string_buffer_free_not_data(buffer);
}
/* ═══════════════════════════════════════════════════════════════════════════
 *  Instructions  —  single bytecode entries
 * ═══════════════════════════════════════════════════════════════════════════
 */

/** Map opcode enum to human-readable string.  Uses a static lookup table. */
const char *vm_instruction_opcode_to_string(vm_instruction_opcode opcode) {
  // TODO use switch
  static const char *strings[] = {"NOP",
                                  "INC",
                                  "DEC",
                                  "ADD",
                                  "MUL",
                                  "DIV",
                                  "EXP",
                                  "POW",
                                  "SUB",
                                  "NEG",
                                  "NOT",
                                  "SET_VAR",
                                  "CREATE_SET_VAR",
                                  "GET_OBJ_VAR",
                                  "SET_OBJ_VAR",
                                  "STACK_POP",
                                  "STACK_PUSH",
                                  "CALL_METHOD",
                                  "CALL_OBJ_METHOD",
                                  "CALL_EXT_METHOD",
                                  "RETURN",
                                  "CONDITION_GTE",
                                  "CONDITION_LTE",
                                  "CONDITION_GT",
                                  "CONDITION_LT",
                                  "CONDITION_NE",
                                  "CONDITION_EQ",
                                  "CONDITION_TRUE",
                                  "CONDITIONAL_AND",
                                  "CONDITIONAL_OR",
                                  "JUMP_IF_TRUE",
                                  "JUMP_IF_FALSE",
                                  "JUMP",
                                  "LABEL",
                                  "YIELD",
                                  "XPATH",
                                  "JSON",
                                  "COMMENT"};

  if ((size_t)opcode > sizeof(strings)) {
    return "UNKNOWN";
  }

  return strings[opcode];
}

/**
 * Allocate a new instruction with all fields initialised.
 * Param strings are NOT cloned — they are assumed to be symbol-tracked.
 */
vm_instruction *vm_instruction_create_with_meta(vm_instruction_opcode opcode,
                                                char *param1, char *param2,
                                                char *param3,
                                                size_t meta_source_line) {
  vm_instruction *instruction = VM_NEW(vm_instruction);
  instruction->opcode = opcode;
  instruction->param1 = param1;
  instruction->param2 = param2;
  instruction->param3 = param3;
  instruction->param_type_1 = NULL;
  instruction->param_type_2 = NULL;
  instruction->param_type_3 = NULL;
  instruction->param_is_constant_1 = false;
  instruction->param_is_constant_2 = false;
  instruction->param_is_constant_3 = false;
  instruction->param_is_register_1 = false;
  instruction->param_is_register_2 = false;
  instruction->param_is_register_3 = false;
  instruction->opJUMP_jump_to_instruction_pos = 0;
  instruction->source_line = meta_source_line;
  instruction->var1 = NULL;
  instruction->var2 = NULL;
  instruction->var3 = NULL;
  instruction->call_method_function = NULL;
  instruction->call_method_verb = NULL;

  return instruction;
}

/** Convenience wrapper: create instruction without source line metadata. */
vm_instruction *vm_instruction_create(vm_instruction_opcode opcode,
                                      char *param1, char *param2,
                                      char *param3) {
  return vm_instruction_create_with_meta(opcode, param1, param2, param3, 0);
}

/** Free an instruction struct (params are symbol-tracked, so not freed here).
 */
void vm_instruction_free(vm_instruction *instruction) {
  VM_ASSERT_NOT_NULL(instruction);
  //
  //    if (instruction->opcode == opCOMMENT && instruction->param1 &&
  //    instruction->param2) {
  ////        vm_string_free(instruction->param1);
  //        vm_string_free(instruction->param2);
  //    }

  VM_FREE(instruction);
}

/** Append opcode string (right-aligned 20 chars) to a string buffer. */
void vm_instruction_opcode_to_vm_string_buffer(vm_string_buffer *buffer,
                                               vm_instruction_opcode opcode) {
  vm_string_buffer_append_format(buffer, "%20s",
                                 vm_instruction_opcode_to_string(opcode));
}

/** Append one instruction parameter's details to a string buffer (for IL
 * dump).
 */
void vm_instruction_params_to_vm_string_buffer(
    vm_string_buffer *buffer, bool param_is_constant, bool param_is_register,
    size_t param_register_id, char *param, char *param_type,
    vm_param_type_class param_type_class) {
  if (param_is_constant || param_is_register) {
    if (param_is_register) {
      vm_string_buffer_append_format(
          buffer, "\t{%sr[%zu]}%10s:%-10s", !param_is_constant ? " " : "c",
          param_register_id, !param ? "" : param,
          !param_type ? (param ? "?" : "") : param_type);
    } else {
      vm_string_buffer_append_format(
          buffer, "\t{%s    }%10s:%-10s", !param_is_constant ? " " : "c",
          !param ? "" : param, !param_type ? (param ? "?" : "") : param_type);
    }
  } else {
    vm_string_buffer_append_format(
        buffer, "\t       %10s:%-10s", !param ? "" : param,
        !param_type ? (param ? "?" : "") : param_type);
  }

  if (param && param_type) {
    vm_string_buffer_append_format(
        buffer, " {%s}",
        vm_variable_param_type_class_to_string_abbreviated(param_type_class));
  } else {
    vm_string_buffer_append_format(
        buffer, "    ",
        vm_variable_param_type_class_to_string_abbreviated(param_type_class));
  }
}

/** Render a full instruction as a heap-allocated string (for IL dump output).
 */
char *vm_instruction_to_string(vm_instruction *instruction) {
  vm_string_buffer *buffer = vm_string_buffer_create_empty();

  switch (instruction->opcode) {
  case opLABEL:
    vm_string_buffer_append_format(buffer, "%s:", instruction->param1);
    break;

  case opCOMMENT:
    vm_string_buffer_append_format(buffer, "// *** Line %s: %s",
                                   instruction->param1, instruction->param2);
    break;

  case opJUMP:
  case opJUMP_IF_FALSE:
  case opJUMP_IF_TRUE:
    vm_instruction_opcode_to_vm_string_buffer(buffer, instruction->opcode);
    vm_instruction_params_to_vm_string_buffer(
        buffer, instruction->param_is_constant_1,
        instruction->param_is_register_1, instruction->param_register_id_1,
        instruction->param1, instruction->param_type_1,
        instruction->param_type_class_1);
    vm_instruction_params_to_vm_string_buffer(
        buffer, instruction->param_is_constant_2,
        instruction->param_is_register_2, instruction->param_register_id_2,
        instruction->param2, instruction->param_type_2,
        instruction->param_type_class_2);
    vm_string_buffer_append_format(
        buffer, "\t%25s [%d]", !instruction->param3 ? "" : instruction->param3,
        instruction->opJUMP_jump_to_instruction_pos);
    break;

  default:
    vm_instruction_opcode_to_vm_string_buffer(buffer, instruction->opcode);
    vm_instruction_params_to_vm_string_buffer(
        buffer, instruction->param_is_constant_1,
        instruction->param_is_register_1, instruction->param_register_id_1,
        instruction->param1, instruction->param_type_1,
        instruction->param_type_class_1);
    vm_instruction_params_to_vm_string_buffer(
        buffer, instruction->param_is_constant_2,
        instruction->param_is_register_2, instruction->param_register_id_2,
        instruction->param2, instruction->param_type_2,
        instruction->param_type_class_2);
    vm_instruction_params_to_vm_string_buffer(
        buffer, instruction->param_is_constant_3,
        instruction->param_is_register_3, instruction->param_register_id_3,
        instruction->param3, instruction->param_type_3,
        instruction->param_type_class_3);
    break;
  }

  return vm_string_buffer_free_not_data(buffer);
}

/* convert.h removed */

/* ═══════════════════════════════════════════════════════════════════════════
 *  Variables  —  runtime values (stack, register, named)
 * ═══════════════════════════════════════════════════════════════════════════
 */

// #define CONFIG_VM_VARIABLES_PREDECLARE

#ifdef CONFIG_VM_VARIABLES_PREDECLARE

#define VM_MAX_VARIABLES 100

static vm_variable variables[VM_MAX_VARIABLES];
static size_t variables_size = 0;

#ifdef VM_ENABLE_VAR_NAME
vm_variable *_vm_variable_create(char *name, const char *filename,
                                 size_t line) {
#else
vm_variable *_vm_variable_create(const char *filename, size_t line) {
#endif

  if (variables_size == VM_MAX_VARIABLES) {
    printf("Max variables reached %d\n", VM_MAX_VARIABLES);
    exit(1);
  }

  //        vm_variable* variable = _vm_memory_malloc(sizeof(vm_variable),
  //        filename, line);
  vm_variable *variable = NULL;

  for (size_t i = 0; i < variables_size; i++) {
    vm_variable *inspect_variable = &variables[i];

    if (inspect_variable->is_used == false) {
      variable = inspect_variable;
      break;
    }
  }

  if (!variable) {
    variable = &variables[variables_size++];
  }

  //    memset(variable, 0, sizeof(vm_variable));
#ifdef VM_ENABLE_VAR_NAME

  // TODO: max length check
  if (vm_string_length(name) >= sizeof(variable->name)) {
    vm_log_error("vm_variable", "Variable name too long name=%s", name);
    return NULL;
  }

  string_copy(variable->name, name);
#endif

  variable->type_class = ptcNONE;
  variable->value.string = NULL;
  variable->is_used = true;
  variable->is_preallocated = true;

  return variable;
}

void vm_variable_dispose(vm_variable *variable) {
  VM_ASSERT_NOT_NULL(variable);

  vm_variable_clear(variable);

  //  vm_string_free(variable->name);
  variable->type_class = ptcNONE;
  variable->is_constant = false;
  variable->is_register = false;
  variable->is_used = false;

  if (!variable->is_preallocated) {
    VM_FREE(variable);
  }
}

void vm_variable_clear(vm_variable *variable) {
  VM_ASSERT_NOT_NULL(variable);

  if (variable->type_class == ptcSTRING && variable->value.string) {
    vm_string_free(variable->value.string);
    variable->value.string = NULL;
  }
}

#else

vm_variable *_vm_variable_create(const char *filename, size_t line) {
  vm_variable *variable = (vm_variable *)malloc(sizeof(vm_variable));
  variable->type_class = ptcNONE;
  variable->value.string = NULL;
  variable->is_used = true;
  variable->is_preallocated = false;
  return variable;
}

void vm_variable_dispose(vm_variable *variable) {
  VM_ASSERT_NOT_NULL(variable);

  vm_variable_clear(variable);

  //  vm_string_free(variable->name);
  variable->type_class = ptcNONE;
  variable->is_constant = false;
  variable->is_register = false;
  variable->is_used = false;

  VM_FREE(variable);
}

void vm_variable_clear(vm_variable *variable) {
  VM_ASSERT_NOT_NULL(variable);

  if (variable->type_class == ptcSTRING && variable->value.string) {
    vm_string_free(variable->value.string);
    variable->value.string = NULL;
  }
}

#endif

/**
 * Assign a constant literal to a vm_variable_value based on its type class.
 * Converts the string representation to the appropriate type.
 *
 * @return true on success, false on unsupported type.
 */
bool vm_variable_value_assign_constant(vm_param_type_class param_type,
                                       char *param_value,
                                       vm_variable_value *value) {
  VM_ASSERT_NOT_NULL(param_value);
  VM_ASSERT_NOT_NULL(value);

  switch (param_type) {
  case ptcBOOLEAN:
    value->boolean = vm_string_equals("true", param_value);
    break;

  case ptcNUMBER:
    value->number = atof(param_value); // convert_string_to_double
    break;

  case ptcSTRING:
    value->string = param_value;
    break;

  case ptcOTHER: {
    // Must be a custom type eg. JSON
    char err_msg[1024];
    snprintf(err_msg, sizeof(err_msg),
             "ptcOTHER as a constant is not supported for param_value %s and "
             "param_type %s",
             param_value, vm_variable_param_type_class_to_string(param_type));
    vm_log_error(err_msg, NULL);
    return false;
  }

  case ptcNONE:
  default: {
    char err_msg[1024];
    snprintf(err_msg, sizeof(err_msg),
             "ptcNONE detected for param_value %s and param_type %s",
             param_value, vm_variable_param_type_class_to_string(param_type));
    vm_log_error(err_msg, NULL);
    return false;
  }
  }

  return true;
}

/**
 * Copy a variable's value into a target value slot.
 * Validates type consistency and clones strings.
 *
 * @return true on success, false on type mismatch.
 */
bool vm_variable_value_assign(vm_variable *source,
                              vm_param_type_class target_type_class,
                              vm_variable_value *target_value) {
  VM_ASSERT_NOT_NULL(source);
  VM_ASSERT_NOT_NULL(target_value);

  if (source->type_class != target_type_class) {
    // This shouldn't ever happen, if it does the compiler verifier should
    // have picked this up
    char err_msg[1024];
    snprintf(err_msg, sizeof(err_msg),
             "Incorrect type assignment. The assigned variable/register of "
             "type %s cannot be set to the stack return value type of type %s",
             vm_variable_param_type_class_to_string(target_type_class),
             vm_variable_param_type_class_to_string(source->type_class));
    vm_log_error(err_msg, NULL);
    return false;
  }

  switch (source->type_class) {
  case ptcBOOLEAN:
    target_value->boolean = source->value.boolean;
    break;

  case ptcNUMBER:
    target_value->number = source->value.number;
    break;

  case ptcSTRING:
    target_value->string = vm_string_clone(source->value.string);
    break;

  case ptcOTHER:
    vm_log_error("ptcOTHER assignment is not supported", NULL);
    return false;

  case ptcNONE:
  default:
    vm_log_error("ptcNONE assignment is not supported", NULL);
    return false;
  }

  return true;
}

/** Convert type class enum to its full string name (e.g. "number"). */
const char *
vm_variable_param_type_class_to_string(vm_param_type_class param_type_class) {
  switch (param_type_class) {
  case ptcNONE:
    return "none";

  case ptcBOOLEAN:
    return "boolean";

  case ptcNUMBER:
    return "number";

  case ptcSTRING:
    return "string";

  case ptcOTHER:
    return "other";

  default:
    return "unknown";
  }
}

/** Convert type class enum to single-char abbreviation (n/s/b/o/?). */
const char *vm_variable_param_type_class_to_string_abbreviated(
    vm_param_type_class param_type_class) {
  switch (param_type_class) {
  case ptcNONE:
    return "?";

  case ptcBOOLEAN:
    return "b";

  case ptcNUMBER:
    return "n";

  case ptcSTRING:
    return "s";

  case ptcOTHER:
    return "o";

  default:
    return "u";
  }
}

/** Parse a type name string ("boolean", "number", "string") into its enum. */
vm_param_type_class
vm_variable_param_type_name_to_param_type_class(char *type) {
  VM_ASSERT_NOT_NULL(type);

  if (vm_string_equals(type, "boolean")) {
    return ptcBOOLEAN;
  }

  if (vm_string_equals(type, "number")) {
    return ptcNUMBER;
  }

  if (vm_string_equals(type, "string")) {
    return ptcSTRING;
  }

  return ptcOTHER;
}

/**
 * Convert a variable to a string representation.
 * WARNING: uses a static buffer for numbers — not thread-safe.
 */
char *vm_variable_to_string(vm_variable *variable) {
  VM_ASSERT_NOT_NULL(variable);

  static char buffer[64];

  switch (variable->type_class) {
  case ptcBOOLEAN:
    return (char *)(variable->value.boolean ? "true" : "false");

  case ptcNUMBER:
    // TODO: use convert
    snprintf(buffer, sizeof(buffer), "%f", variable->value.number);
    return buffer;

  case ptcSTRING:
    return variable->value.string;

  case ptcNONE:
  case ptcOTHER:
  default:
    VM_ASSERT_NOT_NULL(variable->value.other);
    return (char *)variable->value.other;
  }
}

/** Append a variable's value to a string buffer in "name=value" format. */
void vm_variable_to_vm_string_buffer(vm_string_buffer *buffer, const char *name,
                                     vm_variable *var) {
  switch (var->type_class) {
  case ptcNONE:
    vm_string_buffer_append_format(buffer, "%s=<none>", name);
    break;

  case ptcBOOLEAN:
    vm_string_buffer_append_format(buffer, "%s=%s", name,
                                   var->value.boolean ? "true" : "false");
    break;

  case ptcNUMBER:
    vm_string_buffer_append_format(buffer, "%s=%g", name, var->value.number);
    break;

  case ptcSTRING:
    vm_string_buffer_append_format(buffer, "%s=%s", name, var->value.string);
    break;

  case ptcOTHER:
    vm_string_buffer_append_format(buffer, "%s=<other>", name);
    break;
  }
}
