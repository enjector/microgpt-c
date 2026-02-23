/*
 * microgpt_vm.h  —  MicroGPT-C Virtual Machine Engine (public header)
 *
 * Copyright (c) 2026 Ajay Soni.  MIT License.
 *
 * Self-contained header declaring every type, enum, and function used by
 * the TypeScript-ish virtual machine.  Organised into logical sections:
 *
 *   0. Support Layer    — C99 utilities (memory, strings, containers)
 *   1. Variables        — typed value containers (vm_variable)
 *   2. Engine API       — high-level C API (vm_engine)
 *   3. Instructions     — bytecode opcodes and instruction structs
 *   4. Functions        — compiled function descriptors (vm_function)
 *   5. Modules          — compilation units holding functions (vm_module)
 *   6. Compiler         — parser-side code builder (vm_compiler)
 *   7. Runtime          — execution context (vm_module_runtime)
 *   8. Code Generator   — IL emitter (vm_module_generator)
 *   9. Post-Processing  — optimisation / verification passes
 *  10. Parser           — lexer/parser interface
 *  11. Eval             — one-shot expression evaluator (vm_eval)
 *  12. Legacy Runtime   — older vm_queue-based runtime (vm_runtime)
 */

#pragma once

#include <assert.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ═══════════════════════════════════════════════════════════════════════════
 *  0. Support Layer  —  Pure-C99 VM utilities
 *
 *  Lightweight implementations of: vm_memory, string utilities,
 *  vm_string_buffer, vm_list, vm_map, vm_queue, VM_ASSERT/log, verb_context.
 * ═══════════════════════════════════════════════════════════════════════════
 */

/* ── core_result ── */

typedef enum { VM_OK = 0, VM_UNKNOWN = 1, VM_FAIL = 2 } vm_result;

/* ── verb_context (forward decl — full definition at end of section 0) ── */

typedef struct verb_context_t verb_context;

/* ── vm_memory ── */

#define VM_NEW(T) ((T *)calloc(1, sizeof(T)))
#define VM_CREATE(T) ((T *)calloc(1, sizeof(T)))
#define VM_FREE(p)                                                             \
  do {                                                                         \
    free(p);                                                                   \
    (p) = NULL;                                                                \
  } while (0)

/* ── string utilities ── */

/** Duplicate a C string onto the heap.  Returns NULL if `s` is NULL. */
static inline char *vm_string_clone(const char *s) {
  if (!s)
    return NULL;
  size_t n = strlen(s) + 1;
  char *d = (char *)malloc(n);
  if (d)
    memcpy(d, s, n);
  return d;
}

/** Free a heap-allocated string (NULL-safe). */
static inline void vm_string_free(char *s) { free(s); }

/** Return the length of `s`, or 0 if NULL. */
static inline size_t vm_string_length(const char *s) {
  return s ? strlen(s) : 0;
}

/** Compare two C strings for equality.  Returns false if either is NULL. */
static inline bool vm_string_equals(const char *a, const char *b) {
  if (!a || !b)
    return false;
  return strcmp(a, b) == 0;
}

/** Copy up to `len` characters from `src` into `dst` and null-terminate. */
static inline void vm_string_copy_length(char *dst, const char *src,
                                         size_t len) {
  if (dst && src) {
    strncpy(dst, src, len);
    dst[len] = '\0';
  }
}

/** Single-character replacement in a heap-allocated copy.  Caller frees. */
static inline char *vm_string_replace(const char *s, const char *old_sub,
                                      const char *new_sub) {
  if (!s || !old_sub || !new_sub)
    return NULL;
  char *m = vm_string_clone(s);
  char *p = m;
  while (*p) {
    if (*p == old_sub[0])
      *p = new_sub[0];
    p++;
  }
  return m;
}

/** In-place single-character replacement across the entire string. */
static inline void vm_string_replace_inplace(char *s, char old_c, char new_c) {
  if (!s)
    return;
  while (*s) {
    if (*s == old_c)
      *s = new_c;
    s++;
  }
}

/** Convert a decimal string to int (returns 0 on NULL). */
static inline int vm_string_to_int(const char *s) {
  if (!s)
    return 0;
  return atoi(s);
}

/** Convert an int to a heap-allocated decimal string.  Caller frees. */
static inline char *vm_int_to_string(int val) {
  char buf[32];
  snprintf(buf, sizeof(buf), "%d", val);
  return vm_string_clone(buf);
}

/* ── vm_string_buffer (growable C-string builder) ── */

typedef struct {
  char *data;
  size_t len;
  size_t cap;
} vm_string_buffer;

/** Create a new empty string buffer (initial capacity 256). */
static inline vm_string_buffer *vm_string_buffer_create_empty(void) {
  vm_string_buffer *b = (vm_string_buffer *)calloc(1, sizeof(vm_string_buffer));
  if (!b)
    return NULL;
  b->cap = 256;
  b->data = (char *)malloc(b->cap);
  if (b->data)
    b->data[0] = '\0';
  return b;
}

/** Append a null-terminated string, growing as needed. */
static inline void vm_string_buffer_append(vm_string_buffer *b, const char *s) {
  if (!b || !s)
    return;
  size_t slen = strlen(s);
  if (b->len + slen + 1 > b->cap) {
    b->cap = (b->len + slen + 1) * 2;
    b->data = (char *)realloc(b->data, b->cap);
  }
  if (b->data) {
    memcpy(b->data + b->len, s, slen + 1);
    b->len += slen;
  }
}

/** Append a printf-style formatted string. */
static inline void vm_string_buffer_append_format(vm_string_buffer *b,
                                                  const char *fmt, ...) {
  if (!b)
    return;
  char tmp[4096];
  va_list args;
  va_start(args, fmt);
  vsnprintf(tmp, sizeof(tmp), fmt, args);
  va_end(args);
  vm_string_buffer_append(b, tmp);
}

/** Extract the internal buffer and free the struct.  Caller owns vm_result. */
static inline char *vm_string_buffer_free_not_data(vm_string_buffer *b) {
  if (!b)
    return NULL;
  char *r = b->data;
  free(b);
  return r;
}

/** Free both the buffer data and the struct. */
static inline void vm_string_buffer_free(vm_string_buffer *b) {
  if (!b)
    return;
  free(b->data);
  free(b);
}

/* ── VM_ASSERT / log ── */

#define VM_ASSERT_NOT_NULL(x) assert((x) != NULL)

#ifndef VM_ASSERT
#define VM_ASSERT(x) assert((x))
#endif

#define VM_SPRINTF_S snprintf

/** Log an error to stderr with an optional detail string. */
static inline void vm_log_error(const char *msg, const char *detail) {
  if (detail)
    fprintf(stderr, "[vm] ERROR: %s — %s\n", msg, detail);
  else
    fprintf(stderr, "[vm] ERROR: %s\n", msg);
}

#define vm_log_warn(fmt, ...) fprintf(stderr, fmt, ##__VA_ARGS__)

/* ── vm_list (growable void* array with per-item type metadata) ── */

typedef struct {
  void *value;
  char *type;
} vm_list_item;

typedef struct {
  vm_list_item *items;
  size_t count;
  size_t cap;
} vm_list;

/** Create a new empty vm_list (initial capacity 8). */
static inline vm_list *vm_list_create(void) {
  vm_list *s = (vm_list *)calloc(1, sizeof(vm_list));
  if (!s)
    return NULL;
  s->cap = 8;
  s->items = (vm_list_item *)calloc(s->cap, sizeof(vm_list_item));
  return s;
}

/** Append an item, doubling capacity when full. */
static inline void vm_list_add(vm_list *s, void *item) {
  if (!s)
    return;
  if (s->count == s->cap) {
    s->cap *= 2;
    s->items = (vm_list_item *)realloc(s->items, s->cap * sizeof(vm_list_item));
  }
  s->items[s->count].value = item;
  s->items[s->count].type = NULL;
  s->count++;
}

/** Return the item count, or 0 if NULL. */
static inline size_t vm_list_count(const vm_list *s) {
  return s ? s->count : 0;
}

/** Free backing array and struct (does NOT free stored items). */
static inline void vm_list_free(vm_list *s) {
  if (!s)
    return;
  free(s->items);
  free(s);
}

/** Reset count to zero without freeing backing array. */
static inline void vm_list_clear(vm_list *s) {
  if (s)
    s->count = 0;
}

/** Free every item's value and type, then reset count to zero. */
static inline void vm_list_dispose_items(vm_list *s) {
  if (s) {
    for (size_t i = 0; i < s->count; i++) {
      free(s->items[i].value);
      free(s->items[i].type);
    }
    s->count = 0;
  }
}

/** Append all items from src into dst (deep-copies type strings). */
static inline void vm_list_merge(vm_list *dst, vm_list *src) {
  if (!dst || !src)
    return;
  for (size_t i = 0; i < src->count; i++) {
    vm_list_add(dst, src->items[i].value);
    dst->items[dst->count - 1].type =
        src->items[i].type ? vm_string_clone(src->items[i].type) : NULL;
  }
}

/** Check if the vm_list contains a specific pointer value. */
static inline bool vm_list_contains(const vm_list *s, void *item) {
  if (!s)
    return false;
  for (size_t i = 0; i < s->count; i++) {
    if (s->items[i].value == item)
      return true;
  }
  return false;
}

/** Get value at index, or NULL if out of bounds. */
static inline void *vm_list_get_value(const vm_list *s, size_t index) {
  if (!s || index >= s->count)
    return NULL;
  return s->items[index].value;
}

/** Get pointer to item at index, or NULL if out of bounds. */
static inline vm_list_item *vm_list_get_item(const vm_list *s, size_t index) {
  if (!s || index >= s->count)
    return NULL;
  return &s->items[index];
}

/** Return the raw item array. */
static inline vm_list_item *vm_list_enumerable(const vm_list *s) {
  return s ? s->items : NULL;
}

/** Get last item's value cast to Type, or NULL if empty. */
#define vm_list_last_of(seq_ptr, Type)                                         \
  ((Type *)((seq_ptr) && (seq_ptr)->count > 0                                  \
                ? (seq_ptr)->items[(seq_ptr)->count - 1].value                 \
                : NULL))

/**
 * Iterate over vm_list items with a typed loop variable.
 * Usage:  vm_list_foreach_of(errors, vm_module_error*, err) { ... }
 */
#define vm_list_foreach_of(seq_ptr, Type, var)                                 \
  for (size_t _sf_i = 0; (seq_ptr) && _sf_i < (seq_ptr)->count; _sf_i++)       \
    for (int _sf_once_##var = 1; _sf_once_##var; _sf_once_##var = 0)           \
      for (Type var = (Type)(seq_ptr)->items[_sf_i].value; _sf_once_##var;     \
           _sf_once_##var = 0)

/* ── vm_map (open-addressing string-keyed hash map) ── */

typedef struct {
  char *key;
  void *value;
} vm_map_item;

typedef struct {
  vm_map_item *buckets;
  size_t cap;
  size_t count;
} vm_map;

/** DJB2 hash masked to fit within cap (must be power of 2). */
static inline size_t _vm_map_hash(const char *key, size_t cap) {
  size_t h = 5381;
  while (*key)
    h = ((h << 5) + h) ^ (unsigned char)*key++;
  return h & (cap - 1);
}

/** Create a new empty map (initial capacity 16). */
static inline vm_map *vm_map_create(void) {
  vm_map *m = (vm_map *)calloc(1, sizeof(vm_map));
  if (!m)
    return NULL;
  m->cap = 16;
  m->buckets = (vm_map_item *)calloc(m->cap, sizeof(vm_map_item));
  return m;
}

/** Insert or update.  Keys are cloned on insert.  Grows at 75% load. */
static inline void vm_map_set(vm_map *m, const char *key, void *value) {
  if (!m || !key)
    return;
  if (m->count * 4 >= m->cap * 3) {
    size_t new_cap = m->cap * 2;
    vm_map_item *nb = (vm_map_item *)calloc(new_cap, sizeof(vm_map_item));
    if (!nb)
      return;
    for (size_t i = 0; i < m->cap; i++) {
      if (!m->buckets[i].key)
        continue;
      size_t j = _vm_map_hash(m->buckets[i].key, new_cap);
      while (nb[j].key)
        j = (j + 1) & (new_cap - 1);
      nb[j] = m->buckets[i];
    }
    free(m->buckets);
    m->buckets = nb;
    m->cap = new_cap;
  }
  size_t i = _vm_map_hash(key, m->cap);
  while (m->buckets[i].key) {
    if (strcmp(m->buckets[i].key, key) == 0) {
      m->buckets[i].value = value;
      return;
    }
    i = (i + 1) & (m->cap - 1);
  }
  m->buckets[i].key = vm_string_clone(key);
  m->buckets[i].value = value;
  m->count++;
}

/** Look up a bucket by key.  Returns NULL if not found. */
static inline vm_map_item *vm_map_get_item(const vm_map *m, const char *key) {
  if (!m || !key)
    return NULL;
  size_t i = _vm_map_hash(key, m->cap);
  size_t probed = 0;
  while (m->buckets[i].key && probed < m->cap) {
    if (strcmp(m->buckets[i].key, key) == 0)
      return &m->buckets[i];
    i = (i + 1) & (m->cap - 1);
    probed++;
  }
  return NULL;
}

/** Get value for a key, or NULL. */
static inline void *vm_map_get_value(const vm_map *m, const char *key) {
  vm_map_item *it = vm_map_get_item(m, key);
  return it ? it->value : NULL;
}

/** Check whether a key exists. */
static inline bool vm_map_contains(const vm_map *m, const char *key) {
  return vm_map_get_item(m, key) != NULL;
}

/** Free keys, buckets, and map struct (does NOT free values). */
static inline void vm_map_free(vm_map *m) {
  if (!m)
    return;
  for (size_t i = 0; i < m->cap; i++)
    free(m->buckets[i].key);
  free(m->buckets);
  free(m);
}

/** Remove all entries (frees keys only). */
static inline void vm_map_clear(vm_map *m) {
  if (!m)
    return;
  for (size_t i = 0; i < m->cap; i++) {
    if (m->buckets[i].key) {
      free(m->buckets[i].key);
      m->buckets[i].key = NULL;
      m->buckets[i].value = NULL;
    }
  }
  m->count = 0;
}

/** Remove a single entry by key (frees key only, NOT value). */
static inline bool vm_map_remove(vm_map *m, const char *key) {
  if (!m || !key)
    return false;
  size_t idx = _vm_map_hash(key, m->cap);
  for (size_t i = 0; i < m->cap; i++) {
    size_t pos = (idx + i) & (m->cap - 1);
    if (!m->buckets[pos].key)
      return false;
    if (strcmp(m->buckets[pos].key, key) == 0) {
      free(m->buckets[pos].key);
      m->buckets[pos].key = NULL;
      m->buckets[pos].value = NULL;
      m->count--;
      return true;
    }
  }
  return false;
}

/** Copy all entries from src into dst. */
static inline void vm_map_merge(vm_map *dst, vm_map *src) {
  if (!dst || !src)
    return;
  for (size_t i = 0; i < src->cap; i++) {
    if (src->buckets[i].key)
      vm_map_set(dst, src->buckets[i].key, src->buckets[i].value);
  }
}

/**
 * Iterate over occupied vm_map buckets.
 * Usage:  vm_map_foreach_of(map, key, vm_function*, fn) { ... }
 */
#define vm_map_foreach_of(cmap_ptr, key_var, Type, val_var)                    \
  for (size_t _cm_i = 0; (cmap_ptr) && _cm_i < (cmap_ptr)->cap; _cm_i++)       \
    for (const char *key_var = (cmap_ptr)->buckets[_cm_i].key; key_var;        \
         key_var = NULL)                                                       \
      for (int _cm_once_##val_var = 1; _cm_once_##val_var;                     \
           _cm_once_##val_var = 0)                                             \
        for (Type val_var = (Type)(cmap_ptr)->buckets[_cm_i].value;            \
             _cm_once_##val_var; _cm_once_##val_var = 0)

/* ── vm_queue (singly-linked FIFO) ── */

typedef struct _vm_queue_node {
  void *data;
  struct _vm_queue_node *next;
} _vm_queue_node;

typedef struct {
  _vm_queue_node *head;
  _vm_queue_node *tail;
  size_t count;
} vm_queue;

/** Create a new empty vm_queue. */
static inline vm_queue *vm_queue_create(void) {
  return (vm_queue *)calloc(1, sizeof(vm_queue));
}

/** Push an item onto the front. */
static inline void vm_queue_push(vm_queue *q, void *item) {
  if (!q)
    return;
  _vm_queue_node *n = (_vm_queue_node *)malloc(sizeof(_vm_queue_node));
  if (!n)
    return;
  n->data = item;
  n->next = q->head;
  q->head = n;
  if (!q->tail)
    q->tail = n;
  q->count++;
}

/** Pop and return the front item, or NULL if empty. */
static inline void *vm_queue_pop(vm_queue *q) {
  if (!q || !q->head)
    return NULL;
  _vm_queue_node *n = q->head;
  void *v = n->data;
  q->head = n->next;
  if (!q->head)
    q->tail = NULL;
  free(n);
  q->count--;
  return v;
}

/** Return item count, or 0 if NULL. */
static inline size_t vm_queue_count(const vm_queue *q) {
  return q ? q->count : 0;
}

/** Check whether the vm_queue is empty (or NULL). */
static inline bool vm_queue_is_empty(const vm_queue *q) {
  return !q || q->count == 0;
}

/** Free all nodes and the vm_queue struct (does NOT free data). */
static inline void vm_queue_free(vm_queue *q) {
  if (!q)
    return;
  while (q->head) {
    _vm_queue_node *n = q->head;
    q->head = n->next;
    free(n);
  }
  free(q);
}

/** Remove all nodes without freeing data. */
static inline void vm_queue_clear(vm_queue *q) {
  if (!q)
    return;
  while (q->head) {
    _vm_queue_node *n = q->head;
    q->head = n->next;
    free(n);
  }
  q->tail = NULL;
  q->count = 0;
}

/** Remove all nodes AND free each node's data pointer. */
static inline void vm_queue_dispose_items(vm_queue *q) {
  if (!q)
    return;
  while (q->head) {
    _vm_queue_node *n = q->head;
    q->head = n->next;
    free(n->data);
    free(n);
  }
  q->tail = NULL;
  q->count = 0;
}

/** Transfer all items from src into dst.  src is emptied. */
static inline void vm_queue_merge(vm_queue *dst, vm_queue *src) {
  if (!dst || !src)
    return;
  while (src->head) {
    vm_queue_push(dst, src->head->data);
    _vm_queue_node *o = src->head;
    src->head = o->next;
    free(o);
  }
  src->tail = NULL;
  src->count = 0;
}

/**
 * Iterate over vm_queue items.
 * Usage:  vm_queue_foreach_of(labels, char*, lbl) { ... }
 */
#define vm_queue_foreach_of(q_ptr, Type, var)                                  \
  for (_vm_queue_node *_qn_##var = (q_ptr) ? (q_ptr)->head : NULL; _qn_##var;  \
       _qn_##var = _qn_##var->next)                                            \
    for (int _qn_once_##var = 1; _qn_once_##var; _qn_once_##var = 0)           \
      for (Type var = (Type)_qn_##var->data; _qn_once_##var; _qn_once_##var = 0)

/* ═══════════════════════════════════════════════════════════════════════════
 *  verb_context  —  full DSL dispatch layer
 *
 *  Ported from experiments/engines/microgpt-verb.  Provides:
 *    verb_register / verb_unregister / verb_find / verb_exists / verb_enum
 *    verb_compile / verb_exec        (sentence → lookup → dispatch)
 *    verb_definition_dispose / verb_compile_dispose
 *    verb_result                     (error code → string)
 *
 *  Lightweight inline helpers live here; the heavy verb_compile / verb_exec
 *  implementations are in microgpt_vm.c.
 * ═══════════════════════════════════════════════════════════════════════════
 */

/* Verb error codes (kept as positive values — distinct from vm runtime codes)
 */
#define RESULT_CORE_VERB_ERROR_NO_MATCH 2300
#define RESULT_CORE_VERB_ERROR_ALREADY_REGISTERED 2301
#define RESULT_CORE_VERB_ERROR_ARGUMENT_NULL 2302
#define RESULT_CORE_VERB_ERROR_INCORRECT_USAGE 2303
#define RESULT_CORE_VERB_ERROR_ARGUMENT_PARSING 2304
#define RESULT_CORE_VERB_ERROR_EXEC_NOT_ENOUGH_PARAMS 2306

/* Sentence parse ←→ dispatch limit */
#define VM_SENTENCE_MESSAGE_MAX_SIZE 4096

struct verb_context_t {
  vm_map *verb_definition_map;
};

typedef char *(*verb_function)(verb_context *vcontext, const vm_map *args,
                               void *fcontext);
typedef verb_function verb_exec_function; /* alias used by some call sites */

typedef struct verb_definition_t {
  const char *name;              /**< Verb name (heap-allocated). */
  const char *definition_params; /**< Usage string, e.g. "<name> <message>". */
  size_t name_length;            /**< Cached strlen(name) for prefix match. */
  vm_list *param_list;           /**< Sequence of parameter-name strings. */
  size_t params_count;           /**< Number of parameters. */
  verb_function function;        /**< User callback. */
  void *fcontext;                /**< Opaque user context passed to callback. */
  struct verb_definition_t *next_verb_definition; /**< Legacy linked list. */
} verb_definition;

/** Holds the vm_result of verb_compile (parsed sentence + matched verb). */
typedef struct verb_compiled_t {
  int vm_result; /**< VM_OK on success, verb error code otherwise. */
  verb_definition *verb_definition_; /**< Matched verb (may be NULL). */
  vm_map *verb_arg_list; /**< Parsed argument map (param name → value). */
  char *sentence_values; /**< Cloned sentence (owns memory for value ptrs). */
  void *context;         /**< Caller context passed through compile. */
} verb_compiled;

/* ── lightweight inlines ── */

static inline verb_context *verb_context_create(void) {
  verb_context *ctx = (verb_context *)VM_NEW(struct verb_context_t);
  ctx->verb_definition_map = vm_map_create();
  return ctx;
}

static inline verb_definition *verb_find(verb_context *ctx, const char *name) {
  return ctx ? (verb_definition *)vm_map_get_value(ctx->verb_definition_map,
                                                   name)
             : NULL;
}

static inline bool verb_exists(verb_context *ctx, const char *name) {
  return verb_find(ctx, name) != NULL;
}

static inline vm_map *verb_enum(verb_context *ctx) {
  return ctx ? ctx->verb_definition_map : NULL;
}

/* ── macros ── */

#define verb(function)                                                         \
  char *function(verb_context *vcontext, const vm_map *args, void *context)
#define verb_arg(_name) (const char *)vm_map_get_value(args, _name)

/* ── prototypes (implemented in microgpt_vm.c) ── */

void verb_definition_dispose(verb_definition *def);
void verb_context_dispose(verb_context *ctx);
int verb_register(verb_context *ctx, const char *name, const char *params,
                  verb_function fn, void *fctx);
int verb_unregister(verb_context *ctx, const char *name);
verb_compiled *verb_compile(verb_context *ctx, const char *sentence,
                            void *context);
void verb_compile_dispose(verb_compiled *compiled);
int verb_exec(verb_context *ctx, const char *sentence, char **response);
const char *verb_result(int code);
/* ═══════════════════════════════════════════════════════════════════════════
 *  1. Variables  —  typed value containers
 *
 *  Every runtime value in the VM is represented as a vm_variable.
 *  Variables carry a type-class tag, a union value, and bookkeeping flags.
 * ═══════════════════════════════════════════════════════════════════════════
 */

/** When _DEBUG_TRACE is defined, variables carry a human-readable name. */
#ifdef _DEBUG_TRACE
#define VM_ENABLE_VAR_NAME
#endif

#ifdef VM_ENABLE_VAR_NAME
#define VM_VARIABLE_NAME_MAX_LENGTH 64
#endif

/** Type classification for VM values and function parameters. */
typedef enum vm_param_type_class_t {
  ptcNONE,    /**< Untyped / void. */
  ptcBOOLEAN, /**< Boolean (true / false). */
  ptcNUMBER,  /**< IEEE 754 double. */
  ptcSTRING,  /**< Heap-allocated C string. */
  ptcOTHER    /**< Opaque pointer to caller-owned data. */
} vm_param_type_class;

/** Convert a type-class enum to its full string name (e.g. "number"). */
const char *
vm_variable_param_type_class_to_string(vm_param_type_class param_type_class);

/** Convert a type-class enum to a short abbreviation (e.g. "n"). */
const char *vm_variable_param_type_class_to_string_abbreviated(
    vm_param_type_class param_type_class);

/** Map a type-name string (e.g. "number") to its type-class enum. */
vm_param_type_class vm_variable_param_type_name_to_param_type_class(char *type);

/** Tagged union holding the actual runtime value. */
typedef union vm_variable_value_t {
  bool boolean;  /**< ptcBOOLEAN value. */
  double number; /**< ptcNUMBER value. */
  char *string;  /**< ptcSTRING value (heap-allocated). */
  void *other;   /**< ptcOTHER value (caller-owned). */
} vm_variable_value;

/** A single VM variable: type tag + value + lifecycle flags. */
typedef struct vm_variable_t {
#ifdef VM_ENABLE_VAR_NAME
  char name[VM_VARIABLE_NAME_MAX_LENGTH]; /**< Debug-only variable name. */
#endif
  vm_param_type_class type_class; /**< Runtime type discriminator. */
  vm_variable_value value;        /**< The stored value. */
  bool is_register;               /**< True if this is a temporary register. */
  bool is_constant;               /**< True if immutable after assignment. */
  bool is_used;                   /**< True if referenced during execution. */
  bool is_preallocated;           /**< True if managed by pool; skip dispose. */
} vm_variable;

/* Variable lifecycle */
#ifdef VM_ENABLE_VAR_NAME
vm_variable *_vm_variable_create(char *name, const char *filename, size_t line);
#define vm_variable_create(name) _vm_variable_create(name, __FILE__, __LINE__)
#else
vm_variable *_vm_variable_create(const char *filename, size_t line);
#define vm_variable_create() _vm_variable_create(__FILE__, __LINE__)
#endif

/** Free a variable and its owned string value (if any). */
void vm_variable_dispose(vm_variable *variable);

/** Reset a variable's value to zero/NULL without freeing the struct. */
void vm_variable_clear(vm_variable *variable);

/** Parse a string constant and assign it into a vm_variable_value union. */
bool vm_variable_value_assign_constant(vm_param_type_class param_type,
                                       char *param_value,
                                       vm_variable_value *value);

/** Copy `source`'s value into `target_value`, performing type coercion. */
bool vm_variable_value_assign(vm_variable *source,
                              vm_param_type_class target_type_class,
                              vm_variable_value *target_value);

/** Render a variable's value as a heap-allocated string. Caller owns vm_result.
 */
char *vm_variable_to_string(vm_variable *variable);

/** Append "name=value\n" to a vm_string_buffer for diagnostics. */
void vm_variable_to_vm_string_buffer(vm_string_buffer *buffer, const char *name,
                                     vm_variable *var);

/* ═══════════════════════════════════════════════════════════════════════════
 *  2. Engine API  —  high-level C interface for embedding the VM
 *
 *  Typical usage:
 *    vm_engine *e = vm_engine_create();
 *    if (vm_engine_load(e, source_code) == 0) {
 *        vm_engine_run(e, "main");
 *        printf("vm_result = %f\n", vm_engine_result_number(e));
 *    }
 *    vm_engine_dispose(e);
 * ═══════════════════════════════════════════════════════════════════════════
 */

/** Opaque engine handle. */
typedef struct vm_engine_t vm_engine;

/**
 * Native function callback — invoked when VM script calls a registered
 * C function.  The return value (double) is placed into the engine's
 * vm_result slot.
 */
typedef double (*vm_native_fn)(int argc, const double *argv);

/* Lifecycle */

/** Create a new VM engine instance.  Returns NULL on allocation failure. */
vm_engine *vm_engine_create(void);

/** Dispose of an engine and all resources it owns (module, runtime, etc). */
void vm_engine_dispose(vm_engine *e);

/* Loading */

/**
 * Parse and compile TypeScript-ish source into the engine.
 * @param e       Engine instance.
 * @param source  Null-terminated source string.
 * @return 0 on success, non-zero on parse/compile error.
 *         Call vm_engine_last_error() for a description on failure.
 */
int vm_engine_load(vm_engine *e, const char *source);

/* Native function registration */

/**
 * Register a native C function callable from VM scripts.
 * Must be called before vm_engine_load() if the function is referenced.
 *
 * @param name  Function name as used in the script.
 * @param fn    C callback.
 */
void vm_engine_register_fn(vm_engine *e, const char *name, vm_native_fn fn);

/* Execution */

/**
 * Execute a named function in the loaded module.
 * @param fn_name  E.g. "main" or any declared function name.
 * @return 0 on success, non-zero on runtime error.
 */
int vm_engine_run(vm_engine *e, const char *fn_name);

/* Results */

/** Numeric return value from the last vm_engine_run() call. */
double vm_engine_result_number(const vm_engine *e);

/**
 * String return value from the last vm_engine_run() call.
 * Pointer valid until the next engine call or vm_engine_dispose().
 */
const char *vm_engine_result_string(const vm_engine *e);

/** Boolean return value (0 = false, 1 = true). */
int vm_engine_result_bool(const vm_engine *e);

/* Error reporting */

/** Last error message.  Valid after a non-zero return from load/run. */
const char *vm_engine_last_error(const vm_engine *e);

/* Diagnostics */

/**
 * Dump the compiled IL (intermediate language) of the loaded module
 * to stdout.  Useful for debugging generated bytecode.
 */
void vm_engine_dump_il(const vm_engine *e);

/* ═══════════════════════════════════════════════════════════════════════════
 *  3. Instructions  —  bytecode opcodes and instruction descriptors
 * ═══════════════════════════════════════════════════════════════════════════
 */

/** Every VM bytecode operation. */
typedef enum vm_instruction_opcode_t {
  opNOP,             /**< No operation. */
  opINC,             /**< Increment variable. */
  opDEC,             /**< Decrement variable. */
  opADD,             /**< Add two operands. */
  opMUL,             /**< Multiply two operands. */
  opDIV,             /**< Divide. */
  opEXP,             /**< Exponentiation. */
  opPOW,             /**< Power (synonym). */
  opSUB,             /**< Subtract. */
  opNEG,             /**< Negate. */
  opNOT,             /**< Logical NOT. */
  opSET_VAR,         /**< Assign to existing variable. */
  opCREATE_SET_VAR,  /**< Declare and assign a new variable. */
  opGET_OBJ_VAR,     /**< Read object property. */
  opSET_OBJ_VAR,     /**< Write object property. */
  opSTACK_POP,       /**< Pop value from runtime stack. */
  opSTACK_PUSH,      /**< Push value onto runtime stack. */
  opCALL_METHOD,     /**< Call a function defined in the same module. */
  opCALL_OBJ_METHOD, /**< Call an object method. */
  opCALL_EXT_METHOD, /**< Call an external/native function. */
  opRETURN,          /**< Return from function. */
  opCONDITION_GTE,   /**< Compare >=. */
  opCONDITION_LTE,   /**< Compare <=. */
  opCONDITION_GT,    /**< Compare >. */
  opCONDITION_LT,    /**< Compare <. */
  opCONDITION_NE,    /**< Compare !=. */
  opCONDITION_EQ,    /**< Compare ==. */
  opCONDITION_TRUE,  /**< Test truthiness. */
  opCONDITIONAL_AND, /**< Logical AND. */
  opCONDITIONAL_OR,  /**< Logical OR. */
  opJUMP_IF_TRUE,    /**< Conditional jump (true). */
  opJUMP_IF_FALSE,   /**< Conditional jump (false). */
  opJUMP,            /**< Unconditional jump. */
  opLABEL,           /**< Label target for jumps. */
  opYIELD,           /**< Co-routine yield. */
  opXPATH,           /**< XPath evaluation. */
  opJSON,            /**< JSON access. */
  opCOMMENT          /**< Source-level comment (no-op at runtime). */
} vm_instruction_opcode;

/** Convert an opcode enum to its string name. */
const char *vm_instruction_opcode_to_string(vm_instruction_opcode opcode);

/** Forward declaration (full definition in Functions section). */
typedef struct vm_function_t vm_function;

/**
 * A single bytecode instruction with up to three operands.
 *
 * Each operand (param1..3) carries its string representation, an optional
 * type name and type-class, and resolution metadata (constant, register,
 * or named variable).
 */
typedef struct vm_instruction_t {
  vm_instruction_opcode opcode; /**< The operation to perform. */

  /* Operand string values */
  char *param1;
  char *param2;
  char *param3;

  /* Operand type names (e.g. "number", "string") */
  char *param_type_1;
  char *param_type_2;
  char *param_type_3;

  /* Operand type classifications */
  vm_param_type_class param_type_class_1;
  vm_param_type_class param_type_class_2;
  vm_param_type_class param_type_class_3;

  /* Constant flags */
  bool param_is_constant_1;
  bool param_is_constant_2;
  bool param_is_constant_3;

  /* Register flags and IDs */
  bool param_is_register_1;
  bool param_is_register_2;
  bool param_is_register_3;
  size_t param_register_id_1;
  size_t param_register_id_2;
  size_t param_register_id_3;

  size_t opJUMP_jump_to_instruction_pos; /**< Resolved jump target index. */
  size_t source_line;                    /**< Original source line number. */

  /* Resolved variable pointers (set during post-processing) */
  vm_variable *var1;
  vm_variable *var2;
  vm_variable *var3;

  vm_function *call_method_function; /**< Resolved callee (opCALL_METHOD). */
  void *call_method_verb;            /**< Resolved verb (opCALL_EXT_METHOD). */
} vm_instruction;

/** Create an instruction with source-line metadata. */
vm_instruction *vm_instruction_create_with_meta(vm_instruction_opcode opcode,
                                                char *param1, char *param2,
                                                char *param3,
                                                size_t source_line);

/** Create an instruction (source line defaults to 0). */
vm_instruction *vm_instruction_create(vm_instruction_opcode opcode,
                                      char *param1, char *param2, char *param3);

/** Free an instruction and all its owned string fields. */
void vm_instruction_free(vm_instruction *instruction);

/** Render an instruction as "OPCODE param1, param2, param3". Caller frees. */
char *vm_instruction_to_string(vm_instruction *instruction);

/* ═══════════════════════════════════════════════════════════════════════════
 *  4. Functions  —  compiled function descriptors
 *
 *  A vm_function holds the bytecode (instruction list), parameter
 *  declarations, local variables, registers, constants, and type traits
 *  for a single compiled function.
 * ═══════════════════════════════════════════════════════════════════════════
 */

/** Compile-time type annotation for symbols. */
typedef struct {
  char *type;       /**< Type name (e.g. "number"). */
  bool is_constant; /**< True if the symbol is declared const. */
} vm_type_trait;

/** Create a new type trait.  `type` is cloned. */
vm_type_trait *vm_type_trait_create(char *type, bool is_constant);

/** Dispose a type trait and free its type string. */
void vm_type_trait_dispose(vm_type_trait *type_trait);

/**
 * A compiled function within a VM module.
 *
 * Contains the instruction stream plus all metadata needed for execution:
 * parameters, variables, constants, registers, type traits, and labels.
 */
struct vm_function_t {
  char *name;            /**< Function name (heap-allocated). */
  vm_list *parameters;   /**< Parameter list (vm_variable*). */
  vm_list *instructions; /**< Bytecode (vm_instruction*). */

  vm_param_type_class return_type_class; /**< Return type classification. */
  char *return_type;                     /**< Return type name string. */

  vm_list *symbols;          /**< Source-level symbol records. */
  vm_queue *tracking_labels; /**< Label stack for loop/branch generation. */
  vm_list *label_names;      /**< All label name strings. */
  vm_list *register_names;   /**< Temporary register name strings. */
  vm_map *trait_types;       /**< Maps symbol name → vm_type_trait*. */

  bool is_executing; /**< True while running (for recursion guard). */

  vm_list *registers;      /**< Temporary register pool (vm_variable*). */
  vm_map *constants;       /**< Named constants (vm_variable*). */
  vm_map *variables;       /**< Named variables (vm_variable*). */
  vm_list *variables_list; /**< Variable list preserving declaration order. */

  size_t instruction_pointer; /**< Current IP during execution. */
};

/** Create a new function with the given name. */
vm_function *vm_function_create(char *name);

/** Free a function and all its owned sub-structures. */
void vm_function_dispose(vm_function *function);

/** Reset a function's variables and registers for re-execution. */
void vm_function_clear(vm_function *function);

/** Add a typed parameter to the function's parameter list. */
void vm_function_parameter_add(vm_function *function, char *name, char *type);

/** Render all variables as "name=value\n" pairs. Caller frees vm_result. */
char *vm_function_variables_to_string(vm_function *function);

/* ═══════════════════════════════════════════════════════════════════════════
 *  5. Modules  —  compilation units
 *
 *  A vm_module is the top-level container produced by the compiler.
 *  It holds a map of functions and an error list.
 * ═══════════════════════════════════════════════════════════════════════════
 */

/** Compilation error record. */
typedef struct {
  char *message;             /**< Human-readable error description. */
  size_t source_line_number; /**< 1-based line number in source. */
  size_t source_line_column; /**< 1-based column number in source. */
} vm_module_error;

/** A compiled module containing all functions from a single source. */
typedef struct {
  verb_context *verb_context_; /**< Optional verb dispatch context. */
  vm_map *functions;           /**< Maps function name → vm_function*. */
  vm_list *errors;             /**< List of vm_module_error*. */
  vm_list *functions_list;     /**< Functions in declaration order. */
} vm_module;

/** Create a new empty module (verb_context may be NULL). */
vm_module *vm_module_create(verb_context *verb_context);

/** Free a module and all its functions and errors. */
void vm_module_dispose(vm_module *module);

/** Look up a function by name.  Returns NULL if not found. */
vm_function *vm_module_fetch_function(vm_module *module, const char *name);

/** Record a compilation error on the module. */
void vm_module_error_add(vm_module *module, size_t source_line_number,
                         size_t source_line_column, char *message);

/** Free a single error record. */
void vm_module_error_dispose(vm_module_error *error);

/** Render the entire module's IL as a string. Caller frees vm_result. */
char *vm_module_to_string(vm_module *module);

/* ═══════════════════════════════════════════════════════════════════════════
 *  6. Compiler  —  parser-side code builder (vm_compiler)
 *
 *  Used by the legacy Flex/Bison-based parser to build functions
 *  one instruction at a time.
 * ═══════════════════════════════════════════════════════════════════════════
 */

/** State for the legacy single-module compiler. */
typedef struct {
  const char *vm_parser_state_input;            /**< Source text pointer. */
  size_t vm_parser_state_input_len;             /**< Source text length. */
  size_t vm_parser_state_input_index;           /**< Current scan index. */
  size_t vm_parser_state_input_index_previous;  /**< Previous scan index. */
  size_t vm_parser_state_input_lineno;          /**< Current line number. */
  size_t vm_parser_state_input_lineno_previous; /**< Previous line number. */

  vm_module *vm_parser_state_current_module;     /**< Module being built. */
  vm_function *vm_parser_state_current_function; /**< Function being built. */
} vm_compiler;

/** Create a new compiler instance. */
vm_compiler *vm_compiler_create(void);

/** Dispose a compiler and its internal state. */
void vm_compiler_dispose(vm_compiler *generator);

/** Begin compilation with a specified default return type. */
void vm_compiler_begin(vm_compiler *generator,
                       vm_param_type_class return_type_class);

/** Finalise compilation. */
void vm_compiler_end(vm_compiler *generator);

/** Begin a new function definition within the current module. */
vm_function *vm_compiler_function_begin(vm_compiler *generator, char *name);

/** End the current function definition. */
void vm_compiler_function_end(vm_compiler *generator);

/** Add a typed parameter to the current function. */
void vm_compiler_function_parameter(vm_compiler *generator, char *name,
                                    char *type);

/** Set the return type of the current function. */
void vm_compiler_function_return_type(vm_compiler *generator, char *type);

/** Emit a bytecode instruction into the current function. */
void vm_compiler_function_emit(vm_compiler *generator,
                               vm_instruction_opcode opcode, char *param1,
                               char *param2, char *param3);

/** Emit a comment instruction. */
void vm_compiler_function_emit_comment(vm_compiler *generator, char *message);

/** Allocate a new temporary register name (e.g. "r$0"). */
char *vm_compiler_tmp_register_create(vm_compiler *generator);

/** Reset the temporary register counter. */
void vm_compiler_tmp_registers_reset(vm_compiler *generator);

/** Allocate a new unique label name (e.g. "L0"). */
char *vm_compiler_tmp_label_create(vm_compiler *generator);

/** Push a label onto the tracking stack. */
void vm_compiler_tracking_labels_push(vm_compiler *generator, char *label);

/** Pop a label from the tracking stack. Caller frees vm_result. */
char *vm_compiler_tracking_labels_pop(vm_compiler *generator);

/** Clear all labels from the tracking stack. */
void vm_compiler_tracking_labels_clear(vm_compiler *generator);

/** Register a symbol name and return its interned copy. */
char *vm_compiler_symbol_track(vm_compiler *generator, char *symbol);

/** Get the type-trait string for a tracked symbol. */
char *vm_compiler_trait_type_get(vm_compiler *generator, char *symbol);

/** Set the type-trait for a tracked symbol. */
void vm_compiler_trait_type_set(vm_compiler *generator, char *symbol,
                                char *type, bool is_constant);

/* ═══════════════════════════════════════════════════════════════════════════
 *  7. Runtime  —  module execution context (vm_module_runtime)
 *
 *  Manages the execution stack and dispatches external method callbacks.
 * ═══════════════════════════════════════════════════════════════════════════
 */

/** Callback type for handling opCALL_EXT_METHOD during execution. */
struct vm_module_runtime_t;
typedef void (*vm_call_ext_method_callback)(struct vm_module_runtime_t *runtime,
                                            vm_function *function);

/* Runtime error codes */
#define RESULT_CORE_VM_RUNTIME_METHOD_DOESNT_EXIST -10
#define RESULT_CORE_VM_RUNTIME_RECURSION_NOT_SUPPORTED -11
#define RESULT_CORE_VM_RUNTIME_INCORRECT_PARAMS -12
#define RESULT_CORE_VERB_ERROR -13
#define RESULT_CORE_VM_RUNTIME_CALL_EXT_METHOD_CALLBACK_NOT_SET -14
#define RESULT_CORE_VM_RUNTIME_JUMP_INDEX_INVALID -15
#define RESULT_CORE_VM_EMPTY_STACK -16
#define RESULT_CORE_VM_RUNTIME_ERROR -17

/** Maximum depth of the runtime evaluation stack. */
#define VM_MAX_STACK 100

/** Execution context for a compiled module. */
typedef struct vm_module_runtime_t {
  vm_module *module;                /**< The module to execute. */
  vm_variable *stack[VM_MAX_STACK]; /**< Fixed-size eval stack. */
  size_t stack_size;                /**< Current stack depth. */
  vm_call_ext_method_callback call_ext_method_callback; /**< Ext dispatch. */
} vm_module_runtime;

/** Create a runtime bound to a module. */
vm_module_runtime *vm_module_runtime_create(vm_module *module);

/** Dispose a runtime (does not dispose the module). */
void vm_module_runtime_dispose(vm_module_runtime *runtime);

/** Clear the runtime stack for re-execution. */
void vm_module_runtime_clear(vm_module_runtime *runtime);

/** Push a boolean onto the evaluation stack. */
void vm_module_runtime_stack_push_boolean(vm_module_runtime *runtime,
                                          bool value);

/** Push a string onto the evaluation stack (cloned internally). */
void vm_module_runtime_stack_push_string(vm_module_runtime *runtime,
                                         const char *value);

/** Push a number onto the evaluation stack. */
void vm_module_runtime_stack_push_number(vm_module_runtime *runtime,
                                         double value);

/** Push an opaque pointer onto the evaluation stack. */
void vm_module_runtime_stack_push_other(vm_module_runtime *runtime,
                                        void *other);

/** Pop the top value off the evaluation stack. */
vm_result vm_module_runtime_stack_pop(vm_module_runtime *runtime,
                                      vm_variable **out_variable);

/** Set the external method dispatch callback. */
void vm_module_runtime_set_call_ext_method_callback(
    vm_module_runtime *runtime, vm_call_ext_method_callback callback);

/** Execute a function within this runtime. */
vm_result vm_module_runtime_run(vm_module_runtime *runtime,
                                vm_function *function);

/* ═══════════════════════════════════════════════════════════════════════════
 *  8. Code Generator  —  IL emitter (vm_module_generator)
 *
 *  Used by the Flex/Bison module parser to build a vm_module from source.
 *  Supports deferred code fragments for complex control flow.
 * ═══════════════════════════════════════════════════════════════════════════
 */

/** Code generator state. */
typedef struct vm_module_generator_t {
  vm_module *vm_module_parser_state_current_module; /**< Module target. */
  vm_function
      *vm_module_parser_state_current_function; /**< Current function. */

  size_t meta_state_input_index_previous;       /**< Tracks source position. */
  size_t meta_state_input_line_number_previous; /**< Tracks source line. */

  vm_queue *code_fragments; /**< Active deferred code fragment queues. */
  vm_queue *completed_code_fragments; /**< Completed deferred fragments. */
} vm_module_generator;

/** Create a generator bound to a module. */
vm_module_generator *vm_module_generator_create(vm_module *module);

/** Dispose a generator and its deferred fragment queues. */
void vm_module_generator_dispose(vm_module_generator *generator);

/** Begin a new function definition. */
vm_function *vm_module_generator_function_begin(vm_module_generator *generator,
                                                char *name);

/** End the current function definition. */
void vm_module_generator_function_end(vm_module_generator *generator);

/** Add a typed parameter to the current function. */
void vm_module_generator_function_parameter(vm_module_generator *generator,
                                            char *name, char *type);

/** Set the return type for a named function. */
void vm_module_generator_function_return_type_set(
    vm_module_generator *generator, char *name, char *type);

/** Get the return type for a named function. */
char *
vm_module_generator_function_return_type_get(vm_module_generator *generator,
                                             char *name);

/** Emit an instruction with source-position metadata. */
void vm_module_generator_function_emit_with_meta(vm_module_generator *generator,
                                                 vm_instruction_opcode opcode,
                                                 char *param1, char *param2,
                                                 char *param3, char *source,
                                                 size_t line_number,
                                                 size_t source_index);

/** Emit an instruction without metadata. */
void vm_module_generator_function_emit(vm_module_generator *generator,
                                       vm_instruction_opcode opcode,
                                       char *param1, char *param2,
                                       char *param3);

/** Emit a comment instruction with source-position metadata. */
void vm_module_generator_function_emit_comment_with_meta(
    vm_module_generator *generator, char *message, char *meta_source,
    size_t meta_line_number, size_t meta_source_index);

/** Emit a comment instruction without metadata. */
void vm_module_generator_function_emit_comment(vm_module_generator *generator,
                                               char *message);

/** Allocate a temporary register name (e.g. "r$0"). */
char *vm_module_generator_tmp_register_create(vm_module_generator *generator);

/** Reset the temporary register counter. */
void vm_module_generator_tmp_registers_reset(vm_module_generator *generator);

/** Allocate a unique label name (e.g. "L0"). */
char *vm_module_generator_tmp_label_create(vm_module_generator *generator);

/** Push a label onto the tracking stack. */
void vm_module_generator_tracking_labels_push(vm_module_generator *generator,
                                              char *label);

/** Pop a label from the tracking stack. */
char *vm_module_generator_tracking_labels_pop(vm_module_generator *generator);

/** Clear all tracking labels. */
void vm_module_generator_tracking_labels_clear(vm_module_generator *generator);

/** Begin a deferred code fragment (for out-of-order emission). */
void vm_module_generator_defer_push_begin(vm_module_generator *generator);

/** End and save the current deferred code fragment. */
void vm_module_generator_defer_push_end(vm_module_generator *generator);

/** Pop and emit all deferred code fragments into the current function. */
void vm_module_generator_defer_pop(vm_module_generator *generator);

/** Register a symbol name and return its interned copy. */
const char *vm_module_generator_symbol_track(vm_module_generator *generator,
                                             const char *symbol);

/** Get the type-trait string for a symbol. */
char *vm_module_generator_trait_type_get(vm_module_generator *generator,
                                         char *symbol);

/** Set the type-trait for a symbol. */
void vm_module_generator_trait_type_set(vm_module_generator *generator,
                                        char *symbol, char *type,
                                        bool is_constant);

/* ═══════════════════════════════════════════════════════════════════════════
 *  9. Post-Processing & Verification
 *
 *  Compiler passes that run after initial code generation: jump resolution,
 *  variable resolution, type checking, and null-pointer verification.
 * ═══════════════════════════════════════════════════════════════════════════
 */

/** Resolve jumps, intern variables/constants, and optimise the module. */
void vm_compiler_post_processing_process(vm_module *module);

/** Verify that all required fields are non-NULL across the module. */
void vm_compiler_verifier_verify(vm_module *module);

/** Verify non-NULL invariants for the module-generator pipeline. */
void vm_module_generator_verifier_VM_ASSERT_NOT_NULL(vm_module *module);

/** Post-processing pass for the module-generator pipeline. */
void vm_module_generator_post_processing_process(vm_module *module);

/* ═══════════════════════════════════════════════════════════════════════════
 *  10. Parser  —  lexer / parser interface
 * ═══════════════════════════════════════════════════════════════════════════
 */

/** Parse source text and produce a module (using the legacy compiler). */
vm_module *vm_parser_load(const char *text, vm_param_type_class return_type);

/** Fetch the next character from the compiler's input stream. */
int vm_parser_char_fetch_next(vm_compiler *generator);

/* ═══════════════════════════════════════════════════════════════════════════
 *  10b. Module Parser  —  Flex/Bison based module parser
 * ═══════════════════════════════════════════════════════════════════════════
 */

/** State for the Flex/Bison module parser. */
typedef struct {
  const char *vm_module_parser_state_input;        /**< Source text pointer. */
  size_t vm_module_parser_state_input_len;         /**< Source text length. */
  size_t vm_module_parser_state_input_index;       /**< Current scan index. */
  size_t vm_module_parser_state_input_line_number; /**< Current line. */
  size_t vm_module_parser_state_input_line_column; /**< Current column. */

  vm_module_generator *generator; /**< Associated code generator. */
} vm_module_parser;

/**
 * Compile source code into a module using the Flex/Bison parser.
 *
 * @param verb_context_  Optional verb context (may be NULL).
 * @param source         Null-terminated source string.
 * @param out_module     Output: the compiled module.
 * @return VM_OK on success.
 */
vm_result vm_module_parser_generate(verb_context *verb_context_,
                                    const char *source, vm_module **out_module);

/** Compile source code into a module (convenience wrapper). */
vm_result vm_module_compile(verb_context *verb_context, const char *source,
                            vm_module **out_module);

/* ═══════════════════════════════════════════════════════════════════════════
 *  11. Eval  —  one-shot expression evaluator
 *
 *  Convenience layer that compiles + executes a single expression and
 *  extracts the vm_result in one call.
 * ═══════════════════════════════════════════════════════════════════════════
 */

/** Persistent evaluation context (holds module + runtime + function). */
typedef struct vm_eval_t {
  vm_module *module;          /**< Compiled module. */
  vm_module_runtime *runtime; /**< Execution runtime. */
  vm_function *function;      /**< The compiled eval function. */

  vm_param_type_class result_type; /**< Expected return type. */
  vm_variable *vm_result;          /**< Result variable after execution. */
} vm_eval;

/** Evaluation vm_result wrapper. */
typedef struct vm_eval_result_t {
  vm_variable *return_var; /**< The return variable (caller disposes). */
} vm_eval_result;

/** Compile an expression into a reusable eval context. */
vm_result vm_eval_create(const char *expression,
                         vm_param_type_class return_type, vm_eval **out_eval);

/** Execute the eval context and retrieve the vm_result. */
vm_result vm_eval_run(vm_eval *eval, vm_eval_result **out_result);

/** Dispose an eval context and its compiled module. */
vm_result vm_eval_dispose(vm_eval *eval);

/** Dispose an eval vm_result. */
vm_result vm_eval_result_dispose(vm_eval_result *vm_result);

/* ═══════════════════════════════════════════════════════════════════════════
 *  12. Legacy Runtime  —  older vm_queue-based runtime (vm_runtime)
 *
 *  Pre-dates vm_module_runtime.  Uses a linked-list vm_queue instead of a
 *  fixed-size stack.  Retained for backward compatibility.
 * ═══════════════════════════════════════════════════════════════════════════
 */

/** Legacy runtime with vm_queue-based stack. */
typedef struct {
  vm_module *module; /**< The module to execute. */
  vm_queue *stack;   /**< Evaluation stack (linked-list). */
} vm_runtime;

/** Create a legacy runtime bound to a module. */
vm_runtime *vm_runtime_create(vm_module *module);

/** Dispose a legacy runtime. */
void vm_runtime_dispose(vm_runtime *runtime);

/** Clear the legacy runtime stack for re-execution. */
void vm_runtime_clear(vm_runtime *runtime);

/** Push a boolean onto the legacy stack. */
void vm_runtime_stack_push_boolean(vm_runtime *runtime, bool value);

/** Push a string onto the legacy stack. */
void vm_runtime_stack_push_string(vm_runtime *runtime, const char *value);

/** Push a number onto the legacy stack. */
void vm_runtime_stack_push_number(vm_runtime *runtime, double value);

/** Push an opaque pointer onto the legacy stack. */
void vm_runtime_stack_push_other(vm_runtime *runtime, void *other);

/** Pop the top value off the legacy stack. */
vm_result vm_runtime_stack_pop(vm_runtime *runtime, vm_variable **out_variable);

/** Execute a function using the legacy runtime. */
bool vm_runtime_run(vm_runtime *runtime, vm_function *function);
