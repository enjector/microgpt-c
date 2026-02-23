#include "../src/microgpt_vm.h"
#include "bench.h"

// --- Added dynamic file loader ---
#include <stdio.h>
#include <stdlib.h>

static char *load_file_content(const char *filepath) {
  FILE *f = fopen(filepath, "rb");
  if (!f)
    return NULL;
  fseek(f, 0, SEEK_END);
  long fsize = ftell(f);
  fseek(f, 0, SEEK_SET);
  char *string = malloc(fsize + 1);
  fread(string, 1, fsize, f);
  fclose(f);
  string[fsize] = 0;
  return string;
}
// ---------------------------------

// --- vm_benchmarks.c ---
/*
 * vm_benchmarks.c
 *
 * Copyright 2024 Enjector Software, Ltd. All Rights Reserved.
 *
 */

#include <assert.h>

static int benchmark_simple1() {
  int count = 5000000;

#ifdef _DEBUG
  count /= 10;
#endif

  int n = count;

  char *source = load_file_content("resources/vm/runtime/runtime_simple1.ts");
  assert(source != NULL);

  vm_module *module = NULL;
  result r = vm_module_compile(NULL, source, &module);
  assert(RESULT_OK == r);
  assert(module != NULL);
  assert(0 == sequence_count(module->errors));

  vm_module_runtime *runtime = vm_module_runtime_create(module);
  assert(runtime != NULL);

  vm_function *function = vm_module_fetch_function(module, "main");
  assert(function != NULL);

  while (n-- > 0) {
    r = vm_module_runtime_run(runtime, function);
    assert(RESULT_OK == r);
    vm_variable *var =
        (vm_variable *)cmap_get_value(function->variables, "total");
    assert(var != NULL);

    if (var->type_class != ptcNUMBER) {
      printf("Got wrong type. Expected %s, but got %s",
             vm_variable_param_type_class_to_string(ptcNUMBER),
             vm_variable_param_type_class_to_string(var->type_class));
      return 0;
    }

    if (var->value.number != 362) {
      printf("Got wrong value. Expected %d, but got %f", 362,
             var->value.number);
      return 0;
    }

    if (runtime->stack_size != 1) {
      printf("Stack should have a single item");
      return 0;
    }

    vm_variable *return_var = NULL;
    r = vm_module_runtime_stack_pop(runtime, &return_var);
    assert(RESULT_OK == r);
    assert(return_var != NULL);
    assert(ptcNUMBER == return_var->type_class);
    assert(362 == return_var->value.number);
    vm_variable_dispose(return_var);

    vm_module_runtime_clear(runtime);
  }

  vm_module_runtime_dispose(runtime);
  vm_module_dispose(module);
  free(source);

#ifdef _DEBUG
  xmemory_report_exit_on_leaks();
#endif

  return count;
}

static int benchmark_conditions1() {
  int count = 5000000;

#ifdef _DEBUG
  count /= 10;
#endif

  int n = count;

  vm_module *module = NULL;
  puts("Starting compile...");
  fflush(stdout);
  char *source =
      load_file_content("resources/vm/runtime/runtime_conditions1.ts");
  assert(source != NULL);

  result r = vm_module_compile(NULL, source, &module);
  puts("Compile finished!");
  fflush(stdout);
  assert(RESULT_OK == r);
  assert(module != NULL);
  assert(0 == sequence_count(module->errors));

  vm_module_runtime *runtime = vm_module_runtime_create(module);
  assert(runtime != NULL);

  vm_function *function = vm_module_fetch_function(module, "main");
  assert(function != NULL);

  while (n-- > 0) {
    r = vm_module_runtime_run(runtime, function);
    assert(RESULT_OK == r);

    /* Verify amount stays at 100 (never modified by the script) */
    vm_variable *var_amount =
        (vm_variable *)cmap_get_value(function->variables, "amount");
    assert(var_amount != NULL);

    if (var_amount->type_class != ptcNUMBER) {
      printf("Got wrong type. Expected %s, but got %s",
             vm_variable_param_type_class_to_string(ptcNUMBER),
             vm_variable_param_type_class_to_string(var_amount->type_class));
      return 0;
    }

    if (var_amount->value.number != 100) {
      printf("Got wrong value. Expected %d, but got %f", 100,
             var_amount->value.number);
      return 0;
    }

    /* Verify condition branches executed correctly:
       a=1 (amount > 10), b=1 (amount >= 100), c=1 (amount-1 == 99),
       d=1 (amount < 200 && amount <= 100), f=2 (amount < 1000, f=1+1) */
    vm_variable *var_a =
        (vm_variable *)cmap_get_value(function->variables, "a");
    assert(var_a != NULL && var_a->value.number == 1);

    vm_variable *var_b =
        (vm_variable *)cmap_get_value(function->variables, "b");
    assert(var_b != NULL && var_b->value.number == 1);

    vm_variable *var_c =
        (vm_variable *)cmap_get_value(function->variables, "c");
    assert(var_c != NULL && var_c->value.number == 1);

    vm_variable *var_d =
        (vm_variable *)cmap_get_value(function->variables, "d");
    assert(var_d != NULL && var_d->value.number == 1);

    vm_variable *var_f =
        (vm_variable *)cmap_get_value(function->variables, "f");
    assert(var_f != NULL && var_f->value.number == 2);

    vm_module_runtime_clear(runtime);
  }

  vm_module_runtime_dispose(runtime);
  vm_module_dispose(module);

#ifdef _DEBUG
  xmemory_report_exit_on_leaks();
#endif

  return count;
}

// --- verb benchmarks (ported from microgpt-verb) ---

static verb(bench_echo) {
  const char *message = verb_arg("message");
  return message ? (char *)message : NULL;
}

static int benchmark_simple_verb_call_single_param() {
  int count = 5000000;

#ifdef _DEBUG
  count /= 10000;
#endif

  int n = count;
  char *response = NULL;

  verb_context *context = verb_context_create();

  int result = verb_register(context, "echo", "message", bench_echo, NULL);

  if (result != RESULT_OK) {
    verb_context_dispose(context);
    return BENCHMARK_RESULT_CORE_FAILED;
  }

  while (n-- > 0) {
    result = verb_exec(context, "echo hello", &response);

    if (result != RESULT_OK) {
      verb_context_dispose(context);
      return BENCHMARK_RESULT_CORE_FAILED;
    }
  }

  verb_context_dispose(context);

  return count;
}

// --- opaque handle benchmark (rolling_mean pipeline) ---

typedef struct {
  double *data;
  size_t length;
} bench_signal;

#define BENCH_MAX_HANDLES 8
static bench_signal *_bench_handles[BENCH_MAX_HANDLES];
static size_t _bench_handle_count = 0;

static bench_signal *_bench_signal_create(size_t len) {
  bench_signal *s = (bench_signal *)malloc(sizeof(bench_signal));
  s->data = (double *)calloc(len, sizeof(double));
  s->length = len;
  return s;
}

static void _bench_reset(void) {
  for (size_t i = 0; i < _bench_handle_count; i++) {
    free(_bench_handles[i]->data);
    free(_bench_handles[i]);
  }
  _bench_handle_count = 0;
}

static verb(bench_create_signal) {
  size_t len = (size_t)atoi(verb_arg("length"));
  bench_signal *s = _bench_signal_create(len);
  for (size_t i = 0; i < len; i++)
    s->data[i] = (double)(i + 1);
  size_t id = _bench_handle_count++;
  _bench_handles[id] = s;
  char buf[32];
  snprintf(buf, sizeof(buf), "%zu", id);
  return string_clone(buf);
}

static verb(bench_rolling_mean) {
  size_t src_id = (size_t)atoi(verb_arg("handle"));
  size_t window = (size_t)atoi(verb_arg("window"));
  bench_signal *src = _bench_handles[src_id];
  bench_signal *dst = _bench_signal_create(src->length);
  for (size_t i = 0; i < src->length; i++) {
    double sum = 0;
    size_t cnt = 0;
    for (size_t j = (i >= window ? i - window + 1 : 0); j <= i; j++) {
      sum += src->data[j];
      cnt++;
    }
    dst->data[i] = sum / (double)cnt;
  }
  size_t id = _bench_handle_count++;
  _bench_handles[id] = dst;
  char buf[32];
  snprintf(buf, sizeof(buf), "%zu", id);
  return string_clone(buf);
}

static verb(bench_signal_value_at) {
  size_t id = (size_t)atoi(verb_arg("handle"));
  size_t idx = (size_t)atoi(verb_arg("index"));
  char buf[64];
  snprintf(buf, sizeof(buf), "%.6f", _bench_handles[id]->data[idx]);
  return string_clone(buf);
}

static int benchmark_opaque_handle_rolling_mean() {
  int count = 100000;

#ifdef _DEBUG
  count /= 100;
#endif

  int n = count;

  verb_context *ctx = verb_context_create();
  verb_register(ctx, "create_signal", "length", bench_create_signal, NULL);
  verb_register(ctx, "rolling_mean", "handle window", bench_rolling_mean, NULL);
  verb_register(ctx, "signal_value_at", "handle index", bench_signal_value_at,
                NULL);

  while (n-- > 0) {
    _bench_reset();

    char *h1 = NULL, *h2 = NULL, *val = NULL;
    verb_exec(ctx, "create_signal 100", &h1);
    verb_exec(ctx, "rolling_mean 0 10", &h2);
    verb_exec(ctx, "signal_value_at 1 50", &val);

    free(h1);
    free(h2);
    free(val);
  }

  _bench_reset();
  verb_context_dispose(ctx);
  return count;
}

benchmark vm_benchmarks[] = {
    BENCHMARK_CASE(benchmark_simple1),
    BENCHMARK_CASE(benchmark_conditions1),
    BENCHMARK_CASE(benchmark_simple_verb_call_single_param),
    BENCHMARK_CASE(benchmark_opaque_handle_rolling_mean),
    BENCHMARK_END,
};

// --- main.c ---
/*
 * main.c
 *
 * Copyright 2024 Enjector Software, Ltd. All Rights Reserved.
 *
 */

#include "bench.h"
#include <stdio.h>

static benchmark_suite benchmarks[] = {BENCHMARK_CASE(vm_benchmarks),

                                       BENCHMARK_END};

#ifdef _WIN32
#include <tchar.h>
#include <windows.h>
#endif

int main(int argc, const char *argv[]) {

  const char *optional_output_path = NULL;

  if (argc) {
    optional_output_path = argv[1];
    printf("Output Path: %s\n", optional_output_path);
  }

  // #ifdef _DEBUG
  //     printf("Benchmarks must be built using Release configuration\n");
  //
  // #ifdef _WIN32    // For running via Visual Studio
  //     printf("\nEnd, press key to close\n");
  //     getchar();
  // #endif
  //
  //     return 1;
  // #endif

#ifdef _WIN32
  HANDLE process = GetCurrentProcess();
  // DWORD_PTR coreNumber = 15*2+1;

  // if (!SetThreadAffinityMask(GetCurrentThread(), (1 << coreNumber))) {
  //    _tprintf(TEXT("Failed to set SetThreadAffinityMask\n"));
  //    return 0;
  //}

  if (!SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_TIME_CRITICAL)) {
    _tprintf(TEXT("Failed to enter THREAD_PRIORITY_TIME_CRITICAL mode\n"));
    return 0;
  }

#endif
  //   abc:
  bool result = benchmark_suite_run(benchmarks, optional_output_path);

#ifdef _WIN32

  /*   if (!SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_BELOW_NORMAL))
     { _tprintf(TEXT("Failed to enter THREAD_PRIORITY_BELOW_NORMAL mode\n"));
     }*/

#endif

#ifdef _WIN32 // For running via Visual Studio
  printf("\nEnd, press key to close\n");
  getchar();
#endif

  // Return 0 for success and 1 fo failure
  return result ? 0 : 1;
}
