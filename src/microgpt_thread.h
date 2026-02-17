/*
 * microgpt_thread.h — Portable threading for MicroGPT-C
 *
 * Provides a minimal cross-platform abstraction over:
 *   - Thread creation / join  (pthread on Unix, Win32 on Windows)
 *   - CPU core count detection (sysconf / GetSystemInfo)
 *
 * Usage:
 *   mgpt_thread_t threads[N];
 *   mgpt_thread_create(&threads[i], worker_fn, &args[i]);
 *   mgpt_thread_join(threads[i]);
 *   int ncpu = mgpt_cpu_count();
 *
 * Copyright (c) 2026 Ajay Soni, Enjector Software Ltd. MIT License.
 */

#ifndef MICROGPT_THREAD_H
#define MICROGPT_THREAD_H

#ifdef _WIN32
/* ---- Windows ---- */
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <process.h>
#include <windows.h>

typedef HANDLE mgpt_thread_t;

/* Win32 thread proc signature: unsigned __stdcall fn(void*) */
typedef unsigned(__stdcall *mgpt_win32_fn)(void *);

/*
 * Wrapper: the user provides a void*(*fn)(void*) signature (like pthread).
 * We wrap it into the __stdcall unsigned(void*) that _beginthreadex expects.
 */
typedef struct {
  void *(*fn)(void *);
  void *arg;
} mgpt_thread_trampoline_t;

static unsigned __stdcall mgpt_thread_trampoline_(void *p) {
  mgpt_thread_trampoline_t *t = (mgpt_thread_trampoline_t *)p;
  t->fn(t->arg);
  return 0;
}

/*
 * Note: caller must keep the trampoline struct alive until thread completes.
 * The Shakespeare demo allocates these on the stack alongside the thread array.
 */
static int mgpt_thread_create(mgpt_thread_t *thread,
                              mgpt_thread_trampoline_t *tramp,
                              void *(*fn)(void *), void *arg) {
  tramp->fn = fn;
  tramp->arg = arg;
  *thread =
      (HANDLE)_beginthreadex(NULL, 0, mgpt_thread_trampoline_, tramp, 0, NULL);
  return (*thread == 0) ? -1 : 0;
}

static int mgpt_thread_join(mgpt_thread_t thread) {
  WaitForSingleObject(thread, INFINITE);
  CloseHandle(thread);
  return 0;
}

static int mgpt_cpu_count(void) {
  SYSTEM_INFO si;
  GetSystemInfo(&si);
  return (int)si.dwNumberOfProcessors;
}

/* rand_r is not available on Windows; provide a simple replacement */
#ifndef HAVE_RAND_R
static unsigned int mgpt_rand_r(unsigned int *seed) {
  *seed = *seed * 1103515245u + 12345u;
  return (*seed >> 16) & 0x7fff;
}
#define rand_r(s) mgpt_rand_r(s)
#endif

#else
/* ---- POSIX (Linux, macOS, etc.) ---- */
#include <pthread.h>
#include <unistd.h>

typedef pthread_t mgpt_thread_t;

/* Dummy trampoline struct (not needed on POSIX, but keeps API uniform) */
typedef struct {
  void *(*fn)(void *);
  void *arg;
} mgpt_thread_trampoline_t;

static int mgpt_thread_create(mgpt_thread_t *thread,
                              mgpt_thread_trampoline_t *tramp,
                              void *(*fn)(void *), void *arg) {
  (void)tramp; /* unused on POSIX */
  return pthread_create(thread, NULL, fn, arg);
}

static int mgpt_thread_join(mgpt_thread_t thread) {
  return pthread_join(thread, NULL);
}

static int mgpt_cpu_count(void) {
  long n = sysconf(_SC_NPROCESSORS_ONLN);
  return (n > 0) ? (int)n : 1;
}

#endif /* _WIN32 */

/*
 * mgpt_default_threads - Recommend a thread count for training.
 *   Returns min(cpu_count, batch_size) so we never have idle threads.
 */
static int mgpt_default_threads(int batch_size) {
  int ncpu = mgpt_cpu_count();
  return (ncpu < batch_size) ? ncpu : batch_size;
}

#endif /* MICROGPT_THREAD_H */
