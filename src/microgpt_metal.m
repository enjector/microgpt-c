/*
 * microgpt_metal.m - Objective-C Metal GPU bridge for MicroGPT-C
 *
 * Copyright (c) 2026 Ajay Soni (ajay.soni@enjector.com), Enjector Software Ltd.
 * MIT License — see LICENSE file for details.
 *
 * Implements the C API declared in microgpt_metal.h using Apple's Metal
 * framework for GPU compute.
 *
 * Key design decisions:
 * - Metal only supports float32, so we convert double↔float at the boundary.
 *   This trades a small precision loss for massive GPU speedup.
 * - Uses MTLResourceStorageModeShared for zero-copy on Apple Silicon.
 * - Pre-allocates reusable scratch buffers to avoid per-call allocation.
 * - Synchronous execution (waitUntilCompleted) for correctness.
 */

#include "microgpt_metal.h"
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ---- Metal state (module-global) ---- */
static id<MTLDevice> g_device = nil;
static id<MTLCommandQueue> g_queue = nil;
static id<MTLComputePipelineState> g_lin_fwd = nil;
static id<MTLComputePipelineState> g_lin_bwd_dx = nil;
static id<MTLComputePipelineState> g_lin_bwd_dW = nil;
static int g_initialized = 0;

/* ---- Scratch buffers for double↔float conversion ---- */
/* Pre-allocated to avoid malloc per dispatch call.
 * Sized for the largest expected matrix (MLP_DIM × N_EMBD = 512×128 = 65536).
 */
#define SCRATCH_SIZE (512 * 512)   /* 256K floats = 1MB, enough for any layer */
static float *g_scratch_x = NULL;  /* input vector (float32) */
static float *g_scratch_W = NULL;  /* weight matrix (float32) */
static float *g_scratch_y = NULL;  /* output vector (float32) */
static float *g_scratch_dy = NULL; /* upstream gradient (float32) */
static float *g_scratch_dx = NULL; /* input gradient (float32) */
static float *g_scratch_dW = NULL; /* weight gradient (float32) */

/* ---- Helper: double[] → float[] conversion ---- */
static void d2f(const double *src, float *dst, size_t n) {
  for (size_t i = 0; i < n; i++)
    dst[i] = (float)src[i];
}

/* ---- Helper: float[] → double[] conversion (overwrite) ---- */
static void f2d(const float *src, double *dst, size_t n) {
  for (size_t i = 0; i < n; i++)
    dst[i] = (double)src[i];
}

/* ---- Helper: float[] → double[] conversion (accumulate) ---- */
static void f2d_acc(const float *src, double *dst, size_t n) {
  for (size_t i = 0; i < n; i++)
    dst[i] += (double)src[i];
}

/* ---- Helper: load shader from file ---- */
static id<MTLLibrary> load_metal_library(id<MTLDevice> device) {
  NSError *err = nil;

  /* Try loading from default library first (Xcode-compiled metallib) */
  id<MTLLibrary> lib = [device newDefaultLibrary];
  if (lib)
    return lib;

  /* Fall back to compiling from source at runtime */
  NSArray *searchPaths = @[
    /* Same directory as the executable */
    [[[NSProcessInfo processInfo].arguments[0]
        stringByDeletingLastPathComponent]
        stringByAppendingPathComponent:@"microgpt_metal.metal"],
    /* Development paths */
    @"../src/microgpt_metal.metal",
    @"src/microgpt_metal.metal",
    @"microgpt_metal.metal",
  ];

  for (NSString *path in searchPaths) {
    if ([[NSFileManager defaultManager] fileExistsAtPath:path]) {
      NSString *source = [NSString stringWithContentsOfFile:path
                                                   encoding:NSUTF8StringEncoding
                                                      error:&err];
      if (!source)
        continue;

      MTLCompileOptions *opts = [[MTLCompileOptions alloc] init];
      lib = [device newLibraryWithSource:source options:opts error:&err];
      if (lib) {
        fprintf(stderr, "[Metal] Compiled shaders from: %s\n",
                [path UTF8String]);
        return lib;
      } else {
        fprintf(stderr, "[Metal] Shader compile error: %s\n",
                [[err localizedDescription] UTF8String]);
      }
    }
  }

  fprintf(stderr, "[Metal] ERROR: Could not find microgpt_metal.metal\n");
  return nil;
}

/* ---- Public API ---- */

int metal_init(void) {
  if (g_initialized)
    return 0;

  @autoreleasepool {
    /* 1. Get the default Metal device (GPU) */
    g_device = MTLCreateSystemDefaultDevice();
    if (!g_device) {
      fprintf(stderr, "[Metal] No Metal-capable GPU found\n");
      return -1;
    }
    fprintf(stderr, "[Metal] GPU: %s\n", [[g_device name] UTF8String]);

    /* 2. Create command queue */
    g_queue = [g_device newCommandQueue];
    if (!g_queue) {
      fprintf(stderr, "[Metal] Failed to create command queue\n");
      return -1;
    }

    /* 3. Load and compile shader library */
    id<MTLLibrary> lib = load_metal_library(g_device);
    if (!lib)
      return -1;

    /* 4. Create compute pipeline states for each kernel */
    NSError *err = nil;
    id<MTLFunction> fn;

    fn = [lib newFunctionWithName:@"lin_fwd_kernel"];
    if (!fn) {
      fprintf(stderr, "[Metal] lin_fwd_kernel not found\n");
      return -1;
    }
    g_lin_fwd = [g_device newComputePipelineStateWithFunction:fn error:&err];
    if (!g_lin_fwd) {
      fprintf(stderr, "[Metal] Pipeline error: %s\n",
              [[err localizedDescription] UTF8String]);
      return -1;
    }

    fn = [lib newFunctionWithName:@"lin_bwd_dx_kernel"];
    if (!fn) {
      fprintf(stderr, "[Metal] lin_bwd_dx_kernel not found\n");
      return -1;
    }
    g_lin_bwd_dx = [g_device newComputePipelineStateWithFunction:fn error:&err];
    if (!g_lin_bwd_dx)
      return -1;

    fn = [lib newFunctionWithName:@"lin_bwd_dW_kernel"];
    if (!fn) {
      fprintf(stderr, "[Metal] lin_bwd_dW_kernel not found\n");
      return -1;
    }
    g_lin_bwd_dW = [g_device newComputePipelineStateWithFunction:fn error:&err];
    if (!g_lin_bwd_dW)
      return -1;

    /* 5. Allocate scratch buffers for double↔float conversion */
    g_scratch_x = (float *)malloc(SCRATCH_SIZE * sizeof(float));
    g_scratch_W = (float *)malloc(SCRATCH_SIZE * sizeof(float));
    g_scratch_y = (float *)malloc(SCRATCH_SIZE * sizeof(float));
    g_scratch_dy = (float *)malloc(SCRATCH_SIZE * sizeof(float));
    g_scratch_dx = (float *)malloc(SCRATCH_SIZE * sizeof(float));
    g_scratch_dW = (float *)malloc(SCRATCH_SIZE * sizeof(float));
    if (!g_scratch_x || !g_scratch_W || !g_scratch_y || !g_scratch_dy ||
        !g_scratch_dx || !g_scratch_dW) {
      fprintf(stderr, "[Metal] Failed to allocate scratch buffers\n");
      return -1;
    }

    g_initialized = 1;
    fprintf(stderr,
            "[Metal] Initialised (float32 compute, max threadgroup: %lu)\n",
            (unsigned long)g_lin_fwd.maxTotalThreadsPerThreadgroup);
    return 0;
  }
}

void metal_cleanup(void) {
  g_lin_fwd = nil;
  g_lin_bwd_dx = nil;
  g_lin_bwd_dW = nil;
  g_queue = nil;
  g_device = nil;
  free(g_scratch_x);
  g_scratch_x = NULL;
  free(g_scratch_W);
  g_scratch_W = NULL;
  free(g_scratch_y);
  g_scratch_y = NULL;
  free(g_scratch_dy);
  g_scratch_dy = NULL;
  free(g_scratch_dx);
  g_scratch_dx = NULL;
  free(g_scratch_dW);
  g_scratch_dW = NULL;
  g_initialized = 0;
}

int metal_available(void) { return g_initialized; }

/* ---- GPU dispatch helper ---- */
static void dispatch_kernel(id<MTLComputePipelineState> pipeline,
                            id<MTLBuffer> bufs[], int nbuf, uint32_t params[],
                            int nparams, size_t grid_size) {
  @autoreleasepool {
    id<MTLCommandBuffer> cmdBuf = [g_queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];

    [enc setComputePipelineState:pipeline];
    for (int i = 0; i < nbuf; i++)
      [enc setBuffer:bufs[i] offset:0 atIndex:i];

    for (int i = 0; i < nparams; i++)
      [enc setBytes:&params[i] length:sizeof(uint32_t) atIndex:nbuf + i];

    NSUInteger tpg = pipeline.maxTotalThreadsPerThreadgroup;
    if (tpg > grid_size)
      tpg = grid_size;
    MTLSize gridSz = MTLSizeMake(grid_size, 1, 1);
    MTLSize tgSize = MTLSizeMake(tpg, 1, 1);

    [enc dispatchThreads:gridSz threadsPerThreadgroup:tgSize];
    [enc endEncoding];
    [cmdBuf commit];
    [cmdBuf waitUntilCompleted];
  }
}

/* ---- lin_fwd: y = W @ x ---- */
void metal_lin_fwd(const double *x, const double *W, size_t nin, size_t nout,
                   double *y) {
  @autoreleasepool {
    /* Convert double → float for GPU */
    d2f(x, g_scratch_x, nin);
    d2f(W, g_scratch_W, nout * nin);

    /* Create Metal buffers wrapping our float scratch arrays.
     * StorageModeShared = zero-copy on Apple Silicon unified memory. */
    id<MTLBuffer> buf_x =
        [g_device newBufferWithBytesNoCopy:g_scratch_x
                                    length:nin * sizeof(float)
                                   options:MTLResourceStorageModeShared
                               deallocator:nil];
    id<MTLBuffer> buf_W =
        [g_device newBufferWithBytesNoCopy:g_scratch_W
                                    length:nout * nin * sizeof(float)
                                   options:MTLResourceStorageModeShared
                               deallocator:nil];
    id<MTLBuffer> buf_y =
        [g_device newBufferWithBytesNoCopy:g_scratch_y
                                    length:nout * sizeof(float)
                                   options:MTLResourceStorageModeShared
                               deallocator:nil];

    uint32_t params[2] = {(uint32_t)nin, (uint32_t)nout};
    id<MTLBuffer> bufs[3] = {buf_x, buf_W, buf_y};
    dispatch_kernel(g_lin_fwd, bufs, 3, params, 2, nout);

    /* Convert float → double result */
    f2d(g_scratch_y, y, nout);
  }
}

/* ---- lin_bwd: dx = W^T@dy, dW += dy⊗x ---- */
void metal_lin_bwd(const double *x, const double *W, const double *dy,
                   size_t nin, size_t nout, double *dx, double *dW) {
  @autoreleasepool {
    uint32_t params[2] = {(uint32_t)nin, (uint32_t)nout};

    if (dx) {
      /* Convert inputs to float */
      d2f(dy, g_scratch_dy, nout);
      d2f(W, g_scratch_W, nout * nin);
      /* Zero the dx scratch since the kernel accumulates */
      memset(g_scratch_dx, 0, nin * sizeof(float));

      id<MTLBuffer> buf_dy =
          [g_device newBufferWithBytesNoCopy:g_scratch_dy
                                      length:nout * sizeof(float)
                                     options:MTLResourceStorageModeShared
                                 deallocator:nil];
      id<MTLBuffer> buf_W =
          [g_device newBufferWithBytesNoCopy:g_scratch_W
                                      length:nout * nin * sizeof(float)
                                     options:MTLResourceStorageModeShared
                                 deallocator:nil];
      id<MTLBuffer> buf_dx =
          [g_device newBufferWithBytesNoCopy:g_scratch_dx
                                      length:nin * sizeof(float)
                                     options:MTLResourceStorageModeShared
                                 deallocator:nil];

      id<MTLBuffer> bufs[3] = {buf_dy, buf_W, buf_dx};
      dispatch_kernel(g_lin_bwd_dx, bufs, 3, params, 2, nin);

      /* Accumulate float result into double dx */
      f2d_acc(g_scratch_dx, dx, nin);
    }

    if (dW) {
      /* Convert inputs to float */
      d2f(dy, g_scratch_dy, nout);
      d2f(x, g_scratch_x, nin);
      /* Zero the dW scratch since the kernel accumulates */
      memset(g_scratch_dW, 0, nout * nin * sizeof(float));

      id<MTLBuffer> buf_dy =
          [g_device newBufferWithBytesNoCopy:g_scratch_dy
                                      length:nout * sizeof(float)
                                     options:MTLResourceStorageModeShared
                                 deallocator:nil];
      id<MTLBuffer> buf_x =
          [g_device newBufferWithBytesNoCopy:g_scratch_x
                                      length:nin * sizeof(float)
                                     options:MTLResourceStorageModeShared
                                 deallocator:nil];
      id<MTLBuffer> buf_dW =
          [g_device newBufferWithBytesNoCopy:g_scratch_dW
                                      length:nout * nin * sizeof(float)
                                     options:MTLResourceStorageModeShared
                                 deallocator:nil];

      id<MTLBuffer> bufs[3] = {buf_dy, buf_x, buf_dW};
      dispatch_kernel(g_lin_bwd_dW, bufs, 3, params, 2, nout * nin);

      /* Accumulate float result into double dW */
      f2d_acc(g_scratch_dW, dW, nout * nin);
    }
  }
}
