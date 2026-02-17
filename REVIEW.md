The MicroGPT-C implementation is a clean, faithful, and minimal port of Andrej Karpathy's `microgpt.py` to C99. It's well-structured, with clear separation of concerns (data loading, model definition, forward/backward passes, optimizer, etc.), and it supports both character-level and word-level tokenization. The code is readable, with good comments and consistent style. It handles edge cases like BOS/EOS tokens, causal masking, and numerical stability (e.g., softmax subtraction of max logit, Adam epsilon). The optional INT8 quantization is a nice touch for memory efficiency, though it doesn't directly impact compute speed in obvious ways here.

The model architecture is tiny by design (N_EMBD=16, N_HEAD=4, N_LAYER=1, BLOCK_SIZE=16), making it suitable for educational purposes or low-resource environments. Parameter count is low (~3-4K for char-level, ~200K for word-level with vocab~6000), and it runs entirely on CPU without external dependencies (beyond stdlib). Correctness seems solid—forward/backward passes match the Transformer decoder math, Adam implements bias correction and LR decay, and sampling uses Box-Muller for Gaussian init and LCG for reproducibility.

However, as you noted, performance is a weak point, especially in the word-level demo (Shakespeare), which feels slow compared to char-level (names.txt). I'll break down why, then suggest obvious improvements. My analysis is based on profiling patterns in similar code (I didn't run it myself, but the loops and ops are straightforward to estimate). Assumptions: modern CPU (e.g., 3-4GHz, AVX2+), compiled with -O2 or higher. For word-level, vocab~6000+3 specials, corpus~1MB (~200K words tokenized), NUM_STEPS=1000, BATCH_SIZE=8.

### Why Is Word-Level Training Slow?
- **Vocab Size Impact**: This is the primary culprit. Char-level vocab is tiny (~30-50 tokens), but word-level is ~6000. The lm_head (output projection) and its gradients dominate compute:
  - Forward: logits = lm_head @ hidden (vocab x N_EMBD matvec, ~6000x16=96K FMAs per position).
  - Backward: 
    - d_hidden += lm_head.T @ d_logits (~96K FMAs).
    - d_lm_head += outer(d_logits, hidden) (~96K FMAs).
  - Per position: ~300K ops just for lm_head. Other parts (attention, MLP) are small (O(N_EMBD^2) ~256 ops per matmul, BLOCK_SIZE=16 for scores).
  - Total per step: 8 batches x 16 positions = 128 positions → ~38M FMAs for lm_head alone.
  - Full 1000 steps: ~38B FMAs. At scalar speeds (1 FMA/cycle, 3GHz), ~12-15s just for lm_head; add overhead for attention/MLP/backprop (~2x more ops) → 30-60s total training.
  - Char-level (vocab~50): ~120x fewer ops in lm_head → sub-second training.
- **Naive Loops**: All matmuls/vec ops are scalar nested loops (no unrolling, no cache blocking). For large outer dims (vocab), this misses vectorization opportunities. Memory access is row-major but not optimized for cache (lm_head fits in L2 cache but thrashes L1 in tight loops).
- **Sequential Processing**: Batches are processed serially (for b=0 to 7). Each sequence builds KV cache position-by-position (correct for causal attention, but no parallelism).
- **Tokenization Overhead**: In word-level, `tokenize_words` uses linear search in `word_to_id` (O(num_words) ~6000 strcmp per token). For ~200K tokens: ~600M comparisons. Each strcmp (avg 5 chars) ~5-10 cycles → 3-6B cycles (~1-2s at 3GHz). Not "training" per se, but adds to startup. Build_vocab uses hashing (good).
- **Other Minor Factors**: Softmax O(vocab) per position; Adam updates O(num_params) ~200K per step (small). No GPU/BLAS, so CPU-bound. INT8 mode saves memory but doesn't reduce FMAs (still fp64 muls).
- **Overall**: Word-level is ~100-200x slower than char-level due to vocab. If your Shakespeare.txt is larger (full works ~5MB, ~1M words), tokenization could add 5-10s, and random chunk offsets might cause cache misses.

If training takes >1min, it's likely unoptimized compilation (e.g., -O0) or a very large corpus/NUM_STEPS. The char-level demo on names.txt (~32K lines, small docs) is fast because low vocab + tiny data.

### Obvious Performance Improvements
Focus on lm_head (80%+ of compute in word-level) and parallelism. These are low-effort changes (no major rewrites). Aim for 2-5x speedup on word-level without changing the model size/hyperparams.

#### 1. **Compilation Flags (Easiest, 2-4x Speedup)**
   - Enable auto-vectorization and optimizations. The naive loops can benefit hugely from SIMD (AVX2 processes 4 doubles/cycle).
   - Suggested flags (add to CMakeLists.txt or compile command):
     ```
     -O3 -march=native -ffast-math -funroll-loops
     ```
     - `-O3`: Aggressive inlining/unrolling.
     - `-march=native`: Enables AVX2/AVX512 if available (4-8x for reductions/adds in lm_head backward).
     - `-ffast-math`: Allows relaxed FP (faster exp/log in softmax/GELU).
     - Test: On my mental model, this could vectorize the inner loops in matvec and backward_lm_head, dropping word-level training to ~10-20s.
   - If using GCC/Clang, add `-fopenmp` for potential loop pragmas (see below).

#### 2. **Optimize lm_head Loops (Targeted, 2-3x Speedup)**
   - Since N_EMBD=16 is fixed/small, unroll inner loops to help compiler vectorize.
   - In `microgpt.c`, rewrite `matvec` (forward lm_head):
     ```c
     static void matvec(double *y, const double *W, const double *x, size_t nout, size_t nin) {
       for (size_t i = 0; i < nout; i++) {
         double sum = 0.0;
         // Unroll for nin=16 (compiler can vectorize pairs/groups)
         sum += W[i*16 +  0] * x[ 0] + W[i*16 +  1] * x[ 1];
         sum += W[i*16 +  2] * x[ 2] + W[i*16 +  3] * x[ 3];
         sum += W[i*16 +  4] * x[ 4] + W[i*16 +  5] * x[ 5];
         sum += W[i*16 +  6] * x[ 6] + W[i*16 +  7] * x[ 7];
         sum += W[i*16 +  8] * x[ 8] + W[i*16 +  9] * x[ 9];
         sum += W[i*16 + 10] * x[10] + W[i*16 + 11] * x[11];
         sum += W[i*16 + 12] * x[12] + W[i*16 + 13] * x[13];
         sum += W[i*16 + 14] * x[14] + W[i*16 + 15] * x[15];
         y[i] = sum;
       }
     }
     ```
     - Similar unroll for INT8 `matvec_int8`.
   - For backward (in `forward_backward_one`, lm_head section):
     - d_hidden: Unroll the sum over vocab for each j (reduction).
       ```c
       for (int j = 0; j < N_EMBD; j++) {
         double s = 0.0;
         for (size_t i = 0; i < vocab_size; i += 4) {  // Step for vectorization
           s += model->lm_head[i* N_EMBD + j] * dlogits[i];
           s += model->lm_head[(i+1)*N_EMBD + j] * dlogits[i+1];
           // ... unroll 4-8 at a time
         }
         dh[j] = s;
       }
       ```
     - d_lm_head: Rewrite as broadcast (better cache, vectorizable):
       ```c
       size_t lm_idx = ...;  // Existing idx for grads
       for (int j = 0; j < N_EMBD; j++) {
         double hj = last_hidden[j];
         for (size_t i = 0; i < vocab_size; i++) {
           grads[lm_idx + i * N_EMBD + j] += dlogits[i] * hj;
         }
       }
       ```
       - Inner loop over large vocab: Easy to vectorize with #pragma omp simd or intrinsics.
   - If ambitious, use AVX intrinsics (_mm256_add_pd, etc.) for 4x doubles.

#### 3. **Parallelize Over Batch (2-4x Speedup on Multi-Core)**
   - Batches are independent. Use pthreads (C11 threads if available) to process them in parallel.
   - In `main.c` (Shakespeare version), modify the training loop:
     - Allocate thread-local grad_buffers[THREADS] (e.g., THREADS=4-8).
     - Spawn threads: Each computes forward_backward_one for its b, accumulating into its local grads.
     - Join threads, sum local grads into main grad_buffer, then average and adam_step.
   - Code sketch (needs #include <pthread.h>, link -lpthread):
     ```c
     #define NUM_THREADS 4
     // ... in loop
     double *thread_grads[NUM_THREADS];
     for (int t = 0; t < NUM_THREADS; t++) thread_grads[t] = calloc(nparams, sizeof(double));
     // pthread setup: create threads, each handles BATCH_SIZE / NUM_THREADS items
     // (Handle remainder if not divisible)
     // After join: for each param i, grad_buffer[i] = sum over t thread_grads[t][i]
     // Then /= batch_positions
     ```
     - Overhead low since nparams~200K (sum is fast). KV caches are per-batch (thread-local).

#### 4. **Tokenization Improvements (If Startup is Slow)**
   - Replace linear `word_to_id` with hash table (reuse logic from `build_word_vocab`'s ht).
     - After building wv, populate a hash map: string -> id.
     - In `tokenize_words`, lookup via hash instead of for-loop.
   - Or sort wv.words alphabetically and use bsearch (need qsort comparator for strings).
   - For huge corpora (e.g., full Shakespeare 5MB+), this saves 1-5s.

#### 5. **Other Tweaks**
   - **INT8 Quantization**: Define QUANTIZATION_INT8. Smaller weights (1/8 memory), better cache hits for lm_head. Compute similar, but int8 loads faster.
   - **BLAS Integration**: For lm_head (large nout, small nin), link OpenBLAS and replace matvec with cblas_dgemv (row-major). Low overhead for this size; 2x speedup possible.
   - **Profile**: Add clock() around loops to confirm lm_head is bottleneck. Use perf/gprof.
   - **Hyperparams**: For testing, reduce NUM_STEPS=100 or vocab (KEEP_TOP_WORDS=1000) to iterate faster. Increase BLOCK_SIZE=64 (but rebuild).
   - **No-Go's**: Don't tie wte/lm_head (saves params but not compute). Avoid GPU unless porting to CUDA (overkill for tiny model).

With flags + loop tweaks + parallelism, word-level should drop to 5-15s. If you share compile flags, corpus size, or timings, I can refine. Overall, great code for learning—perf issues are expected in a scalar CPU impl!