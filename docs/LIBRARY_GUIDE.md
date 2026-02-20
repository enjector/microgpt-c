# Using MicroGPT-C as a Library

Add two files to your project — no build system changes needed beyond compiling one extra `.c`:

```c
#include "microgpt.h"
```

---

## Character-Level Pipeline

Best for short text: names, codes, identifiers.

```c
MicrogptConfig cfg = microgpt_default_config();

Docs docs;
load_docs("names.txt", &docs, cfg.max_docs);  // Load line-separated training data

Vocab vocab;
build_vocab(&docs, &vocab);           // Build character vocabulary (auto-sized)

size_t ids[256];
size_t n = tokenize("alice", 5, &vocab, ids, 256);

Model *model = model_create(vocab.vocab_size, &cfg);
// ... train with forward_backward_one + adam_step ...
// ... or use TrainWorker + train_worker_run for multi-threaded batches ...
// ... generate with forward_inference + sample_token ...
model_free(model);
```

---

## Word-Level Pipeline

Best for prose, dialogue, poetry. Uses O(1) hash-based `word_to_id` lookup.

```c
size_t len;
char *text = load_file("shakespeare.txt", &len);

WordVocab wv;
build_word_vocab(text, len, 10000, &wv);  // Keep top 10,000 words

size_t ids[8192];
size_t n = tokenize_words(text, len, &wv, ids, 8192);

MicrogptConfig cfg = microgpt_default_config();
Model *model = model_create(wv.vocab_size, &cfg);
// ... train and generate ...
free_word_vocab(&wv);
model_free(model);
```

---

## Character-Level vs Word-Level — Which to Use?

Both tokenisation strategies are available, but they suit different model scales:

| Factor | Character-level | Word-level |
|--------|----------------|-----------|
| **Vocab size** | ~50–100 tokens | ~5,000–10,000+ tokens |
| **`<unk>` tokens** | **Zero** — every byte is in vocab | Common for rare words |
| **lm_head cost** | ~100 × N_EMBD ≈ tiny | ~10,000 × N_EMBD ≈ dominates model |
| **Training signal** | Every character seen thousands of times | Rare words get few examples |
| **Best for** | Small models (< 1M params) | Large models (millions of params) |

**Why character-level wins at this scale:** Shakespeare has ~20,000 unique words. Even keeping the top 8,000, the output layer (`lm_head`) alone would consume more parameters than the rest of the model combined. The model can't learn meaningful distinctions between thousands of words, so output floods with `<unk>`. With character-level (~84 tokens), the entire vocabulary fits comfortably and the model masters every symbol.

**Rule of thumb:** Use character-level unless your model has enough capacity (N_EMBD ≥ 256, N_LAYER ≥ 4) to handle a large word vocabulary.

---

## Training Checkpoints

Save and resume training without losing optimizer momentum:

```c
// Save: model weights + Adam m/v state + step counter
checkpoint_save(model, m_adam, v_adam, step, "checkpoint.bin");

// Resume: restores everything needed to continue training
MicrogptConfig cfg = microgpt_default_config();
Model *model = checkpoint_load("checkpoint.bin", vocab_size,
                               &cfg, m_adam, v_adam, &step);
// Continue training from 'step' onwards — momentum and LR schedule are preserved
```

---

## Multi-Threaded Training Helpers

The library provides shared training infrastructure so demos don't need to duplicate boilerplate:

```c
#include "microgpt.h"

// TrainWorker struct holds per-thread state (grads, KV cache, loss, positions)
TrainWorker workers[N_THREADS];
// Set workers[t].model, .docs, .vocab, .batch_start, .batch_end, etc.

// Spawn threads with the shared entry point
for (int t = 0; t < nthreads; t++)
    pthread_create(&threads[t], NULL, train_worker_run, &workers[t]);
for (int t = 0; t < nthreads; t++)
    pthread_join(threads[t], NULL);

// Aggregate: workers[t].loss, workers[t].positions, workers[t].grads
```

Also available: `shuffle_docs(&docs)` for Fisher-Yates document shuffling and `rand_u()` for uniform random `scalar_t` in [0, 1).

---

## Complete Examples

See [`demos/character-level/names/main.c`](../demos/character-level/names/main.c) (character-level), [`demos/character-level/shakespeare/main.c`](../demos/character-level/shakespeare/main.c) (character-level, multi-threaded), and [`experiments/organelles/c_codegen/main.c`](../experiments/organelles/c_codegen/main.c) (875K-param C code generation) for full working programs.

Detailed guides:
- [Character-level tokenisation](foundation/CHARACTER_LEVEL.md)
- [Word-level tokenisation](foundation/WORD_LEVEL.md)
