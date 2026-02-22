# Word-Level Tokenisation

---

## Spear Summary

**Point:** Word-level tokenisation trades vocabulary size for semantic density — each token carries a whole word's meaning instead of a single letter.

**Picture:** It's like reading a book word-by-word instead of letter-by-letter. You understand faster but you'll skip any word you've never seen before — and for open vocabulary at this model scale that happens a lot.

**Proof:** The Shakespeare corpus has ~8,000 unique word forms, but text normalisation (lowercase + punctuation splitting) reduces effective vocabulary to ~4,000. With 5,000 word slots, the UNK rate drops to ~7%. For **constrained vocabularies** (like the VM codegen DSL with ~500 words) UNK drops to <0.1%, making word-level the superior choice.

**Push:** Word-level tokenisation is ideal for **controlled vocabularies** (DSLs, forms, structured data). For open-ended prose, compare with the character-level demo to see the tradeoff.

---

Word-level tokenisation splits text on whitespace, maps each word to a token ID, and preserves newlines as distinct tokens. It works best when **the vocabulary is bounded or repeating** — fewer tokens per document, but a much larger embedding table.

## When to Use

| ✅ Good fits | ❌ Poor fits |
|-------------|------------|
| Constrained DSLs (VM code, SQL) | Highly varied open vocabulary |
| Prose with frequent word reuse | Very short text (< 5 words) |
| Dialogue with stock phrases | Character-level patterns (names) |
| Forms and templates | Subword morphology needed |

> **Key insight from vm_codegen:** Word-level tokenisation pushed syntax pass rate from 0% → 80% on a constrained VM DSL, while character-level achieved 0% on the same model. Representation beats capacity.

## API

```c
#include "microgpt.h"

/* 1. Load the entire text file */
size_t text_len;
char *text = load_file("shakespeare.txt", &text_len);

/* 2. Build word vocabulary (top N by frequency) */
WordVocab wv;
build_word_vocab(text, text_len, 8000, &wv);
/* wv.words[i]    = string for token i
   wv.vocab_size  = kept_words + 3 (unk + newline + bos)
   wv.unk_id      = unknown word token
   wv.newline_id  = newline token
   wv.bos_id      = beginning-of-sequence token */

/* 3. Tokenise entire text */
size_t *ids = malloc(text_len * sizeof(size_t));
size_t n = tokenize_words(text, text_len, &wv, ids, text_len);

/* 4. Train on sliding windows of BLOCK_SIZE tokens */
size_t offset = rand() % (n - BLOCK_SIZE);
forward_backward_one(model, ids[offset+pos], pos,
                     ids[offset+pos+1], ...);

/* 5. Generate words */
forward_inference(model, token_id, pos, ...);
size_t next = sample_token(logits, wv.vocab_size, temperature);
printf("%s ", wv.words[next]);  /* print the word string */

/* 6. Cleanup */
free_word_vocab(&wv);
free(text);
```

## Demos

| Demo | Vocab | UNK Rate | Notes |
|------|-------|----------|-------|
| [`demos/word-level/shakespeare/`](file:///Users/user/dev/projects/microgpt-c/demos/word-level/shakespeare) | 5,000 words | ~7% | Normalised open vocab (lowercase + punct split) |
| [`demos/character-level/shakespeare/`](file:///Users/user/dev/projects/microgpt-c/demos/character-level/shakespeare) | ~80 chars | 0% | No UNK but must learn spelling |
| [`experiments/organelles/vm_codegen/`](file:///Users/user/dev/projects/microgpt-c/experiments/organelles/vm_codegen) | 500 words | <0.1% | Constrained DSL — **80% syntax pass** |

> **Character vs Word — when to use which:**
> - **Character-level** — zero unknowns, learns spelling, needs deeper models (4 layers). Best for open vocabulary at sub-1M scale.
> - **Word-level** — coherent phrases, needs normalisation to control vocab, shallower models (1 layer). Best for constrained vocabularies or when phrase structure matters more than spelling.

## Text Normalisation

Raw word-level tokenisation wastes vocab slots on punctuation variants: `"the"`, `"the,"`, `"the."`, `"The"` all consume separate entries. Pre-processing fixes this:

1. **Lowercase**: `"The"` → `"the"` — merges case variants
2. **Punctuation splitting**: `"art,"` → `"art" ","` — separates words from punctuation

This reduces effective unique words from ~8,000 to ~4,000, so 5,000 vocab slots cover 93% of Shakespeare:

```
Before: [John.] [I] [wonder] [thou] [UNK] [art,] [born]     (20% UNK)
After:  [john] [.] [i] [wonder] [thou] [being] [,] [art]    (7% UNK)
```

See the Shakespeare word-level demo for the full implementation.

## Vocabulary Sizing

The `max_words` parameter to `build_word_vocab` controls how many unique words to keep. Words outside the top-N become `<unk>`.

| Corpus type | Unique words | Recommended top-N | `<unk>` rate |
|-------------|-------------|-------------------|-------------|
| Constrained DSL | ~200–500 | All (500) | <0.1% |
| Small prose (< 1MB) | ~2,000 | 1,000–2,000 | ~5% |
| Medium prose (1–10MB) | ~8,000 | 5,000–8,000 | ~15% |
| Large prose (> 10MB) | ~20,000+ | 8,000–10,000 | ~20% |

**Key tradeoff**: more words = fewer `<unk>` in output, but each rare word gets less training signal, and the output layer grows linearly with vocab size.

## Performance Note

`build_word_vocab` uses O(n²) deduplication. For large corpora (>1MB), sample a subset for vocabulary discovery:

```c
/* Use first 500KB for vocab discovery (fast) */
size_t sample_len = text_len > 500000 ? 500000 : text_len;
build_word_vocab(text, sample_len, max_words, &wv);
/* Then tokenize full corpus with discovered vocab */
tokenize_words(text, text_len, &wv, ids, max_ids);
```

## Model Sizing

Word-level models need more capacity than character-level since the vocabulary is much larger:

| Parameter | Open Vocabulary | Constrained DSL |
|-----------|----------------|----------------|
| N_EMBD | 48 | 48–64 |
| BLOCK_SIZE | 64 | 256 |
| N_LAYER | 1 | 1 |
| MAX_VOCAB | 5,000 | 500 |
| Training steps | 10,000 | 15,000 |

These are set via CMake compile definitions (see `CMakeLists.txt` for the `shakespeare_word_demo` and `vm_codegen` targets).
