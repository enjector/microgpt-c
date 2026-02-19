# Word-Level Tokenisation

---

## Spear Summary

**Point:** Word-level tokenisation trades vocabulary size for semantic density — each token carries a whole word's meaning instead of a single letter.

**Picture:** It's like reading a book word-by-word instead of letter-by-letter. You understand faster but you'll skip any word you've never seen before — and at this model scale that happens a lot.

**Proof:** The Shakespeare corpus has ~8,000 unique words. At the model sizes MicroGPT-C uses (~875K params) the `<unk>` rate becomes problematic. The Shakespeare demo was rewritten from word-level to character-level precisely because of this.

**Push:** Only use word-level tokenisation when you have a controlled vocabulary (forms, templates, structured data). For open-ended text at sub-1M param scale, character-level is strictly better.

---

Word-level tokenisation splits text on whitespace, maps each word to a token ID, and preserves newlines as distinct tokens. It works best for **prose, dialogue, and longer text** where whole-word semantics matter more than individual characters.

## When to Use

| ✅ Good fits | ❌ Poor fits |
|-------------|------------|
| Prose / literary text | Very short text (< 5 words) |
| Dialogue generation | Character-level patterns (names) |
| Poetry (verse structure) | Highly varied vocabulary |
| Text with natural word boundaries | Subword morphology needed |

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

## Example

The word-level API is available in `microgpt.h` for use in your own programs. See the API section above for a complete usage pattern.

> **Note:** The bundled Shakespeare demo (`examples/shakespeare/main.c`) uses **character-level** tokenisation for better quality at this model scale. Word-level tokenisation is most effective with larger models and vocabularies.

## Vocabulary Sizing

The `max_words` parameter to `build_word_vocab` controls how many unique words to keep. Words outside the top-N become `<unk>`.

| Corpus | Unique words | Recommended top-N | Model params |
|--------|-------------|-------------------|-------------|
| Small (< 1MB) | ~2,000 | 1,000–2,000 | ~100k |
| Medium (1–10MB) | ~8,000 | 3,000–5,000 | ~300k |
| Large (> 10MB) | ~20,000+ | 5,000–10,000 | ~500k+ |

**Key tradeoff**: more words = fewer `<unk>` in output, but each rare word gets less training signal, and the output layer grows quadratically with N_EMBD.

## Model Sizing

Word-level models need more capacity than character-level since the vocabulary is much larger:

| Parameter | Recommended |
|-----------|-------------|
| N_EMBD | 32–64 |
| BLOCK_SIZE | 32–64 |
| N_LAYER | 1–2 |
| MAX_VOCAB | top_words + 3 (round up) |
| Training steps | 3,000–10,000 |

These are set via CMake compile definitions (see `CMakeLists.txt` for the shakespeare target).
