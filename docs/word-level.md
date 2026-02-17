# Word-Level Tokenisation

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

See [examples/shakespeare/main.c](../examples/shakespeare/main.c) for a complete working example.

```bash
cmake --build build --target shakespeare_demo
./build/shakespeare_demo
```

Sample output:
```
My lord,
 Should ... and his ...
 And I, for ... and I upon the same ...
```

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
