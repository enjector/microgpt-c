# Character-Level Tokenisation

Character-level tokenisation maps each byte in the input to a unique token ID. This is the simplest form of tokenisation and works best for **short, structured text** where individual characters carry strong signal.

## When to Use

| ✅ Good fits | ❌ Poor fits |
|-------------|------------|
| Name generation | Prose / dialogue |
| Code completion | Long-form text |
| Chemical formulas | Large vocabularies |
| Short text (< 32 chars) | Texts needing semantic understanding |

## API

```c
#include "microgpt.h"

/* 1. Load data (one document per line) */
Docs docs;
load_docs("names.txt", &docs);

/* 2. Build character vocabulary */
Vocab vocab;
build_vocab(&docs, &vocab);
/* vocab.chars[i] = i-th unique byte
   vocab.vocab_size = unique chars + 1 (BOS)
   vocab.bos_id = beginning-of-sequence token */

/* 3. Tokenise a document */
size_t ids[BLOCK_SIZE + 2];
size_t n = tokenize(doc, doc_len, &vocab, ids, BLOCK_SIZE + 2);
/* ids = [BOS, char_0, char_1, ..., char_n, BOS(EOS)] */

/* 4. Train (see examples/names/main.c for full loop) */
forward_backward_one(model, ids[pos], pos, ids[pos+1], ...);
adam_step(model, grads, m, v, step);

/* 5. Generate */
forward_inference(model, token_id, pos, ...);
size_t next = sample_token(logits, vocab.vocab_size, temperature);
char ch = vocab.chars[next];  /* convert back to character */
```

## Example

See [examples/names/main.c](../examples/names/main.c) for a complete working example that trains on 32k human names and generates new ones.

```bash
cmake --build build --target names_demo
./build/names_demo
```

Sample output:
```
sample  1: Karina
sample  2: Jaxon
sample  3: Ailina
```

## Model Sizing

For character-level tasks, the default model size is sufficient:

| Parameter | Recommended |
|-----------|-------------|
| N_EMBD | 16–32 |
| BLOCK_SIZE | 16–32 |
| N_LAYER | 1–2 |
| Vocab size | ~50–100 (auto from data) |
| Training steps | 1,000–5,000 |
