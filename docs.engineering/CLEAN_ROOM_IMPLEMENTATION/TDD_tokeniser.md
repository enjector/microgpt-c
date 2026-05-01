# TDD_tokeniser — Technical Design Document

**Document ID:** TDD-TOK-001
**Version:** 1.0
**Status:** DRAFT
**Paired BS:** `BS_tokeniser.md`
**Sources:** `src/microgpt.h` (declarations), `src/microgpt.c` §1 + §9.

---

## 1. Overview

Two tokenisation strategies are exposed by the core engine: character-level and word-level. Both are deliberately minimal — neither implements BPE, SentencePiece, or any subword scheme. The strategies trade off at different model scales (see `docs/FUNCTIONAL_SPEC.md` "Character-Level vs Word-Level").

## 2. Architecture

```
   raw text file
        │
        ▼
  load_docs() / load_file()         ← reads bytes verbatim into memory
        │
        ▼
  ┌───────────────────────┐    ┌────────────────────────────────┐
  │ Char path:            │    │ Word path:                      │
  │  build_vocab(docs, v) │    │  build_word_vocab(buf, sz, max, wv)│
  │  tokenize(s, l, v, …) │    │  tokenize_words(buf, sz, wv, …)│
  └───────────────────────┘    └────────────────────────────────┘
        │                               │
        ▼                               ▼
  size_t ids[]                    size_t ids[]
```

## 3. Data flow

### Char level

`build_vocab` makes a single pass over every byte of every document, sets a 256-entry "seen" bitmap, then collects the seen bytes in sorted order into `Vocab.chars`. The BOS token is appended at the highest index (`bos_id = vocab_size − 1`).

`tokenize` writes BOS, then for each byte of the input scans `Vocab.chars` for a match (linear, ≤ 128 entries) and writes the index. A trailing BOS is appended if there is room.

### Word level

`build_word_vocab` runs a two-pass count:

1. Pass 1: walk whitespace-delimited spans, hash each into a temporary count map.
2. Pass 2: heap-select the top-N by frequency, assign token IDs `[0..N-1]`. Append `<unk>` (N), newline (N+1), `<bos>` (N+2).

`tokenize_words` splits on whitespace; newlines emit `wv->newline_id`. `word_to_id` queries the hash table; on miss it returns `wv->unk_id`.

The hash table is open-addressed with linear probing; capacity is the next power of two ≥ 2 × `num_words`.

## 4. Key data structures

### 4.1 `Docs`

```c
typedef struct {
  char    *data;        /* slurped file buffer, NUL-terminated for convenience */
  char   **lines;       /* lines[i] points into data */
  size_t   num_docs;
  size_t  *doc_lens;
} Docs;
```

`load_docs` reads the entire file into `data`, then walks it splitting at `\n`. Lines are NOT NUL-terminated; consumers MUST use `doc_lens[i]` to bound the read.

### 4.2 `Vocab`

```c
typedef struct {
  unsigned char *chars;   /* sorted unique bytes, length vocab_size − 1 */
  size_t         vocab_size;
  size_t         bos_id;  /* always vocab_size − 1 */
} Vocab;
```

### 4.3 `WordVocab`

```c
typedef struct {
  char    **words;       /* words[id] → string */
  size_t    vocab_size;  /* total = num_words + 3 */
  size_t    num_words;
  size_t    unk_id, newline_id, bos_id;
  /* O(1) lookup table */
  char    **ht_keys;     /* not owned; alias of words[] entries */
  size_t   *ht_ids;
  size_t    ht_cap;
} WordVocab;
```

## 5. Algorithms

### 5.1 Char vocabulary build

```
for each doc d:
    for each byte b of d:
        seen[b] = 1
collect: chars[] = sorted bytes where seen[b] = 1
vocab_size = len(chars) + 1
bos_id = vocab_size − 1
```

Cost: O(total_bytes + 256 log 256) — linear in corpus size.

### 5.2 Char tokenise

```
ids[0] = bos_id
n = 1
for each byte b in input (up to max_len − 2):
    j = linear_search(chars, b)
    ids[n++] = j     (j == 0 if not found — silent fallback)
ids[n++] = bos_id    (only if room)
return n
```

The linear search per character is acceptable because `vocab_size ≤ 128` for any reasonable corpus.

### 5.3 Word vocabulary build (frequency-ranked top-N)

```
1. allocate temp count table ht of size max(max_words * 4, 1024) — load factor ~0.25
2. single pass over input:
     for each whitespace-delimited word w (delimiters: space/'\n'/'\r'):
         ht[word_ht_find_or_insert(w)].count++
3. compact non-empty ht entries into a `sorted[]` array, qsort by frequency descending
4. keep = min(num_unique, max_words); assign IDs [0..keep-1] in sorted order
5. append <unk> (keep), newline (keep+1), <bos> (keep+2); vocab_size = keep + 3
6. allocate the lookup table at ht_cap = max(vocab_size * 4, 64)
   — open-addressed, linear probing, modulo is NOT power-of-two
   — populate with each word's id; ht_keys aliases into wv->words[] (not owned)
```

Note: the reference implementation does NOT use heap-select; it does a full qsort over the unique-word table. M (unique-word count) is bounded by `ht_cap - 1`, so the cost is acceptable in the small-corpus regime.

### 5.4 Word tokenise

```
for each whitespace-delimited span:
    if span is "\n":
        ids[n++] = newline_id
    else:
        ids[n++] = word_to_id(wv, span) (= unk_id if not found)
```

The newline token preserves line-structure (essential for poetry / verse).

## 6. Concurrency model

Vocabularies and document arrays are read-only after construction; they are safe to share across threads (the training harness in `TrainWorker` does so). The hash table is not modified after `build_word_vocab` returns.

## 7. Trade-offs considered

| Decision | Chosen | Rejected | Rationale |
|---|---|---|---|
| Subword tokenisation | None (char + word only) | BPE / SentencePiece | Adds significant build complexity; char + word + the proven game / Shakespeare demos do not need it. Recorded as future work in `VISION.md` §7. |
| Char vocabulary linear scan | Linear (≤ 128 entries) | Hash / 256-entry direct lookup | Direct lookup would cost a 256-byte table per Vocab; saves ~1 cycle/char at the cost of cache footprint. Linear is fine in the small-model regime. |
| Word vocab structure | Frequency-ranked top-N | All words | Top-N caps the `lm_head` size (the dominant cost when vocab is large). Trade is a non-zero `<unk>` rate. |
| OOV handling | Single `<unk>` token | Char-level fallback | Simple and matches the reference. A char-level fallback is left to user code if needed. |

## 8. Known limitations

- No Unicode awareness: bytes are treated as units. Multi-byte UTF-8 codepoints become multiple tokens. For ASCII-dominant corpora this is invisible; for CJK or emoji-heavy text it is a real cost.
- `tokenize` linear scan is O(prompt_len × vocab_size); acceptable when vocab_size ≤ 128.
- Word tokenisation does not do casing / punctuation normalisation; `"Hello,"` and `"Hello"` are different word tokens. Users SHOULD pre-clean their corpora.
- `MAX_WORD_LEN = 48` truncates word tokens; longer tokens become `<unk>`.

## 9. References

- `docs/FUNCTIONAL_SPEC.md` — usage guide.
- `docs/foundation/CHARACTER_LEVEL.md`, `docs/foundation/WORD_LEVEL.md` — narrative explanations.

## 10. Revision history

| Version | Date | Change |
|---|---|---|
| 1.0 | 2026-04-30 | Initial extraction. |
