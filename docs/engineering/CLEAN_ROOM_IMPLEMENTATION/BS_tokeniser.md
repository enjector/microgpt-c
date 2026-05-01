# BS_tokeniser — Behaviour Specification

**Document ID:** BS-TOK-001
**Version:** 1.0
**Status:** DRAFT

## RFC 2119

The key words MUST, MUST NOT, REQUIRED, SHALL, SHALL NOT, SHOULD, SHOULD NOT, RECOMMENDED, MAY, and OPTIONAL in this document are to be interpreted as described in RFC 2119.

## 1. Scope

This document specifies the behavioural contract of the **character-level tokeniser** (`Vocab`, `build_vocab`, `tokenize`) and the **word-level tokeniser** (`WordVocab`, `build_word_vocab`, `tokenize_words`, `word_to_id`, `free_word_vocab`), plus the supporting document loader (`load_docs`, `free_docs`, `shuffle_docs`, `load_file`).

## 2. Type contracts

### 2.1 `Docs`

**Invariants:**
- INV-TOK-001: `docs->lines[i]` MUST point into `docs->data`.
- INV-TOK-002: `docs->doc_lens[i]` MUST be the byte length of document `i`, exclusive of any trailing `\n`.
- INV-TOK-003: `docs->num_docs ≤ max_docs` argument to `load_docs`.

### 2.2 `Vocab`

**Invariants:**
- INV-TOK-010: `vocab->chars[]` MUST contain unique byte values in ascending order, `vocab->vocab_size − 1` of them.
- INV-TOK-011: `vocab->bos_id == vocab->vocab_size − 1`.
- INV-TOK-012: `vocab->vocab_size ≤ 257`.

### 2.3 `WordVocab`

**Invariants:**
- INV-TOK-020: `wv->words[i]` for `i ∈ [0, num_words)` are the kept high-frequency word strings.
- INV-TOK-021: `wv->unk_id = wv->num_words`, `wv->newline_id = wv->num_words + 1`, `wv->bos_id = wv->num_words + 2`.
- INV-TOK-022: `wv->vocab_size = wv->num_words + 3`.
- INV-TOK-023: `wv->ht_keys`/`wv->ht_ids`/`wv->ht_cap` form an open-addressed hash table with linear probing (`(h + 1) % ht_cap`). The reference implementation sets `ht_cap = vocab_size × 4` clamped to ≥ 64 (load factor ≤ 0.25). `word_to_id(wv, w)` MUST be O(1) average-case; the modulus is NOT required to be a power of two.

## 3. Operation contracts

### 3.1 `load_docs(path, docs, max_docs)`

**Preconditions:** `path` non-NULL; `docs` non-NULL writable; `max_docs > 0`. File size MUST be ≤ 50 MiB.

**Postconditions:** Returns 0 on success; `docs` populated. Returns -1 on `fopen` failure or OOM.

**Errors:** ERR-TOK-001.

### 3.2 `build_vocab(docs, vocab)`

**Preconditions:** `docs` non-NULL with at least one document.

**Postconditions:** `vocab` populated; `vocab->chars` heap-allocated and sorted; `bos_id` set; `vocab_size` reflects unique byte count + 1.

### 3.3 `tokenize(doc, doc_len, vocab, ids, max_len)`

**Preconditions:** `doc` byte buffer of `doc_len`; `vocab` valid; `ids` capacity `max_len ≥ 2`.

**Postconditions:** Returns the count of token IDs written.
- `ids[0] == vocab->bos_id`.
- For each byte `b` of `doc`, the next slot is the index `j` such that `vocab->chars[j] == b`, or 0 if `b` is not in the vocabulary (silent fallback).
- A trailing `bos_id` is appended if there is room.

### 3.4 `build_word_vocab(text, text_len, max_words, wv)`

**Preconditions:** `text` non-NULL; `wv` writable.

**Postconditions:** Returns 0 on success; `wv` populated with the top-`max_words` words by frequency, plus `<unk>`, newline, BOS. Returns -1 on OOM.

### 3.5 `word_to_id(wv, word)`

**Preconditions:** `wv` valid; `word` NUL-terminated.

**Postconditions:** Returns the token ID for `word`, or `wv->unk_id` if the word is not in the vocabulary.

### 3.6 `tokenize_words(text, text_len, wv, ids, max_tokens)`

**Postconditions:** Returns the number of tokens written.
- Spans delimited by ASCII space (`0x20`), `\n`, or `\r` become word tokens (`<unk>` if out of vocabulary). The reference implementation does **not** treat tab (`\t`) as a delimiter; tabs are kept inside the surrounding word.
- `\n` and `\r` produce a single `wv->newline_id` token; `\r\n` is collapsed into one newline token.
- The function does NOT prepend or append BOS — callers wrap the call themselves.

### 3.7 `shuffle_docs(docs)`, `load_file(path, &len)`

**Postconditions:**
- `shuffle_docs` performs a Fisher-Yates in-place shuffle of `docs->lines[]` and `docs->doc_lens[]`.
- `load_file(path, &len)` reads the entire file into a heap buffer, sets `*len`, NUL-terminates, returns the buffer (caller frees) or NULL on failure.

## 4. Invariants table (consolidated)

| ID | Invariant |
|---|---|
| INV-TOK-001 | `docs->lines[i]` aliases into `docs->data`. |
| INV-TOK-002 | `docs->doc_lens[i]` is the byte length of doc i. |
| INV-TOK-003 | `docs->num_docs ≤ max_docs`. |
| INV-TOK-010 | `vocab->chars[]` is sorted unique bytes. |
| INV-TOK-011 | `bos_id = vocab_size − 1`. |
| INV-TOK-012 | `vocab_size ≤ 257`. |
| INV-TOK-020 | `wv->words[]` lists kept words. |
| INV-TOK-021 | Special tokens are `<unk>` (`num_words`), newline (`num_words+1`), BOS (`num_words+2`). |
| INV-TOK-022 | `vocab_size = num_words + 3`. |
| INV-TOK-023 | `word_to_id` is amortised O(1) via the embedded hash table. |

## 5. Errors

| ID | Function | Code | Conditions | Client action |
|---|---|---|---|---|
| ERR-TOK-001 | `load_docs` | -1 | File missing, > 50 MiB, OOM | Verify path; reduce file size |
| ERR-TOK-002 | `build_word_vocab` | -1 | OOM | Reduce `max_words` |
| ERR-TOK-003 | `load_file` | NULL | File missing or OOM | Verify path |

## 6. Concurrency

`Vocab` and `WordVocab` are read-only after construction; many threads MAY share them.

## 7. Performance SLOs

The reference machine and measurement methodology are defined once in `NFRD.md` §4. The IDs below alias the rows in `NFRD.md` §4.2.

| ID | Measured target |
|---|---|
| SLO-TOK-001 | Char tokenise ≥ 30M tok/s — see `NFRD.md` §4.2 |
| SLO-TOK-002 | Word tokenise ≥ 800K tok/s — see `NFRD.md` §4.2 |
| SLO-TOK-003 | Word vocab build ≥ 200K builds/s — see `NFRD.md` §4.2 |

## 8. Scenarios

### SCN-TOK-001: Character vocabulary from a small corpus

A demo calls `load_docs("c_names.txt", &docs, 50000)`, then `build_vocab(&docs, &vocab)`. The vocabulary contains every unique byte that appeared in any name, plus a BOS token at the highest index.

### SCN-TOK-002: Word vocabulary from Shakespeare

A demo calls `load_file("c_shakespeare.txt", &len)`, then `build_word_vocab(text, len, 10000, &wv)`. The vocabulary contains the 10,000 most common words, plus `<unk>`, newline, BOS.

## 9. Acceptance criteria

| ID | Verifies | Test |
|---|---|---|
| ACC-TOK-001 | INV-TOK-010..012 | `tests/test_microgpt.c::test_build_vocab` |
| ACC-TOK-002 | INV-TOK-020..023 | `tests/test_microgpt.c::test_build_word_vocab` |
| ACC-TOK-003 | SLO-TOK-001..003 | `tests/bench_microgpt.c` |

## 10. Cross-references

- **TDD:** `TDD_tokeniser.md`
- **Source:** `src/microgpt.h`, `src/microgpt.c` §1, §9.
- **Upstream:** none.
- **Downstream:** `BS_core.md` consumes `Vocab`/`WordVocab` for `forward_*`.

## 11. Revision history

| Version | Date | Change |
|---|---|---|
| 1.0 | 2026-04-30 | Initial extraction. |
