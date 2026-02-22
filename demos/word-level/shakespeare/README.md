# Shakespeare Word-Level Generation

A word-level Transformer learns to write Shakespeare word by word — using the library's `WordVocab` and `tokenize_words` API.

---

## Spear Summary

**Point:** Word-level tokenisation compresses Shakespeare into ~1/5th the sequence length, letting a much smaller model (510K vs 900K params) learn prose structure instead of spelling.

**Picture:** Where the character-level demo needs 300 tokens to generate ~60 words, this demo generates 100 words in 100 tokens. Each token carries a whole word's meaning.

**Proof:** Text normalisation (lowercase + punctuation splitting) reduces vocabulary fragmentation so 5,000 word slots cover ~93% of the corpus. The model generates coherent Shakespearean phrases with only ~7% unknown words displayed as `...`:

```
o, bid thee. so... me, i think all the king. let thee, some my lord...
```

**Push:** Compare with the character-level demo: word-level produces more coherent phrases but can't spell rare words; character-level never has unknowns but struggles with long-range structure.

---

## How It Works

1. Load Shakespeare's complete works (line-per-document)
2. **Normalise** text: lowercase all characters, split punctuation into separate tokens
3. Build a word vocabulary (top 5,000 by frequency + UNK + newline + BOS)
4. Pre-tokenize all documents into word token sequences
5. Train a decoder-only Transformer using single-threaded forward-backward
6. Generate new text from seed words ("the", "o", "what", "my", "how")

### Text Normalisation

The demo pre-processes all text before vocabulary building and tokenisation:

- **Lowercase**: `"The"` → `"the"` — merges case variants into one token
- **Punctuation splitting**: `"art,"` → `"art" ","` — separates words from punctuation

This reduces effective unique words from ~8,000 to ~4,000, so 5,000 vocab slots cover 93% of the corpus:

```
Before: [John.] [I] [wonder] [thou] [UNK] [art,] [born]
After:  [john] [.] [i] [wonder] [thou] [(] [being] [,] [art] [,]
```

## Architecture

| Parameter | Value |
|-----------|-------|
| Tokenisation | Word-level (normalised + whitespace split) |
| N_EMBD | 48 |
| N_LAYER | 1 |
| N_HEAD | 4 |
| MLP_DIM | 192 |
| BLOCK_SIZE | 64 |
| Max vocab | 5,000 words |
| Params | ~510K |
| Training steps | 10,000 |
| Learning rate | 0.001 |
| UNK rate | ~7% (displayed as `...`) |
| Gen length | 100 words/sample |
| Temperature | 0.5 |
| Samples | 5 |

## Build & Run

```bash
mkdir build && cd build
cmake .. && cmake --build . --target shakespeare_word_demo
./shakespeare_word_demo
```

Auto-resumes from checkpoint (`shakespeare_word.ckpt`).

## Comparison: Character-Level vs Word-Level

| Aspect | Character-Level | Word-Level |
|--------|----------------|-----------|
| **What it learns** | Spelling + structure | Phrase structure |
| **Why** | Each char is a token — must learn letter patterns | Each word is a token — skips spelling |
| Vocab size | ~80 chars | ~5,000 words |
| Tokens per word | ~5 characters | 1 word token |
| Model size | ~900K params | ~510K params |
| Training steps | 30,000 | 10,000 |
| Training time | ~5 min | ~2 min |
| N_LAYER | 4 | 1 |
| Unknown rate | 0% | ~7% (shown as `...`) |
| **Result** | Spells correctly, struggles with coherence | Coherent phrases, gaps for rare words |

**Key insight:** Word-level needs a much smaller model because sequences are ~5× shorter. But it trades spelling ability for phrase-level coherence. Both approaches are valid — the right choice depends on whether your vocabulary is open (character-level) or constrained (word-level).

## Sample Output

```
the truth of..., and with such born to have heard a son,
and..., and, what you... the line... fire. we... eyes out,
be. i will marry... time, the pure... suffolk? i tell you...

o, bid thee. so... me, i think all the king. let thee,
some my lord... that after..., sir andrew. would not the...
even so... audrey gone. i dare not say thou art thou art...
```

## Corpus

`shakespeare.txt` — Shakespeare's complete works (~5.4 MB), shared with the character-level demo.

---

*MicroGPT-C — Enjector Software Ltd. MIT License.*
