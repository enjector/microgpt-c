# VM Code Generation Organelle

A word-level Transformer trained on ~1,597 VM DSL functions to test whether targeting a constrained grammar improves code generation syntax pass rates compared to raw C. Two variants exist: the **original** (custom word-level infrastructure) and **V2** (refactored to use the organelle word-level API).

---

## Spear Summary

**Point:** Targeting the VM's constrained DSL with word-level tokenisation (~861 word tokens vs character-level ~56 chars) should dramatically reduce token-sampling noise, pushing syntax validation pass rate from C's 0% toward >80%.

**Picture:** Instead of asking a model to generate raw C code with pointers, memory management, and preprocessor directives, we ask it to generate TypeScript-like scripts with typed variables, simple loops, and function calls. The VM parser (`vm_module_compile`) provides instant, deterministic syntax validation — no `gcc` dependency.

**Proof:** Word-level tokenisation achieved **60% validated syntax pass rate** (6/10, multi-seed verified, ±0% std dev), with **80% pass on control prompts** and **40% on novel prompts**. This is up from 0% with character-level tokenisation.

**Push:** The word-level result validates the **representation beats capacity** thesis from the 8-puzzle experiments. Better input encoding (word tokens vs character tokens) produces dramatic improvement without changing model capacity.

---

## How It Works

```
   Prompt                              Generated Output
   ────────────────────                ──────────────────────────────────
   // compute factorial of n      →    function factorial(n: number): number {
                                           var result = 1;
                                           for (var i = 2; i <= n; i++) {
                                               result = result * i;
                                           }
                                           return result;
                                       }
```

1. Feed `// comment\n` as prompt, tokenised word-by-word
2. Model builds internal context from the comment tokens
3. Autoregressive generation produces function body (temp=0.3)
4. **Brace-balanced stopping**: generation halts when `}` depth returns to 0
5. **Validation gate**: pipe output through `vm_module_compile()` — instant VALID/INVALID
6. Confidence score gates output quality (~60% threshold separates known from unknown)

## Data Flow

```
 vm_functions_combined.txt (726 functions)
         │
         ▼
 ┌─────────────────────────┐
 │   vm_scan_token()       │  VM-DSL-aware lexer: splits identifiers,
 │   (custom scanner)      │  keywords, operators, numbers, punctuation
 └────────┬────────────────┘
          │  word stream
          ▼
 ┌─────────────────────────┐
 │   build_vm_word_vocab()  │  Frequency-ranked vocabulary
 │   (DJB2 hash table)     │  Top-N words + UNK + \n + BOS
 └────────┬────────────────┘
          │  token IDs
          ▼
 ┌─────────────────────────┐
 │   forward_backward_one() │  Per-position training
 │   + adam_step()          │  Cosine LR schedule, 15K steps
 └────────┬────────────────┘
          │  trained model
          ▼
 ┌─────────────────────────┐
 │   sample_token()         │  Autoregressive generation
 │   + brace-balanced stop  │  Stops at matching }, temp=0.3
 └────────┬────────────────┘
          │  generated word tokens
          ▼
 ┌─────────────────────────┐
 │   vm_detokenize()        │  Words → source text
 │   (spacing heuristics)   │
 └────────┬────────────────┘
          │  VM DSL source code
          ▼
 ┌─────────────────────────┐
 │   vm_module_compile()    │  Deterministic syntax gate
 │   (Flex/Bison parser)    │  VALID / INVALID (0% false positives)
 └─────────────────────────┘
```

## Architecture

| Parameter | Original (`main.c`) | V2 (`main_v2.c`) |
|-----------|---------------------|------------------|
| Tokenisation | VM-aware scanner | Pre-tokenized + generic word API |
| N_EMBD | 96 | 96 |
| N_LAYER | 2 | 2 |
| N_HEAD | 4 | 4 |
| MLP_DIM | 384 | 384 |
| BLOCK_SIZE | 256 | 256 |
| Corpus | 1,597 functions | 1,597 functions (pre-tokenized) |
| Vocab | 863 word tokens | 861 word tokens |
| Training | 15K steps, batch=16 (multi-threaded) | 5K steps (single-threaded, organelle API) |
| Params | ~411K | ~411K |

## Training Corpus

1,597 VM DSL functions across six domains (146 hand-crafted + 580 variations + additional expansions):

| Domain | Count | Examples |
|--------|-------|---------|
| Arithmetic / math | ~30 | factorial, fibonacci, power, gcd |
| String processing | ~5 | (limited by VM types) |
| Aggregation / stats | ~15 | sum, average, running total |
| Conditionals / logic | ~20 | abs, clamp, sign, min/max |
| Finance / physics | ~40 | interest, energy, momentum, area |
| Multi-function wiring | ~20 | compound_interest → compound, clamped_sigmoid → sigmoid + clamp |
| Prompt variations | 580+ | Paraphrased comments for each original function |

Each function has a `// descriptive comment` header — this is the prompt conditioning signal.

### Pre-Tokenized Corpus (V2)

V2 uses `pretokenize_corpus.py` to convert `vm_functions_combined.txt` into `vm_functions_pretok.txt` where all tokens are space-separated. This allows the generic `build_word_vocab()` / `tokenize_words()` API to produce the same quality tokenization as the custom VM scanner:

```
Original:  function factorial(n: number): number {
Pre-tok:   function factorial ( n : number ) : number {
```

## Key Innovations

### 1. VM-Aware Word Tokeniser

Instead of whitespace-splitting or character-level tokenisation, `vm_scan_token()` understands VM DSL grammar:

```
Character-level:  f u n c t i o n   f a c t o r i a l ( n : ...  → 175 tokens
Word-level:       function factorial ( n : number ) : number { ... →  40 tokens
```

### 2. Brace-Balanced Stopping

Generation tracks `{` and `}` depth, halting when braces balance. This prevents trailing garbage that was the sole cause of syntax failures.

### 3. VM Validation Gate

```c
vm_module *module = NULL;
result r = vm_module_compile(NULL, generated_code, &module);
if (r == RESULT_OK && module && sequence_count(module->errors) == 0) {
    printf("  >> VM SYNTAX: ✅ VALID\n");
}
```

Deterministic, in-memory syntax gate with zero external dependencies.

## Results

### Original (`vm_codegen`, custom word infra)

| Metric | Char-Level | Word-Level (96-dim, 1597 fn) |
|--------|------------|-----------------------------|
| Vocab size | 56 chars | 863 words |
| Tokens/function | ~175 | ~54 |
| Training time | 3 min | 17 min (15K steps, 16 threads) |
| Final loss | 0.32 | 1.14 |
| **Syntax pass** | **0%** | **10/10 (100%)** |

> ⚠️ **Validation note**: The original tests on prompts drawn from the training corpus. This 100% measures memorisation, not generalisation. See Validation section below.

### V2 (`vm_codegen_v2`, organelle API) — Validated

| Metric | V2 (5K steps) | V2 (10K steps) |
|--------|:---:|:---:|
| Vocab size | 861 words | 861 words |
| Training time | 5 min (single-threaded) | 10 min |
| Best loss | 0.110 | 0.091 |
| **Controls (in-corpus)** | **4/5 (80%)** | **4/5 (80%)** |
| **Novel (out-of-corpus)** | **2/5 (40%)** | **1/5 (20%)** |
| **Total** | **6/10 (60%)** | **5/10 (50%)** |
| Multi-seed verified | ✅ 60% ± 0% (2 seeds) | Single run |

> Overfitting detected: 5K→10K steps improves loss but degrades accuracy.

### Validation Summary

Validated per `/validate-experiment` workflow:
- **§1.1 Reproducibility**: ✅ PASSED — 60% ± 0% across 2 seeds
- **§1.3 Train/Novel split**: ✅ Clean separation
- **§1.4 Error analysis**: Systematic failures (same prompts fail every seed)
- **§2.3 Training curve**: ⚠️ Overfitting above 5K steps
- **§2.7 Significance**: N=10 too small for statistical significance

## Build & Run

```bash
mkdir build && cd build
cmake ..

# Original (custom word infra, 15K steps, ~17 min)
cmake --build . --target vm_codegen && ./vm_codegen

# V2 (organelle word API, 5K steps, ~5 min)
cmake --build . --target vm_codegen_v2 && ./vm_codegen_v2
```

Both auto-resume from checkpoint.

### Pre-tokenizing the corpus (V2 only)

```bash
cd demos/character-level/vm_codegen
python3 pretokenize_corpus.py   # generates vm_functions_pretok.txt
```

## Files

| File | Description |
|------|-------------|
| `main.c` | Original — custom VM-aware word scanner + training loop (755 lines) |
| `main_v2.c` | V2 — organelle word API + ensemble voting (382 lines, 49% reduction) |
| `pretokenize_corpus.py` | Converts corpus to space-separated tokens for generic word API |
| `vm_functions_combined.txt` | Training corpus (1,597 functions) |
| `vm_functions_pretok.txt` | Pre-tokenized corpus (for V2) |
| `generate_variations.py` | Generates paraphrased prompt variations |

## Connection to ORGANELLE_GENERALISATION_VM.md

This experiment implements **Phase 1–2** of the roadmap described in `ORGANELLE_GENERALISATION_VM.md`:
- Phase 1: Corpus generation (vm_functions.txt + variations)
- Phase 2: Train vm_codegen organelle (word-level, 60% validated syntax pass)

Phase 3 (vm_compose pipeline) is implemented in `../vm_compose/`.

---

*MicroGPT-C — Enjector Software Ltd. MIT License.*
