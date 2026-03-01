# VM Code Generation Organelle

A word-level Transformer trained on ~726 VM DSL functions to test whether targeting a constrained grammar improves code generation syntax pass rates compared to raw C.

---

## Spear Summary

**Point:** Targeting the VM's constrained DSL with word-level tokenisation (~474 word tokens vs character-level ~56 chars) should dramatically reduce token-sampling noise, pushing syntax validation pass rate from C's 28% toward >80%.

**Picture:** Instead of asking a model to generate raw C code with pointers, memory management, and preprocessor directives, we ask it to generate TypeScript-like scripts with typed variables, simple loops, and function calls. The VM parser (`vm_module_compile`) provides instant, deterministic syntax validation — no `gcc` dependency.

**Proof:** Word-level tokenisation achieved **60% syntax pass rate** (6/10), with **100% pass on control prompts**. This is up from 0% with character-level tokenisation.

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

| Parameter | Value |
|-----------|-------|
| Tokenisation | Word-level (VM-aware scanner) |
| N_EMBD | 64 |
| N_LAYER | 1 |
| N_HEAD | 4 |
| MLP_DIM | 256 |
| BLOCK_SIZE | 256 |
| Corpus | ~726 functions (146 + 580 variations) |
| Vocab | 500 word tokens |
| Training | 15K steps, batch=16 |
| Params | ~129K |

## Training Corpus

726 VM DSL functions across six domains (146 hand-crafted + 580 prompt variations):

| Domain | Count | Examples |
|--------|-------|---------|
| Arithmetic / math | ~30 | factorial, fibonacci, power, gcd |
| String processing | ~5 | (limited by VM types) |
| Aggregation / stats | ~15 | sum, average, running total |
| Conditionals / logic | ~20 | abs, clamp, sign, min/max |
| Finance / physics | ~40 | interest, energy, momentum, area |
| Multi-function wiring | ~20 | compound_interest → compound, clamped_sigmoid → sigmoid + clamp |
| Prompt variations | 580 | Paraphrased comments for each original function |

Each function has a `// descriptive comment` header — this is the prompt conditioning signal.

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

| Metric | Char-Level (48-dim) | Word-Level (48-dim) | Word-Level (64-dim, 726 fn) |
|--------|-------------------|-------------------|--------------------------|
| Vocab size | 56 chars | 474 words | 500 words |
| Tokens/function | ~175 | ~53 | ~53 |
| Training time | 3 min | 3 min | 5 min |
| Final loss | 0.32 | 0.087 | ~0.10 |
| **Syntax pass (controls)** | **0%** | **100% (5/5)** | **100% (5/5)** |
| **Syntax pass (novel)** | **0%** | **20% (1/5)** | **60% (3/5)** |
| **Syntax pass (total)** | **0%** | **60% (6/10)** | **80% (8/10)** |

## Build & Run

```bash
mkdir build && cd build
cmake .. && cmake --build . --target vm_codegen
./vm_codegen    # trains from scratch or loads checkpoint
```

Auto-resumes from checkpoint (`vm_codegen.ckpt`).

## Connection to ORGANELLE_GENERALISATION_VM.md

This experiment implements **Phase 1–2** of the roadmap described in `ORGANELLE_GENERALISATION_VM.md`:
- Phase 1: Corpus generation (vm_functions.txt + variations)
- Phase 2: Train vm_codegen organelle (word-level, 60% syntax pass)

If successful at >80%, Phase 3 integrates this into the `vm_compose` pipeline.

---

*MicroGPT-C — Enjector Software Ltd. MIT License.*