# C Code Generation Organelle

An 875K-parameter Transformer trained on 2,081 C functions achieves byte-perfect code retrieval from natural language prompts — but cannot write a single line it hasn't memorised.

---

## Spear Summary

**Point:** This model is a neural grep — it maps description strings to memorised function bodies with byte-level fidelity but has zero ability to compose new code.

**Picture:** It's like a librarian who knows exactly where every book is but can't write a single sentence. Ask for "bubble sort" and you get a flawless copy. Ask for "reverse array" — something any programmer could write in 10 seconds — and you get gibberish.

**Proof:** `/* sort values in ascending order */` produces a perfect `bubble_sort` at 100% confidence. `/* ascending sort */` — the exact same concept in different words — produces garbled token soup at 35%. The model memorised the string not the meaning.

**Push:** Stop trying to make one model do everything. Chain a "wiring" organelle (maps novel descriptions to known patterns) with this code organelle (retrieves implementations). That's exactly what organelle pipelines are for.

---

## How It Works

```
   Prompt                              Generated Output
   ────────────────────                ──────────────────────────────────
   /* FFT with Hamming window */  →    void fft_windowed_hamming(double *re,
                                         double *im, const double *signal,
                                         int n) {
                                         double w[1024];
                                         hamming_window(w, n);
                                         for (int i = 0; i < n; i++) {
                                           re[i] = signal[i] * w[i];
                                           im[i] = 0;
                                         }
                                         fft_radix2(re, im, n);
                                       }
```

1. Feed `/* comment */` as prompt, character by character
2. Model builds internal context from the comment string
3. Autoregressive generation produces the function body (temp=0.3)
4. Confidence score gates output quality (~80% threshold separates known from unknown)

## Architecture

| Parameter | Small Model | Scale-Up |
|-----------|-------------|----------|
| N_EMBD | 64 | **128** |
| N_LAYER | 2 | **4** |
| N_HEAD | 4 | 4 |
| BLOCK_SIZE | 512 | 512 |
| Total params | 142,080 | **875,008** |
| Corpus | 190 functions (46 KB) | **2,081 functions (492 KB)** |
| Params/byte | 3.1:1 | **1.78:1** |
| Training | 10K steps (~10 min) | **50K steps (~9 hours)** |

## Training Corpus

2,081 numerical-recipes-style C functions across five domains:

| Domain | Count | Examples |
|--------|-------|---------|
| Core numerical | ~50 | FFT, sorting, searching, root-finding |
| Linear algebra | ~20 | LU, Cholesky, matrix multiply, dot product |
| Statistics | ~40 | Skewness, kurtosis, percentile, z-score, KDE |
| Signal processing | ~40 | Goertzel, STFT, cepstrum, biquad, Welch PSD |
| Technical analysis | ~40 | RSI, MACD, Bollinger, Stochastic, ATR, VWAP |

Each function has a `/* descriptive comment */` header — this is the prompt conditioning signal.

## Results

### Progression Across Training

| Prompt | 27K (loss 0.15) | 30K (loss 0.10) | 50K (loss 0.03) |
|--------|----------------|----------------|----------------|
| `mean()` | Garbled | ✅ Perfect | ✅ **100% conf** |
| `bubble_sort()` | Garbled | ⚠️ 1 char error | ✅ **100% conf** |
| `FFT Hamming` | `hamm_wing_window` | ✅ Perfect | ✅ **100% conf** |
| `MACD histogram` | Correct sig | ❌ Regressed | ✅ **100% conf** |
| **Novel prompts** | ❌ 0/N | ❌ 0/N | ❌ **0/10** |

### Final Checkpoint (50K Steps, Loss 0.034)

**Corpus matches: 7/7 byte-perfect at 100% generation confidence.**

```c
/* sort values in ascending order */
void bubble_sort(double *a, int n) {
  for (int i = 0; i < n - 1; i++)
    for (int j = 0; j < n - i - 1; j++)
      if (a[j] > a[j + 1]) { double t = a[j]; a[j] = a[j + 1]; a[j + 1] = t; }
}
```

```c
/* smoothed RSI */
void rsi_smoothed(double *out, const double *close, int n,
                  int rsi_period, int ma_period) {
  double raw_rsi[1024];
  rsi(raw_rsi, close, n, rsi_period);
  sma(out, raw_rsi, n - rsi_period, ma_period);
}
```

**Novel prompts: 0/10 — all garbled.** Including trivially simple operations like "reverse array" and "array of square numbers."

### Confidence as Quality Gate

```
  Corpus matches:  82–91% confidence  →  100% generation confidence
  Novel prompts:   35–74% confidence  →  garbled output
  Threshold:       ~80%               →  zero false positives/negatives
```

The confidence score reliably distinguishes "known" from "unknown" prompts.

## Key Findings

### 1. It's a retrieval system not a generator

The paraphrase test is definitive: same concept, different wording → total failure. The model memorised strings, not semantics.

### 2. Comment→code document structure is critical

Treating `/* comment */ + function body` as a single training document was the breakthrough. Line-by-line splitting failed completely because the model never learned the association.

### 3. The model learns function dependency graphs

MACD histogram → calls `macd()`. Smoothed RSI → chains `rsi()` → `sma()`. FFT Hamming → chains `hamming_window()` → `fft_radix2()`. Inter-function relationships are learned from the corpus.

### 4. More training improves recall, not composition

Training from 27K → 50K steps eliminated every character-level error in corpus matches. But novel composition remained at 0/10 throughout. More training = better lookup table, not smarter programmer.

## Build & Run

```bash
# Scale-up (875K params, ~9h training)
mkdir build && cd build
cmake .. && cmake --build . --target c_codegen
./c_codegen    # trains from scratch or loads checkpoint

# Edit prompts[] in main.c to test different generation targets
```

Auto-resumes from checkpoint (`c_codegen.ckpt`). Best checkpoints saved as `c_codegen_best_stepN_lossX.ckpt`.

## Recommended Next Steps

| Priority | Change | Impact |
|----------|--------|--------|
| **P0** | Chain with wiring organelle (c_wiringgen) | Enable novel composition via two-stage pipeline |
| **P1** | Push params/byte below 1:1 (more corpus or smaller model) | Force compositional learning |
| **P1** | Use confidence gate in production | ~80% threshold for reliable quality filtering |
| **P2** | Test with Metal GPU acceleration | Faster training on Apple Silicon |

---

*MicroGPT-C — Enjector Software Ltd. MIT License.*
