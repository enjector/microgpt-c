# OPA Code Composition Pipeline

**Status: Two experiments complete – key findings on capacity scaling**

Trains two organelles (c_planner + c_judge) to decompose natural language intents
into flat-string function composition plans, then validates them against a registry
of 523 known C functions.

## Architecture

```
/* comment */ -> c_planner -> "seq|fn1|fn2" -> c_judge -> PASS/FAIL
```

- **c_planner**: Maps `/* smooth then differentiate */` to `seq|rolling_mean|diff_central`
- **c_judge**: Validates plans as PASS or FAIL
- **Registry**: 523 known function names from c_codegen + c_wiringgen
- **Constrained decoding**: Edit-distance nearest-name filter (v2)

## Corpus

| Corpus | v1 (Baseline) | v2 (Expanded) |
|--------|---------------|---------------|
| c_planner.txt | 512 train | 1452 train (+940 synonym variations) |
| c_judge.txt | 1321 pairs | 3779 pairs |
| test_intents.txt | 128 held-out | 128 held-out (same split) |
| c_registry.txt | 523 functions | 523 functions |

## Experiment Results

### v1 Baseline (462K params, 25K steps, 512 train)

| Metric | Result |
|--------|--------|
| **Plan parse rate** | **96%** (123/128) |
| All fns in registry | 4% (5/128) |
| Judge PASS | 65% (83/128) |
| Exact match | 2% (2/128) |
| Planner best loss | 0.072 |
| Judge best loss | ~0.14 |
| Training time | 276s planner + 156s judge = **7 min** |

### v2 All Improvements (1.2M params, 50K steps, 1452 train, constrained decoding)

| Metric | Result | vs Baseline |
|--------|--------|------------|
| **Plan parse rate** | **20%** (26/128) | ↓76% |
| All fns in registry | 11% (14/128) | ↑7% |
| Judge PASS | 2% (2/128) | ↓63% |
| Exact match | 0% (0/128) | ↓2% |
| Parse errors | 102 | ↑97 |
| Planner best loss | 0.14 | ↑0.07 |
| Judge best loss | ~0.26 | ↑0.12 |
| Training time | 1821s planner + 899s judge = **45 min** |

> [!CAUTION]
> **v2 regressed catastrophically.** The 1.2M model diverged after step ~7K
> (lr=0.001 too aggressive), producing mostly garbage output. The best-checkpoint
> mechanism preserved the step-7K weights, but those were still severely underfit.

## Key Findings

### 1. Format is trivially learnable (v1: 96% parse)
The char-level GPT learns the `seq|fn1|fn2` flat-string structure perfectly at 462K params.
This validates the OPA approach for code composition.

### 2. Function names need more capacity (v1: 4% registry hit)
73 unique function names × ~10 chars each is a much harder vocabulary than
single-character game moves. The model generates garbled approximations
(e.g., `difff_cententpe` instead of `diff_central`).

### 3. Capacity without LR tuning is destructive
Scaling from 462K → 1.2M params at the same lr=0.001 causes catastrophic
divergence. Both organelles diverged after step ~5-10K and never recovered.
**Larger models need lower learning rates** (e.g., 0.0003 or lr warmup).

### 4. Constrained decoding needs valid raw output first
The edit-distance filter (threshold ≤3) boosted registry hit from 4% → 11%
where plans parsed, but can't help when 80% of outputs are unparseable garbage.

### 5. Corpus expansion alone isn't sufficient
Expanding from 512 → 1452 with synonym variations provided more diversity,
but the benefits were masked by the lr divergence issue.

## Recommended Next Steps

1. **Reduce learning rate**: Try lr=0.0003 or lr=0.0005 for 1.2M model
2. **Add LR warmup/decay**: Linear warmup → cosine decay schedule
3. **Intermediate capacity**: Try 96/8/4 with 50K steps (same as working v1 but longer)
4. **Temperature tuning**: Lower planner temperature for more deterministic output

## Build & Run

```bash
cd build && cmake .. && cmake --build . --target c_compose
./c_compose
```

## Files

- `generate_corpus.py` — Corpus generator with synonym-based variation expansion
- `main.c` — Pipeline orchestrator with ensemble voting + constrained decoding
- `c_planner.txt` / `c_judge.txt` — Training corpora
- `c_registry.txt` — Function name registry (523 names)
- `test_intents.txt` — Held-out test intents (128)
