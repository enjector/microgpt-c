# FS_checkpoint — Functional / Format Specification

**Document ID:** FS-CKPT-001
**Version:** 1.0
**Status:** DRAFT
**Last updated:** 2026-04-30
**Source of truth:** `src/microgpt.c` — `model_save`, `model_load`, `checkpoint_save`, `checkpoint_load`, `write_doubles`, `read_doubles`.

---

## RFC 2119

The key words MUST, MUST NOT, REQUIRED, SHALL, SHALL NOT, SHOULD, SHOULD NOT, RECOMMENDED, MAY, and OPTIONAL in this document are to be interpreted as described in RFC 2119.

## 1. Format overview

This document specifies the byte-level layout of two files produced by the MicroGPT-C engine:

1. **Weights file** — written by `model_save`, consumed by `model_load`. Contains a vocabulary-size header and the model's flat-array weights. Used for inference-only deployments.
2. **Training checkpoint** — written by `checkpoint_save`, consumed by `checkpoint_load`. Contains the same weights plus the Adam optimiser's first- and second-moment buffers and the current step counter. Used for resumable training.

Both files are little-endian-implied — the engine writes raw native bytes via `fwrite`. Implementations that target a big-endian host MUST byte-swap to the producer's endianness before writing and after reading. (The reference implementation does not handle cross-endian portability; the format MUST NOT be assumed portable across endianness.)

**Out of scope:** when `QUANTIZATION_INT8` (or `QUANTISATION_INT8`) is defined, both `model_save` and `checkpoint_save` SHALL return -1 (the format is undefined for INT8 builds in V1.0). The corresponding load functions SHALL return NULL.

## 2. Common types and conventions

| Type | Size | Notes |
|---|---|---|
| `int` | 4 bytes (typical platforms) | Signed two's-complement; used only for the step counter |
| `size_t` | 8 bytes on 64-bit hosts, 4 bytes on 32-bit hosts | Used for `vocab_size` |
| `scalar_t` | 4 bytes if `MICROGPT_USE_FLOAT` is defined at build time, else 8 bytes | The IEEE-754 binary type that matches the producer's build |

A consumer MUST be built with the same `MICROGPT_USE_FLOAT` setting and the same `size_t` width as the producer; otherwise loading SHALL fail. (The current reference implementation does not encode this in the file; it is the deployer's responsibility to keep build settings consistent.)

The architecture macros `N_EMBD`, `BLOCK_SIZE`, `MLP_DIM`, `N_LAYER` MUST also match the producer; mismatches will produce silent corruption because the layout is computed from those macros. (This is `GAP-FMT-001` — the format does not self-describe its architecture; see `TRACEABILITY.md`.)

## 3. Weights file (`model_save` / `model_load`)

### 3.1 Byte-level layout

| Offset | Size | Field | Type | Notes |
|---:|---:|---|---|---|
| 0 | `sizeof(size_t)` | `vocab_size` | `size_t` | MUST equal the consumer's expected `vocab_size`; mismatch SHALL cause `model_load` to return NULL |
| H | `vocab_size × N_EMBD × sizeof(scalar_t)` | `wte` | `scalar_t[]` | Token embedding, row-major `[vocab_size × N_EMBD]` |
| H + W | `BLOCK_SIZE × N_EMBD × sizeof(scalar_t)` | `wpe` | `scalar_t[]` | Position embedding, row-major `[BLOCK_SIZE × N_EMBD]` |
| H + W + P | `vocab_size × N_EMBD × sizeof(scalar_t)` | `lm_head` | `scalar_t[]` | Output projection, row-major `[vocab_size × N_EMBD]` |
| ... | per-layer block, repeated `N_LAYER` times | (see §3.2) | | |
| ... | per-layer attention-residual block, repeated `N_LAYER` times if `MICROGPT_ATTN_RES` was defined at build time | (see §3.3) | | |

`H = sizeof(size_t)`, `W = vocab_size × N_EMBD × sizeof(scalar_t)`, `P = BLOCK_SIZE × N_EMBD × sizeof(scalar_t)`.

### 3.2 Per-layer block

For each layer L in `0..N_LAYER-1`, in order:

| Order | Field | Shape | Bytes |
|---:|---|---|---|
| 1 | `attn_wq[L]` | `[N_EMBD × N_EMBD]` | `N_EMBD² × sizeof(scalar_t)` |
| 2 | `attn_wk[L]` | `[N_EMBD × N_EMBD]` | as above |
| 3 | `attn_wv[L]` | `[N_EMBD × N_EMBD]` | as above |
| 4 | `attn_wo[L]` | `[N_EMBD × N_EMBD]` | as above |
| 5 | `mlp_fc1[L]` | `[MLP_DIM × N_EMBD]` | `MLP_DIM × N_EMBD × sizeof(scalar_t)` |
| 6 | `mlp_fc2[L]` | `[N_EMBD × MLP_DIM]` | as above |

All matrices are stored row-major with the convention `[output_dim × input_dim]`.

### 3.3 Per-layer attention-residual block (only if `MICROGPT_ATTN_RES`)

If and only if the producer was built with `-DMICROGPT_ATTN_RES`, the file additionally contains, after the layer block above, for each layer L in `0..N_LAYER-1`:

| Order | Field | Shape | Bytes |
|---:|---|---|---|
| 1 | `attn_res_proj[L]` | `[N_EMBD]` | `N_EMBD × sizeof(scalar_t)` |
| 2 | `mlp_res_proj[L]` | `[N_EMBD]` | as above |

A consumer that lacks `MICROGPT_ATTN_RES` MUST NOT attempt to read these fields; a consumer that has `MICROGPT_ATTN_RES` MUST read them.

This is a known portability hazard (`GAP-FMT-002`): nothing in the file marks whether the residual block is present. The consumer is responsible for matching its build flags to the producer's.

## 4. Training checkpoint (`checkpoint_save` / `checkpoint_load`)

### 4.1 Byte-level layout

| Offset | Size | Field | Type | Notes |
|---:|---:|---|---|---|
| 0 | `sizeof(int)` | `step` | `int` | Current training step (0-indexed) |
| 4 | `sizeof(size_t)` | `vocab_size` | `size_t` | MUST equal the consumer's expected `vocab_size` |
| ... | (weight body, identical to §3.1) | | | |
| ... | `N × sizeof(scalar_t)` | Adam `m` | `scalar_t[]` | First-moment estimates, flat array of length `N = model_num_params(model)` |
| ... | `N × sizeof(scalar_t)` | Adam `v` | `scalar_t[]` | Second-moment estimates, same length and ordering |

Where `N` (`model_num_params`) is the total parameter count:
```
N = vocab_size × N_EMBD                  [wte]
  + BLOCK_SIZE × N_EMBD                  [wpe]
  + vocab_size × N_EMBD                  [lm_head]
  + N_LAYER × ( 4 × N_EMBD²
              + 2 × MLP_DIM × N_EMBD )   [per-layer attention + MLP]
  + N_LAYER × 2 × N_EMBD                 [per-layer attn_res, if MICROGPT_ATTN_RES]
```

### 4.2 Adam buffer ordering

The Adam `m` and `v` buffers are laid out in the **same order as the model's parameters**, that is:

```
[ wte | wpe | lm_head | (per layer L=0..N_LAYER-1: wq, wk, wv, wo, fc1, fc2) ]
```

This ordering matches the diagram in `microgpt.c` (the "MEMORY LAYOUT (flat gradient / Adam buffers)" comment) and is critical: a consumer that loads the buffers in any other order SHALL produce silent training corruption.

When `MICROGPT_ATTN_RES` is defined, the residual projections (`attn_res_proj`, `mlp_res_proj`) are appended to each per-layer chunk in the model's parameter accounting; they are written as part of the buffer in the same chunk order.

## 5. Field semantics

### 5.1 `vocab_size`

The vocabulary size used to allocate the model. For a character-level model this is the count of unique characters seen in the training corpus plus one (for the BOS token). For a word-level model this is `num_kept_words + 3` (for `<unk>`, newline, BOS).

### 5.2 `step`

The training step at which the checkpoint was saved (0-indexed). Used by the LR scheduler to resume the cosine decay phase.

### 5.3 Weight matrices

Row-major, `[output_dim × input_dim]`. To compute `y = W @ x` from the stored bytes, treat the buffer as a flat `scalar_t[output_dim × input_dim]` and accumulate:

```
y[j] = sum_{i=0}^{input_dim-1} W[j * input_dim + i] * x[i]
```

## 6. Versioning

The format is **unversioned** — there is no version byte at offset 0. This is a known limitation (`GAP-FMT-003`).

Compatibility rules in the absence of an explicit version field:

- A consumer SHALL refuse to load a file whose `vocab_size` header does not match its own expected vocabulary.
- A consumer SHALL refuse to load a file that runs out of bytes before the expected layout is filled.
- A consumer SHOULD validate that the resulting weights produce reasonable logits on a known test prompt before deploying the model.

A future format revision MUST add a magic number and a version byte at offset 0, and MUST self-describe `(N_EMBD, N_HEAD, N_LAYER, BLOCK_SIZE, MLP_DIM, scalar_t_width)`. This is recorded as `GAP-FMT-003` in `TRACEABILITY.md` with disposition `DEFERRED` to V2.0 of the format.

## 7. Error codes

The save/load surface uses C return codes:

| ID | Function | Code | Conditions | Client action |
|---|---|---|---|---|
| ERR-CKPT-001 | `model_save` | -1 | `fopen` failed (cannot open path for writing); a `fwrite` short-write occurred | Verify path is writable; ensure adequate disk space; do NOT retry without changing inputs |
| ERR-CKPT-002 | `model_load` | NULL | `fopen` failed; vocab-size mismatch; short-read on any buffer | Verify path; verify `vocab_size` matches; verify build flags match producer |
| ERR-CKPT-003 | `checkpoint_save` | -1 | As ERR-CKPT-001 | As ERR-CKPT-001 |
| ERR-CKPT-004 | `checkpoint_load` | NULL | As ERR-CKPT-002 plus failure to allocate `m`/`v` buffers | Verify path; verify `vocab_size`; ensure caller's `m` and `v` buffers are at least `model_num_params(model)` × `sizeof(scalar_t)` bytes |
| ERR-CKPT-005 | All four functions in INT8 build | -1 / NULL | Format is undefined when `QUANTIZATION_INT8` is set | Use a non-quantised build to save / load; quantised checkpoints are out of scope for V1.0 |

## 8. Normative example

A names-demo checkpoint with `N_EMBD=16`, `N_HEAD=4`, `N_LAYER=1`, `BLOCK_SIZE=16`, `MLP_DIM=64`, `vocab_size=27`, `MICROGPT_USE_FLOAT=ON`, no `MICROGPT_ATTN_RES`:

```
Offset (bytes)   Field              Length              Cumulative
   0             step (int)           4                       4
   4             vocab_size (size_t)  8                      12
  12             wte                  27 × 16 × 4 =  1,728 1,740
1,740            wpe                  16 × 16 × 4 =  1,024 2,764
2,764            lm_head              27 × 16 × 4 =  1,728 4,492
4,492            attn_wq[0]          16 × 16 × 4 =  1,024 5,516
5,516            attn_wk[0]                       1,024 6,540
6,540            attn_wv[0]                       1,024 7,564
7,564            attn_wo[0]                       1,024 8,588
8,588            mlp_fc1[0]          64 × 16 × 4 =  4,096 12,684
12,684           mlp_fc2[0]          16 × 64 × 4 =  4,096 16,780
16,780           Adam m              N × 4 = 4,192 × 4 = 16,768  33,548
33,548           Adam v                                  16,768  50,316
```

A weights-only file would omit the `step` header and the `m` / `v` blocks, starting from offset 0 with the `vocab_size` field, total length 16,772 bytes.

## 9. Reference implementation

- Producer (full): `src/microgpt.c::checkpoint_save`
- Consumer (full): `src/microgpt.c::checkpoint_load`
- Producer (weights only): `src/microgpt.c::model_save`
- Consumer (weights only): `src/microgpt.c::model_load`
- Helpers: `src/microgpt.c::write_doubles`, `read_doubles`

## 10. Test vectors

A reproducible test vector exists in `tests/test_microgpt.c` — the named `TEST(checkpoint_roundtrip)` (or equivalent) saves a deterministic-seed-initialised model, reloads it, and asserts byte-identity of the in-memory model state. New format implementations SHOULD be validated against this vector.

## 11. Cross-references

- `BS_core.md` for the model lifecycle invariants that produce the parameters this format serialises.
- `BS_tokeniser.md` for the meaning of `vocab_size`.
- `FRD.md` REQ-CKPT-001 .. REQ-CKPT-006.
- `NFRD.md` SLO-CORE-007 (`checkpoint_save` + `load` round-trip throughput).

## 12. Revision history

| Version | Date | Change |
|---|---|---|
| 1.0 | 2026-04-30 | Initial extraction. |
