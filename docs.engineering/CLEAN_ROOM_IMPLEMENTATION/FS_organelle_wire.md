# FS_organelle_wire — Functional / Format Specification

**Document ID:** FS-ORG-001
**Version:** 1.0
**Status:** DRAFT
**Last updated:** 2026-04-30
**Source of truth:** `src/microgpt_organelle.c` — `organelle_generate`, `organelle_generate_multiline`, `opa_load_docs_multiline`, `opa_extract_pipe_value`, `opa_pipe_starts_with`. `OpaTrace` corpus output: `opa_trace_to_corpus`.

---

## RFC 2119

The key words MUST, MUST NOT, REQUIRED, SHALL, SHALL NOT, SHOULD, SHOULD NOT, RECOMMENDED, MAY, and OPTIONAL in this document are to be interpreted as described in RFC 2119.

## 1. Format overview

This document specifies the textual formats used by the Organelle Pipeline Architecture (OPA) for:

1. **Inference protocol** — how a prompt is fed to an organelle and how the response is delimited.
2. **Pipe-string format** — the key/value wire used by planner / player / judge organelles to exchange state.
3. **Multi-line corpus format** — how training corpora for organelles are laid out on disk.
4. **Reasoning trace corpus format** — the line-per-step format produced by `opa_trace_to_corpus`.

These are not byte-level binary formats; they are line-oriented ASCII (or UTF-8) conventions. The "schema" is enforced by the producer / consumer pair, not by a parser. Schema mismatches are a frequent failure mode and are documented as gaps in `TRACEABILITY.md`.

## 2. Inference protocol (single-line)

The default `organelle_generate` protocol feeds a prompt to an organelle and reads a single response line. The exchange uses the character-level vocabulary built from the corpus.

### 2.1 Prompt encoding

A consumer SHALL feed tokens to the model in this order:

1. The vocabulary's BOS token.
2. Each character of the prompt, mapped via `Vocab.chars` to its token ID. (Linear scan; the corpus's character set MUST contain every prompt character.)
3. A single newline character (token ID for `'\n'`). This separator is mandatory and signals "respond now".

### 2.2 Response decoding

After feeding the prompt, the engine SHALL auto-regressively sample tokens until ONE of:

- A newline character is produced — terminates the response (the newline is NOT included in the output buffer).
- The BOS token is produced — terminates the response.
- `max_len` characters have been emitted to the output buffer.
- Position `pos` reaches `cfg->block_size` — context window full.

The output buffer SHALL be NUL-terminated. Newlines within the response are forbidden under this protocol; corpora that need newlines in the response MUST use the multiline protocol (§3).

## 3. Inference protocol (multi-line)

`organelle_generate_multiline` extends the protocol to corpora whose responses span multiple lines (e.g., C function bodies). The prompt encoding (§2.1) is unchanged. Response decoding terminates on:

- A blank-line marker — the byte sequence `\n\n` (or `\n` followed by EOF) signals end of response.
- BOS — terminates as in §2.2.
- `max_len` reached.

The output buffer SHALL contain the response including any internal newlines but SHALL NOT include the terminating blank-line marker.

## 4. Pipe-string format

Planner / player / judge organelles communicate via flat key/value strings. The format is:

```
KEY1=value1|KEY2=value2|KEY3=value3
```

### 4.1 Lexical rules

- The separator between fields is the ASCII pipe `|` (`0x7C`).
- The separator between key and value is `=` (`0x3D`).
- Keys are short identifiers — `[A-Za-z][A-Za-z0-9_]*` is the convention. The reference parser does not enforce this.
- Values may contain any byte that is not `|` or `\n`.
- Trailing newlines (`\n`) terminate a record.
- Empty values (`KEY=|`) are permitted.
- An optional **prefix tag** MAY precede the first key with no `=`, e.g. `STATE|board=12345...|empties=4`. Use `opa_pipe_starts_with(buf, "STATE|")` to detect.

### 4.2 Reading semantics

A consumer SHALL read fields with `opa_extract_pipe_value(buf, "KEY")` which:

- Searches for the substring `KEY=` in the buffer.
- Returns a pointer to the byte after `=`.
- NUL-terminates the value at the next `|` or `\n` by writing `'\0'` into the buffer (the buffer is mutated in place).
- Returns NULL if the key is not found.

### 4.3 Writing semantics

Producers SHALL render pipe strings with `snprintf` or equivalent:

```
snprintf(buf, sz, "STATE|board=%s|empties=%d", board_str, empties);
```

Producers SHALL NOT include `|` or `\n` in any value. (No escape mechanism is defined in V1.0; this is documented as `GAP-WIRE-001`.)

### 4.4 Examples

| Use case | Example string |
|---|---|
| 8-puzzle state | `STATE|board=123456780|empties=1|moves=R,D,L` |
| Connect-4 plan | `PLAN|next=4|reason=block_diag` |
| Mastermind guess | `GUESS|code=AABC|score=2_1` |
| Generic action | `ACTION|move=up` |

## 5. Multi-line corpus format

`opa_load_docs_multiline` loads training corpora where each document spans multiple lines and documents are separated by blank lines. This is the standard format for OPA organelle training data.

### 5.1 Layout

```
<document 1 line 1>\n
<document 1 line 2>\n
\n
<document 2 line 1>\n
<document 2 line 2>\n
<document 2 line 3>\n
\n
...
```

### 5.2 Parsing rules

A reader SHALL:

1. Slurp the entire file into a single contiguous heap buffer.
2. Walk the buffer; treat one or more consecutive `\n` as a document boundary.
3. Group consecutive non-empty lines into a single document, **including** the terminating newline of each in the document length.
4. Produce a `Docs` struct with `data` pointing at the slurped buffer and `lines[i]` pointing at the start of document `i` within `data`. Document `i` has length `doc_lens[i]`.

### 5.3 Constraints

- File size SHALL be ≤ `2 GiB - 1` (`long` ftell range).
- The number of documents SHALL be ≤ `max_docs` passed to the loader; excess documents SHALL be silently dropped at the end of the file.
- The corpus SHALL NOT contain an embedded NUL byte; the loader appends a terminating NUL but treats earlier NULs as undefined behaviour.
- Lines MAY be of any length up to `max_doc_len` (consumer-defined; the loader does not truncate).

### 5.4 Standard organelle training shape

A document for an organelle SHOULD have the form:

```
PROMPT_LINE
RESPONSE_LINE
```

separated by blank lines. The training loop tokenises the entire document and trains the model to predict the response from the prompt.

For multi-line responses the document SHOULD have the form:

```
PROMPT_LINE
RESPONSE_LINE_1
RESPONSE_LINE_2
...
```

with the blank line still being the document terminator.

## 6. Reasoning trace corpus format

`opa_trace_to_corpus` (and `opa_trace_write`) emit a line-per-step trace of one pipeline run. Each line has the form:

```
<step>|<action>|<outcome>|<metric_before>→<metric_after>|<blocked_snapshot>|<src>
```

| Field | Meaning |
|---|---|
| `step` | 1-indexed step number |
| `action` | The proposed action string (e.g. `up`, `4`, `ABCD`) |
| `outcome` | One of `ACCEPTED`, `REJECTED`, `STALL`, `REPLAN`, `CYCLE_BREAK` (corresponding to `OpaStepOutcome`) |
| `metric_before` | Integer progress metric before the step |
| `metric_after` | Integer progress metric after the step, or `-1` when the step was rejected and the metric did not advance |
| `blocked_snapshot` | The kanban `blocked` string at the time of the decision; comma-separated, MAY be empty |
| `src` | `model` if the action came from the organelle, `fallback` if it came from the deterministic valid-move fallback |

Trace files MAY contain multiple traces concatenated, separated by blank lines (so they are themselves valid `opa_load_docs_multiline` corpora — this is intentional, the trace format is designed to be re-trainable on).

## 7. Versioning

These formats are **unversioned** in V1.0. They are project conventions enforced by the producer / consumer pair, not by a self-describing schema. Format changes are coordinated by updating this document, the producer's render code, and the consumer's parse code together.

## 8. Error conditions

| ID | Function | Failure mode | Client action |
|---|---|---|---|
| ERR-WIRE-001 | `opa_extract_pipe_value` | Key not found | Returns NULL; caller MUST handle |
| ERR-WIRE-002 | `opa_load_docs_multiline` | `fopen` failed, OOM, file > 2 GiB | Returns -1 |
| ERR-WIRE-003 | `opa_trace_to_corpus` | Output buffer too small | Returns -1; caller SHOULD enlarge buffer and retry |
| ERR-WIRE-004 | `organelle_generate` | Prompt character missing from vocabulary | The character is silently mapped to token 0; output may be degraded. The producer SHOULD pre-validate that the prompt's character set is a subset of the corpus's |

## 9. Reference implementation

- §2 protocol: `organelle_generate` in `microgpt_organelle.c`.
- §3 multi-line: `organelle_generate_multiline` in same file.
- §4 pipe strings: `opa_extract_pipe_value`, `opa_pipe_starts_with`.
- §5 corpus loader: `opa_load_docs_multiline`.
- §6 trace corpus: `opa_trace_to_corpus`, `opa_trace_write`, with the `OpaStepOutcome` enum.

## 10. Cross-references

- `BS_organelle.md` for the behavioural contracts of organelle inference.
- `FRD.md` REQ-ORG-013, REQ-ORG-014, REQ-ORG-016.
- `book/` Chapter 8 (Pipelines) — narrative context.

## 11. Revision history

| Version | Date | Change |
|---|---|---|
| 1.0 | 2026-04-30 | Initial extraction. |
