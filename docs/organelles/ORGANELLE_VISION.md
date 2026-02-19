# Organelle CLI — Design Document

**Author:** Ajay Soni, Enjector Software Ltd.

**Status:** Draft — February 2026

---

## Spear Summary

**Point:** Seven separate demo executables with 300–600 lines of boilerplate each is a scaling problem that one CLI binary with an INI config file can eliminate.

**Picture:** It's like building a new kitchen from scratch every time you want to cook a different meal. The CLI is the universal kitchen — bring your recipe (corpus + config) and it handles the rest.

**Proof:** Every demo (names, Shakespeare, tic-tac-toe, Connect-4, 8-puzzle, c_codegen, c_wiringgen) shares the same 5-step pattern: load corpus → build vocab → create model → train → infer. That pattern belongs in one binary, not seven.

**Push:** Ship Phase 1 (single organelle `microgpt train/infer`) with runtime config. Accept the ~2× perf hit at small model sizes and add tiered compilation later for large models.

---

## 1. Problem Statement

MicroGPT-C currently ships 7 separate demo executables, each with its own `main.c`. Every new organelle or pipeline requires:

1. A new `main.c` with ~300–600 lines of boilerplate (corpus loading, vocab build, training loop, checkpointing, inference)
2. A new CMake target with architecture-specific `DEFINES`
3. Manual coordination for multi-organelle setups (Kanban struct, judge logic, game loops)

This does not scale. A user who wants to train a custom organelle — even a simple one — must write C code, understand CMake, and know which macros to override. The "stem cell" vision ([VISION.md](VISION.md)) promises composable intelligence blocks; the developer experience should match.

---

## 2. Proposed Solution

A single CLI binary, `microgpt`, that reads a declarative config file and handles the full organelle lifecycle:

```
microgpt create  organelle.ini     # scaffold corpus + config
microgpt train   organelle.ini     # train one or more organelles
microgpt infer   organelle.ini     # run inference or pipeline simulation
microgpt info    model.bin         # inspect a saved model
```

> [!IMPORTANT]
> **Zero-dependency constraint**: the CLI must remain pure C99 with no external libraries. Config parsing uses a simple INI-style format, not YAML/JSON.

---

## 3. Key Design Decision: Compile-Time vs Runtime

The hot paths in `microgpt.c` read compile-time `#define` macros (`N_EMBD`, `N_LAYER`, etc.) for maximum compiler optimisation. This creates a fundamental tension:

| Approach | Performance | Flexibility | Complexity |
|----------|-------------|-------------|------------|
| **A. One binary per config** | Full (macros) | Low — must pre-compile variants | CMake targets |
| **B. Runtime config reads** | ~2× slower | Full — any config at runtime | Minimal |
| **C. Tiered — builder + runner** | Full for training | Full for inference | Medium |

### Recommended: Approach C — Tiered Architecture

```
┌─────────────────────────────────┐
│ microgpt CLI (runtime config)   │  ← "builder" mode
│  • create, info, simple infer   │
│  • Reads organelle.ini          │
│  • Acceptable perf for tooling  │
└─────────────┬───────────────────┘
              │ generates
              ▼
┌─────────────────────────────────┐
│ Per-organelle binary (macros)   │  ← "runner" mode
│  • train, fast infer, pipeline  │
│  • Compiled with -DN_EMBD=128   │
│  • Maximum performance          │
└─────────────────────────────────┘
```

- **Builder mode**: The `microgpt` CLI uses `microgpt_default_config()` and runtime struct reads. Acceptable for scaffolding, inspection, and light inference (e.g., single-token predictions).
- **Runner mode**: For training workloads, `microgpt train` invokes CMake to compile an optimised per-config binary (or uses a pre-compiled one). This is analogous to how `cargo build` compiles specialised release binaries.

> [!NOTE]
> Phase 1 can ship with runtime-only (Approach B) for simplicity. The ~2× perf hit is acceptable when N_EMBD ≤ 48. Tiered compilation can be added later for larger models.

---

## 4. Config File Format

INI-style, parsable with `strtok`/`sscanf` in ~150 lines of C. No external parser required.

### Single Organelle

```ini
[organelle:address_validator]
type       = single
tokenizer  = character
corpus     = addresses.txt
checkpoint = address_validator.ckpt

# Architecture
n_embd     = 48
n_head     = 4
n_layer    = 2
block_size = 128
mlp_dim    = 192

# Training
num_steps  = 5000
batch_size = 8
lr         = 0.001
threads    = auto
```

### Multi-Organelle Pipeline

```ini
[organelle:ttt_planner]
type       = pipeline_component
role       = planner
tokenizer  = character
corpus     = tictactoe_planner.txt
checkpoint = ttt_planner.ckpt
n_embd     = 48
n_layer    = 2
block_size = 128

[organelle:ttt_player]
type       = pipeline_component
role       = player
tokenizer  = character
corpus     = tictactoe_player.txt
checkpoint = ttt_player.ckpt
n_embd     = 48
n_layer    = 2
block_size = 128

[pipeline:ttt_game]
organelles = ttt_planner, ttt_player
judge      = deterministic
flow       = planner -> player -> judge -> repeat
replan_threshold = 3
test_games = 100
opponent   = random
```

### Scaling Up: Connect-4 (5 organelles)

```ini
[organelle:c4_planner]
type       = pipeline_component
role       = planner
corpus     = connect4_planner.txt
checkpoint = c4_planner.ckpt
n_embd     = 48
n_layer    = 2
block_size = 128

[organelle:c4_player]
type       = pipeline_component
role       = player
corpus     = connect4_player.txt
checkpoint = c4_player.ckpt
n_embd     = 48
n_layer    = 2
block_size = 128

[organelle:c4_evaluator]
type       = pipeline_component
role       = evaluator
corpus     = connect4_evaluator.txt
checkpoint = c4_evaluator.ckpt
n_embd     = 48
n_layer    = 2
block_size = 128

[organelle:c4_opening_book]
type       = pipeline_component
role       = opening
corpus     = connect4_openings.txt
checkpoint = c4_opening.ckpt
n_embd     = 32
n_layer    = 1
block_size = 64

[organelle:c4_endgame]
type       = pipeline_component
role       = endgame
corpus     = connect4_endgame.txt
checkpoint = c4_endgame.ckpt
n_embd     = 48
n_layer    = 2
block_size = 128

[pipeline:c4_game]
organelles = c4_planner, c4_player, c4_evaluator, c4_opening_book, c4_endgame
judge      = deterministic
flow       = opening -> planner -> player -> judge -> evaluator -> repeat
replan_threshold = 3
test_games = 200
opponent   = random
```

> [!NOTE]
> Each organelle can have a different architecture — the opening book uses a smaller model (N_EMBD=32, N_LAYER=1) since it memorises fixed patterns rather than reasoning about game state.

### Format Rules

- Sections: `[organelle:name]` or `[pipeline:name]`
- Keys: lowercase, alphanumeric + underscores
- Values: string or numeric, trimmed whitespace
- Comments: lines starting with `#` or `;`
- `auto` keyword: computed at runtime (e.g., `threads = auto` → `mgpt_default_threads()`)

### Config Validation

`parse_ini()` validates on load and exits with a clear message on any error:

| Rule | Check | Error Message |
|------|-------|---------------|
| Required keys | `corpus`, `n_embd`, `n_layer` must exist | `"ERROR: [organelle:X] missing required key 'corpus'"` |
| Corpus exists | `fopen(corpus, "r")` succeeds | `"ERROR: corpus file 'Y.txt' not found"` |
| Architecture sanity | `n_embd % n_head == 0` | `"ERROR: n_embd (48) must be divisible by n_head (5)"` |
| Positive values | `n_embd > 0`, `n_layer > 0`, `block_size > 0` | `"ERROR: n_embd must be positive"` |
| Pipeline refs | All names in `organelles=` have matching `[organelle:X]` | `"ERROR: pipeline 'Y' references unknown organelle 'Z'"` |
| Memory estimate | `count_params() * sizeof(scalar_t) < 25MB` per organelle | `"WARNING: model requires ~32MB — may exceed embedded RAM"` |

---

## 5. CLI Commands

| Command | Description | Example |
|---------|-------------|---------|
| `create` | Generate skeleton config + empty corpus | `microgpt create --name validator --type single` |
| `train` | Train organelle(s) from config | `microgpt train organelle.ini --threads 4 --resume` |
| `infer` | Run inference with a prompt | `microgpt infer organelle.ini --prompt "456 Oak|"` |
| `pipeline` | Execute a multi-organelle pipeline | `microgpt pipeline organelle.ini --games 100` |
| `info` | Inspect a saved `.bin` or `.ckpt` file | `microgpt info model.bin` |

### Argument Parsing

Use `getopt`-style parsing (custom, zero-dep). Subcommand is `argv[1]`, config file is `argv[2]`, flags follow.

---

## 6. Architecture

```
cli_main.c
├── parse_ini()          Config parser (~150 LoC)
├── cmd_create()         Scaffold generator
├── cmd_train()          Training orchestrator
│   ├── load corpus (load_docs / load_file)
│   ├── build vocab (build_vocab / build_word_vocab)
│   ├── model_create() with parsed config
│   ├── Spawn TrainWorker threads
│   └── checkpoint_save() on completion
├── cmd_infer()          Single-organelle inference
│   ├── model_load()
│   ├── forward_inference() loop
│   └── Confidence scoring (softmax entropy)
├── cmd_pipeline()       Multi-organelle orchestrator
│   ├── Load all organelle models
│   ├── Kanban state management
│   ├── organelle_generate() per role
│   └── Deterministic or neural judge
└── cmd_info()           Model inspector
```

### Dependencies

- Links against `microgpt_lib` (for runtime-config mode)
- Uses `microgpt_thread.h` for parallel training
- No additional source files beyond `cli_main.c`

### Code Sketch: Data Structures

```c
#define MAX_ORGANELLES 16
#define MAX_KEY_LEN    64
#define MAX_VAL_LEN    256
#define MAX_PATH_LEN   256

/* Per-organelle configuration parsed from [organelle:name] sections */
typedef struct {
  char name[MAX_KEY_LEN];           /* section name (e.g. "ttt_planner")  */
  char type[MAX_KEY_LEN];           /* "single" or "pipeline_component"   */
  char role[MAX_KEY_LEN];           /* "planner", "player", etc.          */
  char tokenizer[MAX_KEY_LEN];      /* "character" or "word"              */
  char corpus[MAX_PATH_LEN];        /* path to training corpus            */
  char checkpoint[MAX_PATH_LEN];    /* path to .ckpt file                 */

  /* Architecture — mirrors MicrogptConfig */
  int n_embd;
  int n_head;
  int n_layer;
  int block_size;
  int mlp_dim;

  /* Training */
  int    num_steps;
  int    batch_size;
  double lr;
  int    threads;                   /* 0 = auto-detect                    */
} OrgConfig;

/* Pipeline definition parsed from [pipeline:name] sections */
typedef struct {
  char name[MAX_KEY_LEN];
  char judge[MAX_KEY_LEN];          /* "deterministic" or "neural"        */
  int  replan_threshold;
  int  test_games;
  char opponent[MAX_KEY_LEN];       /* "random", "minimax", etc.          */

  /* Ordered list of organelle names in this pipeline */
  char org_names[MAX_ORGANELLES][MAX_KEY_LEN];
  int  num_orgs;
} PipeConfig;

/* Top-level config: all organelles + all pipelines in one file */
typedef struct {
  OrgConfig  orgs[MAX_ORGANELLES];
  int        num_orgs;
  PipeConfig pipes[4];              /* max 4 pipelines per file           */
  int        num_pipes;
} CliConfig;
```

### Code Sketch: INI Parser

```c
/*
 * parse_ini — Read an INI-style config file into a CliConfig struct.
 * Returns 0 on success, -1 on error (with message printed to stderr).
 */
static int parse_ini(const char *path, CliConfig *out) {
  memset(out, 0, sizeof(*out));

  FILE *f = fopen(path, "r");
  if (!f) { fprintf(stderr, "ERROR: cannot open '%s'\n", path); return -1; }

  char line[512];
  OrgConfig  *cur_org  = NULL;
  PipeConfig *cur_pipe = NULL;

  while (fgets(line, sizeof(line), f)) {
    /* Strip newline and leading whitespace */
    char *p = line;
    while (*p == ' ' || *p == '\t') p++;
    size_t len = strlen(p);
    if (len > 0 && p[len - 1] == '\n') p[--len] = '\0';

    /* Skip blank lines and comments */
    if (len == 0 || *p == '#' || *p == ';') continue;

    /* Section header: [organelle:name] or [pipeline:name] */
    if (*p == '[') {
      char kind[32], name[MAX_KEY_LEN];
      if (sscanf(p, "[%31[^:]:%63[^]]]" , kind, name) == 2) {
        if (strcmp(kind, "organelle") == 0 && out->num_orgs < MAX_ORGANELLES) {
          cur_org = &out->orgs[out->num_orgs++];
          cur_pipe = NULL;
          memset(cur_org, 0, sizeof(*cur_org));
          strncpy(cur_org->name, name, MAX_KEY_LEN - 1);
          /* Set defaults */
          cur_org->n_head = 4;
          cur_org->mlp_dim = 0; /* 0 = auto (4 × n_embd) */
        } else if (strcmp(kind, "pipeline") == 0 && out->num_pipes < 4) {
          cur_pipe = &out->pipes[out->num_pipes++];
          cur_org = NULL;
          memset(cur_pipe, 0, sizeof(*cur_pipe));
          strncpy(cur_pipe->name, name, MAX_KEY_LEN - 1);
        }
      }
      continue;
    }

    /* Key = value */
    char key[MAX_KEY_LEN], val[MAX_VAL_LEN];
    if (sscanf(p, "%63[^= ] = %255[^\n]", key, val) == 2) {
      /* Trim trailing whitespace from val */
      size_t vlen = strlen(val);
      while (vlen > 0 && (val[vlen-1]==' ' || val[vlen-1]=='\t')) val[--vlen]='\0';

      if (cur_org)  apply_org_key(cur_org, key, val);
      if (cur_pipe) apply_pipe_key(cur_pipe, key, val);
    }
  }
  fclose(f);

  /* Auto-compute mlp_dim where not set */
  for (int i = 0; i < out->num_orgs; i++)
    if (out->orgs[i].mlp_dim == 0)
      out->orgs[i].mlp_dim = out->orgs[i].n_embd * 4;

  return validate_config(out);
}
```

> [!TIP]
> The `apply_org_key()` / `apply_pipe_key()` helpers use a chain of `strcmp`/`sscanf` calls (~30 LoC each) to map key strings to struct fields. `validate_config()` implements the rules from §4 Config Validation.

---

## 7. Multi-Organelle Pipeline Runtime

Based on the proven pattern from `tictactoe/main.c`, `connect4/main.c`, and `puzzle8/main.c`:

```
┌──────────┐    prompt     ┌──────────┐    action    ┌──────────┐
│ Planner  │──────────────▶│  Player  │─────────────▶│  Judge   │
│(neural)  │               │(neural)  │              │(determ.) │
└──────────┘               └──────────┘              └────┬─────┘
      ▲                                                   │
      │                    ┌──────────┐                   │
      └────replan──────────│  Kanban  │◀──────────────────┘
                           │  State   │   result + blocked cells
                           └──────────┘
```

**Kanban State** tracks: board state, blocked positions, last moves, stall count. When stalls exceed `replan_threshold`, the Planner is re-invoked with updated context.

The pipeline orchestrator in `cli_main.c` generalises this loop:

1. Parse `[pipeline]` section → ordered list of organelles + judge type
2. Load each organelle's `.ckpt` or `.bin`
3. Initialise Kanban state
4. Loop: generate from each organelle in `flow` order, apply judge, update state

---

## 8. Implementation Phases

### Phase 1: Single Organelle (MVP) — Target: Q1 2026

- [ ] INI parser: `parse_ini()` with section/key/value extraction and validation
- [ ] `cmd_train`: corpus → vocab → model → training loop → checkpoint
- [ ] `cmd_infer`: load model → prompt → generate → confidence score
- [ ] `cmd_info`: read `.bin` header, print architecture summary
- [ ] `cmd_create`: generate skeleton `.ini` + empty corpus template
- [ ] CMake target: `add_executable(microgpt cli_main.c)` linking `microgpt_lib`
- [ ] **Validation gate:** convert `names_demo` and `shakespeare_demo` to `.ini` configs; **verify identical training loss and inference output** vs the standalone demos

**Estimated effort:** ~400–600 LoC in `cli_main.c`

### Phase 2: Multi-Organelle Pipelines — Target: Q2 2026

- [ ] `cmd_pipeline`: load multiple models, orchestrate Kanban loop
- [x] Generalised `organelle_generate()` — **implemented in `src/microgpt_organelle.c|h`**
- [x] `OpaKanban` state management — **implemented in shared library**
- [ ] Support for `judge = deterministic` (rules-based) and `judge = neural` (model-based)
- [ ] Pipeline metrics: win rate, avg moves, replan count
- [ ] **Validation gate:** convert `tictactoe_demo` and `connect4_demo` to `.ini` configs; **verify comparable win rates** vs standalone demos

**Estimated effort:** ~300–400 LoC added

### Phase 3: Tiered Compilation (Performance) — Target: Q3 2026

- [ ] `cmd_train --compile`: generate a temporary CMake target with `-DN_EMBD=X` flags
- [ ] Invoke `cmake --build` for a per-config optimised binary
- [ ] Cache compiled variants by config hash (e.g., `.microgpt/cache/<hash>/`)
- [ ] Fallback to runtime mode if CMake is unavailable
- [ ] **Validation gate:** benchmark CLI-compiled training vs standalone demo; **within 5% throughput**

**Estimated effort:** ~200 LoC + build system integration

### Benchmark Targets

| Metric | Phase 1 (runtime) | Phase 3 (compiled) | Standalone demo |
|--------|--------------------|--------------------|----------------|
| Names training (1K steps) | ≤ 0.20 s | ≤ 0.10 s | **0.09 s** |
| Shakespeare training (1K steps) | Acceptable | Match standalone | ~12 s |
| c_codegen training (1K steps) | Acceptable | Match standalone | — |
| Pipeline: TTT 100 games | ≤ 2× standalone | Match standalone | — |

---

## 9. Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| INI parser edge cases (quoting, escapes) | Config misreads | Keep format strict — no quoting, no escape sequences, validate on parse |
| Runtime config ~2× slower for large models | Training bottleneck | Phase 3 tiered compilation; Phase 1 only targets small models (N_EMBD ≤ 48) |
| `.ckpt` format not self-describing | Can't load checkpoints across configs | Add architecture header to `.ckpt` format (n_embd, n_layer, etc.) |
| Pipeline generalisation too rigid | Can't express novel flows | Start with linear `A → B → judge → repeat`; extend to DAGs later |
| Zero-dep constraint limits features | No rich TUI, no YAML | Acceptable — target audience is C developers and embedded engineers |

---

## 10. Relationship to Existing Docs

| Document | Scope | This doc extends |
|----------|-------|------------------|
| [VISION.md](VISION.md) | Stem cell philosophy — why composable micro-models | Provides the **tooling** to realise the vision |
| [ROADMAP.md](ROADMAP.md) | Organelle Toolkit (Q1) + Chaining Protocol (Q2) | Concrete implementation plan for both |
| [ORGANELLES.md](ORGANELLES.md) | Organelle concepts and patterns | CLI operationalises these patterns |
| [ORGANELLE_PIPELINE.md](ORGANELLE_PIPELINE.md) | Pipeline architecture detail | CLI's `cmd_pipeline` implements this |

---

*Copyright © 2026 Ajay Soni, Enjector Software Ltd. MIT License.*