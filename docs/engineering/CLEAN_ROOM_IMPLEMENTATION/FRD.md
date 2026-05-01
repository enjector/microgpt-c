# FRD — MicroGPT-C Functional Requirements Document

**Document ID:** MGC-FRD-001
**Version:** 1.0
**Status:** DRAFT
**Last updated:** 2026-04-30

---

## 1. Purpose and conventions

This document enumerates the functional surface of the MicroGPT-C platform. Each requirement carries a stable `REQ-<SUBSYSTEM>-NNN` ID and is the link between the strategic intent in `BRD.md` and the per-subsystem `BS_*.md` documents that pin down the contracts in RFC 2119 voice.

This document uses descriptive voice. Where a requirement reads like a contract, the corresponding contract lives in the relevant `BS_*.md`; that BS is the binding artefact.

### Subsystem keys

| Key | Subsystem |
|---|---|
| `CORE` | Transformer engine (forward, backward, Adam, KV cache) |
| `TOK` | Tokenisation (character + word) |
| `CKPT` | Checkpoint serialisation |
| `ORG` | Organelle pipeline (training, inference, ensemble, kanban) |
| `MSA` | Memory Sparse Attention |
| `QUANT` | TurboQuant / RotorQuant KV compression |
| `PIPE` | Pipeline IR (typed graph + verifier + DOT renderer + text round-trip) |
| `WIRE` | Wiring Organelle (NL → graph) and anchor retrieval |
| `GEO` | Geodesic state-space metrics |
| `VR` | Vietoris-Rips persistent cohomology |
| `EKAN` | EKAN B-spline edge activations |
| `VM` | Virtual machine compiler + runtime |
| `METAL` | Apple Metal GPU bridge |
| `BUILD` | Build system, feature flags, CMake variants |

## 2. Core transformer engine (`CORE`)

The core engine implements a decoder-only GPT-2-style transformer in C99.

| ID | Requirement |
|---|---|
| REQ-CORE-001 | The engine SHALL implement embedding, RMSNorm, multi-head causal self-attention, MLP (two-layer with ReLU), residual connections, output projection (`lm_head`) and softmax. |
| REQ-CORE-002 | Architecture dimensions (`N_EMBD`, `N_HEAD`, `N_LAYER`, `BLOCK_SIZE`, `MLP_DIM`) SHALL be compile-time constants. |
| REQ-CORE-003 | A `MicrogptConfig` struct SHALL mirror the compile-time constants for runtime introspection (banner printing, API convenience). |
| REQ-CORE-004 | A `microgpt_default_config()` helper SHALL return a config populated from the compile-time constants. |
| REQ-CORE-005 | A `microgpt_verify_config(cfg)` helper SHALL detect mismatches between a runtime config and the compile-time constants and return -1 on mismatch. |
| REQ-CORE-006 | Scalar precision SHALL be a compile-time toggle: `MICROGPT_USE_FLOAT` defines `scalar_t = float`; otherwise `scalar_t = double`. |
| REQ-CORE-007 | Optimiser hyperparameters (`BETA1`, `BETA2`, `EPS_ADAM`, `LEARNING_RATE`) SHALL remain `double` regardless of `scalar_t`. |
| REQ-CORE-008 | The forward+backward primitive `forward_backward_one(model, token_id, pos_id, target_id, keys, values, cache_len, grad_buffer)` SHALL run one position through the transformer, compute the cross-entropy loss against `target_id`, accumulate gradients into `grad_buffer`, and return the loss. |
| REQ-CORE-009 | The inference primitive `forward_inference(model, token_id, pos_id, keys, values, cache_len, logits_out)` SHALL produce raw next-token logits, with no gradient accumulation. |
| REQ-CORE-010 | The Adam optimiser primitive `adam_step(model, grads, m, v, step)` SHALL implement Adam with bias correction and a cosine learning-rate schedule with linear warmup. |
| REQ-CORE-011 | Adam SHALL apply decoupled weight decay (AdamW) to all matrices except token / position embeddings (`wte`, `wpe`) when `WEIGHT_DECAY > 0`. |
| REQ-CORE-012 | Optional global gradient clipping SHALL be supported via `clip_gradients` and the `GRAD_CLIP` macro (no-op when `GRAD_CLIP <= 0`). |
| REQ-CORE-013 | Optional label smoothing SHALL be supported via the `LABEL_SMOOTH` compile-time macro. |
| REQ-CORE-014 | A flat KV cache helper API (`kv_cache_alloc`, `kv_cache_free`, `kv_cache_reset`, `kv_cache_copy`) SHALL be provided. |
| REQ-CORE-015 | A demand-paged KV cache (`PagedKVCache`) SHALL be provided behind `MICROGPT_PAGED_KV`, with `KV_PAGE_SIZE` configurable (default 64). |
| REQ-CORE-016 | A model lifecycle API SHALL be provided: `model_create`, `model_free`, `model_num_params`, `model_save`, `model_load`. |
| REQ-CORE-017 | A weight-transfer helper `model_transfer_weights(src, dst, cfg)` SHALL copy `wpe`, attention `Q/K/V/O`, and MLP `fc1/fc2` from a source model to a destination model with the same architecture, leaving `wte` and `lm_head` to be freshly randomised. |
| REQ-CORE-018 | A model-soup helper `model_soup_average(dst, sources, n)` SHALL element-wise average `n` source models' weights into `dst`. |
| REQ-CORE-019 | A token sampler `sample_token(logits, vocab_size, temperature)` SHALL implement temperature-scaled softmax sampling. |
| REQ-CORE-020 | An RNG helper API SHALL be provided: `seed_rng(seed)` and `rand_u()` returning a uniform `scalar_t` in `[0, 1)`. |
| REQ-CORE-021 | The engine SHALL support optional INT8 weight quantisation via `QUANTIZATION_INT8` (or the British spelling `QUANTISATION_INT8`). |
| REQ-CORE-022 | Weight initialisation SHALL be Gaussian with standard deviation `INIT_STD` (default 0.08). |
| REQ-CORE-023 | The engine SHALL provide a multi-threaded training harness `TrainWorker` + `train_worker_run`, plus a portable thread API (`mgpt_thread_create`, `mgpt_thread_join`, `mgpt_cpu_count`, `mgpt_default_threads`). |
| REQ-CORE-024 | The engine SHALL provide optional Block Attention Residuals via `MICROGPT_ATTN_RES`. |
| REQ-CORE-025 | The engine SHALL provide optional Attention Sink (`MICROGPT_ATTN_SINK` with `ATTN_SINK_LOGIT`), Q/K RMSNorm pre-dot (`MICROGPT_QK_NORM`), and Partial RoPE (`MICROGPT_PARTIAL_ROPE` with `ROPE_DIMS`, `ROPE_BASE`) — the "DeepSeek-V4 port stack". All four flags SHALL default OFF and SHALL compose orthogonally. |
| REQ-CORE-026 | A `microgpt_print_config(name, cfg)` helper SHALL emit a human-readable configuration banner for demos to call at startup. |

## 3. Tokenisation (`TOK`)

| ID | Requirement |
|---|---|
| REQ-TOK-001 | The platform SHALL provide a character-level tokeniser: `Vocab` struct, `build_vocab(docs, vocab)`, `tokenize(doc, doc_len, vocab, ids, max_len)`. |
| REQ-TOK-002 | The character-level vocabulary SHALL be auto-discovered from the corpus and SHALL include a BOS token at the highest index. |
| REQ-TOK-003 | `tokenize` SHALL prepend BOS to its output and SHALL append a trailing BOS as an EOS sentinel when space allows. |
| REQ-TOK-004 | The platform SHALL provide a word-level tokeniser: `WordVocab` struct, `build_word_vocab(text, len, max_words, wv)`, `word_to_id(wv, word)`, `tokenize_words(text, len, wv, ids, max_tokens)`, `free_word_vocab(wv)`. |
| REQ-TOK-005 | The word-level vocabulary SHALL be frequency-ranked and SHALL keep the top `max_words` most common words; any out-of-vocabulary word SHALL map to `<unk>` (`unk_id`). |
| REQ-TOK-006 | The word-level tokeniser SHALL preserve newline structure with a dedicated `newline_id` token. |
| REQ-TOK-007 | The word-level tokeniser SHALL use an O(1) hash table for `word → id` lookup. |
| REQ-TOK-008 | A document loader `load_docs(path, docs, max_docs)` SHALL read a text file (≤ 50 MiB) into a `Docs` struct holding a flat data buffer plus per-line pointers and lengths. |
| REQ-TOK-009 | A `shuffle_docs(docs)` helper SHALL Fisher-Yates shuffle the document list in place. |
| REQ-TOK-010 | A generic `load_file(path, &len)` SHALL read an entire file into a heap buffer that is NUL-terminated for convenience. |

## 4. Checkpoint serialisation (`CKPT`)

| ID | Requirement |
|---|---|
| REQ-CKPT-001 | The platform SHALL provide a binary checkpoint format that captures model weights, Adam optimiser state (`m`, `v`), and the current training step. |
| REQ-CKPT-002 | The save / load API SHALL be `checkpoint_save(model, m, v, step, path)` and `checkpoint_load(path, vocab_size, cfg, m, v, &step_out)`. |
| REQ-CKPT-003 | Loading a checkpoint SHALL be sufficient to fully resume training without loss of momentum, learning-rate schedule, or step counter. |
| REQ-CKPT-004 | A weights-only API `model_save(model, path)` and `model_load(path, vocab_size, cfg)` SHALL also exist for inference-only deployments. |
| REQ-CKPT-005 | When `QUANTIZATION_INT8` is active, both checkpoint and weights-only paths SHALL be disabled (return -1 / NULL). |
| REQ-CKPT-006 | The byte-level format of checkpoint and weights files SHALL be specified in `FS_checkpoint.md`. |

## 5. Organelle pipeline (`ORG`)

The organelle module wraps a trained model with its tokeniser and training data, and provides scaffolding for multi-organelle coordination.

| ID | Requirement |
|---|---|
| REQ-ORG-001 | An `Organelle` struct SHALL bundle: a trained `Model *`, a character-level `Vocab`, a word-level `WordVocab`, a `Docs` corpus, and a `word_level` flag. |
| REQ-ORG-002 | `organelle_train(name, corpus_path, ckpt_path, cfg, num_steps)` SHALL train an organelle from a corpus, resuming from a checkpoint if one exists at `ckpt_path`. |
| REQ-ORG-003 | `organelle_train_words(...)` SHALL train a word-level organelle with a frequency-ranked vocabulary capped at `max_words`. |
| REQ-ORG-004 | `organelle_generate(org, cfg, prompt, output, max_len, temperature)` SHALL feed the prompt character-by-character into the model and sample up to `max_len` output characters, terminating on newline or BOS. |
| REQ-ORG-005 | `organelle_generate_multiline` SHALL behave like `organelle_generate` but terminate on a blank line ("`}\n\n`") for corpora whose responses span multiple lines. |
| REQ-ORG-006 | `organelle_generate_words` SHALL provide the word-level analogue. |
| REQ-ORG-007 | `organelle_generate_from_cache` SHALL allow autoregressive decoding to start from a pre-filled KV cache, avoiding redundant prompt processing. |
| REQ-ORG-008 | `organelle_generate_speculative(draft, target, ...)` SHALL implement SSD-style speculative decoding with a `spec_k` parameter; the draft and target organelles MUST share a vocabulary. |
| REQ-ORG-009 | `organelle_generate_ensemble(org, cfg, prompt, output, max_len, n_votes, base_temp, &confidence)` SHALL run `n_votes` inferences with temperature jitter and majority-vote the result, returning the agreement fraction in `confidence`. |
| REQ-ORG-010 | `organelle_generate_words_ensemble` SHALL provide the word-level analogue. |
| REQ-ORG-011 | An `OpaKanban` struct SHALL track blocked actions, recent action history, stall counts, and replans, with `opa_kanban_init`, `opa_kanban_add_blocked`, `opa_kanban_is_blocked`, `opa_kanban_clear_blocked`, `opa_kanban_add_last`. |
| REQ-ORG-012 | An `OpaCycleDetector` SHALL detect A↔B oscillation patterns over an `OPA_CYCLE_WINDOW` history (default 8) with `opa_cycle_init`, `opa_cycle_detected`, `opa_cycle_other`, `opa_cycle_record`. |
| REQ-ORG-013 | Pipe-string helpers `opa_extract_pipe_value(buf, key)` and `opa_pipe_starts_with(buf, prefix)` SHALL parse organelle wire strings of the form `key1=value1\|key2=value2\|...`. |
| REQ-ORG-014 | A multi-line corpus loader `opa_load_docs_multiline(path, docs, max_docs)` SHALL load corpora where each document is a prompt line + newline + response line, separated by blank lines. |
| REQ-ORG-015 | A valid-move filter `opa_valid_filter(action, valid_csv)` and a fallback selector `opa_valid_fallback(kb, valid_csv, fallback, sz)` SHALL be provided. |
| REQ-ORG-016 | An `OpaTrace` recorder SHALL capture each pipeline step's action, outcome (`OPA_STEP_*`), progress metric, kanban snapshot, and source (`from_model`), with `opa_trace_init`, `opa_trace_record`, `opa_trace_finalise`, `opa_trace_to_corpus`, `opa_trace_write`, `opa_trace_count`, `opa_trace_has_recovery`. |
| REQ-ORG-017 | A weight-transfer training helper `organelle_train_transfer(...)` SHALL train an organelle whose vocab-agnostic weights are copied from a source model. |
| REQ-ORG-018 | A model-soup training helper `organelle_train_soup(...)` SHALL train `n_seeds` models with different RNG seeds and average their weights. |
| REQ-ORG-019 | All pipe-string and Kanban formats SHALL be specified in `FS_organelle_wire.md`. |

## 6. Memory Sparse Attention (`MSA`)

| ID | Requirement |
|---|---|
| REQ-MSA-001 | The platform SHALL provide an `MsaPool` arena holding compressed KV chunks with `msa_pool_create(capacity, n_layer, n_embd)` and `msa_pool_free`. |
| REQ-MSA-002 | `msa_pool_chunk(pool, active_keys, active_values, chunk_len)` SHALL pool a chunk of recent active KV vectors into a single latent entry via mean (or weighted) pooling. |
| REQ-MSA-003 | A pool-weighting policy SHALL be selectable at compile time via `MSA_POOL_MODE`: `0` uniform mean (default), `1` linear ramp, `2` exponential recency, `3` content-aware (softmax of cosine to anchor). |
| REQ-MSA-004 | `msa_route_top_1(pool, query_keys)` SHALL return the index of the single best chunk by cross-layer cosine similarity, or `-1` if the pool is empty. |
| REQ-MSA-005 | `msa_route_top_k(pool, query_keys, k, indices_out, scores_out)` SHALL return the top-K chunks by ReLU-summed cross-layer scoring (Lightning Indexer style), in descending score order with stable tie-break. |
| REQ-MSA-006 | `msa_expand_context(pool, chunk_idx, active_keys, active_values, pos)` SHALL inject a pooled chunk back into the active KV cache at position `pos`. |
| REQ-MSA-007 | A `MsaRecency` ring buffer SHALL be provided to retain the last `n_win` uncompressed K/V tokens across chunking events, with `msa_recency_create`, `msa_recency_free`, `msa_recency_reset`, `msa_recency_push`, `msa_recency_inject`. |
| REQ-MSA-008 | When `MICROGPT_PARTIAL_ROPE` is enabled, the platform SHALL provide RoPE-aware pool/inject variants `msa_pool_chunk_rope`, `msa_expand_context_rope`, `msa_recency_inject_rope` that store pool entries in position-zero space and re-rotate at injection time. |

## 7. KV compression — TurboQuant and RotorQuant (`QUANT`)

| ID | Requirement |
|---|---|
| REQ-QUANT-001 | The platform SHALL provide a TurboQuant 4-bit dual-state quantiser combining MSE codebook indices (≤ b-1 bits) and a 1-bit QJL signature (sign of a random projection) plus a residual norm. |
| REQ-QUANT-002 | The TurboQuant API SHALL be `turboquant_init`, `turboquant_free`, `turboquant_quant_prod`, `turboquant_dequant_prod`, plus pure-MSE variants `turboquant_quant_mse` / `turboquant_dequant_mse`. |
| REQ-QUANT-003 | The platform SHALL provide a RotorQuant variant with the same API surface and a mode parameter (`RQ_MODE_PLANAR` 2D Givens, `RQ_MODE_ISO` 4D quaternion). |
| REQ-QUANT-004 | When `ENABLE_TURBOQUANT` or `ENABLE_ROTORQUANT` is set, `MsaPool` SHALL store quantised KV state instead of raw `scalar_t`. |
| REQ-QUANT-005 | The dequantisation primitives SHALL be inner-product-optimal and unbiased, suitable for cosine routing in MSA without accuracy regression on the demos that integrate them. |

## 8. Pipeline IR (`PIPE`)

| ID | Requirement |
|---|---|
| REQ-PIPE-001 | The platform SHALL provide a typed-graph IR (`Pipeline`, `PipelineNode`, `PipelineEdge`, `PipelineType`, `PipelinePort`, `PipelineConfig`, `PipelineValue`). |
| REQ-PIPE-002 | The type system SHALL include `VOID`, `INT`, `FLOAT`, `STRING`, `LIST`, `TENSOR`, `RECORD`, `ANY`, with structural equality and ANY as a polymorphic wildcard. |
| REQ-PIPE-003 | Construction primitives SHALL be: `pipeline_create`, `pipeline_free`, `pipeline_add_node`, `pipeline_add_subgraph`, `pipeline_connect`, `pipeline_set_signature`, `pipeline_connect_signature_in`, `pipeline_connect_signature_out`, `pipeline_node_set_config_{int,float,string}`. |
| REQ-PIPE-004 | A verifier `pipeline_verify(p)` SHALL run, in order: unique-id check, edge endpoint validity, dangling-port detection, signature-input/output coverage, type compatibility, cycle detection, topological sort. |
| REQ-PIPE-005 | A partial verifier `pipeline_verify_partial(p, &missing)` SHALL accept incomplete graphs (dangling inputs, unconnected sig outputs) but reject hard errors (duplicate ids, type mismatches, cycles), returning the count of "missing" elements. |
| REQ-PIPE-006 | A graph-repair pass `pipeline_repair(p, &report)` SHALL drop fragments of an internally inconsistent graph (dead nodes whose inputs cannot be satisfied) so the residual subgraph can verify cleanly. Repair SHALL only subtract; it MUST NOT add nodes, edges, or ports. |
| REQ-PIPE-007 | An execution primitive `pipeline_execute(p, inputs, outputs, dispatch, user_data)` SHALL walk the cached topological order, calling a host-provided `PipelineDispatchFn` for each leaf primitive. |
| REQ-PIPE-008 | A VM-backed dispatch primitive `pipeline_execute_vm(p, vm, inputs, outputs)` SHALL resolve each leaf node's primitive name via `vm_engine_find_fn` (REQ-VM-007), marshal `PipelineValue ↔ double[]`, and dispatch the resolved `vm_native_fn`. INT/FLOAT/VOID ports only; non-numeric ports return `PIPE_ERR_EXEC` with a message identifying the offending node and port. Subgraph nodes recurse. The implementation lives in the opt-in TU `src/microgpt_pipeline_vm.c`; `microgpt_lib.a` itself does NOT link the VM. Demos / tests calling this function MUST link `microgpt_pipeline_vm.c` AND the VM library. (V1.0.4 RESOLVES the prior `GAP-PIPE-003` stub.) |
| REQ-PIPE-009 | A text serialiser `pipeline_render_text(p)` SHALL emit a deterministic canonical text form (nodes in topological order). |
| REQ-PIPE-010 | A strict text parser `pipeline_parse_text(src)` SHALL be the inverse of the serialiser for verified graphs (round-trip identity). |
| REQ-PIPE-011 | A tolerant text parser `pipeline_parse_text_tolerant(src)` SHALL apply three named repairs (dedup signature inputs, auto-promote referenced sig inputs, auto-promote referenced sig outputs). |
| REQ-PIPE-012 | A DOT-format renderer `pipeline_render_dot(p)` SHALL emit a GraphViz visualisation. |
| REQ-PIPE-013 | An error retrieval helper `pipeline_last_error()` SHALL return the most recent thread-local human-readable error message. |
| REQ-PIPE-014 | The text format SHALL be specified in `FS_pipeline_ir_text.md`. |
| REQ-PIPE-015 | Errors SHALL be reported as the negative codes `PIPE_ERR_DUP_NODE_ID`, `PIPE_ERR_UNKNOWN_NODE`, `PIPE_ERR_UNKNOWN_PORT`, `PIPE_ERR_DANGLING_PORT`, `PIPE_ERR_TYPE_MISMATCH`, `PIPE_ERR_CYCLE`, `PIPE_ERR_BAD_SIGNATURE`, `PIPE_ERR_OOM`, `PIPE_ERR_PARSE`, `PIPE_ERR_EXEC`. |

## 9. Wiring Organelle (`WIRE`)

The wiring organelle is a 540K-parameter word-level transformer that emits pipeline-IR text for natural-language prompts, augmented by a manifold-retrieval anchor table.

| ID | Requirement |
|---|---|
| REQ-WIRE-001 | The platform SHALL provide a wiring organelle demo that emits @graph text from natural-language prompts. |
| REQ-WIRE-002 | A best-of-N + verify-as-Judge loop SHALL re-rank candidate graphs by `pipeline_verify` success and by domain heuristics. |
| REQ-WIRE-003 | An anchor library SHALL ship a curated set of canonical @graph entries indexed by domain keywords. |
| REQ-WIRE-004 | A Geodesic-based anchor classifier SHALL embed prompts into a 20-dimensional state space and select the top-1 anchor by Geodesic top-1 prediction (Phase 2c mechanism). |
| REQ-WIRE-005 | A TF-IDF centroid classifier SHALL be available as an alternative to the Geodesic classifier; on the expanded Phase 4 corpus this classifier SHALL achieve ≥ 90% on the adversarial axis-2 stress test. |
| REQ-WIRE-006 | A composition fallback `wiring_compose_for_prompt(...)` SHALL chain top-2/3 fragments by output→input linkage when no single anchor matches a multi-stage prompt. |
| REQ-WIRE-007 | CLI flags `--no-anchor`, `--clean-only`, `--composition`, `--no-composition` SHALL reproduce the leakage-audited honest baselines documented in `RESEARCH_PIPELINE_IR.md`. |
| REQ-WIRE-008 | Native primitives SHALL be registered via `wiring_natives.{h,c}` (≥ 40 primitives) for the dispatch path. |
| REQ-WIRE-009 | Reference answers for the held-out NL prompts SHALL be defined in `wiring_references.{h,c}`. |
| REQ-WIRE-010 | A pre-registered Phase 4 corpus expander `tools/corpus_expand` SHALL deterministically generate ≥ 4,000 prompts from per-family synonym tables and sentence templates. |

## 10. Geodesic state-space metrics (`GEO`)

| ID | Requirement |
|---|---|
| REQ-GEO-001 | The platform SHALL provide a Riemannian geodesic solver fixed at `GEO_DIMS` dimensions (default 40 in this fork; previously 12). |
| REQ-GEO-002 | The solver SHALL implement RK4 geodesic integration with metric identity caching and Cholesky decomposition for SPD metric inversion. |
| REQ-GEO-003 | A gauge-field hook SHALL allow coercion / romance-scam detection metrics to introduce additional path work. |
| REQ-GEO-004 | Built-in metrics SHALL include `geo_metric_flat`, `geo_metric_diagonal`, `geo_metric_behavioral`, `geo_metric_fraud`. |
| REQ-GEO-005 | The API SHALL be `geo_solver_init`, `geo_compute_tension`, `geo_compute_euclidean`, plus the public matrix utilities `geo_dot`, `geo_norm`, `geo_norm_sq`, `geo_mat_vec`, `geo_quadratic_form`, `geo_identity`, `geo_is_identity`, `geo_invert_matrix`, `geo_christoffel`. |

## 11. Vietoris-Rips persistent cohomology (`VR`)

| ID | Requirement |
|---|---|
| REQ-VR-001 | The platform SHALL provide a Vietoris-Rips persistent cohomology engine with bounded buffers (`VR_MAX_DIMS=12`, `VR_MAX_PTS=64`, `VR_MAX_INTERVALS=512`). |
| REQ-VR-002 | The engine SHALL implement an L2 distance matrix, flag-complex filtration with bitmask clique expansion, F₂ persistent cohomology (apparent pairs + clearing), persistence diagrams, and Betti numbers β₀, β₁, β₂. |
| REQ-VR-003 | The API SHALL be `vr_engine_init`, `vr_compute`, `vr_betti_numbers`, `vr_betti_at`, `vr_make_point`. |

## 12. EKAN B-spline activations (`EKAN`)

| ID | Requirement |
|---|---|
| REQ-EKAN-001 | The platform SHALL provide a fixed-point cubic B-spline edge activation primitive with `MAX_EKAN_EDGES = 128` and `MAX_SPLINE_GRID_SIZE = 64`. |
| REQ-EKAN-002 | The fixed-point scale SHALL be `BONSAI_FP_SCALE = 1,000,000` (64-bit-safe). |
| REQ-EKAN-003 | Core primitives `fp_mul`, `fp_div` (with zero-division protection), `ekan_find_knot_span_fp` (binary search), `ekan_bspline_basis_fp`, `ekan_edge_pulse` SHALL be force-inlined. |

## 13. Virtual Machine (`VM`)

| ID | Requirement |
|---|---|
| REQ-VM-001 | The platform SHALL provide a stack-based virtual machine with a TypeScript-flavoured surface language compiled through Flex/Bison to bytecode. |
| REQ-VM-002 | The compilation pipeline SHALL be: lexer (Flex) → parser (Bison ≥ 3.0) → AST → bytecode → 6-pass verifier → runtime. |
| REQ-VM-003 | Pre-generated parser sources (`microgpt_vm_parser.l.c`, `microgpt_vm_parser.tab.c`, `microgpt_vm_parser.tab.h`) SHALL be committed so a build that lacks Flex/Bison ≥ 3.0 still succeeds. |
| REQ-VM-004 | The VM SHALL expose a public C API (`vm_engine`, `vm_module`, `vm_function`, `vm_compiler`, `vm_module_runtime`, `vm_module_generator`, `vm_eval`, `vm_runtime`) with utility containers (`vm_list`, `vm_map`, `vm_queue`, `vm_string_buffer`). |
| REQ-VM-005 | The VM SHALL allow registration of native C functions reachable from the surface language (`vm_engine_register_fn`). |
| REQ-VM-006 | The bytecode format SHALL be specified in `FS_vm_bytecode.md`. |
| REQ-VM-007 | The VM engine SHALL expose a public `vm_native_fn vm_engine_find_fn(const vm_engine *e, const char *name)` lookup that returns the previously-registered function pointer (or NULL). Lookup is O(n) over the registration table (typically ≤ 50 entries) and not on the inner-loop hot path. Used by `pipeline_execute_vm` to resolve graph-node primitives at execute time. |
| REQ-WIRE-011 | The wiring layer SHALL ship a primitive manifest (`demos/wiring_organelle/wiring_primitive_manifest.{h,c}`) listing every native primitive with its typed input/output signature plus a NL keyword set. Public API: `wiring_primitive_manifest(out_count)` and `wiring_primitive_find(name)`. |
| REQ-WIRE-012 | The wiring layer SHALL ship a deterministic type-directed compositional search (`demos/wiring_organelle/wiring_compositional_search.{h,c}`) that synthesises a verified `Pipeline *` from an NL prompt and the primitive manifest. The search is greedy, beam-width 1 in V1.0.4. Public API: `wiring_compositional_search(prompt, &report)` and `wiring_compositional_search_render(prompt, &pipeline_out, &report)`. The returned graph SHALL pass `pipeline_verify`. |

## 14. Metal GPU bridge (`METAL`)

| ID | Requirement |
|---|---|
| REQ-METAL-001 | When `MICROGPT_METAL` is enabled on macOS, the platform SHALL provide GPU-accelerated dense linear forward/backward primitives (`metal_lin_fwd`, `metal_lin_bwd`) via Apple Metal compute shaders. |
| REQ-METAL-002 | The Metal bridge SHALL convert between `double` and `float` at the CPU/GPU boundary; Metal natively supports float32 only. |
| REQ-METAL-003 | The platform SHALL expose `metal_init`, `metal_cleanup`, `metal_available` for capability detection. |
| REQ-METAL-004 | When Metal is unavailable or fails to initialise, the engine SHALL fall back to CPU code paths transparently. |

## 15. Build system (`BUILD`)

| ID | Requirement |
|---|---|
| REQ-BUILD-001 | The build system SHALL be CMake ≥ 3.10. |
| REQ-BUILD-002 | A `add_demo()` CMake helper SHALL register demo executables, copy data files post-build, link threading and accelerator libraries on demand, and select the correct compile-flag-variant of the library. |
| REQ-BUILD-003 | A `_microgpt_lib_for_defines()` helper SHALL cache library targets by their sorted compile-define set, MD5-hashed, so that two demos with identical architecture flags share one library. |
| REQ-BUILD-004 | Feature flags SHALL be available as CMake options: `MICROGPT_SIMD`, `MICROGPT_USE_FLOAT`, `MICROGPT_HEAD_PARALLEL`, `MICROGPT_PAGED_KV`, `QUANTIZATION_INT8`, `MICROGPT_BLAS`, `MICROGPT_METAL`, `MICROGPT_ATTN_RES`, `MICROGPT_ATTN_SINK`, `MICROGPT_QK_NORM`, `MICROGPT_PARTIAL_ROPE`, `ENABLE_TURBOQUANT`, `ENABLE_ROTORQUANT`, plus `MSA_POOL_MODE` and `ROPE_DIMS` / `ROPE_BASE` / `ATTN_SINK_LOGIT` magnitude knobs. |
| REQ-BUILD-005 | Demos SHALL run from `build/` (Linux/macOS) or `build/Release/` (Windows); each demo's data files are copied next to its binary by `add_demo()` POST_BUILD steps. |
| REQ-BUILD-006 | Bootstrap scripts (`bootstrap.sh`, `bootstrap.bat`) SHALL produce a working build with default flags. |
| REQ-BUILD-007 | A CI configuration SHALL build and run `test_microgpt` + `bench_microgpt` on Ubuntu (gcc, clang), macOS (clang), Windows (cl). |

## 16. Demos (informative)

The following demos are *evidence* the platform satisfies BREQ-010 through BREQ-019 and SHOULD remain runnable. They are not normative requirements in themselves but their continued operation is the "smoke test" that the corpus's contracts have not silently regressed.

| Demo | Evidence for |
|---|---|
| `c_names_demo` | Tiny model, < 1 s training, sub-200 KB checkpoint |
| `c_shakespeare_demo`, `w_shakespeare_demo` | BREQ-010, BREQ-011 |
| `c_puzzle8_demo` (and reasoning variants) | BREQ-012 |
| `c_connect4_demo` | BREQ-013 |
| `c_lottery_demo` | BREQ-014 (negative control) |
| `wiring_organelle_demo` | BREQ-015, BREQ-016 |
| `manifold_tfidf_demo` | BREQ-017 |
| `msa_infinite_shakespeare`, `msa_infinite_shakespeare_v4`, `msa_*` | BREQ-018 |
| `tq_*`, `rq_*` | BREQ-019 |
| `c_vm_codegen`, `w_vm_codegen_*`, `c_vm_compose` | VM exercise |
| `c_tictactoe_demo`, `c_othello_demo`, `c_mastermind_demo`, `c_sudoku_demo`, `c_klotski_demo`, `c_reddonkey_demo`, `c_lightsout_demo`, `c_hex_demo`, `c_pentago_demo` | Game leaderboard (`README.md` table) |
| `c_tictactoe_transfer_othello`, `c_tictactoe_transfer_planner_to_player` | Weight transfer (REQ-CORE-017) |

## 17. Cross-references

- `BRD.md` for business rationale.
- `NFRD.md` for performance and portability targets.
- `BS_*.md` for the per-subsystem RFC 2119 contracts.
- `FS_checkpoint.md`, `FS_pipeline_ir_text.md`, `FS_organelle_wire.md`, `FS_vm_bytecode.md` for byte-level formats.
- `TRACEABILITY.md` for ID-to-source linkage.

## 18. Revision history

| Version | Date | Change |
|---|---|---|
| 1.0 | 2026-04-30 | Initial extraction. |
