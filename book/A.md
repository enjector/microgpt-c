# Glossary and References {-}

This appendix provides a comprehensive glossary of key terms used throughout the book, along with a curated list of references. The glossary defines concepts in simple, accessible language, drawing from the explanations in the chapters. Terms are listed alphabetically for easy reference. The references include foundational papers and resources that influenced the principles of MicroGPT-C, such as transformer architectures and optimization techniques. These are cited in a standard format (APA style) for further reading. Note that while the book focuses on practical implementation, these sources offer deeper theoretical insights.

## Glossary

- **Adam Optimizer**: An adaptive optimization algorithm used in training that adjusts learning rates for each parameter based on historical gradients. It incorporates momentum and variance scaling to improve convergence, especially on noisy data (see Chapter 3).

- **AdamW (Decoupled Weight Decay)**: A variant of Adam that separates the weight decay term from the gradient update, ensuring all parameters are regularised equally. Prevents embedding table bloat during early training. Formula: $\theta_t = \theta_{t-1} - \alpha m_t / (\sqrt{v_t} + \epsilon) - \alpha \lambda \theta_{t-1}$ (see Chapter 3).

- **Anomaly Detection**: The process of identifying unusual patterns in data, such as spikes in sensor readings, using models to flag deviations from normal behavior (see Chapter 11).

- **Attention Mechanism**: A core component of transformers that allows the model to weigh the importance of different parts of the input data, focusing on relevant elements while ignoring others (see Chapters 2 and 8).

- **Batch**: A group of training examples processed together in one iteration to stabilize updates and improve efficiency (see Chapter 3).

- **Bias (in Data)**: Systematic favoritism in training data toward certain groups or outcomes, leading to unfair model predictions; mitigated through curation and balancing (see Chapter 12).

- **Block Size**: The maximum length of input sequences a model can handle, determining context capacity (see Chapter 2).

- **Catastrophic Forgetting**: The tendency of a model to lose previously learned knowledge when trained on new data; addressed with replay buffers (see Chapters 3 and 12).

- **Character-Level Tokenization**: Breaking input into individual characters or bytes, ideal for short, structured data with no unknown tokens (see Chapter 2).

- **Command-Line Interface (CLI)**: A text-based tool for executing commands like training or inference, simplifying workflows without full scripting (see Chapter 9).

- **Confidence Gating**: Rejecting model outputs below a probability threshold to avoid overconfident errors (see Chapters 4, 10, and 12).

- **Corpus**: A collection of training data examples, such as text pairs or simulations, used to differentiate organelles (see Chapter 4).

- **Cosine Decay**: A learning rate schedule that smoothly reduces the learning rate following a cosine curve after warmup, preventing overfitting in later training (see Chapter 3).

- **Coordination Funnel**: The empirical pattern where pipeline coordination converts weak individual model outputs (~50% invalid) into high system-level success rates (85-90%). Validated across 14 experiments (see Chapters 5 and 6).

- **c99_compose**: The code composition experiment using a Planner→Judge pipeline (1.2M params each with LR scheduling) to generate function composition plans, achieving 83% exact match (see Chapter 10).

- **Cross-Entropy Loss**: A measure of prediction error in generative models, penalizing low probabilities for correct targets (see Chapter 3).

- **Cycle Detection**: Identifying and breaking repetitive loops in pipelines, such as oscillating moves, using history windows (see Chapter 5).

- **Differentiation (of Organelles)**: The process of training a generic stem cell model on a specific corpus to create a specialist (see Chapter 4).

- **Drift Detection**: Monitoring for changes in data distribution that degrade model performance, triggering retraining (see Chapters 11 and 12).

- **Edge AI**: Running AI directly on peripheral devices (e.g., sensors) rather than central servers, emphasizing low latency and privacy (see Chapter 11).

- **Embeddings**: Vector representations of tokens that capture semantic meaning in a fixed-dimensional space (see Chapter 2).

- **Ensemble Voting**: Running multiple inferences with slight variations and selecting the majority output for improved reliability (see Chapter 4).

- **Epoch**: A complete pass through the entire training dataset during optimization (see Chapter 3).

- **Federated Differentiation**: Collaborative training across devices where only model updates (not raw data) are shared for privacy (see Chapters 11 and 13).

- **Feed-Forward Network**: A simple neural layer in transformers that processes features after attention, using ReLU activation (see Chapter 2).

- **Flat-String Protocol**: A simple, pipe-delimited format for inter-organelle communication, reducing complexity over nested structures (see Chapter 10).

- **Gradient Clipping**: Max-norm scaling of gradients before the optimizer step, preventing exploding gradients in deep models. Formula: if $\|g\| > g_{\max}$, scale $g$ by $g_{\max} / \|g\|$ (see Chapter 3).

- **Gradient Descent**: The core method for updating model parameters by following the direction of steepest error reduction (see Chapter 3).

- **Grouped Query Attention (GQA)**: An efficient attention variant that shares keys and values across query head groups, reducing memory (see Chapter 8).

- **Hidden Markov Model (HMM)**: A statistical model with hidden states, transition probabilities, and emission distributions. Useful for sequential data where observed outputs depend on unobserved states. Trained via Baum-Welch (EM) algorithm (see Chapter 6).

- **Hurst Exponent**: A measure of long-range dependence in time series. $H = 0.5$ indicates a random walk; $H > 0.5$ indicates trending; $H < 0.5$ indicates mean-reverting. Used to characterise the predictability of sequential data (see Chapter 6).

- **Inference**: The phase where a trained model generates outputs from new inputs, without updating parameters (see Chapter 3).

- **Internet of Things (IoT)**: A network of connected devices that collect and exchange data, enhanced by edge AI for local processing (see Chapter 11).

- **Judge (in Pipelines)**: A deterministic component that validates outputs, such as checking move legality or syntax (see Chapter 5).

- **Kanban Architecture**: A coordination system using shared state (todo, blocked, history) to manage pipeline workflows and handle failures (see Chapter 5).

- **KV Cache**: Stored keys and values from past attention computations, speeding up sequential inference (see Chapters 3 and 8).

- **Label Smoothing**: A regularisation technique that replaces hard targets with a soft mixture: $y_{\text{smooth}} = (1-\alpha) \cdot y_{\text{hard}} + \alpha/V$. Calibrates model confidence and raises the loss floor (see Chapter 3).

- **Learning Rate Scheduling**: Gradually adjusting the step size in optimization, often with warmup and decay for stability (see Chapter 3).

- **Low-Rank Adaptation (LoRA)**: A parameter-efficient fine-tuning technique that freezes main weights and trains small rank-decomposition matrices, significantly reducing memory footprints (see Chapter 13).

- **LR-Capacity Scaling**: The empirical rule that larger models require lower learning rates: lr $\propto$ 1/$\sqrt{\text{params}}$. At 460K params lr=0.001 works; at 1.2M params lr=0.0005 is needed to prevent divergence (see Chapters 3 and 4).

- **Manhattan Distance (MD-delta)**: A board-state encoding for sliding puzzles: $MD = \Sigma |\text{row}_i - \text{goal\_row}_i| + |\text{col}_i - \text{goal\_col}_i|$. The MD-delta variant encodes per-move changes, reducing the problem to greedy selection (see Chapter 6).

- **Model Soup**: Weight averaging of multiple training runs: $\theta_{\text{soup}} = (1/N) \Sigma \theta_i$. Finds flatter loss landscape basins for better generalisation (see Chapter 7).

- **Multi-Head Attention (MHA)**: Parallel attention computations where each head learns different relationships (see Chapter 8).

- **Multi-Query Attention (MQA)**: An extreme efficiency variant sharing one set of keys/values across all queries (see Chapter 8).

- **Organelle**: A small, specialized AI model differentiated from a stem cell base for focused tasks (see Chapter 4).

- **Organelle Pipeline Architecture (OPA)**: A framework for coordinating multiple organelles via planners, workers, and judges (see Chapter 5).

- **Overconfidence**: When a model assigns high probability to incorrect outputs; mitigated by ensembles and gating (see Chapter 12).

- **Overfitting**: When a model memorizes training data but fails on new inputs; detected by comparing train/test losses (see Chapter 3).

- **Paged KV Cache**: A memory-efficient cache that allocates in fixed pages, handling long sequences without fragmentation (see Chapter 7).

- **Paraphrase Blindness**: Model failure on reworded inputs due to literal matching; addressed by decomposition (see Chapter 10).

- **Permutation Test**: A validation method that shuffles training labels to test whether a model's accuracy exceeds chance. Stratified variants isolate signal in specific subsets (see Chapter 6).

- **Persistence Baseline**: The naive prediction baseline "predict same class as today." For sequential prediction tasks, persistence is often a surprisingly strong baseline — the true bar to beat (see Chapter 6).

- **Planner (in Pipelines)**: An organelle that decomposes problems into task lists (see Chapter 5).

- **Quantization**: Reducing parameter precision (e.g., to INT8) for smaller models and faster inference (see Chapter 7).

- **Replay Buffer**: A storage of past examples mixed into new training to prevent forgetting (see Chapters 3 and 12).

- **Reproducibility**: Ensuring the same results across runs via seeding random generators (see Chapter 3).

- **Retrieval-Based Intelligence**: Model behavior focused on reproducing trained patterns rather than true generation (see Chapter 4).

- **RMSNorm**: A normalization technique that stabilizes activations by dividing by root-mean-square (see Chapter 2).

- **Self-Monitoring**: Organelles that track their own confidence and trigger retraining on drift (see Chapter 13).

- **Sliding Window Attention (SWA)**: Limiting attention to recent tokens for efficiency on long sequences (see Chapter 8).

- **Softmax**: A function that converts raw scores into probabilities summing to 1 (see Chapter 2).

- **Stem Cell Philosophy**: The idea of starting with generic models that differentiate into specialists (see Chapter 4).

- **Structured Outputs**: Generating data in fixed formats like JSON, validated by judges (see Chapter 10).

- **Temperature (in Sampling)**: A parameter controlling randomness in output generation—low for deterministic, high for creative (see Chapter 3).

- **Tokenization**: Converting raw input into numerical tokens for model processing (see Chapter 2).

- **Training Loop**: The iterative process of forward passes, loss computation, and backward updates (see Chapter 3).

- **Transformer Block**: The repeating unit in models, combining attention and feed-forward layers with residuals (see Chapter 2).

- **Vectorization (SIMD)**: CPU technique for parallel data processing, speeding up operations like matrix multiplies (see Chapter 7).

- **Word-Level Tokenization**: Breaking input into words, suitable for semantic-rich text (see Chapter 2).

- **Worker (in Pipelines)**: An organelle that executes specific tasks, like suggesting a move (see Chapter 5).

- **Warmup Ratio**: The fraction of total training steps spent ramping the learning rate from zero to peak. Typical values: 3-5% of total steps. Larger models require longer warmup (see Chapter 3).

## References

Ainslie, J., et al. (2023). *GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints*. arXiv preprint arXiv:2305.13245.

Beltagy, I., Peters, M. E., & Cohan, A. (2020). *Longformer: The Long-Document Transformer*. arXiv preprint arXiv:2004.05150.

DeepSeek-AI. (2024). *DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model*. Technical report.

Jiang, A. Q., et al. (2023). *Mistral 7B*. arXiv preprint arXiv:2310.06825.

Shazeer, N. (2019). *Fast Transformer Decoding: One Write-Head is All You Need*. arXiv preprint arXiv:1911.02150.

Vaswani, A., et al. (2017). *Attention Is All You Need*. In Advances in Neural Information Processing Systems (NeurIPS).

Zaheer, M., et al. (2020). *Big Bird: Transformers for Longer Sequences*. In Advances in Neural Information Processing Systems (NeurIPS).

These references represent seminal works on transformers and efficiency improvements. For implementation details, refer to the MicroGPT-C source code and documentation, which adapts these ideas for small-scale, C99-based systems. Further reading is encouraged for those interested in theoretical foundations.

Loshchilov, I., & Hutter, F. (2019). *Decoupled Weight Decay Regularization*. In International Conference on Learning Representations (ICLR).

Szegedy, C., Vanhoucke, V., Ioffe, S., Shlens, J., & Wojna, Z. (2016). *Rethinking the Inception Architecture for Computer Vision*. In CVPR. (Introduces label smoothing.)

Velickovic, P., et al. (2022). *The CLRS Algorithmic Reasoning Benchmark*. In Proceedings of ICML.

Wortsman, M., et al. (2022). *Model Soups: Averaging Weights of Multiple Fine-tuned Models Improves Accuracy without Increasing Inference Time*. In Proceedings of ICML.

Baum, L. E., et al. (1970). *A Maximization Technique Occurring in the Statistical Analysis of Probabilistic Functions of Markov Chains*. Annals of Mathematical Statistics. (Baum-Welch / EM for HMMs.)

## Appendix B: Full Code Listings

To maintain the flow of the text, many chapters only present truncated snippets. The complete, compilable source code for all examples is available in the repository:

- **Core Engine:** `src/microgpt.c`
- **Organelle API & Kanban:** `src/microgpt_organelle.c`
- **CPU Parallelism (OpenMP):** `examples/names/main.c`
- **Metal GPU Offloading:** `src/microgpt_metal.m`

## Appendix C: Benchmarks and Experimental Data

Performance claims and game win-rates are backed by reproducible benchmark scripts. 

- **Game Organelle Experiments:** Explore the READMEs inside `demos/character-level/` (e.g., `8puzzle`, `mastermind`, `pentago`).
- **Autonomous Codegen (c99_compose):** Test the 1.2M parameter planner/judge pipeline in `experiments/c99_compose/`.

## Appendix D: Datasets and Corpora

The "stem cell" organelles were differentiated using specific generated datasets. For generation scripts:

- **Name Generation Corpus:** `demos/character-level/names/c_names.txt`
- **Shakespeare Character Corpus:** `examples/shakespeare/`

## Appendix E: Changelog

A version history of the book, reconstructed from the git commit log.

| Version | Date | Commit | Changes |
|---------|------|--------|---------|
| **1.3.0** | Feb 28, 2026 | — | Publication improvement pass: Ch.2 competitive positioning + guiding principles, Ch.3 AdamW/gradient clipping/label smoothing/K-V gradient accuracy, Ch.4 entropy gating + transfer learning, Ch.6 MD-delta/permutation tests/persistence baseline/BFS corpus generation, Ch.7 model soup/corpus-to-steps scaling, Ch.13 BPE/pruning/distillation/speculative decoding/WASM, Ch.14 O(1) rejection cost/neural vs TF-IDF retrieval, Appendix glossary+references+validation checklist. |
| **1.2.0** | Feb 27, 2026 | — | Chapter restructuring and content updates. Chapter 15 NAR section updated. |
| **1.1.0** | Feb 23, 2026 | `ea87a68` | Hex topology uplift (4%->27%), Red Donkey corpus expansion (12%->19%), MCTS corpus generation, Chapter 7 leaderboard updated, Appendix E changelog added |
| **1.0.8** | Feb 23, 2026 | `851182a` | Fresh benchmark data across PERFORMANCE.md and README |
| **1.0.7** | Feb 23, 2026 | `2783ad3` | Chapter 16 added: "The Nature of Reasoning — Unified Synthesis", reasoning conclusion document |
| | | `000e748` | Learning frontier section: what must be engineered vs what can be learned |
| **1.0.4** | Feb 23, 2026 | `9dd512e` | VM engine chapter updates, VM experiment findings integrated, test and benchmark fixes |
| **1.0.3** | Feb 21, 2026 | `d8a3b9b` | Inference cost economics ($5/experiment) added across VALUE_PROPOSITION and book chapters |
| | | `905434f` | Neural Algorithmic Reasoning (NAR) framing integrated across project docs |
| | | `1f3772d` | NAR reframing: "generalist monolith problem" — models waste budget on 30-80 line algorithms |
| | | `9519576` | Details and diagrams improved throughout |
| | | `b4c09d7` | Formatting pass across all chapters |
| **1.0.2** | Feb 21, 2026 | `e50e4c8` | Book logo updated |
| **1.0.1** | Feb 21, 2026 | `04dc7cf` | Cover updated with NAR framing |
| **1.0.0** | Feb 20, 2026 | `59f7f85` | First complete edition — 16 chapters + Appendix A, initial research integrated |
| | | `d4e5c76` | Chapter content updates |
| | | `2565c4a` | OPA biology analogy infographic added to Chapter 4 |
| | | `17bc4eb` | Chapter links added to table of contents |

## Appendix F: Validation Checklist

A systematic methodology for validating organelle experiments, derived from the research documented in `docs/testing/VALIDATION_CHECKLIST.md`.

### Pre-Training Checks

1. **Corpus integrity**: Verify corpus file exists, is non-empty, and has the expected format (one example per line, consistent delimiters).
2. **Corpus statistics**: Count examples, unique patterns, and vocabulary. Log `corpus_size`, `vocab_size`, `avg_example_length`.
3. **Parameter sizing**: Verify `params / corpus_size` falls within the 5-20x operating envelope. Flag if outside.
4. **Config validation**: Ensure `BLOCK_SIZE` >= longest example, `N_EMBD` divides evenly by `N_HEAD`.

### Training Checks

5. **Loss convergence**: Confirm loss decreases monotonically (smoothed over 100-step windows). Flag if loss at step 1000 > loss at step 100.
6. **Learning rate schedule**: Verify warmup completes before peak LR. For models >500K params, confirm LR <= 0.0005.
7. **Gradient health**: If gradient clipping is enabled, log the fraction of steps where clipping fires. If >50%, the learning rate may be too high.
8. **Overfitting detection**: Compare train loss vs. held-out validation loss every 5K steps. If gap exceeds 0.5, training should stop.

### Post-Training Validation

9. **Baseline comparison**: Run the trained model against (a) random baseline and (b) persistence baseline. The model must exceed random by a statistically significant margin.
10. **Permutation test**: Shuffle labels, retrain, and compare. If $\Delta < 5\%$, the model may not be learning genuine patterns.
11. **Seed variance**: Train 3 seeds (42, 7961, 15880). Compute mean and standard deviation. If std > 10% of mean, consider model soup.
12. **Edge deployment check**: Verify checkpoint size fits target device RAM. Measure inference latency (must be <10ms for real-time applications).

### Pipeline Validation (for OPA experiments)

13. **Invalid move rate**: Measure raw model invalids before pipeline filtering. Log the pipeline's correction rate.
14. **Coordination overhead**: Measure replans per game/task. If >5 replans per decision, the model may need more training data.
15. **Intelligence gap**: Compare trained model pipeline results vs. random model pipeline results. The gap must be statistically significant to claim genuine learning.
