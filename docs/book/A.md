# Appendix A: Glossary and References

This appendix provides a comprehensive glossary of key terms used throughout the book, along with a curated list of references. The glossary defines concepts in simple, accessible language, drawing from the explanations in the chapters. Terms are listed alphabetically for easy reference. The references include foundational papers and resources that influenced the principles of MicroGPT-C, such as transformer architectures and optimization techniques. These are cited in a standard format (APA style) for further reading. Note that while the book focuses on practical implementation, these sources offer deeper theoretical insights.

## Glossary

- **Adam Optimizer**: An adaptive optimization algorithm used in training that adjusts learning rates for each parameter based on historical gradients. It incorporates momentum and variance scaling to improve convergence, especially on noisy data (see Chapter 3).

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

- **c_compose**: The code composition experiment using a Planner→Judge pipeline (1.2M params each with LR scheduling) to generate function composition plans, achieving 83% exact match (see Chapter 10).

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

- **Gradient Descent**: The core method for updating model parameters by following the direction of steepest error reduction (see Chapter 3).

- **Grouped Query Attention (GQA)**: An efficient attention variant that shares keys and values across query head groups, reducing memory (see Chapter 8).

- **Inference**: The phase where a trained model generates outputs from new inputs, without updating parameters (see Chapter 3).

- **Internet of Things (IoT)**: A network of connected devices that collect and exchange data, enhanced by edge AI for local processing (see Chapter 11).

- **Judge (in Pipelines)**: A deterministic component that validates outputs, such as checking move legality or syntax (see Chapter 5).

- **Kanban Architecture**: A coordination system using shared state (todo, blocked, history) to manage pipeline workflows and handle failures (see Chapter 5).

- **KV Cache**: Stored keys and values from past attention computations, speeding up sequential inference (see Chapters 3 and 8).

- **Learning Rate Scheduling**: Gradually adjusting the step size in optimization, often with warmup and decay for stability (see Chapter 3).

- **LR-Capacity Scaling**: The empirical rule that larger models require lower learning rates: lr ∝ 1/√params. At 460K params lr=0.001 works; at 1.2M params lr=0.0005 is needed to prevent divergence (see Chapters 3 and 4).

- **Multi-Head Attention (MHA)**: Parallel attention computations where each head learns different relationships (see Chapter 8).

- **Multi-Query Attention (MQA)**: An extreme efficiency variant sharing one set of keys/values across all queries (see Chapter 8).

- **Organelle**: A small, specialized AI model differentiated from a stem cell base for focused tasks (see Chapter 4).

- **Organelle Pipeline Architecture (OPA)**: A framework for coordinating multiple organelles via planners, workers, and judges (see Chapter 5).

- **Overconfidence**: When a model assigns high probability to incorrect outputs; mitigated by ensembles and gating (see Chapter 12).

- **Overfitting**: When a model memorizes training data but fails on new inputs; detected by comparing train/test losses (see Chapter 3).

- **Paged KV Cache**: A memory-efficient cache that allocates in fixed pages, handling long sequences without fragmentation (see Chapter 7).

- **Paraphrase Blindness**: Model failure on reworded inputs due to literal matching; addressed by decomposition (see Chapter 10).

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