# MicroGPT-C Project Roadmap

This roadmap outlines the planned development for MicroGPT-C, a minimal C99 implementation of a GPT-style language model inspired by Andrej Karpathy's `microgpt.py`. The project aims to provide an educational, efficient, and portable tool for learning about Transformers, with a focus on reproducibility, low resource usage, and extensibility. As an open-source project under the MIT License, contributions are welcome via pull requests or issues on the repository (assuming hosted on GitHub or similar).

The roadmap is divided into phases: **Short-Term (Q1-Q2 2026)** for immediate improvements, **Medium-Term (Q3-Q4 2026)** for enhancements, and **Long-Term (2027+)** for ambitious expansions. Priorities are based on user feedback, performance benchmarks, and alignment with educational goals. Milestones will be tracked via GitHub issues/projects.

## Short-Term Goals (Q1-Q2 2026): Stability and Usability
Focus on polishing the core implementation, fixing issues from the initial review, and making it easier for beginners to get started.

1. **Bug Fixes and Robustness**:
   - Address memory leaks and improve error handling (e.g., use `goto cleanup` patterns in allocation functions like `model_create`).
   - Fix numerical stability in attention softmax by adding per-row max subtraction.
   - Enhance quantization: Implement round-to-nearest in `quantize_fp64_to_int8` and add safeguards against zero-scale divisions.
   - Target: Zero known crashes; add unit tests for forward/backward passes using a framework like Unity or CTest.

2. **Documentation Improvements**:
   - Create a comprehensive README with build instructions, examples (e.g., training on Shakespeare), and benchmarks (e.g., training time on CPU).
   - Add Doxygen-style comments for all public API functions.
   - Include tutorials: "Building Your First Model" and "Custom Tokenization".

3. **Basic Optimizations**:
   - Integrate simple SIMD (e.g., AVX2 for matrix multiplies in `lin_fwd` on x86).
   - Enable multi-threading for batch training using the existing `microgpt_thread.h` abstraction (parallelize `forward_backward_one` across batch items).
   - Milestone: 2x speedup on small models.

4. **Testing and CI**:
   - Set up GitHub Actions for automated builds/tests on Linux, macOS, and Windows.
   - Add equivalence tests against the Python reference (e.g., compare logits on toy datasets).

## Medium-Term Goals (Q3-Q4 2026): Features and Performance
Expand capabilities while maintaining the minimalistic ethos, targeting broader use cases like embedded systems or teaching.

1. **Advanced Features**:
   - Support for FP16/BF16 data types for GPU/embedded compatibility (e.g., via optional CUDA backend).
   - Dynamic limits: Replace hardcoded caps (e.g., `MAX_VOCAB`) with realloc-based growth.
   - Unicode support in tokenization (e.g., UTF-8 handling for non-English corpora).

2. **Performance Enhancements**:
   - Integrate BLAS libraries (e.g., OpenBLAS) for matrix operations as an optional dependency.
   - Add profiling tools (e.g., integrate with gprof) and optimize KV cache for larger `BLOCK_SIZE`.
   - Quantization extensions: Per-tensor or asymmetric quantization for better accuracy.

3. **Extensibility**:
   - Modular backends: Allow custom activation functions (e.g., GELU instead of ReLU) via callbacks.
   - Dataset loaders: Add support for mmap-based large file handling and streaming datasets.
   - Community demos: Add examples for word-level models on larger texts (e.g., Tiny Shakespeare integration).

4. **Community and Tools**:
   - Release v1.0 with a changelog and binary releases for major platforms.
   - Set up a discussion forum or Discord for users.
   - Add visualization tools (e.g., export weights to TensorBoard-compatible formats).

## Long-Term Goals (2027+): Scalability and Ecosystem
Scale to more advanced applications, potentially evolving into a lightweight library for AI experimentation.

1. **Scalability**:
   - Distributed training: Basic MPI support for multi-node setups.
   - Larger architectures: Increase defaults (e.g., N_EMBD=128, N_LAYER=6) with configurable presets.
   - Inference optimizations: Add beam search and top-k/top-p sampling.

2. **Integrations**:
   - Embeddings export: Compatibility with Hugging Face Transformers for model sharing.
   - Bindings: Python/C++ wrappers via SWIG or pybind11 for easier integration.
   - Hardware acceleration: Full CUDA/ROCm support for GPU training/inference.

3. **Research and Education**:
   - Ablation studies: Tools to experiment with variants (e.g., no positional embeddings).
   - Educational resources: Video tutorials, blog posts on internals, and a "build your own GPT" workshop.
   - Sustainability: Explore funding via sponsors or grants for open AI tools.

## Risks and Dependencies
- **Risks**: Limited contributor bandwidth; prioritize based on issues filed. Performance gains depend on hardware testing.
- **Dependencies**: Rely on community for ports (e.g., ARM for embedded). Monitor C standards for future compatibility (e.g., C23 features).
- **Metrics for Success**: Aim for 1K+ GitHub stars, 100+ forks, and positive feedback in AI education communities.

This roadmap is flexible and will be updated quarterly based on progress and feedback. If you'd like to contribute or discuss specifics, feel free to open an issue!
