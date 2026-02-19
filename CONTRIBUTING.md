# Contributing to MicroGPT-C

Thank you for your interest in MicroGPT-C! We welcome contributions from everyone.

---

## Ways to Contribute

| Contribution | How |
|-------------|-----|
| **Bug reports** | Open an [issue](../../issues) with a minimal reproduction case |
| **Feature requests** | Open an issue describing the use case and proposed approach |
| **Code contributions** | Fork → branch → PR (see below) |
| **Documentation** | Fix typos, improve explanations, add examples |
| **Experiments** | Add new organelle experiments with Spear-formatted READMEs |

---

## Pull Request Process

1. **Fork** the repository and create a feature branch from `main`
2. **Keep it focused** — one PR per feature or fix
3. **Follow the style** — C99, no external dependencies in core (`microgpt.h` + `microgpt.c`)
4. **Add tests** if touching core engine code — see `tests/test_microgpt.c`
5. **Run the test suite** before submitting:
   ```bash
   mkdir build && cd build
   cmake ..
   cmake --build .
   ./test_microgpt       # All tests must pass
   ./bench_microgpt      # No performance regressions
   ```
6. **Document your changes** — update relevant docs in `docs/` or add a Spear summary

---

## Code Style

- **C99 standard** — no C11/C23 features in core engine
- **Zero dependencies** — core engine uses only `libc` + `libm`
- **Optional accelerators** are fine behind `#ifdef` guards (Metal, BLAS, etc.)
- Use `scalar_t` for all weights/activations (never hardcode `float` or `double`)
- Prefer explicit loops over macros — readability over cleverness
- Comment the *why*, not the *what*

---

## Adding a New Organelle Experiment

1. Create a directory under `experiments/organelles/your_experiment/`
2. Include a `main.c` with the training and inference loop
3. Add a `README.md` using the [Spear framework](docs/organelles/) (Point, Picture, Proof, Push)
4. Register the target in `CMakeLists.txt` using the `add_demo()` macro
5. Document results — we value honest reporting, including failures

---

## Ethical Guidelines

When contributing organelle experiments or use cases:

- **Document limitations honestly** — if a model fails at novel tasks, say so
- **Include confidence scoring** — organelles should report when they don't know
- **Consider misuse potential** — if your organelle could be used harmfully, note it
- **No training on private data** without consent, even if the model runs locally
- **Bias awareness** — small models trained on narrow corpora inherit the biases of that corpus

See the [Responsible Use](#responsible-use) section in the README for the project's position.

---

## Questions?

Open a [discussion](../../discussions) or email the maintainer at ajay.soni@enjector.com.

---

*MicroGPT-C — Enjector Software Ltd. MIT License.*
