# VibeThinker-3B

<p align="center">
  <a href="VibeThinker-3B.pdf">Paper (PDF)</a> &nbsp;|&nbsp;
  <a href="https://arxiv.org/abs/2606.16140">arXiv:2606.16140v1</a> &nbsp;|&nbsp;
  <a href="https://github.com/WeiboAI/VibeThinker">GitHub</a> &nbsp;|&nbsp;
  <a href="https://huggingface.co/WeiboAI/VibeThinker-3B">HuggingFace</a> &nbsp;|&nbsp;
  <a href="IDEAS.md">Ideas for microgpt-c →</a>
</p>

> *VibeThinker-3B: Exploring the Frontier of Verifiable Reasoning in Small Language Models.*
> Sen Xu, Shixi Liu, Wei Wang, Jixin Min, Yingwei Dai, Zhibin Yin, Yirong Chen, Xin Zhou, Junlin Zhang — Sina Weibo Inc., June 2026.

---

## Thesis

A **3B dense** model (built on Qwen2.5-Coder-3B) reaches *frontier-level* performance on **verifiable** reasoning tasks — 94.3 on AIME26 (97.1 with test-time scaling), 80.2 Pass@1 on LiveCodeBench v6, 96.1% acceptance on unseen LeetCode contests — matching or exceeding models 200–300× larger (DeepSeek V3.2 671B, GLM-5 744B, Kimi K2.5 1T) on these tasks, while a visible gap remains on knowledge-heavy benchmarks (GPQA-Diamond). The lift comes entirely from **post-training**, not scale.

## Method — the "Spectrum-to-Signal" pipeline

SFT builds a broad, diverse solution space (the *Spectrum*); RL amplifies the high-value reasoning signals within it (the *Signal*). Stages (Fig. 3):

- **Two-stage curriculum SFT** — Stage 1 broad coverage; Stage 2 hard-reasoning subset via joint length-difficulty filtration (8 rollouts/query, keep only problems with error rate > 0.75).
- **Diversity-Exploring Distillation** — preserve multi-path solutions in SFT; save intermediate checkpoints, pick per-domain specialists by **Pass@K** (most valid solutions, not lowest val-loss), merge at the parameter level.
- **MaxEnt-Guided Policy Optimization (MGPO)** — GRPO-style RL that up-weights prompts near the model's capability boundary (empirical accuracy `p(q) ≈ 0.5`, maximum entropy): `w(q) = exp(-γ·D_ME(p(q)‖0.5))`.
- **Multi-domain Reasoning RL** — sequential Math → Code → STEM, each with its own verifier (final-answer / sandbox-execution / option-matching).
- **Long2Short Math RL** — "from accuracy to efficiency": a zero-sum, brevity-centered reward shift redistributed *only* among correct trajectories, preferring shorter correct reasoning without changing the group baseline.
- **Offline Self-Distillation** — distil verified RL-stage traces back into one student, prioritising traces by a length-normalised *learning-potential score* (traces the student does not yet model well).
- **Instruct RL** — rule-based constraint validators + rubric-based reward models restore strict instruction-following after the reasoning push.
- **Claim-Level Reliability Assessment (CLR)** — test-time scaling: generate `K=32` trajectories, extract `M=5` decision-relevant claims each, self-verify each claim to a binary verdict `v_{k,m}`, score each trajectory nonlinearly `r_k = ((1/M)Σ v_{k,m})^M`, cluster answers by equivalence, and pick the answer maximising summed reliability. Adds +2–3 points on verifiable-math benchmarks without touching weights.

## The framing claim

**Parametric Compression-Coverage Hypothesis.** Foundational capabilities differ not just in *how much* parameter capacity they need but in the *structural form* of that demand:

- **Verifiable reasoning** is *parameter-dense* — its core is search, constraint satisfaction, error correction, and multi-step composition in a structured solution space. It compresses into a compact, reusable reasoning core.
- **Open-domain knowledge / general competence** is *parameter-expansive* — it needs broad coverage over facts, concepts, semantic associations, and long-tail scenarios, which scales with raw parameter count.

Under the companion **Reasoning-Knowledge Decoupling Paradigm**, small models can host a high-density reasoning engine for verifiable, structured tasks, while large models remain the natural vehicle for broad knowledge.

---

See [**IDEAS.md**](IDEAS.md) for an evaluation of which of these ideas transfer to microgpt-c and its organelles (and which are already rejected or falsified here).
