# Phase 0 Decisions

Locked choices coming out of Phase 0 setup. This is the reference the rest of
the project builds on; later phases should be able to name their tooling without
re-litigating these. Deferred items list explicit re-entry criteria so "decide
later" doesn't become "never decide."

Last updated: end of Phase 0 (Unit 0.6).

## Locked

### RL framework — TRL (v1.0+)
- **Source**: Unit 0.4 research + smoke ([docs/spike_findings.md](spike_findings.md)).
- TRL's `GRPOTrainer` supports our multi-turn attacker↔target shape via
  `rollout_func`, which lets us own the dialogue loop (matches `env/dialogue.py`
  and the Attacker/Target protocols). The tool-calling-shaped `environment_factory`
  path is a worse fit for conversational attacks.
- OpenRLHF / verl are **not** needed as fallbacks.
- Note: TRL's fast generation uses vLLM (CUDA), so real GRPO training is a cloud
  activity. TRL 1.6 removed `GRPOConfig(max_prompt_length=...)` — only
  `max_completion_length` remains.

### Base attacker model — Qwen3-0.6B
- **Source**: Unit 0.5 A/B ([spikes/base_model_ab_outputs.md](../spikes/base_model_ab_outputs.md)).
- Current small Qwen (supersedes Qwen2.5-0.5B). Apache 2.0, 32k context,
  instruct + separate base, optional thinking mode.
- Chosen for sweeping the multi-turn coherence axis — the prior GRPO cannot
  easily instill. Its persuasion gap is exactly what GRPO trains; thinking mode
  is a reserved lever (disabled by default for now).
- **Band note**: 0.6B total / 0.44B non-embedding is above the original
  "200-350M" headline. The headline reconciliation (concept doc + plan wording,
  e.g. "sub-1B / ~0.6B specialist") is **deferred** — see below.

### Serving backends — MLX local, vLLM cloud
- **Source**: Unit 0.4.
- **Local dev (Apple M4 Pro, 24 GB, MPS)**: MLX / mlx-lm. Confirmed serving a
  4-bit target. vLLM does not run on Metal.
- **Cloud (CUDA)**: vLLM for training-rollout serving.
- Both sit behind the `Target` protocol — the dialogue env is backend-agnostic.

## Deferred (with re-entry criteria)

### Cloud provider — DEFERRED
- Candidates: Lambda / RunPod / Modal.
- **Re-entry**: decide when we first need a CUDA box — i.e., at the start of the
  fair-baseline fine-tune (Unit 3.4) or the first real SFT run (Unit 4.3),
  whichever comes first. At that point: pick one provider, run the deferred
  CUDA spikes (vLLM serving + cloud training hello-world), and record here.

### API eval budget cap — DEFERRED
- A soft $ ceiling per eval sweep against closed models (Claude / GPT-4o / Gemini).
- **Re-entry**: set before the first API eval sweep (Unit 6.2 / Phase 7 evals).
  Size the eval suite to fit once the figure is chosen.

### "200-350M" band headline — DEFERRED
- The Qwen3-0.6B choice exceeds the original band. Update the concept doc and
  plan wording to reflect a "~0.6B / sub-1B specialist" framing.
- **Re-entry**: fold into the Phase 8 writeup prep at the latest; can be done
  sooner as a docs touch-up. The "small beats large" thesis still holds — 0.6B
  vs frontier targets is still a >10× size gap.

## Confirmed environment facts (for resumption)

- Dev box: Apple M4 Pro — 20-core GPU, 24 GB unified memory, Metal 4; PyTorch
  MPS works. No CUDA locally.
- Python 3.11, `uv` for env management.
- Spike deps (torch, trl, transformers, mlx-lm, datasets, accelerate) are
  installed in the venv but **not** in `pyproject.toml`; real deps land in their
  own units.
