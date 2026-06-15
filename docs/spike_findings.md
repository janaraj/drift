# Unit 0.4 — Toolchain Spike Findings

Date: 2026-06-14

## Environment

- **Dev box**: Apple M4 Pro — 14 CPU cores, 20-core GPU, 24 GB unified memory, Metal 4.
- **PyTorch MPS**: available (`torch 2.12.0`, `torch.backends.mps.is_available() == True`).
- **No CUDA / NVIDIA GPU.** Consequences: vLLM cannot serve locally; CUDA-only
  accelerators (bitsandbytes 4-bit, flash-attention) are unavailable on this machine.

## The load-bearing question: does TRL support our multi-turn rollout shape?

**Verdict: YES. TRL is viable. No switch to OpenRLHF/verl needed.**

Answered by research against current docs (TRL v1.0+, April 2026), confirmed by the
installed `trl 1.6.0`. `GRPOTrainer` supports multi-turn RL via two mechanisms:

| Mechanism | Shape | Fit for Drift |
|---|---|---|
| `environment_factory` (their recommended) | Tool-calling: policy emits tool calls, env returns tool results. Trainer auto-runs the loop. | **Worse.** Forces the attacker to emit structured tool calls (e.g. a generic `send_message` tool, which their docs discourage), polluting the SFT format and fighting the "pure sequence model" design. |
| `rollout_func` | You own the entire generation + environment interaction loop, returning `{prompt_ids, completion_ids, logprobs, env_reward}`. | **Right fit.** Lets us own the attacker↔target loop — exactly the `env/dialogue.py` we are building on top of the Attacker/Target protocols (Unit 0.2). Attacker stays a plain message-generating model. |

Our attacks are **conversational** (attacker emits a natural-language message → target
LLM replies → repeat → judge gives a terminal reward), not tool-calling. So
`rollout_func` is the path. This aligns cleanly with the 0.2 architecture and de-risks
all of Phase 5.

Caveat: TRL's fast generation path uses vLLM (`use_vllm=True`), which is CUDA-only.
Real GRPO training is therefore a **cloud** activity; the local Mac validates API shape
and small-scale dev only.

Sources:
- TRL OpenEnv integration — https://huggingface.co/docs/trl/main/en/openenv
- RL Posttraining for Tool-Using Agents (GRPO, 2026) — https://zylos.ai/en/research/2026-04-10-rl-posttraining-tool-using-agents-grpo-async-rl/

## Hands-on spike results

### `spikes/spike_mlx_serve.py` — local target serving (MLX)

**PASS.** Loaded `mlx-community/Qwen2.5-0.5B-Instruct-4bit` in ~10 s, generated a correct
chat-templated response in ~0.5 s. Confirms the M4 Pro can serve local dialogue targets
for dev. The Phase 1 `env/targets/local_mlx.py` adapter will wrap this path; real targets
(4-bit 7-8B) use the same API and fit in 24 GB.

### `spikes/spike_trl_grpo.py` — TRL GRPO plumbing on MPS

**PASS.** `GRPOTrainer` constructed and ran 2 steps on MPS with
`HuggingFaceTB/SmolLM2-135M-Instruct`, `use_vllm=False`, transformers generation.
First step ~9 s, second ~1.7 s. Toy length-reward moved from −113.5 → −76 across the two
steps, confirming the generate→reward→update loop is live. This is a plumbing check only —
not representative of real training, which runs on cloud CUDA with vLLM.

API note: TRL 1.6.0 removed `GRPOConfig(max_prompt_length=...)`; only
`max_completion_length` remains. Recorded so the Phase 5 trainer config doesn't trip on it.

## Deferred (CUDA-only; re-entry when a cloud box is provisioned)

- **vLLM serving spike** — cannot run on Metal.
- **Cloud-provider training hello-world** — gated on the cloud-provider decision, itself
  deferred to Unit 0.6.

## Inputs this hands to Unit 0.6 (decisions doc)

- **RL framework**: TRL (v1.0+), using `rollout_func` for the multi-turn loop. Locked.
- **Local serving backend**: MLX / mlx-lm (Apple Silicon). Cloud serving: vLLM (CUDA).
- **Still open for 0.6**: base attacker model (0.5 spike), cloud provider, API eval budget cap.
