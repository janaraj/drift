# Drift

**A study of conversational manipulation in language models.**

Drift investigates whether *manipulation is a learnable specialization* — that
is, whether a small, focused model can be trained to steer much larger frontier
models into undesired behavior through sustained, multi-turn dialogue, at rates
exceeding what a general-purpose large model achieves at the same task.

The headline question: **does specialization beat scale in adversarial
conversation?** If a ~0.6B attacker, trained on open-weight targets, can
reliably elicit target behaviors from frontier models — and *transfer* to closed
models it never trained against — that challenges the implicit assumption that
safety scales with model size, and surfaces a class of vulnerability in how
current models handle prolonged adversarial pressure.

> **Status: early / pre-alpha.** The evaluation environment is being built
> (Phase 1). There are no trained attacker weights yet, and no empirical
> claims — only the methodology and infrastructure. See
> [Status](#status) below for exactly what works today.

---

## Why this matters

Single-turn jailbreaking is largely a solved research area. The open frontier is
**multi-turn manipulation** — dialogues where an attacker adapts, builds context,
exploits sycophancy and role-consistency, and applies graduated pressure across
many turns. Prior work (LM-vs-LM red-teaming, PAIR, TAP, Crescendo) mapped the
shape of this problem, but did so with *off-the-shelf* large attackers. No one
has trained a small, dedicated specialist and asked whether specialization, not
scale, is what makes an effective multi-turn attacker.

The contribution Drift aims for is **methodological**: a clean, reproducible
benchmark and training recipe for multi-turn manipulation, with **programmatic
judges** (not LLM-as-judge) for the target behaviors — which avoids the
judge-quality trap that undermines much red-team research.

## What "manipulation" means here

V1 deliberately targets **capability demonstrations and enterprise concerns**,
not weaponizable harm:

- **E — System prompt extraction.** Get the target to reveal its hidden system
  instructions. Crisp, verifiable metric (verbatim / near-duplicate match); a
  real confidentiality concern for deployed assistants.
- **B — Behavioral / persona violation.** Get the target to break its assigned
  role — e.g. claim to be human, violate an assigned persona name, or contradict
  its own prior statements. Capability demonstration, deterministically judged.

Generic harmful-content jailbreaking is **explicitly out of scope** — it is
crowded, weaponizable, and not the research question here. Multi-agent
goal-hijacking is potential future work.

## Responsible use

This is **defensive safety research**. The goal is to measure and understand a
vulnerability class so it can be defended against, and to give safety teams a
reproducible way to evaluate multi-turn robustness.

- The target behaviors are chosen to be **demonstrative, not harmful** (no
  harmful-content generation).
- Judging is **programmatic**, so results are reproducible and auditable.
- Release of any trained checkpoint is gated on an explicit **harm-vector
  audit** (a planned, documented step) made with results in hand — not assumed.

If you build on this, please use it to make models more robust, and follow
responsible-disclosure norms for anything you find against production systems.

## How it works

```
            ┌───────────┐   attacker turn (user)    ┌──────────┐
            │  Attacker │ ────────────────────────▶ │  Target  │
            │  (~0.6B)  │ ◀──────────────────────── │  (LLM)   │
            └───────────┘   target turn (assistant)  └──────────┘
                  │                                        │
                  │            full dialogue               │
                  ▼                                        ▼
            ┌──────────────────────────────────────────────────┐
            │  Judge (programmatic)  →  success / metadata       │
            └──────────────────────────────────────────────────┘
```

- An **attacker** model generates messages aimed at eliciting a target behavior.
- A **target** model (local open-weight, or a frontier model via API) responds.
- A multi-turn **dialogue environment** runs the exchange and logs every turn.
- A **judge** scores the completed dialogue programmatically.
- The attacker is bootstrapped with supervised fine-tuning, then improved with
  GRPO (reinforcement learning) against the judge's reward.

The training thesis is validated by **transfer**: train the attacker against
open-weight targets, then evaluate whether its attacks succeed against closed
frontier models — and whether it beats a fairly-matched, *larger* fine-tuned
attacker baseline.

### Plug-in architecture

The codebase is built so that new behaviors, targets, and baselines plug in
without touching the environment, training loop, or evaluation harness. Adding a
new behavior requires only: a new judge, a scenario set, and a behavior
registration. This contract is enforced by a test.

## Status

Pre-alpha. Building the evaluation environment (Phase 1).

| Area | State |
|---|---|
| Core protocols & plug-in registry | ✅ done |
| Local target adapter (MLX / Apple Silicon) | ✅ done |
| API target adapters (Anthropic, OpenAI, Google) | ✅ done (Google live-verified) |
| Multi-turn dialogue loop | 🚧 in progress |
| Rollout logging | ⬜ planned |
| Judges (E, then B) | ⬜ planned |
| Scenario suites | ⬜ planned |
| Baselines, SFT, GRPO, transfer eval | ⬜ planned |
| Trained attacker weights | ⬜ none yet |

Key decisions so far: attacker base **Qwen3-0.6B**; RL via **TRL** (`GRPOTrainer`
`rollout_func` for the multi-turn loop); local serving via **MLX**, cloud via
**vLLM**. See [docs/decisions.md](docs/decisions.md).

## Project layout

```
src/drift/
  core/        protocols (Target, Attacker, Judge), registry, Behavior
  env/         dialogue loop, rollout logging, target adapters (local + API)
  judges/      programmatic judges (system-prompt extraction, persona)
  scenarios/   scenario schema + loader
  attacker/    SFT + GRPO training, reward composition
  baselines/   naive, Crescendo, PAIR-style, fine-tuned-LLM baselines
  eval/        ASR matrices, transfer experiments, attack-pattern catalog
  behaviors/   behavior registrations (E, B)
docs/          decisions, spike findings, (later) writeup + harm audit
spikes/        throwaway toolchain validation scripts
```

## Development

Requires Python 3.11+. Uses [uv](https://docs.astral.sh/uv/).

```bash
uv venv
uv pip install -e ".[dev]"        # core + test/lint tooling

uv run pytest                     # tests
uv run ruff check .               # lint
```

Optional extras (install only what you need):

```bash
uv pip install -e ".[local]"      # MLX local target serving (Apple Silicon)
uv pip install -e ".[google]"     # Gemini API target  (also: anthropic, openai, api)
```

API targets read credentials from environment variables
(`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GEMINI_API_KEY`); a gitignored `.env`
is supported. No keys are stored in the repo.

## Design docs

- [Research concept](internal_docs/Drift%20%E2%80%94%20A%20Study%20of%20Conversational%20Manipu.md)
- [Project plan](internal_docs/Plan.md)
- [Phase-0 decisions](docs/decisions.md)
- [Toolchain spike findings](docs/spike_findings.md)

## License

To be determined (see [LICENSE](LICENSE)) — finalized alongside the release
decision and harm-vector audit. Treat as all-rights-reserved until then.
