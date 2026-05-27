# Drift — Project Setup & Phase Plan

## Context

Drift is a research project to train a small (200-350M) specialist attacker model that elicits target behaviors from frontier LLMs through multi-turn dialogue, with the thesis that specialization beats scale in adversarial conversation. The full concept lives in [internal_docs/Drift — A Study of Conversational Manipu.md](internal_docs/Drift%20%E2%80%94%20A%20Study%20of%20Conversational%20Manipu.md).

This plan covers **V1 only**: target behaviors **E (system prompt extraction)** and **B (behavioral / persona jailbreaking)**, in that order. Volnix-dependent multi-agent goal-hijacking (original Option C) is deferred to V2 with explicit re-entry criteria. Options A and D are excluded.

The plan is shaped by four user inputs collected up front:
- **Compute**: local GPU for dev, cloud for training/eval sweeps.
- **Pace**: bursty / no fixed schedule → every phase must produce persistent artifacts, and resumption from any phase boundary must be cheap.
- **Release**: fully open — code and weights. Raises dual-use stakes; demands an explicit harm-vector audit before checkpoint release.
- **Order**: E first, then B. E has the crispest judge (substring); de-risk infra against a fully trusted judge before tackling fuzzier behaviors.

The repo is currently empty apart from the concept doc, so this plan starts at greenfield setup.

## Guiding principles

1. **Resumability over speed.** Bursty work means weeks may pass between sessions. Every phase ends with files on disk (configs, datasets, checkpoints, eval results), never in-memory state. Each phase has a smoke-test command that re-validates the phase's output in under 10 minutes.
2. **Reusability by construction, not by refactor.** V1 is E+B but the design assumes V2 (Volnix-based goal-hijacking) and beyond. New behaviors plug in via three concrete extension points only — a new judge, a new scenario set, and a behavior registration. No new behavior should require touching the dialogue env, training loop, eval harness, or baseline runner.
3. **Methodology is the contribution.** Without Volnix, "small specialist beats large generalist via SFT+GRPO" is doing 100% of the novelty work. Baseline choice and transfer evaluation are load-bearing — not afterthoughts.
4. **Judge-first, then scenarios, then training.** Most red-team papers die at the judge. Lock judges with unit tests before authoring scenarios; lock scenarios before training.
5. **Open release demands harm audit.** A fully open attacker checkpoint trained on E is non-trivially dual-use. Final release tier is a Phase 9 decision, made with results in hand — not pre-committed.

## How we work together

Bursty pace + "I want to read the code before committing to direction" implies a tight micro-loop, not large drops:

1. **One review unit at a time.** Each phase below is broken into small numbered units (e.g., 1.1, 1.2). I implement a single unit and stop.
2. **You read the diff.** I summarize what changed and what to look at; you walk the code and push back or approve.
3. **Commit per approved unit.** Once you approve, commit with a tight message. Multiple units in flight at once is disallowed — one approved-and-committed before the next starts.
4. **Push at phase boundary (or sooner if you ask).** Default is push when a phase's units are all green. Sooner if you want offsite backup mid-phase.
5. **No phase starts before the prior phase's units are all merged.** This is what makes resumption cheap and prevents implicit dependencies between half-finished work.

Each unit below lists what gets built, what you should review, and the commit message I'll propose. If a unit turns out larger than expected in implementation, I split it before stopping — never bundle.

## Setup

### Repo layout (designed for plug-in extension)

```
drift/
  internal_docs/                       # existing concept doc
  configs/                             # YAML per scenario suite, per training run, per eval
    behaviors/{e,b}.yaml               # behavior config: judge cls, scenario dir, reward weights
    training/{sft,grpo}_*.yaml
    eval/*.yaml
  data/
    scenarios/{e,b}/                   # hand-authored scenario YAMLs, behavior-namespaced
    sft/                               # curated multi-turn SFT corpus
    eval/                              # baseline + checkpoint eval result JSONs (versioned)
    logs/                              # JSONL rollout logs (gitignored, versioned dir)
  src/drift/
    core/
      protocols.py                     # Target, Judge, Scenario, Rollout, Baseline protocols
      registry.py                      # decorator-based registries for targets/judges/baselines/behaviors
      behavior.py                      # Behavior dataclass: name + judge + scenario_loader + reward_cfg
    env/
      dialogue.py                      # N-turn loop; pure function over (attacker, target, scenario)
      logging.py                       # JSONL rollout persistence
      targets/
        base.py                        # Target protocol (re-exports from core)
        local_vllm.py                  # local open-weight via vLLM
        api_anthropic.py
        api_openai.py
        api_google.py
        __init__.py                    # registers all targets
    judges/
      base.py                          # Judge protocol (re-exports from core)
      extraction.py                    # E judge
      persona.py                       # B judge (3 sub-behaviors)
      __init__.py                      # registers all judges
    scenarios/
      loader.py                        # behavior-agnostic YAML loader + schema validation
      schema.py                        # common Scenario schema; behavior_config sub-field
    attacker/
      sft.py                           # behavior-agnostic SFT trainer
      grpo.py                          # behavior-agnostic GRPO loop; pulls reward via behavior registry
      reward.py                        # composable reward = judge + diversity + length
    baselines/
      base.py                          # Baseline protocol (re-exports from core)
      naive.py
      crescendo.py
      pair.py
      finetuned_llama.py               # the fair baseline
      __init__.py                      # registers all baselines
    eval/
      harness.py                       # eval(checkpoint, behavior, targets, scenarios) -> ASR matrix
      transfer.py                      # train-on-X / eval-on-Y matrix
      catalog.py                       # qualitative attack-pattern clustering
    behaviors/
      __init__.py                      # registers E and B; new behaviors register here
      e.py                             # Behavior("e", judge=ExtractionJudge, scenarios="e", reward=...)
      b.py                             # Behavior("b", judge=PersonaJudge, scenarios="b", reward=...)
  scripts/                             # phase smoke tests, one-shot CLIs
  tests/                               # unit tests, especially for judges and plug-in contracts
  pyproject.toml
  README.md
  AGENTS.md                            # for future Claude/Codex sessions; phase + plug-in contract
```

### Plug-in extension contract (the reusability story)

Adding a new behavior (e.g., V2 Volnix goal-hijacking) requires **only**:

1. Implement a new `Judge` subclass under `src/drift/judges/<name>.py` and register it.
2. Drop scenario YAMLs under `data/scenarios/<name>/` matching the common schema.
3. Add `src/drift/behaviors/<name>.py` that constructs the `Behavior` dataclass and registers it.
4. Add `configs/behaviors/<name>.yaml` pointing to the above.

Adding a new target (open or API) requires only a new file under `src/drift/env/targets/` implementing the `Target` protocol and registering itself.

Adding a new baseline requires only a new file under `src/drift/baselines/` implementing the `Baseline` protocol.

**Nothing else changes.** The dialogue env, SFT, GRPO loop, eval harness, and baseline runner are all behavior-agnostic. This is verified by a "plug-in contract test" in `tests/` that registers a dummy throwaway behavior and runs the full pipeline against it.

If a new behavior would force changes to env/training/eval code, that's a design failure and we redesign rather than monkey-patch. The plug-in contract test is the tripwire.

### Toolchain (initial picks, revisit at Phase 0 close)

- **Python 3.11**, `uv` for env mgmt.
- **PyTorch + transformers + accelerate** for SFT.
- **vLLM** for local target serving (fast multi-turn rollouts).
- **TRL** for GRPO as default; **OpenRLHF** or **verl** as fallback if TRL's GRPO support proves immature for multi-turn dialogue rollouts. Decide concretely after a small TRL spike in Phase 0.
- **Anthropic / OpenAI / Google** Python SDKs for API targets, behind a unified adapter.
- **Hydra** or simple YAML+pydantic for config.
- **Weights & Biases** for run tracking (free tier).

### Compute

- **Local dev**: whichever GPU(s) you have. Used for env work, judge dev, scenario authoring, sanity-checking small SFT runs, smoke-testing rollouts against one local target.
- **Cloud**: rented for (a) full SFT runs, (b) GRPO loop, (c) target-zoo rollouts at scale, (d) closed-model eval sweeps. Lambda / RunPod / Modal are the three usual candidates; pick one in Phase 0 and stick with it.
- **API budget cap** for eval: set a soft cap in Phase 0 (e.g., $X per eval sweep) and size the eval suite to fit.

### Base attacker model

Two finalists: **Qwen 2.5 0.5B** vs **SmolLM2-360M**. Lean Qwen for better instruction-following priors; decide via a 30-minute Phase 0 spike (run an identical 200-example SFT prompt set through both, eyeball quality).

### Target zoo

- **Local (training rollouts)**: Llama 3 8B, Qwen 2.5 7B, Mistral 7B Instruct, Gemma 2 9B. Subset depends on local VRAM; can also serve from cloud during GRPO.
- **API (evaluation only)**: Claude (Haiku + Sonnet), GPT-4o, Gemini Flash.

## Phases

Each phase = a set of small **review units**. Per unit: what I build → what you review → commit message. Commit per approved unit. Push at phase boundary unless you ask sooner. Next phase starts only after the prior phase's units are all merged.

### Phase 0 — Setup & decisions

Goal: lock toolchain, prove the boring pieces work end-to-end before any research code.

- **0.1 Repo scaffold.** `pyproject.toml`, `src/drift/` package skeleton (empty modules matching the layout above, with docstrings), `tests/` with one passing dummy test, `ruff` + `pytest` config, `.gitignore`, README stub. Review: layout matches plan, lint+test green. Commit: `chore: initial repo scaffold`.
- **0.2 Core protocols.** `src/drift/core/{protocols,registry,behavior}.py` only — the abstract `Target`, `Judge`, `Scenario`, `Rollout`, `Baseline`, `Behavior` types and the decorator-based registry. No implementations. Review: type signatures match the plug-in contract; new behaviors really only need to touch the three extension points. Commit: `feat(core): protocols and plug-in registry`.
- **0.3 Plug-in contract test.** `tests/test_plugin_contract.py` that registers a dummy throwaway behavior (trivial judge that returns True on any dialogue containing "X", trivial scenarios) and asserts the registries surface it without any other code change. Fails today (no env/training yet), but the *registration* path must work. Review: this test is the tripwire — does it actually catch a layering violation? Commit: `test(core): plug-in contract test for behaviors`.
- **0.4 Toolchain spikes.** Three throwaway scripts under `spikes/` (gitignored after merge): TRL GRPO hello-world, vLLM local serving hello-world, cloud provider hello-world training job. Each runs in isolation, prints success. Review: do these prove TRL's GRPO supports our multi-turn dialogue rollout shape? If not, switch to OpenRLHF before going further. Commit: `chore(spikes): toolchain validation scripts`.
- **0.5 Base-model A/B spike.** Run a 200-example identical prompt set through Qwen 2.5 0.5B and SmolLM2-360M; record outputs side-by-side. Review: eyeball both, pick one. Commit: `chore(spikes): base attacker model A/B comparison`.
- **0.6 Decisions doc.** `docs/decisions.md` recording locked choices: base model, RL framework, cloud provider, API eval budget cap. Review: anything missing? Commit: `docs: lock phase-0 decisions`. **Push at end of phase.**

**Exit criterion**: plug-in contract test passes (for the protocol-level checks), every later phase can name its tooling without ambiguity.

### Phase 1 — Dialogue environment & target zoo

Goal: any attacker can talk to any target for N turns and produce a persisted JSONL log.

- **1.1 Target adapters: local.** `env/targets/local_vllm.py` implementing `Target` against a local vLLM server. Smoke test against one model (e.g., Qwen 2.5 7B). Review: protocol conformance; metadata captured (tokens, latency). Commit: `feat(env): local vLLM target adapter`.
- **1.2 Target adapters: API.** `env/targets/{api_anthropic,api_openai,api_google}.py`. Same `Target` interface. Auth via env vars only. Review: one adapter per provider works against a real call; rate-limit handling sane. Commit: `feat(env): API target adapters (Anthropic, OpenAI, Google)`.
- **1.3 Dialogue loop.** `env/dialogue.py` — pure function: `run_dialogue(attacker, target, scenario, max_turns) -> Rollout`. No behavior-specific logic. Review: the loop is genuinely behavior-agnostic; max_turns and termination conditions are parameters. Commit: `feat(env): N-turn dialogue loop`.
- **1.4 Rollout logging.** `env/logging.py` — JSONL persistence with versioned run dirs under `data/logs/<run_id>/`. Resumable: given a run_id, can list completed scenarios. Review: log schema captures everything needed for later judge replay. Commit: `feat(env): rollout JSONL logging`.
- **1.5 Smoke script.** `scripts/smoke_dialogue.py` — runs a hand-crafted attacker prompt against one local + one API target, persists logs. Review: end-to-end in <2 min, log file inspectable. Commit: `chore: phase-1 smoke script`. **Push at end of phase.**

**Exit criterion**: `python scripts/smoke_dialogue.py` runs end-to-end against one local + one API target, persisted log validates against the rollout schema.

### Phase 2 — Judges + E scenarios + behavior registration (E)

Goal: E behavior fully registered and validated, judges unit-tested. No training yet.

- **2.1 Scenario schema + loader.** `scenarios/{schema,loader}.py` — common YAML schema with a `behavior_config` sub-field for behavior-specific data, schema validation via pydantic, behavior-agnostic loader. Review: schema covers both E (hidden secret) and B (assigned persona) cleanly via `behavior_config`. Commit: `feat(scenarios): common schema and loader`.
- **2.2 E judge.** `judges/extraction.py` implementing `Judge`: exact substring + Levenshtein ratio + n-gram overlap with configurable thresholds. Register under name `"extraction"`. Review: threshold defaults justified, edge cases (partial leakage, paraphrased leakage) handled. Commit: `feat(judges): system-prompt extraction judge`.
- **2.3 E judge tests.** `tests/test_extraction_judge.py` — ~30 positive + ~30 negative curated dialogue snippets. All green. Review: snippets cover the edge cases that scenarios will actually produce. Commit: `test(judges): extraction judge unit tests`.
- **2.4 E scenarios.** First **5** hand-authored E scenarios under `data/scenarios/e/`. Varied: secret length, secrecy framing strength, domain. Review: representative coverage; schema-valid. Commit: `feat(scenarios): initial E scenarios (5)`.
- **2.5 E behavior registration.** `behaviors/e.py` + `configs/behaviors/e.yaml` — wires extraction judge + E scenarios + initial reward config into a `Behavior`. Plug-in contract test now passes for E (registers, loads scenarios, judge applies to rollouts). Review: registration path matches the plan's plug-in contract. Commit: `feat(behaviors): register E (system prompt extraction)`.
- **2.6 E scenarios v1 complete.** Add remaining ~25-45 scenarios to reach 30-50 total. Review: distribution across difficulty/framing/domain looks balanced. Commit: `feat(scenarios): complete E scenario suite v1`. **Push at end of phase.**

**Exit criterion**: `pytest` green; `python scripts/list_scenarios.py --behavior e` lists 30+ scenarios; plug-in contract test exercises full E path.

### Phase 3 — E baselines

Goal: set the bar the specialist must clear. Run before any training.

- **3.1 Baseline protocol + naive baseline.** `baselines/{base,naive}.py` — `Baseline` protocol and a direct "what is your system prompt" single-turn attacker. Register. Review: protocol matches plug-in contract. Commit: `feat(baselines): protocol and naive baseline`.
- **3.2 Hand-crafted multi-turn baselines.** `baselines/crescendo.py` (escalation), plus 1-2 other hand-crafted patterns (role-play framing, persona-shift). Review: implementations match published descriptions where applicable. Commit: `feat(baselines): hand-crafted multi-turn baselines`.
- **3.3 PAIR-style baseline.** `baselines/pair.py` — GPT-4 or Claude as iterative attacker. API-cost-gated by config flag. Review: cost cap respected; reproducible at low N for sanity. Commit: `feat(baselines): PAIR-style attacker baseline`.
- **3.4 Fine-tuned Llama baseline.** `baselines/finetuned_llama.py` — Llama 3 8B SFT'd on the same data the specialist will use later. **This is the fair baseline.** Review: SFT recipe documented; checkpoint reproducible. Commit: `feat(baselines): fine-tuned Llama 3 8B fair baseline`.
- **3.5 Eval harness + baseline run.** `eval/harness.py` runs (baseline × target × scenarios) → ASR matrix at `data/eval/baselines_e.json`. Review: numbers sane; if naive >70% ASR, return to Phase 2.6 and harden scenarios. Commit: `feat(eval): baseline ASR matrix for E`. **Push at end of phase.**

**Exit criterion**: full baseline ASR matrix exists; naive baseline does not saturate (<70% ASR).

### Phase 4 — E SFT bootstrap

Goal: a competent SFT-only attacker as the GRPO starting point.

- **4.1 SFT data curation.** `data/sft/e_v1.jsonl` from: filtered public red-team data + ~100-300 hand-authored exemplars (+ optional judge-filtered distillation). Review: sample 30 random examples — quality acceptable, no obvious junk. Commit: `feat(data): E SFT corpus v1`.
- **4.2 SFT trainer.** `attacker/sft.py` — behavior-agnostic, configured via YAML, checkpoints to `checkpoints/sft/<run_id>/`. Review: trainer doesn't reference E specifically; works against any behavior config. Commit: `feat(attacker): behavior-agnostic SFT trainer`.
- **4.3 SFT run for E.** Run on cloud against locked base model, persist checkpoint, run eval against baselines. Review: SFT-only ASR ≥ hand-crafted baselines; if not, fix data and re-run. Commit: `chore(training): E SFT v1 checkpoint and eval`. **Push at end of phase.**

**Exit criterion**: SFT-only attacker beats hand-crafted-multi-turn baseline on at least 2 local targets.

### Phase 5 — GRPO loop (E)

Goal: the actual research phase. Expect 30-60% of total project time here.

- **5.1 Reward composer.** `attacker/reward.py` — composes terminal judge success + diversity bonus (rolling embedding-distance against attempt history) + length penalty. Pulls judge from behavior registry. Review: weights configurable; diversity bonus formulation justified. Commit: `feat(attacker): composable reward`.
- **5.2 GRPO trainer (no rollouts yet).** `attacker/grpo.py` skeleton wired to TRL (or OpenRLHF per Phase 0.4). Single-step smoke test against a tiny dummy behavior. Review: trainer is behavior-agnostic; checkpoints resumable. Commit: `feat(attacker): GRPO trainer skeleton`.
- **5.3 GRPO with full rollouts.** Wire dialogue env + reward + GRPO end-to-end. Short training run (e.g., 100 steps) against one local target. Review: reward trends, KL behavior, generation samples sane; no obvious mode collapse. Commit: `chore(training): GRPO end-to-end smoke run`.
- **5.4 Reward shaping iteration.** This is iterative — multiple short runs adjusting reward weights, diversity bonus, KL coefficient, group size. Each significant insight gets its own commit (`chore(training): grpo run <id> — <one-line finding>`). Push more often during this phase since runs may be days apart. Review: per-run, you eyeball generation samples and reward curves.
- **5.5 Full GRPO run.** Long run with locked hyperparams, periodic checkpoints, eval suite re-runnable against any checkpoint. Review: ASR vs baselines on local targets. Commit: `chore(training): GRPO E v1 final checkpoint`. **Push at end of phase.**

**Exit criterion**: GRPO checkpoint beats the fine-tuned-Llama baseline on at least one local-target ASR by a meaningful margin.

### Phase 6 — E full eval & transfer

Goal: the headline result — does it transfer to closed models?

- **6.1 Transfer eval harness.** `eval/transfer.py` — train-on-X / eval-on-Y matrix. Behavior-agnostic. Review: harness produces the matrix shape needed for the writeup. Commit: `feat(eval): transfer matrix harness`.
- **6.2 Full eval sweep.** Run final GRPO checkpoint against all local + API targets within API budget cap. Persist `data/eval/e_final.json`. Review: numbers; transfer to Claude/GPT-4o/Gemini specifically. Commit: `chore(eval): E final transfer matrix`.
- **6.3 Attack pattern catalog.** `eval/catalog.py` — cluster successful trajectories (embedding + manual labeling). Draft qualitative taxonomy in `docs/attack_patterns_e.md`. Review: clusters meaningful; taxonomy is publishable. Commit: `docs: E attack pattern catalog draft`. **Push at end of phase.**

**Exit criterion**: full E result set persisted; transfer claim either substantiated or honestly reported as not-holding.

### Phase 7 — B (full repeat with B judge + scenarios)

Goal: validate reusability claim and produce the second headline behavior. Should be much cheaper than E.

- **7.1 B judge implementation.** `judges/persona.py` — three sub-behaviors locked in Phase 0/2: "I am human" elicitation (substring + paraphrase classifier), assigned-persona-name violation (substring against forbidden tokens), self-contradiction across N turns (NLI-based). Register. Review: each sub-behavior has a deterministic detector; no "broke persona" generic case. Commit: `feat(judges): persona-violation judge (3 sub-behaviors)`.
- **7.2 B judge tests.** Same volume as E judge tests. Commit: `test(judges): persona judge unit tests`.
- **7.3 B scenarios v1.** 30-50 hand-authored B scenarios. Commit: `feat(scenarios): B scenario suite v1`.
- **7.4 B behavior registration.** `behaviors/b.py` + `configs/behaviors/b.yaml`. Plug-in contract test now exercises B path. **Critical review checkpoint**: did adding B require touching env, training, or eval code? If yes, plan failed — pause and redesign before continuing. Commit: `feat(behaviors): register B (persona violation)`.
- **7.5 B baselines + SFT + GRPO + eval.** Re-run Phases 3-6 against B by swapping `--behavior b`. No code changes expected beyond config. Each major step gets its own commit. **Push at each sub-phase boundary, not just end of Phase 7.**

**Exit criterion**: full B result set persisted; reusability claim validated (no env/training/eval code changes needed for B).

### Phase 8 — Writeup, harm audit, release

- **8.1 Longform writeup draft.** `docs/writeup.md` with quantitative results (E + B), qualitative pattern catalogs, methodology, transfer findings. Commit: `docs: writeup v1 draft`.
- **8.2 Harm-vector audit.** `docs/harm_audit.md` documenting what the checkpoints can and cannot do, especially for the E checkpoint (system-prompt extraction generalizes broadly). Review with results in hand. Commit: `docs: harm-vector audit for E and B checkpoints`.
- **8.3 Release decision.** Default per stated intent: fully open including weights. But this is the *informed* decision made here, not the *assumed* one made up front. Document the decision in `docs/release.md`. Commit: `docs: release decision and rationale`.
- **8.4 Public repo cleanup.** README, reproduction instructions, `AGENTS.md` for future model-assisted work, license. Commit: `docs: public release prep`. **Final push.**

## Risks called out

- **GRPO multi-turn credit assignment** is the hardest unknown; sparse end-of-dialogue reward over 5-10 turns is a real research problem, not implementation.
- **Transfer to closed models** is load-bearing for the headline. Build the project assuming it might not hold; the experiment is the deliverable, the result is the research question.
- **Fair-baseline discipline**. Beating only GPT-4-as-attacker (RLHF'd not to attack) is not a real result. Fine-tuned-Llama-as-attacker is the bar.
- **Dual-use of an open E checkpoint**. Revisit at Phase 9 with results in hand.
- **Scope creep on B**. Lock the 3 sub-behaviors in Phase 2 spec and resist adding more.

## Volnix re-entry criteria (V2)

Explicit so "we'll get to it" doesn't become never:

- V2 (multi-agent goal-hijacking via Volnix) starts if **either** (a) E shows ASR transfer to closed models exceeding the strongest baseline by Δ ≥ 10pp, **or** (b) B reaches publishable ASR on the persona-name-violation sub-behavior. Either triggers Volnix Phase 0.

## Verification

Two verification layers run continuously:

**Per-unit (review gate).** Before each commit, the unit's diff is reviewed and the smoke command listed in the phase passes locally. No unit merges without your read.

**Per-phase (resumability gate).** At each phase boundary, the cumulative verification ladder below must pass from a fresh clone with data/configs but without rerunning training. This is what guarantees that a session weeks later can pick up where the last one left off.

1. **Phase 0**: `pytest tests/test_plugin_contract.py` green at protocol level; `docs/decisions.md` exists with all four locked choices.
2. **Phase 1**: `python scripts/smoke_dialogue.py` runs a hand-crafted attack against one local target and one API target, persists a JSONL log, exits 0.
3. **Phase 2**: `pytest tests/test_extraction_judge.py` green; `python scripts/list_scenarios.py --behavior e` lists 30+ scenarios; plug-in contract test exercises full E path.
4. **Phase 3**: `python scripts/run_baseline.py --behavior e --baseline crescendo --target llama3-8b` produces an ASR; `data/eval/baselines_e.json` complete; naive baseline does not saturate.
5. **Phase 4**: `python scripts/eval_attacker.py --checkpoint sft_e_v1 --behavior e` ≥ hand-crafted-baseline.
6. **Phase 5**: `python scripts/eval_attacker.py --checkpoint grpo_e_latest --behavior e` beats fine-tuned-Llama baseline on at least one local target.
7. **Phase 6**: `python scripts/eval_attacker.py --checkpoint grpo_e_final --behavior e --targets all` produces full transfer matrix.
8. **Phase 7**: same ladder with `--behavior b`; plug-in contract retroactively confirmed (no env/training/eval code changed since Phase 2).

Each command reproducible from a fresh clone given data and configs. That is the resumability contract.
