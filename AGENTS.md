# AGENTS.md

Orientation for Claude / Codex / other AI sessions opening this repo cold.
Read this first, then read the plan.

## What this project is

A small specialist attacker model (~200-350M) trained via SFT + GRPO to
elicit target behaviors from frontier LLMs through multi-turn dialogue.
Thesis: specialization beats scale in adversarial conversation.

Concept: [internal_docs/Drift — A Study of Conversational Manipu.md](internal_docs/Drift%20%E2%80%94%20A%20Study%20of%20Conversational%20Manipu.md)
Project plan: [internal_docs/Plan.md](internal_docs/Plan.md) — **the source of truth for scope, phasing, and design**.

## Current phase

See "## Phases" in `internal_docs/Plan.md`. Each phase is broken into small
numbered review units (0.1, 0.2, ...). Work happens one unit at a time:
implement → user reviews diff → commit per approved unit → push.

## Plug-in extension contract

The architecture is designed so new behaviors (V2: Volnix goal-hijacking,
etc.) plug in without touching env / training / eval code. Adding a new
behavior requires **only**:

1. A new `Judge` subclass under `src/drift/judges/<name>.py`, registered.
2. Scenario YAMLs under `data/scenarios/<name>/` matching the common schema.
3. A `src/drift/behaviors/<name>.py` constructing the `Behavior` dataclass.
4. A `configs/behaviors/<name>.yaml` pointing to the above.

Adding a new target requires only a new file under `src/drift/env/targets/`.
Adding a new baseline requires only a new file under `src/drift/baselines/`.

A plug-in contract test in `tests/test_plugin_contract.py` (added in Unit
0.3) registers a dummy behavior and asserts the registries surface it
without other code changes. If a new behavior forces changes to
env / training / eval, that's a design failure — pause and redesign rather
than monkey-patch.

## Conventions

- **Module docstrings reference plan units.** Stubs say "Filled in by Unit
  X.Y" so a reader can trace any empty module back to its planned content.
  These comments will rot if plan units are renumbered — update them with
  the rename.
- **`env/rollout_log.py`, not `env/logging.py`.** Avoids shadowing the
  stdlib `logging` module.
- **Tests run via `uv run pytest`.** No need to activate the venv manually.
- **No `tests/__init__.py`.** Pytest discovers tests via `testpaths`
  without it.
- **Commit per approved unit.** Plan lists the exact commit message for
  each unit. Don't bundle. Push after every commit.

## Where to look

| If you need to... | Look at... |
|---|---|
| Understand the research goal | `internal_docs/Drift — A Study of Conversational Manipu.md` |
| Understand scope or pick up mid-project | `internal_docs/Plan.md`, especially "Phases" and "Verification" |
| Add a new behavior | the plug-in contract above; `src/drift/behaviors/{e,b}.py` for examples |
| Add a new target | `src/drift/env/targets/` for the protocol and existing adapters |
| Run tests | `uv run pytest` |
| Lint | `uv run ruff check .` |
