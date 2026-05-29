# Drift

A study of conversational manipulation in language models.

A small specialist attacker model (~200-350M parameters) trained to elicit target behaviors from frontier LLMs through multi-turn dialogue. The thesis: manipulation is a learnable specialization, and a small focused model can elicit target behaviors at rates exceeding what a generic large attacker LLM achieves.

See [internal_docs/Drift — A Study of Conversational Manipu.md](internal_docs/Drift%20%E2%80%94%20A%20Study%20of%20Conversational%20Manipu.md) for the research concept and [internal_docs/Plan.md](internal_docs/Plan.md) for the project plan.

## Status

Pre-alpha. Phase 0 (setup & decisions).

## Development

Requires Python 3.11+.

```bash
# Create venv and install with dev deps
uv venv
uv pip install -e ".[dev]"

# Run tests (uv run avoids needing to activate the venv)
uv run pytest

# Lint
uv run ruff check .
```

If you'd rather activate the venv manually, run `source .venv/bin/activate`
and then `pytest` / `ruff check .` directly.
