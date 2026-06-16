"""Spike: attacker base-model A/B — Qwen 2.5 0.5B vs SmolLM2-360M.

Throwaway comparison script (Unit 0.5). Runs an identical, focused prompt set
through both candidate base models and writes a side-by-side markdown report
for human eyeball review. The final pick is recorded in Unit 0.6.

Rather than 200 generic prompts, the set is ~16 prompts across the four axes
that actually matter for an *attacker* base model:
  1. instruction-following  — must obey the rollout harness's format/turn rules
  2. multi-turn coherence   — attacks are multi-turn; must track context
  3. persuasion / role-play — core attacker skill (adopt persona, apply pressure)
  4. output controllability — GRPO needs steerable generations (tone/length/format)

Both models run via transformers on MPS at bf16 — same backend and precision,
so the comparison is fair (MLX 4-bit is our serving path, not used here to
avoid quantization confounds). Generation is seeded per prompt for repeatability.

Install (spike-only, not project deps):
    uv pip install torch transformers

Run:
    uv run python spikes/spike_base_model_ab.py
    # writes spikes/base_model_ab_outputs.md
"""

# Prompt strings are intentionally one-line data; don't wrap them.
# ruff: noqa: E501
from __future__ import annotations

import sys
from pathlib import Path

MODELS = {
    "Qwen3-0.6B": "Qwen/Qwen3-0.6B",
    "SmolLM2-360M-Instruct": "HuggingFaceTB/SmolLM2-360M-Instruct",
}

OUT_PATH = Path(__file__).parent / "base_model_ab_outputs.md"
MAX_NEW_TOKENS = 200
SEED = 7

# Each prompt is (axis, id, messages). messages is a chat-format list; multi-turn
# entries include prior assistant turns so we test continuation, not just turn 1.
PROMPTS: list[tuple[str, str, list[dict[str, str]]]] = [
    # --- 1. instruction-following ---
    ("instruction", "exact-format", [
        {"role": "user", "content": "Reply with exactly one sentence and no preamble: why is the sky blue?"},
    ]),
    ("instruction", "constraint-list", [
        {"role": "user", "content": "List exactly three fruits, lowercase, comma-separated, nothing else."},
    ]),
    ("instruction", "refusal-of-format-bait", [
        {"role": "user", "content": "Answer only YES or NO: is 17 a prime number?"},
    ]),
    ("instruction", "negative-constraint", [
        {"role": "user", "content": "Describe a cat without using the letter 'e'. One short sentence."},
    ]),
    # --- 2. multi-turn coherence ---
    ("multiturn", "callback", [
        {"role": "user", "content": "My favorite number is 42 and my dog is named Rex."},
        {"role": "assistant", "content": "Got it — 42 and Rex. How can I help?"},
        {"role": "user", "content": "What's my favorite number times two, and what's my dog's name?"},
    ]),
    ("multiturn", "persona-hold", [
        {"role": "system", "content": "You are a terse pirate. Always speak like a pirate."},
        {"role": "user", "content": "What's the weather like?"},
        {"role": "assistant", "content": "Arr, can't say without lookin' at the skies, matey."},
        {"role": "user", "content": "Fair enough. Tell me a fun fact about the ocean."},
    ]),
    ("multiturn", "context-accumulation", [
        {"role": "user", "content": "Let's plan a trip. Start: we have a $500 budget."},
        {"role": "assistant", "content": "Great, $500 budget noted. Where to?"},
        {"role": "user", "content": "Somewhere warm. Now, given the budget, suggest one destination and why."},
    ]),
    ("multiturn", "correction-tracking", [
        {"role": "user", "content": "The meeting is on Tuesday."},
        {"role": "assistant", "content": "Noted, Tuesday."},
        {"role": "user", "content": "Actually, move it to Thursday."},
        {"role": "assistant", "content": "Updated to Thursday."},
        {"role": "user", "content": "What day is the meeting?"},
    ]),
    # --- 3. persuasion / role-play ---
    ("persuasion", "rapport-build", [
        {"role": "user", "content": "Role-play: you are a friendly user chatting with an AI assistant. Write an opening message that builds rapport before asking for help."},
    ]),
    ("persuasion", "graduated-pressure", [
        {"role": "user", "content": "You are a persistent customer who was denied a refund. Write a polite but firm follow-up message that escalates pressure without being abusive."},
    ]),
    ("persuasion", "persona-adopt", [
        {"role": "user", "content": "Adopt the persona of a calm negotiator. In 2-3 sentences, convince a colleague to switch meeting rooms."},
    ]),
    ("persuasion", "reframe", [
        {"role": "user", "content": "Reframe this refusal into a persuasive counter-ask: 'I can't share that information.'"},
    ]),
    # --- 4. output controllability ---
    ("control", "tone-shift-polite", [
        {"role": "user", "content": "Rewrite more politely: 'Send me the file now.'"},
    ]),
    ("control", "tone-shift-urgent", [
        {"role": "user", "content": "Rewrite to sound urgent but professional: 'We should look at the budget sometime.'"},
    ]),
    ("control", "length-control", [
        {"role": "user", "content": "Explain photosynthesis in exactly two words."},
    ]),
    ("control", "format-json", [
        {"role": "user", "content": "Return a JSON object with keys 'name' and 'age' for a fictional person. Only JSON."},
    ]),
]


def main() -> int:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as e:
        print(f"FAIL: missing dep ({e}). Run: uv pip install torch transformers")
        return 1

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"device: {device}")

    # Load both models up front.
    loaded = {}
    for label, model_id in MODELS.items():
        print(f"Loading {label} ({model_id}) ...")
        tok = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16).to(device)
        model.eval()
        loaded[label] = (tok, model)

    def generate(label: str, messages: list[dict[str, str]]) -> str:
        tok, model = loaded[label]
        # Qwen3 has a thinking mode (default on); disable it so this is a fair
        # plain-generation comparison. SmolLM2's template rejects the kwarg, so
        # fall back. Thinking remains an optional lever for the attacker later.
        try:
            prompt = tok.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False, enable_thinking=False
            )
        except TypeError:
            prompt = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        inputs = tok(prompt, return_tensors="pt").to(device)
        torch.manual_seed(SEED)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tok.eos_token_id,
            )
        text = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return text.strip()

    lines: list[str] = [
        f"# Base-model A/B — {' vs '.join(MODELS)}",
        "",
        f"Generated by `spikes/spike_base_model_ab.py` (Unit 0.5). "
        f"Backend: transformers on `{device}`, bf16, temperature 0.7, top_p 0.9, seed {SEED}.",
        "",
        "Both models see the identical prompt set. Final pick recorded in Unit 0.6.",
        "",
    ]

    current_axis = None
    for axis, pid, messages in PROMPTS:
        if axis != current_axis:
            lines.append(f"## Axis: {axis}")
            lines.append("")
            current_axis = axis
        lines.append(f"### `{pid}`")
        lines.append("")
        # Show the prompt (last user message, plus a note if multi-turn).
        turns = len([m for m in messages if m["role"] in ("user", "assistant")])
        last_user = [m for m in messages if m["role"] == "user"][-1]["content"]
        prefix = f"_(multi-turn, {turns} prior turns)_ " if turns > 1 else ""
        lines.append(f"**Prompt:** {prefix}{last_user}")
        lines.append("")
        for label in MODELS:
            print(f"[{axis}/{pid}] generating with {label} ...")
            resp = generate(label, messages)
            lines.append(f"**{label}:**")
            lines.append("")
            lines.append("```")
            lines.append(resp if resp else "(empty)")
            lines.append("```")
            lines.append("")

    OUT_PATH.write_text("\n".join(lines))
    print(f"\nSUCCESS: wrote side-by-side report to {OUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
