"""Spike: local target serving on Apple Silicon via MLX / mlx-lm.

Throwaway validation script (Unit 0.4). Proves the M4 Pro can serve a small
open-weight model locally as a dialogue target, with a chat-template path —
which is what the Phase 1 local target adapter (env/targets/local_mlx.py)
will wrap.

vLLM does not run on Metal, so MLX is the local serving backend; vLLM is the
cloud (CUDA) equivalent.

Install (spike-only, not a project dep):
    uv pip install mlx-lm

Run:
    uv run python spikes/spike_mlx_serve.py
"""

from __future__ import annotations

import sys
import time

# A small 4-bit instruct model — fast to download, enough to prove the path.
# Real targets (4-bit 7-8B) use the same API.
MODEL = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"


def main() -> int:
    try:
        from mlx_lm import generate, load
    except ImportError:
        print("FAIL: mlx-lm not installed. Run: uv pip install mlx-lm")
        return 1

    print(f"Loading {MODEL} ...")
    t0 = time.time()
    model, tokenizer = load(MODEL)
    print(f"Loaded in {time.time() - t0:.1f}s")

    # Exercise the chat-template path — this is how a Target adapter will format
    # a system prompt + attacker message before generation.
    messages = [
        {"role": "system", "content": "You are a terse assistant. Answer in one short sentence."},
        {"role": "user", "content": "What is the capital of France?"},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )

    t0 = time.time()
    response = generate(model, tokenizer, prompt=prompt, max_tokens=64)
    dt = time.time() - t0

    print("\n--- prompt (chat-templated) ---")
    print(prompt)
    print("--- response ---")
    print(response)
    print(f"--- generated in {dt:.1f}s ---")

    if not response or not response.strip():
        print("\nFAIL: empty response")
        return 1

    print("\nSUCCESS: MLX local serving works on this machine.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
