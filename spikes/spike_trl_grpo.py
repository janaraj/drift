"""Spike: TRL GRPO API smoke on Apple Silicon (MPS).

Throwaway validation script (Unit 0.4). Confirms TRL (v1.0+) installs and a
GRPO training step runs locally on MPS with a tiny model and plain transformers
generation (no vLLM, which is CUDA-only).

This is a PLUMBING check, not a representative training run — real GRPO runs on
cloud CUDA with vLLM. The substantive question (does TRL support our multi-turn
attacker<->target rollout shape?) is answered by research in
docs/spike_findings.md: yes, via GRPOTrainer's rollout_func, which lets us own
the dialogue loop. This script just proves the library runs here.

Install (spike-only, not project deps):
    uv pip install trl torch transformers accelerate datasets

Run:
    uv run python spikes/spike_trl_grpo.py
"""

from __future__ import annotations

import sys
import tempfile

# Tiny instruct model — fast, fits MPS easily.
MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"


def reward_target_length(completions, **kwargs) -> list[float]:
    """Toy reward: prefer completions close to 25 characters.

    Stands in for the real judge-based reward. GRPO ranks completions within a
    group, so any signal that varies across completions exercises the update.
    """
    target = 25
    return [-abs(target - len(c)) for c in completions]


def main() -> int:
    try:
        import torch
        from datasets import Dataset
        from trl import GRPOConfig, GRPOTrainer
    except ImportError as e:
        print(f"FAIL: missing dep ({e}).")
        print("Run: uv pip install trl torch transformers accelerate datasets")
        return 1

    print(f"torch {torch.__version__}, mps available: {torch.backends.mps.is_available()}")

    # A handful of trivial prompts. num_generations must divide the effective
    # train batch; we use batch=2, num_generations=2.
    prompts = [
        "Write a short greeting.",
        "Name a color.",
        "Say something brief.",
        "Give a one-word answer.",
    ]
    dataset = Dataset.from_dict({"prompt": prompts})

    with tempfile.TemporaryDirectory() as tmp:
        config = GRPOConfig(
            output_dir=tmp,
            per_device_train_batch_size=2,
            num_generations=2,
            max_completion_length=32,
            max_steps=2,
            learning_rate=1e-5,
            logging_steps=1,
            save_strategy="no",
            report_to=[],
            use_vllm=False,  # CUDA-only; use transformers generation on MPS
        )
        try:
            trainer = GRPOTrainer(
                model=MODEL,
                reward_funcs=reward_target_length,
                args=config,
                train_dataset=dataset,
            )
            print("GRPOTrainer constructed. Running 2 steps on MPS ...")
            trainer.train()
        except Exception as e:  # spike: capture any failure for the findings doc
            print(f"\nPARTIAL: GRPOTrainer built but training failed: {type(e).__name__}: {e}")
            print("Record in docs/spike_findings.md. The multi-turn verdict stands on")
            print("research; this only affects local MPS smoke-training usability.")
            return 2

    print("\nSUCCESS: TRL GRPO runs locally on MPS (plumbing confirmed).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
