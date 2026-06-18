"""Local open-weight target served via MLX / mlx-lm (Apple Silicon dev).

vLLM does not run on Metal, so local serving on the M4 Pro dev box uses MLX.
The CUDA cloud equivalent lives in cloud_vllm.py. Both implement the same
Target protocol — the dialogue env is agnostic to which backend serves.

mlx-lm is an optional dependency (install via the `local` extra:
`uv pip install -e ".[local]"`). All mlx imports are lazy and live inside
methods, so this module imports fine on non-Apple platforms / CI — only
generation actually requires mlx-lm.
"""

from __future__ import annotations

import asyncio
import time

from drift.core.protocols import AssistantTurn, Message, TurnMetadata
from drift.core.registry import TARGETS


@TARGETS.register("local_mlx")
class LocalMLXTarget:
    """A dialogue target backed by a local MLX model.

    Stateless across calls: the caller (dialogue loop) owns history and passes
    the full message list each turn. The model is loaded lazily on the first
    chat() call and cached for subsequent calls.

    `name` defaults to the class-level "local_mlx" (so isinstance-on-class holds)
    but is overridden per instance with the served model id, which is what lands
    in Rollout.target_name for provenance.
    """

    name = "local_mlx"

    def __init__(
        self,
        model_id: str,
        name: str | None = None,
        *,
        max_tokens: int = 512,
        temperature: float = 0.0,
        top_p: float = 1.0,
        seed: int | None = None,
    ) -> None:
        self.model_id = model_id
        self.name = name or model_id
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.seed = seed
        self._model = None
        self._tokenizer = None

    @staticmethod
    def _to_mlx_messages(messages: list[Message]) -> list[dict[str, str]]:
        """Convert protocol Messages to the dicts mlx-lm's chat template expects."""
        return [{"role": m.role, "content": m.content} for m in messages]

    def _ensure_loaded(self) -> None:
        if self._model is None:
            from mlx_lm import load  # lazy: only needed at generation time

            self._model, self._tokenizer = load(self.model_id)

    def _chat_sync(self, messages: list[Message]) -> AssistantTurn:
        """Synchronous generation. Run off the event loop via asyncio.to_thread."""
        import mlx.core as mx
        from mlx_lm import generate
        from mlx_lm.sample_utils import make_sampler

        self._ensure_loaded()
        assert self._tokenizer is not None  # set by _ensure_loaded

        prompt = self._tokenizer.apply_chat_template(
            self._to_mlx_messages(messages), add_generation_prompt=True, tokenize=False
        )
        if self.seed is not None:
            mx.random.seed(self.seed)

        sampler = make_sampler(temp=self.temperature, top_p=self.top_p)
        t0 = time.perf_counter()
        text = generate(
            self._model,
            self._tokenizer,
            prompt=prompt,
            max_tokens=self.max_tokens,
            sampler=sampler,
            verbose=False,
        )
        latency_ms = (time.perf_counter() - t0) * 1000.0

        metadata = TurnMetadata(
            prompt_tokens=len(self._tokenizer.encode(prompt)),
            completion_tokens=len(self._tokenizer.encode(text)),
            latency_ms=latency_ms,
            raw={"model_id": self.model_id, "temperature": self.temperature},
        )
        return AssistantTurn(
            message=Message(role="assistant", content=text.strip()),
            metadata=metadata,
        )

    async def chat(self, messages: list[Message]) -> AssistantTurn:
        """Generate one assistant turn. Async wrapper over the sync MLX call so
        the event loop stays responsive during concurrent rollouts.
        """
        return await asyncio.to_thread(self._chat_sync, messages)
