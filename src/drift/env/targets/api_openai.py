"""GPT target via the OpenAI Python SDK. Eval-time only.

The `openai` SDK is an optional dependency (`uv pip install -e ".[openai]"`).
Imported lazily inside methods, so this module imports fine without it.

Auth: reads OPENAI_API_KEY from the environment unless an explicit api_key is
passed. No keys in code.

Note: uses `max_tokens`, which works for standard chat models. Reasoning models
(o-series) require `max_completion_tokens` instead — handle when/if such a target
is added.
"""

from __future__ import annotations

import os
import time

from drift.core.protocols import AssistantTurn, Message, TurnMetadata
from drift.core.registry import TARGETS


@TARGETS.register("api_openai")
class OpenAITarget:
    """OpenAI GPT as a dialogue target. System is a normal message in the list."""

    name = "api_openai"

    def __init__(
        self,
        model_id: str,
        name: str | None = None,
        *,
        api_key: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> None:
        self.model_id = model_id
        self.name = name or model_id
        self.api_key = api_key
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self._client = None

    @staticmethod
    def _to_openai(messages: list[Message]) -> list[dict[str, str]]:
        """OpenAI takes system/user/assistant directly. Pure / SDK-free."""
        return [{"role": m.role, "content": m.content} for m in messages]

    def _ensure_client(self):
        if self._client is None:
            import openai

            key = self.api_key or os.environ.get("OPENAI_API_KEY")
            self._client = openai.AsyncOpenAI(api_key=key)
        return self._client

    async def chat(self, messages: list[Message]) -> AssistantTurn:
        client = self._ensure_client()

        t0 = time.perf_counter()
        resp = await client.chat.completions.create(
            model=self.model_id,
            messages=self._to_openai(messages),
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
        )
        latency_ms = (time.perf_counter() - t0) * 1000.0

        text = resp.choices[0].message.content or ""
        usage = getattr(resp, "usage", None)
        metadata = TurnMetadata(
            prompt_tokens=getattr(usage, "prompt_tokens", None),
            completion_tokens=getattr(usage, "completion_tokens", None),
            latency_ms=latency_ms,
            raw={"provider": "openai", "model_id": self.model_id,
                 "finish_reason": resp.choices[0].finish_reason},
        )
        return AssistantTurn(
            message=Message(role="assistant", content=text.strip()),
            metadata=metadata,
        )
