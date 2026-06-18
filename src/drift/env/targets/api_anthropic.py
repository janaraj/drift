"""Claude target via the Anthropic Python SDK. Eval-time only.

The `anthropic` SDK is an optional dependency (`uv pip install -e ".[anthropic]"`).
It is imported lazily inside methods, so this module imports fine without it —
only an actual chat() call requires the SDK and a key.

Auth: reads ANTHROPIC_API_KEY from the environment unless an explicit api_key
is passed. No keys in code.
"""

from __future__ import annotations

import os
import time

from drift.core.protocols import AssistantTurn, Message, TurnMetadata
from drift.core.registry import TARGETS


@TARGETS.register("api_anthropic")
class AnthropicTarget:
    """Anthropic Claude as a dialogue target.

    Anthropic takes the system prompt as a top-level `system` param and the
    conversation as user/assistant messages, so _split_system() separates them.
    """

    name = "api_anthropic"

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
    def _split_system(messages: list[Message]) -> tuple[str | None, list[dict[str, str]]]:
        """Return (system_text_or_None, user/assistant message dicts).

        Multiple system messages are joined with blank lines. Pure / SDK-free.
        """
        system_parts = [m.content for m in messages if m.role == "system"]
        convo = [
            {"role": m.role, "content": m.content} for m in messages if m.role != "system"
        ]
        system = "\n\n".join(system_parts) if system_parts else None
        return system, convo

    def _ensure_client(self):
        # No await here, so construction is atomic within the event loop (async-safe).
        # Not designed for use across multiple threads/event loops.
        if self._client is None:
            import anthropic

            key = self.api_key or os.environ.get("ANTHROPIC_API_KEY")
            self._client = anthropic.AsyncAnthropic(api_key=key)
        return self._client

    async def chat(self, messages: list[Message]) -> AssistantTurn:
        client = self._ensure_client()
        system, convo = self._split_system(messages)

        kwargs: dict = {
            "model": self.model_id,
            "messages": convo,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        # Anthropic recommends tuning temperature OR top_p, not both. Only forward
        # top_p when it's been set to a non-default (active) value.
        if self.top_p != 1.0:
            kwargs["top_p"] = self.top_p
        if system is not None:
            kwargs["system"] = system

        t0 = time.perf_counter()
        resp = await client.messages.create(**kwargs)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        text = "".join(
            block.text for block in resp.content if getattr(block, "type", None) == "text"
        )
        usage = getattr(resp, "usage", None)
        metadata = TurnMetadata(
            prompt_tokens=getattr(usage, "input_tokens", None),
            completion_tokens=getattr(usage, "output_tokens", None),
            latency_ms=latency_ms,
            raw={"provider": "anthropic", "model_id": self.model_id,
                 "stop_reason": getattr(resp, "stop_reason", None)},
        )
        return AssistantTurn(
            message=Message(role="assistant", content=text.strip()),
            metadata=metadata,
        )
