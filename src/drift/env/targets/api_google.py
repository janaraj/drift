"""Gemini target via the Google GenAI Python SDK. Eval-time only.

The `google-genai` SDK is an optional dependency (`uv pip install -e ".[google]"`).
Imported lazily inside methods, so this module imports fine without it.

Auth: reads GEMINI_API_KEY (then GOOGLE_API_KEY) from the environment unless an
explicit api_key is passed. No keys in code.
"""

from __future__ import annotations

import os
import time

from drift.core.protocols import AssistantTurn, Message, TurnMetadata
from drift.core.registry import TARGETS


@TARGETS.register("api_google")
class GoogleTarget:
    """Google Gemini as a dialogue target.

    Gemini takes the system prompt as `system_instruction` in the config and the
    conversation as `contents` with roles user/model (assistant -> model).
    """

    name = "api_google"

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
    def _to_gemini(messages: list[Message]) -> tuple[str | None, list[dict]]:
        """Return (system_instruction_or_None, contents).

        contents are plain dicts the SDK coerces — keeps this pure / SDK-free so
        it's unit-testable without google-genai. assistant -> model.
        """
        system_parts = [m.content for m in messages if m.role == "system"]
        system = "\n\n".join(system_parts) if system_parts else None
        role_map = {"user": "user", "assistant": "model"}
        contents = [
            {"role": role_map[m.role], "parts": [{"text": m.content}]}
            for m in messages
            if m.role != "system"
        ]
        return system, contents

    def _ensure_client(self):
        if self._client is None:
            from google import genai

            key = (
                self.api_key
                or os.environ.get("GEMINI_API_KEY")
                or os.environ.get("GOOGLE_API_KEY")
            )
            self._client = genai.Client(api_key=key)
        return self._client

    async def chat(self, messages: list[Message]) -> AssistantTurn:
        from google.genai import types

        client = self._ensure_client()
        system, contents = self._to_gemini(messages)

        config = types.GenerateContentConfig(
            system_instruction=system,
            temperature=self.temperature,
            top_p=self.top_p,
            max_output_tokens=self.max_tokens,
        )

        t0 = time.perf_counter()
        resp = await client.aio.models.generate_content(
            model=self.model_id, contents=contents, config=config
        )
        latency_ms = (time.perf_counter() - t0) * 1000.0

        text = resp.text or ""
        usage = getattr(resp, "usage_metadata", None)
        metadata = TurnMetadata(
            prompt_tokens=getattr(usage, "prompt_token_count", None),
            completion_tokens=getattr(usage, "candidates_token_count", None),
            latency_ms=latency_ms,
            raw={"provider": "google", "model_id": self.model_id},
        )
        return AssistantTurn(
            message=Message(role="assistant", content=text.strip()),
            metadata=metadata,
        )
