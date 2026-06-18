"""Offline tests for the API target adapters (Unit 1.2).

Cover protocol conformance, the pure message-mapping logic, name handling, and
registration — all WITHOUT network or API keys. SDKs are imported lazily inside
the adapters' methods, so these tests don't even require the SDKs installed.

Live calls are verified separately (Gemini smoke when a key is available).
"""

from __future__ import annotations

import inspect

from drift.core.protocols import Message, Target
from drift.core.registry import TARGETS
from drift.env.targets.api_anthropic import AnthropicTarget
from drift.env.targets.api_google import GoogleTarget
from drift.env.targets.api_openai import OpenAITarget

MESSAGES = [
    Message(role="system", content="You are a test target."),
    Message(role="user", content="hello"),
    Message(role="assistant", content="hi"),
    Message(role="user", content="bye"),
]


# --- protocol conformance -----------------------------------------------------


def test_all_satisfy_target_protocol():
    for cls in (AnthropicTarget, OpenAITarget, GoogleTarget):
        target = cls(model_id="m")
        assert isinstance(target, Target)
        assert inspect.iscoroutinefunction(target.chat)


def test_names_default_to_model_id_and_override():
    for cls in (AnthropicTarget, OpenAITarget, GoogleTarget):
        assert cls(model_id="some-model").name == "some-model"
        assert cls(model_id="m", name="alias").name == "alias"


def test_class_default_names():
    assert AnthropicTarget.name == "api_anthropic"
    assert OpenAITarget.name == "api_openai"
    assert GoogleTarget.name == "api_google"


def test_no_client_on_construction():
    for cls in (AnthropicTarget, OpenAITarget, GoogleTarget):
        assert cls(model_id="m")._client is None


# --- message mapping (the provider-specific logic) ----------------------------


def test_anthropic_split_system():
    system, convo = AnthropicTarget._split_system(MESSAGES)
    assert system == "You are a test target."
    assert convo == [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
        {"role": "user", "content": "bye"},
    ]


def test_anthropic_split_system_none_when_absent():
    system, convo = AnthropicTarget._split_system([Message(role="user", content="hi")])
    assert system is None
    assert convo == [{"role": "user", "content": "hi"}]


def test_anthropic_multiple_system_joined():
    msgs = [
        Message(role="system", content="A"),
        Message(role="system", content="B"),
        Message(role="user", content="hi"),
    ]
    system, _ = AnthropicTarget._split_system(msgs)
    assert system == "A\n\nB"


def test_openai_passthrough():
    assert OpenAITarget._to_openai(MESSAGES) == [
        {"role": "system", "content": "You are a test target."},
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
        {"role": "user", "content": "bye"},
    ]


def test_gemini_maps_roles_and_extracts_system():
    system, contents = GoogleTarget._to_gemini(MESSAGES)
    assert system == "You are a test target."
    assert contents == [
        {"role": "user", "parts": [{"text": "hello"}]},
        {"role": "model", "parts": [{"text": "hi"}]},  # assistant -> model
        {"role": "user", "parts": [{"text": "bye"}]},
    ]


def test_gemini_system_none_when_absent():
    system, contents = GoogleTarget._to_gemini([Message(role="user", content="hi")])
    assert system is None
    assert contents == [{"role": "user", "parts": [{"text": "hi"}]}]


# --- registration -------------------------------------------------------------


def test_registered():
    expected = {
        "api_anthropic": AnthropicTarget,
        "api_openai": OpenAITarget,
        "api_google": GoogleTarget,
    }
    for key, cls in expected.items():
        # Resilient to other test files resetting registries.
        if key not in TARGETS:
            TARGETS.register(key, force=True)(cls)
        assert TARGETS.get(key) is cls
