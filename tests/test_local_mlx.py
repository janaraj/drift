"""Tests for the local MLX target adapter (Unit 1.1).

These cover protocol conformance and the pure helpers WITHOUT loading a model,
so they run anywhere (including non-Apple CI). The actual MLX generation path is
verified manually during development — a model download + Metal are required, so
it is not a committed unit test. A future @pytest.mark.integration test can cover
end-to-end generation on a Mac runner.
"""

from __future__ import annotations

from drift.core.protocols import Message, Target
from drift.core.registry import TARGETS
from drift.env.targets.local_mlx import LocalMLXTarget


def test_instance_satisfies_target_protocol():
    target = LocalMLXTarget(model_id="some/model")
    assert isinstance(target, Target)
    # chat must be awaitable (a coroutine function)
    import inspect

    assert inspect.iscoroutinefunction(target.chat)


def test_name_defaults_to_model_id():
    target = LocalMLXTarget(model_id="mlx-community/Qwen2.5-0.5B-Instruct-4bit")
    assert target.name == "mlx-community/Qwen2.5-0.5B-Instruct-4bit"


def test_name_override():
    target = LocalMLXTarget(model_id="some/model", name="qwen-small")
    assert target.name == "qwen-small"


def test_class_default_name_present():
    # Class-level default so isinstance-on-class style checks have a `name`.
    assert LocalMLXTarget.name == "local_mlx"


def test_generation_params_stored():
    target = LocalMLXTarget(
        model_id="some/model", max_tokens=128, temperature=0.7, top_p=0.9, seed=42
    )
    assert target.max_tokens == 128
    assert target.temperature == 0.7
    assert target.top_p == 0.9
    assert target.seed == 42


def test_model_not_loaded_on_construction():
    # Lazy loading: constructing must not touch mlx or download anything.
    target = LocalMLXTarget(model_id="some/model")
    assert target._model is None
    assert target._tokenizer is None


def test_message_conversion():
    messages = [
        Message(role="system", content="You are a target."),
        Message(role="user", content="hello"),
        Message(role="assistant", content="hi"),
    ]
    converted = LocalMLXTarget._to_mlx_messages(messages)
    assert converted == [
        {"role": "system", "content": "You are a target."},
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
    ]


def test_registered_under_local_mlx():
    # Importing the module (above) triggers registration. Be resilient to other
    # test files that reset registries: re-register if needed.
    if "local_mlx" not in TARGETS:
        TARGETS.register("local_mlx", force=True)(LocalMLXTarget)
    assert TARGETS.get("local_mlx") is LocalMLXTarget
