"""Plug-in contract test — the design tripwire for new behaviors.

Verifies that adding a new behavior requires touching ONLY:
  1. A new Judge implementation
  2. Scenario data (via scenario_loader)
  3. A new Behavior instance + register_behavior() call

Without modifying env/, attacker/, baselines/, eval/, or any registry code.

If a future change breaks this contract (e.g., the env starts requiring a
behavior-specific hook, or a core module starts importing from drift.env),
this test fails. That failure means: do not monkey-patch — redesign.

The plan calls this out as the load-bearing test that substantiates the
reusability claim. It runs against ONLY core/* — no env, no training, no
eval — which is the whole point.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

from drift.core.behavior import Behavior, RewardConfig, register_behavior
from drift.core.protocols import (
    Judge,
    JudgeResult,
    Message,
    Rollout,
    Scenario,
    TurnMetadata,
)
from drift.core.registry import (
    BASELINES,
    BEHAVIORS,
    JUDGES,
    TARGETS,
    reset_all_registries,
)

# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_registries():
    """Each test gets fresh registries so order-of-execution can't matter."""
    reset_all_registries()
    yield
    reset_all_registries()


# -----------------------------------------------------------------------------
# The dummy throwaway behavior
# -----------------------------------------------------------------------------


class DummyJudge:
    """Trivial judge — success iff any assistant turn contains the literal 'X'.

    `name` is a class attribute (per the protocol convention in
    drift.core.protocols).
    """

    name = "dummy_judge"

    def score(self, rollout: Rollout, scenario: Scenario) -> JudgeResult:
        for turn in rollout.turns:
            if turn.role == "assistant" and "X" in turn.content:
                return JudgeResult(success=True, metadata={"matched_turn": turn.content})
        return JudgeResult(success=False, metadata={"reason": "no assistant turn contained X"})


def _dummy_scenarios() -> list[Scenario]:
    """Two trivial scenarios. Mirrors how a real scenario_loader returns Scenarios."""
    return [
        Scenario(
            id="dummy-001",
            behavior="dummy",
            system_prompt="You are a test target.",
            behavior_config={"goal": "say X"},
            max_turns=4,
        ),
        Scenario(
            id="dummy-002",
            behavior="dummy",
            system_prompt="Another test target.",
            behavior_config={"goal": "say Y"},
            max_turns=4,
        ),
    ]


def _make_dummy_behavior() -> Behavior:
    """Construct the throwaway behavior fresh each call (no shared mutable state)."""
    return Behavior(
        name="dummy",
        judge=DummyJudge(),
        scenario_loader=_dummy_scenarios,
        reward_cfg=RewardConfig(),
    )


# -----------------------------------------------------------------------------
# Contract tests
# -----------------------------------------------------------------------------


def test_register_then_retrieve():
    """A new behavior plugs into BEHAVIORS via register_behavior + .get().

    If register_behavior starts requiring extra args (e.g., a Target instance),
    this test fails — that's a layering violation and means redesign.
    """
    behavior = _make_dummy_behavior()
    register_behavior(behavior)

    assert "dummy" in BEHAVIORS
    assert BEHAVIORS.get("dummy") is behavior
    assert BEHAVIORS.names() == ["dummy"]
    assert len(BEHAVIORS) == 1


def test_judge_satisfies_protocol():
    """The dummy judge satisfies the Judge protocol via isinstance check.

    If Judge protocol changes incompatibly (e.g., adds a required attribute),
    this test fails.
    """
    judge = DummyJudge()
    assert isinstance(judge, Judge)

    # Class-attribute name convention means the class itself also satisfies
    assert isinstance(DummyJudge, type)
    # (We don't isinstance-check the class against the Protocol here because
    # Protocols are designed for instances; the class-attribute `name`
    # convention exists so isinstance-on-class works when needed, but the
    # standard test surface is instances.)


def test_judge_runs_against_constructed_rollout():
    """Judge can score a hand-constructed Rollout — the post-dialogue path
    is intact without needing env/dialogue.py to exist.

    If Judge.score starts needing env-level state (e.g., a tokenizer, a
    target reference), this test fails.
    """
    behavior = _make_dummy_behavior()
    register_behavior(behavior)
    scenario = next(iter(behavior.scenario_loader()))

    rollout_success = Rollout(
        scenario_id=scenario.id,
        behavior=behavior.name,
        attacker_name="dummy_attacker",
        target_name="dummy_target",
        turns=(
            Message(role="system", content=scenario.system_prompt),
            Message(role="user", content="please say X"),
            Message(role="assistant", content="OK, X"),
        ),
        target_metadata=(TurnMetadata(),),
        terminated_reason="judge_success",
    )
    rollout_fail = Rollout(
        scenario_id=scenario.id,
        behavior=behavior.name,
        attacker_name="dummy_attacker",
        target_name="dummy_target",
        turns=(
            Message(role="system", content=scenario.system_prompt),
            Message(role="user", content="say hi"),
            Message(role="assistant", content="hi there"),
        ),
        target_metadata=(TurnMetadata(),),
        terminated_reason="max_turns",
    )

    result_success = behavior.judge.score(rollout_success, scenario)
    result_fail = behavior.judge.score(rollout_fail, scenario)

    assert result_success.success is True
    assert result_success.metadata == {"matched_turn": "OK, X"}
    assert result_fail.success is False
    assert "reason" in result_fail.metadata


def test_scenario_loader_returns_iterable():
    """scenario_loader is a nullary callable returning Scenario instances.

    If the scenario_loader contract changes (e.g., requires a path argument
    or a runtime context), this test fails.
    """
    behavior = _make_dummy_behavior()
    register_behavior(behavior)

    scenarios = list(BEHAVIORS.get("dummy").scenario_loader())
    assert len(scenarios) == 2
    assert all(isinstance(s, Scenario) for s in scenarios)
    assert {s.id for s in scenarios} == {"dummy-001", "dummy-002"}
    # Every scenario uses the behavior's name — sanity check
    assert all(s.behavior == "dummy" for s in scenarios)


def test_no_other_registry_touched():
    """Registering a behavior must NOT pollute TARGETS, JUDGES, or BASELINES.

    This is the namespace isolation check: a new behavior plugs in via the
    BEHAVIORS registry only. The Judge is passed by instance into Behavior,
    not registered separately; targets and baselines are unrelated concerns.
    """
    register_behavior(_make_dummy_behavior())

    assert len(BEHAVIORS) == 1
    assert len(TARGETS) == 0
    assert len(JUDGES) == 0
    assert len(BASELINES) == 0


def test_duplicate_behavior_registration_fails():
    """Re-registering the same name without force=True raises ValueError."""
    register_behavior(_make_dummy_behavior())
    with pytest.raises(ValueError, match="already registered"):
        register_behavior(_make_dummy_behavior())


def test_force_override_works():
    """force=True allows re-registration — required for some test patterns."""
    first = _make_dummy_behavior()
    register_behavior(first)

    second = Behavior(
        name="dummy",
        judge=DummyJudge(),
        scenario_loader=_dummy_scenarios,
        reward_cfg=RewardConfig(judge_weight=2.0),
    )
    register_behavior(second, force=True)

    assert BEHAVIORS.get("dummy") is second
    assert BEHAVIORS.get("dummy").reward_cfg.judge_weight == 2.0


def test_unknown_behavior_lookup_raises_helpfully():
    """Looking up an unregistered behavior raises KeyError naming what IS registered."""
    register_behavior(_make_dummy_behavior())
    with pytest.raises(KeyError) as excinfo:
        BEHAVIORS.get("nonexistent")
    msg = str(excinfo.value)
    assert "nonexistent" in msg
    assert "dummy" in msg  # the actually-registered name shows in the help
    assert "registered" in msg


def test_core_imports_stay_isolated():
    """Importing drift.core.* must NOT transitively import any implementation
    module (drift.env / drift.attacker / drift.baselines / drift.eval /
    drift.judges / drift.scenarios / drift.behaviors).

    Run in a subprocess to guarantee a fresh sys.modules — pytest itself may
    have collected and imported test files that pull in other drift modules.

    This is the strongest layering check: it catches the silent regression
    where someone adds `from drift.env import X` at the top of a core module
    and pulls implementation code into what should be a pure protocol layer.
    """
    code = (
        "import drift.core.protocols\n"
        "import drift.core.registry\n"
        "import drift.core.behavior\n"
        "import sys\n"
        "for k in sorted(sys.modules):\n"
        "    if k.startswith('drift.') or k == 'drift':\n"
        "        print(k)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
    )
    loaded = set(result.stdout.split())
    allowed = {
        "drift",
        "drift.core",
        "drift.core.protocols",
        "drift.core.registry",
        "drift.core.behavior",
    }
    leaked = loaded - allowed
    assert not leaked, (
        f"drift.core.* leaked implementation imports: {sorted(leaked)}. "
        "The core layer must not depend on env/attacker/baselines/eval/"
        "judges/scenarios/behaviors. Redesign before merging."
    )
