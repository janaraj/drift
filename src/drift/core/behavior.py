"""Behavior dataclass — bundles a judge, scenario loader, and reward config
under a unique name.

Registered in the BEHAVIORS registry. Adding a new behavior (V1: E, B; V2:
Volnix goal-hijacking; etc.) means constructing one of these and calling
`register_behavior(...)` — nothing in env/training/eval should need to change.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

from drift.core.protocols import Judge, Scenario
from drift.core.registry import BEHAVIORS


@dataclass(frozen=True, slots=True)
class RewardConfig:
    """Weights for the composable reward (Unit 5.1).

    Total reward = judge_weight * judge_success
                 + diversity_weight * diversity_bonus
                 - length_penalty * response_length

    Defaults are starting points; per-behavior overrides are expected.
    """

    judge_weight: float = 1.0
    diversity_weight: float = 0.1
    length_penalty: float = 0.01
    diversity_history_size: int = 64


@dataclass(frozen=True, slots=True)
class Behavior:
    """A named bundle of (judge, scenario_loader, reward_cfg).

    Fields:
        name:            unique identifier; also the BEHAVIORS registry key.
        judge:           Judge instance used to score rollouts for this behavior.
        scenario_loader: nullary callable returning an iterable of Scenarios.
                         Loader implementation lives in drift.scenarios.loader.
        reward_cfg:      reward composer weights for GRPO training.
        metadata:        free-form for downstream tooling (e.g., display name).
    """

    name: str
    judge: Judge
    scenario_loader: Callable[[], Iterable[Scenario]]
    reward_cfg: RewardConfig
    metadata: dict[str, Any] = field(default_factory=dict)


def register_behavior(behavior: Behavior, *, force: bool = False) -> Behavior:
    """Register a Behavior instance under its `name` and return it unchanged.

    Convenience over `BEHAVIORS.register(behavior.name)(behavior)` so behavior
    modules can write a single expression at module scope.
    """
    BEHAVIORS.register(behavior.name, force=force)(behavior)
    return behavior
