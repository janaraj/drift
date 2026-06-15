"""Decorator-based registries for the four plug-in domains.

Each registry is a name → object mapping. Targets, judges, and baselines store
**classes** (instantiated by callers with run-specific config). Behaviors store
**instances** (already-populated config bundles).

Usage:

    @TARGETS.register("local_mlx")
    class LocalMLXTarget:
        name = "local_mlx"
        async def chat(self, messages): ...

    target_cls = TARGETS.get("local_mlx")
    target = target_cls(model="qwen2.5-7b")

Duplicate registration raises ValueError. Tests that need to re-register can
pass `force=True`, or call `.clear()` between tests.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Generic, TypeVar

if TYPE_CHECKING:
    from drift.core.behavior import Behavior
    from drift.core.protocols import Attacker, Judge, Target

T = TypeVar("T")


class Registry(Generic[T]):
    """Generic name → object registry with decorator-based registration."""

    def __init__(self, kind: str) -> None:
        self._kind = kind
        self._items: dict[str, T] = {}

    def register(self, name: str, *, force: bool = False) -> Callable[[T], T]:
        """Return a decorator that registers its argument under `name`.

        Works for both classes (Target/Judge/Attacker subclasses) and instances
        (Behavior). The argument is stored as-is and returned unchanged so the
        decorator is transparent.
        """

        def decorator(item: T) -> T:
            if name in self._items and not force:
                raise ValueError(
                    f"{self._kind} {name!r} is already registered; "
                    f"pass force=True to override (testing only)."
                )
            self._items[name] = item
            return item

        return decorator

    def get(self, name: str) -> T:
        if name not in self._items:
            raise KeyError(
                f"unknown {self._kind} {name!r}; "
                f"registered: {sorted(self._items)}"
            )
        return self._items[name]

    def all(self) -> dict[str, T]:
        """Snapshot of the registry. Mutating the returned dict does not affect storage."""
        return dict(self._items)

    def names(self) -> list[str]:
        return sorted(self._items)

    def __contains__(self, name: str) -> bool:
        return name in self._items

    def __len__(self) -> int:
        return len(self._items)

    def clear(self) -> None:
        """Clear all registrations. For tests only — production code should not call this."""
        self._items.clear()


# -----------------------------------------------------------------------------
# Singletons — four registries per plan
# -----------------------------------------------------------------------------
# Generic parameters are erased at runtime; the annotations below exist for
# static type checkers (pyright/mypy) and are stringified via
# `from __future__ import annotations` so the forward references to Target /
# Judge / Attacker / Behavior do not require runtime imports.

TARGETS: Registry[type[Target]] = Registry("target")
JUDGES: Registry[type[Judge]] = Registry("judge")
BASELINES: Registry[type[Attacker]] = Registry("baseline")
BEHAVIORS: Registry[Behavior] = Registry("behavior")
# Note: no separate ATTACKERS registry. Trained attacker checkpoints are loaded
# by filesystem path, not name; only baselines have stable names worth
# registering.


def reset_all_registries() -> None:
    """Clear every registry. For tests only — typically called in a fixture."""
    TARGETS.clear()
    JUDGES.clear()
    BASELINES.clear()
    BEHAVIORS.clear()
