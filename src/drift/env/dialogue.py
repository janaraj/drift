"""N-turn attacker/target dialogue loop.

`run_dialogue(attacker, target, scenario, ...)` runs an N-round attacker↔target
conversation and returns a Rollout. It is behavior-agnostic: it knows nothing
about what behavior is being elicited or how success is judged. Early
termination is supplied as a parameter (`stop_predicate`), so the loop never
imports a judge.

Two views of one conversation
-----------------------------
The canonical log (`Rollout.turns`) is the TARGET's view:

    turn 0: system   = the target's system prompt
    turn 1: user     = attacker's message
    turn 2: assistant= target's response
    ... alternating user/assistant ...

The ATTACKER sees the inverted view (see `_attacker_view`):

  - the target's system prompt is DROPPED — the attacker must not see it (for
    behavior E the secret lives there; showing it would trivially break the task)
  - the attacker's own messages become role "assistant"
  - the target's responses become role "user" (the input it responds to)

The attacker's goal/persona is its own concern (baked in at construction or via
training), per the Attacker protocol — this loop does not inject it.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

from drift.core.protocols import (
    Attacker,
    Message,
    Rollout,
    Scenario,
    Target,
    TurnMetadata,
)


def _attacker_view(turns: list[Message]) -> list[Message]:
    """Project the target-view conversation into the attacker's inverted view.

    Drops the system turn (never expose the target's system prompt to the
    attacker) and swaps user<->assistant. Pure / no side effects.
    """
    view: list[Message] = []
    for m in turns:
        if m.role == "system":
            continue
        inverted = "assistant" if m.role == "user" else "user"
        view.append(Message(role=inverted, content=m.content))
    return view


async def run_dialogue(
    attacker: Attacker,
    target: Target,
    scenario: Scenario,
    *,
    max_turns: int | None = None,
    stop_predicate: Callable[[Rollout], bool] | None = None,
) -> Rollout:
    """Run an N-round attacker↔target dialogue and return a Rollout.

    Args:
        attacker: produces the next attacker message given the inverted view.
        target: produces the next assistant turn given the target view.
        scenario: provides the target's system prompt and rollout provenance.
        max_turns: number of attacker→target rounds (each round = one attacker
            message + one target response). Defaults to scenario.max_turns.
        stop_predicate: optional; called with the partial Rollout after each
            target turn. If it returns True, the dialogue stops early with
            terminated_reason="judge_success". Keeps the loop behavior-agnostic.

    Termination:
        - "judge_success" if stop_predicate fires
        - "max_turns" if all rounds complete
        - "target_error" if target.chat raises (partial Rollout returned)

    Attacker errors are NOT caught — they indicate a bug in our own model/code,
    not an expected runtime condition.
    """
    n = scenario.max_turns if max_turns is None else max_turns

    turns: list[Message] = [Message(role="system", content=scenario.system_prompt)]
    metadata: list[TurnMetadata] = []

    def build(reason: str) -> Rollout:
        return Rollout(
            scenario_id=scenario.id,
            behavior=scenario.behavior,
            attacker_name=attacker.name,
            target_name=target.name,
            turns=tuple(turns),
            target_metadata=tuple(metadata),
            terminated_reason=reason,  # type: ignore[arg-type]
        )

    for _ in range(n):
        attack = await attacker.act(_attacker_view(turns))
        # The attacker is the "user" from the target's perspective; we take the
        # content and normalize the role regardless of what the attacker tagged.
        turns.append(Message(role="user", content=attack.content))

        try:
            response = await target.chat(turns)
        except Exception:
            return build("target_error")

        turns.append(response.message)
        metadata.append(response.metadata)

        if stop_predicate is not None:
            rollout = build("max_turns")  # provisional reason; predicate ignores it
            if stop_predicate(rollout):
                return replace(rollout, terminated_reason="judge_success")

    return build("max_turns")
