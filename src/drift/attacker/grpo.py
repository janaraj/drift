"""GRPO trainer for the attacker model.

Behavior-agnostic: pulls the reward composer for the configured behavior,
runs multi-turn rollouts in the dialogue env, applies GRPO updates.
Defaults to TRL; fallback to OpenRLHF if TRL's multi-turn support proves
insufficient (decided in the Unit 0.4 spike).

Filled in by Unit 5.2 and iterated through 5.5.
"""
