"""Composable reward function.

reward = judge_success (terminal binary) + diversity_bonus + length_penalty

Diversity bonus is computed against a rolling embedding-distance history
to prevent mode collapse. Weights are configurable per behavior.

Filled in by Unit 5.1.
"""
