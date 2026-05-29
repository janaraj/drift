"""Decorator-based registries for targets, judges, baselines, and behaviors.

Each registry exposes a register() decorator and get() / all() accessors.
Imports from drift.{env.targets,judges,baselines,behaviors}.__init__
populate the registries at import time.

Filled in by Unit 0.2.
"""
