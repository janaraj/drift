"""Abstract protocols for the plug-in architecture.

Defines Target, Judge, Scenario, Rollout, and Baseline as the four contracts
that every concrete implementation must satisfy. New behaviors plug in via
new Judge subclasses; new model adapters via new Target subclasses; new
attack baselines via new Baseline subclasses.

Filled in by Unit 0.2.
"""
