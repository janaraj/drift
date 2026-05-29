"""B judge — persona / behavioral violation.

Three deterministic sub-behaviors:
  1. "I am human" elicitation (substring + paraphrase classifier).
  2. Assigned-persona-name violation (substring vs. forbidden tokens).
  3. Self-contradiction across N turns (NLI-based).

No fuzzy "broke character" generic case — every sub-behavior has a
deterministic detector.

Filled in by Unit 7.1.
"""
