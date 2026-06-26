"""Rollout persistence as JSONL under data/logs/<run_id>/.

Every dialogue lands on disk as one line in an append-only JSONL file, so work
survives interruption (bursty development) and rollouts can be replayed through
judges later. Given a run_id, completed scenarios can be listed so a resumed
sweep skips what's already done.

Layout:
    data/logs/<run_id>/rollouts.jsonl   # one serialized Rollout per line

data/logs/ is gitignored — these are run artifacts, not committed.
"""

from __future__ import annotations

import json
import threading
from datetime import UTC, datetime
from pathlib import Path

from drift.core.protocols import Message, Rollout, TurnMetadata

SCHEMA_VERSION = 1
ROLLOUTS_FILE = "rollouts.jsonl"


# -----------------------------------------------------------------------------
# (De)serialization — pure, lossless
# -----------------------------------------------------------------------------


def rollout_to_dict(rollout: Rollout) -> dict:
    """Serialize a Rollout to a JSON-able dict (lossless)."""
    return {
        "schema_version": SCHEMA_VERSION,
        "scenario_id": rollout.scenario_id,
        "behavior": rollout.behavior,
        "attacker_name": rollout.attacker_name,
        "target_name": rollout.target_name,
        "terminated_reason": rollout.terminated_reason,
        "turns": [{"role": m.role, "content": m.content} for m in rollout.turns],
        "target_metadata": [
            {
                "prompt_tokens": md.prompt_tokens,
                "completion_tokens": md.completion_tokens,
                "latency_ms": md.latency_ms,
                "raw": md.raw,
            }
            for md in rollout.target_metadata
        ],
    }


def rollout_from_dict(d: dict) -> Rollout:
    """Reconstruct a Rollout from a dict produced by rollout_to_dict.

    Unknown keys (e.g. schema_version, future fields) are ignored.
    """
    turns = tuple(Message(role=t["role"], content=t["content"]) for t in d["turns"])
    target_metadata = tuple(
        TurnMetadata(
            prompt_tokens=md.get("prompt_tokens"),
            completion_tokens=md.get("completion_tokens"),
            latency_ms=md.get("latency_ms"),
            raw=md.get("raw", {}),
        )
        for md in d["target_metadata"]
    )
    return Rollout(
        scenario_id=d["scenario_id"],
        behavior=d["behavior"],
        attacker_name=d["attacker_name"],
        target_name=d["target_name"],
        turns=turns,
        target_metadata=target_metadata,
        terminated_reason=d["terminated_reason"],
    )


def _validate_run_id(run_id: str) -> None:
    """Reject run_ids that would escape or collapse the run directory.

    run_id is interpolated into a filesystem path, so guard against empty values,
    path separators, and parent refs even though it's normally our-code-controlled.
    """
    if not run_id or not run_id.strip():
        raise ValueError("run_id must be a non-empty string")
    if "/" in run_id or "\\" in run_id or run_id in (".", ".."):
        raise ValueError(f"run_id must not contain path separators or be a path ref: {run_id!r}")


def default_run_id(prefix: str = "run") -> str:
    """A timestamped run id, e.g. 'run-20260615T142233Z'. Convenience for scripts."""
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{prefix}-{stamp}"


# -----------------------------------------------------------------------------
# Logger
# -----------------------------------------------------------------------------


class RolloutLogger:
    """Append-only JSONL logger for a single run, with resumability helpers.

    Designed for single-process use; writes are line-atomic and guarded by a
    lock so concurrent callers (incl. threads) don't interleave lines.
    """

    def __init__(self, run_id: str, base_dir: str | Path = "data/logs") -> None:
        _validate_run_id(run_id)
        self.run_id = run_id
        self.run_dir = Path(base_dir) / run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.run_dir / ROLLOUTS_FILE
        self._lock = threading.Lock()

    def log(self, rollout: Rollout) -> None:
        """Append one rollout as a JSONL line.

        Uses default=str as a backstop so an unexpected non-JSON-able value in
        metadata.raw degrades to a string rather than crashing a long sweep.
        """
        line = json.dumps(rollout_to_dict(rollout), default=str, ensure_ascii=False)
        with self._lock:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")

    def completed_scenario_ids(self) -> set[str]:
        """Scenario ids already logged for this run. Empty if nothing logged yet.

        Resumability: skip scenarios whose id is in this set. Malformed lines
        are ignored so a partial final write (e.g. from a hard kill) can't block
        a resume.
        """
        if not self.path.exists():
            return set()
        ids: set[str] = set()
        with self.path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    ids.add(json.loads(line)["scenario_id"])
                except (json.JSONDecodeError, KeyError):
                    continue
        return ids

    def read_all(self) -> list[Rollout]:
        """Reconstruct every logged Rollout (for judge replay). Skips bad lines."""
        if not self.path.exists():
            return []
        rollouts: list[Rollout] = []
        with self.path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rollouts.append(rollout_from_dict(json.loads(line)))
                except (json.JSONDecodeError, KeyError):
                    continue
        return rollouts
