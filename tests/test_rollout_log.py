"""Tests for rollout JSONL logging (Unit 1.4).

Offline; all writes go to pytest's tmp_path, never the real data/logs/.
"""

from __future__ import annotations

import json

from drift.core.protocols import Message, Rollout, TurnMetadata
from drift.env.rollout_log import (
    RolloutLogger,
    default_run_id,
    rollout_from_dict,
    rollout_to_dict,
)


def make_rollout(scenario_id: str = "s-1", reason: str = "max_turns") -> Rollout:
    return Rollout(
        scenario_id=scenario_id,
        behavior="extraction",
        attacker_name="atk",
        target_name="gemini-2.5-flash",
        turns=(
            Message(role="system", content="You are a target. secret=ABC"),
            Message(role="user", content="hi"),
            Message(role="assistant", content="hello"),
        ),
        target_metadata=(
            TurnMetadata(
                prompt_tokens=12,
                completion_tokens=3,
                latency_ms=128.5,
                raw={"provider": "google", "finish_reason": "STOP"},
            ),
        ),
        terminated_reason=reason,
    )


# --- round-trip ---------------------------------------------------------------


def test_dict_round_trip_is_lossless():
    r = make_rollout()
    assert rollout_from_dict(rollout_to_dict(r)) == r


def test_round_trip_through_json_string():
    r = make_rollout()
    restored = rollout_from_dict(json.loads(json.dumps(rollout_to_dict(r))))
    assert restored == r
    # spot-check nested fields survive
    assert restored.target_metadata[0].raw == {"provider": "google", "finish_reason": "STOP"}
    assert restored.target_metadata[0].latency_ms == 128.5
    assert restored.terminated_reason == "max_turns"


def test_to_dict_has_schema_version():
    assert rollout_to_dict(make_rollout())["schema_version"] == 1


def test_from_dict_ignores_unknown_keys():
    d = rollout_to_dict(make_rollout())
    d["some_future_field"] = "whatever"
    # should not raise
    assert rollout_from_dict(d).scenario_id == "s-1"


# --- logger -------------------------------------------------------------------


def test_log_creates_run_dir_and_file(tmp_path):
    logger = RolloutLogger("run-x", base_dir=tmp_path)
    assert logger.run_dir.exists()
    logger.log(make_rollout("s-1"))
    assert logger.path.exists()
    assert logger.path.read_text().count("\n") == 1


def test_log_appends_one_line_per_rollout(tmp_path):
    logger = RolloutLogger("run-x", base_dir=tmp_path)
    logger.log(make_rollout("s-1"))
    logger.log(make_rollout("s-2"))
    logger.log(make_rollout("s-3"))
    assert logger.path.read_text().strip().count("\n") == 2  # 3 lines


def test_completed_scenario_ids(tmp_path):
    logger = RolloutLogger("run-x", base_dir=tmp_path)
    logger.log(make_rollout("s-1"))
    logger.log(make_rollout("s-2"))
    assert logger.completed_scenario_ids() == {"s-1", "s-2"}


def test_completed_empty_when_nothing_logged(tmp_path):
    logger = RolloutLogger("run-x", base_dir=tmp_path)
    assert logger.completed_scenario_ids() == set()


def test_resumability_across_reopen(tmp_path):
    RolloutLogger("run-x", base_dir=tmp_path).log(make_rollout("s-1"))
    # New logger instance, same run_id -> sees prior progress and appends.
    logger2 = RolloutLogger("run-x", base_dir=tmp_path)
    assert logger2.completed_scenario_ids() == {"s-1"}
    logger2.log(make_rollout("s-2"))
    assert logger2.completed_scenario_ids() == {"s-1", "s-2"}


def test_read_all_reconstructs_rollouts(tmp_path):
    logger = RolloutLogger("run-x", base_dir=tmp_path)
    r1 = make_rollout("s-1", reason="max_turns")
    r2 = make_rollout("s-2", reason="judge_success")
    logger.log(r1)
    logger.log(r2)
    restored = logger.read_all()
    assert restored == [r1, r2]


def test_read_all_empty_when_no_file(tmp_path):
    assert RolloutLogger("run-x", base_dir=tmp_path).read_all() == []


def test_malformed_line_is_skipped(tmp_path):
    logger = RolloutLogger("run-x", base_dir=tmp_path)
    logger.log(make_rollout("s-1"))
    # simulate a partial/corrupt trailing line from a hard kill
    with logger.path.open("a", encoding="utf-8") as f:
        f.write('{"scenario_id": "s-2", "behav')  # truncated JSON, no newline
    assert logger.completed_scenario_ids() == {"s-1"}
    assert len(logger.read_all()) == 1


def test_default_run_id_shape():
    rid = default_run_id()
    assert rid.startswith("run-") and rid.endswith("Z")
    assert default_run_id("eval").startswith("eval-")
