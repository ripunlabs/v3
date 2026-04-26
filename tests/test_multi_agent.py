"""Multi-agent oversight compatibility tests."""

from __future__ import annotations

from env.environment import AACEEnvironment
from env.models import AACEAction, ActionKind


def test_observation_includes_oversight_metadata() -> None:
    env = AACEEnvironment()
    env.reset(seed=0, task_id="easy")
    obs = env.step(AACEAction(kind=ActionKind.NOOP))
    oversight = dict((obs.metadata or {}).get("oversight", {}))
    assert "selected_agent" in oversight
    assert "scores" in oversight
    assert "reason" in oversight


def test_invalid_primary_action_preserves_ground_truth_execution() -> None:
    env = AACEEnvironment()
    env.reset(seed=0, task_id="medium")
    obs = env.step(AACEAction(kind=ActionKind.ASSIGN_RUNWAY, flight_id="AC010", runway_id="R2"))
    assert obs.last_action_valid is False
    oversight = dict((obs.metadata or {}).get("oversight", {}))
    assert isinstance(oversight.get("scores"), dict)
