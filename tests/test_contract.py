"""OpenEnv-style contract: reset, step, state, and task selection."""

from __future__ import annotations

import pytest

from env.environment import AACEEnvironment
from env.models import AACEAction, ActionKind, AACEState


def test_reset_returns_task_specific_observation() -> None:
    for tid in ("easy", "medium", "hard"):
        env = AACEEnvironment()
        obs = env.reset(seed=0, episode_id=f"ep-{tid}", task_id=tid)
        assert obs.task_id == tid
        assert obs.done is False
        assert obs.reward == 0.0
        assert obs.max_steps > 0
        assert len(obs.flights) > 0


def test_step_increments_step_count() -> None:
    env = AACEEnvironment()
    env.reset(seed=0, task_id="easy")
    assert env.state.step_count == 0
    env.step(AACEAction(kind=ActionKind.NOOP))
    assert env.state.step_count == 1
    env.step(AACEAction(kind=ActionKind.NOOP))
    assert env.state.step_count == 2


def test_state_reflects_terminal_transition() -> None:
    env = AACEEnvironment()
    obs = env.reset(seed=0, task_id="easy")
    while not obs.done:
        obs = env.step(AACEAction(kind=ActionKind.NOOP))
    st = env.state
    assert isinstance(st, AACEState)
    assert st.episode_terminal is True
    assert st.terminal_reason in ("max_steps", "mission_complete", "fuel_exhaustion")


def test_invalid_task_id_raises() -> None:
    env = AACEEnvironment()
    with pytest.raises(ValueError, match="Unknown task_id"):
        env.reset(seed=0, task_id="nonexistent")
