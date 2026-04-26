"""Environment dynamics, safety rules, rewards, and terminal semantics."""

from __future__ import annotations

from env.environment import AACEEnvironment
from env.inference_policy import scripted_action
from env.models import AACEAction, ActionKind, FlightPhase


def _until_at_fix(env: AACEEnvironment, fid: str, max_steps: int = 30) -> None:
    for _ in range(max_steps):
        if env.snapshot_for_grader().flights[fid].phase == FlightPhase.AT_FIX:
            return
        env.step(AACEAction(kind=ActionKind.NOOP))
    raise AssertionError(f"{fid} did not reach AT_FIX in time")


def test_assign_runway_open_runway_succeeds() -> None:
    env = AACEEnvironment()
    env.reset(seed=0, task_id="easy")
    obs = env.step(
        AACEAction(kind=ActionKind.ASSIGN_RUNWAY, flight_id="AC001", runway_id="R1")
    )
    assert obs.last_action_valid is True


def test_assign_runway_weather_closed_fails_medium() -> None:
    env = AACEEnvironment()
    env.reset(seed=0, task_id="medium")
    obs = env.step(
        AACEAction(kind=ActionKind.ASSIGN_RUNWAY, flight_id="AC010", runway_id="R2")
    )
    assert obs.last_action_valid is False
    assert env.snapshot_for_grader().violation_count >= 1


def test_clear_to_land_without_assignment_fails() -> None:
    env = AACEEnvironment()
    env.reset(seed=0, task_id="easy")
    env.step(
        AACEAction(
            kind=ActionKind.SEQUENCE_LANDING,
            ordered_flight_ids=["AC001", "AC002", "AC003"],
        )
    )
    _until_at_fix(env, "AC001")
    obs = env.step(AACEAction(kind=ActionKind.CLEAR_TO_LAND, flight_id="AC001"))
    assert obs.last_action_valid is False


def test_clear_runway_occupied_fails() -> None:
    env = AACEEnvironment()
    env.reset(seed=0, task_id="easy")
    env.step(
        AACEAction(
            kind=ActionKind.SEQUENCE_LANDING,
            ordered_flight_ids=["AC001", "AC002", "AC003"],
        )
    )
    _until_at_fix(env, "AC001")
    env.step(AACEAction(kind=ActionKind.ASSIGN_RUNWAY, flight_id="AC001", runway_id="R1"))
    env.step(AACEAction(kind=ActionKind.CLEAR_TO_LAND, flight_id="AC001"))
    _until_at_fix(env, "AC002")
    env.step(AACEAction(kind=ActionKind.ASSIGN_RUNWAY, flight_id="AC002", runway_id="R1"))
    obs = env.step(AACEAction(kind=ActionKind.CLEAR_TO_LAND, flight_id="AC002"))
    assert obs.last_action_valid is False


def test_declare_emergency_moves_to_queue_front() -> None:
    env = AACEEnvironment()
    env.reset(seed=0, task_id="easy")
    env.step(
        AACEAction(
            kind=ActionKind.SEQUENCE_LANDING,
            ordered_flight_ids=["AC001", "AC002", "AC003"],
        )
    )
    assert env.snapshot_for_grader().approach_queue[0] == "AC001"
    env.step(AACEAction(kind=ActionKind.DECLARE_EMERGENCY, flight_id="AC003"))
    assert env.snapshot_for_grader().approach_queue[0] == "AC003"


def test_hold_pattern_increases_fuel_burn_vs_noop() -> None:
    env_a = AACEEnvironment()
    env_b = AACEEnvironment()
    env_a.reset(seed=0, task_id="easy")
    env_b.reset(seed=0, task_id="easy")
    env_b.step(AACEAction(kind=ActionKind.HOLD_PATTERN, flight_id="AC001", hold_rounds=4))
    for _ in range(5):
        env_a.step(AACEAction(kind=ActionKind.NOOP))
        env_b.step(AACEAction(kind=ActionKind.NOOP))
    fa = env_a.snapshot_for_grader().flights["AC001"].fuel_minutes
    fb = env_b.snapshot_for_grader().flights["AC001"].fuel_minutes
    assert fb < fa


def test_reroute_removes_from_active_control() -> None:
    env = AACEEnvironment()
    env.reset(seed=0, task_id="easy")
    obs = env.step(
        AACEAction(
            kind=ActionKind.REROUTE_FLIGHT,
            flight_id="AC002",
            alternate_code="KPHL",
        )
    )
    assert obs.last_action_valid is True
    assert "AC002" not in env.snapshot_for_grader().active_flight_ids()


def test_fuel_exhaustion_terminal_hard_noop() -> None:
    env = AACEEnvironment()
    obs = env.reset(seed=0, task_id="hard")
    for _ in range(200):
        if obs.done:
            break
        obs = env.step(AACEAction(kind=ActionKind.NOOP))
    assert obs.done
    assert env.snapshot_for_grader().terminal_reason == "fuel_exhaustion"


def test_mission_complete_scripted_easy() -> None:
    env = AACEEnvironment()
    obs = env.reset(seed=0, task_id="easy")
    for _ in range(120):
        if obs.done:
            break
        obs = env.step(scripted_action(obs))
    assert obs.done
    assert env.snapshot_for_grader().terminal_reason == "mission_complete"


def test_invalid_action_reduces_reward_vs_valid_noop() -> None:
    env = AACEEnvironment()
    env.reset(seed=0, task_id="easy")
    r_bad = float(
        env.step(
            AACEAction(kind=ActionKind.ASSIGN_RUNWAY, flight_id="AC999", runway_id="R1")
        ).reward
        or 0.0
    )
    env.reset(seed=0, task_id="easy")
    r_ok = float(env.step(AACEAction(kind=ActionKind.NOOP)).reward or 0.0)
    assert r_bad < r_ok


def test_landing_produces_higher_reward_than_noop_at_fix() -> None:
    env = AACEEnvironment()
    env.reset(seed=0, task_id="easy")
    env.step(
        AACEAction(
            kind=ActionKind.SEQUENCE_LANDING,
            ordered_flight_ids=["AC001", "AC002", "AC003"],
        )
    )
    _until_at_fix(env, "AC001")
    env.step(AACEAction(kind=ActionKind.ASSIGN_RUNWAY, flight_id="AC001", runway_id="R1"))
    r_land = float(env.step(AACEAction(kind=ActionKind.CLEAR_TO_LAND, flight_id="AC001")).reward or 0.0)

    env = AACEEnvironment()
    env.reset(seed=0, task_id="easy")
    env.step(
        AACEAction(
            kind=ActionKind.SEQUENCE_LANDING,
            ordered_flight_ids=["AC001", "AC002", "AC003"],
        )
    )
    _until_at_fix(env, "AC001")
    env.step(AACEAction(kind=ActionKind.ASSIGN_RUNWAY, flight_id="AC001", runway_id="R1"))
    r_noop = float(env.step(AACEAction(kind=ActionKind.NOOP)).reward or 0.0)
    assert r_land > r_noop


def test_catastrophe_reward_large_negative() -> None:
    env = AACEEnvironment()
    env.reset(seed=0, task_id="hard")
    worst = 0.0
    obs = env.step(AACEAction(kind=ActionKind.NOOP))
    worst = min(worst, float(obs.reward or 0.0))
    for _ in range(80):
        obs = env.step(AACEAction(kind=ActionKind.NOOP))
        worst = min(worst, float(obs.reward or 0.0))
        if obs.done:
            break
    assert worst < -2.0


def test_reward_varies_over_trajectory() -> None:
    env = AACEEnvironment()
    obs = env.reset(seed=0, task_id="easy")
    seen = {float(obs.reward or 0.0)}
    for _ in range(15):
        obs = env.step(scripted_action(obs))
        seen.add(float(obs.reward or 0.0))
    assert len(seen) >= 3


def test_noop_streak_escalates_penalty() -> None:
    env = AACEEnvironment()
    env.reset(seed=0, task_id="easy")
    r1 = float(env.step(AACEAction(kind=ActionKind.NOOP)).reward or 0.0)
    r2 = float(env.step(AACEAction(kind=ActionKind.NOOP)).reward or 0.0)
    r3 = float(env.step(AACEAction(kind=ActionKind.NOOP)).reward or 0.0)
    assert r3 <= r2 <= r1
