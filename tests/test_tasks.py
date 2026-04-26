"""Task builders: determinism, weather schedules, hard-task structure."""

from __future__ import annotations

from env.config import DEFAULT_CONFIG
from env.models import FlightPhase
from env.tasks import TASK_BUILDERS, build_hard_state, build_medium_state


def test_task_builders_deterministic() -> None:
    for tid, builder in TASK_BUILDERS.items():
        a = builder(7)
        b = builder(7)
        assert a.model_dump() == b.model_dump()
        assert a.task.task_id == tid


def test_medium_weather_schedule_covers_horizon() -> None:
    st = build_medium_state(0)
    assert len(st.weather_closure_schedule) > 0
    assert st.weather_closure_schedule.get(0) == ["R2"]
    assert st.weather_closure_schedule.get(11) == []


def test_hard_task_emergency_and_priority_config() -> None:
    st = build_hard_state(0)
    assert st.task.emergency_flight_id == "AC020"
    assert st.flights["AC020"].emergency is True
    thr = DEFAULT_CONFIG.simulation.fuel_critical_minutes
    assert st.flights["AC023"].fuel_minutes <= thr + 10.0
    assert st.flights["AC024"].fuel_minutes <= thr + 10.0
    assert set(st.task.priority_flight_ids) == {"AC020", "AC023", "AC024"}


def test_hard_initial_queue_prioritizes_emergency() -> None:
    st = build_hard_state(0)
    assert st.approach_queue[0] == "AC020"
