"""Grader ranges, separation of good vs bad outcomes, task-specific stress."""

from __future__ import annotations

from env.graders import GRADERS
from env.models import FlightPhase
from env.tasks import build_easy_state, build_hard_state, build_medium_state


def _easy_perfect() -> object:
    st = build_easy_state(0)
    flights = {fid: f.model_copy(update={"phase": FlightPhase.LANDED}) for fid, f in st.flights.items()}
    return st.model_copy(
        update={
            "flights": flights,
            "violation_count": 0,
            "violation_codes": [],
            "fuel_exhaustions": 0,
            "cumulative_delay_steps": 4,
            "landed_count": 3,
            "step_index": 18,
            "episode_terminal": True,
            "terminal_reason": "mission_complete",
        }
    )


def _easy_poor() -> object:
    st = build_easy_state(0)
    return st.model_copy(
        update={
            "violation_count": 9,
            "violation_codes": ["assign_closed_runway"] * 5,
            "fuel_exhaustions": 0,
            "cumulative_delay_steps": 80,
            "landed_count": 0,
            "step_index": 36,
            "episode_terminal": True,
            "terminal_reason": "max_steps",
        }
    )


def _medium_weather_abuse() -> object:
    st = build_medium_state(0)
    return st.model_copy(
        update={
            "violation_count": 8,
            "violation_codes": ["assign_closed_runway", "clear_closed_runway", "assign_closed_runway"],
            "fuel_exhaustions": 0,
            "cumulative_delay_steps": 40,
            "landed_count": 2,
            "step_index": 40,
            "episode_terminal": True,
            "terminal_reason": "max_steps",
        }
    )


def _hard_emergency_fail() -> object:
    st = build_hard_state(0)
    flights = {}
    for fid, f in st.flights.items():
        if fid == "AC020":
            flights[fid] = f.model_copy(update={"phase": FlightPhase.AT_FIX, "assigned_runway": "R1"})
        else:
            flights[fid] = f.model_copy(update={"phase": FlightPhase.LANDED})
    return st.model_copy(
        update={
            "flights": flights,
            "landed_count": 4,
            "violation_count": 0,
            "violation_codes": [],
            "fuel_exhaustions": 0,
            "cumulative_delay_steps": 30,
            "step_index": 30,
            "emergency_resolved_step": None,
            "episode_terminal": True,
            "terminal_reason": "max_steps",
        }
    )


def test_grader_ranges_on_built_states() -> None:
    state = build_easy_state(0)
    r = GRADERS["easy"](state)
    assert 0.0 <= r.score <= 1.0


def test_all_graders_invoke_in_range() -> None:
    for tid in ("easy", "medium", "hard"):
        st = {"easy": build_easy_state(0), "medium": build_medium_state(0), "hard": build_hard_state(0)}[tid]
        r = GRADERS[tid](st)
        assert 0.0 <= r.score <= 1.0


def test_easy_perfect_beats_poor() -> None:
    hi = GRADERS["easy"](_easy_perfect()).score
    lo = GRADERS["easy"](_easy_poor()).score
    assert hi > 0.85
    assert lo < 0.55
    assert hi > lo + 0.2


def test_medium_weather_misuse_lowers_score() -> None:
    baseline = GRADERS["medium"](build_medium_state(0)).score
    abused = GRADERS["medium"](_medium_weather_abuse()).score
    assert abused < baseline
    assert abused < 0.5


def test_hard_emergency_mismanagement_material() -> None:
    ok = GRADERS["hard"](
        build_hard_state(0).model_copy(
            update={
                "flights": {
                    fid: f.model_copy(update={"phase": FlightPhase.LANDED})
                    for fid, f in build_hard_state(0).flights.items()
                },
                "landed_count": 5,
                "emergency_resolved_step": 8,
                "fuel_exhaustions": 0,
                "violation_count": 0,
                "step_index": 25,
            }
        )
    ).score
    bad = GRADERS["hard"](_hard_emergency_fail()).score
    assert ok > 0.75
    assert bad < 0.45
    assert ok > bad + 0.25


def test_fuel_exhaustion_caps_score() -> None:
    st = build_easy_state(0).model_copy(update={"fuel_exhaustions": 1, "violation_count": 0})
    r = GRADERS["easy"](st)
    assert r.score <= 0.2


def test_grader_details_include_subscores_easy() -> None:
    r = GRADERS["easy"](_easy_perfect())
    assert "subscores" in r.details
    assert "metrics_used" in r.details
