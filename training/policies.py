"""Baseline and trainable policies for MACE training/evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from math import exp

from env.config import DEFAULT_CONFIG
from env.models import AACEAction, AACEObservation, ActionKind, FlightPhase

FUEL_CRITICAL_MINUTES = DEFAULT_CONFIG.simulation.fuel_critical_minutes


def _active_flights(obs: AACEObservation):
    flights = [f for f in obs.flights if f.phase not in (FlightPhase.LANDED, FlightPhase.DIVERTED)]
    flights.sort(key=lambda f: (f.eta_steps, f.flight_id))
    return flights


def _open_runways(obs: AACEObservation) -> list[str]:
    closed = set(obs.weather.closed_runways)
    return [r.runway_id for r in sorted(obs.runways, key=lambda x: x.runway_id) if r.open and r.runway_id not in closed]


def baseline_policy(obs: AACEObservation) -> AACEAction:
    """Deterministic baseline used for before/after comparison."""
    active = _active_flights(obs)
    if not active:
        return AACEAction(kind=ActionKind.NOOP)
    if obs.step_index % 6 == 0 and len(active) > 1:
        return AACEAction(kind=ActionKind.SEQUENCE_LANDING, ordered_flight_ids=[f.flight_id for f in active])
    return AACEAction(kind=ActionKind.NOOP)


def _candidate_actions(obs: AACEObservation) -> dict[ActionKind, AACEAction]:
    active = _active_flights(obs)
    open_runways = _open_runways(obs)
    queue = list(obs.approach_queue)

    emergency = sorted([f for f in active if f.emergency], key=lambda f: (f.eta_steps, f.fuel_minutes, f.flight_id))
    critical = sorted(
        [f for f in active if f.fuel_minutes <= FUEL_CRITICAL_MINUTES],
        key=lambda f: (f.fuel_minutes, f.eta_steps, f.flight_id),
    )
    target = emergency[0] if emergency else (critical[0] if critical else (active[0] if active else None))

    actions: dict[ActionKind, AACEAction] = {ActionKind.NOOP: AACEAction(kind=ActionKind.NOOP)}
    if target is None:
        return actions

    if target.phase == FlightPhase.AT_FIX and not target.assigned_runway and open_runways:
        actions[ActionKind.ASSIGN_RUNWAY] = AACEAction(
            kind=ActionKind.ASSIGN_RUNWAY,
            flight_id=target.flight_id,
            runway_id=open_runways[0],
        )

    if target.phase == FlightPhase.AT_FIX and target.assigned_runway:
        if target.emergency or not queue or queue[0] == target.flight_id:
            actions[ActionKind.CLEAR_TO_LAND] = AACEAction(kind=ActionKind.CLEAR_TO_LAND, flight_id=target.flight_id)

    desired_order = [f.flight_id for f in emergency]
    desired_order.extend(f.flight_id for f in critical if f.flight_id not in set(desired_order))
    desired_order.extend(f.flight_id for f in active if f.flight_id not in set(desired_order))
    if desired_order and desired_order != queue:
        actions[ActionKind.SEQUENCE_LANDING] = AACEAction(
            kind=ActionKind.SEQUENCE_LANDING,
            ordered_flight_ids=desired_order,
        )

    if critical and not critical[0].emergency:
        actions[ActionKind.DECLARE_EMERGENCY] = AACEAction(
            kind=ActionKind.DECLARE_EMERGENCY,
            flight_id=critical[0].flight_id,
        )
    return actions


@dataclass
class PreferencePolicy:
    """Lightweight policy with action-kind preferences for PPO-style updates."""

    preferences: dict[str, float]

    @classmethod
    def default(cls) -> "PreferencePolicy":
        return cls(
            preferences={
                ActionKind.NOOP.value: 0.0,
                ActionKind.ASSIGN_RUNWAY.value: 0.2,
                ActionKind.SEQUENCE_LANDING.value: 0.2,
                ActionKind.CLEAR_TO_LAND.value: 0.4,
                ActionKind.DECLARE_EMERGENCY.value: 0.1,
            }
        )

    def choose_action(self, obs: AACEObservation) -> AACEAction:
        actions = _candidate_actions(obs)
        if not actions:
            return AACEAction(kind=ActionKind.NOOP)

        best_kind = max(actions, key=lambda k: self.preferences.get(k.value, -999.0))
        return actions.get(best_kind, AACEAction(kind=ActionKind.NOOP))

    def action_probs(self, obs: AACEObservation) -> dict[str, float]:
        actions = _candidate_actions(obs)
        if not actions:
            return {ActionKind.NOOP.value: 1.0}

        logits = {kind.value: self.preferences.get(kind.value, 0.0) for kind in actions}
        max_logit = max(logits.values())
        exp_values = {k: exp(v - max_logit) for k, v in logits.items()}
        z = sum(exp_values.values()) or 1.0
        return {k: (v / z) for k, v in exp_values.items()}
