"""Training reward utilities for PPO-style policy updates."""

from __future__ import annotations

from dataclasses import dataclass

from env.config import DEFAULT_CONFIG
from env.models import AACEAction, AACEObservation, FlightPhase


@dataclass(frozen=True)
class RewardSignal:
    """Decomposed reward signal used by the trainer."""

    total: float
    landed_bonus: float
    conflict_resolution_bonus: float
    action_quality_bonus: float
    emergency_bonus: float
    violation_penalty: float
    fuel_penalty: float
    validity_penalty: float
    noop_penalty: float
    delay_penalty: float
    env_bridge: float


def _flights_by_id(obs: AACEObservation) -> dict[str, object]:
    return {f.flight_id: f for f in obs.flights}


def _count_active(obs: AACEObservation) -> int:
    return sum(1 for f in obs.flights if f.phase not in (FlightPhase.LANDED, FlightPhase.DIVERTED))


def _landed_delta(prev: AACEObservation, nxt: AACEObservation) -> int:
    prev_map = _flights_by_id(prev)
    nxt_map = _flights_by_id(nxt)
    landed = 0
    for fid, after in nxt_map.items():
        before = prev_map.get(fid)
        if before is None:
            continue
        if before.phase != FlightPhase.LANDED and after.phase == FlightPhase.LANDED:
            landed += 1
    return landed


def _emergency_handled(prev: AACEObservation, nxt: AACEObservation) -> bool:
    prev_emergency = {f.flight_id for f in prev.flights if f.emergency and f.phase != FlightPhase.LANDED}
    if not prev_emergency:
        return False
    next_map = _flights_by_id(nxt)
    for fid in prev_emergency:
        after = next_map.get(fid)
        if after is None or after.phase == FlightPhase.LANDED:
            return True
    return False


def compute_training_reward(
    state: AACEObservation,
    action: AACEAction,
    next_state: AACEObservation,
    info: dict | None = None,
) -> RewardSignal:
    """Compute strong reward signal for behavioral learning pressure.

    The structure mirrors hackathon-oriented shaping:
    reward meaningful control, penalize inactivity/unsafe behavior.
    """

    _ = info or {}
    reward = 0.0

    prev_metrics = state.metrics or {}
    next_metrics = next_state.metrics or {}
    landed_count = _landed_delta(state, next_state)
    violation_prev = int((state.metadata or {}).get("violation_count", 0))
    violation_next = int((next_state.metadata or {}).get("violation_count", 0))
    violation_delta = max(0, violation_next - violation_prev)
    fuel_threshold = DEFAULT_CONFIG.simulation.fuel_critical_minutes
    fuel_critical = any(
        f.phase not in (FlightPhase.LANDED, FlightPhase.DIVERTED) and f.fuel_minutes <= fuel_threshold
        for f in next_state.flights
    )
    delay_prev = float(prev_metrics.get("delay_index", 0.0))
    delay_next = float(next_metrics.get("delay_index", delay_prev))
    delay_worsened = delay_next > delay_prev + 1e-9
    conflict_prev = float(prev_metrics.get("conflict_probability", 0.0))
    conflict_next = float(next_metrics.get("conflict_probability", 0.0))
    conflict_resolved = conflict_prev >= 0.35 and conflict_next < conflict_prev
    handled_emergency = _emergency_handled(state, next_state)

    landed_bonus = 30.0 * float(landed_count)
    conflict_resolution_bonus = 15.0 if conflict_resolved else 0.0
    action_quality_bonus = 5.0 if action.kind.value in {"assign_runway", "sequence_landing"} else 0.0
    emergency_bonus = 20.0 if handled_emergency else 0.0
    violation_penalty = -40.0 * float(violation_delta)
    fuel_penalty = -20.0 if fuel_critical else 0.0
    validity_penalty = 0.0 if next_state.last_action_valid else -10.0
    noop_penalty = -5.0 if action.kind.value == "noop" and _count_active(next_state) > 0 else 0.0
    delay_penalty = -2.0 if delay_worsened else 0.0

    env_bridge = 0.35 * float(next_state.reward or 0.0)
    reward += landed_bonus
    reward += conflict_resolution_bonus
    reward += action_quality_bonus
    reward += emergency_bonus
    reward += violation_penalty
    reward += fuel_penalty
    reward += validity_penalty
    reward += noop_penalty
    reward += delay_penalty
    reward += env_bridge

    return RewardSignal(
        total=reward,
        landed_bonus=landed_bonus,
        conflict_resolution_bonus=conflict_resolution_bonus,
        action_quality_bonus=action_quality_bonus,
        emergency_bonus=emergency_bonus,
        violation_penalty=violation_penalty,
        fuel_penalty=fuel_penalty,
        validity_penalty=validity_penalty,
        noop_penalty=noop_penalty,
        delay_penalty=delay_penalty,
        env_bridge=env_bridge,
    )
