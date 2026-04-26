"""Episode runner for baseline/trained evaluations and demo replay."""

from __future__ import annotations

from dataclasses import dataclass

from env.agents import ATCAgent, AirlineAgent, AirportOpsAgent, OversightAgent, WeatherAgent
from env.config import DEFAULT_CONFIG
from env.environment import AACEEnvironment
from env.models import AACEAction, AACEObservation, ActionKind, FlightPhase
from training.policies import PreferencePolicy, baseline_policy

TASK_CYCLE = ("easy", "medium", "hard")


@dataclass
class StepRecord:
    step_index: int
    max_steps: int
    task_id: str
    phase_id: str
    atc_action: AACEAction
    airline_action: AACEAction
    ops_action: AACEAction
    trained_action: AACEAction | None
    weather_note: str
    oversight_selected_agent: str
    oversight_scores: dict[str, float]
    oversight_reason: str
    conflict_detected: bool
    final_action: AACEAction
    reward: float
    valid: bool
    message: str
    violation_count: int
    done: bool


@dataclass
class EpisodeRecord:
    episode: int
    task_id: str
    reward: float
    violations: int
    completion: bool
    terminal_reason: str
    safety_violations: int
    steps: list[StepRecord]


def _safe_scores(oversight: dict) -> dict[str, float]:
    out = {}
    for k, v in dict(oversight.get("scores", {})).items():
        out[str(k)] = float(v)
    return out


def _phase_id(step_index: int, max_steps: int) -> str:
    """Deterministic cinematic phase schedule for judge demos.

    Uses observed step index windows so the story always progresses in order
    during typical hard-task demos instead of depending on max_steps horizon.
    """
    _ = max_steps  # Retained for signature compatibility.
    if step_index <= 1:
        return "phase1_stable"
    if step_index <= 3:
        return "phase2_stress"
    if step_index <= 6:
        return "phase3_conflict"
    if step_index <= 8:
        return "phase4_oversight"
    if step_index <= 10:
        return "phase5_resolution"
    return "phase6_learning"


def _action_signature(action: AACEAction) -> str:
    return (
        f"{action.kind.value}|{action.flight_id}|{action.runway_id}|"
        f"{','.join(action.ordered_flight_ids or [])}"
    )


def _priority_flight_id(state) -> str | None:
    active = [f for f in state.flights.values() if f.phase not in (FlightPhase.LANDED, FlightPhase.DIVERTED)]
    if not active:
        return None
    active.sort(key=lambda f: (f.fuel_minutes, f.eta_steps, f.flight_id))
    return active[0].flight_id


def _force_conflict_if_needed(state, atc_action: AACEAction, airline_action: AACEAction, ops_action: AACEAction) -> tuple[AACEAction, AACEAction]:
    """Ensure visible proposal divergence during phase-3 storytelling."""
    priority_fid = _priority_flight_id(state)
    open_runways = [
        rid
        for rid in sorted(state.runways)
        if state.runways[rid].open and rid not in set(state.weather.closed_runways)
    ]

    forced_airline = airline_action
    forced_ops = ops_action
    if _action_signature(forced_airline) == _action_signature(atc_action):
        if priority_fid:
            forced_airline = AACEAction(kind=ActionKind.DECLARE_EMERGENCY, flight_id=priority_fid)
        elif state.approach_queue:
            forced_airline = AACEAction(kind=ActionKind.SEQUENCE_LANDING, ordered_flight_ids=list(reversed(state.approach_queue)))

    if _action_signature(forced_ops) == _action_signature(atc_action):
        if atc_action.kind == ActionKind.ASSIGN_RUNWAY and atc_action.flight_id and len(open_runways) > 1:
            alternate = next((rw for rw in open_runways if rw != atc_action.runway_id), open_runways[0])
            forced_ops = AACEAction(kind=ActionKind.ASSIGN_RUNWAY, flight_id=atc_action.flight_id, runway_id=alternate)
        else:
            forced_ops = AACEAction(kind=ActionKind.NOOP)
    return forced_airline, forced_ops


def run_episode_with_orchestration(
    env: AACEEnvironment,
    *,
    task_id: str,
    seed: int,
    trained_policy: PreferencePolicy | None = None,
    mode: str = "baseline",
) -> EpisodeRecord:
    obs = env.reset(seed=seed, task_id=task_id)
    done = False
    total_reward = 0.0
    steps: list[StepRecord] = []

    atc_agent = ATCAgent(DEFAULT_CONFIG)
    airline_agent = AirlineAgent(DEFAULT_CONFIG)
    ops_agent = AirportOpsAgent()
    weather_agent = WeatherAgent()
    oversight_agent = OversightAgent(DEFAULT_CONFIG)

    while not done:
        state = env.snapshot_for_grader()
        phase_id = _phase_id(obs.step_index, obs.max_steps)

        atc_action = atc_agent.act(state)
        airline_action = airline_agent.act(state)
        ops_action = ops_agent.act(state)
        _ = weather_agent.act(state)
        if mode == "demo" and phase_id == "phase3_conflict":
            airline_action, ops_action = _force_conflict_if_needed(state, atc_action, airline_action, ops_action)

        selected_action = atc_action
        selected_agent = "atc"
        scores: dict[str, float] = {}
        oversight_reason = "atc_default"
        trained_action = None

        candidates = {"atc": atc_action, "airline": airline_action, "ops": ops_action}
        if mode in {"trained", "demo"} and trained_policy is not None:
            trained_action = trained_policy.choose_action(obs)
            candidates["trained"] = trained_action

        # Always arbitrate with oversight for non-baseline modes so trained runs
        # are safety-filtered and comparable to the control-tower decision stack.
        oversight = oversight_agent.resolve(candidates, state)
        selected_agent = str(oversight.get("selected_agent", "atc"))
        scores = _safe_scores(oversight)
        oversight_reason = str(oversight.get("reason", "oversight_selection"))
        if selected_agent == "airline":
            selected_action = airline_action
        elif selected_agent == "ops":
            selected_action = ops_action
        elif selected_agent == "trained" and trained_action is not None:
            selected_action = trained_action

        next_obs: AACEObservation = env.step(selected_action)
        total_reward += float(next_obs.reward or 0.0)
        done = bool(next_obs.done)
        violations = int((next_obs.metadata or {}).get("violation_count", 0))
        weather_note = "closed_runways=" + (",".join(sorted(next_obs.weather.closed_runways)) or "none")
        proposals = [atc_action, airline_action, ops_action]
        if trained_action is not None:
            proposals.append(trained_action)
        conflict_detected = len({_action_signature(a) for a in proposals}) > 1

        steps.append(
            StepRecord(
                step_index=next_obs.step_index,
                max_steps=next_obs.max_steps,
                task_id=task_id,
                phase_id=phase_id,
                atc_action=atc_action,
                airline_action=airline_action,
                ops_action=ops_action,
                trained_action=trained_action,
                weather_note=weather_note,
                oversight_selected_agent=selected_agent,
                oversight_scores=scores,
                oversight_reason=oversight_reason,
                conflict_detected=conflict_detected,
                final_action=selected_action,
                reward=float(next_obs.reward or 0.0),
                valid=bool(next_obs.last_action_valid),
                message=str(next_obs.last_action_message),
                violation_count=violations,
                done=done,
            )
        )
        obs = next_obs

    terminal_reason = str((obs.metadata or {}).get("terminal_reason", "unknown"))
    violations = int((obs.metadata or {}).get("violation_count", 0))
    return EpisodeRecord(
        episode=0,
        task_id=task_id,
        reward=total_reward,
        violations=violations,
        completion=terminal_reason == "mission_complete",
        terminal_reason=terminal_reason,
        safety_violations=violations,
        steps=steps,
    )


def evaluate_policy(
    *,
    episodes: int,
    seed: int,
    mode: str,
    trained_policy: PreferencePolicy | None = None,
) -> list[EpisodeRecord]:
    env = AACEEnvironment()
    records: list[EpisodeRecord] = []

    for episode in range(episodes):
        task_id = TASK_CYCLE[episode % len(TASK_CYCLE)]
        if mode == "baseline":
            # Baseline path intentionally does not use oversight and mirrors legacy behavior.
            obs = env.reset(seed=seed + episode, task_id=task_id)
            done = False
            total_reward = 0.0
            while not done:
                action = baseline_policy(obs)
                obs = env.step(action)
                total_reward += float(obs.reward or 0.0)
                done = bool(obs.done)
            terminal_reason = str((obs.metadata or {}).get("terminal_reason", "unknown"))
            violations = int((obs.metadata or {}).get("violation_count", 0))
            records.append(
                EpisodeRecord(
                    episode=episode,
                    task_id=task_id,
                    reward=total_reward,
                    violations=violations,
                    completion=terminal_reason == "mission_complete",
                    terminal_reason=terminal_reason,
                    safety_violations=violations,
                    steps=[],
                )
            )
            continue

        record = run_episode_with_orchestration(
            env,
            task_id=task_id,
            seed=seed + episode,
            trained_policy=trained_policy,
            mode=mode,
        )
        record.episode = episode
        records.append(record)

    return records
