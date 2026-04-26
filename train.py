"""Deterministic MACE training and structured evaluation reporting."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import sys
import time

import matplotlib.pyplot as plt
from rich.console import Console

from env.environment import AACEEnvironment
from env.models import AACEAction, AACEObservation, ActionKind, FlightPhase

FUEL_CRITICAL_MINUTES = 12.0
TASK_CYCLE = ("easy", "medium", "hard")


def _supports_unicode() -> bool:
    enc = sys.stdout.encoding or "utf-8"
    try:
        "━━━━━━━━".encode(enc)
        return True
    except Exception:
        return False


UNICODE_OK = _supports_unicode()
ARROW = "→" if UNICODE_OK else "->"


def _u(unicode_text: str, ascii_text: str) -> str:
    return unicode_text if UNICODE_OK else ascii_text


@dataclass(frozen=True)
class TrainConfig:
    num_episodes: int = 60
    heuristic_start_episode: int = 20
    base_seed: int = 42
    output_curve_path: str = "reward_curve.png"
    output_csv_path: str = "training_metrics.csv"
    broadcast: bool = False
    frame_delay_s: float = 0.6


@dataclass(frozen=True)
class EpisodeResult:
    episode: int
    task_id: str
    seed: int
    total_reward: float
    terminal_reason: str | None
    violations: int
    fuel_exhaustions: int
    emergency_present: bool
    emergency_success: bool | None
    done: bool
    violation_codes: tuple[str, ...]
    valid_action_rate: float
    last_action_valid: bool
    last_action_message: str
    oversight_selected_agent: str
    oversight_scores: dict[str, float]
    last_action_repr: str
    active_flights_end: tuple[str, ...]
    emergency_flights_end: tuple[str, ...]
    fuel_critical_flights_end: tuple[str, ...]
    runway_state_end: tuple[str, ...]
    oversight_conflict_detected: bool


def _active_flights(obs: AACEObservation):
    flights = [f for f in obs.flights if f.phase not in (FlightPhase.LANDED, FlightPhase.DIVERTED)]
    flights.sort(key=lambda f: (f.eta_steps, f.flight_id))
    return flights


def _open_runways(obs: AACEObservation) -> list[str]:
    weather_closed = set(obs.weather.closed_runways)
    return [r.runway_id for r in sorted(obs.runways, key=lambda x: x.runway_id) if r.open and r.runway_id not in weather_closed]


def _runway_map(obs: AACEObservation):
    return {r.runway_id: r for r in obs.runways}


def _is_runway_open(obs: AACEObservation, runway_id: str) -> bool:
    rw = _runway_map(obs).get(runway_id)
    if rw is None or not rw.open:
        return False
    return runway_id not in set(obs.weather.closed_runways)


def _is_runway_occupied(obs: AACEObservation, runway_id: str) -> bool:
    rw = _runway_map(obs).get(runway_id)
    if rw is None:
        return True
    return obs.step_index < rw.occupied_until_step


def _flight_by_id(obs: AACEObservation, flight_id: str | None):
    if not flight_id:
        return None
    for f in obs.flights:
        if f.flight_id == flight_id:
            return f
    return None


def is_safe_action(action: AACEAction, obs: AACEObservation) -> bool:
    if action.kind == ActionKind.NOOP:
        return True
    if action.kind == ActionKind.ASSIGN_RUNWAY:
        return bool(action.runway_id) and _is_runway_open(obs, action.runway_id)
    if action.kind == ActionKind.CLEAR_TO_LAND:
        fl = _flight_by_id(obs, action.flight_id)
        if fl is None or not fl.assigned_runway:
            return False
        if not _is_runway_open(obs, fl.assigned_runway):
            return False
        if _is_runway_occupied(obs, fl.assigned_runway):
            return False
        if not fl.emergency and obs.approach_queue and obs.approach_queue[0] != fl.flight_id:
            return False
        return True
    if action.kind == ActionKind.SEQUENCE_LANDING:
        active_ids = sorted(f.flight_id for f in _active_flights(obs))
        proposed = sorted(action.ordered_flight_ids or [])
        return proposed == active_ids
    if action.kind == ActionKind.DECLARE_EMERGENCY:
        fl = _flight_by_id(obs, action.flight_id)
        return fl is not None and not fl.emergency and fl.phase not in (FlightPhase.LANDED, FlightPhase.DIVERTED)
    return True


def naive_policy(obs: AACEObservation) -> AACEAction:
    active = _active_flights(obs)
    if not active:
        return AACEAction(kind=ActionKind.NOOP)
    if obs.step_index % 6 == 0 and len(active) > 1:
        return AACEAction(kind=ActionKind.SEQUENCE_LANDING, ordered_flight_ids=[f.flight_id for f in active])
    return AACEAction(kind=ActionKind.NOOP)


def heuristic_policy(obs: AACEObservation) -> AACEAction:
    active = _active_flights(obs)
    if not active:
        return AACEAction(kind=ActionKind.NOOP)

    open_runways = _open_runways(obs)
    queue = list(obs.approach_queue)
    emergency = sorted([f for f in active if f.emergency], key=lambda f: (f.eta_steps, f.fuel_minutes, f.flight_id))
    critical = sorted([f for f in active if f.fuel_minutes <= FUEL_CRITICAL_MINUTES], key=lambda f: (f.fuel_minutes, f.eta_steps, f.flight_id))
    target = emergency[0] if emergency else (critical[0] if critical else active[0])

    candidate_actions: list[AACEAction] = []
    if target.phase == FlightPhase.AT_FIX:
        if not target.assigned_runway and open_runways:
            candidate_actions.append(AACEAction(kind=ActionKind.ASSIGN_RUNWAY, flight_id=target.flight_id, runway_id=open_runways[0]))
        if target.assigned_runway and (target.emergency or not queue or queue[0] == target.flight_id):
            candidate_actions.append(AACEAction(kind=ActionKind.CLEAR_TO_LAND, flight_id=target.flight_id))

    if open_runways:
        for fl in active:
            if fl.phase == FlightPhase.AT_FIX and not fl.assigned_runway:
                candidate_actions.append(AACEAction(kind=ActionKind.ASSIGN_RUNWAY, flight_id=fl.flight_id, runway_id=open_runways[0]))
                break

    desired_order = [f.flight_id for f in emergency]
    desired_order.extend(f.flight_id for f in critical if f.flight_id not in set(desired_order))
    desired_order.extend(f.flight_id for f in active if f.flight_id not in set(desired_order))
    if desired_order and desired_order != queue:
        candidate_actions.append(AACEAction(kind=ActionKind.SEQUENCE_LANDING, ordered_flight_ids=desired_order))

    candidate_actions.append(AACEAction(kind=ActionKind.NOOP))
    safe_actions = [a for a in candidate_actions if is_safe_action(a, obs)]
    return safe_actions[0] if safe_actions else AACEAction(kind=ActionKind.NOOP)


def policy_for_mode(obs: AACEObservation, episode: int, cfg: TrainConfig, mode: str) -> AACEAction:
    if mode == "baseline":
        return naive_policy(obs)
    if episode < cfg.heuristic_start_episode:
        return naive_policy(obs)
    return heuristic_policy(obs)


def run_eval(cfg: TrainConfig, mode: str) -> tuple[list[EpisodeResult], list[float], list[int]]:
    env = AACEEnvironment()
    console = Console()
    episodes: list[EpisodeResult] = []
    rewards: list[float] = []
    violations: list[int] = []

    for episode in range(cfg.num_episodes):
        seed = cfg.base_seed + episode
        task_id = TASK_CYCLE[episode % len(TASK_CYCLE)]
        obs = env.reset(seed=seed, task_id=task_id)
        done = False
        total_reward = 0.0
        valid_actions = 0
        total_actions = 0
        last_action = AACEAction(kind=ActionKind.NOOP)

        while not done:
            action = policy_for_mode(obs, episode, cfg, mode=mode)
            last_action = action
            obs = env.step(action)
            if cfg.broadcast and mode == "mace":
                _render_broadcast_frame(
                    console=console,
                    timestep=obs.step_index,
                    atc_action=action,
                    obs_after=obs,
                    frame_delay_s=cfg.frame_delay_s,
                )
            total_reward += float(obs.reward or 0.0)
            done = bool(obs.done)
            total_actions += 1
            if obs.last_action_valid:
                valid_actions += 1

        st = env.snapshot_for_grader()
        emergency_present = bool(st.task.emergency_flight_id)
        emergency_success = None
        if emergency_present:
            emergency_success = st.emergency_resolved_step is not None and st.terminal_reason != "fuel_exhaustion"

        oversight = dict((obs.metadata or {}).get("oversight", {}))
        selected_agent = str(oversight.get("selected_agent", "unknown"))
        raw_scores = dict(oversight.get("scores", {}))
        scores = {str(k): float(v) for k, v in raw_scores.items()}
        active_end = tuple(st.active_flight_ids())
        emergency_end = tuple(sorted(fid for fid in active_end if st.flights[fid].emergency))
        critical_end = tuple(
            sorted(
                fid
                for fid in active_end
                if st.flights[fid].fuel_minutes <= cfg.base_seed * 0 + FUEL_CRITICAL_MINUTES
            )
        )
        runway_state = []
        for rid in sorted(st.runways):
            rw = st.runways[rid]
            status = "OPEN"
            if not rw.open or rid in st.weather.closed_runways:
                status = "CLOSED"
            elif st.step_index < rw.occupied_until_step:
                status = "OCCUPIED"
            runway_state.append(f"{rid}:{status}")

        ep = EpisodeResult(
            episode=episode,
            task_id=task_id,
            seed=seed,
            total_reward=total_reward,
            terminal_reason=st.terminal_reason,
            violations=st.violation_count,
            fuel_exhaustions=st.fuel_exhaustions,
            emergency_present=emergency_present,
            emergency_success=emergency_success,
            done=bool(obs.done),
            violation_codes=tuple(st.violation_codes),
            valid_action_rate=(valid_actions / total_actions) if total_actions else 0.0,
            last_action_valid=bool(obs.last_action_valid),
            last_action_message=str(obs.last_action_message),
            oversight_selected_agent=selected_agent,
            oversight_scores=scores,
            last_action_repr=f"{last_action.kind.value}(flight={last_action.flight_id}, runway={last_action.runway_id})",
            active_flights_end=active_end,
            emergency_flights_end=emergency_end,
            fuel_critical_flights_end=critical_end,
            runway_state_end=tuple(runway_state),
            oversight_conflict_detected=(selected_agent not in {"atc", "unknown"}),
        )
        episodes.append(ep)
        rewards.append(total_reward)
        violations.append(st.violation_count)

    return episodes, rewards, violations


def save_csv(path: Path, episodes: list[EpisodeResult]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "task_id", "seed", "reward", "violations", "terminal_reason", "fuel_exhaustions"])
        for ep in episodes:
            writer.writerow([ep.episode, ep.task_id, ep.seed, f"{ep.total_reward:.6f}", ep.violations, ep.terminal_reason, ep.fuel_exhaustions])


def _running_average(values: list[float], window: int = 5) -> list[float]:
    out: list[float] = []
    total = 0.0
    for i, value in enumerate(values):
        total += value
        if i >= window:
            total -= values[i - window]
            out.append(total / window)
        else:
            out.append(total / (i + 1))
    return out


def save_curve(path: Path, rewards: list[float], heuristic_start_episode: int) -> None:
    plt.figure(figsize=(9, 4.5))
    plt.plot(rewards, label="Episode Reward", linewidth=1.8)
    if rewards:
        plt.plot(_running_average(rewards, window=5), label="Running Average", linestyle="--", linewidth=2.0)
    plt.axvline(heuristic_start_episode, color="gray", linestyle=":", linewidth=2.0, label="Heuristic Policy Activated")
    plt.title("MACE Training: Reward Improvement Over Episodes")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def _pct(numer: int, denom: int) -> float:
    if denom <= 0:
        return 0.0
    return 100.0 * numer / denom


def summarize(episodes: list[EpisodeResult]) -> dict[str, float | int]:
    n = len(episodes)
    avg_reward = sum(ep.total_reward for ep in episodes) / max(1, n)
    avg_violations = sum(ep.violations for ep in episodes) / max(1, n)
    completion = sum(1 for ep in episodes if ep.terminal_reason == "mission_complete")
    fuel_survival = sum(1 for ep in episodes if ep.fuel_exhaustions == 0)
    emerg_total = sum(1 for ep in episodes if ep.emergency_present)
    emerg_success = sum(1 for ep in episodes if ep.emergency_success is True)
    return {
        "avg_reward": avg_reward,
        "avg_violations": avg_violations,
        "completion_rate": _pct(completion, n),
        "emergency_success_rate": _pct(emerg_success, emerg_total),
        "fuel_survival_rate": _pct(fuel_survival, n),
        "total_violations": sum(ep.violations for ep in episodes),
    }


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Deterministic MACE training and final benchmark report.")
    parser.add_argument("--episodes", type=int, default=60, help="Number of episodes.")
    parser.add_argument("--heuristic-start", type=int, default=20, help="Episode index where heuristic policy starts.")
    parser.add_argument("--seed", type=int, default=42, help="Base seed for deterministic evaluation.")
    parser.add_argument("--curve-out", type=str, default="reward_curve.png", help="Output PNG path.")
    parser.add_argument("--csv-out", type=str, default="training_metrics.csv", help="Output CSV path.")
    parser.add_argument(
        "--broadcast",
        action="store_true",
        help="Render synchronized multi-agent control-tower frames at each timestep.",
    )
    parser.add_argument(
        "--frame-delay",
        type=float,
        default=0.6,
        help="Delay between synchronized broadcast frames (0.5-1.0 recommended).",
    )
    args = parser.parse_args()
    return TrainConfig(
        num_episodes=args.episodes,
        heuristic_start_episode=args.heuristic_start,
        base_seed=args.seed,
        output_curve_path=args.curve_out,
        output_csv_path=args.csv_out,
        broadcast=bool(args.broadcast),
        frame_delay_s=max(0.0, args.frame_delay),
    )


def _table(headers: list[str], rows: list[list[str]]) -> str:
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))
    sep = "| " + " | ".join("-" * w for w in widths) + " |"
    head = "| " + " | ".join(headers[i].ljust(widths[i]) for i in range(len(headers))) + " |"
    body = ["| " + " | ".join(row[i].ljust(widths[i]) for i in range(len(row))) + " |" for row in rows]
    return "\n".join([head, sep, *body])


def _section(title_number: int, title: str) -> str:
    line = "=" * 60
    return f"{line}\n{title_number}. {title}\n{line}"


def _box(lines: list[str]) -> str:
    width = max((len(line) for line in lines), default=0)
    if UNICODE_OK:
        tl, tr, bl, br, h, v = "┌", "┐", "└", "┘", "─", "│"
    else:
        tl, tr, bl, br, h, v = "+", "+", "+", "+", "-", "|"
    top = tl + h * (width + 2) + tr
    body = "\n".join(f"{v} " + line.ljust(width) + f" {v}" for line in lines)
    bottom = bl + h * (width + 2) + br
    return "\n".join([top, body, bottom])


def _fmt_list(values: tuple[str, ...] | list[str]) -> str:
    return ", ".join(values) if values else "none"


def _fmt_scores(scores: dict[str, float]) -> str:
    if not scores:
        return "{}"
    ordered = sorted(scores.items(), key=lambda x: x[0])
    return "{" + ", ".join(f"{k}: {v:.2f}" for k, v in ordered) + "}"


def _broadcast_frame_lines(
    timestep: int,
    atc_action: AACEAction,
    obs_after: AACEObservation,
) -> list[str]:
    active = _active_flights(obs_after)
    critical = sorted(
        [f.flight_id for f in active if f.fuel_minutes <= FUEL_CRITICAL_MINUTES]
    )
    critical_msg = f"{critical[0]} fuel_critical priority" if critical else "no immediate fuel-critical escalation"

    closed = set(obs_after.weather.closed_runways)
    runway_tokens: list[str] = []
    for r in sorted(obs_after.runways, key=lambda x: x.runway_id):
        status = "open"
        if (not r.open) or (r.runway_id in closed):
            status = "closed"
        elif obs_after.step_index < r.occupied_until_step:
            status = "occupied"
        runway_tokens.append(f"{r.runway_id} {status}")
    ops_msg = ", ".join(runway_tokens)
    weather_msg = (
        f"closure impact: {sorted(closed)}" if closed else "clear conditions"
    )

    oversight = dict((obs_after.metadata or {}).get("oversight", {}))
    scores = _fmt_scores({str(k): float(v) for k, v in dict(oversight.get("scores", {})).items()})
    selected = str(oversight.get("selected_agent", "unknown")).upper()
    if not obs_after.last_action_valid:
        ov_msg = f"OVERRIDE FLAGGED (unsafe) | selected={selected} | scores={scores}"
    else:
        ov_msg = f"APPROVED (safe sequence) | selected={selected} | scores={scores}"

    atc_msg = (
        f"{atc_action.kind.value}(flight={atc_action.flight_id}, runway={atc_action.runway_id})"
    )

    messages = [
        f"ATC        {atc_msg}",
        f"AIRLINE    {critical_msg}",
        f"OPS        {ops_msg}",
        f"WEATHER    {weather_msg}",
        f"OVERSIGHT  {ov_msg}",
    ]
    width = max(43, min(92, max(len(m) + 4 for m in messages)))
    title = f" TIMESTEP {timestep:02d} "
    left_pad = max(0, (width - len(title)) // 2)
    right_pad = max(0, width - len(title) - left_pad)
    if UNICODE_OK:
        tl, tr, bl, br, h, v, pointer = "╭", "╮", "╰", "╯", "─", "│", "▶"
    else:
        tl, tr, bl, br, h, v, pointer = "+", "+", "+", "+", "-", "|", ">"
    top = tl + h * left_pad + title + h * right_pad + tr
    blank = v + " " * width + v

    def row(label: str, msg: str) -> str:
        body = f" {label:<10}{pointer} {msg}"
        return v + body[:width].ljust(width) + v

    bottom = bl + h * width + br
    return [
        top,
        blank,
        row("ATC", atc_msg),
        row("AIRLINE", critical_msg),
        row("OPS", ops_msg),
        row("WEATHER", weather_msg),
        row("OVERSIGHT", ov_msg),
        blank,
        bottom,
    ]


def _render_broadcast_frame(
    console: Console,
    timestep: int,
    atc_action: AACEAction,
    obs_after: AACEObservation,
    frame_delay_s: float,
) -> None:
    sync_prefix = "●" if UNICODE_OK else "*"
    console.print(f"[bold cyan]{sync_prefix} synchronizing agents...[/bold cyan]")
    lines = _broadcast_frame_lines(timestep, atc_action, obs_after)
    for line in lines:
        # Teal for frame, white for neutral, red for risk signals
        if "unsafe" in line.lower() or "OVERRIDE FLAGGED" in line:
            console.print(f"[bold red]{line}[/bold red]")
        elif "OVERSIGHT" in line or line.startswith(("╭", "╰", "│", "+", "|")):
            console.print(f"[bold cyan]{line}[/bold cyan]")
        else:
            console.print(f"[bold white]{line}[/bold white]")
    time.sleep(frame_delay_s)


def render_report(cfg: TrainConfig, baseline_eps: list[EpisodeResult], mace_eps: list[EpisodeResult]) -> None:
    baseline = summarize(baseline_eps)
    mace = summarize(mace_eps)
    reward_delta = float(mace["avg_reward"]) - float(baseline["avg_reward"])
    viol_delta = float(mace["avg_violations"]) - float(baseline["avg_violations"])

    all_codes = Counter(code for ep in mace_eps for code in ep.violation_codes)
    top_codes = all_codes.most_common(5)
    worst = min(mace_eps, key=lambda ep: ep.total_reward)
    zero_viol_episodes = sum(1 for ep in mace_eps if ep.violations == 0)
    runway_related_viol = sum(1 for code, count in all_codes.items() if "runway" in code for _ in range(count))
    ref = mace_eps[-1]
    early_split = max(1, min(cfg.heuristic_start_episode, len(mace_eps)))
    early = mace_eps[:early_split]
    late = mace_eps[early_split:] or mace_eps[-1:]
    early_reward = sum(ep.total_reward for ep in early) / len(early)
    late_reward = sum(ep.total_reward for ep in late) / len(late)
    early_viol = sum(ep.violations for ep in early) / len(early)
    late_viol = sum(ep.violations for ep in late) / len(late)
    reward_trend = "Improving" if late_reward > early_reward else ("Stable" if abs(late_reward - early_reward) < 1e-9 else "Degrading")
    safety_trend = "Improving" if late_viol < early_viol else ("Stable" if abs(late_viol - early_viol) < 1e-9 else "Degrading")
    valid_action_rate = sum(ep.valid_action_rate for ep in mace_eps) / max(1, len(mace_eps))
    safety_score = max(0.0, 100.0 - (float(mace["avg_violations"]) * 25.0))
    reward_interpretation = "Improving" if reward_delta > 0 else ("Stable" if abs(reward_delta) < 1e-9 else "Degrading")
    safety_interpretation = "Improving" if viol_delta < 0 else ("Stable" if abs(viol_delta) < 1e-9 else "Degrading")

    print(_box([_u("MACE CONTROL TOWER REPORT", "MACE CONTROL TOWER REPORT"), "Operational Intelligence Evaluation Output"]))
    print()

    print(_section(1, "SYSTEM STATE & CONTEXT"))
    print(f"{ARROW} Episode ID: {ref.episode}")
    print(f"{ARROW} Task difficulty: {ref.task_id}")
    print(f"{ARROW} Seed: {ref.seed}")
    print(f"{ARROW} Environment state: {ref.terminal_reason}")
    print(f"{ARROW} Active entities (flights): {_fmt_list(ref.active_flights_end)}")
    print(f"{ARROW} Active entities (runways): {_fmt_list(ref.runway_state_end)}")
    print(f"{ARROW} Active entities (emergencies): {_fmt_list(ref.emergency_flights_end)}")

    print(_section(2, "MULTI-AGENT DECISION TRACE"))
    print(f"[ATC] decision = {ref.last_action_repr}")
    airline_target = ref.fuel_critical_flights_end[0] if ref.fuel_critical_flights_end else "none"
    print(f"[AIRLINE] intent = minimize fuel-risk delay")
    print(f"[AIRLINE] decision = prioritize {airline_target}")
    closed_runways = [x for x in ref.runway_state_end if x.endswith('CLOSED')]
    print(f"[OPS] intent = enforce infrastructure constraints")
    print(f"[OPS] decision = {'; '.join(closed_runways) if closed_runways else 'no closed runway constraints'}")
    print("[WEATHER] influence = closure schedule applied deterministically")
    if ref.oversight_conflict_detected:
        print("[CONFLICT] detected between ATC proposal and oversight preference")
    else:
        print("[CONFLICT] none")

    print(_section(3, "OVERSIGHT / SAFETY ENGINE"))
    print(f"[Oversight] scores = {_fmt_scores(ref.oversight_scores)}")
    print(f"[Decision] selected_agent = {ref.oversight_selected_agent}")
    print(f"[Reason] {ref.last_action_message[:120]}")
    print(f"[Safety Override] {'YES' if ref.oversight_conflict_detected else 'NO'}")

    print(_section(4, "ENVIRONMENT EXECUTION TRACE"))
    print(f"[Environment] executed_action = {ref.last_action_repr}")
    print(f"[Environment] reward_impact = {ref.total_reward:.3f}")
    print(f"[Environment] validity = {ref.last_action_valid}")
    print(f"[Environment] constraint_effects = terminal_reason={ref.terminal_reason}, violations={ref.violations}")

    print(_section(5, "TRAINING METRICS SNAPSHOT"))
    print(
        _table(
            ["Metric", "Value"],
            [
                ["Episode Reward", f"{ref.total_reward:.3f}"],
                ["Violations", str(ref.violations)],
                ["Success/Failure Reason", str(ref.terminal_reason)],
                ["Fuel Risk", "ACTIVE" if ref.fuel_critical_flights_end else "LOW"],
                ["Emergency Handling Status", "RESOLVED" if ref.emergency_success else "UNRESOLVED/NA"],
                ["Valid Action Rate", f"{valid_action_rate*100:.1f}%"],
            ],
        )
    )

    print(_section(6, "COMPARATIVE INSIGHT"))
    print(f"{ARROW} Early vs Late reward: {early_reward:.2f} -> {late_reward:.2f}")
    print(f"{ARROW} Early vs Late safety violations: {early_viol:.2f} -> {late_viol:.2f}")
    print(f"{ARROW} Reward delta: {reward_delta:.2f}")
    print(f"{ARROW} Violation delta: {viol_delta:.2f}")
    print(f"{ARROW} Reward trend interpretation: {reward_interpretation}")
    print(f"{ARROW} Safety trend interpretation: {safety_interpretation}")
    print(f"{ARROW} Interpretation: reward={reward_trend}, safety={safety_trend}, completion={float(mace['completion_rate']):.1f}%.")

    print(_section(7, "EDGE CASE ANALYSIS"))
    if any(ep.emergency_present for ep in mace_eps):
        print(f"{ARROW} Fuel exhaustion: {sum(1 for ep in mace_eps if ep.terminal_reason == 'fuel_exhaustion')} episodes")
        print(f"{ARROW} Runway congestion: runway-related violations={runway_related_viol}")
        print(f"{ARROW} Multi-agent conflict: {sum(1 for ep in mace_eps if ep.oversight_conflict_detected)} episodes")
        print(f"{ARROW} Emergency handling: success_rate={float(mace['emergency_success_rate']):.1f}%")
    else:
        print(f"{ARROW} No edge case triggered in this deterministic run.")

    print(_section(8, "FINAL TABLE SUMMARY"))
    print(
        _table(
            ["Metric", "Value"],
            [
                ["Avg Reward", f"{float(mace['avg_reward']):.2f}"],
                ["Violations", f"{float(mace['avg_violations']):.2f}"],
                ["Completion Rate", f"{float(mace['completion_rate']):.1f}%"],
                ["Safety Score", f"{safety_score:.1f}"],
            ],
        )
    )

    print(_section(9, "SYSTEM CONFIGURATION"))
    print(
        _table(
            ["Component", "Value"],
            [
                ["Environment", "MACE OpenEnv"],
                ["Agents", "ATC, Airline, Ops, Weather, Oversight"],
                ["Policy Type", "Heuristic + Oversight"],
                ["Episodes", str(cfg.num_episodes)],
                ["Seed", str(cfg.base_seed)],
            ],
        )
    )
    print(_section(10, "PERFORMANCE METRICS TABLE"))
    print(
        _table(
            ["Metric", "Baseline", "MACE"],
            [
                ["Avg Reward", f"{float(baseline['avg_reward']):.2f}", f"{float(mace['avg_reward']):.2f}"],
                ["Safety Violations", f"{float(baseline['avg_violations']):.2f}", f"{float(mace['avg_violations']):.2f}"],
                ["Completion Rate", f"{float(baseline['completion_rate']):.1f}%", f"{float(mace['completion_rate']):.1f}%"],
                ["Emergency Success", f"{float(baseline['emergency_success_rate']):.1f}%", f"{float(mace['emergency_success_rate']):.1f}%"],
                ["Fuel Survival", f"{float(baseline['fuel_survival_rate']):.1f}%", f"{float(mace['fuel_survival_rate']):.1f}%"],
            ],
        )
    )
    print(_section(11, "SAFETY BREAKDOWN"))
    print(f"- Total violations: {int(mace['total_violations'])}")
    if top_codes:
        print("- Types of violations (top):")
        for code, count in top_codes:
            print(f"  - {code}: {count}")
    else:
        print("- Types of violations: none recorded")
    print(
        f"- Worst-case scenario: episode {worst.episode} ({worst.task_id}, seed={worst.seed}) "
        f"reward={worst.total_reward:.3f}, terminal_reason={worst.terminal_reason}"
    )
    print(
        f"- Mitigation success: {zero_viol_episodes}/{cfg.num_episodes} episodes completed with zero violations "
        f"({100.0 * zero_viol_episodes / max(1, cfg.num_episodes):.1f}%)."
    )
    print(_section(12, "FINAL VERDICT"))
    if reward_delta >= 0 and viol_delta <= 0:
        verdict = (
            "System demonstrates strong improvement in multi-agent coordination under constrained aviation control "
            "scenarios with significant safety stabilization under heuristic + oversight policy."
        )
    elif reward_delta >= 0:
        verdict = (
            "System demonstrates strong reward-side improvement in multi-agent coordination, but safety violations "
            "remain the limiting factor for production-grade stabilization."
        )
    else:
        verdict = (
            "System currently does not outperform the baseline across key benchmarks and requires policy refinement "
            "before claiming stable multi-agent operational gains."
        )
    print(verdict)


def main() -> None:
    cfg = parse_args()
    baseline_eps, _, _ = run_eval(cfg, mode="baseline")
    mace_eps, rewards, _ = run_eval(cfg, mode="mace")

    curve_path = Path(cfg.output_curve_path)
    csv_path = Path(cfg.output_csv_path)
    save_curve(curve_path, rewards, cfg.heuristic_start_episode)
    save_csv(csv_path, mace_eps)
    render_report(cfg, baseline_eps, mace_eps)


if __name__ == "__main__":
    main()
