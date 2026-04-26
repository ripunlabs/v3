"""Judge-facing MACE live demo entrypoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rich.console import Console
from rich.panel import Panel

from env.environment import AACEEnvironment
from evaluation.runner import evaluate_policy, run_episode_with_orchestration
from training.ppo_loop import load_policy_from_json
from ui.replay_engine import ReplayEngine

BASE_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run live MACE orchestration demo.")
    parser.add_argument("--frame-delay", type=float, default=0.6, help="Delay between demo frames.")
    parser.add_argument(
        "--stepwise",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable stepwise replay (default: enabled).",
    )
    parser.add_argument(
        "--orchestrate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Render orchestration view with all agents (default: enabled).",
    )
    parser.add_argument("--task", type=str, default="hard", choices=["easy", "medium", "hard"], help="Demo task.")
    parser.add_argument("--seed", type=int, default=42, help="Demo seed.")
    parser.add_argument("--model", type=str, default="training/model_latest.json", help="Model json path.")
    parser.add_argument(
        "--judge-mode",
        action="store_true",
        help="Presentation mode: cleaner spacing, stronger arbitration emphasis, lower noise.",
    )
    parser.add_argument(
        "--compare-baseline",
        action="store_true",
        help="Run baseline summary first, then trained cinematic demo.",
    )
    return parser.parse_args()


def _resolve_model_path(model_arg: str) -> Path:
    if model_arg == "latest":
        return BASE_DIR / "training" / "model_latest.json"
    requested = Path(model_arg)
    if requested.is_absolute():
        return requested
    if requested.exists():
        return requested.resolve()
    return BASE_DIR / requested


def main() -> None:
    args = parse_args()
    console = Console()
    console.print(
        Panel(
            "Check operations commands in: operations.txt",
            title="Operations Commands",
            style="black on white",
            border_style="white",
        )
    )
    console.print(
        Panel(
            "Narrative order is fixed for judges:\n"
            "Calm -> Stress -> Conflict -> Oversight -> Resolution -> Learning",
            title="Storytelling Mode" + (" (Judge Mode)" if args.judge_mode else ""),
            border_style="bold blue",
        )
    )
    model_path = _resolve_model_path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    policy_payload = json.loads(model_path.read_text(encoding="utf-8"))
    policy = load_policy_from_json(policy_payload)

    env = AACEEnvironment()

    if args.compare_baseline:
        baseline_record = evaluate_policy(episodes=1, seed=args.seed, mode="baseline")[0]
        console.print("[bold yellow]Baseline Behavior: reactive, inefficient[/bold yellow]")
        console.print(
            Panel(
                (
                    f"reward={baseline_record.reward:.2f} | violations={baseline_record.violations} | "
                    f"completion={baseline_record.completion} | terminal={baseline_record.terminal_reason}"
                ),
                title="Baseline Snapshot (Fast Comparison)",
                border_style="yellow",
            )
        )
        console.print("[bold green]Trained Behavior: proactive, safety-aware[/bold green]")

    episode = run_episode_with_orchestration(
        env,
        task_id=args.task,
        seed=args.seed,
        trained_policy=policy,
        mode="demo",
    )

    engine = ReplayEngine(
        frame_delay=args.frame_delay,
        stepwise=bool(args.stepwise),
        orchestrate=bool(args.orchestrate),
        judge_mode=bool(args.judge_mode),
    )
    engine.replay(episode)
    console.print(
        "[bold]Story arc:[/bold] conflict, oversight arbitration, safety stabilization, reward updates rendered live."
    )
    console.print(
        "[bold bright_white on blue]Judge takeaway:[/bold bright_white on blue] "
        "MACE turns multi-agent disagreement into safety-first arbitration and stable execution."
    )


if __name__ == "__main__":
    main()
