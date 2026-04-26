"""Judge-facing MACE live demo entrypoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rich.console import Console
from rich.panel import Panel

from env.environment import AACEEnvironment
from evaluation.runner import run_episode_with_orchestration
from training.ppo_loop import load_policy_from_json
from ui.replay_engine import ReplayEngine


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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    console = Console()
    console.print(
        Panel(
            "Check operations commands in: e:/v3/operations.txt",
            title="Operations Commands",
            style="black on white",
            border_style="white",
        )
    )
    console.print(
        Panel(
            "Narrative order is fixed for judges:\n"
            "Calm -> Stress -> Conflict -> Oversight -> Resolution -> Learning",
            title="Storytelling Mode",
            border_style="bold blue",
        )
    )
    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    policy_payload = json.loads(model_path.read_text(encoding="utf-8"))
    policy = load_policy_from_json(policy_payload)

    env = AACEEnvironment()
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
    )
    engine.replay(episode)
    console.print(
        "[bold]Story arc:[/bold] conflict, oversight arbitration, safety stabilization, reward updates rendered live."
    )


if __name__ == "__main__":
    main()
