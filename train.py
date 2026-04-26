"""Train MACE with a minimal PPO-style loop."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from evaluation.runner import evaluate_policy
from evaluation.reporting import summarize, write_metrics_json, write_report_json
from reports.plots import save_reward_curve, save_safety_curve
from training.ppo_loop import PPOConfig, PPOTrainer, model_payload

BASE_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MACE using a minimal PPO-style policy update loop.")
    parser.add_argument("--episodes", type=int, default=200, help="Training episodes.")
    parser.add_argument("--seed", type=int, default=42, help="Base seed.")
    parser.add_argument("--learning-rate", type=float, default=0.05, help="PPO update learning rate.")
    parser.add_argument("--clip-epsilon", type=float, default=0.2, help="PPO clipping epsilon.")
    parser.add_argument("--model-out", type=str, default="training/model_latest.json", help="Trained model path.")
    return parser.parse_args()


def _resolve_output_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return BASE_DIR / path


def _print_comparison(console: Console, baseline, trained) -> None:
    table = Table(title="Baseline vs Trained")
    table.add_column("Metric", style="bold cyan")
    table.add_column("Baseline", justify="right")
    table.add_column("Trained", justify="right")
    table.add_row("Avg Reward", f"{baseline.avg_reward:.3f}", f"{trained.avg_reward:.3f}")
    table.add_row("Safety Violations/Episode", f"{baseline.avg_safety_violations:.3f}", f"{trained.avg_safety_violations:.3f}")
    table.add_row("Completion Rate", f"{baseline.completion_rate:.2f}%", f"{trained.completion_rate:.2f}%")
    console.print(table)


def _print_operations_hint(console: Console) -> None:
    console.print(
        Panel(
            "Check operations commands in: operations.txt",
            title="Operations Commands",
            style="black on white",
            border_style="white",
        )
    )


def main() -> None:
    args = parse_args()
    console = Console()
    _print_operations_hint(console)

    trainer = PPOTrainer(
        PPOConfig(
            episodes=args.episodes,
            seed=args.seed,
            learning_rate=args.learning_rate,
            clip_epsilon=args.clip_epsilon,
        )
    )
    train_history = trainer.train()

    model_path = _resolve_output_path(args.model_out)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.write_text(
        json.dumps(model_payload(trainer.policy, trainer.config), indent=2),
        encoding="utf-8",
    )

    baseline_records = evaluate_policy(episodes=args.episodes, seed=args.seed, mode="baseline")
    trained_records = evaluate_policy(episodes=args.episodes, seed=args.seed, mode="trained", trained_policy=trainer.policy)

    baseline_summary = summarize(baseline_records)
    trained_summary = summarize(trained_records)
    _print_comparison(console, baseline_summary, trained_summary)

    reward_curve = [ep.reward for ep in train_history]
    save_reward_curve(BASE_DIR / "reward_curve.png", reward_curve)
    save_safety_curve(
        BASE_DIR / "reports" / "safety_curve.png",
        [x.safety_violations for x in baseline_records],
        [x.safety_violations for x in trained_records],
    )
    write_metrics_json(
        path=BASE_DIR / "reports" / "metrics.json",
        episodes=args.episodes,
        seed=args.seed,
        baseline_records=baseline_records,
        trained_records=trained_records,
    )
    write_report_json(
        path=BASE_DIR / "reports" / "report.json",
        baseline_records=baseline_records,
        trained_records=trained_records,
    )

    console.print(f"[bold green]Model saved:[/bold green] {model_path}")
    console.print("[bold green]Artifacts:[/bold green] reward_curve.png, reports/metrics.json, reports/report.json")


if __name__ == "__main__":
    main()
