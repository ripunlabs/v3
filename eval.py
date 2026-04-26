"""Evaluate MACE baseline vs trained policy and generate reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from evaluation.runner import evaluate_policy
from evaluation.reporting import summarize, write_metrics_json, write_report_json
from reports.plots import save_reward_curve, save_safety_curve
from training.ppo_loop import load_policy_from_json

BASE_DIR = Path(__file__).resolve().parent


def _supports_unicode() -> bool:
    enc = sys.stdout.encoding or "utf-8"
    try:
        "✅⚠️".encode(enc)
        return True
    except Exception:
        return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate MACE policies and write structured reports.")
    parser.add_argument("--model", type=str, default="latest", help="'latest' or explicit path to model json.")
    parser.add_argument("--episodes", type=int, default=50, help="Evaluation episodes.")
    parser.add_argument("--seed", type=int, default=42, help="Base seed.")
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


def _print_table(console: Console, baseline, trained) -> None:
    table = Table(title="MACE Evaluation (Before vs After)")
    table.add_column("Metric", style="bold cyan")
    table.add_column("Baseline", justify="right")
    table.add_column("Trained", justify="right")
    table.add_row("Avg Reward", f"{baseline.avg_reward:.3f}", f"{trained.avg_reward:.3f}")
    table.add_row("Safety Violations/Episode", f"{baseline.avg_safety_violations:.3f}", f"{trained.avg_safety_violations:.3f}")
    table.add_row("Completion Rate", f"{baseline.completion_rate:.2f}%", f"{trained.completion_rate:.2f}%")
    console.print(table)


def _print_before_after_banner(console: Console) -> None:
    console.print("[bold cyan]BEFORE TRAINING vs AFTER TRAINING[/bold cyan]")


def _print_improvement_status(console: Console, baseline, trained) -> None:
    ok = "✅" if _supports_unicode() else "[OK]"
    warn = "⚠️" if _supports_unicode() else "[WARN]"
    if trained.avg_reward > baseline.avg_reward:
        console.print(f"[bold green]{ok} Training improved performance[/bold green]")
    else:
        console.print(f"[bold yellow]{warn} No improvement detected[/bold yellow]")


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
    model_path = _resolve_model_path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    policy_payload = json.loads(model_path.read_text(encoding="utf-8"))
    trained_policy = load_policy_from_json(policy_payload)

    baseline_records = evaluate_policy(episodes=args.episodes, seed=args.seed, mode="baseline")
    trained_records = evaluate_policy(
        episodes=args.episodes,
        seed=args.seed,
        mode="trained",
        trained_policy=trained_policy,
    )

    baseline = summarize(baseline_records)
    trained = summarize(trained_records)
    _print_before_after_banner(console)
    _print_table(console, baseline, trained)
    _print_improvement_status(console, baseline, trained)

    save_reward_curve(BASE_DIR / "reward_curve.png", [x.reward for x in trained_records])
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
    console.print("[bold green]Wrote:[/bold green] reward_curve.png, reports/metrics.json, reports/report.json")


if __name__ == "__main__":
    main()
