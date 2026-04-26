"""Plot utilities for MACE training/evaluation outputs."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


def _running_average(values: list[float], window: int = 5) -> list[float]:
    if not values:
        return []
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


def save_reward_curve(path: Path, rewards: list[float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(9, 4.5))
    plt.plot(rewards, label="Episode Reward", linewidth=1.8)
    if rewards:
        plt.plot(_running_average(rewards), linestyle="--", linewidth=2.0, label="Running Average")
    plt.title("MACE Reward Curve")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def save_safety_curve(path: Path, baseline_violations: list[int], trained_violations: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(9, 4.5))
    plt.plot(baseline_violations, label="Baseline Safety Violations", linewidth=1.6, alpha=0.75)
    plt.plot(trained_violations, label="Trained Safety Violations", linewidth=1.8)
    plt.title("MACE Safety Violations per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Violations")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
