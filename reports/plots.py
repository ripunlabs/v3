"""Plot utilities for MACE training/evaluation outputs."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


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
    x = list(range(len(rewards)))
    plt.plot(x, rewards, label="Episode Reward", linewidth=1.8)
    if rewards:
        running = _running_average(rewards)
        plt.plot(x, running, linestyle="--", linewidth=2.0, label="Running Average")
        if len(rewards) >= 2:
            coeffs = np.polyfit(np.array(x), np.array(rewards), 1)
            slope, intercept = float(coeffs[0]), float(coeffs[1])
            trend = [slope * i + intercept for i in x]
            trend_label = f"Trend (upward)" if slope >= 0 else "Trend (downward)"
            plt.plot(x, trend, linestyle=":", linewidth=2.0, label=trend_label)
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
    bx = list(range(len(baseline_violations)))
    tx = list(range(len(trained_violations)))
    plt.plot(bx, baseline_violations, label="Baseline Safety Violations", linewidth=1.6, alpha=0.75)
    plt.plot(tx, trained_violations, label="Trained Safety Violations", linewidth=1.8)
    if len(trained_violations) >= 2:
        coeffs = np.polyfit(np.array(tx), np.array(trained_violations), 1)
        slope, intercept = float(coeffs[0]), float(coeffs[1])
        trend = [slope * i + intercept for i in tx]
        trend_label = "Trained Trend (downward)" if slope <= 0 else "Trained Trend (upward)"
        plt.plot(tx, trend, linestyle=":", linewidth=2.0, label=trend_label)
    plt.title("MACE Safety Violations per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Violations")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
