"""Structured evaluation metrics and report writers."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from evaluation.runner import EpisodeRecord


@dataclass
class EvalSummary:
    avg_reward: float
    avg_safety_violations: float
    completion_rate: float


def summarize(records: list[EpisodeRecord]) -> EvalSummary:
    n = max(1, len(records))
    avg_reward = sum(r.reward for r in records) / n
    avg_violations = sum(r.safety_violations for r in records) / n
    completion_rate = 100.0 * (sum(1 for r in records if r.completion) / n)
    return EvalSummary(
        avg_reward=avg_reward,
        avg_safety_violations=avg_violations,
        completion_rate=completion_rate,
    )


def before_after_rows(baseline: EvalSummary, trained: EvalSummary) -> list[dict[str, str | float]]:
    return [
        {"metric": "avg_reward", "baseline": round(baseline.avg_reward, 4), "trained": round(trained.avg_reward, 4)},
        {
            "metric": "safety_violations_per_episode",
            "baseline": round(baseline.avg_safety_violations, 4),
            "trained": round(trained.avg_safety_violations, 4),
        },
        {"metric": "completion_rate_percent", "baseline": round(baseline.completion_rate, 2), "trained": round(trained.completion_rate, 2)},
    ]


def write_metrics_json(
    *,
    path: Path,
    episodes: int,
    seed: int,
    baseline_records: list[EpisodeRecord],
    trained_records: list[EpisodeRecord],
) -> dict:
    baseline_summary = summarize(baseline_records)
    trained_summary = summarize(trained_records)
    reward_improvement = trained_summary.avg_reward - baseline_summary.avg_reward
    violation_improvement = baseline_summary.avg_safety_violations - trained_summary.avg_safety_violations
    completion_improvement = trained_summary.completion_rate - baseline_summary.completion_rate
    payload = {
        "episodes": episodes,
        "seed": seed,
        "baseline_avg_reward": round(baseline_summary.avg_reward, 4),
        "trained_avg_reward": round(trained_summary.avg_reward, 4),
        "improvement": round(reward_improvement, 4),
        "baseline_violations": round(baseline_summary.avg_safety_violations, 4),
        "trained_violations": round(trained_summary.avg_safety_violations, 4),
        "baseline_completion_rate": round(baseline_summary.completion_rate, 2),
        "trained_completion_rate": round(trained_summary.completion_rate, 2),
        "violation_improvement": round(violation_improvement, 4),
        "completion_improvement": round(completion_improvement, 2),
        "baseline_rewards": [r.reward for r in baseline_records],
        "trained_rewards": [r.reward for r in trained_records],
        "baseline_violations_per_episode": [r.safety_violations for r in baseline_records],
        "trained_violations_per_episode": [r.safety_violations for r in trained_records],
        "completion_rate_trained": trained_summary.completion_rate,
        "before_vs_after": before_after_rows(baseline_summary, trained_summary),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def write_report_json(
    *,
    path: Path,
    baseline_records: list[EpisodeRecord],
    trained_records: list[EpisodeRecord],
) -> dict:
    baseline_summary = summarize(baseline_records)
    trained_summary = summarize(trained_records)
    reward_improvement = trained_summary.avg_reward - baseline_summary.avg_reward
    violation_improvement = baseline_summary.avg_safety_violations - trained_summary.avg_safety_violations
    completion_improvement = trained_summary.completion_rate - baseline_summary.completion_rate
    if reward_improvement > 0:
        improvement_msg = f"Reward improved by {reward_improvement:.2f} points over baseline."
    else:
        improvement_msg = f"Reward dropped by {abs(reward_improvement):.2f} points vs baseline."

    if violation_improvement > 0:
        safety_msg = f"Safety violations reduced by {violation_improvement:.2f} per episode."
    elif violation_improvement < 0:
        safety_msg = f"Safety violations increased by {abs(violation_improvement):.2f} per episode."
    else:
        safety_msg = "Safety violations remained unchanged from baseline."

    if completion_improvement > 0:
        behavior_msg = f"Completion behavior improved by {completion_improvement:.1f}% with more proactive runway decisions."
    elif completion_improvement < 0:
        behavior_msg = f"Completion behavior regressed by {abs(completion_improvement):.1f}%."
    else:
        behavior_msg = "Completion behavior did not change from baseline."

    payload = {
        "title": "MACE Evaluation Report",
        "baseline_summary": asdict(baseline_summary),
        "trained_summary": asdict(trained_summary),
        "comparison_table": before_after_rows(baseline_summary, trained_summary),
        "improvement": improvement_msg,
        "behavior_change": behavior_msg,
        "safety_change": safety_msg,
        "episodes": len(trained_records),
        "judge_ready": True,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload
