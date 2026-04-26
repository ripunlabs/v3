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
    payload = {
        "episodes": episodes,
        "seed": seed,
        "baseline_rewards": [r.reward for r in baseline_records],
        "trained_rewards": [r.reward for r in trained_records],
        "baseline_violations": [r.safety_violations for r in baseline_records],
        "trained_violations": [r.safety_violations for r in trained_records],
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
    payload = {
        "title": "MACE Evaluation Report",
        "baseline_summary": asdict(baseline_summary),
        "trained_summary": asdict(trained_summary),
        "comparison_table": before_after_rows(baseline_summary, trained_summary),
        "episodes": len(trained_records),
        "judge_ready": True,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload
