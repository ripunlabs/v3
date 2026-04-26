"""Rich view for multi-agent orchestration at each timestep."""

from __future__ import annotations

from rich.table import Table
from rich.text import Text

from evaluation.runner import StepRecord


def action_label(action) -> str:
    return f"{action.kind.value}(flight={action.flight_id}, runway={action.runway_id})"


def _impact_level(agent: str, step: StepRecord) -> str:
    if agent == "weather":
        return "Critical" if "none" not in step.weather_note else "Medium"
    if agent == "oversight":
        return "Critical"
    if agent == "airline":
        return "Critical" if "declare_emergency" in action_label(step.airline_action) else "Medium"
    if agent == "atc":
        return "Critical" if step.phase_id in {"phase3_conflict", "phase4_oversight"} else "Medium"
    if agent == "ops":
        return "Critical" if "closed_runways=none" not in step.weather_note else "Medium"
    return "Low"


def build_orchestration_table(step: StepRecord) -> Table:
    table = Table(title=f"Agent Orchestration View - Step {step.step_index}", expand=True)
    table.add_column("Agent", style="bold cyan")
    table.add_column("Decision")
    table.add_column("Score", justify="right")
    table.add_column("Selected", justify="center")
    table.add_column("Role Impact", justify="center")

    rows = [
        ("atc", action_label(step.atc_action)),
        ("airline", action_label(step.airline_action)),
        ("ops", action_label(step.ops_action)),
        ("weather", step.weather_note),
    ]
    if step.trained_action is not None:
        rows.append(("trained", action_label(step.trained_action)))
    for agent, decision in rows:
        selected = agent == step.oversight_selected_agent
        score_value = step.oversight_scores.get(agent)
        score = f"{score_value:.2f}" if score_value is not None else "-"
        marker = Text("YES", style="bold green") if selected else Text("-")
        impact = _impact_level(agent, step)
        table.add_row(agent.upper(), decision, score, marker, impact)

    oversight_selected = Text("SAFETY CLIMAX", style="bold black on bright_blue")
    table.add_row(
        Text("OVERSIGHT", style="bold black on bright_blue"),
        f"selected={step.oversight_selected_agent} | reason={step.oversight_reason}",
        "-",
        oversight_selected,
        _impact_level("oversight", step),
    )
    return table
