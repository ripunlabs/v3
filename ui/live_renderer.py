"""Stepwise live renderer for judge-facing MACE demos."""

from __future__ import annotations

import time
import sys

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from evaluation.runner import StepRecord
from ui.orchestration_view import action_label, build_orchestration_table


def _supports_unicode() -> bool:
    enc = sys.stdout.encoding or "utf-8"
    try:
        "🟢🟡🔴🧠📊".encode(enc)
        return True
    except Exception:
        return False


UNICODE_OK = _supports_unicode()
PHASE_LABELS: dict[str, tuple[str, str]] = {
    "phase1_stable": (("🟢 " if UNICODE_OK else "[P1] ") + "Phase 1: Stable Airport Operations", "black on green"),
    "phase2_stress": (("🟡 " if UNICODE_OK else "[P2] ") + "Phase 2: Operational Stress Building", "black on yellow"),
    "phase3_conflict": (("🔴 " if UNICODE_OK else "[P3] ") + "Phase 3: Multi-Agent Conflict Detected", "white on red"),
    "phase4_oversight": (
        ("🧠 " if UNICODE_OK else "[P4] ") + "Phase 4: Oversight Arbitration (Safety Decision Engine)",
        "white on blue",
    ),
    "phase5_resolution": (("🟢 " if UNICODE_OK else "[P5] ") + "Phase 5: Safe Resolution & Execution", "black on green"),
    "phase6_learning": (("📊 " if UNICODE_OK else "[P6] ") + "Phase 6: Outcome & System Learning", "black on cyan"),
}


class LiveRenderer:
    def __init__(self, frame_delay: float = 0.6) -> None:
        self.console = Console()
        self.frame_delay = max(0.0, frame_delay)
        self._previous_phase_id: str | None = None
        self._previous_violations: int = 0

    def _progress_bar(self, step: int, total: int, width: int = 34) -> str:
        total_safe = max(1, total)
        ratio = min(1.0, max(0.0, step / total_safe))
        filled = int(width * ratio)
        return "[" + ("#" * filled) + ("-" * (width - filled)) + f"] Step {step}/{total_safe}"

    def render_story_intro(self) -> None:
        self.console.print(
            Panel(
                "Calm -> Stress -> Conflict -> Oversight -> Resolution -> Learning\n"
                "This demo is narrated as a control-tower story arc for judges.",
                title="MACE Cinematic Story Flow",
                border_style="bold cyan",
            )
        )

    def _phase_panel(self, step: StepRecord) -> Panel:
        label, style = PHASE_LABELS.get(step.phase_id, PHASE_LABELS["phase1_stable"])
        progress = self._progress_bar(step.step_index, step.max_steps)
        subtitle = "Conflict visible across agents." if step.conflict_detected else "System remains coordinated."
        return Panel(
            f"{progress}\n{subtitle}",
            title=label,
            style=style,
            border_style="white",
        )

    def _build_oversight_table(self, step: StepRecord) -> Table:
        table = Table(title="Oversight Arbitration Table", expand=True)
        table.add_column("Agent Proposal", style="bold cyan")
        table.add_column("Score", justify="right")
        table.add_column("Status", justify="center")
        table.add_column("Reason")

        for agent_name, score in sorted(step.oversight_scores.items(), key=lambda item: item[0]):
            selected = agent_name == step.oversight_selected_agent
            status = Text("SELECTED", style="bold black on bright_blue") if selected else Text("REJECTED", style="bold red")
            reason = step.oversight_reason if selected else "rejected_by_oversight"
            table.add_row(agent_name.upper(), f"{score:.2f}", status, reason)
        return table

    def _learning_interpretation(self, step: StepRecord) -> str:
        delta = step.violation_count - self._previous_violations
        self._previous_violations = step.violation_count
        if step.valid and step.reward >= 0 and delta <= 0:
            return "System learning signal: stable and safety-aligned progression."
        if not step.valid:
            return "System learning signal: invalid maneuver detected, oversight tightening required."
        if delta > 0:
            return "System learning signal: safety pressure increased, corrective sequencing needed."
        return "System learning signal: mixed step outcome, monitor next arbitration."

    def sync_loading(self) -> None:
        with self.console.status("[bold cyan]Syncing agents...[/bold cyan]", spinner="dots"):
            time.sleep(self.frame_delay)

    def render_step_sequence(self, step: StepRecord, orchestrate: bool = True) -> None:
        if self._previous_phase_id is not None and self._previous_phase_id != step.phase_id:
            self.console.print(
                Panel(
                    "Transitioning to next narrative phase...",
                    title="Story Transition",
                    border_style="bright_white",
                )
            )
            time.sleep(self.frame_delay)
        self._previous_phase_id = step.phase_id

        self.console.print(self._phase_panel(step))
        time.sleep(self.frame_delay)

        # Step 2: ATC decision
        self.console.print(Panel(action_label(step.atc_action), title="Step 2 - ATC Decision", border_style="cyan"))
        time.sleep(self.frame_delay)

        # Step 3: Airline decision
        self.console.print(Panel(action_label(step.airline_action), title="Step 3 - Airline Decision", border_style="yellow"))
        time.sleep(self.frame_delay)

        # Step 4: Ops + Weather
        ops_weather = f"OPS: {action_label(step.ops_action)}\nWEATHER: {step.weather_note}"
        self.console.print(Panel(ops_weather, title="Step 4 - Ops + Weather", border_style="blue"))
        time.sleep(self.frame_delay)

        # Step 5: Oversight arbitration
        oversight_title = Text(f"Step 5 - Oversight Arbitration (selected: {step.oversight_selected_agent.upper()})")
        self.console.print(Panel(step.oversight_reason, title=oversight_title, border_style="magenta"))
        self.console.print(self._build_oversight_table(step))
        time.sleep(self.frame_delay)

        # Step 6: Final environment action
        self.console.print(Panel(action_label(step.final_action), title="Step 6 - Final Environment Action", border_style="green"))
        time.sleep(self.frame_delay)

        # Step 7: Reward update / learning
        reward_line = f"reward={step.reward:.3f} | valid={step.valid} | violations={step.violation_count} | done={step.done}"
        style = "bold green" if step.reward >= 0 else "bold red"
        interpretation = self._learning_interpretation(step)
        payload = Text(reward_line + "\n" + interpretation, style=style)
        title = ("📊 " if UNICODE_OK else "[P6] ") + "Phase 6: Outcome & System Learning"
        self.console.print(Panel(payload, title=title, border_style="bright_cyan"))
        time.sleep(self.frame_delay)

        if orchestrate:
            self.console.print(build_orchestration_table(step))
            time.sleep(self.frame_delay)
