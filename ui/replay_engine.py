"""Replay engine that renders orchestrated multi-agent episodes."""

from __future__ import annotations

from evaluation.runner import EpisodeRecord
from ui.live_renderer import LiveRenderer


class ReplayEngine:
    def __init__(self, frame_delay: float = 0.6, stepwise: bool = True, orchestrate: bool = True) -> None:
        self.renderer = LiveRenderer(frame_delay=frame_delay)
        self.stepwise = stepwise
        self.orchestrate = orchestrate

    def replay(self, episode: EpisodeRecord) -> None:
        self.renderer.console.rule(f"[bold]MACE LIVE DEMO - TASK {episode.task_id.upper()}[/bold]")
        self.renderer.render_story_intro()
        for step in episode.steps:
            # Step 1: loading/sync animation
            self.renderer.sync_loading()
            if self.stepwise:
                self.renderer.render_step_sequence(step, orchestrate=self.orchestrate)
        self.renderer.console.rule(
            f"[bold green]Episode complete: reward={episode.reward:.2f}, violations={episode.violations}, terminal={episode.terminal_reason}[/bold green]"
        )
