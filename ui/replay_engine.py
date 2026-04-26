"""Replay engine that renders orchestrated multi-agent episodes."""

from __future__ import annotations

from collections import Counter

from rich.panel import Panel

from evaluation.runner import EpisodeRecord
from ui.live_renderer import LiveRenderer


class ReplayEngine:
    def __init__(
        self,
        frame_delay: float = 0.6,
        stepwise: bool = True,
        orchestrate: bool = True,
        judge_mode: bool = False,
    ) -> None:
        self.renderer = LiveRenderer(frame_delay=frame_delay, judge_mode=judge_mode)
        self.stepwise = stepwise
        self.orchestrate = orchestrate
        self.judge_mode = judge_mode

    def replay(self, episode: EpisodeRecord) -> None:
        self.renderer.console.rule(f"[bold]MACE LIVE DEMO - TASK {episode.task_id.upper()}[/bold]")
        self.renderer.render_story_intro()
        for step in episode.steps:
            # Step 1: loading/sync animation
            self.renderer.sync_loading()
            if self.stepwise:
                self.renderer.render_step_sequence(step, orchestrate=self.orchestrate)
            if self.judge_mode:
                self.renderer.console.print()
        self.renderer.console.rule(
            f"[bold green]Episode complete: reward={episode.reward:.2f}, violations={episode.violations}, terminal={episode.terminal_reason}[/bold green]"
        )
        self._render_final_summary(episode)

    def _render_final_summary(self, episode: EpisodeRecord) -> None:
        phase_counts = Counter(step.phase_id for step in episode.steps)
        conflict_steps = sum(1 for step in episode.steps if step.conflict_detected)
        arbitration_steps = sum(1 for step in episode.steps if step.oversight_scores)
        valid_steps = sum(1 for step in episode.steps if step.valid)
        total_steps = max(1, len(episode.steps))
        completion_line = "MISSION COMPLETE" if episode.completion else f"TERMINAL: {episode.terminal_reason.upper()}"
        if episode.terminal_reason == "fuel_exhaustion":
            outcome = "FAILURE"
        elif episode.violations > 0 or not episode.completion:
            outcome = "DEGRADED"
        else:
            outcome = "SAFE"
        key_insight = self._key_insight(episode, conflict_steps, arbitration_steps)
        key_decisions = self._key_decisions(episode)
        safety_improved = episode.violations == 0

        recap_lines = [
            f"Episode Type: {episode.task_id}",
            f"Total Reward: {episode.reward:.2f}",
            f"Safety Violations: {episode.violations}",
            f"Completion Status: {completion_line}",
            f"SYSTEM OUTCOME: {outcome}",
            f"KEY INSIGHT: {key_insight}",
            "",
            (
                "Story Coverage: "
                f"P1={phase_counts.get('phase1_stable', 0)} "
                f"P2={phase_counts.get('phase2_stress', 0)} "
                f"P3={phase_counts.get('phase3_conflict', 0)} "
                f"P4={phase_counts.get('phase4_oversight', 0)} "
                f"P5={phase_counts.get('phase5_resolution', 0)} "
                f"P6={phase_counts.get('phase6_learning', 0)}"
            ),
            f"Arbitration Consistency: {arbitration_steps} oversight moments across {total_steps} steps",
            f"Action Quality: {valid_steps}/{total_steps} valid actions",
        ]
        self.renderer.render_episode_summary(
            total_reward=episode.reward,
            violations=episode.violations,
            completion_status=completion_line,
            key_decisions=key_decisions,
            safety_improved=safety_improved,
        )
        self.renderer.console.print(
            Panel(
                "\n".join(recap_lines),
                title="Extended Judge Recap",
                border_style="bright_blue",
            )
        )

    @staticmethod
    def _key_insight(episode: EpisodeRecord, conflict_steps: int, arbitration_steps: int) -> str:
        if episode.terminal_reason == "fuel_exhaustion":
            return "Crisis exceeded safety envelope; arbitration cues indicate where policy hardening is needed."
        if conflict_steps > 0 and arbitration_steps > 0 and episode.violations == 0:
            return "Oversight prevented unsafe runway allocation during fuel-critical conflict."
        if conflict_steps > 0 and episode.violations > 0:
            return "Oversight reduced escalation, but residual violations show remaining coordination debt."
        return "System maintained stable coordination and converted stress signals into safe execution."

    @staticmethod
    def _key_decisions(episode: EpisodeRecord) -> str:
        picks: list[str] = []
        for step in episode.steps:
            if step.final_action.kind.value != "noop":
                picks.append(
                    f"{step.final_action.kind.value} via {step.oversight_selected_agent}"
                )
            if len(picks) >= 3:
                break
        return "; ".join(picks) if picks else "No major intervention required"
