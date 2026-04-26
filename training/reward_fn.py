"""Training reward utilities for PPO-style policy updates."""

from __future__ import annotations

from dataclasses import dataclass

from env.models import AACEObservation


@dataclass(frozen=True)
class RewardSignal:
    """Decomposed reward signal used by the trainer."""

    total: float
    env_reward: float
    safety_penalty: float
    completion_bonus: float
    valid_action_bonus: float


def compute_training_reward(
    observation: AACEObservation,
    previous_violation_count: int,
) -> RewardSignal:
    """Return a stable scalar reward for RL updates.

    The environment already provides a dense reward; this function adds explicit
    shaping terms for safety and completion so the training loop can optimize
    the same metrics that judges care about.
    """

    env_reward = float(observation.reward or 0.0)
    current_violations = int((observation.metadata or {}).get("violation_count", 0))
    violation_delta = max(0, current_violations - previous_violation_count)

    safety_penalty = -0.40 * float(violation_delta)
    completion_bonus = 2.0 if observation.done and (observation.metadata or {}).get("terminal_reason") == "mission_complete" else 0.0
    valid_action_bonus = 0.10 if observation.last_action_valid else -0.10

    total = env_reward + safety_penalty + completion_bonus + valid_action_bonus
    return RewardSignal(
        total=total,
        env_reward=env_reward,
        safety_penalty=safety_penalty,
        completion_bonus=completion_bonus,
        valid_action_bonus=valid_action_bonus,
    )
