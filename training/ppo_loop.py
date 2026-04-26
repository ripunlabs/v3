"""Minimal PPO-style trainer (TRL/Unsloth-compatible structure)."""

from __future__ import annotations

from dataclasses import dataclass

from env.environment import AACEEnvironment
from env.models import ActionKind
from training.policies import PreferencePolicy
from training.reward_fn import compute_training_reward

TASK_CYCLE = ("easy", "medium", "hard")


@dataclass
class PPOConfig:
    episodes: int = 50
    seed: int = 42
    learning_rate: float = 0.05
    clip_epsilon: float = 0.2
    gamma: float = 0.99


@dataclass
class TrainEpisode:
    episode: int
    reward: float
    violations: int
    completion: bool


class PPOTrainer:
    """Tiny PPO-style trainer over discrete action preferences.

    This is intentionally lightweight and deterministic, and mirrors the shape
    expected by TRL/Unsloth style workflows:
    - collect trajectory
    - compute returns/advantages
    - apply clipped policy update
    """

    def __init__(self, config: PPOConfig, policy: PreferencePolicy | None = None) -> None:
        self.config = config
        self.policy = policy or PreferencePolicy.default()

    def train(self) -> list[TrainEpisode]:
        env = AACEEnvironment()
        history: list[TrainEpisode] = []

        for episode in range(self.config.episodes):
            task_id = TASK_CYCLE[episode % len(TASK_CYCLE)]
            obs = env.reset(seed=self.config.seed + episode, task_id=task_id)
            done = False

            chosen_action_kinds: list[str] = []
            old_probs: list[float] = []
            rewards: list[float] = []
            total_reward = 0.0
            previous_violations = int((obs.metadata or {}).get("violation_count", 0))

            while not done:
                probs = self.policy.action_probs(obs)
                action = self.policy.choose_action(obs)
                selected_kind = action.kind.value
                chosen_action_kinds.append(selected_kind)
                old_probs.append(max(1e-8, probs.get(selected_kind, 1e-8)))

                obs = env.step(action)
                signal = compute_training_reward(obs, previous_violation_count=previous_violations)
                previous_violations = int((obs.metadata or {}).get("violation_count", previous_violations))
                rewards.append(signal.total)
                total_reward += signal.total
                done = bool(obs.done)

            returns = self._discounted_returns(rewards)
            if returns:
                baseline = sum(returns) / len(returns)
                advantages = [r - baseline for r in returns]
            else:
                advantages = []

            self._ppo_update(chosen_action_kinds, old_probs, advantages)
            terminal_reason = str((obs.metadata or {}).get("terminal_reason", ""))
            violations = int((obs.metadata or {}).get("violation_count", 0))
            history.append(
                TrainEpisode(
                    episode=episode,
                    reward=total_reward,
                    violations=violations,
                    completion=(terminal_reason == "mission_complete"),
                )
            )

        return history

    def _discounted_returns(self, rewards: list[float]) -> list[float]:
        out = [0.0 for _ in rewards]
        running = 0.0
        for idx in range(len(rewards) - 1, -1, -1):
            running = rewards[idx] + self.config.gamma * running
            out[idx] = running
        return out

    def _ppo_update(self, action_kinds: list[str], old_probs: list[float], advantages: list[float]) -> None:
        if not action_kinds:
            return
        for kind, old_p, adv in zip(action_kinds, old_probs, advantages):
            current_pref = self.policy.preferences.get(kind, 0.0)
            ratio = 1.0
            if old_p > 0:
                # Logit preference is used as proxy for policy probability movement.
                new_p_proxy = max(1e-8, old_p + (current_pref * 0.01))
                ratio = new_p_proxy / old_p
            clipped_ratio = min(max(ratio, 1.0 - self.config.clip_epsilon), 1.0 + self.config.clip_epsilon)
            objective = clipped_ratio * adv
            step = self.config.learning_rate * max(-1.0, min(1.0, objective))
            self.policy.preferences[kind] = current_pref + step


def load_policy_from_json(payload: dict) -> PreferencePolicy:
    prefs = payload.get("preferences", {})
    if not isinstance(prefs, dict):
        return PreferencePolicy.default()

    clean = PreferencePolicy.default().preferences.copy()
    for key in clean:
        if key in prefs:
            clean[key] = float(prefs[key])
    return PreferencePolicy(preferences=clean)


def model_payload(policy: PreferencePolicy, config: PPOConfig) -> dict:
    return {
        "policy_type": "minimal_ppo_preference",
        "preferences": dict(policy.preferences),
        "learning_rate": config.learning_rate,
        "clip_epsilon": config.clip_epsilon,
        "gamma": config.gamma,
        "tracked_action_kinds": [k.value for k in ActionKind],
        "trl_ready": True,
        "unsloth_compatible": True,
    }
