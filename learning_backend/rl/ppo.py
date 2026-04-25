from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import random
from typing import Any, Callable, Protocol

import chess
import numpy as np

from learning_backend.chess_core.board import BoardState
from learning_backend.engines import make_engine
from learning_backend.envs.action_space import ACTION_SPACE_SIZE
from learning_backend.envs.action_space import decode_action, encode_move
from learning_backend.envs.chess960_env import Chess960Env
from learning_backend.envs.observation import observe
from learning_backend.experiments.evaluate import evaluate_engines
from learning_backend.experiments.tensorboard_logging import TensorBoardLogger


class MaskedEnv(Protocol):
    def reset(self, *, seed: int | None = None) -> tuple[np.ndarray, dict[str, Any]]: ...

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]: ...


@dataclass
class PPOConfig:
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    entropy_coef: float = 0.02
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    updates_per_batch: int = 3
    minibatch_size: int = 256
    rollout_steps: int = 2048
    normalize_advantages: bool = True
    target_kl: float = 0.03
    seed: int = 42
    hidden_note: str = "NumPy linear masked policy/value model for correctness and smoke tests."


PPO_PRESETS: dict[str, PPOConfig] = {
    "ppo_debug_fast": PPOConfig(rollout_steps=128, minibatch_size=32, updates_per_batch=4, entropy_coef=0.03),
    "ppo_vs_random": PPOConfig(rollout_steps=2048, minibatch_size=256, updates_per_batch=4, entropy_coef=0.03),
    "ppo_vs_material": PPOConfig(rollout_steps=4096, minibatch_size=256, updates_per_batch=3, entropy_coef=0.02),
    "ppo_vs_heuristic": PPOConfig(rollout_steps=8192, minibatch_size=512, updates_per_batch=2, entropy_coef=0.02),
    "ppo_chess960_long": PPOConfig(rollout_steps=8192, minibatch_size=512, updates_per_batch=4, entropy_coef=0.01),
}


@dataclass
class PPOTrainingResult:
    checkpoint_path: Path
    metrics: list[dict[str, Any]]
    agent: dict[str, Any]


@dataclass
class RolloutBatch:
    observations: np.ndarray
    actions: np.ndarray
    old_log_probs: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray
    values: np.ndarray
    masks: np.ndarray
    returns: np.ndarray
    advantages: np.ndarray


@dataclass
class MaskedLinearPolicyValue:
    obs_size: int
    action_size: int
    rng: np.random.Generator
    policy_w: np.ndarray = field(init=False)
    policy_b: np.ndarray = field(init=False)
    value_w: np.ndarray = field(init=False)
    value_b: float = 0.0

    def __post_init__(self) -> None:
        scale = 0.01
        self.policy_w = self.rng.normal(0.0, scale, size=(self.obs_size, self.action_size)).astype(np.float32)
        self.policy_b = np.zeros(self.action_size, dtype=np.float32)
        self.value_w = self.rng.normal(0.0, scale, size=(self.obs_size,)).astype(np.float32)

    def flatten(self, observation: np.ndarray) -> np.ndarray:
        return observation.astype(np.float32).reshape(-1)

    def distribution(self, observation: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
        x = self.flatten(observation)
        logits = x @ self.policy_w + self.policy_b
        probs = masked_softmax(logits, mask)
        value = float(x @ self.value_w + self.value_b)
        return probs, value, x

    def select_action(
        self,
        observation: np.ndarray,
        mask: np.ndarray,
        *,
        rng: np.random.Generator,
        deterministic: bool = False,
    ) -> tuple[int, float, float, float]:
        probs, value, _ = self.distribution(observation, mask)
        if deterministic:
            action = int(np.argmax(probs))
        else:
            action = int(rng.choice(np.arange(len(probs)), p=probs))
        log_prob = float(np.log(max(probs[action], 1e-12)))
        entropy = float(-np.sum(probs[probs > 0] * np.log(probs[probs > 0])))
        return action, log_prob, value, entropy

    def state_dict(self) -> dict[str, Any]:
        return {
            "obs_size": self.obs_size,
            "action_size": self.action_size,
            "policy_w": self.policy_w.tolist(),
            "policy_b": self.policy_b.tolist(),
            "value_w": self.value_w.tolist(),
            "value_b": self.value_b,
        }

    @classmethod
    def from_state_dict(cls, payload: dict[str, Any]) -> "MaskedLinearPolicyValue":
        model = cls(
            obs_size=int(payload["obs_size"]),
            action_size=int(payload["action_size"]),
            rng=np.random.default_rng(0),
        )
        model.policy_w = np.array(payload["policy_w"], dtype=np.float32)
        model.policy_b = np.array(payload["policy_b"], dtype=np.float32)
        model.value_w = np.array(payload["value_w"], dtype=np.float32)
        model.value_b = float(payload["value_b"])
        return model


def masked_softmax(logits: np.ndarray, mask: np.ndarray) -> np.ndarray:
    valid = mask.astype(bool)
    if not np.any(valid):
        raise ValueError("Legal action mask contains no valid actions")
    masked = np.full_like(logits, -1e9, dtype=np.float64)
    masked[valid] = logits[valid]
    masked -= np.max(masked[valid])
    exp = np.zeros_like(masked, dtype=np.float64)
    exp[valid] = np.exp(masked[valid])
    probs = exp / np.sum(exp)
    return probs.astype(np.float64)


def compute_gae(
    rewards: np.ndarray,
    dones: np.ndarray,
    values: np.ndarray,
    last_value: float,
    *,
    gamma: float,
    gae_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    advantages = np.zeros_like(rewards, dtype=np.float64)
    last_advantage = 0.0
    for t in reversed(range(len(rewards))):
        next_non_terminal = 1.0 - dones[t]
        next_value = last_value if t == len(rewards) - 1 else values[t + 1]
        delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
        last_advantage = delta + gamma * gae_lambda * next_non_terminal * last_advantage
        advantages[t] = last_advantage
    returns = advantages + values
    return advantages, returns


def collect_rollout(
    env: MaskedEnv,
    model: MaskedLinearPolicyValue,
    config: PPOConfig,
    *,
    rng: np.random.Generator,
) -> tuple[RolloutBatch, dict[str, float]]:
    observation, info = env.reset(seed=config.seed)
    observations = []
    actions = []
    old_log_probs = []
    rewards = []
    dones = []
    values = []
    masks = []
    entropies = []
    illegal_count = 0
    episode_lengths: list[int] = []
    current_length = 0

    for _ in range(config.rollout_steps):
        mask = np.array(info["legal_action_mask"], dtype=np.int8)
        action, log_prob, value, entropy = model.select_action(observation, mask, rng=rng)
        next_observation, reward, terminated, truncated, next_info = env.step(action)
        observations.append(observation)
        actions.append(action)
        old_log_probs.append(log_prob)
        rewards.append(reward)
        dones.append(float(terminated or truncated))
        values.append(value)
        masks.append(mask)
        entropies.append(entropy)
        illegal_count += 1 if "illegal_action" in next_info else 0
        current_length += 1
        observation, info = next_observation, next_info
        if terminated or truncated:
            episode_lengths.append(current_length)
            current_length = 0
            observation, info = env.reset(seed=config.seed + len(episode_lengths))

    last_mask = np.array(info["legal_action_mask"], dtype=np.int8)
    _, last_value, _ = model.distribution(observation, last_mask)
    rewards_array = np.array(rewards, dtype=np.float64)
    dones_array = np.array(dones, dtype=np.float64)
    values_array = np.array(values, dtype=np.float64)
    advantages, returns = compute_gae(
        rewards_array,
        dones_array,
        values_array,
        last_value,
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
    )
    if config.normalize_advantages and len(advantages) > 1:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    batch = RolloutBatch(
        observations=np.array(observations, dtype=np.float32),
        actions=np.array(actions, dtype=np.int64),
        old_log_probs=np.array(old_log_probs, dtype=np.float64),
        rewards=rewards_array,
        dones=dones_array,
        values=values_array,
        masks=np.array(masks, dtype=np.int8),
        returns=returns,
        advantages=advantages,
    )
    diagnostics = {
        "mean_reward": float(np.mean(rewards_array)),
        "reward_std": float(np.std(rewards_array)),
        "rollout_reward": float(np.sum(rewards_array)),
        "entropy": float(np.mean(entropies)),
        "illegal_moves": float(illegal_count),
        "episode_length": float(np.mean(episode_lengths)) if episode_lengths else 0.0,
        "episodes_completed": float(len(episode_lengths)),
        "advantage_mean": float(np.mean(advantages)),
        "advantage_std": float(np.std(advantages)),
        "return_mean": float(np.mean(returns)),
        "value_mean": float(np.mean(values_array)),
    }
    return batch, diagnostics


def update_policy(
    model: MaskedLinearPolicyValue,
    batch: RolloutBatch,
    config: PPOConfig,
    *,
    rng: np.random.Generator,
) -> dict[str, float]:
    n = len(batch.actions)
    indices = np.arange(n)
    metrics: list[dict[str, float]] = []

    for _ in range(config.updates_per_batch):
        rng.shuffle(indices)
        for start in range(0, n, config.minibatch_size):
            mb = indices[start : start + config.minibatch_size]
            grad_policy_w = np.zeros_like(model.policy_w, dtype=np.float64)
            grad_policy_b = np.zeros_like(model.policy_b, dtype=np.float64)
            grad_value_w = np.zeros_like(model.value_w, dtype=np.float64)
            grad_value_b = 0.0
            approx_kl = []
            clip_hits = []
            policy_losses = []
            value_losses = []
            entropies = []

            for i in mb:
                probs, value, x = model.distribution(batch.observations[i], batch.masks[i])
                action = int(batch.actions[i])
                old_log_prob = batch.old_log_probs[i]
                log_prob = float(np.log(max(probs[action], 1e-12)))
                ratio = float(np.exp(log_prob - old_log_prob))
                advantage = float(batch.advantages[i])
                clipped_ratio = float(np.clip(ratio, 1.0 - config.clip_epsilon, 1.0 + config.clip_epsilon))
                unclipped = ratio * advantage
                clipped = clipped_ratio * advantage
                clipped_active = clipped < unclipped if advantage >= 0 else clipped > unclipped

                grad_logits = np.zeros_like(probs)
                if not clipped_active:
                    grad_logits = advantage * ratio * probs
                    grad_logits[action] -= advantage * ratio

                valid_probs = probs[batch.masks[i].astype(bool)]
                entropy = float(-np.sum(valid_probs * np.log(np.maximum(valid_probs, 1e-12))))
                entropy_grad = np.zeros_like(probs)
                valid = batch.masks[i].astype(bool)
                entropy_grad[valid] = config.entropy_coef * probs[valid] * (np.log(np.maximum(probs[valid], 1e-12)) + entropy)
                grad_logits += entropy_grad

                value_error = value - float(batch.returns[i])
                grad_value = 2.0 * config.value_coef * value_error

                grad_policy_w += np.outer(x, grad_logits)
                grad_policy_b += grad_logits
                grad_value_w += x * grad_value
                grad_value_b += grad_value

                policy_losses.append(float(-min(unclipped, clipped)))
                value_losses.append(float(value_error * value_error))
                approx_kl.append(float(old_log_prob - log_prob))
                clip_hits.append(float(clipped_active))
                entropies.append(entropy)

            scale = 1.0 / max(len(mb), 1)
            grad_policy_w *= scale
            grad_policy_b *= scale
            grad_value_w *= scale
            grad_value_b *= scale
            grad_norm = float(
                np.sqrt(
                    np.sum(grad_policy_w**2)
                    + np.sum(grad_policy_b**2)
                    + np.sum(grad_value_w**2)
                    + grad_value_b**2
                )
            )
            if grad_norm > config.max_grad_norm:
                factor = config.max_grad_norm / (grad_norm + 1e-12)
                grad_policy_w *= factor
                grad_policy_b *= factor
                grad_value_w *= factor
                grad_value_b *= factor

            model.policy_w -= config.learning_rate * grad_policy_w.astype(np.float32)
            model.policy_b -= config.learning_rate * grad_policy_b.astype(np.float32)
            model.value_w -= config.learning_rate * grad_value_w.astype(np.float32)
            model.value_b -= config.learning_rate * grad_value_b

            metrics.append(
                {
                    "policy_loss": float(np.mean(policy_losses)),
                    "value_loss": float(np.mean(value_losses)),
                    "entropy": float(np.mean(entropies)),
                    "kl": float(np.mean(approx_kl)),
                    "clip_fraction": float(np.mean(clip_hits)),
                    "grad_norm": grad_norm,
                }
            )
        if metrics and metrics[-1]["kl"] > config.target_kl:
            break

    return {key: float(np.mean([row[key] for row in metrics])) for key in metrics[0]}


def train_ppo(
    env: MaskedEnv | None = None,
    *,
    episodes: int | None = None,
    updates: int = 3,
    preset: str = "ppo_debug_fast",
    seed: int = 42,
    output_dir: str | Path = "learning_backend/experiments/runs/ppo_smoke",
    action_size: int | None = None,
    tensorboard_log_dir: str | Path | None = None,
    eval_every: int = 0,
    eval_games: int = 20,
    eval_baseline: str = "random",
    verbose: bool = False,
    on_update: Callable[[int, dict[str, Any], MaskedLinearPolicyValue], None] | None = None,
) -> PPOTrainingResult:
    config = PPO_PRESETS[preset]
    config = PPOConfig(**{**asdict(config), "seed": seed})
    if episodes is not None:
        updates = max(1, episodes)
    env = env or Chess960Env(seed=seed)
    rng = np.random.default_rng(seed)
    random.seed(seed)
    observation, info = env.reset(seed=seed)
    inferred_action_size = action_size or len(info["legal_action_mask"])
    model = MaskedLinearPolicyValue(
        obs_size=int(np.prod(observation.shape)),
        action_size=inferred_action_size,
        rng=rng,
    )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    metrics_path = output_path / "training_log.jsonl"
    metrics: list[dict[str, Any]] = []
    checkpoint_path = output_path / "checkpoints" / "ppo-latest.json"
    tb = TensorBoardLogger(tensorboard_log_dir) if tensorboard_log_dir else None

    try:
        for update in range(1, updates + 1):
            batch, rollout_metrics = collect_rollout(env, model, config, rng=rng)
            train_metrics = update_policy(model, batch, config, rng=rng)
            row = {
                "update": update,
                **rollout_metrics,
                **train_metrics,
                "reward": float(np.sum(batch.rewards)),
                "illegal_move_count": int(rollout_metrics["illegal_moves"]),
            }
            if eval_every and update % eval_every == 0:
                eval_metrics = evaluate_ppo_model(
                    model,
                    baseline=eval_baseline,
                    games=eval_games,
                    seed=seed + update * 1000,
                )
                row.update({f"eval_{key}": value for key, value in eval_metrics.items() if isinstance(value, (int, float))})
                if tb:
                    tb.scalars(f"eval/{eval_baseline}", eval_metrics, update)

            metrics.append(row)
            with metrics_path.open("a") as handle:
                handle.write(json.dumps(row) + "\n")
            if tb:
                tb.scalars("train", row, update)
                tb.flush()
            if verbose:
                print(format_update_metrics(row))
            checkpoint_path = save_ppo_checkpoint(
                model,
                config,
                output_path / "checkpoints" / f"ppo-update-{update:06d}.json",
                extra={"update": update, "metrics_path": str(metrics_path), "metrics": row},
            )
            if on_update:
                on_update(update, row, model)
    finally:
        if tb:
            tb.close()

    return PPOTrainingResult(
        checkpoint_path=checkpoint_path,
        metrics=metrics,
        agent={"type": "ppo_numpy", "checkpoint_path": str(checkpoint_path)},
    )


def save_ppo_checkpoint(
    model: MaskedLinearPolicyValue,
    config: PPOConfig,
    path: str | Path,
    *,
    extra: dict[str, Any] | None = None,
) -> Path:
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.write_text(
        json.dumps(
            {
                "algorithm": "numpy_masked_ppo",
                "config": asdict(config),
                "model": model.state_dict(),
                "extra": extra or {},
            }
        )
    )
    return checkpoint_path


def load_ppo_checkpoint(path: str | Path) -> tuple[MaskedLinearPolicyValue, PPOConfig, dict[str, Any]]:
    payload = json.loads(Path(path).read_text())
    return (
        MaskedLinearPolicyValue.from_state_dict(payload["model"]),
        PPOConfig(**payload["config"]),
        payload.get("extra", {}),
    )


def default_chess_action_size() -> int:
    return ACTION_SPACE_SIZE


class PPOPolicyEngine:
    name = "ppo_policy"

    def __init__(
        self,
        model: MaskedLinearPolicyValue,
        *,
        deterministic: bool = True,
        position_id: int = -1,
    ) -> None:
        self.model = model
        self.deterministic = deterministic
        self.position_id = position_id

    def select_move(
        self,
        position: chess.Board,
        legal_moves: list[chess.Move],
        *,
        rng: random.Random | None = None,
        info: dict[str, Any] | None = None,
    ) -> chess.Move:
        if not legal_moves:
            raise ValueError("PPOPolicyEngine cannot select a move without legal moves")
        observation = observe(position, initial_position_id=self.position_id)
        mask = np.zeros(self.model.action_size, dtype=np.int8)
        for move in legal_moves:
            action = encode_move(move)
            if action < self.model.action_size:
                mask[action] = 1
        action, _, _, _ = self.model.select_action(
            observation,
            mask,
            rng=np.random.default_rng((rng or random).randrange(2**32)),
            deterministic=self.deterministic,
        )
        move = decode_action(action)
        if move in legal_moves:
            return move
        return max(
            legal_moves,
            key=lambda legal_move: self.model.distribution(observation, mask)[0][encode_move(legal_move)],
        )


def evaluate_ppo_model(
    model: MaskedLinearPolicyValue,
    *,
    baseline: str = "random",
    games: int = 20,
    seed: int = 42,
) -> dict[str, float]:
    result = evaluate_engines(
        PPOPolicyEngine(model),
        make_engine(baseline),
        games=games,
        seed=seed,
    )
    return {
        "score_percentage": float(result["score_percentage"]),
        "wins": float(result["wins"]),
        "losses": float(result["losses"]),
        "draws": float(result["draws"]),
        "illegal_move_rate": float(result["illegal_move_rate"]),
        "average_game_length": float(result["average_game_length"]),
    }


def format_update_metrics(row: dict[str, Any]) -> str:
    parts = [
        f"update={row.get('update')}",
        f"reward={row.get('reward', 0.0):.3f}",
        f"mean_reward={row.get('mean_reward', 0.0):.4f}",
        f"policy_loss={row.get('policy_loss', 0.0):.4f}",
        f"value_loss={row.get('value_loss', 0.0):.4f}",
        f"entropy={row.get('entropy', 0.0):.4f}",
        f"kl={row.get('kl', 0.0):.5f}",
        f"clip={row.get('clip_fraction', 0.0):.3f}",
        f"illegal={row.get('illegal_move_count', 0)}",
    ]
    if "eval_score_percentage" in row:
        parts.append(f"eval_score={row['eval_score_percentage']:.3f}")
    return " | ".join(parts)
