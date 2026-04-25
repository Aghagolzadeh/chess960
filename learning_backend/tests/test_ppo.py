from __future__ import annotations

import json

import numpy as np

from learning_backend.envs.toy_env import MaskedToyEnv, TwoActionBanditEnv
from learning_backend.rl.ppo import (
    MaskedLinearPolicyValue,
    PPOConfig,
    collect_rollout,
    compute_gae,
    train_ppo,
    update_policy,
)


def test_gae_returns_have_expected_shape() -> None:
    rewards = np.array([1.0, 0.0, 1.0])
    dones = np.array([0.0, 0.0, 1.0])
    values = np.array([0.1, 0.2, 0.3])
    advantages, returns = compute_gae(rewards, dones, values, 0.0, gamma=0.99, gae_lambda=0.95)
    assert advantages.shape == rewards.shape
    assert returns.shape == rewards.shape
    assert returns[-1] == 1.0


def test_ppo_legal_mask_prevents_illegal_action_selection() -> None:
    env = MaskedToyEnv()
    observation, info = env.reset(seed=1)
    model = MaskedLinearPolicyValue(obs_size=2, action_size=3, rng=np.random.default_rng(1))
    for _ in range(50):
        action, _, _, _ = model.select_action(observation, info["legal_action_mask"], rng=np.random.default_rng(2))
        assert action != 2


def test_ppo_learns_tiny_toy_policy() -> None:
    result = train_ppo(
        TwoActionBanditEnv(),
        updates=20,
        preset="ppo_debug_fast",
        seed=7,
        output_dir="/tmp/chess960_test_ppo_toy",
        action_size=2,
    )
    payload = json.loads(result.checkpoint_path.read_text())
    model = MaskedLinearPolicyValue.from_state_dict(payload["model"])
    probs, _, _ = model.distribution(np.array([1.0], dtype=np.float32), np.array([1, 1], dtype=np.int8))
    assert probs[1] > probs[0]


def test_ppo_update_changes_loss_on_fixed_batch() -> None:
    env = TwoActionBanditEnv()
    config = PPOConfig(rollout_steps=64, minibatch_size=16, updates_per_batch=1, learning_rate=1e-2)
    rng = np.random.default_rng(11)
    observation, _ = env.reset(seed=11)
    model = MaskedLinearPolicyValue(obs_size=observation.size, action_size=2, rng=rng)
    batch, _ = collect_rollout(env, model, config, rng=rng)
    first = update_policy(model, batch, config, rng=rng)
    second = update_policy(model, batch, config, rng=rng)
    assert np.isfinite(first["policy_loss"])
    assert np.isfinite(second["policy_loss"])
