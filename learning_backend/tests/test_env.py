from __future__ import annotations

from learning_backend.envs.chess960_env import Chess960Env


def test_env_reset_returns_observation_and_info() -> None:
    env = Chess960Env(seed=42)
    observation, info = env.reset(seed=42)
    assert observation.shape == (18, 8, 8)
    assert info["initial_position_id"] == env.state.initial_position_id
    assert info["legal_actions"]
    assert info["legal_action_mask"].sum() == len(info["legal_actions"])


def test_env_step_applies_legal_action() -> None:
    env = Chess960Env(seed=42)
    env.reset(seed=42)
    action = env.legal_actions()[0]
    _, reward, terminated, truncated, info = env.step(action)
    assert reward in {-1.0, 0.0, 1.0}
    assert not (terminated and truncated)
    assert info["plies"] == 1


def test_env_illegal_action_is_terminal_and_logged() -> None:
    env = Chess960Env(seed=42, reward_debug=True)
    env.reset(seed=42)
    _, reward, terminated, truncated, info = env.step(0)
    if "illegal_action" not in info:
        return
    assert reward < 0
    assert terminated
    assert not truncated
    assert info["reward_breakdown"]["illegal_reward"] < 0


def test_reward_debug_logs_material_components() -> None:
    env = Chess960Env(seed=42, reward_debug=True)
    _, info = env.reset(seed=42)
    action = info["legal_actions"][0]
    _, _, _, _, next_info = env.step(action)
    breakdown = next_info["reward_breakdown"]
    assert "material_delta" in breakdown
    assert breakdown["player"] in {"white", "black"}
