from __future__ import annotations

import argparse
import json
from typing import Any

import numpy as np

from learning_backend.chess_core.board import BoardState
from learning_backend.engines import make_engine
from learning_backend.envs.toy_env import MaskedToyEnv, TwoActionBanditEnv
from learning_backend.experiments.evaluate import evaluate_engines
from learning_backend.experiments.run_experiment import play_game
from learning_backend.rl.alphazero import MCTS, self_play_game
from learning_backend.rl.ppo import MaskedLinearPolicyValue, PPO_PRESETS, train_ppo


def run_checks() -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    checks.append(_check_random_complete_game())
    checks.append(_check_heuristic_beats_random())
    checks.append(_check_ppo_toy_learning())
    checks.append(_check_ppo_masking())
    checks.append(_check_alphazero_mcts_legal())
    checks.append(_check_alphazero_self_play_targets())
    checks.append(_check_deterministic_evaluation())
    return checks


def _row(name: str, passed: bool, detail: str, severity: str = "error") -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "severity": severity, "detail": str(detail)}


def _check_random_complete_game() -> dict[str, Any]:
    result = play_game(make_engine("random"), make_engine("random"), seed=1)
    return _row("random_agent_complete_game", bool(result["history"]), f"plies={result['plies']} status={result['status']['state']}")


def _check_heuristic_beats_random() -> dict[str, Any]:
    result = evaluate_engines(make_engine("heuristic"), make_engine("random"), games=20, seed=2)
    return _row(
        "heuristic_beats_random",
        result["score_percentage"] >= 0.55,
        f"score={result['score_percentage']:.2f}; use more games before claiming strength",
        severity="warning",
    )


def _check_ppo_toy_learning() -> dict[str, Any]:
    result = train_ppo(
        TwoActionBanditEnv(),
        updates=20,
        preset="ppo_debug_fast",
        seed=3,
        output_dir="/tmp/chess960_ppo_toy_sanity",
        action_size=2,
    )
    payload = json.loads(result.checkpoint_path.read_text())
    learned = MaskedLinearPolicyValue.from_state_dict(payload["model"])
    probs, _, _ = learned.distribution(np.array([1.0], dtype=np.float32), np.array([1, 1], dtype=np.int8))
    return _row("ppo_learns_toy_policy", probs[1] > probs[0], f"prob(action_1)={probs[1]:.3f}")


def _check_ppo_masking() -> dict[str, Any]:
    env = MaskedToyEnv()
    observation, info = env.reset(seed=4)
    model = MaskedLinearPolicyValue(obs_size=2, action_size=3, rng=np.random.default_rng(4))
    illegal_sampled = False
    for _ in range(100):
        action, _, _, _ = model.select_action(observation, info["legal_action_mask"], rng=np.random.default_rng(5))
        illegal_sampled = illegal_sampled or action == 2
    return _row("ppo_legal_mask_blocks_illegal_action", not illegal_sampled, "action 2 is masked")


def _check_alphazero_mcts_legal() -> dict[str, Any]:
    state = BoardState.new_chess960(seed=5)
    move = MCTS(simulations=4).select_move(state.board, temperature=1e-6)
    return _row("alphazero_mcts_selects_legal_move", move in state.board.legal_moves, move.uci())


def _check_alphazero_self_play_targets() -> dict[str, Any]:
    game = self_play_game(seed=6, max_plies=8)
    valid = bool(game["trajectory"]) and all(abs(float(row["policy_target"].sum()) - 1.0) < 1e-5 for row in game["trajectory"])
    return _row("alphazero_self_play_targets_valid", valid, f"examples={len(game['trajectory'])}")


def _check_deterministic_evaluation() -> dict[str, Any]:
    first = evaluate_engines(make_engine("material"), make_engine("random"), games=4, seed=7)
    second = evaluate_engines(make_engine("material"), make_engine("random"), games=4, seed=7)
    return _row("evaluation_fixed_seed_stable", first["results"] == second["results"], "fixed-seed repeated evaluation")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RL pipeline sanity checks.")
    parser.parse_args()
    checks = run_checks()
    print(json.dumps({"checks": checks, "passed": all(row["passed"] or row["severity"] == "warning" for row in checks)}, indent=2))


if __name__ == "__main__":
    main()
