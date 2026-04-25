from __future__ import annotations

import chess

from learning_backend.chess_core.board import BoardState
from learning_backend.rl.alphazero import MCTS, self_play_game


def test_mcts_never_selects_illegal_move() -> None:
    state = BoardState.new_chess960(seed=12)
    move = MCTS(simulations=8).select_move(state.board, temperature=1e-6)
    assert move in state.board.legal_moves


def test_policy_target_sums_to_one_over_legal_moves() -> None:
    state = BoardState.new_chess960(seed=12)
    target = MCTS(simulations=8).policy_target(state.board)
    legal_indices = [index for index, value in enumerate(target) if value > 0]
    assert abs(float(target.sum()) - 1.0) < 1e-6
    assert len(legal_indices) <= len(list(state.board.legal_moves))


def test_self_play_produces_valid_trajectory_targets() -> None:
    game = self_play_game(seed=13, max_plies=6)
    assert game["trajectory"]
    for row in game["trajectory"]:
        assert abs(float(row["policy_target"].sum()) - 1.0) < 1e-6
        assert row["value_target"] in {-1.0, 0.0, 1.0}


def test_value_target_sign_flips_by_player() -> None:
    game = {
        "trajectory": [
            {"player": "white"},
            {"player": "black"},
        ]
    }
    for row in game["trajectory"]:
        row["value_target"] = 1.0 if row["player"] == "white" else -1.0
    assert game["trajectory"][0]["value_target"] == -game["trajectory"][1]["value_target"]
