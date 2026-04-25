from __future__ import annotations

from learning_backend.chess_core.board import BoardState
from learning_backend.envs.action_space import decode_action, encode_move, legal_action_mask


def test_action_round_trip_for_legal_moves() -> None:
    state = BoardState.new_chess960(seed=11)
    for move in list(state.board.legal_moves)[:10]:
        assert decode_action(encode_move(move)) == move


def test_legal_action_mask_marks_legal_moves() -> None:
    state = BoardState.new_chess960(seed=11)
    mask = legal_action_mask(state.board)
    legal_actions = [encode_move(move) for move in state.board.legal_moves]
    assert legal_actions
    assert all(mask[action] == 1 for action in legal_actions)
