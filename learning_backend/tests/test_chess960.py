from __future__ import annotations

import chess

from learning_backend.chess_core.board import BoardState
from learning_backend.chess_core.chess960 import generate_chess960_back_rank, validate_chess960_back_rank


def test_all_chess960_positions_are_unique_and_valid() -> None:
    ranks = [generate_chess960_back_rank(position_id) for position_id in range(960)]
    assert len(set(ranks)) == 960
    assert all(validate_chess960_back_rank(rank) for rank in ranks)


def test_new_board_uses_requested_position_id() -> None:
    state = BoardState.new_chess960(position_id=42)
    assert state.initial_position_id == 42
    assert state.board.chess960
    assert len(list(state.board.legal_moves)) > 0


def test_chess960_castling_is_delegated_to_rules_engine() -> None:
    board = chess.Board.empty(chess960=True)
    board.set_piece_at(chess.B1, chess.Piece(chess.KING, chess.WHITE))
    board.set_piece_at(chess.A1, chess.Piece(chess.ROOK, chess.WHITE))
    board.set_piece_at(chess.H1, chess.Piece(chess.ROOK, chess.WHITE))
    board.set_piece_at(chess.B8, chess.Piece(chess.KING, chess.BLACK))
    board.turn = chess.WHITE
    board.castling_rights = chess.BB_A1 | chess.BB_H1

    legal_moves = list(board.legal_moves)
    assert any(board.is_castling(move) for move in legal_moves)
