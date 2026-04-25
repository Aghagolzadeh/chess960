from __future__ import annotations

import chess

from .board import BoardState


def legal_uci_moves(position: BoardState | chess.Board) -> list[str]:
    board = position.board if isinstance(position, BoardState) else position
    return [move.uci() for move in board.legal_moves]


def legal_move_dicts(position: BoardState | chess.Board) -> list[dict[str, object]]:
    state = position if isinstance(position, BoardState) else BoardState(position, -1)
    return state.to_dict()["legal_moves"]  # type: ignore[return-value]


def apply_uci_move(position: BoardState, uci: str) -> BoardState:
    next_position = position.copy()
    next_position.push_uci(uci)
    return next_position
