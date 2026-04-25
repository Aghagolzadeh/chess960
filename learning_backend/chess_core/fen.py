from __future__ import annotations

import chess

from .board import BoardState


def to_fen(position: BoardState | chess.Board) -> str:
    board = position.board if isinstance(position, BoardState) else position
    return board.fen()


def from_fen(fen: str, *, chess960: bool = True, position_id: int = -1) -> BoardState:
    board = chess.Board(fen=fen, chess960=chess960)
    return BoardState(board=board, initial_position_id=position_id)
