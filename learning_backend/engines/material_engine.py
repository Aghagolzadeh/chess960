from __future__ import annotations

import random
from typing import Any, Sequence

import chess

from .base import Engine, require_legal_moves

PIECE_VALUES = {
    chess.PAWN: 1.0,
    chess.KNIGHT: 3.0,
    chess.BISHOP: 3.0,
    chess.ROOK: 5.0,
    chess.QUEEN: 9.0,
    chess.KING: 0.0,
}


def material_score(board: chess.Board) -> float:
    score = 0.0
    for piece in board.piece_map().values():
        value = PIECE_VALUES[piece.piece_type]
        score += value if piece.color == chess.WHITE else -value
    return score


class MaterialEngine(Engine):
    name = "material"

    def select_move(
        self,
        position: chess.Board,
        legal_moves: Sequence[chess.Move],
        *,
        rng: random.Random | None = None,
        info: dict[str, Any] | None = None,
    ) -> chess.Move:
        require_legal_moves(legal_moves)
        maximizing = position.turn == chess.WHITE
        best_move = legal_moves[0]
        best_score = float("-inf") if maximizing else float("inf")
        for move in legal_moves:
            board = position.copy(stack=False)
            board.push(move)
            score = material_score(board)
            if maximizing and score > best_score:
                best_move, best_score = move, score
            elif not maximizing and score < best_score:
                best_move, best_score = move, score
        return best_move
