from __future__ import annotations

import math
import random
from typing import Any, Sequence

import chess

from .base import Engine, require_legal_moves
from .material_engine import material_score

MATE_SCORE = 10000.0
CENTER = {chess.D4, chess.E4, chess.D5, chess.E5}
EXTENDED_CENTER = {
    chess.C3,
    chess.D3,
    chess.E3,
    chess.F3,
    chess.C4,
    chess.F4,
    chess.C5,
    chess.F5,
    chess.C6,
    chess.D6,
    chess.E6,
    chess.F6,
}


def heuristic_score(board: chess.Board) -> float:
    if board.is_checkmate():
        return -MATE_SCORE if board.turn == chess.WHITE else MATE_SCORE

    score = material_score(board)
    for square, piece in board.piece_map().items():
        sign = 1 if piece.color == chess.WHITE else -1
        if square in CENTER:
            score += sign * 0.12
        elif square in EXTENDED_CENTER:
            score += sign * 0.05
        if piece.piece_type == chess.BISHOP:
            score += sign * 0.02
        if piece.piece_type == chess.KNIGHT and chess.square_file(square) in {0, 7}:
            score -= sign * 0.08
    return score


def minimax(board: chess.Board, depth: int, alpha: float, beta: float) -> float:
    if depth == 0 or board.is_game_over(claim_draw=True):
        return heuristic_score(board)

    legal_moves = list(board.legal_moves)
    legal_moves.sort(key=lambda move: board.is_capture(move), reverse=True)

    if board.turn == chess.WHITE:
        value = -math.inf
        for move in legal_moves:
            board.push(move)
            value = max(value, minimax(board, depth - 1, alpha, beta))
            board.pop()
            alpha = max(alpha, value)
            if alpha >= beta:
                break
        return value

    value = math.inf
    for move in legal_moves:
        board.push(move)
        value = min(value, minimax(board, depth - 1, alpha, beta))
        board.pop()
        beta = min(beta, value)
        if alpha >= beta:
            break
    return value


class HeuristicEngine(Engine):
    name = "heuristic"

    def __init__(self, depth: int = 2) -> None:
        self.depth = max(1, depth)

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
        best_score = -math.inf if maximizing else math.inf

        ordered_moves = list(legal_moves)
        ordered_moves.sort(key=lambda move: position.is_capture(move), reverse=True)
        for move in ordered_moves:
            board = position.copy(stack=False)
            board.push(move)
            score = minimax(board, self.depth - 1, -math.inf, math.inf)
            if maximizing and score > best_score:
                best_move, best_score = move, score
            elif not maximizing and score < best_score:
                best_move, best_score = move, score
        return best_move
