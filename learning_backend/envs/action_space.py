from __future__ import annotations

import chess
import numpy as np

PROMOTIONS = [None, chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]
ACTION_SPACE_SIZE = 64 * 64 * len(PROMOTIONS)


def encode_move(move: chess.Move) -> int:
    promotion_index = PROMOTIONS.index(move.promotion)
    return (move.from_square * 64 + move.to_square) * len(PROMOTIONS) + promotion_index


def decode_action(action: int) -> chess.Move:
    if not 0 <= action < ACTION_SPACE_SIZE:
        raise ValueError(f"Action must be in [0, {ACTION_SPACE_SIZE})")
    base, promotion_index = divmod(action, len(PROMOTIONS))
    from_square, to_square = divmod(base, 64)
    return chess.Move(from_square, to_square, promotion=PROMOTIONS[promotion_index])


def legal_action_mask(board: chess.Board) -> np.ndarray:
    mask = np.zeros(ACTION_SPACE_SIZE, dtype=np.int8)
    for move in board.legal_moves:
        mask[encode_move(move)] = 1
    return mask
