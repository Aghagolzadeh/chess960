from __future__ import annotations

import chess
import numpy as np

PIECE_PLANES = {
    (chess.WHITE, chess.PAWN): 0,
    (chess.WHITE, chess.KNIGHT): 1,
    (chess.WHITE, chess.BISHOP): 2,
    (chess.WHITE, chess.ROOK): 3,
    (chess.WHITE, chess.QUEEN): 4,
    (chess.WHITE, chess.KING): 5,
    (chess.BLACK, chess.PAWN): 6,
    (chess.BLACK, chess.KNIGHT): 7,
    (chess.BLACK, chess.BISHOP): 8,
    (chess.BLACK, chess.ROOK): 9,
    (chess.BLACK, chess.QUEEN): 10,
    (chess.BLACK, chess.KING): 11,
}


def observe(board: chess.Board, *, initial_position_id: int = -1) -> np.ndarray:
    observation = np.zeros((18, 8, 8), dtype=np.float32)
    for square, piece in board.piece_map().items():
        plane = PIECE_PLANES[(piece.color, piece.piece_type)]
        rank = chess.square_rank(square)
        file_index = chess.square_file(square)
        observation[plane, 7 - rank, file_index] = 1.0
    observation[12, :, :] = 1.0 if board.turn == chess.WHITE else 0.0
    observation[13, :, :] = 1.0 if board.has_kingside_castling_rights(chess.WHITE) else 0.0
    observation[14, :, :] = 1.0 if board.has_queenside_castling_rights(chess.WHITE) else 0.0
    observation[15, :, :] = 1.0 if board.has_kingside_castling_rights(chess.BLACK) else 0.0
    observation[16, :, :] = 1.0 if board.has_queenside_castling_rights(chess.BLACK) else 0.0
    if initial_position_id >= 0:
        observation[17, :, :] = initial_position_id / 959.0
    return observation
