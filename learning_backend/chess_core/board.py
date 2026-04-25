from __future__ import annotations

from dataclasses import dataclass, field
import random

import chess

from .chess960 import random_chess960_position_id
from .rules import game_status


PIECE_TO_INT = {
    chess.PAWN: 1,
    chess.KNIGHT: 2,
    chess.BISHOP: 3,
    chess.ROOK: 4,
    chess.QUEEN: 5,
    chess.KING: 6,
}


@dataclass
class BoardState:
    board: chess.Board
    initial_position_id: int
    history: list[dict[str, str]] = field(default_factory=list)

    @classmethod
    def new_chess960(
        cls,
        *,
        seed: int | None = None,
        position_id: int | None = None,
    ) -> "BoardState":
        if position_id is None:
            position_id = random_chess960_position_id(seed)
        board = chess.Board.from_chess960_pos(position_id)
        board.chess960 = True
        return cls(board=board, initial_position_id=position_id)

    def copy(self) -> "BoardState":
        return BoardState(
            board=self.board.copy(stack=True),
            initial_position_id=self.initial_position_id,
            history=[entry.copy() for entry in self.history],
        )

    @property
    def turn_name(self) -> str:
        return "white" if self.board.turn == chess.WHITE else "black"

    def legal_moves(self) -> list[chess.Move]:
        return list(self.board.legal_moves)

    def push(self, move: chess.Move) -> None:
        san = self.board.san(move)
        uci = move.uci()
        self.board.push(move)
        self.history.append({"san": san, "uci": uci})

    def push_uci(self, uci: str) -> chess.Move:
        move = self.board.parse_uci(uci)
        if move not in self.board.legal_moves:
            raise ValueError(f"Illegal move: {uci}")
        self.push(move)
        return move

    def random_legal_move(self, rng: random.Random | None = None) -> chess.Move:
        moves = self.legal_moves()
        if not moves:
            raise ValueError("No legal moves available")
        return (rng or random).choice(moves)

    def to_matrix(self) -> list[list[int]]:
        matrix: list[list[int]] = []
        for rank in range(7, -1, -1):
            row: list[int] = []
            for file_index in range(8):
                piece = self.board.piece_at(chess.square(file_index, rank))
                if piece is None:
                    row.append(0)
                    continue
                value = PIECE_TO_INT[piece.piece_type]
                row.append(value if piece.color == chess.WHITE else -value)
            matrix.append(row)
        return matrix

    def piece_map_for_ui(self) -> list[dict[str, str]]:
        pieces: list[dict[str, str]] = []
        for square, piece in self.board.piece_map().items():
            pieces.append(
                {
                    "square": chess.square_name(square),
                    "piece": piece.symbol(),
                    "color": "white" if piece.color == chess.WHITE else "black",
                    "symbol": piece.unicode_symbol(),
                }
            )
        return pieces

    def to_dict(self) -> dict[str, object]:
        status = game_status(self.board)
        return {
            "fen": self.board.fen(),
            "chess960": True,
            "initial_position_id": self.initial_position_id,
            "turn": self.turn_name,
            "matrix": self.to_matrix(),
            "pieces": self.piece_map_for_ui(),
            "legal_moves": [
                {
                    "uci": move.uci(),
                    "from": chess.square_name(move.from_square),
                    "to": chess.square_name(move.to_square),
                    "promotion": chess.piece_symbol(move.promotion) if move.promotion else None,
                    "san": self.board.san(move),
                    "is_capture": self.board.is_capture(move),
                    "is_castling": self.board.is_castling(move),
                }
                for move in self.board.legal_moves
            ],
            "history": self.history,
            "status": status,
        }


def new_chess960_board(seed: int | None = None, position_id: int | None = None) -> BoardState:
    return BoardState.new_chess960(seed=seed, position_id=position_id)
