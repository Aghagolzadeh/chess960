from __future__ import annotations

import random
from typing import Any, Sequence

import chess

from .base import Engine, require_legal_moves
from .material_engine import PIECE_VALUES


class CapturePreferringEngine(Engine):
    name = "capture_preferring"

    def select_move(
        self,
        position: chess.Board,
        legal_moves: Sequence[chess.Move],
        *,
        rng: random.Random | None = None,
        info: dict[str, Any] | None = None,
    ) -> chess.Move:
        require_legal_moves(legal_moves)
        scored = []
        for move in legal_moves:
            captured = position.piece_at(move.to_square)
            value = PIECE_VALUES.get(captured.piece_type, 0.0) if captured else 0.0
            scored.append((value, move))
        best_value = max(value for value, _ in scored)
        best = [move for value, move in scored if value == best_value]
        return (rng or random).choice(best)
