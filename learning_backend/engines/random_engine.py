from __future__ import annotations

import random
from typing import Any, Sequence

import chess

from .base import Engine, require_legal_moves


class RandomEngine(Engine):
    name = "random"

    def select_move(
        self,
        position: chess.Board,
        legal_moves: Sequence[chess.Move],
        *,
        rng: random.Random | None = None,
        info: dict[str, Any] | None = None,
    ) -> chess.Move:
        require_legal_moves(legal_moves)
        return (rng or random).choice(list(legal_moves))
