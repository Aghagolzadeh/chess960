from __future__ import annotations

import random
from typing import Any, Sequence

import chess

from .base import Engine, require_legal_moves
from .heuristic_engine import HeuristicEngine


class SunfishEngine(Engine):
    name = "sunfish"

    def __init__(self) -> None:
        self._fallback = HeuristicEngine(depth=2)
        self.available = False

    def select_move(
        self,
        position: chess.Board,
        legal_moves: Sequence[chess.Move],
        *,
        rng: random.Random | None = None,
        info: dict[str, Any] | None = None,
    ) -> chess.Move:
        require_legal_moves(legal_moves)
        return self._fallback.select_move(position, legal_moves, rng=rng, info=info)
