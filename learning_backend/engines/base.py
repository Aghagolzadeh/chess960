from __future__ import annotations

from abc import ABC, abstractmethod
import random
from typing import Any, Sequence

import chess


class Engine(ABC):
    name: str = "engine"

    @abstractmethod
    def select_move(
        self,
        position: chess.Board,
        legal_moves: Sequence[chess.Move],
        *,
        rng: random.Random | None = None,
        info: dict[str, Any] | None = None,
    ) -> chess.Move:
        """Choose one move from the provided legal moves."""


def require_legal_moves(legal_moves: Sequence[chess.Move]) -> None:
    if not legal_moves:
        raise ValueError("Engine cannot select a move without legal moves")
