from __future__ import annotations

import chess

from .heuristic_engine import HeuristicEngine, heuristic_score


class Chess960HeuristicEngine(HeuristicEngine):
    """A handcrafted engine that avoids standard-opening piece-square tables."""

    name = "chess960_heuristic"


def chess960_position_score(board: chess.Board) -> float:
    return heuristic_score(board)
