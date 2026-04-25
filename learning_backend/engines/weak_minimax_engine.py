from __future__ import annotations

from .heuristic_engine import HeuristicEngine


class WeakMinimaxEngine(HeuristicEngine):
    name = "weak_minimax"

    def __init__(self) -> None:
        super().__init__(depth=1)
