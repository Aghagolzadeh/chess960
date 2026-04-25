from __future__ import annotations

from typing import Any

from learning_backend.engines import make_engine
from learning_backend.experiments.run_experiment import run_arena


def run_arena_route(payload: dict[str, Any]) -> dict[str, Any]:
    white = make_engine(payload.get("white", "chess960_heuristic"), depth=int(payload.get("depth", 2)))
    black = make_engine(payload.get("black", "random"), depth=int(payload.get("depth", 2)))
    return run_arena(
        white,
        black,
        games=int(payload.get("games", 4)),
        seed=int(payload.get("seed", 42)),
    )
