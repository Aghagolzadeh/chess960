from __future__ import annotations

from .base import Engine
from .capture_engine import CapturePreferringEngine
from .chess960_engine import Chess960HeuristicEngine
from .heuristic_engine import HeuristicEngine
from .learned_engine import LearnedEngine
from .material_engine import MaterialEngine
from .random_engine import RandomEngine
from .sunfish_engine import SunfishEngine
from .weak_minimax_engine import WeakMinimaxEngine


def make_engine(name: str, **kwargs: object) -> Engine:
    normalized = name.lower().replace("-", "_")
    if normalized == "random":
        return RandomEngine()
    if normalized == "material":
        return MaterialEngine()
    if normalized in {"capture", "capture_preferring"}:
        return CapturePreferringEngine()
    if normalized == "weak_minimax":
        return WeakMinimaxEngine()
    if normalized == "heuristic":
        return HeuristicEngine(depth=int(kwargs.get("depth", 2)))
    if normalized in {"chess960", "chess960_heuristic"}:
        return Chess960HeuristicEngine(depth=int(kwargs.get("depth", 2)))
    if normalized == "learned":
        return LearnedEngine(checkpoint_path=kwargs.get("checkpoint_path"))
    if normalized == "sunfish":
        return SunfishEngine()
    raise ValueError(f"Unknown engine: {name}")


AVAILABLE_ENGINES = [
    "random",
    "capture_preferring",
    "material",
    "weak_minimax",
    "heuristic",
    "chess960_heuristic",
    "learned",
    "sunfish",
]

__all__ = [
    "Engine",
    "RandomEngine",
    "CapturePreferringEngine",
    "MaterialEngine",
    "WeakMinimaxEngine",
    "HeuristicEngine",
    "Chess960HeuristicEngine",
    "LearnedEngine",
    "SunfishEngine",
    "make_engine",
    "AVAILABLE_ENGINES",
]
