from __future__ import annotations

from pathlib import Path


RUNS_DIR = Path("learning_backend/experiments/runs")


def list_experiments() -> dict[str, object]:
    if not RUNS_DIR.exists():
        return {"runs": []}
    return {"runs": sorted(path.name for path in RUNS_DIR.iterdir() if path.is_dir())}
