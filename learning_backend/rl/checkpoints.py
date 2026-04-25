from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def save_checkpoint(payload: dict[str, Any], path: str | Path) -> Path:
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.write_text(json.dumps(payload, indent=2))
    return checkpoint_path


def load_checkpoint(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())
