from __future__ import annotations

import sys
from pathlib import Path
import subprocess


def add_repo_root_to_path(start: str | Path | None = None) -> Path:
    current = Path(start or Path.cwd()).resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "learning_backend" / "__init__.py").exists():
            if str(candidate) not in sys.path:
                sys.path.insert(0, str(candidate))
            return candidate
    raise RuntimeError("Could not find repo root containing learning_backend/__init__.py")


def ensure_backend_dependencies(repo_root: str | Path) -> None:
    try:
        import chess  # noqa: F401
        import numpy  # noqa: F401
    except ModuleNotFoundError:
        requirements = Path(repo_root) / "learning_backend" / "requirements.txt"
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", str(requirements)])
