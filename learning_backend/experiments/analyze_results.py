from __future__ import annotations

import argparse
import json
from pathlib import Path


def analyze_arena_file(path: str | Path) -> dict[str, object]:
    payload = json.loads(Path(path).read_text())
    games = payload.get("games", 0)
    scores = payload.get("scores", {})
    return {
        "games": games,
        "scores": scores,
        "draw_rate": (scores.get("draws", 0) / games) if games else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize an arena result JSON file.")
    parser.add_argument("path")
    args = parser.parse_args()
    print(json.dumps(analyze_arena_file(args.path), indent=2))


if __name__ == "__main__":
    main()
