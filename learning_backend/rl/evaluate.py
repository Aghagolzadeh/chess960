from __future__ import annotations

import argparse
import json

from learning_backend.engines import LearnedEngine, make_engine
from learning_backend.experiments.run_experiment import run_arena


def evaluate_checkpoint(
    checkpoint_path: str,
    *,
    opponent: str = "random",
    games: int = 4,
    seed: int = 42,
) -> dict[str, object]:
    white = LearnedEngine(checkpoint_path)
    black = make_engine(opponent)
    return run_arena(white, black, games=games, seed=seed)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a learned Chess960 checkpoint.")
    parser.add_argument("checkpoint")
    parser.add_argument("--opponent", default="random")
    parser.add_argument("--games", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    result = evaluate_checkpoint(args.checkpoint, opponent=args.opponent, games=args.games, seed=args.seed)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
