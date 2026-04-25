from __future__ import annotations

import argparse
import json
from pathlib import Path

from learning_backend.rl.alphazero import ALPHAZERO_PRESETS, AlphaZeroConfig, save_alphazero_checkpoint, self_play_game


def train_alphazero(
    *,
    iterations: int = 1,
    preset: str = "az_debug_fast",
    seed: int = 42,
    output_dir: str | Path = "learning_backend/experiments/runs/alphazero_smoke",
) -> Path:
    config = ALPHAZERO_PRESETS[preset]
    config = AlphaZeroConfig(**{**config.__dict__, "seed": seed})
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    log_path = output_path / "self_play_log.jsonl"
    examples = 0
    for iteration in range(1, iterations + 1):
        for game_index in range(config.num_self_play_games_per_iteration):
            game = self_play_game(seed=seed + iteration * 1000 + game_index, config=config)
            examples += len(game["trajectory"])
            with log_path.open("a") as handle:
                handle.write(
                    json.dumps(
                        {
                            "iteration": iteration,
                            "game": game_index,
                            "examples": len(game["trajectory"]),
                            "status": game["status"],
                        }
                    )
                    + "\n"
                )
    return save_alphazero_checkpoint(
        config,
        output_path / "checkpoints" / f"alphazero-iteration-{iterations:06d}.json",
        extra={"examples": examples, "log_path": str(log_path)},
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run AlphaZero-style Chess960 self-play scaffold.")
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--preset", default="az_debug_fast")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="learning_backend/experiments/runs/alphazero_smoke")
    args = parser.parse_args()
    checkpoint = train_alphazero(
        iterations=args.iterations,
        preset=args.preset,
        seed=args.seed,
        output_dir=args.output_dir,
    )
    print(f"Wrote checkpoint: {checkpoint}")


if __name__ == "__main__":
    main()
