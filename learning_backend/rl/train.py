from __future__ import annotations

import argparse

from learning_backend.envs.chess960_env import Chess960Env
from learning_backend.rl.ppo import train_ppo


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the minimal Chess960 PPO scaffold.")
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--updates", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--preset", default="ppo_debug_fast")
    parser.add_argument("--output-dir", default="learning_backend/experiments/runs/ppo_smoke")
    parser.add_argument("--tensorboard-log-dir", default="")
    parser.add_argument("--eval-every", type=int, default=0)
    parser.add_argument("--eval-games", type=int, default=20)
    parser.add_argument("--eval-baseline", default="random")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    env = Chess960Env(seed=args.seed)
    result = train_ppo(
        env,
        episodes=args.episodes,
        updates=args.updates,
        preset=args.preset,
        seed=args.seed,
        output_dir=args.output_dir,
        tensorboard_log_dir=args.tensorboard_log_dir or None,
        eval_every=args.eval_every,
        eval_games=args.eval_games,
        eval_baseline=args.eval_baseline,
        verbose=args.verbose,
    )
    print(f"Wrote checkpoint: {result.checkpoint_path}")


if __name__ == "__main__":
    main()
