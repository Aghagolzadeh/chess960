# Chess960 Research Platform

This repo is a compact Chess960 research playground with two clear halves:

- `frontend/`: browser UI for human play, engine selection, and arena visualization.
- `learning_backend/`: Python package for Chess960 rules, baseline engines, RL environments, training scaffolds, evaluation, experiment logging, and API routes.

The project is intentionally small. Chess rules live in one shared backend layer, engines share one interface, and the frontend consumes backend APIs rather than reimplementing legality.

## Quickstart

Install backend dependencies:

```bash
python3 -m pip install -r learning_backend/requirements.txt
```

Start the backend API:

```bash
python3 -m learning_backend.api.app
```

Start the frontend:

```bash
cd frontend
npm install
npm run dev
```

Open `http://127.0.0.1:5173`, start a new Chess960 game, and choose a black engine from the controls.

Run an arena match:

```bash
python3 -m learning_backend.experiments.run_experiment --white chess960_heuristic --black random --games 4
```

Train PPO:

```bash
python3 -m learning_backend.rl.train --updates 3 --preset ppo_debug_fast --seed 42
```

Train PPO with per-update logs and TensorBoard:

```bash
python3 -m learning_backend.rl.train --updates 10 --preset ppo_debug_fast --seed 42 --output-dir learning_backend/experiments/runs/ppo_debug --tensorboard-log-dir learning_backend/experiments/runs/ppo_debug/tensorboard --eval-every 1 --eval-games 4 --verbose
tensorboard --logdir learning_backend/experiments/runs/ppo_debug/tensorboard --port 6006
```

Run AlphaZero-style self-play:

```bash
python3 -m learning_backend.rl.train_alphazero --iterations 1 --preset az_debug_fast --seed 42
```

Run the RL preflight sanity checks:

```bash
python3 -m learning_backend.experiments.rl_sanity_check
```

Evaluate against a baseline:

```bash
python3 -m learning_backend.experiments.evaluate --candidate heuristic --baseline random --games 100 --seed 42
```

Diagnose a run:

```bash
python3 -m learning_backend.experiments.diagnose_run learning_backend/experiments/runs/ppo_smoke
```

## Repo Map

- `frontend/src/components/`: board and shared UI components.
- `frontend/src/pages/`: browser app pages.
- `frontend/src/arena/`: arena result visualization.
- `frontend/src/api/`: backend API client.
- `learning_backend/chess_core/`: Chess960 board creation, move legality, FEN, and status helpers.
- `learning_backend/engines/`: baseline and learned engine implementations.
- `learning_backend/envs/`: RL environment, action encoding, and observations.
- `learning_backend/rl/`: training, checkpoint, and evaluation entrypoints.
- `learning_backend/experiments/`: arena scripts and experiment configs.
- `learning_backend/api/`: HTTP API consumed by the frontend.
- `docs/`: navigation, Chess960, engine, RL, and experiment documentation.

## Current Status

Works now:

- Chess960 start position generation through `python-chess`
- legal move generation, Chess960 castling, checkmate, stalemate, and draw detection
- human vs engine browser play
- engine vs engine arena matches
- baseline random, material, heuristic, Chess960 heuristic, and learned-engine placeholder
- Gym-like Chess960 environment with action encoding and observations
- NumPy masked PPO implementation with GAE, clipped objective, legal-action masks, diagnostics, and JSON checkpoints
- AlphaZero-style legal-only MCTS and self-play target generation
- curriculum and fixed-seed evaluation utilities
- diagnostic plot and markdown generation
- API smoke tests and backend unit tests

Experimental:

- PPO uses a lightweight NumPy linear policy/value model. It is correct enough for toy learning and plumbing checks, but serious Chess960 learning should move to a neural policy/value implementation.
- AlphaZero uses MCTS with heuristic priors/value. Neural network training is scaffolded but not implemented yet.
- LearnedEngine reads the scaffold checkpoint metadata and uses heuristic move selection with exploration.
- The frontend is intentionally minimal and does not yet include long-running training dashboards.

## Development Principles

- Minimalism: remove dead code and keep modules small.
- Shared code: one backend rules layer owns board state and legal moves.
- Reproducibility: seeds are exposed for games, arenas, and training.
- Clear separation: frontend displays and sends intents; backend owns rules, engines, training, and experiments.
