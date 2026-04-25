# Repo Navigation

The repo has two main top-level concepts:

- `frontend/`: browser UI, dashboard, human play, and arena visualization.
- `learning_backend/`: Python package for rules, engines, environments, RL, experiments, and APIs.

## Frontend

Frontend code belongs in `frontend/src/`.

- `components/`: reusable UI like the board and move history.
- `pages/`: top-level app screens.
- `arena/`: arena controls and summaries.
- `api/`: backend client calls.
- `styles/`: CSS.

Add a dashboard page by creating a page module, wiring it in `src/pages/app.js`, and consuming backend endpoints through `src/api/client.js`.

## Learning Backend

- `chess_core/`: all board rules, Chess960 generation, FEN, move helpers, and status.
- `engines/`: every engine that implements the shared interface.
- `envs/`: RL environments, observation encoding, and action encoding.
- `rl/`: algorithms, checkpointing, training, and evaluation.
- `experiments/`: arena scripts, result analysis, and configs.
- `api/`: HTTP routes consumed by the frontend.
- `notebooks/`: importable research notebooks.
- `tests/`: backend tests.

Experiment outputs go under `learning_backend/experiments/runs/`, which is ignored by git.

## Add a New Engine

Create a module in `learning_backend/engines/`, subclass `Engine`, register it in `engines/__init__.py`, document it in `engines/README.md`, and add a test.

## Add a New RL Algorithm

Create a module in `learning_backend/rl/`, expose a training function with explicit seeds and output directories, write metrics as JSON lines, and add a CLI entrypoint.
