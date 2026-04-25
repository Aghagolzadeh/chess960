# Learning Backend

The backend is an importable Python package for Chess960 rules, engines, environments, training, evaluation, experiment logging, and HTTP APIs.

## Setup

```bash
python3 -m pip install -r learning_backend/requirements.txt
```

Run tests:

```bash
pytest
```

Start the backend API:

```bash
python3 -m learning_backend.api.app
```

The API listens on `http://127.0.0.1:8000`.

## Python Usage

```python
from learning_backend.envs.chess960_env import Chess960Env
from learning_backend.engines.heuristic_engine import HeuristicEngine
from learning_backend.rl.train import train_ppo

env = Chess960Env(seed=42)
agent = train_ppo(env)
```

## Arena Evaluation

```bash
python3 -m learning_backend.experiments.run_experiment --white chess960_heuristic --black random --games 8 --seed 42
```

Write results to a file:

```bash
python3 -m learning_backend.experiments.run_experiment --output learning_backend/experiments/runs/arena/result.json
```

## Training

```bash
python3 -m learning_backend.rl.train --updates 3 --preset ppo_debug_fast --seed 42 --output-dir learning_backend/experiments/runs/ppo_smoke
```

Training logs are written as JSON lines under the chosen output directory. Checkpoints include the NumPy policy/value weights, PPO config, and latest metrics.

Verbose training with TensorBoard:

```bash
python3 -m learning_backend.rl.train \
  --updates 10 \
  --preset ppo_debug_fast \
  --output-dir learning_backend/experiments/runs/ppo_debug \
  --tensorboard-log-dir learning_backend/experiments/runs/ppo_debug/tensorboard \
  --eval-every 1 \
  --eval-games 4 \
  --verbose
```

Open TensorBoard:

```bash
tensorboard --logdir learning_backend/experiments/runs/ppo_debug/tensorboard --port 6006
```

Run AlphaZero-style self-play:

```bash
python3 -m learning_backend.rl.train_alphazero --iterations 1 --preset az_debug_fast --seed 42 --output-dir learning_backend/experiments/runs/alphazero_smoke
```

## Resuming Training

PPO checkpoints store model weights and config. Optimizer momentum is not used by the NumPy optimizer, so resuming means loading the checkpoint model and continuing future updates from those weights. The AlphaZero scaffold stores config and run metadata; neural optimizer state is not present yet.

## Evaluation

```bash
python3 -m learning_backend.experiments.evaluate --candidate learned --baseline random --games 100 --seed 42
```

Curriculum evaluation:

```bash
python3 -m learning_backend.experiments.curriculum --candidate learned --seed 42
```

Run all preflight checks:

```bash
python3 -m learning_backend.experiments.rl_sanity_check
```

Diagnose a run:

```bash
python3 -m learning_backend.experiments.diagnose_run learning_backend/experiments/runs/ppo_smoke
```

## Notebook Imports

The included notebook bootstraps `sys.path` automatically. If you create a new notebook, either start Jupyter from the repo root or copy the bootstrap cell from `learning_backend/notebooks/train_chess960_agent.ipynb`.

For a fully visible PPO loop, use `learning_backend/notebooks/train_chess960_ppo_verbose.ipynb`. It prints rollout stats, batch reward/advantage stats, update losses, evaluation score, checkpoint paths, and writes TensorBoard scalars each update.

You can also launch Jupyter with the repo root on `PYTHONPATH`:

```bash
PYTHONPATH=. jupyter notebook learning_backend/notebooks
```

Then imports like `from learning_backend.envs.chess960_env import Chess960Env` work consistently.
