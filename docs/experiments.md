# Experiments

Experiments are small, repeatable scripts under `learning_backend/experiments/`.

## Arena Runs

```bash
python3 -m learning_backend.experiments.run_experiment --white chess960_heuristic --black random --games 8 --seed 42
```

Use `--output` to write a JSON result file under `learning_backend/experiments/runs/`.

## Analyze Results

```bash
python3 -m learning_backend.experiments.analyze_results learning_backend/experiments/runs/arena/result.json
```

## Reproducibility

Use explicit seeds for arenas and training. Keep generated outputs under `learning_backend/experiments/runs/`, which is ignored by git.
