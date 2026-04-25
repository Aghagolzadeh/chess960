# Engines Overview

The shared engine interface is:

```python
class Engine:
    name: str

    def select_move(self, position, legal_moves, *, rng=None, info=None):
        ...
```

The rules layer provides `legal_moves`, so engines are responsible only for choosing among valid moves.

## Available Engines

- `random`: stochastic legal-move baseline.
- `material`: deterministic one-ply material baseline.
- `heuristic`: deterministic alpha-beta search with handcrafted material and positional priors.
- `chess960_heuristic`: deterministic Chess960-aware baseline that avoids standard opening piece-square tables.
- `learned`: experimental checkpoint-backed engine.

## Determinism

`random` is stochastic unless seeded through the arena. `material`, `heuristic`, and `chess960_heuristic` are deterministic. `learned` can include exploration based on checkpoint metadata.

## A Priori Tuning

A priori tuned means the engine receives chess knowledge before learning: piece values, center bonuses, search depth, and tactical ordering. These are useful baselines but can bias research conclusions.

Classical chess piece-square tables can be biased for Chess960 because they assume standard starting positions and standard opening development patterns. A knight on a corner may still be bad, but many other standard-opening preferences are less reliable when the back rank is randomized.

## Add a New Engine

Add a file in `learning_backend/engines/`, subclass `Engine`, register it in `learning_backend/engines/__init__.py`, and add tests. Keep the engine from duplicating move generation.

## Run an Engine Match

```bash
python3 -m learning_backend.experiments.run_experiment --white chess960_heuristic --black material --games 8
```
