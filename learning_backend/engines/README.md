# Engines

All engines implement the same interface:

```python
class Engine:
    name: str

    def select_move(self, position, legal_moves, *, rng=None, info=None):
        ...
```

`position` is a `python-chess` board and `legal_moves` is the already validated move list. Engines do not generate rules independently.

## Current Engines

- `RandomEngine`: picks uniformly from legal moves. Stochastic.
- `MaterialEngine`: chooses the move with the best immediate material balance. Deterministic.
- `HeuristicEngine`: depth-limited alpha-beta search using material, center control, and small handcrafted bonuses. Deterministic.
- `Chess960HeuristicEngine`: uses the same search shape while avoiding classical opening piece-square tables that assume the standard starting position. Deterministic.
- `LearnedEngine`: loads a lightweight JSON checkpoint and currently behaves as a heuristic engine with checkpoint-controlled exploration. Experimental.

## A Priori Tuned Engines

The handcrafted engines use chess knowledge supplied before learning: piece values, center preferences, mobility, and search depth. That is a priori tuning. It can be useful as a baseline, but it can also bias results.

Classical piece-square tables are especially risky for Chess960 because they encode assumptions about standard starting squares and opening development. A Chess960 agent may need different priors because bishops, rooks, queen, king, and knights start in many legal layouts.

## Adding an Engine

1. Add a module in `learning_backend/engines/`.
2. Subclass `Engine`.
3. Implement `select_move`.
4. Register it in `learning_backend/engines/__init__.py`.
5. Add a smoke test in `learning_backend/tests/test_engines.py`.

## Running Engines Head to Head

```bash
python3 -m learning_backend.experiments.run_experiment --white heuristic --black random --games 10
```
