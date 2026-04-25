# RL Learning Audit

## What Was Wrong

- PPO was a placeholder stochastic loop, not PPO. It did not store old log probabilities, compute GAE, use clipped policy updates, optimize a value function, report KL/clip fraction, or checkpoint model state.
- AlphaZero-style training did not exist in the refactored repo.
- Rewards were effectively White-centric. That makes training incorrect when the learning agent plays Black or alternates colors.
- Observations omitted castling rights and the Chess960 start id.
- Evaluation was too noisy and did not alternate colors, fix starting positions, report confidence intervals, or track illegal move rate.
- There was no preflight command to prove masking, self-play target generation, toy PPO learning, or deterministic evaluation before long runs.

## What Was Fixed

- Added explicit reward configuration and reward breakdown logging with move, player, terminal status, material delta, shaped reward, illegal penalty, and truncation penalty.
- Expanded observations to include pieces, side to move, castling rights, and normalized Chess960 position id.
- Added a NumPy masked PPO implementation with GAE, clipped objective, entropy, KL, clip fraction, gradient clipping, value loss, and checkpoints.
- Added toy environments and tests that prove PPO can learn a trivial masked policy.
- Added AlphaZero-style MCTS, legal expansion, visit-count policy targets, temperature handling, value labels from the current player perspective, and self-play trajectories.
- Added curriculum stages, fixed-seed evaluation, confidence intervals, side split, start-id split, illegal move rate, and Elo estimates.
- Added diagnostics that generate plots and a markdown diagnosis summary.
- Added `rl_sanity_check` as the required pre-long-run checklist.

## PPO Config Recommendation

Start with:

```bash
python3 -m learning_backend.rl.train --updates 3 --preset ppo_debug_fast --seed 42
```

For the first real stage:

```bash
python3 -m learning_backend.rl.train --updates 50 --preset ppo_vs_random --seed 42 --output-dir learning_backend/experiments/runs/ppo_vs_random
```

The preset uses the requested stable PPO shape: learning rate around `3e-4`, gamma `0.99`, GAE lambda `0.95`, clip epsilon `0.2`, entropy bonus, value coefficient `0.5`, gradient clipping, advantage normalization, and target KL.

## AlphaZero Config Recommendation

Start with:

```bash
python3 -m learning_backend.rl.train_alphazero --iterations 1 --preset az_debug_fast --seed 42
```

Then move to:

```bash
python3 -m learning_backend.rl.train_alphazero --iterations 10 --preset az_cpu_small --seed 42 --output-dir learning_backend/experiments/runs/az_cpu_small
```

Do not compare AlphaZero to Sunfish until MCTS self-play targets are valid and the agent beats random/material baselines.

## Success Criteria

- Stage 0: agent selects only legal moves.
- Stage 1: agent beats random over 100 games.
- Stage 2: agent beats capture-preferring and material engines.
- Stage 3: agent is competitive with weak minimax depth-1.
- Stage 4: agent is competitive with heuristic baseline.
- Stage 5: agent can take games from Sunfish.

## Required Commands

Run sanity checks:

```bash
python3 -m learning_backend.experiments.rl_sanity_check
```

Run curriculum evaluation:

```bash
python3 -m learning_backend.experiments.curriculum --candidate learned --seed 42 --output learning_backend/experiments/runs/curriculum.json
```

Evaluate against baselines:

```bash
python3 -m learning_backend.experiments.evaluate --candidate learned --baseline random --games 100 --seed 42
python3 -m learning_backend.experiments.evaluate --candidate learned --baseline material --games 100 --seed 42
python3 -m learning_backend.experiments.evaluate --candidate learned --baseline heuristic --games 200 --seed 42
python3 -m learning_backend.experiments.evaluate --candidate learned --baseline sunfish --games 200 --seed 42
```

Diagnose a run:

```bash
python3 -m learning_backend.experiments.diagnose_run learning_backend/experiments/runs/ppo_vs_random
```

## Interpretation

Do not claim learning from 16-game batches. Use at least 100 games for random/material stages and 200 games for stronger baselines. Watch illegal move rate first, then score percentage with confidence intervals, then entropy, KL, policy loss, value loss, material balance, and episode length.
