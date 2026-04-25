# Reinforcement Learning

Reinforcement learning is a way to train an agent by letting it act in an environment and improve from rewards. In this repo, the environment is Chess960: the agent sees a board, chooses a legal move, receives a reward, and repeats until the game ends.

## Core Terms

- Environment: the system the agent interacts with. Here it is `Chess960Env`.
- Observation: the data the agent sees. Here it is an 18-plane tensor with pieces, side to move, castling rights, and normalized Chess960 start id.
- Action: a move encoded as an integer.
- Reward: feedback after a move. Checkmate gives a win/loss reward; most non-terminal moves are zero.
- Policy: the move-selection rule the agent follows.
- Value function: an estimate of how good a position is.

## Why Chess Is Hard

Chess has sparse rewards: most moves do not immediately win or lose. It also has a huge action space, long credit-assignment chains, and difficult exploration. An agent may need thousands or millions of games before a signal becomes clear.

Chess960 adds useful pressure because opening memorization is reduced. The agent must learn transferable principles rather than one fixed opening layout.

## Self-Play and Curriculum Learning

Self-play trains an agent against versions of itself. Curriculum learning starts with easier tasks or weaker opponents and gradually increases difficulty. Both are common ways to make chess RL less brittle.

## PPO Intuition

PPO, or Proximal Policy Optimization, improves a policy while limiting how far each update can move from the previous policy. The clipping objective discourages destructive jumps. PPO usually tracks reward, entropy, approximate KL divergence, value loss, and episode length.

## Current PPO Algorithm

`train_ppo` now implements masked PPO with old log-prob storage, GAE, clipped policy objective, value loss, entropy bonus, approximate KL, clip fraction, gradient clipping, legal-action masking, JSON checkpoints, and JSONL metrics.

The policy/value model is intentionally small and NumPy-based because this machine currently fails on `torch` import due duplicate OpenMP runtime loading. The implementation is scientifically useful for toy learning, masking, reward, checkpoint, and diagnostic tests. It is not expected to beat strong chess engines without replacing the linear policy/value model with a neural network.

## Logging And TensorBoard

PPO writes JSONL metrics to `training_log.jsonl`. With `--tensorboard-log-dir`, it also writes TensorBoard scalars without importing Torch.

Useful tags include:

- `train/policy_loss`
- `train/value_loss`
- `train/entropy`
- `train/kl`
- `train/clip_fraction`
- `train/illegal_move_count`
- `eval/<baseline>/score_percentage`
- `eval/<baseline>/wins`
- `eval/<baseline>/losses`
- `eval/<baseline>/draws`

Run:

```bash
python3 -m learning_backend.rl.train --updates 10 --preset ppo_debug_fast --tensorboard-log-dir learning_backend/experiments/runs/ppo_debug/tensorboard --eval-every 1 --eval-games 4 --verbose
tensorboard --logdir learning_backend/experiments/runs/ppo_debug/tensorboard --port 6006
```

For notebook work, open `learning_backend/notebooks/train_chess960_ppo_verbose.ipynb`.

## AlphaZero-Style Training

`learning_backend.rl.alphazero` provides legal-only MCTS, visit-count policy targets, player-perspective value labels, and self-play trajectory generation. It currently uses heuristic priors/value estimates instead of a trained neural network. This validates the data pipeline before adding network training.

## Metrics To Watch

- Reward: terminal outcome signal.
- Win rate: fraction of evaluation games won.
- Entropy: policy randomness; collapsing too early can harm exploration.
- KL: how far the policy changes between updates.
- Value loss: how well the value function predicts returns.
- Episode length: short losses and long aimless games both reveal problems.

## Reading Learning Signals

An agent is probably learning if reward and win rate improve against fixed baselines while entropy decreases gradually and KL stays controlled. If reward is flat, entropy collapses, or games end by repeated illegal moves, the training loop is learning the wrong thing or not receiving enough signal.

## Current Limitations

- No neural policy or value network is included yet.
- PPO checkpoints include NumPy weights; AlphaZero checkpoints are still metadata and config.
- Rewards are sparse and simple.
- There is no self-play league.

## Recommended Next Experiments

- Add a small PyTorch policy/value network.
- Add reward shaping for material and checkmate distance.
- Train first against `random`, then `material`, then `chess960_heuristic`.
- Compare Chess960-aware priors against standard chess piece-square tables.

## Named PPO Presets

- `ppo_debug_fast`: tiny smoke checks.
- `ppo_vs_random`: first real curriculum stage.
- `ppo_vs_material`: longer rollouts against material baselines.
- `ppo_vs_heuristic`: conservative settings for harder opponents.
- `ppo_chess960_long`: longer sparse-reward experiments.

## Named AlphaZero Presets

- `az_debug_fast`: MCTS/self-play smoke checks.
- `az_cpu_small`: CPU-feasible early runs.
- `az_chess960_medium`: larger self-play batches.
- `az_chess960_long`: longer runs with lower learning rate.
