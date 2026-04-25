# Reinforcement Learning Algorithms

Reinforcement learning trains an agent by interaction. The agent observes a state, chooses an action, receives a reward, and updates its policy so future actions become better.

## Environment, Observation, Action, Reward

An environment is the world the agent acts in. In this repo, `Chess960Env` owns a Chess960 game.

An observation is what the agent sees. Here it is a tensor with piece planes and side to move.

An action is what the agent does. Here each action is an encoded chess move.

A reward is feedback. Checkmate creates a win or loss reward; most intermediate positions currently give zero.

The policy chooses actions. The value function estimates expected future reward from a position.

## Why Chess Is Hard For RL

Chess has sparse rewards, long games, many legal actions, and subtle credit assignment. A move on turn 8 may matter only because of a tactic on turn 35. Exploration is difficult because random play rarely discovers strong plans.

## Sparse Reward Problem

If the agent only receives reward at game end, most training steps carry no immediate signal. This can make learning slow or unstable.

## Exploration Problem

The agent must try unfamiliar moves to discover better strategies, but too much randomness prevents stable improvement. Entropy is a useful metric because it shows whether the policy still explores.

## Self-Play and Curriculum

Self-play trains against current or older versions of the same agent. Curriculum learning starts with easier opponents or shorter tasks before increasing difficulty. For this repo, a natural curriculum is random engine, material engine, heuristic engine, then self-play.

## PPO Intuition

PPO improves a policy while limiting update size. It compares the new policy to the old policy and clips the objective when the change is too large. This helps avoid updates that destroy previously learned behavior.

PPO optimizes a clipped policy objective, a value prediction objective, and often an entropy bonus.

## Current Repo Algorithm

`learning_backend.rl.ppo.train_ppo` is a minimal scaffold. It runs seeded Chess960 episodes, records reward-like metrics, and writes JSON checkpoints. It does not yet include a neural policy, a value network, minibatch optimization, or true clipped PPO updates.

## Metrics To Watch

- Reward: terminal outcome signal.
- Win rate: performance against fixed baselines.
- Entropy: policy randomness and exploration.
- KL: approximate policy movement between updates.
- Value loss: value prediction quality.
- Episode length: whether games are short losses, long draws, or improving conversions.

## Is The Agent Learning?

Look for improving win rate against fixed baselines, reward trends that survive different seeds, stable KL, entropy that decreases gradually rather than instantly, and game histories that show coherent plans rather than repeated accidental moves.

## Limitations And Failure Modes

- Sparse rewards can produce flat metrics.
- Exploration can collapse too early.
- A learned agent can overfit one starting-position distribution.
- Handcrafted baselines can leak too much prior chess knowledge.
- JSON checkpoints are placeholders until a neural implementation lands.

## Recommended Next Experiments

- Add a compact PyTorch policy and value network.
- Add material and mobility reward shaping.
- Train against a curriculum of fixed engines.
- Add self-play snapshots.
- Compare standard chess priors with Chess960-aware priors.
