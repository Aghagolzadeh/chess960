from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import random
from typing import Any

import chess
import numpy as np

from learning_backend.chess_core.board import BoardState
from learning_backend.chess_core.rules import game_status
from learning_backend.engines.heuristic_engine import heuristic_score
from learning_backend.envs.action_space import ACTION_SPACE_SIZE, encode_move


@dataclass
class AlphaZeroConfig:
    num_self_play_games_per_iteration: int = 25
    mcts_simulations: int = 50
    temperature_moves: int = 12
    c_puct: float = 1.5
    replay_buffer_size: int = 50_000
    training_batch_size: int = 128
    training_steps_per_iteration: int = 100
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    value_loss_weight: float = 1.0
    policy_loss_weight: float = 1.0
    seed: int = 42


ALPHAZERO_PRESETS: dict[str, AlphaZeroConfig] = {
    "az_debug_fast": AlphaZeroConfig(num_self_play_games_per_iteration=2, mcts_simulations=8, training_steps_per_iteration=2),
    "az_cpu_small": AlphaZeroConfig(num_self_play_games_per_iteration=25, mcts_simulations=25, training_steps_per_iteration=100),
    "az_chess960_medium": AlphaZeroConfig(num_self_play_games_per_iteration=50, mcts_simulations=50, training_steps_per_iteration=250),
    "az_chess960_long": AlphaZeroConfig(num_self_play_games_per_iteration=100, mcts_simulations=100, training_steps_per_iteration=1000, learning_rate=1e-4),
}


class HeuristicPolicyValueNet:
    """Small deterministic policy/value provider used until a neural net is added."""

    def predict(self, board: chess.Board) -> tuple[dict[chess.Move, float], float]:
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return {}, _value_from_white_score(board, heuristic_score(board))

        scores = []
        for move in legal_moves:
            child = board.copy(stack=False)
            child.push(move)
            score = heuristic_score(child)
            scores.append(score if board.turn == chess.WHITE else -score)
        shifted = np.array(scores, dtype=np.float64)
        shifted -= np.max(shifted)
        priors_array = np.exp(shifted)
        priors_array /= priors_array.sum()
        priors = {move: float(priors_array[i]) for i, move in enumerate(legal_moves)}
        value = _value_from_white_score(board, heuristic_score(board))
        return priors, value


def _value_from_white_score(board: chess.Board, score: float) -> float:
    bounded = math.tanh(score / 10.0)
    return bounded if board.turn == chess.WHITE else -bounded


@dataclass
class MCTSNode:
    prior: float
    visit_count: int = 0
    value_sum: float = 0.0
    children: dict[chess.Move, "MCTSNode"] | None = None

    @property
    def value(self) -> float:
        return 0.0 if self.visit_count == 0 else self.value_sum / self.visit_count

    def expanded(self) -> bool:
        return self.children is not None


class MCTS:
    def __init__(
        self,
        network: HeuristicPolicyValueNet | None = None,
        *,
        simulations: int = 50,
        c_puct: float = 1.5,
    ) -> None:
        self.network = network or HeuristicPolicyValueNet()
        self.simulations = simulations
        self.c_puct = c_puct

    def run(self, board: chess.Board) -> MCTSNode:
        root = MCTSNode(prior=1.0)
        self._expand(root, board)
        for _ in range(self.simulations):
            scratch = board.copy(stack=False)
            search_path = [root]
            node = root
            while node.expanded() and node.children:
                move, node = self._select_child(node)
                scratch.push(move)
                search_path.append(node)

            status = game_status(scratch)
            if status["state"] == "checkmate":
                value = -1.0
            elif status["state"] in {"stalemate", "draw"}:
                value = 0.0
            else:
                value = self._expand(node, scratch)
            self._backpropagate(search_path, value)
        return root

    def select_move(self, board: chess.Board, *, temperature: float = 1.0) -> chess.Move:
        root = self.run(board)
        if not root.children:
            raise ValueError("MCTS could not select a move without legal children")
        moves = list(root.children.keys())
        visits = np.array([root.children[move].visit_count for move in moves], dtype=np.float64)
        if temperature <= 1e-6:
            return moves[int(np.argmax(visits))]
        visits = visits ** (1.0 / temperature)
        probs = visits / visits.sum()
        return moves[int(np.random.choice(np.arange(len(moves)), p=probs))]

    def policy_target(self, board: chess.Board, *, temperature: float = 1.0) -> np.ndarray:
        root = self.run(board)
        target = np.zeros(ACTION_SPACE_SIZE, dtype=np.float32)
        if not root.children:
            return target
        visits = np.array([node.visit_count for node in root.children.values()], dtype=np.float64)
        if temperature <= 1e-6:
            visits = np.eye(len(visits))[int(np.argmax(visits))]
        else:
            visits = visits ** (1.0 / temperature)
        visits /= visits.sum()
        for move, probability in zip(root.children.keys(), visits):
            target[encode_move(move)] = float(probability)
        return target

    def _expand(self, node: MCTSNode, board: chess.Board) -> float:
        priors, value = self.network.predict(board)
        node.children = {move: MCTSNode(prior=prior) for move, prior in priors.items()}
        return value

    def _select_child(self, node: MCTSNode) -> tuple[chess.Move, MCTSNode]:
        assert node.children is not None
        total_visits = max(1, node.visit_count)
        best_score = -math.inf
        best_item: tuple[chess.Move, MCTSNode] | None = None
        for move, child in node.children.items():
            exploration = self.c_puct * child.prior * math.sqrt(total_visits) / (1 + child.visit_count)
            score = -child.value + exploration
            if score > best_score:
                best_score = score
                best_item = (move, child)
        assert best_item is not None
        return best_item

    def _backpropagate(self, search_path: list[MCTSNode], value: float) -> None:
        for node in reversed(search_path):
            node.value_sum += value
            node.visit_count += 1
            value = -value


def self_play_game(
    *,
    seed: int = 42,
    position_id: int | None = None,
    config: AlphaZeroConfig | None = None,
    max_plies: int = 240,
) -> dict[str, Any]:
    config = config or AlphaZeroConfig(seed=seed)
    rng = random.Random(seed)
    np.random.seed(seed)
    state = BoardState.new_chess960(seed=seed, position_id=position_id)
    mcts = MCTS(simulations=config.mcts_simulations, c_puct=config.c_puct)
    trajectory: list[dict[str, Any]] = []

    for ply in range(max_plies):
        if state.board.is_game_over(claim_draw=True):
            break
        temperature = 1.0 if ply < config.temperature_moves else 1e-6
        board_before = state.board.copy(stack=False)
        target = mcts.policy_target(board_before, temperature=temperature)
        legal_sum = float(target.sum())
        if legal_sum <= 0:
            break
        legal_actions = [encode_move(move) for move in board_before.legal_moves]
        move = _sample_from_policy_target(board_before, target, rng)
        trajectory.append(
            {
                "fen": board_before.fen(),
                "player": "white" if board_before.turn == chess.WHITE else "black",
                "policy_target": target,
                "legal_actions": legal_actions,
            }
        )
        state.push(move)

    status = game_status(state.board)
    outcome = _outcome_for_white(status)
    for row in trajectory:
        row["value_target"] = outcome if row["player"] == "white" else -outcome
    return {
        "initial_position_id": state.initial_position_id,
        "status": status,
        "trajectory": trajectory,
        "history": state.history,
    }


def _sample_from_policy_target(board: chess.Board, target: np.ndarray, rng: random.Random) -> chess.Move:
    legal_moves = list(board.legal_moves)
    weights = [float(target[encode_move(move)]) for move in legal_moves]
    total = sum(weights)
    if total <= 0:
        return rng.choice(legal_moves)
    threshold = rng.random() * total
    running = 0.0
    for move, weight in zip(legal_moves, weights):
        running += weight
        if running >= threshold:
            return move
    return legal_moves[-1]


def _outcome_for_white(status: dict[str, Any]) -> float:
    if status["state"] == "checkmate":
        return 1.0 if status["winner"] == "white" else -1.0
    return 0.0


def save_alphazero_checkpoint(config: AlphaZeroConfig, path: str | Path, *, extra: dict[str, Any] | None = None) -> Path:
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.write_text(
        json.dumps(
            {
                "algorithm": "alphazero_mcts_scaffold",
                "config": asdict(config),
                "extra": extra or {},
            },
            indent=2,
        )
    )
    return checkpoint_path
