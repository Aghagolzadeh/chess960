from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Sequence

import chess
import numpy as np

from .base import Engine, require_legal_moves
from .heuristic_engine import heuristic_score
from learning_backend.envs.action_space import decode_action, encode_move
from learning_backend.envs.observation import observe


class LearnedEngine(Engine):
    name = "learned"

    def __init__(self, checkpoint_path: str | Path | None = None) -> None:
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        self.metadata: dict[str, Any] = {}
        self.ppo_model = None
        if self.checkpoint_path and self.checkpoint_path.exists():
            self.metadata = json.loads(self.checkpoint_path.read_text())
            if self.metadata.get("algorithm") == "numpy_masked_ppo":
                from learning_backend.rl.ppo import MaskedLinearPolicyValue

                self.ppo_model = MaskedLinearPolicyValue.from_state_dict(self.metadata["model"])

    def select_move(
        self,
        position: chess.Board,
        legal_moves: Sequence[chess.Move],
        *,
        rng: random.Random | None = None,
        info: dict[str, Any] | None = None,
    ) -> chess.Move:
        require_legal_moves(legal_moves)
        if self.ppo_model is not None:
            observation = observe(position)
            mask = np.zeros(self.ppo_model.action_size, dtype=np.int8)
            for move in legal_moves:
                action = encode_move(move)
                if action < self.ppo_model.action_size:
                    mask[action] = 1
            action, _, _, _ = self.ppo_model.select_action(
                observation,
                mask,
                rng=np.random.default_rng((rng or random).randrange(2**32)),
                deterministic=True,
            )
            move = decode_action(action)
            if move in legal_moves:
                return move

        exploration = float(self.metadata.get("exploration", 0.05))
        if (rng or random).random() < exploration:
            return (rng or random).choice(list(legal_moves))

        maximizing = position.turn == chess.WHITE
        best_move = legal_moves[0]
        best_score = float("-inf") if maximizing else float("inf")
        for move in legal_moves:
            board = position.copy(stack=False)
            board.push(move)
            score = heuristic_score(board)
            if maximizing and score > best_score:
                best_move, best_score = move, score
            elif not maximizing and score < best_score:
                best_move, best_score = move, score
        return best_move
