from __future__ import annotations

from dataclasses import dataclass, field
import random
from typing import Any

import chess
import numpy as np

from learning_backend.chess_core.board import BoardState
from learning_backend.chess_core.rules import game_status
from learning_backend.engines import Engine
from learning_backend.envs.rewards import RewardBreakdown, RewardConfig, reward_for_transition

from .action_space import decode_action, encode_move, legal_action_mask
from .observation import observe


@dataclass
class Chess960Env:
    seed: int | None = None
    position_id: int | None = None
    max_plies: int = 240
    reward_config: RewardConfig = field(default_factory=RewardConfig)
    reward_debug: bool = False
    agent_color: chess.Color | None = None
    opponent_engine: Engine | None = None

    def __post_init__(self) -> None:
        self.rng = random.Random(self.seed)
        self.state = BoardState.new_chess960(seed=self.seed, position_id=self.position_id)
        self.plies = 0
        self.reward_log: list[dict[str, Any]] = []

    def reset(
        self,
        *,
        seed: int | None = None,
        position_id: int | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if seed is not None:
            self.seed = seed
            self.rng.seed(seed)
        if position_id is not None:
            self.position_id = position_id
        self.state = BoardState.new_chess960(seed=self.seed, position_id=self.position_id)
        self.plies = 0
        self.reward_log = []
        return self.observation(), self.info()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        move = decode_action(action)
        before = self.state.board.copy(stack=False)
        acting_color = self.state.board.turn
        perspective = self.agent_color if self.agent_color is not None else acting_color
        if move not in self.state.board.legal_moves:
            status = game_status(self.state.board)
            breakdown = reward_for_transition(
                before,
                self.state.board,
                move=move,
                acting_color=acting_color,
                perspective=perspective,
                status=status,
                config=self.reward_config,
                illegal=True,
            )
            info = self.info()
            info["illegal_action"] = action
            info["reward_breakdown"] = breakdown.to_dict()
            self._record_reward(breakdown)
            return self.observation(), breakdown.reward, True, False, info

        self.state.push(move)
        self.plies += 1
        status = game_status(self.state.board)
        terminated = status["state"] in {"checkmate", "stalemate", "draw"}
        truncated = self.plies >= self.max_plies and not terminated
        breakdown = reward_for_transition(
            before,
            self.state.board,
            move=move,
            acting_color=acting_color,
            perspective=perspective,
            status=status,
            config=self.reward_config,
            truncated=truncated,
        )
        self._record_reward(breakdown)

        if self.opponent_engine is not None and not (terminated or truncated):
            self._play_opponent_move()
            status = game_status(self.state.board)
            terminated = status["state"] in {"checkmate", "stalemate", "draw"}
            truncated = self.plies >= self.max_plies and not terminated

        info = self.info()
        info["reward_breakdown"] = breakdown.to_dict()
        return self.observation(), breakdown.reward, terminated, truncated, info

    def legal_actions(self) -> list[int]:
        return [encode_move(move) for move in self.state.board.legal_moves]

    def legal_action_mask(self) -> np.ndarray:
        return legal_action_mask(self.state.board)

    def info(self) -> dict[str, Any]:
        return {
            "position": self.state,
            "board": self.state.board,
            "fen": self.state.board.fen(),
            "turn": self.state.turn_name,
            "legal_actions": self.legal_actions(),
            "legal_action_mask": self.legal_action_mask(),
            "status": game_status(self.state.board),
            "plies": self.plies,
            "initial_position_id": self.state.initial_position_id,
            "reward_log": self.reward_log[-20:],
        }

    def observation(self) -> np.ndarray:
        return observe(self.state.board, initial_position_id=self.state.initial_position_id)

    def _record_reward(self, breakdown: RewardBreakdown) -> None:
        if self.reward_debug:
            self.reward_log.append(breakdown.to_dict())

    def _play_opponent_move(self) -> None:
        if self.opponent_engine is None:
            return
        legal_moves = list(self.state.board.legal_moves)
        if not legal_moves:
            return
        move = self.opponent_engine.select_move(self.state.board.copy(stack=False), legal_moves, rng=self.rng)
        self.state.push(move)
        self.plies += 1
