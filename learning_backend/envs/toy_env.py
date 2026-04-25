from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class TwoActionBanditEnv:
    """Tiny masked env where action 1 is always better than action 0."""

    episode_length: int = 1

    def __post_init__(self) -> None:
        self.steps = 0

    def reset(self, *, seed: int | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        self.steps = 0
        return np.array([1.0], dtype=np.float32), self.info()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        self.steps += 1
        reward = 1.0 if action == 1 else -1.0
        terminated = self.steps >= self.episode_length
        return np.array([1.0], dtype=np.float32), reward, terminated, False, self.info()

    def info(self) -> dict[str, Any]:
        return {"legal_action_mask": np.array([1, 1], dtype=np.int8)}


@dataclass
class MaskedToyEnv:
    """Action 2 is illegal; action 1 is the only rewarding legal action."""

    def reset(self, *, seed: int | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        return np.array([1.0, 0.0], dtype=np.float32), self.info()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if action == 2:
            return np.array([1.0, 0.0], dtype=np.float32), -10.0, True, False, {"legal_action_mask": self.info()["legal_action_mask"], "illegal_action": action}
        return np.array([1.0, 0.0], dtype=np.float32), (1.0 if action == 1 else -1.0), True, False, self.info()

    def info(self) -> dict[str, Any]:
        return {"legal_action_mask": np.array([1, 1, 0], dtype=np.int8)}
