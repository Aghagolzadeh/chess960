from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import chess

from learning_backend.engines.material_engine import material_score


@dataclass(frozen=True)
class RewardConfig:
    terminal_win: float = 1.0
    terminal_loss: float = -1.0
    draw: float = 0.0
    illegal_move: float = -1.0
    material_delta_coef: float = 0.01
    truncation_penalty: float = -0.05
    draw_penalty: float = 0.0


@dataclass
class RewardBreakdown:
    move: str | None
    player: str
    reward: float
    terminal_status: dict[str, Any]
    material_before: float
    material_after: float
    material_delta: float
    material_reward: float
    terminal_reward: float
    illegal_reward: float = 0.0
    truncation_reward: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def color_name(color: chess.Color) -> str:
    return "white" if color == chess.WHITE else "black"


def reward_for_transition(
    before: chess.Board,
    after: chess.Board,
    *,
    move: chess.Move | None,
    acting_color: chess.Color,
    perspective: chess.Color,
    status: dict[str, Any],
    config: RewardConfig,
    truncated: bool = False,
    illegal: bool = False,
) -> RewardBreakdown:
    sign = 1.0 if perspective == chess.WHITE else -1.0
    material_before = material_score(before)
    material_after = material_score(after)
    material_delta = sign * (material_after - material_before)
    material_reward = config.material_delta_coef * material_delta

    terminal_reward = 0.0
    if illegal:
        illegal_reward = config.illegal_move
    else:
        illegal_reward = 0.0
        state = status["state"]
        winner = status.get("winner")
        if state == "checkmate":
            terminal_reward = config.terminal_win if winner == color_name(perspective) else config.terminal_loss
        elif state in {"stalemate", "draw", "claimable_draw"}:
            terminal_reward = config.draw + config.draw_penalty

    truncation_reward = config.truncation_penalty if truncated else 0.0
    reward = terminal_reward + material_reward + illegal_reward + truncation_reward
    return RewardBreakdown(
        move=move.uci() if move else None,
        player=color_name(acting_color),
        reward=reward,
        terminal_status=status,
        material_before=material_before,
        material_after=material_after,
        material_delta=material_delta,
        material_reward=material_reward,
        terminal_reward=terminal_reward,
        illegal_reward=illegal_reward,
        truncation_reward=truncation_reward,
    )
