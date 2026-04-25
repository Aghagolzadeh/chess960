from __future__ import annotations

import argparse
import json
import math
import random
from typing import Any

import chess

from learning_backend.chess_core.board import BoardState
from learning_backend.engines import Engine, make_engine


def wilson_interval(wins: float, total: int, z: float = 1.96) -> tuple[float, float]:
    if total == 0:
        return (0.0, 0.0)
    p = wins / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * total)) / total) / denominator
    return (max(0.0, center - margin), min(1.0, center + margin))


def elo_delta(score: float) -> float | None:
    if score <= 0.0 or score >= 1.0:
        return None
    return -400.0 * math.log10(1.0 / score - 1.0)


def evaluate_engines(
    candidate: Engine,
    baseline: Engine,
    *,
    games: int = 100,
    seed: int = 42,
    start_ids: list[int] | None = None,
) -> dict[str, Any]:
    start_ids = start_ids or [i % 960 for i in range(max(1, games // 2))]
    results = []
    wins = losses = draws = illegal_moves = total_plies = 0
    side_split = {"white": {"wins": 0, "losses": 0, "draws": 0}, "black": {"wins": 0, "losses": 0, "draws": 0}}
    by_start: dict[str, dict[str, int]] = {}

    for game_index in range(games):
        rng = random.Random(seed + game_index)
        candidate_color = chess.WHITE if game_index % 2 == 0 else chess.BLACK
        position_id = start_ids[(game_index // 2) % len(start_ids)]
        state = BoardState.new_chess960(position_id=position_id)
        engines = {
            candidate_color: candidate,
            not candidate_color: baseline,
        }
        plies = 0
        while not state.board.is_game_over(claim_draw=True) and plies < 240:
            legal = list(state.board.legal_moves)
            move = engines[state.board.turn].select_move(state.board.copy(stack=False), legal, rng=rng)
            if move not in legal:
                illegal_moves += 1
                break
            state.push(move)
            plies += 1

        result = state.board.result(claim_draw=True)
        total_plies += plies
        side = "white" if candidate_color == chess.WHITE else "black"
        start_key = str(position_id)
        by_start.setdefault(start_key, {"wins": 0, "losses": 0, "draws": 0})
        if result == "1/2-1/2" or result == "*":
            draws += 1
            side_split[side]["draws"] += 1
            by_start[start_key]["draws"] += 1
            outcome = "draw"
        else:
            white_won = result == "1-0"
            candidate_won = white_won == (candidate_color == chess.WHITE)
            if candidate_won:
                wins += 1
                side_split[side]["wins"] += 1
                by_start[start_key]["wins"] += 1
                outcome = "win"
            else:
                losses += 1
                side_split[side]["losses"] += 1
                by_start[start_key]["losses"] += 1
                outcome = "loss"
        results.append({"game": game_index, "position_id": position_id, "candidate_color": side, "result": result, "outcome": outcome, "plies": plies})

    score = (wins + 0.5 * draws) / games if games else 0.0
    ci_low, ci_high = wilson_interval(wins + 0.5 * draws, games)
    return {
        "candidate": candidate.name,
        "baseline": baseline.name,
        "games": games,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "score_percentage": score,
        "score_confidence_interval": [ci_low, ci_high],
        "estimated_elo_delta": elo_delta(score),
        "average_game_length": total_plies / games if games else 0.0,
        "side_split": side_split,
        "performance_by_start_id": by_start,
        "illegal_move_rate": illegal_moves / games if games else 0.0,
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a candidate engine against a baseline.")
    parser.add_argument("--candidate", default="chess960_heuristic")
    parser.add_argument("--baseline", default="random")
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    result = evaluate_engines(make_engine(args.candidate), make_engine(args.baseline), games=args.games, seed=args.seed)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
