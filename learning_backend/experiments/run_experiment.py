from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
from typing import Any

import chess

from learning_backend.chess_core.board import BoardState
from learning_backend.chess_core.rules import game_status
from learning_backend.engines import Engine, make_engine


def play_game(
    white_engine: Engine,
    black_engine: Engine,
    *,
    seed: int,
    position_id: int | None = None,
    max_plies: int = 240,
) -> dict[str, Any]:
    rng = random.Random(seed)
    state = BoardState.new_chess960(seed=seed, position_id=position_id)
    engines = {chess.WHITE: white_engine, chess.BLACK: black_engine}
    plies = 0
    snapshots = [
        {
            "ply": 0,
            "fen": state.board.fen(),
            "turn": state.turn_name,
            "move": None,
            "san": None,
            "uci": None,
            "pieces": state.piece_map_for_ui(),
        }
    ]

    while not state.board.is_game_over(claim_draw=True) and plies < max_plies:
        legal_moves = list(state.board.legal_moves)
        move = engines[state.board.turn].select_move(state.board.copy(stack=False), legal_moves, rng=rng)
        san = state.board.san(move)
        uci = move.uci()
        state.push(move)
        plies += 1
        snapshots.append(
            {
                "ply": plies,
                "fen": state.board.fen(),
                "turn": state.turn_name,
                "move": f"{san} ({uci})",
                "san": san,
                "uci": uci,
                "pieces": state.piece_map_for_ui(),
            }
        )

    status = game_status(state.board)
    if plies >= max_plies and status["state"] == "ongoing":
        status = {"state": "max_plies", "winner": None, "check": state.board.is_check(), "result": "1/2-1/2"}

    return {
        "initial_position_id": state.initial_position_id,
        "white_engine": white_engine.name,
        "black_engine": black_engine.name,
        "plies": plies,
        "status": status,
        "history": state.history,
        "snapshots": snapshots,
        "final_fen": state.board.fen(),
    }


def run_arena(
    white_engine: Engine,
    black_engine: Engine,
    *,
    games: int = 4,
    seed: int = 42,
) -> dict[str, Any]:
    results = []
    scores = {"white_wins": 0, "black_wins": 0, "draws": 0}
    for game_index in range(games):
        result = play_game(white_engine, black_engine, seed=seed + game_index)
        results.append(result)
        winner = result["status"]["winner"]
        if winner == "white":
            scores["white_wins"] += 1
        elif winner == "black":
            scores["black_wins"] += 1
        else:
            scores["draws"] += 1
    return {"games": games, "scores": scores, "results": results}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a Chess960 arena match.")
    parser.add_argument("--white", default="chess960_heuristic")
    parser.add_argument("--black", default="random")
    parser.add_argument("--games", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    result = run_arena(make_engine(args.white), make_engine(args.black), games=args.games, seed=args.seed)
    encoded = json.dumps(result, indent=2)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(encoded)
    print(encoded)


if __name__ == "__main__":
    main()
