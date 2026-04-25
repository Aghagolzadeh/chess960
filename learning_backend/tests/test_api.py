from __future__ import annotations

from learning_backend.api.routes_arena import run_arena_route
from learning_backend.api.routes_game import GameStore


def test_game_store_starts_game_and_applies_move() -> None:
    store = GameStore()
    game = store.new_game({"seed": 42, "black_engine": "random"})
    move = game["legal_moves"][0]
    updated = store.move({"game_id": game["game_id"], "uci": move["uci"], "depth": 1})
    assert updated["game_id"] == game["game_id"]
    assert updated["history"]


def test_arena_route_smoke() -> None:
    result = run_arena_route({"white": "random", "black": "random", "games": 1, "seed": 9})
    assert result["games"] == 1
    assert len(result["results"]) == 1
