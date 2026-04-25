from __future__ import annotations

from learning_backend.chess_core.board import BoardState
from learning_backend.engines import AVAILABLE_ENGINES, make_engine
from learning_backend.experiments.run_experiment import run_arena


def test_all_registered_engines_select_legal_moves() -> None:
    state = BoardState.new_chess960(seed=7)
    legal_moves = list(state.board.legal_moves)
    for engine_name in AVAILABLE_ENGINES:
        engine = make_engine(engine_name)
        move = engine.select_move(state.board.copy(stack=False), legal_moves)
        assert move in legal_moves


def test_arena_smoke() -> None:
    result = run_arena(make_engine("random"), make_engine("material"), games=2, seed=3)
    assert result["games"] == 2
    assert len(result["results"]) == 2
    assert sum(result["scores"].values()) == 2


def test_arena_includes_replay_snapshots() -> None:
    result = run_arena(make_engine("random"), make_engine("random"), games=1, seed=4)
    game = result["results"][0]
    assert len(game["snapshots"]) == game["plies"] + 1
    assert game["snapshots"][0]["ply"] == 0
    assert game["snapshots"][-1]["fen"] == game["final_fen"]
    assert "pieces" in game["snapshots"][0]
