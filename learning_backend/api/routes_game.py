from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any
from uuid import uuid4

import chess

from learning_backend.chess_core.board import BoardState
from learning_backend.chess_core.rules import game_status
from learning_backend.engines import make_engine


@dataclass
class GameSession:
    id: str
    state: BoardState
    mode: str = "human_vs_engine"
    white_engine: str | None = None
    black_engine: str | None = "chess960_heuristic"
    white_ms: int = 300_000
    black_ms: int = 300_000
    last_tick: float = 0.0

    def tick(self) -> None:
        status = game_status(self.state.board)
        if status["state"] != "ongoing":
            self.last_tick = time.monotonic()
            return
        now = time.monotonic()
        if self.last_tick == 0.0:
            self.last_tick = now
            return
        elapsed = int((now - self.last_tick) * 1000)
        if self.state.board.turn == chess.WHITE:
            self.white_ms = max(0, self.white_ms - elapsed)
        else:
            self.black_ms = max(0, self.black_ms - elapsed)
        self.last_tick = now

    def as_dict(self) -> dict[str, Any]:
        self.tick()
        payload = self.state.to_dict()
        status = payload["status"]
        if self.white_ms <= 0 and status["state"] == "ongoing":
            status = {"state": "timeout", "winner": "black", "check": self.state.board.is_check(), "result": "0-1"}
        if self.black_ms <= 0 and status["state"] == "ongoing":
            status = {"state": "timeout", "winner": "white", "check": self.state.board.is_check(), "result": "1-0"}
        payload.update(
            {
                "game_id": self.id,
                "mode": self.mode,
                "white_engine": self.white_engine,
                "black_engine": self.black_engine,
                "status": status,
                "clocks": {"white_ms": self.white_ms, "black_ms": self.black_ms},
            }
        )
        return payload


class GameStore:
    def __init__(self) -> None:
        self.games: dict[str, GameSession] = {}

    def new_game(self, payload: dict[str, Any]) -> dict[str, Any]:
        session = GameSession(
            id=str(uuid4()),
            state=BoardState.new_chess960(seed=payload.get("seed"), position_id=payload.get("position_id")),
            mode=payload.get("mode", "human_vs_engine"),
            white_engine=payload.get("white_engine"),
            black_engine=payload.get("black_engine", "chess960_heuristic"),
            last_tick=time.monotonic(),
        )
        self.games[session.id] = session
        return session.as_dict()

    def move(self, payload: dict[str, Any]) -> dict[str, Any]:
        game_id = payload["game_id"]
        session = self.games[game_id]
        session.tick()
        move = session.state.board.parse_uci(payload["uci"])
        if move not in session.state.board.legal_moves:
            raise ValueError(f"Illegal move: {payload['uci']}")
        session.state.push(move)
        session.last_tick = time.monotonic()

        if session.mode == "human_vs_engine" and session.black_engine and not session.state.board.is_game_over():
            engine = make_engine(session.black_engine, depth=int(payload.get("depth", 2)))
            legal_moves = list(session.state.board.legal_moves)
            engine_move = engine.select_move(session.state.board.copy(stack=False), legal_moves)
            session.state.push(engine_move)
            session.last_tick = time.monotonic()
        return session.as_dict()
