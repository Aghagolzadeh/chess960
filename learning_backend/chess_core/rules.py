from __future__ import annotations

import chess


def game_status(board: chess.Board) -> dict[str, object]:
    if board.is_checkmate():
        winner = "black" if board.turn == chess.WHITE else "white"
        return {"state": "checkmate", "winner": winner, "check": True, "result": board.result()}
    if board.is_stalemate():
        return {"state": "stalemate", "winner": None, "check": False, "result": "1/2-1/2"}
    if board.is_insufficient_material():
        return {"state": "draw", "winner": None, "check": board.is_check(), "result": "1/2-1/2"}
    if board.can_claim_fifty_moves() or board.can_claim_threefold_repetition():
        return {"state": "claimable_draw", "winner": None, "check": board.is_check(), "result": "*"}
    return {"state": "ongoing", "winner": None, "check": board.is_check(), "result": "*"}


def is_terminal(board: chess.Board) -> bool:
    return game_status(board)["state"] in {"checkmate", "stalemate", "draw"}
