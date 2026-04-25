from __future__ import annotations

import random

PIECE_SYMBOLS = ("B", "Q", "N", "R", "K")


def random_chess960_position_id(seed: int | None = None) -> int:
    rng = random.Random(seed)
    return rng.randrange(960)


def generate_chess960_back_rank(position_id: int) -> str:
    """Return the white back rank for a Scharnagl Chess960 position id."""
    if not 0 <= position_id < 960:
        raise ValueError("Chess960 position id must be between 0 and 959")

    n = position_id
    squares: list[str | None] = [None] * 8

    dark_bishop_file = (n % 4) * 2 + 1
    n //= 4
    squares[dark_bishop_file] = "B"

    light_bishop_file = (n % 4) * 2
    n //= 4
    squares[light_bishop_file] = "B"

    empty_files = [i for i, piece in enumerate(squares) if piece is None]
    queen_index = n % 6
    n //= 6
    squares[empty_files[queen_index]] = "Q"

    empty_files = [i for i, piece in enumerate(squares) if piece is None]
    knight_pairs = [
        (0, 1),
        (0, 2),
        (0, 3),
        (0, 4),
        (1, 2),
        (1, 3),
        (1, 4),
        (2, 3),
        (2, 4),
        (3, 4),
    ]
    knight_a, knight_b = knight_pairs[n]
    squares[empty_files[knight_a]] = "N"
    squares[empty_files[knight_b]] = "N"

    empty_files = [i for i, piece in enumerate(squares) if piece is None]
    for file_index, piece in zip(empty_files, ("R", "K", "R")):
        squares[file_index] = piece

    rank = "".join(piece or "" for piece in squares)
    validate_chess960_back_rank(rank)
    return rank


def validate_chess960_back_rank(rank: str) -> bool:
    if len(rank) != 8:
        raise ValueError("Back rank must contain 8 pieces")
    if sorted(rank) != sorted("RNBQKBNR"):
        raise ValueError("Back rank must contain RNBQKBNR material")

    bishops = [i for i, piece in enumerate(rank) if piece == "B"]
    if len(bishops) != 2 or bishops[0] % 2 == bishops[1] % 2:
        raise ValueError("Chess960 bishops must start on opposite colors")

    king_file = rank.index("K")
    rook_files = [i for i, piece in enumerate(rank) if piece == "R"]
    if not (rook_files[0] < king_file < rook_files[1]):
        raise ValueError("Chess960 king must start between the two rooks")

    return True
