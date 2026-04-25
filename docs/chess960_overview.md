# Chess960 Overview

Chess960, also called Fischer Random Chess, keeps normal chess pieces and rules while randomizing the back rank. There are 960 legal starting positions.

## Starting Position Rules

- White's back rank contains the same pieces as standard chess.
- The bishops must start on opposite-colored squares.
- The king must start between the two rooks.
- Black mirrors White's back rank.
- Pawns start on the usual second and seventh ranks.

## Why Chess960 Is Interesting

Chess960 reduces opening memorization. Players and agents must solve unfamiliar development problems from move one. That makes it useful for algorithmic learning because memorized standard openings are less helpful, and learned policies must rely more on transferable chess principles.

## Castling

After castling in Chess960, the king and rook finish on the same squares they would in standard chess:

- Kingside castling leaves the king on g-file and rook on f-file.
- Queenside castling leaves the king on c-file and rook on d-file.

The path rules depend on the starting layout. Engines and environments must be careful to use Chess960-aware castling logic. This repo delegates legality to `python-chess` through `learning_backend/chess_core`.

## Engine and Environment Care

Classical opening assumptions can be wrong in Chess960. Engines should avoid hardcoding standard starting-square development. Environments should expose the Chess960 position id, preserve castling legality, and seed position generation for repeatability.
