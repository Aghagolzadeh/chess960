# Frontend

The frontend is a Vite-powered browser client for the local Chess960 backend. It renders the board, move history, game status, clocks, engine selectors, and arena summaries.

## Setup

```bash
cd frontend
npm install
npm run dev
```

Open `http://127.0.0.1:5173`.

The backend should be running separately:

```bash
python3 -m learning_backend.api.app
```

By default the frontend calls `http://127.0.0.1:8000`. The API field in the top bar can point at a different local backend.

## Starting a Chess960 Game

Use the Play tab. Choose a mode, pick the black engine, set a deterministic seed, and press New. The backend returns the Chess960 position id, board pieces, legal moves, clocks, status, and history.

## Human Play

The human plays from the browser. The board highlights legal destinations returned by the backend. When the human submits a move in human-vs-engine mode, the backend validates it, applies it, selects an engine reply, and returns the updated state.

## Arena

Use the Arena tab to choose white and black engines, number of games, and seed. The frontend calls `/api/arena/run` and displays white wins, black wins, and draws.

## Engine Selection

The engine list comes from `/api/engines`. Current choices are:

- `random`
- `material`
- `heuristic`
- `chess960_heuristic`
- `learned`

## Key Files

- `src/api/client.js`: API calls.
- `src/components/Board.js`: board rendering.
- `src/components/MoveHistory.js`: move list rendering.
- `src/pages/GamePage.js`: human play page.
- `src/arena/ArenaView.js`: arena controls and result summary.
- `src/styles/main.css`: app styling.
