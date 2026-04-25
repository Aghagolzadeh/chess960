import { playMove, startGame } from "../api/client.js";
import { renderBoard } from "../components/Board.js";
import { renderMoveHistory } from "../components/MoveHistory.js";

function formatClock(ms) {
  const totalSeconds = Math.max(0, Math.floor(ms / 1000));
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;
  return `${String(minutes).padStart(2, "0")}:${String(seconds).padStart(2, "0")}`;
}

export function createGamePage({ engines }) {
  let state = null;
  let selectedSquare = null;
  const settings = {
    mode: "human_vs_engine",
    engine: "chess960_heuristic",
    depth: 2,
    seed: 42,
  };
  const root = document.createElement("section");
  root.className = "game-layout";

  async function newGame() {
    selectedSquare = null;
    state = await startGame({
      mode: settings.mode,
      black_engine: settings.engine,
      seed: settings.seed,
    });
    render();
  }

  async function selectSquare(square) {
    if (!state || state.status.state !== "ongoing") {
      return;
    }
    const move = state.legal_moves.find((candidate) => candidate.from === selectedSquare && candidate.to === square);
    if (move) {
      state = await playMove({
        game_id: state.game_id,
        uci: move.uci,
        depth: settings.depth,
      });
      selectedSquare = null;
      render();
      return;
    }
    const hasLegalMove = state.legal_moves.some((candidate) => candidate.from === square);
    selectedSquare = hasLegalMove ? square : null;
    render();
  }

  function render() {
    const options = engines.map((engine) => `<option value="${engine}">${engine}</option>`).join("");
    const boardHost = document.createElement("div");
    boardHost.className = "board-host";
    if (state) {
      boardHost.appendChild(renderBoard({ state, selectedSquare, onSelect: selectSquare }));
    }

    root.innerHTML = `
      <section class="panel controls-panel">
        <div class="panel-heading">
          <h1>Chess960 Lab</h1>
          <button class="command" data-new-game>New</button>
        </div>
        <div class="control-grid">
          <label>Mode<select data-mode><option value="human_vs_engine">Human vs engine</option><option value="analysis">Human only</option></select></label>
          <label>Black engine<select data-engine>${options}</select></label>
          <label>Search depth<input data-depth type="number" min="1" max="4" value="2" /></label>
          <label>Seed<input data-seed type="number" value="42" /></label>
        </div>
      </section>
      <section class="play-surface"></section>
      <aside class="panel status-panel">
        <h2>Game</h2>
        <dl>
          <div><dt>Position</dt><dd>${state?.initial_position_id ?? "-"}</dd></div>
          <div><dt>Turn</dt><dd>${state?.turn ?? "-"}</dd></div>
          <div><dt>Status</dt><dd>${state?.status.state ?? "-"}</dd></div>
          <div><dt>Winner</dt><dd>${state?.status.winner ?? "-"}</dd></div>
          <div><dt>White clock</dt><dd>${state ? formatClock(state.clocks.white_ms) : "05:00"}</dd></div>
          <div><dt>Black clock</dt><dd>${state ? formatClock(state.clocks.black_ms) : "05:00"}</dd></div>
        </dl>
        <h2>Moves</h2>
        <div data-history></div>
      </aside>
    `;
    root.querySelector("[data-mode]").value = settings.mode;
    root.querySelector("[data-engine]").value = settings.engine;
    root.querySelector("[data-depth]").value = String(settings.depth);
    root.querySelector("[data-seed]").value = String(settings.seed);
    root.querySelector("[data-mode]").addEventListener("change", (event) => {
      settings.mode = event.target.value;
    });
    root.querySelector("[data-engine]").addEventListener("change", (event) => {
      settings.engine = event.target.value;
    });
    root.querySelector("[data-depth]").addEventListener("change", (event) => {
      settings.depth = Number(event.target.value);
    });
    root.querySelector("[data-seed]").addEventListener("change", (event) => {
      settings.seed = Number(event.target.value);
    });
    root.querySelector("[data-new-game]").addEventListener("click", newGame);
    root.querySelector(".play-surface").appendChild(boardHost);
    root.querySelector("[data-history]").appendChild(renderMoveHistory(state?.history || []));
  }

  render();
  setTimeout(newGame, 0);
  return root;
}
