import { runArena } from "../api/client.js";
import { renderBoard } from "../components/Board.js";

export function renderArenaPage({ engines, onResult, initialResult = null }) {
  const section = document.createElement("section");
  section.className = "panel arena-panel";
  const options = engines.map((engine) => `<option value="${engine}">${engine}</option>`).join("");
  let latestResult = initialResult;
  let selectedGameIndex = 0;
  let selectedPly = 0;

  function selectedGame() {
    return latestResult?.results?.[selectedGameIndex] || null;
  }

  function renderResults() {
    const resultBox = section.querySelector("[data-results]");
    if (!latestResult) {
      resultBox.innerHTML = "";
      return;
    }

    const game = selectedGame();
    const snapshots = game?.snapshots || [];
    selectedPly = Math.min(selectedPly, Math.max(0, snapshots.length - 1));
    const snapshot = snapshots[selectedPly];
    const boardState = snapshot ? { pieces: snapshot.pieces, legal_moves: [] } : null;
    const gameOptions = latestResult.results
      .map((gameResult, index) => {
        const label = `Game ${index + 1}: ${gameResult.white_engine} vs ${gameResult.black_engine} (${gameResult.status.result})`;
        return `<option value="${index}">${label}</option>`;
      })
      .join("");

    resultBox.innerHTML = `
      <div class="score-grid">
        <span>White wins <strong>${latestResult.scores.white_wins}</strong></span>
        <span>Black wins <strong>${latestResult.scores.black_wins}</strong></span>
        <span>Draws <strong>${latestResult.scores.draws}</strong></span>
      </div>
      <div class="arena-replay">
        <div class="replay-toolbar">
          <label>Game<select data-game-picker>${gameOptions}</select></label>
          <div class="replay-buttons">
            <button class="command secondary" data-first-ply type="button">First</button>
            <button class="command secondary" data-prev-ply type="button">Prev</button>
            <button class="command secondary" data-next-ply type="button">Next</button>
            <button class="command secondary" data-last-ply type="button">Last</button>
          </div>
        </div>
        <div class="replay-grid">
          <div class="arena-board" data-board></div>
          <aside class="replay-details">
            <dl>
              <div><dt>Position</dt><dd>${game?.initial_position_id ?? "-"}</dd></div>
              <div><dt>Ply</dt><dd>${selectedPly} / ${Math.max(0, snapshots.length - 1)}</dd></div>
              <div><dt>Move</dt><dd>${snapshot?.move || "Start"}</dd></div>
              <div><dt>Turn</dt><dd>${snapshot?.turn || "-"}</dd></div>
              <div><dt>Status</dt><dd>${game?.status?.state || "-"}</dd></div>
              <div><dt>Result</dt><dd>${game?.status?.result || "-"}</dd></div>
            </dl>
            <ol class="move-history arena-move-list" data-move-list></ol>
          </aside>
        </div>
      </div>
    `;

    resultBox.querySelector("[data-game-picker]").value = String(selectedGameIndex);
    resultBox.querySelector("[data-game-picker]").addEventListener("change", (event) => {
      selectedGameIndex = Number(event.target.value);
      selectedPly = 0;
      renderResults();
    });
    resultBox.querySelector("[data-first-ply]").addEventListener("click", () => {
      selectedPly = 0;
      renderResults();
    });
    resultBox.querySelector("[data-prev-ply]").addEventListener("click", () => {
      selectedPly = Math.max(0, selectedPly - 1);
      renderResults();
    });
    resultBox.querySelector("[data-next-ply]").addEventListener("click", () => {
      selectedPly = Math.min(snapshots.length - 1, selectedPly + 1);
      renderResults();
    });
    resultBox.querySelector("[data-last-ply]").addEventListener("click", () => {
      selectedPly = Math.max(0, snapshots.length - 1);
      renderResults();
    });

    if (boardState) {
      resultBox.querySelector("[data-board]").appendChild(
        renderBoard({
          state: boardState,
          selectedSquare: null,
          onSelect: () => {},
        }),
      );
    }

    const moveList = resultBox.querySelector("[data-move-list]");
    for (const entry of game?.history || []) {
      const item = document.createElement("li");
      item.className = entry.uci === snapshot?.uci ? "current-move" : "";
      item.textContent = `${entry.san} (${entry.uci})`;
      item.addEventListener("click", () => {
        selectedPly = snapshots.findIndex((candidate) => candidate.uci === entry.uci);
        renderResults();
      });
      moveList.appendChild(item);
    }
  }

  section.innerHTML = `
    <div class="panel-heading">
      <h2>Arena</h2>
      <button class="command" data-run-arena>Run</button>
    </div>
    <div class="control-grid">
      <label>White engine<select data-white>${options}</select></label>
      <label>Black engine<select data-black>${options}</select></label>
      <label>Games<input data-games type="number" min="1" max="20" value="4" /></label>
      <label>Seed<input data-seed type="number" value="42" /></label>
    </div>
    <div class="arena-results" data-results></div>
  `;

  section.querySelector("[data-white]").value = "chess960_heuristic";
  section.querySelector("[data-black]").value = "random";
  section.querySelector("[data-run-arena]").addEventListener("click", async () => {
    const resultBox = section.querySelector("[data-results]");
    resultBox.textContent = "Running arena...";
    const result = await runArena({
      white: section.querySelector("[data-white]").value,
      black: section.querySelector("[data-black]").value,
      games: Number(section.querySelector("[data-games]").value),
      seed: Number(section.querySelector("[data-seed]").value),
    });
    latestResult = result;
    selectedGameIndex = 0;
    selectedPly = 0;
    renderResults();
    onResult(result);
  });

  renderResults();
  return section;
}
