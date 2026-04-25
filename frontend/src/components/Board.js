const files = ["a", "b", "c", "d", "e", "f", "g", "h"];

export function renderBoard({ state, selectedSquare, onSelect }) {
  const board = document.createElement("div");
  board.className = "board";

  const pieces = new Map((state?.pieces || []).map((piece) => [piece.square, piece]));
  const legalMoves = state?.legal_moves || [];
  const selectedMoves = legalMoves.filter((move) => move.from === selectedSquare);
  const destinations = new Map(selectedMoves.map((move) => [move.to, move]));

  for (let rank = 8; rank >= 1; rank -= 1) {
    for (let fileIndex = 0; fileIndex < 8; fileIndex += 1) {
      const squareName = `${files[fileIndex]}${rank}`;
      const piece = pieces.get(squareName);
      const square = document.createElement("button");
      square.className = `square ${(rank + fileIndex) % 2 === 0 ? "dark" : "light"}`;
      square.type = "button";
      square.dataset.square = squareName;
      if (selectedSquare === squareName) {
        square.classList.add("selected");
      }
      if (destinations.has(squareName)) {
        square.classList.add(destinations.get(squareName).is_capture ? "capture" : "legal");
      }
      square.innerHTML = `<span class="piece ${piece?.color || ""}">${piece?.symbol || ""}</span>`;
      square.addEventListener("click", () => onSelect(squareName));
      board.appendChild(square);
    }
  }

  return board;
}
