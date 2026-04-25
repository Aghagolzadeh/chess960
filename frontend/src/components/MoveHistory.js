export function renderMoveHistory(history = []) {
  const list = document.createElement("ol");
  list.className = "move-history";
  history.forEach((move, index) => {
    const item = document.createElement("li");
    item.textContent = `${index + 1}. ${move.san}`;
    list.appendChild(item);
  });
  return list;
}
