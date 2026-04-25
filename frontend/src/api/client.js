const DEFAULT_API_BASE = "http://127.0.0.1:8000";

export function getApiBase() {
  return window.localStorage.getItem("chess960_api_base") || DEFAULT_API_BASE;
}

export function setApiBase(value) {
  window.localStorage.setItem("chess960_api_base", value || DEFAULT_API_BASE);
}

async function request(path, options = {}) {
  const response = await fetch(`${getApiBase()}${path}`, {
    headers: {
      "Content-Type": "application/json",
      ...(options.headers || {}),
    },
    ...options,
  });
  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.error || `Request failed: ${response.status}`);
  }
  return payload;
}

export function getEngines() {
  return request("/api/engines");
}

export function startGame(payload) {
  return request("/api/game/new", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function playMove(payload) {
  return request("/api/game/move", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function runArena(payload) {
  return request("/api/arena/run", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}
