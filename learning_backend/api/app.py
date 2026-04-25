from __future__ import annotations

from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from typing import Any, Callable
from urllib.parse import urlparse

from learning_backend.api.routes_arena import run_arena_route
from learning_backend.api.routes_experiments import list_experiments
from learning_backend.api.routes_game import GameStore
from learning_backend.engines import AVAILABLE_ENGINES


def create_handler(game_store: GameStore) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def do_OPTIONS(self) -> None:
            self._send_json({})

        def do_GET(self) -> None:
            path = urlparse(self.path).path
            if path == "/health":
                self._send_json({"ok": True})
            elif path == "/api/engines":
                self._send_json({"engines": AVAILABLE_ENGINES})
            elif path == "/api/experiments":
                self._send_json(list_experiments())
            else:
                self._send_json({"error": "Not found"}, HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:
            path = urlparse(self.path).path
            routes: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {
                "/api/game/new": game_store.new_game,
                "/api/game/move": game_store.move,
                "/api/arena/run": run_arena_route,
            }
            if path not in routes:
                self._send_json({"error": "Not found"}, HTTPStatus.NOT_FOUND)
                return
            try:
                payload = self._read_json()
                self._send_json(routes[path](payload))
            except (KeyError, ValueError) as exc:
                self._send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)

        def log_message(self, format: str, *args: object) -> None:
            return

        def _read_json(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length", "0"))
            if length == 0:
                return {}
            return json.loads(self.rfile.read(length).decode("utf-8"))

        def _send_json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
            encoded = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

    return Handler


def run(host: str = "127.0.0.1", port: int = 8000) -> None:
    server = ThreadingHTTPServer((host, port), create_handler(GameStore()))
    print(f"Chess960 backend API running at http://{host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    run()
