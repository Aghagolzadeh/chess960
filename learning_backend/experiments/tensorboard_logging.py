from __future__ import annotations

from pathlib import Path
import time
from typing import Any


class TensorBoardLogger:
    """Small TensorBoard scalar writer that does not import torch."""

    def __init__(self, log_dir: str | Path) -> None:
        try:
            from tensorboard.compat.proto import event_pb2, summary_pb2
            from tensorboard.summary.writer.event_file_writer import EventFileWriter
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "TensorBoard support requires the standalone 'tensorboard' package. "
                "Install it with: python -m pip install tensorboard"
            ) from exc

        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._event_pb2 = event_pb2
        self._summary_pb2 = summary_pb2
        self._writer = EventFileWriter(str(self.log_dir))

    def scalar(self, tag: str, value: float | int, step: int) -> None:
        summary = self._summary_pb2.Summary(
            value=[self._summary_pb2.Summary.Value(tag=tag, simple_value=float(value))]
        )
        event = self._event_pb2.Event(wall_time=time.time(), step=int(step), summary=summary)
        self._writer.add_event(event)

    def scalars(self, prefix: str, metrics: dict[str, Any], step: int) -> None:
        for key, value in metrics.items():
            if isinstance(value, bool):
                self.scalar(f"{prefix}/{key}", int(value), step)
            elif isinstance(value, (int, float)):
                self.scalar(f"{prefix}/{key}", value, step)

    def flush(self) -> None:
        self._writer.flush()

    def close(self) -> None:
        self._writer.flush()
        self._writer.close()

    def __enter__(self) -> "TensorBoardLogger":
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()
