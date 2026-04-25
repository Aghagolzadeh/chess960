from __future__ import annotations

from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from learning_backend.experiments.tensorboard_logging import TensorBoardLogger


def test_tensorboard_logger_writes_scalar(tmp_path: Path) -> None:
    with TensorBoardLogger(tmp_path) as logger:
        logger.scalar("train/loss", 1.25, 1)

    accumulator = EventAccumulator(str(tmp_path))
    accumulator.Reload()
    assert "train/loss" in accumulator.Tags()["scalars"]
    event = accumulator.Scalars("train/loss")[0]
    assert event.step == 1
    assert event.value == 1.25
