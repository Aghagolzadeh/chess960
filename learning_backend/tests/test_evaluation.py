from __future__ import annotations

from learning_backend.engines import make_engine
from learning_backend.experiments.evaluate import evaluate_engines


def test_evaluation_reports_statistical_fields() -> None:
    result = evaluate_engines(make_engine("material"), make_engine("random"), games=4, seed=21)
    assert result["games"] == 4
    assert "score_confidence_interval" in result
    assert "side_split" in result
    assert "performance_by_start_id" in result
    assert result["wins"] + result["losses"] + result["draws"] == 4
