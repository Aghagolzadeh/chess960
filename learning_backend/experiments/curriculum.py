from __future__ import annotations

from dataclasses import asdict, dataclass, field
import argparse
import json
from pathlib import Path
from typing import Any

from learning_backend.engines import make_engine
from learning_backend.experiments.evaluate import evaluate_engines


@dataclass
class CurriculumStage:
    opponent: str
    promote_when_win_rate_above: float
    eval_games: int


@dataclass
class CurriculumConfig:
    enabled: bool = True
    stages: list[CurriculumStage] = field(
        default_factory=lambda: [
            CurriculumStage("random", 0.65, 100),
            CurriculumStage("capture_preferring", 0.62, 100),
            CurriculumStage("material", 0.60, 100),
            CurriculumStage("weak_minimax", 0.55, 100),
            CurriculumStage("heuristic", 0.50, 200),
        ]
    )


SUCCESS_STAGES = [
    "Stage 0: agent selects only legal moves",
    "Stage 1: agent beats random over 100 games",
    "Stage 2: agent beats capture-preferring/material engines",
    "Stage 3: agent is competitive with weak minimax depth-1",
    "Stage 4: agent is competitive with heuristic baseline",
    "Stage 5: agent can take games from Sunfish",
]


def evaluate_curriculum_stage(candidate: str, stage: CurriculumStage, *, seed: int = 42) -> dict[str, Any]:
    result = evaluate_engines(
        make_engine(candidate),
        make_engine(stage.opponent),
        games=stage.eval_games,
        seed=seed,
    )
    result["promoted"] = result["score_percentage"] >= stage.promote_when_win_rate_above
    result["threshold"] = stage.promote_when_win_rate_above
    return result


def run_curriculum_evaluation(candidate: str, *, seed: int = 42, output: str | Path | None = None) -> dict[str, Any]:
    config = CurriculumConfig()
    stage_results = []
    for stage in config.stages:
        result = evaluate_curriculum_stage(candidate, stage, seed=seed)
        stage_results.append(result)
        if not result["promoted"]:
            break
    payload = {"candidate": candidate, "config": asdict(config), "success_stages": SUCCESS_STAGES, "stage_results": stage_results}
    if output:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2))
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate staged curriculum promotion.")
    parser.add_argument("--candidate", default="learned")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    print(json.dumps(run_curriculum_evaluation(args.candidate, seed=args.seed, output=args.output or None), indent=2))


if __name__ == "__main__":
    main()
