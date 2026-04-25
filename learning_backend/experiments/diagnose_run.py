from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


METRICS = [
    "reward",
    "mean_reward",
    "win_rate",
    "entropy",
    "kl",
    "value_loss",
    "policy_loss",
    "illegal_move_count",
    "episode_length",
]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def diagnose_run(run_dir: str | Path) -> dict[str, Any]:
    run_path = Path(run_dir)
    rows = load_jsonl(run_path / "training_log.jsonl")
    plots_dir = run_path / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    findings = []
    if not rows:
        findings.append("No training_log.jsonl was found, so no learning curves can be diagnosed.")
    else:
        x = [row.get("update", row.get("episode", i + 1)) for i, row in enumerate(rows)]
        for metric in METRICS:
            values = [row.get(metric) for row in rows if row.get(metric) is not None]
            if not values:
                continue
            plt.figure()
            plt.plot(x[: len(values)], values)
            plt.title(metric.replace("_", " ").title())
            plt.xlabel("update")
            plt.ylabel(metric)
            plt.tight_layout()
            plt.savefig(plots_dir / f"{metric}.png")
            plt.close()
        if any(row.get("illegal_move_count", 0) for row in rows):
            findings.append("Illegal moves occurred during training; inspect action masking before trusting rewards.")
        if rows[-1].get("entropy", 1.0) < 0.05:
            findings.append("Policy entropy is very low; exploration may have collapsed.")
        if abs(rows[-1].get("reward", 0.0)) < 1e-9:
            findings.append("Final logged reward is near zero; sparse rewards may be too weak or evaluation may be missing.")
        if rows[-1].get("value_loss", 0.0) > 10.0:
            findings.append("Value loss is large; inspect reward scale and return normalization.")
    summary = {
        "run_dir": str(run_path),
        "rows": len(rows),
        "plots_dir": str(plots_dir),
        "likely_failure_diagnosis": findings or ["No obvious failure detected from available metrics."],
    }
    (run_path / "diagnosis_summary.md").write_text(render_summary(summary))
    return summary


def render_summary(summary: dict[str, Any]) -> str:
    lines = ["# Run Diagnosis", "", f"Rows: {summary['rows']}", f"Plots: `{summary['plots_dir']}`", "", "## Likely Failure Diagnosis"]
    lines.extend(f"- {item}" for item in summary["likely_failure_diagnosis"])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate plots and diagnosis for an RL run.")
    parser.add_argument("run_dir")
    args = parser.parse_args()
    print(json.dumps(diagnose_run(args.run_dir), indent=2))


if __name__ == "__main__":
    main()
