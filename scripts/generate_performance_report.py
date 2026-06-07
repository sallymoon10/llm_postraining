from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


REPORT_COLUMNS = [
    ("run_name", "Run"),
    ("count", "N"),
    ("bleu_1", "BLEU-1"),
    ("bleu_2", "BLEU-2"),
    ("bleu_4", "BLEU-4"),
    ("rouge_1_f1", "ROUGE-1 F1"),
    ("rouge_2_f1", "ROUGE-2 F1"),
    ("rouge_l_f1", "ROUGE-L F1"),
    ("prediction_tokens", "Pred Tokens"),
    ("length_ratio", "Len Ratio"),
    ("judge_overall_count", "Judge N"),
    ("judge_overall_mean", "Judge Overall"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a performance report for an experiment run.")
    parser.add_argument("--experiment-dir", required=True, help="Experiment directory with manifest.json.")
    parser.add_argument("--output", default=None, help="Markdown report path.")
    parser.add_argument("--csv-output", default=None, help="CSV report path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    experiment_dir = Path(args.experiment_dir)
    manifest = load_manifest(experiment_dir)
    records = load_records(experiment_dir)
    if not records:
        raise RuntimeError(f"No aggregate results found under {experiment_dir}")

    output = Path(args.output) if args.output else experiment_dir / "performance_report.md"
    csv_output = Path(args.csv_output) if args.csv_output else experiment_dir / "performance_report.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    csv_output.parent.mkdir(parents=True, exist_ok=True)

    output.write_text(render_markdown(experiment_dir, records, manifest), encoding="utf-8")
    write_csv(csv_output, records)
    print(json.dumps({"report": str(output), "csv": str(csv_output), "runs": len(records)}, indent=2))


def load_manifest(experiment_dir: Path) -> dict[str, Any]:
    manifest_path = experiment_dir / "manifest.json"
    if not manifest_path.exists():
        return {}
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def load_records(experiment_dir: Path) -> list[dict[str, Any]]:
    manifest_path = experiment_dir / "manifest.json"
    records: list[dict[str, Any]] = []
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for item in manifest.get("experiments", []):
            aggregate_path = Path(item.get("aggregate_path", ""))
            if aggregate_path.exists():
                record = json.loads(aggregate_path.read_text(encoding="utf-8"))
                record.setdefault("run_name", item.get("name"))
                record["config"] = item.get("config")
                record.update(load_judge_means(Path(record.get("metrics_path", ""))))
                records.append(record)
        return sorted(records, key=lambda row: str(row.get("run_name", "")))

    for aggregate_path in sorted(experiment_dir.glob("**/*_aggregate.json")):
        record = json.loads(aggregate_path.read_text(encoding="utf-8"))
        record.setdefault("run_name", aggregate_path.stem.replace("_aggregate", ""))
        record.update(load_judge_means(Path(record.get("metrics_path", ""))))
        records.append(record)
    return records


def load_judge_means(metrics_path: Path) -> dict[str, float]:
    if not metrics_path.exists():
        return {}

    judge_values: dict[str, list[float]] = {}
    with metrics_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            judge = row.get("llm_judge") or {}
            for key in ["faithfulness", "coverage", "concision", "clinical_relevance", "overall"]:
                value = judge.get(key)
                if isinstance(value, int | float):
                    judge_values.setdefault(key, []).append(float(value))

    output: dict[str, float] = {}
    for key, values in judge_values.items():
        if not values:
            continue
        output[f"judge_{key}_mean"] = mean(values)
        output[f"judge_{key}_count"] = float(len(values))
    return output


def render_markdown(
    experiment_dir: Path,
    records: list[dict[str, Any]],
    manifest: dict[str, Any],
) -> str:
    split = manifest.get("split")
    limit = manifest.get("limit")
    max_new_tokens = manifest.get("max_new_tokens")
    lines = [
        "# Zero-Shot Prompt Performance Report",
        "",
        f"Experiment directory: `{experiment_dir}`",
        f"Dataset split: `{split or 'unknown'}`",
        f"Sample limit: `{format_cell(limit) if limit is not None else 'all'}`",
        f"Max new tokens: `{format_cell(max_new_tokens) if max_new_tokens is not None else 'config default'}`",
        f"Generated at: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "| " + " | ".join(label for _, label in REPORT_COLUMNS) + " |",
        "| " + " | ".join("---" for _ in REPORT_COLUMNS) + " |",
    ]
    for record in records:
        lines.append("| " + " | ".join(format_cell(record.get(key)) for key, _ in REPORT_COLUMNS) + " |")

    best = best_record(records, "rouge_l_f1")
    if best:
        lines.extend(
            [
                "",
                f"Best ROUGE-L F1: `{best.get('run_name')}` "
                f"({format_cell(best.get('rouge_l_f1'))}).",
            ]
        )

    lines.extend(
        [
            "",
            "Notes:",
            "- BLEU is computed with `sacrebleu`; ROUGE is computed with `rouge-score`.",
            "- Local LLM-judge columns appear only when evaluation was run with a judge config.",
            "- Judge averages use rows with parseable numeric local-judge scores.",
            "- This report summarizes the predictions already saved in this experiment directory.",
            "",
        ]
    )
    return "\n".join(lines)


def write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    keys = [key for key, _ in REPORT_COLUMNS]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for record in records:
            writer.writerow({key: record.get(key) for key in keys})


def best_record(records: list[dict[str, Any]], metric: str) -> dict[str, Any] | None:
    numeric = [record for record in records if isinstance(record.get(metric), int | float)]
    if not numeric:
        return None
    return max(numeric, key=lambda record: float(record[metric]))


def format_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


if __name__ == "__main__":
    main()
