from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.llm_judge import JudgeConfig, LocalTransformersJudge
from src.evaluation.metrics import aggregate_metrics, compute_summary_metrics
from src.generation.base import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate summarization prediction JSONL files.")
    parser.add_argument("--predictions", required=True, help="JSONL file from validation inference.")
    parser.add_argument("--output-dir", default="outputs/evaluation")
    parser.add_argument("--metrics-output", default=None, help="Optional exact JSONL metrics path.")
    parser.add_argument("--aggregate-output", default=None, help="Optional exact aggregate JSON path.")
    parser.add_argument("--run-name", default=None, help="Experiment/run label to store in aggregate JSON.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--judge-config", default=None, help="Optional LLM judge config JSON.")
    parser.add_argument("--skip-llm-judge", action="store_true")
    parser.add_argument(
        "--continue-on-judge-error",
        action="store_true",
        help="Compute automatic metrics even if local judge generation/parsing fails.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictions_path = Path(args.predictions)
    rows = load_jsonl(predictions_path, limit=args.limit)

    judge = None
    if args.judge_config and not args.skip_llm_judge:
        judge_config = JudgeConfig.from_dict(load_config(args.judge_config))
        judge = LocalTransformersJudge(judge_config)

    evaluated_rows: list[dict[str, Any]] = []
    for row in rows:
        prediction = row.get("prediction") or ""
        reference = row.get("target_summary") or row.get("reference") or ""
        metrics = compute_summary_metrics(prediction, reference)
        evaluated = {
            "sample_id": row.get("sample_id"),
            "split": row.get("split"),
            "metrics": metrics,
            "prediction": prediction,
            "target_summary": reference,
        }

        if judge is not None and prediction.strip():
            try:
                evaluated["llm_judge"] = judge.judge(row)
            except RuntimeError as exc:
                if not args.continue_on_judge_error:
                    raise
                evaluated["llm_judge_error"] = str(exc)

        evaluated_rows.append(evaluated)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = args.run_name or predictions_path.stem
    metrics_path = Path(args.metrics_output) if args.metrics_output else output_dir / f"{stem}_metrics_{timestamp}.jsonl"
    aggregate_path = (
        Path(args.aggregate_output)
        if args.aggregate_output
        else output_dir / f"{stem}_aggregate_{timestamp}.json"
    )
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    aggregate_path.parent.mkdir(parents=True, exist_ok=True)

    with metrics_path.open("w", encoding="utf-8") as handle:
        for row in evaluated_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    aggregate = aggregate_metrics(evaluated_rows)
    aggregate.update(
        {
            "run_name": args.run_name,
            "predictions": str(predictions_path),
            "metrics_path": str(metrics_path),
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }
    )
    aggregate_path.write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    print(json.dumps({"aggregate": aggregate, "aggregate_path": str(aggregate_path)}, indent=2))


def load_jsonl(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break
    return rows


if __name__ == "__main__":
    main()
