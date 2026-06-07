from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.generation.base import load_config


DEFAULT_CONFIGS = [
    "configs/zero_shot_baseline.json",
    "configs/zero_shot/structured_sections.json",
    "configs/zero_shot/problem_oriented.json",
    "configs/zero_shot/timeline_synthesis.json",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run and evaluate zero-shot prompt variants.")
    parser.add_argument("--configs", nargs="+", default=DEFAULT_CONFIGS)
    parser.add_argument("--split", default="val")
    parser.add_argument(
        "--limit",
        type=parse_optional_int,
        default=3,
        help="Samples per config. Use 'all' or 'none' for the full split.",
    )
    parser.add_argument("--suite-name", default="zero_shot_prompt_variants")
    parser.add_argument("--experiment-root", default=".experiments")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--save-prompts", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--max-input-tokens", type=int, default=None)
    parser.add_argument("--judge-config", default=None)
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--continue-on-judge-error", action="store_true")
    parser.add_argument("--skip-evaluation", action="store_true")
    parser.add_argument("--skip-report", action="store_true")
    parser.add_argument(
        "--allow-downloads",
        action="store_true",
        help="Allow Hugging Face downloads. By default, model loading is local-cache only.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = Path(args.experiment_root) / args.suite_name / run_id
    predictions_dir = experiment_dir / "predictions"
    evaluations_dir = experiment_dir / "evaluations"
    logs_dir = experiment_dir / "logs"
    for path in [predictions_dir, evaluations_dir, logs_dir]:
        path.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {
        "suite_name": args.suite_name,
        "run_id": run_id,
        "experiment_dir": str(experiment_dir),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "split": args.split,
        "limit": args.limit,
        "max_new_tokens": args.max_new_tokens,
        "max_input_tokens": args.max_input_tokens,
        "local_files_only": not args.allow_downloads,
        "experiments": [],
    }
    manifest_path = experiment_dir / "manifest.json"

    for config in [Path(value) for value in args.configs]:
        config_data = load_config(config)
        name = safe_name(config_data.get("experiment_name") or config.stem)
        prediction_path = predictions_dir / f"{name}.jsonl"
        metrics_path = evaluations_dir / f"{name}_metrics.jsonl"
        aggregate_path = evaluations_dir / f"{name}_aggregate.json"

        item = {
            "name": name,
            "config": str(config),
            "prediction_path": str(prediction_path),
            "metrics_path": str(metrics_path),
            "aggregate_path": str(aggregate_path),
            "status": "running",
        }
        manifest["experiments"].append(item)
        write_manifest(manifest_path, manifest)

        inference_cmd = build_inference_command(args, config, prediction_path)
        run_command(inference_cmd, logs_dir / f"{name}_inference.log")

        if not args.skip_evaluation:
            evaluation_cmd = build_evaluation_command(args, prediction_path, metrics_path, aggregate_path, name)
            run_command(evaluation_cmd, logs_dir / f"{name}_evaluation.log")

        item["status"] = "completed"
        write_manifest(manifest_path, manifest)

    if not args.skip_report:
        report_cmd = [
            sys.executable,
            str(PROJECT_ROOT / "scripts/generate_performance_report.py"),
            "--experiment-dir",
            str(experiment_dir),
        ]
        run_command(report_cmd, logs_dir / "performance_report.log")

    print(json.dumps({"experiment_dir": str(experiment_dir), "manifest": str(manifest_path)}, indent=2))


def build_inference_command(args: argparse.Namespace, config: Path, prediction_path: Path) -> list[str]:
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts/run_zero_shot_validation.py"),
        "--config",
        str(config),
        "--split",
        args.split,
        "--output",
        str(prediction_path),
    ]
    if args.limit is not None:
        command.extend(["--limit", str(args.limit)])
    if args.resume:
        command.append("--resume")
    if args.save_prompts:
        command.append("--save-prompts")
    if not args.allow_downloads:
        command.append("--local-files-only")
    if args.max_new_tokens is not None:
        command.extend(["--max-new-tokens", str(args.max_new_tokens)])
    if args.max_input_tokens is not None:
        command.extend(["--max-input-tokens", str(args.max_input_tokens)])
    if args.continue_on_error:
        command.append("--continue-on-error")
    return command


def build_evaluation_command(
    args: argparse.Namespace,
    prediction_path: Path,
    metrics_path: Path,
    aggregate_path: Path,
    name: str,
) -> list[str]:
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts/evaluate_summaries.py"),
        "--predictions",
        str(prediction_path),
        "--metrics-output",
        str(metrics_path),
        "--aggregate-output",
        str(aggregate_path),
        "--run-name",
        name,
    ]
    if args.judge_config:
        command.extend(["--judge-config", args.judge_config])
    else:
        command.append("--skip-llm-judge")
    if args.continue_on_judge_error:
        command.append("--continue-on-judge-error")
    return command


def run_command(command: list[str], log_path: Path) -> None:
    print(" ".join(command), flush=True)
    env = os.environ.copy()
    env.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    env.setdefault("TRANSFORMERS_VERBOSITY", "error")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    result = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    log_path.write_text(result.stdout, encoding="utf-8")
    if result.stdout:
        print(summarize_command_output(result.stdout), flush=True)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {result.returncode}: {' '.join(command)}")


def parse_optional_int(value: str) -> int | None:
    if value.lower() in {"all", "full", "none", "null"}:
        return None
    return int(value)


def safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip().lower())
    return cleaned.strip("._-") or "run"


def write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def summarize_command_output(text: str, max_lines: int = 40) -> str:
    lines = [line for line in text.splitlines() if line.strip()]
    if len(lines) <= max_lines:
        return text

    interesting = [
        line
        for line in lines
        if line.lstrip().startswith(("{", "}"))
        or '"written"' in line
        or '"aggregate"' in line
        or '"report"' in line
        or '"experiment_dir"' in line
    ]
    selected = interesting[-max_lines:] if interesting else lines[-max_lines:]
    return "\n".join(selected) + "\n"


if __name__ == "__main__":
    main()
