from __future__ import annotations

import argparse
import copy
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.generation.base import load_config
from src.generation.zero_shot import ZeroShotDischargeSummarizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run zero-shot discharge-summary inference on a MIMIC split."
    )
    parser.add_argument("--config", default="configs/zero_shot_baseline.json")
    parser.add_argument("--split", default="val", help="Dataset split to run, e.g. val/test/train.")
    parser.add_argument("--limit", type=int, default=None, help="Optional sample cap.")
    parser.add_argument(
        "--sample-id",
        type=int,
        action="append",
        dest="sample_ids",
        help="Specific SAMPLE_ID to run. Can be supplied multiple times.",
    )
    parser.add_argument("--output", default=None, help="JSONL output path. Defaults to timestamped file.")
    parser.add_argument("--resume", action="store_true", help="Skip sample_ids already present in output.")
    parser.add_argument("--dry-run", action="store_true", help="Save prompts without model inference.")
    parser.add_argument("--local-files-only", action="store_true", help="Use cached Hugging Face files only.")
    parser.add_argument("--save-prompts", action="store_true", help="Persist rendered prompts in JSONL rows.")
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--max-input-tokens", type=int, default=None)
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Write error rows instead of stopping on generation failures.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    config = copy.deepcopy(load_config(config_path))
    config.setdefault("data", {})["split"] = args.split
    config["data"]["limit"] = args.limit
    if args.sample_ids:
        config["data"]["sample_ids"] = args.sample_ids
    if args.local_files_only:
        config.setdefault("model", {})["local_files_only"] = True
    if args.save_prompts:
        config.setdefault("runtime", {})["save_prompts"] = True

    generation_overrides: dict[str, Any] = {}
    if args.max_new_tokens is not None:
        generation_overrides["max_new_tokens"] = args.max_new_tokens
    if args.max_input_tokens is not None:
        generation_overrides["max_input_tokens"] = args.max_input_tokens

    runner = ZeroShotDischargeSummarizer(config, config_path=config_path)
    output_path = resolve_output_path(args.output, runner.output_dir(), args.split, args.dry_run)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    seen_sample_ids = read_existing_sample_ids(output_path) if args.resume else set()
    examples = runner.build_loader().load_examples(limit=args.limit, sample_ids=args.sample_ids)

    mode = "a" if args.resume else "w"
    written = 0
    skipped = 0
    started_at = datetime.now().isoformat(timespec="seconds")

    with output_path.open(mode, encoding="utf-8") as handle:
        for index, example in enumerate(examples, start=1):
            if example.sample_id in seen_sample_ids:
                skipped += 1
                continue

            prompt = runner.build_prompt(example)
            row: dict[str, Any] = {
                "sample_id": example.sample_id,
                "split": example.split,
                "metadata": example.metadata,
                "target_summary": example.target_summary,
                "prediction": None,
                "error": None,
                "run": {
                    "config": str(config_path),
                    "started_at": started_at,
                    "dry_run": args.dry_run,
                    "index": index,
                },
            }
            if config.get("runtime", {}).get("save_prompts", False):
                row["prompt"] = prompt

            try:
                if not args.dry_run:
                    row["prediction"] = runner.generate_text(prompt, **generation_overrides)
            except Exception as exc:
                row["error"] = {"type": type(exc).__name__, "message": str(exc)}
                if not args.continue_on_error:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                    raise

            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            written += 1
            print(
                json.dumps(
                    {
                        "written": written,
                        "skipped": skipped,
                        "sample_id": example.sample_id,
                        "output": str(output_path),
                        "error": row["error"],
                    }
                ),
                flush=True,
            )

    sidecar = output_path.with_suffix(".config.json")
    sidecar.write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "done": True,
                "written": written,
                "skipped": skipped,
                "examples_loaded": len(examples),
                "output": str(output_path),
                "config_snapshot": str(sidecar),
            },
            indent=2,
        )
    )


def resolve_output_path(output: str | None, output_dir: Path, split: str, dry_run: bool) -> Path:
    if output:
        return Path(output)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = "dry_run" if dry_run else "predictions"
    return output_dir / f"zero_shot_{split}_{suffix}_{timestamp}.jsonl"


def read_existing_sample_ids(path: Path) -> set[int]:
    if not path.exists():
        return set()

    sample_ids: set[int] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            sample_id = row.get("sample_id")
            if sample_id is not None:
                sample_ids.add(int(sample_id))
    return sample_ids


if __name__ == "__main__":
    main()
