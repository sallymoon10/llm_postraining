from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from src.generation.base import load_config
from src.generation.zero_shot import ZeroShotDischargeSummarizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MIMIC discharge-summary baselines.")
    parser.add_argument(
        "--config",
        default="configs/zero_shot_baseline.json",
        help="Path to a JSON/YAML experiment config.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Override number of samples.")
    parser.add_argument(
        "--sample-id",
        type=int,
        action="append",
        dest="sample_ids",
        help="Specific SAMPLE_ID to run. Can be supplied multiple times.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Format data and render prompts without loading the model.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Load the Hugging Face model from local cache only.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=None, help="Override generation length.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    config = copy.deepcopy(load_config(config_path))

    if args.local_files_only:
        config.setdefault("model", {})["local_files_only"] = True

    generation_overrides = {}
    if args.max_new_tokens is not None:
        generation_overrides["max_new_tokens"] = args.max_new_tokens

    runner = ZeroShotDischargeSummarizer(config, config_path=config_path)
    results = runner.run(
        limit=args.limit,
        sample_ids=args.sample_ids,
        dry_run=args.dry_run,
        generation_overrides=generation_overrides,
    )

    print(json.dumps(summarize_results(results, dry_run=args.dry_run), indent=2))


def summarize_results(results: list[dict], *, dry_run: bool) -> dict:
    if not results:
        return {"count": 0, "dry_run": dry_run}

    first = results[0]
    prediction_preview = first.get("prediction")
    if prediction_preview:
        prediction_preview = prediction_preview[:1200]

    return {
        "count": len(results),
        "dry_run": dry_run,
        "first_sample_id": first["sample_id"],
        "first_metadata": first["metadata"],
        "prompt_preview": first["prompt"][:1200],
        "prediction_preview": prediction_preview,
    }


if __name__ == "__main__":
    main()
