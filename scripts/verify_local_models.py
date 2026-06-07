from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from huggingface_hub import snapshot_download

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.generation.base import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify configured Hugging Face models exist locally.")
    parser.add_argument("--summarizer-config", default="configs/zero_shot_baseline.json")
    parser.add_argument("--judge-config", default="configs/llm_judge_local.json")
    return parser.parse_args()


def main() -> None:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    args = parse_args()
    summarizer_config = load_config(args.summarizer_config)
    judge_config = load_config(args.judge_config)
    models = {
        "summarizer": summarizer_config["model"]["model_name"],
        "judge": judge_config["model"]["model_name"],
    }

    results: dict[str, Any] = {}
    for role, model_name in models.items():
        path = resolve_local_snapshot(model_name)
        results[role] = {
            "model_name": model_name,
            "snapshot_path": str(path),
            "has_config": (path / "config.json").exists(),
            "has_tokenizer": any(
                (path / filename).exists()
                for filename in ("tokenizer.json", "tokenizer.model", "vocab.json")
            ),
        }

    print(json.dumps(results, indent=2))


def resolve_local_snapshot(model_name: str) -> Path:
    path = Path(model_name)
    if path.exists():
        return path.resolve()
    return Path(snapshot_download(repo_id=model_name, local_files_only=True)).resolve()


if __name__ == "__main__":
    main()
