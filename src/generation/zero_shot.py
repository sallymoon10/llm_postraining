from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from src.data.mimic_discharge import DischargeSummaryExample, MimicDischargeDataLoader
from src.generation.base import InferenceRunner


class ZeroShotDischargeSummarizer(InferenceRunner):
    """Zero-shot discharge-summary generator using formatted MIMIC source notes."""

    def build_loader(self) -> MimicDischargeDataLoader:
        data_config = self.config.get("data", {})
        return MimicDischargeDataLoader(
            data_dir=data_config["data_dir"],
            split=data_config.get("split", "val"),
            include_categories=data_config.get("include_categories"),
            exclude_categories=data_config.get("exclude_categories", ["Discharge summary"]),
            exclude_iserror=bool(data_config.get("exclude_iserror", True)),
            max_notes_per_sample=data_config.get("max_notes_per_sample"),
            max_chars_per_note=data_config.get("max_chars_per_note", 2500),
            max_total_chars=data_config.get("max_total_chars", 30000),
            text_truncation=data_config.get("text_truncation", "head_tail"),
        )

    def load_prompt_template(self) -> str:
        prompt_path = self.resolve_path(self.config["prompt"]["template_path"])
        return prompt_path.read_text(encoding="utf-8")

    def build_prompt(self, example: DischargeSummaryExample) -> str:
        template = self.load_prompt_template()
        return template.format(
            patient_context=example.patient_context,
            source_notes=example.source_notes,
        ).strip()

    def run(
        self,
        *,
        limit: int | None = None,
        sample_ids: list[int] | None = None,
        dry_run: bool = False,
        generation_overrides: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        data_config = self.config.get("data", {})
        if limit is None:
            limit = data_config.get("limit")
        if sample_ids is None:
            sample_ids = data_config.get("sample_ids")

        examples = self.build_loader().load_examples(limit=limit, sample_ids=sample_ids)
        results = []
        for example in examples:
            prompt = self.build_prompt(example)
            result: dict[str, Any] = {
                "sample_id": example.sample_id,
                "split": example.split,
                "metadata": example.metadata,
                "prompt": prompt,
                "target_summary": example.target_summary,
            }
            if dry_run:
                result["prediction"] = None
            else:
                result["prediction"] = self.generate_text(prompt, **(generation_overrides or {}))
            results.append(result)

        if results and self.config.get("runtime", {}).get("save_generations", True):
            self.save_results(results, dry_run=dry_run)

        return results

    def save_results(self, results: list[dict[str, Any]], *, dry_run: bool = False) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        suffix = "dry_run" if dry_run else "generations"
        path = self.output_dir() / f"zero_shot_{suffix}_{timestamp}.jsonl"
        save_prompts = bool(self.config.get("runtime", {}).get("save_prompts", True))

        with path.open("w", encoding="utf-8") as handle:
            for result in results:
                row = dict(result)
                if not save_prompts:
                    row.pop("prompt", None)
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        return path
