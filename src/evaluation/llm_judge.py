from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from src.generation.base import InferenceRunner


@dataclass(frozen=True)
class JudgeConfig:
    provider: str
    model: dict[str, Any]
    generation: dict[str, Any]
    include_prompt_context: bool = False
    max_prompt_chars: int = 8000
    max_reference_chars: int = 12000
    max_prediction_chars: int = 6000

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "JudgeConfig":
        return cls(**data)


class LocalTransformersJudge:
    """LLM-as-judge wrapper that runs a local Hugging Face model only."""

    def __init__(self, config: JudgeConfig) -> None:
        self.config = config
        model_config = dict(config.model)
        model_config["local_files_only"] = True
        self.runner = InferenceRunner(
            {
                "model": model_config,
                "generation": config.generation,
                "runtime": {"output_dir": "outputs/evaluation"},
            }
        )

    def judge(self, row: dict[str, Any]) -> dict[str, Any]:
        prompt = self._prompt(row)
        content = self.runner.generate_text(prompt)
        parsed = parse_json_object_or_fallback(content)
        parsed["raw_response"] = first_json_object_text(content) or content
        parsed["judge_model"] = self.config.model["model_name"]
        parsed["judge_provider"] = self.config.provider
        return parsed

    def _prompt(self, row: dict[str, Any]) -> str:
        reference = row.get("target_summary") or row.get("reference") or ""
        prediction = row.get("prediction") or ""
        prompt_context = row.get("prompt") or ""
        if not self.config.include_prompt_context:
            prompt_context = ""
        return build_judge_prompt(
            reference[: self.config.max_reference_chars],
            prediction[: self.config.max_prediction_chars],
            prompt_context[: self.config.max_prompt_chars],
        )


def build_judge_prompt(reference: str, prediction: str, prompt_context: str = "") -> str:
    context_block = ""
    if prompt_context:
        context_block = f"\nSOURCE PROMPT CONTEXT:\n{prompt_context}\n"

    return f"""You are a strict clinical summarization evaluator. Score only the candidate summary quality.
Do not give credit for information that appears in the reference but is missing from the candidate.
If the candidate is only a title, header, generic phrase, or very short fragment, coverage and overall should be 1.

Evaluate the candidate discharge summary against the reference discharge summary.
{context_block}
REFERENCE DISCHARGE SUMMARY:
{reference}

CANDIDATE SUMMARY:
{prediction}

Return a single JSON object with these keys:
- faithfulness: integer 1-5, where 5 means all claims are supported by the reference/context.
- coverage: integer 1-5, where 5 means it captures the clinically important content.
- concision: integer 1-5, where 5 means concise without omitting critical information.
- clinical_relevance: integer 1-5, where 5 means clinically useful and well prioritized.
- overall: integer 1-5.
- rationale: one short paragraph explaining the scores.

Return JSON only. Do not include markdown.
"""


def parse_json_object_or_fallback(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        if stripped.lower().startswith("json"):
            stripped = stripped[4:].strip()

    parsed = try_parse_first_json_object(stripped)
    if parsed is not None:
        return parsed

    start = stripped.find("{")
    if start >= 0:
        parsed = try_parse_first_json_object(stripped[start:])
        if parsed is not None:
            return parsed

    return {
        "faithfulness": None,
        "coverage": None,
        "concision": None,
        "clinical_relevance": None,
        "overall": None,
        "rationale": "Local judge response was not valid JSON.",
    }


def try_parse_first_json_object(text: str) -> dict[str, Any] | None:
    try:
        parsed, _ = json.JSONDecoder().raw_decode(text)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def first_json_object_text(text: str) -> str | None:
    stripped = text.strip()
    start = stripped.find("{")
    if start < 0:
        return None

    candidate = stripped[start:]
    try:
        _, end = json.JSONDecoder().raw_decode(candidate)
    except json.JSONDecodeError:
        return None
    return candidate[:end]
