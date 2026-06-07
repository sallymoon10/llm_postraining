from __future__ import annotations

import re
from statistics import mean
from typing import Any

from rouge_score import rouge_scorer
from sacrebleu.metrics import BLEU


TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?")
ROUGE_SCORER = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
BLEU_SCORERS = {
    1: BLEU(smooth_method="exp", max_ngram_order=1, effective_order=True),
    2: BLEU(smooth_method="exp", max_ngram_order=2, effective_order=True),
    4: BLEU(smooth_method="exp", max_ngram_order=4, effective_order=True),
}


def compute_summary_metrics(prediction: str, reference: str) -> dict[str, float]:
    prediction = prediction or ""
    reference = reference or ""
    pred_tokens = tokenize(prediction)
    ref_tokens = tokenize(reference)
    rouge_scores = ROUGE_SCORER.score(reference, prediction)

    metrics = {
        "bleu_1": bleu_score(prediction, reference, max_order=1),
        "bleu_2": bleu_score(prediction, reference, max_order=2),
        "bleu_4": bleu_score(prediction, reference, max_order=4),
        "rouge_1_precision": rouge_scores["rouge1"].precision,
        "rouge_1_recall": rouge_scores["rouge1"].recall,
        "rouge_1_f1": rouge_scores["rouge1"].fmeasure,
        "rouge_2_precision": rouge_scores["rouge2"].precision,
        "rouge_2_recall": rouge_scores["rouge2"].recall,
        "rouge_2_f1": rouge_scores["rouge2"].fmeasure,
        "rouge_l_precision": rouge_scores["rougeL"].precision,
        "rouge_l_recall": rouge_scores["rougeL"].recall,
        "rouge_l_f1": rouge_scores["rougeL"].fmeasure,
        "prediction_tokens": float(len(pred_tokens)),
        "reference_tokens": float(len(ref_tokens)),
        "length_ratio": safe_div(len(pred_tokens), len(ref_tokens)),
    }
    return metrics


def aggregate_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    if not rows:
        return {"count": 0.0}

    metric_keys = [
        key
        for key, value in rows[0].get("metrics", {}).items()
        if isinstance(value, int | float)
    ]
    aggregate = {"count": float(len(rows))}
    for key in metric_keys:
        aggregate[key] = mean(float(row["metrics"][key]) for row in rows)
    return aggregate


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall((text or "").lower())


def bleu_score(prediction: str, reference: str, max_order: int) -> float:
    if not prediction.strip() or not reference.strip():
        return 0.0
    score = BLEU_SCORERS[max_order].corpus_score([prediction], [[reference]])
    return bounded_score(score.score / 100.0)


def safe_div(numerator: int | float, denominator: int | float) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator) / float(denominator)


def bounded_score(value: int | float) -> float:
    return max(0.0, min(1.0, float(value)))
