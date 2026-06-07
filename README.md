# LLM Post Training

Interview-facing LLM post-training showcase for clinical discharge-summary generation.

## Zero-Shot Baseline

The first baseline formats each MIMIC sample as one target discharge summary plus the
chronological supporting notes that share the same `SAMPLE_ID`.

Key files:

- `configs/zero_shot_baseline.json` - data/model/generation settings.
- `configs/zero_shot/*.json` - zero-shot prompt-variant configs.
- `src/data/mimic_discharge.py` - pandas JSON loader and prompt-ready formatter.
- `src/prompts/zero_shot_*.txt` - zero-shot prompt templates.
- `src/generation/base.py` - shared model/config/generation utilities.
- `src/generation/zero_shot.py` - zero-shot discharge summarizer.
- `scripts/run_zero_shot_experiments.py` - batch prompt-variant runner.
- `scripts/generate_performance_report.py` - Markdown/CSV performance report generator.

Render a prompt without loading the model:

```bash
.venv/bin/python main.py --dry-run --limit 1
```

Run a short local-cache smoke test:

```bash
.venv/bin/python main.py --sample-id 47 --local-files-only --max-new-tokens 16
```

Run the configured validation baseline:

```bash
.venv/bin/python main.py --config configs/zero_shot_baseline.json
```

Run resumable validation inference:

```bash
.venv/bin/python scripts/run_zero_shot_validation.py \
  --config configs/zero_shot_baseline.json \
  --split val \
  --local-files-only \
  --output outputs/zero_shot/val_predictions.jsonl \
  --resume
```

For a quick smoke test, add `--limit 1 --max-new-tokens 16`.

Evaluate saved predictions with package-backed BLEU and ROUGE metrics (`sacrebleu`,
`rouge-score`):

```bash
.venv/bin/python scripts/evaluate_summaries.py \
  --predictions outputs/zero_shot/val_predictions.jsonl
```

Verify both Hugging Face models are available from local cache only:

```bash
.venv/bin/python scripts/verify_local_models.py \
  --summarizer-config configs/zero_shot_baseline.json \
  --judge-config configs/llm_judge_local.json
```

Optionally run local LLM-as-judge:

```bash
.venv/bin/python scripts/evaluate_summaries.py \
  --predictions outputs/zero_shot/val_predictions.jsonl \
  --judge-config configs/llm_judge_local.json
```

The baseline and judge configs default to `local_files_only: true`; no clinical text is sent
to external APIs in these runs.

## Prompt Variant Experiments

Experiment artifacts are saved under `.experiments/`, which is ignored by git because the
files can contain generated clinical text, references, prompts, and local run metadata.

Use validation for prompt development. Run the zero-shot prompt-variant suite on a
small validation subset:

```bash
.venv/bin/python scripts/run_zero_shot_experiments.py \
  --split val \
  --limit 3 \
  --max-new-tokens 256 \
  --continue-on-error
```

Generate the interview-facing report on the test split:

```bash
.venv/bin/python scripts/run_zero_shot_experiments.py \
  --split test \
  --limit all \
  --resume \
  --continue-on-error
```

For a quick local test-set report before a full run, use a small limit:

```bash
.venv/bin/python scripts/run_zero_shot_experiments.py \
  --split test \
  --limit 3 \
  --max-new-tokens 256 \
  --judge-config configs/llm_judge_local.json \
  --continue-on-error \
  --continue-on-judge-error
```

Run the same suite on the full validation split if you want a larger prompt-development
comparison:

```bash
.venv/bin/python scripts/run_zero_shot_experiments.py \
  --split val \
  --limit all \
  --resume \
  --continue-on-error
```

The runner writes:

- `.experiments/<suite>/<run_id>/predictions/*.jsonl`
- `.experiments/<suite>/<run_id>/evaluations/*_metrics.jsonl`
- `.experiments/<suite>/<run_id>/evaluations/*_aggregate.json`
- `.experiments/<suite>/<run_id>/performance_report.md`
- `.experiments/<suite>/<run_id>/performance_report.csv`

Regenerate a report for an existing experiment directory:

```bash
.venv/bin/python scripts/generate_performance_report.py \
  --experiment-dir .experiments/zero_shot_prompt_variants/<run_id>
```

To include the local LLM-as-judge in a prompt-variant run, add:

```bash
--judge-config configs/llm_judge_local.json --continue-on-judge-error
```
