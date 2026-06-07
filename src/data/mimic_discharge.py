from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import pandas as pd


@dataclass(frozen=True)
class DischargeSummaryExample:
    sample_id: int
    split: str
    patient_context: str
    source_notes: str
    target_summary: str
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class MimicDischargeDataLoader:
    """Loads MIMIC discharge-summary splits and formats notes for LLM prompts."""

    required_columns = {
        "ROW_ID",
        "SUBJECT_ID",
        "HADM_ID",
        "CHARTDATE",
        "CHARTTIME",
        "CATEGORY",
        "DESCRIPTION",
        "TEXT",
        "SAMPLE_ID",
        "SPLIT",
    }

    def __init__(
        self,
        data_dir: str | Path,
        split: str = "val",
        *,
        include_categories: Sequence[str] | None = None,
        exclude_categories: Sequence[str] | None = ("Discharge summary",),
        exclude_iserror: bool = True,
        max_notes_per_sample: int | None = None,
        max_chars_per_note: int | None = 2500,
        max_total_chars: int | None = 30000,
        text_truncation: str = "head_tail",
    ) -> None:
        self.data_dir = Path(data_dir)
        self.split = split
        self.include_categories = set(include_categories or [])
        self.exclude_categories = set(exclude_categories or [])
        self.exclude_iserror = exclude_iserror
        self.max_notes_per_sample = max_notes_per_sample
        self.max_chars_per_note = max_chars_per_note
        self.max_total_chars = max_total_chars
        self.text_truncation = text_truncation

    def load_examples(
        self,
        *,
        limit: int | None = None,
        sample_ids: Iterable[int] | None = None,
    ) -> list[DischargeSummaryExample]:
        inputs, outputs = self.load_split()
        inputs = self._filter_inputs(inputs)
        outputs = self._filter_outputs(outputs)

        requested_ids = [int(sample_id) for sample_id in sample_ids or []]
        if requested_ids:
            sample_order = requested_ids
        else:
            sample_order = [int(sample_id) for sample_id in outputs["SAMPLE_ID"].drop_duplicates()]

        if limit is not None:
            sample_order = sample_order[:limit]

        input_groups = inputs.groupby("SAMPLE_ID", sort=False)
        output_groups = outputs.groupby("SAMPLE_ID", sort=False)

        examples: list[DischargeSummaryExample] = []
        for sample_id in sample_order:
            if sample_id not in output_groups.groups:
                continue

            source_rows = (
                input_groups.get_group(sample_id)
                if sample_id in input_groups.groups
                else inputs.iloc[0:0].copy()
            )
            target_rows = output_groups.get_group(sample_id)
            examples.append(self._format_example(sample_id, source_rows, target_rows))

        return examples

    def load_split(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        input_path = self.data_dir / f"{self.split}_inputs.json"
        output_path = self.data_dir / f"{self.split}_outputs.json"

        if not input_path.exists():
            raise FileNotFoundError(f"Missing input split file: {input_path}")
        if not output_path.exists():
            raise FileNotFoundError(f"Missing output split file: {output_path}")

        inputs = pd.read_json(input_path)
        outputs = pd.read_json(output_path)
        self._validate_columns(inputs, input_path)
        self._validate_columns(outputs, output_path)
        return inputs, outputs

    def _format_example(
        self,
        sample_id: int,
        source_rows: pd.DataFrame,
        target_rows: pd.DataFrame,
    ) -> DischargeSummaryExample:
        target_row = self._sort_notes(target_rows).iloc[0]
        source_rows = self._sort_notes(source_rows)

        if self.max_notes_per_sample is not None:
            source_rows = source_rows.head(self.max_notes_per_sample)

        metadata = self._metadata(sample_id, source_rows, target_row)
        patient_context = self._format_patient_context(metadata)
        source_notes = self._format_source_notes(source_rows)
        target_summary = normalize_note_text(target_row.get("TEXT", ""))

        return DischargeSummaryExample(
            sample_id=sample_id,
            split=str(target_row.get("SPLIT", self.split)),
            patient_context=patient_context,
            source_notes=source_notes,
            target_summary=target_summary,
            metadata=metadata,
        )

    def _filter_inputs(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df[df["SAMPLE_ID"].notna()].copy()

        if self.exclude_iserror and "ISERROR" in df.columns:
            df = df[df["ISERROR"].fillna(0) != 1]

        if self.include_categories:
            df = df[df["CATEGORY"].isin(self.include_categories)]

        if self.exclude_categories:
            df = df[~df["CATEGORY"].isin(self.exclude_categories)]

        df["TEXT"] = df["TEXT"].map(normalize_note_text)
        df = df[df["TEXT"].str.len() > 0]
        return df

    def _filter_outputs(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df[df["SAMPLE_ID"].notna()].copy()
        df["TEXT"] = df["TEXT"].map(normalize_note_text)
        df = df[df["TEXT"].str.len() > 0]
        return df

    def _format_patient_context(self, metadata: dict[str, Any]) -> str:
        lines = [
            f"Sample ID: {metadata['sample_id']}",
            f"Split: {metadata['split']}",
            f"Subject ID: {metadata.get('subject_id') or 'unknown'}",
            f"Admission ID: {metadata.get('hadm_id') or 'unknown'}",
            f"Target discharge note date: {metadata.get('target_chartdate') or 'unknown'}",
            f"Supporting notes included: {metadata['notes_included']}",
        ]
        if metadata["notes_total"] != metadata["notes_included"]:
            lines.append(f"Supporting notes available before limits: {metadata['notes_total']}")
        return "\n".join(lines)

    def _format_source_notes(self, rows: pd.DataFrame) -> str:
        if rows.empty:
            return "No supporting notes were available for this sample."

        chunks: list[str] = []
        current_date: str | None = None
        total_chars = 0
        notes_written = 0

        for _, row in rows.iterrows():
            date_label, time_label = format_note_datetime(row)
            if date_label != current_date:
                date_header = f"\n### {date_label}\n"
                if self._would_exceed_budget(total_chars, date_header):
                    chunks.append(self._truncation_notice(notes_written, rows))
                    break
                chunks.append(date_header)
                total_chars += len(date_header)
                current_date = date_label

            raw_text = normalize_note_text(row.get("TEXT", ""))
            note_text = truncate_text(raw_text, self.max_chars_per_note, self.text_truncation)
            header = self._format_note_header(row, time_label, notes_written + 1)
            note_block = f"{header}\n{note_text}\n"

            if self._would_exceed_budget(total_chars, note_block):
                chunks.append(self._truncation_notice(notes_written, rows))
                break

            chunks.append(note_block)
            total_chars += len(note_block)
            notes_written += 1

        return "".join(chunks).strip()

    def _format_note_header(self, row: pd.Series, time_label: str | None, note_number: int) -> str:
        category = clean_scalar(row.get("CATEGORY")) or "Unknown category"
        description = clean_scalar(row.get("DESCRIPTION")) or "No description"
        row_id = clean_id(row.get("ROW_ID"))
        time_text = time_label or "date only"
        return f"[{note_number}] {time_text} | {category} / {description} | row_id={row_id or 'unknown'}"

    def _metadata(
        self,
        sample_id: int,
        source_rows: pd.DataFrame,
        target_row: pd.Series,
    ) -> dict[str, Any]:
        target_date, _ = format_note_datetime(target_row)
        return {
            "sample_id": sample_id,
            "split": clean_scalar(target_row.get("SPLIT")) or self.split,
            "subject_id": clean_id(target_row.get("SUBJECT_ID"))
            or most_common_id(source_rows, "SUBJECT_ID"),
            "hadm_id": clean_id(target_row.get("HADM_ID")) or most_common_id(source_rows, "HADM_ID"),
            "target_row_id": clean_id(target_row.get("ROW_ID")),
            "target_chartdate": target_date,
            "notes_total": int(len(source_rows)),
            "notes_included": self._count_notes_that_fit(source_rows),
            "source_categories": source_rows["CATEGORY"].value_counts().to_dict()
            if not source_rows.empty
            else {},
        }

    def _count_notes_that_fit(self, rows: pd.DataFrame) -> int:
        if self.max_notes_per_sample is not None:
            return min(int(len(rows)), self.max_notes_per_sample)
        if self.max_total_chars is None:
            return int(len(rows))

        total_chars = 0
        notes_written = 0
        current_date: str | None = None
        for _, row in rows.iterrows():
            date_label, time_label = format_note_datetime(row)
            if date_label != current_date:
                total_chars += len(f"\n### {date_label}\n")
                current_date = date_label

            note_text = truncate_text(
                normalize_note_text(row.get("TEXT", "")),
                self.max_chars_per_note,
                self.text_truncation,
            )
            note_block = f"{self._format_note_header(row, time_label, notes_written + 1)}\n{note_text}\n"
            if total_chars + len(note_block) > self.max_total_chars:
                break
            total_chars += len(note_block)
            notes_written += 1
        return notes_written

    def _would_exceed_budget(self, current_chars: int, next_text: str) -> bool:
        return self.max_total_chars is not None and current_chars + len(next_text) > self.max_total_chars

    def _truncation_notice(self, notes_written: int, rows: pd.DataFrame) -> str:
        remaining = max(int(len(rows)) - notes_written, 0)
        return (
            f"\n[Context truncated after {notes_written} notes; "
            f"{remaining} additional notes omitted by loader budget.]\n"
        )

    def _sort_notes(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df.copy()

        df = df.copy()
        df["_SORT_CHARTDATE"] = pd.to_datetime(df["CHARTDATE"], unit="ms", errors="coerce")
        df["_SORT_CHARTTIME"] = pd.to_datetime(df["CHARTTIME"], errors="coerce")
        df["_SORT_ROW_ID"] = pd.to_numeric(df["ROW_ID"], errors="coerce")
        return df.sort_values(
            ["_SORT_CHARTDATE", "_SORT_CHARTTIME", "_SORT_ROW_ID"],
            na_position="last",
        ).drop(columns=["_SORT_CHARTDATE", "_SORT_CHARTTIME", "_SORT_ROW_ID"])

    def _validate_columns(self, df: pd.DataFrame, path: Path) -> None:
        missing = self.required_columns.difference(df.columns)
        if missing:
            missing_list = ", ".join(sorted(missing))
            raise ValueError(f"{path} is missing required columns: {missing_list}")


def normalize_note_text(value: Any) -> str:
    value = clean_scalar(value)
    if value is None:
        return ""

    text = value.replace("\r\n", "\n").replace("\r", "\n").strip()
    lines = [line.rstrip() for line in text.splitlines()]

    compacted: list[str] = []
    blank_count = 0
    for line in lines:
        if line.strip():
            compacted.append(line)
            blank_count = 0
        else:
            blank_count += 1
            if blank_count <= 1:
                compacted.append("")

    return "\n".join(compacted).strip()


def truncate_text(text: str, max_chars: int | None, strategy: str = "head_tail") -> str:
    if max_chars is None or len(text) <= max_chars:
        return text

    marker = "\n[... note text truncated ...]\n"
    if max_chars <= len(marker) + 20:
        return text[:max_chars].rstrip()

    if strategy == "tail":
        return marker + text[-(max_chars - len(marker)) :].lstrip()
    if strategy == "head":
        return text[: max_chars - len(marker)].rstrip() + marker

    head_chars = (max_chars - len(marker)) // 2
    tail_chars = max_chars - len(marker) - head_chars
    return text[:head_chars].rstrip() + marker + text[-tail_chars:].lstrip()


def format_note_datetime(row: pd.Series) -> tuple[str, str | None]:
    chart_time = clean_scalar(row.get("CHARTTIME"))
    if chart_time:
        parsed_time = pd.to_datetime(chart_time, errors="coerce")
        if not pd.isna(parsed_time):
            return parsed_time.strftime("%Y-%m-%d"), parsed_time.strftime("%H:%M")

    chart_date = row.get("CHARTDATE")
    parsed_date = pd.to_datetime(chart_date, unit="ms", errors="coerce")
    if pd.isna(parsed_date):
        return "unknown date", None
    return parsed_date.strftime("%Y-%m-%d"), None


def clean_scalar(value: Any) -> str | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    return text


def clean_id(value: Any) -> int | None:
    text = clean_scalar(value)
    if text is None:
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


def most_common_id(df: pd.DataFrame, column: str) -> int | None:
    if df.empty or column not in df.columns:
        return None
    values = df[column].dropna()
    if values.empty:
        return None
    return clean_id(values.mode().iloc[0])
