from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .constants import (
    FEEDBACK_COLUMNS,
    FEEDBACK_PATH,
    SEED_DATASET_PATH,
    SESSION_COLUMNS,
    SESSION_LOG_PATH,
    TRAINING_COLUMNS,
)


def _ensure_csv(path: Path, headers: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)


def ensure_runtime_files() -> None:
    _ensure_csv(SESSION_LOG_PATH, SESSION_COLUMNS)
    _ensure_csv(FEEDBACK_PATH, FEEDBACK_COLUMNS)


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_seed_dataset() -> pd.DataFrame:
    frame = pd.read_csv(SEED_DATASET_PATH).fillna("")
    return frame[TRAINING_COLUMNS].copy()


def load_feedback_entries() -> pd.DataFrame:
    ensure_runtime_files()
    frame = pd.read_csv(FEEDBACK_PATH).fillna("")
    return frame


def _feedback_to_training_rows(seed_dataset: pd.DataFrame, feedback: pd.DataFrame) -> pd.DataFrame:
    if feedback.empty:
        return pd.DataFrame(columns=TRAINING_COLUMNS)

    metadata_lookup = (
        seed_dataset.sort_values("recommended_drug")
        .drop_duplicates("recommended_drug")
        .set_index("recommended_drug")[["drug_category", "advisor_note", "seek_care_if"]]
        .to_dict("index")
    )

    rows: list[dict[str, str]] = []
    for entry in feedback.to_dict("records"):
        confirmed_drug = str(entry.get("confirmed_drug", "")).strip()
        metadata = metadata_lookup.get(confirmed_drug)
        if not confirmed_drug or metadata is None:
            continue
        rows.append(
            {
                "age_group": str(entry.get("age_group", "adult")).strip() or "adult",
                "gender": str(entry.get("gender", "Prefer not to say")).strip() or "Prefer not to say",
                "symptoms": str(entry.get("symptoms", "")).strip(),
                "recommended_drug": confirmed_drug,
                "drug_category": metadata["drug_category"],
                "advisor_note": metadata["advisor_note"],
                "seek_care_if": metadata["seek_care_if"],
                "source": "confirmed-feedback",
            }
        )

    return pd.DataFrame(rows, columns=TRAINING_COLUMNS)


def load_training_dataset() -> pd.DataFrame:
    seed_dataset = load_seed_dataset()
    feedback = load_feedback_entries()
    feedback_rows = _feedback_to_training_rows(seed_dataset, feedback)
    if feedback_rows.empty:
        return seed_dataset
    return pd.concat([seed_dataset, feedback_rows], ignore_index=True)


def append_session_log(record: dict[str, object]) -> None:
    ensure_runtime_files()
    row = {column: record.get(column, "") for column in SESSION_COLUMNS}
    with SESSION_LOG_PATH.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SESSION_COLUMNS)
        writer.writerow(row)


def append_feedback_entry(record: dict[str, object]) -> None:
    ensure_runtime_files()
    row = {column: record.get(column, "") for column in FEEDBACK_COLUMNS}
    with FEEDBACK_PATH.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FEEDBACK_COLUMNS)
        writer.writerow(row)


def read_recent_rows(path: Path, limit: int = 10) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path).fillna("")
    if frame.empty:
        return frame
    return frame.tail(limit).iloc[::-1].reset_index(drop=True)


def load_recent_sessions(limit: int = 10) -> pd.DataFrame:
    return read_recent_rows(SESSION_LOG_PATH, limit=limit)


def load_recent_feedback(limit: int = 10) -> pd.DataFrame:
    return read_recent_rows(FEEDBACK_PATH, limit=limit)

