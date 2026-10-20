"""Read-only loading and validation of the supplied historical dataset.

The original CSV is never modified. Cleaning happens in memory only.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import pandas as pd

from . import config as C


class DataValidationError(Exception):
    """Raised when the dataset is missing, unreadable or lacks required columns."""


@dataclass
class QualityReport:
    path: str
    rows_raw: int
    rows_clean: int
    exact_duplicate_rows: int
    duplicate_dates_after_dedupe: int
    missing_dates: List[str]
    date_min: str
    date_max: str
    missing_values: dict
    class_counts: dict
    notes: List[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return self.__dict__.copy()


def _check_columns(df: pd.DataFrame) -> None:
    missing = [c for c in C.REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise DataValidationError(f"Dataset is missing required column(s): {missing}")


def load_raw(path: Optional[Path] = None) -> pd.DataFrame:
    """Load the CSV exactly as stored (Date parsed). Raises DataValidationError."""
    path = Path(path) if path else C.DATA_PATH
    if not path.exists():
        raise DataValidationError(
            f"Dataset not found at {path}. Place delhi_heatwave_airquality.csv in the data/ folder.")
    try:
        df = pd.read_csv(path)
    except Exception as exc:  # pragma: no cover - defensive
        raise DataValidationError(f"Could not read {path}: {exc}") from exc
    _check_columns(df)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)   # mixed-format inference notice is irrelevant here
        df[C.DATE_COL] = pd.to_datetime(df[C.DATE_COL], errors="coerce")
    if df[C.DATE_COL].isna().any():
        raise DataValidationError("Dataset contains unparseable dates.")
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Drop exact duplicate rows, sort chronologically, enforce unique dates."""
    out = df.drop_duplicates().sort_values(C.DATE_COL).reset_index(drop=True)
    if out[C.DATE_COL].duplicated().any():
        # Same date with different values: keep the first and let the report note it.
        out = out.drop_duplicates(subset=C.DATE_COL, keep="first").reset_index(drop=True)
    return out


def find_missing_dates(df: pd.DataFrame) -> List[pd.Timestamp]:
    full = pd.date_range(df[C.DATE_COL].min(), df[C.DATE_COL].max(), freq="D")
    return [d for d in full if d not in set(df[C.DATE_COL])]


def quality_report(raw: pd.DataFrame, cleaned: pd.DataFrame, path: Optional[Path] = None) -> QualityReport:
    exact = int(raw.duplicated().sum())
    dup_dates = int(raw.drop_duplicates()[C.DATE_COL].duplicated().sum())
    missing_dates = [d.strftime("%Y-%m-%d") for d in find_missing_dates(cleaned)]
    notes = []
    if exact:
        notes.append(f"{exact} exact duplicate row(s) removed in memory (original file unchanged).")
    if missing_dates:
        notes.append(f"{len(missing_dates)} calendar date(s) absent from the file: {', '.join(missing_dates)}.")
    notes.append("Data is DAILY. No hourly observations exist in the supplied file.")
    return QualityReport(
        path=str(path or C.DATA_PATH), rows_raw=len(raw), rows_clean=len(cleaned),
        exact_duplicate_rows=exact, duplicate_dates_after_dedupe=dup_dates,
        missing_dates=missing_dates,
        date_min=cleaned[C.DATE_COL].min().strftime("%Y-%m-%d"),
        date_max=cleaned[C.DATE_COL].max().strftime("%Y-%m-%d"),
        missing_values={c: int(v) for c, v in cleaned.isna().sum().items() if v > 0},
        class_counts=cleaned[C.TARGET_COL].value_counts().to_dict(), notes=notes)


def load_dataset(path: Optional[Path] = None):
    """Return (clean_dataframe, QualityReport). Adds derived five-level columns."""
    from .risk_scoring import categorize_series  # local import avoids cycles

    raw = load_raw(path)
    cleaned = clean(raw)
    report = quality_report(raw, cleaned, path)
    cleaned["Score_Category_5L"] = categorize_series(cleaned[C.SCORE_COL])
    return cleaned, report


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    """UTF-8 CSV bytes for download buttons."""
    return df.to_csv(index=False).encode("utf-8")
