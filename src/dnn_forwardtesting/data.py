from __future__ import annotations

from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = ("date", "open", "high", "low", "close")


class DataQualityError(ValueError):
    """Raised when an OHLCV frame violates the package data contract."""


def validate_ohlcv(data: pd.DataFrame, min_rows: int = 2) -> pd.DataFrame:
    """Validate and return a normalized copy without silently changing row order."""
    if not isinstance(data, pd.DataFrame):
        raise DataQualityError("data must be a pandas DataFrame")
    frame = data.copy()
    frame.columns = [str(column).strip().lower() for column in frame.columns]
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise DataQualityError(f"missing required columns: {', '.join(missing)}")

    columns = list(REQUIRED_COLUMNS)
    if "volume" in frame.columns:
        columns.append("volume")
    frame = frame.loc[:, columns]
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce", format="mixed")
    if frame["date"].isna().any():
        raise DataQualityError("date contains invalid or missing values")
    if frame["date"].duplicated().any():
        raise DataQualityError("date contains duplicates")
    if not frame["date"].is_monotonic_increasing:
        raise DataQualityError("date must be sorted in ascending order")

    numeric_columns = [column for column in columns if column != "date"]
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    invalid = [column for column in numeric_columns if frame[column].isna().any()]
    if invalid:
        raise DataQualityError(f"non-numeric or missing values: {', '.join(invalid)}")

    price_columns = ["open", "high", "low", "close"]
    if (frame[price_columns] <= 0).any().any():
        raise DataQualityError("OHLC prices must be positive")
    if (frame["high"] < frame[["open", "low", "close"]].max(axis=1)).any():
        raise DataQualityError("high must be at least open, low, and close")
    if (frame["low"] > frame[["open", "high", "close"]].min(axis=1)).any():
        raise DataQualityError("low must be at most open, high, and close")
    if len(frame) < min_rows:
        raise DataQualityError(f"expected at least {min_rows} rows, got {len(frame)}")
    return frame.reset_index(drop=True)


def load_ohlcv(path: str | Path, min_rows: int = 2) -> pd.DataFrame:
    """Load a CSV and enforce the OHLCV contract."""
    return validate_ohlcv(pd.read_csv(path), min_rows=min_rows)
