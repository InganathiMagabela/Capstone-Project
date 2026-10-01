"""Data loading and cleaning - reproduces notebook section 2 exactly."""
import io
import numpy as np
import pandas as pd
from config import config as C
from utils.validation import validate_holdings_data


def load_dataset(source):
    """Read a CSV from a path, bytes or file-like object."""
    if isinstance(source, (bytes, bytearray)):
        source = io.BytesIO(source)
    return pd.read_csv(source)


def clean_raw(df):
    """Remove the mislabelled 'robin_dataset' block, parse time, drop duplicate (name, timestamp), sort.
    Additionally drops rows whose timestamp / holder count cannot be parsed (never occurs in the original data)."""
    df = df.copy()
    missing = [c for c in C.REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Data must contain columns {C.REQUIRED_COLUMNS}; missing {missing}.")
    df = df[df["name"] != C.MERGE_ERROR_NAME]
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["users_holding"] = pd.to_numeric(df["users_holding"], errors="coerce")
    df = df.dropna(subset=["name", "timestamp", "users_holding"])
    df = (df.drop_duplicates(["name", "timestamp"])
            .sort_values(["name", "timestamp"]).reset_index(drop=True))
    return df[["name", "timestamp", "users_holding"]]


def load_and_validate(source):
    """Load -> validate -> clean. Returns (clean_df | None, ValidationReport)."""
    try:
        raw = load_dataset(source)
    except Exception as exc:  # unreadable file
        from utils.validation import ValidationReport
        rep = ValidationReport(errors=[f"The file could not be read as a CSV table. Details: {exc}"])
        return None, rep
    rep = validate_holdings_data(raw)
    if not rep.ok:
        return None, rep
    return clean_raw(raw), rep


def find_default_data():
    """Prefer the original raw file; otherwise the cleaned file shipped in data/processed."""
    for p in (C.RAW_DATA_FILE, C.CLEAN_DATA_FILE):
        if p.exists():
            return p
    return None
