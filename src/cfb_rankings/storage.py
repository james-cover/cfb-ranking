from __future__ import annotations

import os
import tempfile
from collections.abc import Iterable
from pathlib import Path

import pandas as pd


def read_csv(path: Path, **kwargs) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path, **kwargs)


def atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    """Write a CSV without leaving a partial file if the process is interrupted."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(handle, "w", newline="", encoding="utf-8") as stream:
            frame.to_csv(stream, index=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def upsert_csv(
    new_rows: pd.DataFrame,
    path: Path,
    keys: Iterable[str],
    *,
    sort_by: Iterable[str] | None = None,
) -> pd.DataFrame:
    keys = list(keys)
    if new_rows.empty:
        return read_csv(path)
    old_rows = read_csv(path)
    combined = pd.concat([old_rows, new_rows], ignore_index=True)
    missing = [key for key in keys if key not in combined.columns]
    if missing:
        raise ValueError(f"Cannot upsert {path.name}; missing keys: {missing}")
    combined = combined.drop_duplicates(keys, keep="last")
    if sort_by:
        available = [column for column in sort_by if column in combined.columns]
        if available:
            combined = combined.sort_values(available, kind="stable")
    combined = combined.reset_index(drop=True)
    atomic_write_csv(combined, path)
    return combined


def append_snapshot(frame: pd.DataFrame, path: Path, keys: Iterable[str]) -> pd.DataFrame:
    """Append immutable timestamped observations, de-duplicating exact snapshots."""
    return upsert_csv(frame, path, keys, sort_by=keys)
