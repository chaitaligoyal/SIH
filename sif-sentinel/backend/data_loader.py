"""
Corpus loading for the clustering buffer.

Three bugs are fixed here relative to the original:

  1. `if os.path.exists(...)` sat at the same indentation level as the `for`
     loop, so it executed once after the loop with `file_path` frozen on the
     last entry. Only MSHA data was ever loaded.
  2. The code looked for `SIRDataDownload_2.csv`; the file on disk is
     `SIRDataDownload.csv`. It would never have matched anyway.
  3. The OSHA export is malformed (line 29 has 28 fields where 27 are
     expected) and `pd.read_csv` raised. The exception was swallowed by a bare
     `except Exception` that printed to a console nobody reads during a demo,
     so the failure was silent.

Rather than globbing a hardcoded list, this scans the data directory. Adding a
CSV is now the whole integration step.
"""

from __future__ import annotations

import glob
import os
from typing import Any, Dict, List

import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

TEXT_COLUMN_HINTS = ["narrative", "description", "summary", "text", "detail", "observation"]
ID_COLUMN_HINTS = ["id", "incident", "report", "ref", "case"]

MAX_ROWS_PER_FILE = 2000
MIN_TEXT_LENGTH = 25


def _pick_column(columns: List[str], hints: List[str]) -> str | None:
    for hint in hints:
        for col in columns:
            if hint in col.lower():
                return col
    return None


def _read_csv_resilient(path: str) -> pd.DataFrame | None:
    """Read a CSV that may be malformed, rather than raising on it."""
    attempts = [
        {"on_bad_lines": "skip"},
        {"on_bad_lines": "skip", "engine": "python"},
        {"on_bad_lines": "skip", "engine": "python", "sep": None},
    ]
    for kwargs in attempts:
        try:
            df = pd.read_csv(path, **kwargs)
            if not df.empty:
                return df
        except Exception:  # noqa: BLE001 - try the next parsing strategy
            continue
    return None


def load_corpus(data_dir: str = DATA_DIR) -> List[Dict[str, Any]]:
    """Load every CSV in `data_dir` into a flat list of {id, text, source}."""
    corpus: List[Dict[str, Any]] = []

    if not os.path.isdir(data_dir):
        print(f"[data_loader] Data directory not found: {data_dir}")
        return corpus

    csv_files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))

    # labelled.csv is the training set built from another file already in this
    # directory; loading it too would double-count every narrative.
    csv_files = [f for f in csv_files
                 if os.path.basename(f) not in {"labelled.csv", "feedback.csv"}]

    # Fictional demo fixtures are only used when there is no real data. Once
    # fetch_real_data.py has run, they are dropped so the clustering corpus is
    # entirely real narratives.
    real = [f for f in csv_files if not os.path.basename(f).startswith("DEMO_FIXTURE_")]
    if real:
        skipped = len(csv_files) - len(real)
        if skipped:
            print(f"[data_loader] Real data present — ignoring {skipped} demo fixture file(s).")
        csv_files = real

    if not csv_files:
        print(f"[data_loader] No CSV files in {data_dir}")
        return corpus

    for path in csv_files:
        name = os.path.basename(path)
        df = _read_csv_resilient(path)

        if df is None:
            print(f"[data_loader] {name}: could not be parsed, skipped.")
            continue

        text_col = _pick_column(list(df.columns), TEXT_COLUMN_HINTS)
        if text_col is None:
            print(f"[data_loader] {name}: no narrative column found, skipped.")
            continue

        id_col = _pick_column(list(df.columns), ID_COLUMN_HINTS) or df.columns[0]

        df = df.dropna(subset=[text_col]).head(MAX_ROWS_PER_FILE)
        source = os.path.splitext(name)[0]
        kept = 0

        for _, row in df.iterrows():
            text = str(row[text_col]).strip()
            if len(text) < MIN_TEXT_LENGTH:
                continue
            corpus.append({
                "id": f"{source}::{row[id_col]}",
                "text": text,
                "source": source,
                "site": str(row.get("city") or row.get("Address1") or "Unknown"),
            })
            kept += 1

        print(f"[data_loader] {name}: loaded {kept} reports (column '{text_col}').")

    print(f"[data_loader] Corpus size: {len(corpus)} reports.")
    return corpus
