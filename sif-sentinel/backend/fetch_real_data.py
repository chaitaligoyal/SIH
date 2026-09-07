"""
Download real incident narratives from US regulator open-data portals.

    python backend/fetch_real_data.py

Replaces the three LLM-generated CSVs in data/ with actual regulator data.

MSHA is the primary source and it is a genuinely good one for this problem:
~250,000 free-text accident narratives from 2000 onward, each one already
carrying a structured degree-of-injury code and an immediate-notification code.
That structure is what build_labels.py turns into a training set without any
manual annotation. See:
    https://arlweb.msha.gov/opengovernmentdata/ogimsha.asp

Files are pipe-delimited with a header row and are refreshed every Friday.

PHMSA and BSEE need a browser click (PHMSA gates its zips, BSEE publishes
xlsx behind a query form), so this script prints their URLs rather than
pretending it can fetch them.
"""

from __future__ import annotations

import argparse
import io
import os
import sys
import urllib.request
import zipfile

import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

MSHA_ACCIDENTS_URL = "https://arlweb.msha.gov/opengovernmentdata/DataSets/Accidents.zip"

MANUAL_SOURCES = {
    "PHMSA pipeline incidents (has NARRATIVE + Serious/Significant flags)":
        "https://www.phmsa.dot.gov/data-and-statistics/pipeline/pipeline-incident-flagged-files",
    "BSEE offshore incident statistics (xlsx per year)":
        "https://www.bsee.gov/stats-facts/offshore-incident-statistics",
    "BSEE raw data tables":
        "https://www.data.bsee.gov/Main/RawData.aspx",
    "OSHA Severe Injury Reports (you already have this one)":
        "https://www.osha.gov/severeinjury",
}

# Columns worth keeping. The rest of the ~57 columns are administrative.
KEEP = [
    "MINE_ID", "DOCUMENT_NO", "ACCIDENT_DT", "CAL_YR", "SUBUNIT",
    "DEGREE_INJURY_CD", "DEGREE_INJURY",
    "IMMED_NOTIFY_CD", "IMMED_NOTIFY",
    "CLASSIFICATION", "ACCIDENT_TYPE", "OCCUPATION", "ACTIVITY",
    "MINING_EQUIP", "INJURY_SOURCE", "NATURE_INJURY", "INJ_BODY_PART",
    "UG_LOCATION", "CONTRACTOR_ID", "OPERATOR_NAME",
    "NO_INJURIES", "DAYS_LOST", "COAL_METAL_IND",
    "NARRATIVE",
]


def download(url: str) -> bytes:
    print(f"Downloading {url}")
    print("  (MSHA Accidents.zip is roughly 30-40 MB — give it a minute.)")
    req = urllib.request.Request(url, headers={"User-Agent": "SIF-Sentinel/1.0"})
    with urllib.request.urlopen(req, timeout=300) as resp:
        return resp.read()


def fetch_msha(limit: int | None) -> None:
    try:
        blob = download(MSHA_ACCIDENTS_URL)
    except Exception as exc:  # noqa: BLE001
        print(f"\nDownload failed: {exc}")
        print("If you are behind a proxy or the site is down, download the zip")
        print(f"manually from {MSHA_ACCIDENTS_URL} and unzip it into data/.")
        sys.exit(1)

    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        inner = zf.namelist()[0]
        print(f"Extracting {inner}")
        with zf.open(inner) as fh:
            # Pipe-delimited, header row, latin-1 (the file has some
            # non-UTF8 bytes in older narratives).
            df = pd.read_csv(
                fh, sep="|", dtype=str, encoding="latin-1",
                on_bad_lines="skip", low_memory=False,
            )

    print(f"Parsed {len(df):,} rows, {len(df.columns)} columns.")

    df = df[df["NARRATIVE"].notna()]
    df = df[df["NARRATIVE"].str.len() >= 30]
    print(f"{len(df):,} rows have a usable narrative.")

    cols = [c for c in KEEP if c in df.columns]
    df = df[cols]

    if limit:
        # Most recent first — reporting practice and vocabulary drift over
        # 25 years, and recent narratives look more like what OIL files today.
        df = df.sort_values("CAL_YR", ascending=False).head(limit)
        print(f"Keeping the most recent {len(df):,} rows.")

    os.makedirs(DATA_DIR, exist_ok=True)
    out = os.path.join(DATA_DIR, "MSHA_Accidents_real.csv")
    df.to_csv(out, index=False)
    print(f"\nWrote {out}  ({len(df):,} rows)")

    print("\nDegree of injury distribution:")
    print(df["DEGREE_INJURY"].value_counts().head(12).to_string())
    print("\nImmediate notification distribution:")
    print(df["IMMED_NOTIFY"].value_counts().head(12).to_string())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=60000,
                    help="Cap rows kept (0 = keep everything). Default 60000, "
                         "which is plenty for training and keeps the repo small.")
    args = ap.parse_args()

    fetch_msha(args.limit or None)

    print("\n" + "=" * 68)
    print("Sources that need a manual browser download:")
    print("=" * 68)
    for label, url in MANUAL_SOURCES.items():
        print(f"  {label}\n    {url}\n")
    print("Drop whatever you download into data/ — the loader autodetects")
    print("narrative columns, so no code change is needed.")
    print("\nOnce MSHA_Accidents_real.csv exists, run:")
    print("    python backend/build_labels.py")


if __name__ == "__main__":
    main()
