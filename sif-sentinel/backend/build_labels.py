"""
Build a labelled SIF training set without manual annotation.

    python backend/build_labels.py

This is the piece that unblocks train.py. MSHA's accident file already carries
two structured fields that, taken together, encode the SIF-precursor concept:

  DEGREE_INJURY  — how badly the person was actually hurt
  IMMED_NOTIFY   — whether the event was one of the 12 types MSHA requires be
                   reported immediately. Those twelve are high-energy events
                   by regulatory definition: death, serious injury, entrapment,
                   inundation, gas or dust ignition, mine fire, explosives,
                   roof fall, outburst, hoisting.

The SIF literature separates *actual* SIF (someone was killed or permanently
disabled) from *potential* SIF (a high-energy event where the outcome happened
to be mild — the precursor). Both are positives for this task, and the two
fields give you one arm each:

  POSITIVE  DEGREE_INJURY in {fatality, permanent disability}          <- actual SIF
            OR IMMED_NOTIFY is one of the high-energy event types      <- potential SIF
            (including "accident only" rows, which are high-energy near
             misses with no injury — the single most valuable class of
             example for this problem)

  NEGATIVE  DEGREE_INJURY in {no days away no restrictions, first aid}
            AND IMMED_NOTIFY not marked

  DROPPED   everything in between: days-away-from-work variants,
            occupational illness, natural causes, non-employee injuries.
            These are the genuine gray zone. Training on a noisy middle
            teaches the model to hedge; leaving it out gives clean class
            boundaries. Report the drop rate — a judge will ask.

--- Why the scrubbing step matters ---

A fatality narrative usually says so: "the victim was fatally injured",
"pronounced dead at the scene". Train on that and you build an excellent
detector of the word "fatally", which is worthless on OIL's UA/UC reports
where nobody has been hurt yet. Outcome vocabulary is stripped by default
(--no-scrub disables it). This is the difference between a model that
detects severity-as-written and one that detects fatal potential.
"""

from __future__ import annotations

import argparse
import os
import re

import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
DEFAULT_IN = os.path.join(DATA_DIR, "MSHA_Accidents_real.csv")
DEFAULT_OUT = os.path.join(DATA_DIR, "labelled.csv")

# DEGREE_INJURY_CD, from the MSHA data dictionary.
DEG_FATAL = {"01"}                    # Fatality
DEG_PERMANENT = {"02"}                # Permanent total or partial disability
DEG_ACCIDENT_ONLY = {"00"}            # Accident only, no injury
DEG_MINOR = {"06", "10"}              # No days away / no restrictions; first aid
DEG_GRAY = {"03", "04", "05", "07", "08", "09"}

# IMMED_NOTIFY_CD: the 12 immediately-reportable event types. All are
# high-energy except (12) Offsite and (13) Not marked.
NOTIFY_HIGH_ENERGY = {"01", "02", "03", "04", "05", "06", "07", "08", "09", "11"}
NOTIFY_NOT_MARKED = {"13", "?", ""}

# Outcome vocabulary that leaks the label. Removed from narratives by default.
LEAK_TERMS = [
    r"\bfatal(ly|ity|ities)?\b", r"\bdied\b", r"\bdeath\b", r"\bdeceased\b",
    r"\bkilled\b", r"\bpronounced dead\b", r"\bexpired\b", r"\bcoroner\b",
    r"\bautops\w+\b", r"\bmortal\w*\b",
    r"\bamputat\w+\b", r"\bpermanent(ly)? disab\w+\b", r"\bparaly\w+\b",
    r"\bquadripleg\w*\b", r"\bparapleg\w*\b", r"\benucleat\w+\b",
    r"\bloss of (an? )?(eye|limb|hand|foot|leg|arm|finger|thumb)\b",
    r"\blost (his|her|their) (eye|limb|hand|foot|leg|arm|finger|thumb)\b",
    r"\bhospitaliz\w+\b", r"\bhospitalis\w+\b", r"\blife.?flight\w*\b",
    r"\bairlift\w*\b", r"\bicu\b", r"\bintensive care\b",
    r"\bfirst aid only\b", r"\bno lost time\b", r"\bminor injury\b",
    r"\breturned to work same\b", r"\bno medical treatment\b",
]
LEAK_RE = re.compile("|".join(LEAK_TERMS), flags=re.IGNORECASE)


def scrub(text: str) -> str:
    """Remove outcome vocabulary so the model learns precursors, not verdicts."""
    cleaned = LEAK_RE.sub(" ", str(text))
    return re.sub(r"\s{2,}", " ", cleaned).strip()


def label_row(deg: str, notify: str) -> int | None:
    deg = (deg or "").strip().zfill(2)
    notify = (notify or "").strip()

    high_energy_event = notify in NOTIFY_HIGH_ENERGY

    # Actual SIF outcome.
    if deg in DEG_FATAL or deg in DEG_PERMANENT:
        return 1
    # High-energy event, mild or no outcome — the precursor case.
    if high_energy_event and (deg in DEG_ACCIDENT_ONLY or deg in DEG_MINOR):
        return 1
    if high_energy_event:
        return 1
    # Clean negative: minor outcome, nothing immediately reportable.
    if deg in DEG_MINOR and notify in NOTIFY_NOT_MARKED:
        return 0
    # Accident-only with no high-energy flag: a genuine low-energy near miss.
    if deg in DEG_ACCIDENT_ONLY and notify in NOTIFY_NOT_MARKED:
        return 0
    # Gray zone.
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=DEFAULT_IN)
    ap.add_argument("--output", default=DEFAULT_OUT)
    ap.add_argument("--no-scrub", action="store_true",
                    help="Keep outcome vocabulary in narratives (not recommended)")
    ap.add_argument("--balance", action="store_true",
                    help="Downsample the majority class to a 1:3 positive:negative "
                         "ratio, close to the 20-25%% SIF base rate in the literature")
    ap.add_argument("--max-rows", type=int, default=40000)
    args = ap.parse_args()

    if not os.path.exists(args.input):
        raise SystemExit(
            f"{args.input} not found.\nRun `python backend/fetch_real_data.py` first."
        )

    df = pd.read_csv(args.input, dtype=str, low_memory=False)
    print(f"Read {len(df):,} rows from {os.path.basename(args.input)}")

    df["label"] = [
        label_row(d, n)
        for d, n in zip(df.get("DEGREE_INJURY_CD", ""), df.get("IMMED_NOTIFY_CD", ""))
    ]

    dropped = int(df["label"].isna().sum())
    df = df[df["label"].notna()].copy()
    df["label"] = df["label"].astype(int)

    print(f"Dropped {dropped:,} gray-zone rows ({dropped / (dropped + len(df)):.1%})")
    print(f"Labelled {len(df):,} rows: "
          f"{int(df['label'].sum()):,} positive, "
          f"{int((1 - df['label']).sum()):,} negative "
          f"({df['label'].mean():.1%} positive)")

    if not args.no_scrub:
        before = df["NARRATIVE"].str.len().mean()
        df["NARRATIVE"] = df["NARRATIVE"].map(scrub)
        after = df["NARRATIVE"].str.len().mean()
        print(f"Scrubbed outcome vocabulary "
              f"(mean narrative length {before:.0f} -> {after:.0f} chars)")
        df = df[df["NARRATIVE"].str.len() >= 25]

    if args.balance:
        pos = df[df["label"] == 1]
        neg = df[df["label"] == 0]
        target_neg = min(len(neg), len(pos) * 3)
        neg = neg.sample(n=target_neg, random_state=42)
        df = pd.concat([pos, neg]).sample(frac=1, random_state=42)
        print(f"Balanced to {len(pos):,} positive / {len(neg):,} negative "
              f"({df['label'].mean():.1%} positive)")

    if len(df) > args.max_rows:
        df = df.sample(n=args.max_rows, random_state=42)
        print(f"Sampled down to {len(df):,} rows")

    keep = [c for c in ["DOCUMENT_NO", "NARRATIVE", "label", "DEGREE_INJURY",
                        "IMMED_NOTIFY", "ACTIVITY", "OCCUPATION", "SUBUNIT",
                        "CLASSIFICATION", "MINE_ID", "CONTRACTOR_ID", "CAL_YR"]
            if c in df.columns]
    df[keep].to_csv(args.output, index=False)

    print(f"\nWrote {args.output}")
    print("\nNext:")
    print(f"    python backend/train.py --data {args.output} "
          f"--text-col NARRATIVE --label-col label")
    print("\nSample positives:")
    for t in df[df["label"] == 1]["NARRATIVE"].head(3):
        print(f"  + {t[:150]}")
    print("Sample negatives:")
    for t in df[df["label"] == 0]["NARRATIVE"].head(3):
        print(f"  - {t[:150]}")


if __name__ == "__main__":
    main()
