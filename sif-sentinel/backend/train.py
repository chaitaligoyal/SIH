"""
Train and evaluate the SIF classifier.

    python backend/train.py --data data/labelled.csv

Expects a CSV with a text column and a binary `label` column
(1 = SIF-potential, 0 = not).

This script exists because the submission currently reports no accuracy, no
F1, no baseline and no test split. For an AI/NLP problem statement that is the
single largest scoring gap, and it is fixable in an afternoon once a few
hundred reports are labelled.

Where to get labels without weeks of annotation:

  * OSHA SIR records are severity-labelled by construction — every entry is a
    hospitalisation, amputation or loss of eye, so the outcome was serious.
    Whether the *precursor* was a SIF precursor still needs a human read, but
    the prior is strong.
  * Route the rule baseline over your unlabelled corpus, then have an HSE
    officer adjudicate only the 200 reports nearest the decision boundary.
    That buys far more signal per minute of expert time than labelling at
    random.
  * Corrections submitted through /feedback accumulate in backend/feedback.csv
    and can be concatenated here.

Reported metrics are PR-AUC and precision/recall across candidate thresholds,
not accuracy. SIF precursors are roughly 20-25% of reports, so accuracy is
dominated by the majority class and will read high while the model is useless.
"""

from __future__ import annotations

import argparse
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
MODEL_PATH = os.path.join(MODEL_DIR, "sif_clf.joblib")

TEXT_HINTS = ["narrative", "description", "text", "summary"]


def build_pipeline(n_min_class: int) -> Pipeline:
    """
    Calibration is what makes the 0.40 / 0.70 triage thresholds mean something.
    The original code called its output "calibrated probabilities" while
    producing a hand-tuned linear ramp.

    CalibratedClassifierCV needs at least `cv` examples of the minority class,
    so the fold count adapts and calibration is skipped outright on tiny sets
    rather than crashing — you will hit this the first time you test with a
    small hand-labelled file.
    """
    vectoriser = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=1 if n_min_class < 50 else 2,
        max_features=30000,
        sublinear_tf=True,
        strip_accents="unicode",
    )
    base = LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0)

    if n_min_class < 10:
        print(f"Only {n_min_class} examples in the smaller class — skipping "
              f"probability calibration. Scores will be uncalibrated, so the "
              f"0.40/0.70 thresholds are not yet meaningful. Get to a few "
              f"hundred labels per class before quoting any number.")
        clf = base
    else:
        folds = min(5, n_min_class // 2)
        method = "isotonic" if n_min_class >= 100 else "sigmoid"
        print(f"Calibrating with {folds}-fold {method} scaling.")
        clf = CalibratedClassifierCV(base, method=method, cv=folds)

    return Pipeline([("tfidf", vectoriser), ("clf", clf)])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="CSV with text + label columns")
    ap.add_argument("--text-col", default=None)
    ap.add_argument("--label-col", default="label")
    ap.add_argument("--test-size", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = pd.read_csv(args.data, on_bad_lines="skip")

    text_col = args.text_col
    if text_col is None:
        for hint in TEXT_HINTS:
            match = [c for c in df.columns if hint in c.lower()]
            if match:
                text_col = match[0]
                break
    if text_col is None or args.label_col not in df.columns:
        raise SystemExit(
            f"Could not find text/label columns. Columns present: {list(df.columns)}"
        )

    df = df.dropna(subset=[text_col, args.label_col])

    # Deduplicate before splitting. Operators copy-paste narratives, so the
    # same text appears many times across the file. If a duplicate lands in
    # both train and test, the model gets to memorise the answer and your
    # reported score is fiction. This one line is the difference between an
    # honest number and a meaningless one.
    before = len(df)
    df = df.drop_duplicates(subset=[text_col])
    if before != len(df):
        print(f"Removed {before - len(df):,} duplicate narratives "
              f"({(before - len(df)) / before:.1%}) to prevent train/test leakage.")

    X = df[text_col].astype(str).values
    y = df[args.label_col].astype(int).values

    if len(np.unique(y)) < 2:
        raise SystemExit("Need both classes present to train.")

    print(f"Loaded {len(X)} labelled reports "
          f"({y.sum()} positive, {(1 - y).sum()} negative, "
          f"{y.mean():.1%} positive rate).")

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=args.test_size, stratify=y, random_state=args.seed
    )

    pipe = build_pipeline(int(min(y.sum(), (1 - y).sum())))
    pipe.fit(X_tr, y_tr)
    probs = pipe.predict_proba(X_te)[:, 1]

    pr_auc = average_precision_score(y_te, probs)

    print("\n" + "=" * 64)
    print("HELD-OUT EVALUATION")
    print("=" * 64)
    print(f"PR-AUC (average precision) : {pr_auc:.4f}")
    print(f"ROC-AUC                    : {roc_auc_score(y_te, probs):.4f}")
    print(f"Positive rate in test set  : {y_te.mean():.1%}  <- accuracy floor")

    if pr_auc > 0.98:
        print("\n  !! PR-AUC above 0.98. On free-text safety narratives this")
        print("  !! almost always means leakage, not a good model. Check:")
        print("  !!   1. Did you scrub outcome vocabulary? If the text still")
        print("  !!      says 'fatally injured', you built a word detector.")
        print("  !!   2. Are narratives near-duplicates rather than exact ones?")
        print("  !!      Exact dupes are removed above; near-dupes are not.")
        print("  !!   3. Is your label derivable from something still in the text?")
        print("  !! Do not put this number on a slide until you have ruled")
        print("  !! all three out. A judge who asks 'why 0.99?' and gets no")
        print("  !! answer has found the weak point of your submission.")

    print("\nThreshold sweep — pick the threshold, then quote its numbers:")
    print(f"{'thresh':>7} {'precision':>10} {'recall':>8} {'F1':>7} {'flagged':>8}")
    for t in [0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]:
        pred = (probs >= t).astype(int)
        if pred.sum() == 0:
            print(f"{t:>7.2f} {'—':>10} {'—':>8} {'—':>7} {0:>8}")
            continue
        p, r, f, _ = precision_recall_fscore_support(
            y_te, pred, average="binary", zero_division=0
        )
        print(f"{t:>7.2f} {p:>10.3f} {r:>8.3f} {f:>7.3f} {pred.sum():>8}")

    print("\nFor SIF triage, choose the threshold that holds recall at or above")
    print("~0.95. A missed fatal precursor is unrecoverable; a false positive")
    print("costs an HSE officer twenty minutes.\n")

    pred_default = (probs >= 0.40).astype(int)
    print(classification_report(y_te, pred_default,
                                target_names=["Non-SIF", "SIF-potential"],
                                zero_division=0))
    print("Confusion matrix at 0.40 (rows = true, cols = predicted):")
    print(confusion_matrix(y_te, pred_default))

    os.makedirs(MODEL_DIR, exist_ok=True)
    pipe.fit(X, y)  # refit on everything before shipping
    joblib.dump(pipe, MODEL_PATH)
    print(f"\nSaved model to {MODEL_PATH}")
    print("The API will pick it up automatically on next start.")


if __name__ == "__main__":
    main()
