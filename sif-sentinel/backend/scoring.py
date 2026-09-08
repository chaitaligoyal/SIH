"""
SIF-potential scoring.

Two modes, selected automatically at import time:

  1. TRAINED   - if backend/models/sif_clf.joblib exists, a calibrated
                 TF-IDF + LogisticRegression classifier is used. Train it with
                 `python backend/train.py --data <labelled.csv>`.

  2. BASELINE  - otherwise, a transparent rule baseline built on the standard
                 SIF-precursor definition (DEKRA Martin & Black 2015; EEI SIF
                 Precursor model). This is NOT a machine-learned model and is
                 not described as one anywhere in the UI. It exists so the
                 pipeline is runnable before labelled data is available.

The baseline is a three-part conjunction, not a keyword bag:

    SIF potential = high-energy source present
                    AND a direct control was absent / failed / bypassed
                    AND a person was (or plausibly could have been) exposed

That structure is what makes the score explainable by construction: the API
returns which energy source fired, which barrier failed and what the exposure
evidence was, rather than a post-hoc token attribution over a model that has
no internal structure to attribute to.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

import numpy as np

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "sif_clf.joblib")

# Operating points, read off the held-out threshold sweep in train.py.
# Named here so the API, the triage bands and the dashboard gauge cannot
# drift apart -- previously the API flagged at a hardcoded 0.40 while
# triage_action() banded at 0.08, so the counts on screen disagreed with
# the decisions beside them.
T_REVIEW = 0.08      # recall 0.949, precision 0.517
T_ESCALATE = 0.30    # recall 0.839, precision 0.734

# ---------------------------------------------------------------------------
# High-energy sources.
# The SIF literature defines "high energy" as roughly >1500 ft-lbs, or any
# release of gravitational, mechanical, pressure, electrical, thermal,
# chemical or motion energy capable of causing a life-altering injury.
# ---------------------------------------------------------------------------

ENERGY_SOURCES: Dict[str, List[str]] = {
    "gravity / dropped object": [
        r"\bdropp?(ed|ing)\b", r"\bfell\b", r"\bfalling\b", r"\bsuspended load\b",
        r"\boverhead\b", r"\bhoist(ed|ing)?\b", r"\bsling\b", r"\bwinch\b",
        r"\bcrane\b", r"\brigging\b", r"\bderrick\b", r"\bcollapse[d]?\b",
        r"\bslid (down|off|out)\b", r"\bdislodged\b", r"\bdetach(ed|ing)?\b",
        r"\bswung\b", r"\bswing(ing)?\b", r"\btoppl(e|ed|ing)\b",
    ],
    "pressure / stored energy": [
        r"\bpressur(e|ised|ized)\b", r"\bsurge\b", r"\bburst\b", r"\brupture[d]?\b",
        r"\bblow ?out\b", r"\bbop\b", r"\bkick\b", r"\bpurg(e|ing)\b",
        r"\brelief valve\b", r"\bdepressuris|depressuriz", r"\bhydraulic\b",
        r"\bpneumatic\b", r"\bstored energy\b",
        r"\bgas (release|leak|escap\w*|came out|come out|coming out|blow)",
        r"\b(release|leak|escape) of (gas|steam|fluid|hydrocarbon)",
    ],
    "mechanical / rotating equipment": [
        r"\brotating\b", r"\bconveyor\b", r"\bdrive motor\b", r"\bpinch point\b",
        r"\bentangle", r"\bcaught between\b", r"\bcrush(ed|ing)?\b",
        r"\bnip point\b", r"\btop ?drive\b", r"\bdraw ?works\b",
    ],
    "electrical": [
        r"\barc flash\b", r"\benergis|energiz", r"\blive (circuit|conductor|panel)\b",
        r"\bsubstation\b", r"\bswitchgear\b", r"\bhigh voltage\b", r"\b(11|33)\s?kv\b",
        r"\bshock\b", r"\belectrocut",
    ],
    "thermal / fire": [
        r"\bhot work\b", r"\bwelding\b", r"\bgrinding\b", r"\bnaked flame\b",
        r"\bignition\b", r"\bfire\b", r"\bexplosion\b", r"\bflash fire\b",
        r"\bburn(s|ed|ing)?\b", r"\bsteam\b",
    ],
    "chemical / toxic": [
        r"\bh2s\b", r"\bhydrogen sulphide|hydrogen sulfide\b", r"\btoxic\b",
        r"\bhydrocarbon\b", r"\bcondensate\b", r"\bsour gas\b", r"\bcorrosive\b",
        r"\bacid\b", r"\basphyxiat", r"\boxygen deficien",
    ],
    "motion / vehicle": [
        r"\bhaul truck\b", r"\bvehicle\b", r"\bcollision\b", r"\breversing\b",
        r"\bmobile plant\b", r"\bforklift\b", r"\bruna?way\b", r"\bbrake[s]? fail",
    ],
    "height": [
        r"\bworking at height\b", r"\bscaffold", r"\bplatform edge\b",
        r"\bfall arrest\b", r"\bharness\b", r"\bopen (hatch|grating|hole)\b",
        r"\bmezzanine\b", r"\b\d+\s?(m|meter|metre|ft|foot|feet)\s+(fall|drop|above)\b",
    ],
    "confined space": [
        r"\bconfined space\b", r"\bvessel entry\b", r"\btank entry\b",
        r"\bsump\b", r"\bexcavation\b", r"\btrench\b", r"\bmanhole\b",
    ],
}

# ---------------------------------------------------------------------------
# Barrier / direct-control status.
# ---------------------------------------------------------------------------

BARRIER_FAILED = [
    r"\bbypass(ed|ing)?\b", r"\boverrid(e|den|ing)\b", r"\bdefeat(ed)?\b",
    # Allows modifiers: "without an active banksman", "without a valid permit".
    r"\bwithout (a |an |the )?(\w+\s){0,2}?(permit|ptw|authoris|authoriz|isolation|"
    r"loto|lock(ing|ed)?[ -]?out|banksman|spotter|standby|barricade|gas test|"
    r"supervision|harness|guard|watch)\b",
    # Simplified / code-mixed phrasing common in frontline reports.
    r"\bno (lockout|loto|permit|isolation|guard|barricade|tag|test|banksman)\b",
    r"\bnot (isolated|locked out|tagged|barricaded|de-?energis|de-?energiz)\b",
    r"\bfail(ed|ure) to (isolate|lock ?out|tag|test|barricade)\b",
    r"\bno (banksman|spotter|permit|barricade|gas test|standby)\b",
    r"\bexpired permit\b", r"\binterlock (disabled|bypassed|jumped)\b",
    r"\bpermit (not|was not) (obtained|closed|valid)\b",
    r"\bguard (removed|missing)\b", r"\bmissing (guard|cover|barrier)\b",
    r"\bloto (bypass|not|was not)\b",
    r"\bdetector failed\b", r"\bfailed to alarm\b", r"\balarm (disabled|inhibited)\b",
    r"\bcorrosion\b", r"\bfrayed\b", r"\bworn\b", r"\bdefective\b",
    r"\buncertified\b", r"\bunauthoris|unauthoriz",
]

BARRIER_HELD = [
    r"\bharness prevented\b", r"\bproper ppe\b", r"\bwearing (proper|correct|full) ppe\b",
    r"\bpermit (was )?in place\b", r"\bisolation (was )?verified\b",
    r"\bbarricade[sd]? (in place|installed)\b", r"\bstopped the job\b",
    r"\bstop work\b", r"\bgas test (completed|clear)\b",
    r"\bran(away)? ramp\b", r"\bsafely (stopped|shut down|isolated)\b",
]

# ---------------------------------------------------------------------------
# Exposure: was a person in the line of fire?
# ---------------------------------------------------------------------------

EXPOSURE_PRESENT = [
    r"\bnarrowly miss(ed|ing)\b", r"\bnear miss\b", r"\bstruck (the |a )?(worker|employee|operator|technician|person)\b",
    r"\bstruck by\b", r"\bhit the\b", r"\bline of fire\b",
    r"\bworker (was|were)\b", r"\bemployee (was|were)\b", r"\bpersonnel (were|was) (present|working|standing)\b",
    r"\bin the (vicinity|area|path)\b", r"\bstanding (under|beneath|beside)\b",
    r"\boperator (was|were)\b", r"\bcrew (was|were)\b",
    r"\binjur(y|ed|ies)\b", r"\bamputat", r"\bfracture", r"\blaceration\b",
    r"\bhospitalis|hospitaliz", r"\bmedical treatment\b",
]

EXPOSURE_ABSENT = [
    r"\bno personnel (in|present|nearby|were)\b", r"\barea (was )?(clear|evacuated|cordoned)\b",
    r"\bno one (was )?(present|nearby|in the area|injured)\b",
    r"\bunmanned\b", r"\bduring (a )?shutdown\b",
]

# A realised life-altering injury is proof that high energy was released,
# regardless of whether the narrative names the energy source. Without this,
# terse reports that only state the outcome ("thumb partially amputated")
# score as low-energy, which is the wrong answer in the worst possible case.
SEVERE_OUTCOME = [
    r"\bamputat", r"\bfractur", r"\bdegloving\b", r"\bcrushed (hand|arm|leg|foot|limb)\b",
    r"\bhospitalis|hospitaliz", r"\bloss of (an )?eye\b", r"\bsevere burn",
    r"\bunconscious\b", r"\bfatal|fatalit", r"\blife-?(threatening|altering)\b",
    r"\bmajor laceration\b", r"\binternal injur",
]

LOW_ENERGY_NOISE = [
    r"\bhousekeeping\b", r"\bminor spill\b", r"\btrip hazard\b", r"\bwalkway\b",
    r"\bcanteen\b", r"\blitter\b", r"\bsignage (missing|faded)\b",
    r"\bpaper ?work\b", r"\bfiling\b",
]


# Negation cues. A match preceded by one of these inside a short left-window is
# discarded — this is what stops "No personnel injured" being read as exposure
# evidence, which is a common failure mode of flat keyword scorers.
NEGATORS = re.compile(
    r"\b(no|not|never|without|avoid(ed|ing)?|prevent(ed|ing)?|"
    r"nobody|none|free of|clear of)\b[^.;]{0,40}$",
    flags=re.IGNORECASE,
)

NEGATION_WINDOW = 45


def _matches(patterns: List[str], text: str, negation_aware: bool = False) -> List[str]:
    """Return the surface forms of every pattern that fires in `text`."""
    hits: List[str] = []
    for p in patterns:
        for m in re.finditer(p, text, flags=re.IGNORECASE):
            if negation_aware:
                left = text[max(0, m.start() - NEGATION_WINDOW): m.start()]
                if NEGATORS.search(left):
                    continue
            hits.append(m.group(0).strip())
            break
    return hits


@dataclass
class SIFAssessment:
    """Structured, human-readable justification for a score."""
    sif_probability: float
    method: str
    energy_sources: Dict[str, List[str]] = field(default_factory=dict)
    barrier_failures: List[str] = field(default_factory=list)
    barrier_intact_evidence: List[str] = field(default_factory=list)
    exposure_evidence: List[str] = field(default_factory=list)
    exposure_ruled_out: List[str] = field(default_factory=list)
    rationale: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class BaselineSIFScorer:
    """
    Transparent energy/barrier/exposure baseline.

    Weighting is deliberate and stated rather than tuned to a demo narrative:
    energy is necessary but not sufficient, barrier failure is the strongest
    single multiplier, and exposure evidence pushes a report over the
    escalation threshold. These weights are a starting prior to be replaced by
    a trained model as soon as labelled reports exist.
    """

    W_ENERGY = 0.30          # any high-energy source present
    W_ENERGY_MULTI = 0.08    # per additional distinct energy category, capped
    W_BARRIER = 0.30         # a direct control was absent, bypassed or failed
    W_EXPOSURE = 0.22        # a person was in, or plausibly in, the line of fire
    BASE = 0.06

    def assess(self, text: str) -> SIFAssessment:
        t = text or ""

        energy: Dict[str, List[str]] = {}
        for source, patterns in ENERGY_SOURCES.items():
            hits = _matches(patterns, t)
            if hits:
                energy[source] = hits

        severe = _matches(SEVERE_OUTCOME, t, negation_aware=True)
        if severe and not energy:
            energy["realised serious injury (energy source not stated)"] = severe

        failed = _matches(BARRIER_FAILED, t)
        held = _matches(BARRIER_HELD, t)
        exposed = _matches(EXPOSURE_PRESENT, t, negation_aware=True)
        not_exposed = _matches(EXPOSURE_ABSENT, t)
        noise = _matches(LOW_ENERGY_NOISE, t)

        score = self.BASE

        if energy:
            score += self.W_ENERGY
            score += min((len(energy) - 1) * self.W_ENERGY_MULTI, 0.16)

        if failed:
            # Barrier failure only counts toward SIF potential if there was
            # meaningful energy for the barrier to be controlling.
            score += self.W_BARRIER if energy else self.W_BARRIER * 0.25
        elif held and energy:
            score -= 0.10

        if exposed and energy:
            score += self.W_EXPOSURE
        elif not_exposed:
            score -= 0.08

        # A report with no high-energy source and explicit low-energy framing
        # should land clearly in the auto-file band.
        if not energy and noise:
            score = min(score, 0.10)

        # A realised serious injury is never auto-filed, whatever else the
        # narrative does or does not say.
        if severe:
            score = max(score, 0.75)

        score = float(np.clip(score, 0.02, 0.97))

        parts = []
        if energy:
            parts.append(f"high-energy source(s): {', '.join(energy.keys())}")
        else:
            parts.append("no high-energy source identified")
        if failed:
            parts.append(f"control failure indicated ({len(failed)} cue(s))")
        elif held:
            parts.append("control appears to have held")
        else:
            parts.append("control status not stated")
        if exposed:
            parts.append("person exposed / in line of fire")
        elif not_exposed:
            parts.append("exposure explicitly ruled out")
        else:
            parts.append("exposure not stated")

        return SIFAssessment(
            sif_probability=score,
            method="baseline-energy-barrier-exposure",
            energy_sources=energy,
            barrier_failures=failed,
            barrier_intact_evidence=held,
            exposure_evidence=exposed,
            exposure_ruled_out=not_exposed,
            rationale="; ".join(parts),
        )


class TrainedSIFScorer:
    """Wraps a calibrated sklearn pipeline saved by train.py."""

    def __init__(self, path: str):
        import joblib
        self.pipeline = joblib.load(path)
        self.baseline = BaselineSIFScorer()

    def assess(self, text: str) -> SIFAssessment:
        prob = float(self.pipeline.predict_proba([text])[0][1])
        # Structural evidence is still extracted, because a probability with no
        # stated reason is not actionable for an HSE officer.
        struct = self.baseline.assess(text)
        struct.sif_probability = prob
        struct.method = "trained-tfidf-logreg-calibrated"
        return struct


def load_scorer():
    if os.path.exists(MODEL_PATH):
        try:
            print(f"[scoring] Loaded trained classifier from {MODEL_PATH}")
            return TrainedSIFScorer(MODEL_PATH)
        except Exception as exc:  # noqa: BLE001
            print(f"[scoring] Failed to load trained model ({exc}); using baseline.")
    print("[scoring] No trained model found. Using transparent rule baseline.")
    return BaselineSIFScorer()


SCORER = load_scorer()


def score_texts(texts: List[str]) -> np.ndarray:
    """Batch scorer. Returns [[P(non-SIF), P(SIF)], ...]."""
    out = []
    for t in texts:
        p = SCORER.assess(t).sif_probability
        out.append([1.0 - p, p])
    return np.array(out)


def triage_action(prob: float) -> str:
    """
    Thresholds are set for RECALL, not accuracy.

    Missing a genuine fatal precursor costs a life; a false positive costs an
    HSE officer twenty minutes. The 0.40 boundary is therefore deliberately
    low. Re-fit these on a validation set once labelled data exists — see
    train.py, which reports precision/recall at each candidate threshold.
    """
    if prob >= T_ESCALATE:
        return "ESCALATE_IMMEDIATE_SUPERVISOR"
    if prob >= T_REVIEW:
        return "HUMAN_REVIEW_SECONDARY_QUEUE"
    return "AUTO_FILE_LOW_RISK"
