"""
IOGP Life-Saving Rules multi-label tagger.

The original version compared a 60-word narrative against the two-word string
"Hot Work" with a cosine threshold of 0.28. Sentence embeddings of a long
narrative and a two-token label sit in very different regions of the space, so
that comparison is close to noise and a 0.28 floor fires on most rules at once.

Two changes fix it:

  1. Each rule is embedded from a full descriptive paragraph derived from the
     published IOGP Life-Saving Rules, not from its title.
  2. Cosine similarity is combined with lexical anchors. A narrative that
     literally says "confined space" should tag Confined Space regardless of
     what MiniLM thinks; a narrative that only resembles it semantically needs
     a higher similarity to qualify.

Rule embeddings are computed once at import, not on every request.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List

import numpy as np
from sentence_transformers import SentenceTransformer, util

_EMBEDDER: SentenceTransformer | None = None


def get_embedder() -> SentenceTransformer:
    global _EMBEDDER
    if _EMBEDDER is None:
        _EMBEDDER = SentenceTransformer("all-MiniLM-L6-v2")
    return _EMBEDDER


IOGP_RULES: Dict[str, Dict[str, Any]] = {
    "Bypassing Safety Controls": {
        "definition": (
            "Obtain authorisation before overriding, disabling or bypassing any "
            "safety control, interlock, alarm, trip, guard or protective device. "
            "Covers defeated interlocks, inhibited alarms, removed machine guards "
            "and disabled gas or fire detection."
        ),
        "anchors": [r"bypass", r"overrid", r"interlock", r"inhibit", r"defeat",
                    r"guard (removed|missing)", r"alarm disabled", r"jumper"],
    },
    "Confined Space": {
        "definition": (
            "Obtain authorisation before entering a confined space. Covers entry "
            "into tanks, vessels, sumps, pits, excavations, trenches, sewers and "
            "any enclosure with restricted egress or a potentially oxygen-deficient "
            "or toxic atmosphere. Requires gas testing and a standby attendant."
        ),
        "anchors": [r"confined space", r"vessel entry", r"tank entry", r"manhole",
                    r"excavation", r"trench", r"sump", r"oxygen deficien", r"gas test"],
    },
    "Energy Isolation": {
        "definition": (
            "Verify isolation and the absence of stored energy before work begins. "
            "Covers lockout tagout, LOTO, blinding, de-energisation, depressurisation, "
            "draining and the verification of zero energy state on electrical, "
            "hydraulic, pneumatic, pressure, thermal and mechanical sources."
        ),
        "anchors": [r"loto", r"lock ?out", r"tag ?out", r"isolat", r"de-?energis",
                    r"de-?energiz", r"depressuris", r"depressuriz", r"blind(ed|ing)?",
                    r"stored energy", r"block valve"],
    },
    "Driving Safety": {
        "definition": (
            "Follow safe driving rules. Covers seat belts, speed limits, journey "
            "management, fatigue, mobile phone use while driving, and the operation "
            "of light vehicles, haul trucks and mobile plant on site roads."
        ),
        "anchors": [r"driv(e|ing|er)", r"vehicle", r"seat ?belt", r"haul truck",
                    r"speed", r"journey management", r"reversing", r"forklift",
                    r"mobile plant", r"collision"],
    },
    "Hot Work": {
        "definition": (
            "Control flammables and ignition sources before hot work. Covers welding, "
            "cutting, grinding, brazing, open flame, spark-producing tools and any "
            "ignition source introduced into a hydrocarbon or flammable atmosphere."
        ),
        "anchors": [r"hot work", r"weld", r"grind", r"cutting torch", r"naked flame",
                    r"spark", r"ignition source", r"flammable"],
    },
    "Line of Fire": {
        "definition": (
            "Keep yourself and others out of the line of fire. Covers stored energy "
            "release, dropped and falling objects, suspended loads, pressurised lines, "
            "moving or rotating equipment, pinch points, whipping hoses and any "
            "position where a person is exposed to a released hazard."
        ),
        "anchors": [r"line of fire", r"struck by", r"dropped object", r"suspended load",
                    r"narrowly miss", r"pinch point", r"whip", r"flying", r"recoil",
                    r"crush", r"trapped between"],
    },
    "Safe Mechanical Lifting": {
        "definition": (
            "Plan lifting operations and control the area. Covers cranes, hoists, "
            "winches, slings, shackles, rigging, load charts, banksman or signaller "
            "presence, exclusion zones beneath suspended loads and lifting in wind."
        ),
        "anchors": [r"lift(ing)?", r"crane", r"hoist", r"winch", r"sling", r"shackle",
                    r"rigging", r"banksman", r"suspended load", r"load chart"],
    },
    "Work Authorization": {
        "definition": (
            "Work with a valid permit when required. Covers permit to work, job "
            "safety analysis, toolbox talk, method statement, task risk assessment "
            "and the authorisation of non-routine or high-risk activity."
        ),
        "anchors": [r"permit", r"ptw", r"authoris", r"authoriz", r"jsa",
                    r"toolbox talk", r"method statement", r"risk assessment"],
    },
    "Working at Height": {
        "definition": (
            "Protect against a fall when working at height. Covers scaffolding, "
            "ladders, elevated platforms, open gratings, roof work, fall arrest "
            "harnesses, anchor points, guardrails and unprotected edges."
        ),
        "anchors": [r"working at height", r"scaffold", r"ladder", r"harness",
                    r"fall arrest", r"guard ?rail", r"open grating", r"unprotected edge",
                    r"platform edge", r"mezzanine", r"anchor point"],
    },
}

RULE_NAMES: List[str] = list(IOGP_RULES.keys())

# Computed once at import, not per request.
_RULE_EMBEDDINGS = None


def _rule_embeddings():
    global _RULE_EMBEDDINGS
    if _RULE_EMBEDDINGS is None:
        definitions = [IOGP_RULES[name]["definition"] for name in RULE_NAMES]
        _RULE_EMBEDDINGS = get_embedder().encode(
            definitions, convert_to_tensor=True, normalize_embeddings=True
        )
    return _RULE_EMBEDDINGS


# Thresholds. A lexical anchor is strong evidence, so it clears a lower bar.
SEMANTIC_ONLY_THRESHOLD = 0.42
ANCHORED_THRESHOLD = 0.20
MAX_RULES_RETURNED = 3


def tag_iogp_rules(text: str) -> List[Dict[str, Any]]:
    """Return the IOGP rules implicated by a narrative, most confident first."""
    if not text or not text.strip():
        return []

    text_emb = get_embedder().encode(
        text, convert_to_tensor=True, normalize_embeddings=True
    )
    scores = util.cos_sim(text_emb, _rule_embeddings())[0].cpu().numpy()

    matched: List[Dict[str, Any]] = []
    for idx, name in enumerate(RULE_NAMES):
        sim = float(scores[idx])
        anchors_hit = [
            a for a in IOGP_RULES[name]["anchors"]
            if re.search(a, text, flags=re.IGNORECASE)
        ]
        threshold = ANCHORED_THRESHOLD if anchors_hit else SEMANTIC_ONLY_THRESHOLD

        if sim >= threshold:
            # Anchor hits raise confidence above raw cosine, which otherwise
            # understates a narrative that names the hazard outright.
            confidence = min(1.0, sim + 0.12 * len(anchors_hit))
            matched.append({
                "rule": name,
                "confidence": round(confidence, 3),
                "cosine_similarity": round(sim, 3),
                "matched_terms": anchors_hit,
                "evidence": "lexical + semantic" if anchors_hit else "semantic only",
            })

    matched.sort(key=lambda r: r["confidence"], reverse=True)
    return matched[:MAX_RULES_RETURNED]
