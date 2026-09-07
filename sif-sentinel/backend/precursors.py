"""
Precursor extraction: activity, location, equipment, barrier.

The original entity_ruler had nine patterns — "rig" + digit, "deck",
"routine maintenance", "tripping pipe", "main compressor valve", "metal pipe
fitting", "loto", "bypassed", "high-pressure release". Every one of them
appears verbatim in the three hardcoded demo narratives in the Streamlit app.
On any report a judge types themselves, that extractor returns nothing.

This replaces them with domain gazetteers that cover upstream oil and gas
operations generally. It is still lexical, and it is still the weakest part of
the pipeline — the right long-term answer is a few hundred hand-annotated OIL
reports and a fine-tuned spaCy NER model. But it degrades gracefully on unseen
text instead of returning an empty list.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List

ACTIVITIES = {
    "drilling": [r"drill(ing)?\b", r"\bspud", r"\bmud pump", r"\btop ?drive"],
    "tripping pipe": [r"\btripping\b", r"\btrip (in|out)\b", r"\bdrill (collar|pipe|string)"],
    "well intervention": [r"\bwireline\b", r"\bcoiled tubing\b", r"\bslickline\b",
                          r"\bworkover\b", r"\bwell intervention\b", r"\bperforat"],
    "cementing / stimulation": [r"\bcementing\b", r"\bacidis|acidiz", r"\bfrac(king|turing)?\b"],
    "maintenance": [r"\bmaintenance\b", r"\bservicing\b", r"\brepair(ing|s)?\b",
                    r"\boverhaul\b", r"\bshutdown\b", r"\bturnaround\b"],
    "lifting / crane operations": [r"\blift(ing)?\b", r"\bcrane\b", r"\bhoist(ing)?\b",
                                   r"\brigging\b", r"\bslinging\b"],
    "hot work": [r"\bhot work\b", r"\bwelding\b", r"\bgrinding\b", r"\bcutting\b"],
    "excavation": [r"\bexcavat", r"\btrench(ing)?\b", r"\bdigging\b", r"\bbackfill"],
    "pigging": [r"\bpigg(ing|er)\b", r"\bpig (launcher|receiver)\b"],
    "electrical work": [r"\belectrical work\b", r"\bswitchgear\b", r"\bsubstation\b",
                        r"\bcable pulling\b", r"\bmegger\b"],
    "confined space entry": [r"\bconfined space entry\b", r"\bvessel entry\b", r"\btank entry\b"],
    "transport / driving": [r"\bdriving\b", r"\bhaul(age|ing)?\b", r"\btransport(ing)?\b",
                            r"\breversing\b", r"\bconvoy\b"],
    "gas / chemical handling": [r"\bpurging\b", r"\bventing\b", r"\bnitrogen\b",
                                r"\bchemical (injection|handling|transfer)\b"],
    "inspection": [r"\binspect(ion|ing)?\b", r"\bsurvey\b", r"\bndt\b", r"\bpatrolling\b"],
}

EQUIPMENT = {
    "valve": [r"\b(block|gate|relief|check|choke|control|mainline|discharge)? ?valve\b"],
    "compressor": [r"\bcompressor\b"],
    "pump": [r"\b(mud |centrifugal |booster )?pump\b"],
    "pipeline / flowline": [r"\bpipe ?line\b", r"\bflow ?line\b", r"\bmainline\b", r"\briser\b"],
    "BOP / wellhead": [r"\bbop\b", r"\bblow ?out preventer\b", r"\bwell ?head\b",
                       r"\bchristmas tree\b", r"\bannular\b"],
    "crane / hoist": [r"\bcrane\b", r"\bhoist\b", r"\bwinch\b", r"\bderrick\b"],
    "sling / cable": [r"\bsling\b", r"\bshackle\b", r"\bwire rope\b", r"\bcable\b", r"\btensioner\b"],
    "conveyor": [r"\bconveyor\b", r"\bbelt drive\b"],
    "scaffold / platform": [r"\bscaffold", r"\bplatform\b", r"\bgrating\b", r"\bladder\b"],
    "vessel / tank": [r"\bvessel\b", r"\btank\b", r"\bseparator\b", r"\bsump\b", r"\bknockout drum\b"],
    "electrical panel": [r"\bswitchgear\b", r"\bpanel\b", r"\bbreaker\b", r"\btransformer\b",
                         r"\bsubstation\b"],
    "vehicle / mobile plant": [r"\bhaul truck\b", r"\bforklift\b", r"\bexcavator\b",
                               r"\bmobile plant\b", r"\bloader\b"],
    "gas detector": [r"\bgas detector\b", r"\bdetector\b", r"\bsensor\b", r"\bmonitor\b"],
}

LOCATION_PATTERNS = [
    r"\brig\s*(?:no\.?\s*)?\d+\b",
    r"\bwell\s*(?:no\.?\s*)?[A-Z]{0,3}[-\s]?\d+\b",
    r"\bpump station\s*\d+\b",
    r"\b(?:drill|rig|main|upper|lower|cellar|monkey)\s*(?:floor|deck|board)\b",
    r"\bplatform\s*[A-Z0-9-]+\b",
    r"\b(?:station|block|unit|area|zone|bay|shed|yard)\s*[A-Z]?-?\d+\b",
    r"\bcompressor (?:station|house)\b",
    r"\bgas (?:collecting|compressor) station\b",
    r"\bwell ?pad\b",
    r"\bmoonpool\b", r"\bwellbay\b", r"\bshale shakers?\b", r"\bmud pit\b",
]

BARRIER_PATTERNS = {
    "energy isolation / LOTO": [r"\bloto\b", r"\block ?out\b", r"\btag ?out\b",
                                r"\bisolat(ion|ed|e)\b", r"\bde-?energis|de-?energiz"],
    "permit to work": [r"\bpermit\b", r"\bptw\b", r"\bwork authoris|work authoriz"],
    "gas testing / detection": [r"\bgas test\b", r"\bdetector\b", r"\bsniffer\b", r"\balarm\b"],
    "physical guarding / barricade": [r"\bguard\b", r"\bbarricade\b", r"\bfenc(e|ing)\b",
                                      r"\bexclusion zone\b", r"\bcordon"],
    "fall protection": [r"\bharness\b", r"\bfall arrest\b", r"\blanyard\b", r"\bguard ?rail\b"],
    "banksman / spotter": [r"\bbanksman\b", r"\bspotter\b", r"\bsignall?er\b", r"\bstandby (man|person)\b"],
    "interlock / trip system": [r"\binterlock\b", r"\btrip system\b", r"\besd\b",
                                r"\bemergency shut ?down\b"],
    "PPE": [r"\bppe\b", r"\bhelmet\b", r"\bgoggles\b", r"\bface shield\b", r"\bgloves\b"],
}


def _find(gazetteer: Dict[str, List[str]], text: str) -> List[str]:
    found = []
    for label, patterns in gazetteer.items():
        if any(re.search(p, text, flags=re.IGNORECASE) for p in patterns):
            found.append(label)
    return found


def infer_site(text: str) -> str:
    """Best-effort site identifier, for the /aggregate rollup."""
    for pattern in LOCATION_PATTERNS[:4]:
        m = re.search(pattern, text, flags=re.IGNORECASE)
        if m:
            return m.group(0).strip().title()
    return "Unspecified"


def extract(text: str) -> Dict[str, Any]:
    locations = []
    for pattern in LOCATION_PATTERNS:
        for m in re.finditer(pattern, text, flags=re.IGNORECASE):
            token = m.group(0).strip()
            if token.lower() not in {l.lower() for l in locations}:
                locations.append(token)

    return {
        "activities": _find(ACTIVITIES, text),
        "equipment": _find(EQUIPMENT, text),
        "locations": locations[:5],
        "barriers_referenced": _find(BARRIER_PATTERNS, text),
    }
