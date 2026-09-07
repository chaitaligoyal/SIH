# SIF-Sentinel

Detects Serious Injury & Fatality (SIF) precursors in free-text UA/UC observations,
near-miss and incident reports. Built for Smart India Hackathon 2026, Oil India Limited
problem statement on AI/NLP triage of HSSE reports.

The premise, from the problem statement: low-severity incidents and fatalities do not
share the same causes. Non-fatal US accidents fell 51% over fifteen years while
fatalities fell only 25.5%. Triaging by severity-as-reported therefore misses the
~20–25% of reports that carry genuine fatal potential, and OIL currently triages
manually on a monthly or quarterly cycle.

---

## What it does

1. **Classifies** each report as SIF-potential or not, with a stated reason.
2. **Tags** it to the relevant IOGP Life-Saving Rule (multi-label).
3. **Ranks** sites, activities, energy sources and barrier failures by SIF-precursor
   *density*, so HSE intervention goes where fatal potential is highest.

---

## How the classifier works

The score is not a keyword count. It implements the standard SIF-precursor definition
(DEKRA Martin & Black 2015; EEI SIF Precursor model) as an explicit three-part
conjunction:

```
SIF potential  =  a high-energy source was present
              AND a direct control was absent, bypassed or failed
              AND a person was, or plausibly could have been, in the line of fire
```

Each term is extracted separately and returned in the API response. That means the
explanation is structural rather than post-hoc: instead of a token-attribution plot,
an HSE officer sees *"high-energy source: pressurised gas; control failure: LOTO
bypassed; exposure: worker narrowly missed."* Every HSE professional on a judging panel
will recognise this framework immediately.

Two scoring modes, selected automatically:

| Mode | When | What it is |
|---|---|---|
| `baseline-energy-barrier-exposure` | no trained model on disk | transparent rule logic over the three terms above. Runnable today, labelled as a baseline everywhere it appears in the UI. |
| `trained-tfidf-logreg-calibrated` | `backend/models/sif_clf.joblib` exists | calibrated TF-IDF + LogisticRegression trained on your labelled reports |

The baseline is **not** described as a machine-learned model anywhere. Do not describe
it as one during judging either — the first question after "how does it work" is
usually "show me the training run."

---

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Backend
cd backend && uvicorn api:app --reload --port 8000

# Frontend, in a second terminal
streamlit run frontend/app.py
```

If `hdbscan` fails to build, install a compiler first
(`sudo apt install build-essential python3-dev`) or drop it — the app detects its
absence and disables clustering rather than crashing.

Set `SIF_API_URL` if the backend is not on `localhost:8000`.

---

## Getting labelled data and training

No manual annotation needed. MSHA's open accident file carries the labels already.

```bash
python backend/fetch_real_data.py     # ~250k real narratives, one download
python backend/build_labels.py        # derives SIF labels from structured fields
python backend/train.py --data data/labelled.csv --text-col NARRATIVE --label-col label
```

### Where the labels come from

MSHA Form 7000-1 records two structured fields alongside every free-text narrative:

- `DEGREE_INJURY` — fatality, permanent disability, days away, first aid, accident only
- `IMMED_NOTIFY` — whether the event was one of the twelve types requiring immediate
  notification: death, serious injury, entrapment, inundation, gas or dust ignition,
  mine fire, explosives, roof fall, outburst, hoisting

The SIF literature separates *actual* SIF (someone was killed or permanently disabled)
from *potential* SIF (a high-energy event where the outcome happened to be mild — the
precursor). Those two fields give you one arm each:

| | Rule |
|---|---|
| **Positive** | fatality or permanent disability, **or** a high-energy immediate-notification event regardless of outcome |
| **Negative** | first aid / no days lost / no restrictions, **and** nothing immediately reportable |
| **Dropped** | days-away variants, occupational illness, natural causes, non-employee injuries — the genuine gray zone |

The "accident only + high-energy notification" rows are the most valuable examples in
the whole set: a high-energy event where nobody got hurt is exactly a SIF precursor.

`build_labels.py` prints the drop rate. Quote it — a judge will ask what you did with
the ambiguous middle.

### Two leakage traps, both handled

1. **Outcome vocabulary.** A fatality narrative usually says "fatally injured" or
   "pronounced dead". Train on that and you build a detector for the word "fatally",
   which is useless on OIL's UA/UC reports where nobody has been hurt yet.
   `build_labels.py` strips outcome vocabulary by default.
2. **Duplicate narratives.** Operators copy-paste. If the same text lands in both
   train and test, your reported score is fiction. `train.py` deduplicates before
   splitting and prints how many it removed.

`train.py` warns if PR-AUC exceeds 0.98, because on free-text safety narratives that
is nearly always leakage rather than a good model. Do not put a suspiciously perfect
number on a slide.

### Reading the metrics

Metrics are PR-AUC and precision/recall, **not accuracy**. SIF precursors are ~20–25%
of reports, so accuracy is dominated by the majority class and will read ~78% while the
model does nothing.

Choose the threshold that holds **recall at or above ~0.95**. A missed fatal precursor
is unrecoverable; a false positive costs an HSE officer twenty minutes. Say this out
loud during judging — it demonstrates domain reasoning more than any architecture
diagram.

Corrections submitted through the dashboard accumulate in `backend/feedback.csv` and
can be concatenated straight into the training file, so the model improves on OIL's own
vocabulary over time.

---

## API

| Endpoint | Purpose |
|---|---|
| `POST /analyze` | one narrative → score, IOGP tags, structural evidence, cluster |
| `POST /analyze_batch` | bulk scoring of an exported report file |
| `GET /aggregate` | site / activity / energy / rule ranking by SIF density |
| `GET /clusters` | recurring precursor patterns in the corpus |
| `POST /feedback` | HSE reviewer correction, queued for retraining |
| `GET /health` | model status, corpus size, session counters |

---

## Known limitations

Stating these is a strength in judging, not a weakness. The failure mode is being
caught not knowing them.

- **The baseline is lexical.** It will miss SIF precursors described in vocabulary it
  does not contain. This is exactly why the trained path exists.
- **Reports written in simplified or code-mixed English score lower.** OIL's frontline
  workforce in Duliajan will not all write in fluent technical English, and a genuine
  energy-isolation failure described as *"worker no lockout do, gas come out fast"*
  is worth as much as one described in perfect prose. Some simplified phrasing is
  handled explicitly; a fair amount is not. This is a real fairness concern and a
  measurable one — take a SIF narrative, rewrite it in broken English, compare scores.
  That test is worth more than any generic bias dashboard.
- **Precursor extraction uses gazetteers, not a trained NER.** It covers upstream oil
  and gas vocabulary generally, but a few hundred hand-annotated OIL reports plus a
  fine-tuned spaCy model would be substantially better.
- **Clustering needs volume.** With fewer than a few hundred historical reports,
  HDBSCAN will not surface a real pattern and the module says so rather than inventing
  one.
- **Scored reports are held in memory.** `/aggregate` resets when the server restarts.
  Wire up Postgres before any real deployment.

---

## Data

Run `python backend/fetch_real_data.py` and `data/` fills with real regulator
narratives. Until then it holds fictional fixtures.

| File | What it is |
|---|---|
| `MSHA_Accidents_real.csv` | **real** — MSHA Form 7000-1 accident narratives, 2000–present, pipe-delimited open-government export, refreshed weekly |
| `labelled.csv` | derived training set built by `build_labels.py` |
| `DEMO_FIXTURE_*.csv` | **fictional**, 5 rows each — written by `generate_demo_fixtures.py` for offline UI work. Every row carries a `provenance` column saying so, and the loader ignores these files entirely once real data is present. |

Sources needing a browser click (the fetch script prints these):

- PHMSA pipeline incidents, with narratives plus Serious/Significant flags —
  <https://www.phmsa.dot.gov/data-and-statistics/pipeline/pipeline-incident-flagged-files>
- BSEE offshore incident statistics —
  <https://www.bsee.gov/stats-facts/offshore-incident-statistics>
- OSHA Severe Injury Reports — <https://www.osha.gov/severeinjury>

**Nothing in this repo presents invented data as real.** The earlier version shipped
three LLM-generated CSVs named `BSEE_Offshore_Incidents.csv`,
`PHMSA_Pipeline_Reports.csv` and `MSHA_Mining_Data.csv` — indistinguishable from
regulator exports — read by a loader commented "REAL-WORLD DATASET INGESTION". If those
three files are still in your `data/` directory, delete them.

---

## Deployment notes for OIL

- Everything runs locally. Voice transcription uses `faster-whisper` on CPU rather
  than posting narratives to a third-party US endpoint. OIL is a PSU under DGMS
  oversight — being air-gap deployable is a selling point, not a constraint.
- Integrate as an API against the existing HSSE platform rather than replacing its UI.
  HSE teams will not adopt a second place to file reports.
- Every model decision should be written to an audit log before this goes anywhere
  near a regulated environment.
