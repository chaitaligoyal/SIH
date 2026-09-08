"""
SIF-Sentinel dashboard.

Screens compose primitives from theme.py; this file holds flow and state only.

Every call this file makes -- /health, /aggregate, /clusters, /analyze,
/analyze_batch, /feedback -- uses the same paths, payload shapes and response
keys as before. The redesign is presentation; the API contract is untouched.

Structure of a session:

    Analyse   one narrative, read as a barrier chain
    Batch     a whole export, scored in one request
    Density   the deliverable -- where fatal potential concentrates
    Patterns  recurring precursor clusters across the corpus

The order is deliberate. A judge reads left to right and the argument builds:
here is one report, here are thousands, here is where to send the team.
"""

from __future__ import annotations

import io
import os

import pandas as pd
import requests
import streamlit as st

import theme as T

API = os.getenv("SIF_API_URL", "http://localhost:8000")
TIMEOUT = 180

st.set_page_config(
    page_title="SIF-Sentinel - fatal-potential triage",
    layout="wide",
    page_icon="!",
    initial_sidebar_state="collapsed",
)
T.inject_css()


# --------------------------------------------------------------------------
# Transport
# --------------------------------------------------------------------------

def api_get(path: str, **params):
    try:
        r = requests.get(f"{API}{path}", params=params, timeout=TIMEOUT)
        r.raise_for_status()
        return r.json()
    except requests.RequestException:
        return None


def api_post(path: str, payload: dict):
    try:
        r = requests.post(f"{API}{path}", json=payload, timeout=TIMEOUT)
        if r.status_code != 200:
            st.error(f"The engine rejected this request ({r.status_code}). {r.text[:220]}")
            return None
        return r.json()
    except requests.RequestException:
        st.error(
            f"No response from the scoring engine at {API}. "
            "Start it with `uvicorn api:app --port 8000` from the backend folder, "
            "then reload this page."
        )
        return None


# --------------------------------------------------------------------------
# Masthead and status
# --------------------------------------------------------------------------

T.masthead(
    "SIF-Sentinel",
    "Finds the reports that could have killed someone &mdash; including the ones where nobody got hurt",
)

health = api_get("/health")
agg = api_get("/aggregate")

if health is None:
    T.statusbar([{"label": "Engine", "value": "offline", "state": "degraded",
                  "note": "not reachable"}])
    T.empty_state(
        "The scoring engine is not running",
        "Open a terminal, go to the backend folder, and run "
        "<code>uvicorn api:app --port 8000</code>. First start takes a few minutes "
        "while the language model loads and clustering fits. Reload this page once "
        "it prints &ldquo;Uvicorn running&rdquo;.",
    )
    st.stop()

trained = health.get("trained_model_loaded", False)
total = (agg or {}).get("total_reports", 0)
flagged = (agg or {}).get("sif_flagged", 0)

T.statusbar([
    {"label": "Engine", "value": "trained" if trained else "baseline",
     "state": "live" if trained else "degraded",
     "note": "calibrated classifier" if trained else "rule fallback - run train.py"},
    {"label": "Historical corpus", "value": f"{health.get('corpus_size', 0):,}",
     "note": "narratives available for pattern matching"},
    {"label": "Scored this session", "value": f"{health.get('reports_scored_this_session', 0):,}",
     "note": f"{flagged:,} carry fatal potential" if total else "nothing scored yet"},
    {"label": "Flag rate", "value": f"{(flagged / total * 100):.0f}%" if total else "-",
     "note": "published benchmark is 20-25%" if total else "run a batch to populate"},
    {"label": "Reviewer corrections", "value": f"{health.get('feedback_collected', 0):,}",
     "note": "queued for retraining"},
])

if not trained:
    T.note(
        "Running on the rule baseline, not the trained classifier. Scores are "
        "structural rather than learned, and the 0.08 / 0.30 setpoints are not "
        "calibrated yet. Run <code>python backend/train.py</code> to load the model.",
        warn=True,
    )
    st.write("")

tab_one, tab_batch, tab_density, tab_patterns = st.tabs(
    ["Analyse a report", "Score a batch", "Where the risk is", "Recurring patterns"]
)


# --------------------------------------------------------------------------
# Tab 1 - single report
# --------------------------------------------------------------------------

PRESETS = {
    "Write your own": "",
    "Energy isolation bypassed (Rig 42)":
        "During routine maintenance at Rig 42, the main compressor valve was jammed. "
        "Lockout/tagout energy isolation was bypassed. A sudden high-pressure release "
        "caused a metal pipe fitting to detach and fly across the deck, narrowly "
        "missing the worker.",
    "Suspended load, no banksman (Rig 19)":
        "While tripping pipe at Rig 19, winch cable tension slackened abruptly. The "
        "suspended drill collar shifted 2 metres outside the designated rotary envelope "
        "without an active banksman present.",
    "Control held - fall arrested":
        "Worker slipped on the drill floor during tripping operations; safety harness "
        "prevented a fall into the moonpool.",
    "Low energy - housekeeping":
        "Housekeeping notice: minor water spill observed near the kitchen walkway. "
        "Slippery surface warning sign placed. No personnel injured.",
    "Simplified English, same hazard":
        "worker no lockout do. valve open suddenly. gas come out fast. man standing "
        "near, very close, no injury but big risk.",
}

with tab_one:
    left, right = st.columns([5, 6], gap="large")

    with left:
        T.panel_open("Incident narrative", "free text, any length")
        choice = st.selectbox("Load an example", list(PRESETS.keys()),
                              label_visibility="collapsed")
        text = st.text_area(
            "Narrative", PRESETS[choice], height=210,
            placeholder="Paste a UA/UC observation, near-miss or incident report...",
            label_visibility="collapsed",
        )
        go = st.button("Analyse", type="primary")
        T.panel_close()

        st.caption(
            "Try the last two examples together. The harness case shows a control that "
            "held; the simplified-English case shows the same hazard as the first "
            "example, described by someone writing in a second language."
        )

    if go:
        if not text.strip():
            st.warning("Add a narrative first, then press Analyse.")
        else:
            with st.spinner("Reading the narrative..."):
                res = api_post("/analyze", {"report_id": "UI-001", "text": text})
            if res:
                st.session_state["last"] = res
                st.session_state["last_text"] = text

    data = st.session_state.get("last")

    with right:
        if not data:
            T.empty_state(
                "Nothing analysed yet",
                "Pick an example on the left or paste your own report, then press "
                "Analyse. You will get a triage decision, the barrier chain behind it, "
                "and the IOGP Life-Saving Rules it touches.",
            )
        else:
            prob = data["sif_probability"]
            T.verdict(data["triage_action"], prob)
            st.write("")
            T.panel_open(
                "Fatal potential",
                f"{prob:.0%} &middot; {data['scoring_method'].replace('-', ' ')}",
            )
            T.gauge(prob)
            T.panel_close()

    if data:
        st.write("")
        T.panel_open(
            "Why this reading",
            "a precursor needs high energy, a failed control, and someone exposed",
        )
        T.barrier_chain(data["assessment"])
        st.write("")
        T.chips(data["assessment"])
        T.panel_close()

        c1, c2 = st.columns([1, 1], gap="large")

        with c1:
            T.panel_open("IOGP Life-Saving Rules", "up to three")
            T.rule_rows(data.get("iogp_rules", []))
            T.panel_close()

        with c2:
            p = data.get("precursors", {})
            T.panel_open("Extracted context", data.get("site", "site not stated"))
            any_found = False
            for label, key in (
                ("Activity", "activities"),
                ("Equipment", "equipment"),
                ("Location", "locations"),
                ("Controls named", "barriers_referenced"),
            ):
                if p.get(key):
                    any_found = True
                    st.markdown(
                        f"<div style='padding:7px 0;border-bottom:1px solid #D4D9DF;"
                        f"font-size:13px'><span style='color:#79838F;font-size:11px;"
                        f"font-weight:600;letter-spacing:.04em'>{label}</span><br>"
                        f"{', '.join(p[key])}</div>",
                        unsafe_allow_html=True,
                    )
            if not any_found:
                st.markdown(
                    "<p style='font-size:12.5px;color:#79838F;margin:0'>"
                    "No activity, equipment or location vocabulary recognised. "
                    "Extraction uses domain word lists, so unfamiliar phrasing "
                    "returns nothing.</p>",
                    unsafe_allow_html=True,
                )

            cl = data.get("clustering", {})
            if cl.get("cluster_summary"):
                st.markdown(
                    f"<div style='padding-top:10px;font-size:12.5px;color:#4A5563'>"
                    f"{cl['cluster_summary']}</div>",
                    unsafe_allow_html=True,
                )
            T.panel_close()

        T.panel_open("Reviewer correction", "feeds the next training round")
        fb1, fb2, fb3 = st.columns([1, 1, 3])
        payload = {
            "report_id": data["report_id"],
            "text": st.session_state.get("last_text", ""),
            "predicted_probability": prob,
        }
        if fb1.button("Reading is correct"):
            api_post("/feedback", {**payload, "corrected_label": 1 if prob >= 0.08 else 0})
            st.success("Recorded. This becomes a confirmed training example.")
        if fb2.button("Reading is wrong"):
            api_post("/feedback", {**payload, "corrected_label": 0 if prob >= 0.08 else 1})
            st.success("Recorded with the label flipped. It goes into the retraining set.")
        fb3.caption(
            "Corrections are appended to backend/feedback.csv and can be "
            "concatenated straight into the training file."
        )
        T.panel_close()


# --------------------------------------------------------------------------
# Tab 2 - batch
# --------------------------------------------------------------------------

with tab_batch:
    T.note(
        "This is the path that replaces quarterly manual triage. OIL's HSSE platform "
        "exports observations in bulk; the single-report view is the drill-down, not "
        "the workflow."
    )
    st.write("")

    up = st.file_uploader("CSV export with a narrative column", type=["csv"])

    if up is None:
        T.empty_state(
            "No file loaded",
            "Upload a CSV containing one report per row. Any column holding the "
            "narrative will do - you pick it after upload. data/labelled.csv works "
            "as a demonstration set.",
        )
    else:
        df = pd.read_csv(up, on_bad_lines="skip")
        st.caption(f"{len(df):,} rows &middot; {len(df.columns)} columns")

        s1, s2, s3 = st.columns([2, 2, 1])
        guess = next(
            (i for i, c in enumerate(df.columns)
             if any(h in c.lower() for h in ("narrative", "description", "text", "summary"))),
            0,
        )
        text_col = s1.selectbox("Narrative column", df.columns, index=guess)
        site_col = s2.selectbox("Site column (optional)", ["Not in this file"] + list(df.columns))
        s3.write("")
        run = s3.button("Score all", type="primary")

        if run:
            reports = [
                {
                    "report_id": f"BATCH-{i}",
                    "text": str(row[text_col]).strip(),
                    "site": None if site_col == "Not in this file" else str(row[site_col]),
                }
                for i, row in df.iterrows()
                if len(str(row[text_col]).strip()) >= 20
            ]
            with st.spinner(f"Scoring {len(reports):,} narratives..."):
                res = api_post("/analyze_batch", {"reports": reports})
            if res:
                st.session_state["batch"] = res

        res = st.session_state.get("batch")
        if res:
            st.write("")
            b1, b2 = st.columns([2, 3], gap="large")

            with b1:
                T.statusbar([
                    {"label": "Scored", "value": f"{res['total']:,}"},
                    {"label": "Fatal potential", "value": f"{res['sif_flagged']:,}",
                     "state": "degraded", "note": f"{res['sif_rate']:.1%} of batch"},
                    {"label": "Escalated", "value": f"{res['escalated']:,}",
                     "note": "straight to supervisor"},
                ])
                rate = res["sif_rate"]
                if 0.15 <= rate <= 0.35:
                    T.note(
                        f"A flag rate of {rate:.1%} sits inside the 20-25% band that "
                        "published SIF research reports. That is a sanity check passing, "
                        "not a coincidence."
                    )
                else:
                    T.note(
                        f"A flag rate of {rate:.1%} sits outside the 20-25% band the "
                        "literature reports. Worth re-running the threshold sweep before "
                        "reading anything into this.",
                        warn=True,
                    )

            with b2:
                T.panel_open("Score distribution", "every report in the batch")
                T.distribution([r["sif_probability"] for r in res["results"]])
                T.panel_close()

            st.write("")
            rows = [{
                "Score": r["sif_probability"],
                "Action": r["triage_action"].split("_")[0].title(),
                "Site": r["site"],
                "IOGP rule": r["iogp_rules"][0]["rule"] if r["iogp_rules"] else "-",
                "Energy": ", ".join(list(r["assessment"]["energy_sources"].keys())[:2]) or "-",
                "Why": r["assessment"]["rationale"],
                "Report": r["report_id"],
            } for r in res["results"]]
            out = pd.DataFrame(rows).sort_values("Score", ascending=False)

            T.panel_open("Ranked queue", "highest fatal potential first")
            st.dataframe(
                out, use_container_width=True, height=430, hide_index=True,
                column_config={
                    "Score": st.column_config.ProgressColumn(
                        "Score", format="%.2f", min_value=0.0, max_value=1.0, width="small"),
                    "Why": st.column_config.TextColumn("Why", width="large"),
                },
            )
            buf = io.StringIO()
            out.to_csv(buf, index=False)
            st.download_button("Download scored queue", buf.getvalue(),
                               "sif_scored.csv", "text/csv")
            T.panel_close()


# --------------------------------------------------------------------------
# Tab 3 - density (the deliverable)
# --------------------------------------------------------------------------

with tab_density:
    agg = api_get("/aggregate")

    if not agg or not agg.get("total_reports"):
        T.empty_state(
            "No scored reports yet",
            "This screen ranks sites and activities once reports have been scored. "
            "Go to Score a batch, upload an export, and come back.",
        )
    else:
        T.note(
            "Ranked by the share of a site's reports carrying fatal potential, not by "
            "raw count. Counting raw would rank sites by how much paperwork they file, "
            "which penalises the ones reporting honestly."
        )
        st.write("")

        d1, d2 = st.columns(2, gap="large")
        with d1:
            T.panel_open("Sites", "by precursor density")
            T.rank_bars(agg["sites"])
            T.panel_close()
            T.panel_open("Energy sources", "what is releasing")
            T.rank_bars(agg["energy_sources"], unit="mentions")
            T.panel_close()
        with d2:
            T.panel_open("Activities", "what people were doing")
            T.rank_bars(agg["activities"])
            T.panel_close()
            T.panel_open("IOGP rules", "which rule is under strain")
            T.rank_bars(agg["iogp_rules"], unit="reports")
            T.panel_close()

        if agg.get("barriers"):
            T.panel_open("Controls failing most often", "across every scored report")
            bar_rows = agg["barriers"][:10]
            top = max(b["occurrences"] for b in bar_rows) or 1
            html = ""
            for b in bar_rows:
                w = b["occurrences"] / top * 100
                html += (
                    f"<div class='rank'><div>"
                    f"<div class='nm'>{b['barrier_failure']}</div>"
                    f"<div class='track'><div class='fill' style='width:{w:.1f}%;"
                    f"background:{T.WARNING}'></div></div></div>"
                    f"<div class='pct' style='color:{T.WARNING}'>{b['occurrences']}</div></div>"
                )
            st.markdown(html, unsafe_allow_html=True)
            T.panel_close()


# --------------------------------------------------------------------------
# Tab 4 - clusters
# --------------------------------------------------------------------------

with tab_patterns:
    cl = api_get("/clusters")

    if not cl or not cl.get("clusters"):
        T.empty_state(
            "No recurring patterns found",
            f"Clustering ran over {(cl or {}).get('corpus_size', 0):,} historical "
            "narratives and found nothing repeating often enough to call a pattern. "
            "That is the honest answer for a small corpus - load more history into "
            "data/ rather than lowering the bar.",
        )
    else:
        T.note(
            f"{len(cl['clusters'])} recurring precursor patterns across "
            f"{cl['corpus_size']:,} historical narratives. Reports matching none of them "
            "are left unclustered rather than forced into a group, because most safety "
            "reports genuinely are one-offs."
        )
        st.write("")

        for c in cl["clusters"]:
            T.panel_open(
                f"Pattern {c['cluster_id']}",
                f"{c['size']} reports &middot; {', '.join(c['sources'])}",
            )
            st.markdown(
                f"<p style='font-size:13px;line-height:1.55;color:#4A5563;margin:0;"
                f"max-width:90ch'>{c['representative']}</p>",
                unsafe_allow_html=True,
            )
            T.panel_close()
