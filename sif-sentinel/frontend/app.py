"""
SIF-Sentinel dashboard.

Changes from the original that matter:

  * Every KPI is computed from data. The old header hardcoded "34 rigs",
    "148 ingestions", "9 precursors flagged" and "0.02% bias variance", and
    the emerging-threat feed was a static st.info() string rather than
    clustering output. A judge who changes the input and watches the KPIs stay
    frozen has learned something you did not want them to learn.
  * The analyse button works. The old condition was `if analyze_click or
    report_text:` — report_text is never empty, so the pipeline re-ran on
    every Streamlit rerun, i.e. every keystroke, and the button did nothing.
  * There is a batch tab. Deliverable (c) of the problem statement asks for a
    dashboard that ranks sites and activities by SIF-precursor density; the
    single-report analyser is a drill-down inside that, not the product.
  * The backend URL is configurable instead of hardcoded to localhost.
"""

from __future__ import annotations

import io
import os

import pandas as pd
import requests
import streamlit as st

API = os.getenv("SIF_API_URL", "http://localhost:8000")
TIMEOUT = 120

st.set_page_config(page_title="SIF-Sentinel | OIL HSSE", layout="wide", page_icon="🛢️")

st.markdown("""
<style>
    .band-esc  { background:#fee2e2; color:#991b1b; padding:10px 14px; border-radius:8px;
                 font-weight:700; border-left:5px solid #dc2626; }
    .band-rev  { background:#fef3c7; color:#92400e; padding:10px 14px; border-radius:8px;
                 font-weight:700; border-left:5px solid #d97706; }
    .band-file { background:#dcfce7; color:#166534; padding:10px 14px; border-radius:8px;
                 font-weight:700; border-left:5px solid #16a34a; }
    .tag { display:inline-block; padding:3px 9px; border-radius:4px; font-size:12px;
           margin:2px; font-weight:600; }
    .t-energy  { background:#fee2e2; color:#991b1b; }
    .t-barrier { background:#ffedd5; color:#9a3412; }
    .t-expose  { background:#fef9c3; color:#854d0e; }
    .t-neutral { background:#e2e8f0; color:#334155; }
</style>
""", unsafe_allow_html=True)


def api_get(path: str, **params):
    try:
        r = requests.get(f"{API}{path}", params=params, timeout=TIMEOUT)
        r.raise_for_status()
        return r.json()
    except requests.RequestException as exc:
        st.error(f"Backend unreachable at {API}{path} — {exc}")
        return None


def api_post(path: str, payload: dict):
    try:
        r = requests.post(f"{API}{path}", json=payload, timeout=TIMEOUT)
        if r.status_code != 200:
            st.error(f"Backend returned {r.status_code}: {r.text[:300]}")
            return None
        return r.json()
    except requests.RequestException as exc:
        st.error(f"Backend unreachable at {API}{path} — {exc}")
        return None


# --------------------------------------------------------------------------
# Header — every figure below comes from /health and /aggregate.
# --------------------------------------------------------------------------

st.title("🛢️ SIF-Sentinel")
st.caption("Serious Injury & Fatality precursor detection for OIL UA/UC, near-miss and incident reports")

health = api_get("/health")
agg = api_get("/aggregate")

if health:
    method = health.get("scoring_method", "unknown")
    if not health.get("trained_model_loaded"):
        st.warning(
            "Running on the transparent rule baseline (energy → barrier → exposure). "
            "No trained classifier is loaded. Train one with "
            "`python backend/train.py --data <labelled.csv>` and this banner disappears.",
            icon="⚠️",
        )

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Historical corpus", f"{health.get('corpus_size', 0):,}")
    k2.metric("Reports scored this session", f"{health.get('reports_scored_this_session', 0):,}")
    flagged = (agg or {}).get("sif_flagged", 0)
    total = (agg or {}).get("total_reports", 0)
    k3.metric("SIF-potential flagged", f"{flagged:,}",
              f"{flagged / total:.0%} of scored" if total else "no reports yet")
    k4.metric("HSE corrections collected", f"{health.get('feedback_collected', 0):,}")

st.divider()

tab_single, tab_batch, tab_rank, tab_clusters = st.tabs(
    ["🔍 Single report", "📥 Batch ingest", "📊 Precursor density", "🧭 Emerging clusters"]
)

# --------------------------------------------------------------------------
# Single report
# --------------------------------------------------------------------------

with tab_single:
    st.subheader("Analyse a narrative")

    presets = {
        "— type your own —": "",
        "LOTO bypass, pressure release (Rig 42)":
            "During routine maintenance at Rig 42, the main compressor valve was jammed. "
            "Lockout/tagout energy isolation was bypassed. A sudden high-pressure release "
            "caused a metal pipe fitting to detach and fly across the deck, narrowly "
            "missing the worker.",
        "Suspended load, no banksman (Rig 19)":
            "While tripping pipe at Rig 19, winch cable tension slackened abruptly. The "
            "suspended drill collar shifted 2 metres outside the designated rotary envelope "
            "without an active banksman present.",
        "Housekeeping spill (canteen)":
            "Housekeeping notice: minor water spill observed near the kitchen walkway. "
            "Slippery surface warning sign placed. No personnel injured.",
        "Simplified / code-mixed phrasing":
            "worker no lockout do. valve open suddenly. gas come out fast. man standing "
            "near, very close, no injury but big risk.",
    }

    choice = st.selectbox("Preset", list(presets.keys()))
    text = st.text_area("Narrative", presets[choice], height=140, key="narrative")

    if st.button("Analyse", type="primary"):
        if not text.strip():
            st.warning("Enter a narrative first.")
        else:
            with st.spinner("Scoring…"):
                data = api_post("/analyze", {"report_id": "UI-001", "text": text})

            if data:
                st.session_state["last_result"] = data

    data = st.session_state.get("last_result")
    if data:
        prob = data["sif_probability"]
        action = data["triage_action"]
        assessment = data["assessment"]

        left, right = st.columns([3, 2])

        with left:
            if action.startswith("ESCALATE"):
                st.markdown(
                    f"<div class='band-esc'>🚨 ESCALATE — SIF potential {prob:.0%}</div>",
                    unsafe_allow_html=True)
            elif action.startswith("HUMAN"):
                st.markdown(
                    f"<div class='band-rev'>⚠️ HUMAN REVIEW — SIF potential {prob:.0%}</div>",
                    unsafe_allow_html=True)
            else:
                st.markdown(
                    f"<div class='band-file'>✅ AUTO-FILE — SIF potential {prob:.0%}</div>",
                    unsafe_allow_html=True)

            st.markdown("#### Why")
            st.write(assessment["rationale"])

            chips = ""
            for src, hits in assessment["energy_sources"].items():
                chips += f"<span class='tag t-energy'>ENERGY: {src}</span>"
            for b in assessment["barrier_failures"][:6]:
                chips += f"<span class='tag t-barrier'>BARRIER FAILED: {b}</span>"
            for e in assessment["exposure_evidence"][:4]:
                chips += f"<span class='tag t-expose'>EXPOSURE: {e}</span>"
            for h in assessment["barrier_intact_evidence"][:3]:
                chips += f"<span class='tag t-neutral'>CONTROL HELD: {h}</span>"
            st.markdown(chips or "<i>No structural evidence extracted.</i>",
                        unsafe_allow_html=True)

            st.caption(f"Scoring method: `{data['scoring_method']}`")

        with right:
            st.markdown("#### IOGP Life-Saving Rules")
            if data["iogp_rules"]:
                for rule in data["iogp_rules"]:
                    st.write(
                        f"**{rule['rule']}** — {rule['confidence']:.0%} "
                        f"({rule['evidence']})"
                    )
                    if rule["matched_terms"]:
                        st.caption("matched: " + ", ".join(rule["matched_terms"][:5]))
            else:
                st.info("No Life-Saving Rule matched above threshold.")

            st.markdown("#### Extracted precursors")
            p = data["precursors"]
            for label, key in [("Activity", "activities"), ("Equipment", "equipment"),
                               ("Location", "locations"), ("Barriers referenced", "barriers_referenced")]:
                if p.get(key):
                    st.write(f"**{label}:** {', '.join(p[key])}")

            cl = data.get("clustering", {})
            if cl.get("cluster_summary"):
                st.markdown("#### Pattern match")
                st.caption(cl["cluster_summary"])

        st.divider()
        st.markdown("#### HSE reviewer correction")
        st.caption(
            "Corrections queue into backend/feedback.csv and become the labelled "
            "set that replaces the rule baseline with a trained model."
        )
        c1, c2, _ = st.columns([1, 1, 3])
        if c1.button("✔ Correctly classified"):
            api_post("/feedback", {
                "report_id": data["report_id"], "text": text,
                "predicted_probability": prob,
                "corrected_label": 1 if prob >= 0.40 else 0,
            })
            st.success("Recorded.")
        if c2.button("✘ Wrong — flip the label"):
            api_post("/feedback", {
                "report_id": data["report_id"], "text": text,
                "predicted_probability": prob,
                "corrected_label": 0 if prob >= 0.40 else 1,
            })
            st.success("Recorded. This report goes into the retraining set.")

# --------------------------------------------------------------------------
# Batch
# --------------------------------------------------------------------------

with tab_batch:
    st.subheader("Score a file of reports")
    st.caption(
        "OIL's HSSE platform exports UA/UC observations in bulk. This is the "
        "path that replaces monthly manual triage — the single-report view is "
        "the drill-down, not the workflow."
    )

    upload = st.file_uploader("CSV with a narrative column", type=["csv"])
    if upload is not None:
        df = pd.read_csv(upload, on_bad_lines="skip")
        st.write(f"{len(df):,} rows, {len(df.columns)} columns")

        text_col = st.selectbox("Narrative column", df.columns,
                                index=next((i for i, c in enumerate(df.columns)
                                            if any(h in c.lower() for h in
                                                   ["narrative", "description", "text", "summary"])), 0))
        site_col = st.selectbox("Site column (optional)", ["— none —"] + list(df.columns))

        if st.button("Run pipeline on file", type="primary"):
            reports = []
            for i, row in df.iterrows():
                txt = str(row[text_col]).strip()
                if len(txt) < 20:
                    continue
                reports.append({
                    "report_id": f"BATCH-{i}",
                    "text": txt,
                    "site": None if site_col == "— none —" else str(row[site_col]),
                })

            with st.spinner(f"Scoring {len(reports):,} reports…"):
                res = api_post("/analyze_batch", {"reports": reports})

            if res:
                m1, m2, m3 = st.columns(3)
                m1.metric("Scored", f"{res['total']:,}")
                m2.metric("SIF-potential", f"{res['sif_flagged']:,}", f"{res['sif_rate']:.1%}")
                m3.metric("Escalated", f"{res['escalated']:,}")

                st.caption(
                    "Published benchmarks put genuine SIF potential at roughly "
                    "20–25% of reports. A flag rate far outside that band is a "
                    "signal to re-check your threshold, not a result."
                )

                rows = [{
                    "report_id": r["report_id"],
                    "sif_probability": r["sif_probability"],
                    "triage": r["triage_action"],
                    "site": r["site"],
                    "iogp_rule": r["iogp_rules"][0]["rule"] if r["iogp_rules"] else "",
                    "activity": ", ".join(r["precursors"]["activities"][:2]),
                    "energy": ", ".join(list(r["assessment"]["energy_sources"].keys())[:2]),
                    "rationale": r["assessment"]["rationale"],
                } for r in res["results"]]

                out = pd.DataFrame(rows).sort_values("sif_probability", ascending=False)
                st.dataframe(out, use_container_width=True, height=420)

                buf = io.StringIO()
                out.to_csv(buf, index=False)
                st.download_button("Download scored CSV", buf.getvalue(),
                                   "sif_scored.csv", "text/csv")

# --------------------------------------------------------------------------
# Ranking — the actual deliverable
# --------------------------------------------------------------------------

with tab_rank:
    st.subheader("SIF-precursor density")
    st.caption(
        "Ranked by the share of a site's or activity's reports carrying SIF "
        "potential, not by raw count. Raw counts rank sites by how much they "
        "report, which penalises good reporting culture."
    )

    agg = api_get("/aggregate")
    if not agg or not agg.get("total_reports"):
        st.info("Nothing scored yet. Run a batch in the previous tab.")
    else:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### By site")
            st.dataframe(pd.DataFrame(agg["sites"]), use_container_width=True, height=320)
            st.markdown("#### By IOGP rule")
            st.dataframe(pd.DataFrame(agg["iogp_rules"]), use_container_width=True, height=280)
        with c2:
            st.markdown("#### By activity")
            st.dataframe(pd.DataFrame(agg["activities"]), use_container_width=True, height=320)
            st.markdown("#### By energy source")
            st.dataframe(pd.DataFrame(agg["energy_sources"]), use_container_width=True, height=280)

        st.markdown("#### Most frequent barrier failures")
        if agg["barriers"]:
            st.dataframe(pd.DataFrame(agg["barriers"]), use_container_width=True)
        else:
            st.caption("No barrier failures extracted yet.")

# --------------------------------------------------------------------------
# Clusters
# --------------------------------------------------------------------------

with tab_clusters:
    st.subheader("Recurring precursor patterns")
    st.caption(
        "HDBSCAN over sentence embeddings of the historical corpus, fitted once "
        "at startup. New reports are assigned with approximate_predict rather "
        "than re-clustering the corpus per request."
    )

    cl = api_get("/clusters")
    if not cl or not cl.get("clusters"):
        st.info(
            f"No clusters. Corpus is {(cl or {}).get('corpus_size', 0)} reports — "
            "clustering needs a few hundred at minimum to surface anything real. "
            "Load more historical data into data/."
        )
    else:
        for c in cl["clusters"]:
            with st.expander(f"Pattern #{c['cluster_id']} — {c['size']} reports "
                             f"({', '.join(c['sources'])})"):
                st.write(c["representative"])
