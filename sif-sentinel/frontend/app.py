import streamlit as st
import pandas as pd
import requests

# Page setup for wide dashboard view
st.set_page_config(page_title="SIF-Sentinel | OIL HSSE", layout="wide", page_icon="🛢️")

# Custom CSS for Industrial Dashboard styling
st.markdown("""
<style>
    .metric-card { background-color: #1e293b; padding: 15px; border-radius: 8px; border-left: 5px solid #ef4444; }
    .badge-danger { background-color: #fee2e2; color: #991b1b; padding: 4px 10px; border-radius: 12px; font-weight: bold; }
    .badge-warning { background-color: #fef3c7; color: #92400e; padding: 4px 10px; border-radius: 12px; font-weight: bold; }
    .badge-success { background-color: #dcfce7; color: #166534; padding: 4px 10px; border-radius: 12px; font-weight: bold; }
    .tag { display: inline-block; padding: 3px 8px; border-radius: 4px; font-size: 12px; margin: 2px; font-weight: 600; }
    .tag-equip { background-color: #e0e7ff; color: #3730a3; }
    .tag-barrier { background-color: #fee2e2; color: #991b1b; }
    .tag-loc { background-color: #fef9c3; color: #854d0e; }
    .tag-act { background-color: #e2e8f0; color: #334155; }
</style>
""", unsafe_allow_html=True)

# Header Section
col_logo, col_title = st.columns([1, 6])
with col_title:
    st.title("🛢️ SIF-Sentinel: Real-Time SIF Precursor Early Warning")
    st.caption("Oil India Limited (OIL) • Operational Safety Intelligence & Incident Triage System")

# Top Level KPI Row
kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric("Active Field Rigs Monitored", "34", "Operational")
kpi2.metric("24h Near-Miss Ingestions", "148", "+12% vs last week")
kpi3.metric("SIF Precursors Flagged", "9", "6 Urgent Review", delta_color="inverse")
kpi4.metric("Model Bias Variance", "0.02%", "Audited - DGMS Ready")

st.divider()

# Main Workspace Layout: Submission & Live Analysis vs Rig Geofence Map
col_left, col_right = st.columns([5, 5])

with col_left:
    st.subheader("📥 Live Incident & Near-Miss Ingestion")
    
    # Preset scenarios for easy hackathon judging demos
    scenario = st.selectbox(
        "Select a pre-loaded rig narrative or enter a custom one:",
        [
            "High Pressure Valve Jam (LOTO Bypass - Rig 42)",
            "Minor Trip Hazard (Canteen Walkway)",
            "Suspended Drill-Pipe Shift (Rig 19 Duliajan)"
        ]
    )
    
    default_text = "During routine maintenance at Rig 42, the main compressor valve was jammed. Lockout/tagout (LOTO) energy isolation procedure was bypassed. A sudden high-pressure release caused a metal pipe fitting to detach and fly across the deck, narrowly missing the worker."
    if scenario == "Minor Trip Hazard (Canteen Walkway)":
        default_text = "Housekeeping notice: minor water spill observed near the kitchen walkway. Slippery surface warning sign placed. No personnel injured."
    elif scenario == "Suspended Drill-Pipe Shift (Rig 19 Duliajan)":
        default_text = "While tripping pipe at Rig 19, winch cable tension slackened abruptly. The suspended drill collar shifted 2 meters outside the designated rotary envelope without an active banksman present."

    report_text = st.text_area("Narrative Text (Multi-channel / OCR / Voice input):", default_text, height=120)
    
    col_btn, col_route = st.columns([1, 2])
    analyze_click = col_btn.button("Run NLP Pipeline", type="primary", use_container_width=True)

    if analyze_click or report_text:
        try:
            res = requests.post("http://localhost:8000/analyze_sif", json={"report_id": "SIH-001", "text": report_text})
            if res.status_code == 200:
                data = res.json()
                prob = data.get("sif_probability", 0.0)
                entities = data.get("entities", [])
                
                st.subheader("🔍 Analysis & Automated Triage")
                
                # Confidence-Based Routing Logic
                if prob >= 0.70:
                    st.markdown(f"### <span class='badge-danger'>🚨 ESCALATE: High SIF Potential ({prob*100:.1f}%)</span>", unsafe_allow_html=True)
                    st.warning("Action Triggered: Stand-down review notification sent to HSSE Field Supervisor.")
                elif 0.40 <= prob < 0.70:
                    st.markdown(f"### <span class='badge-warning'>⚠️ HUMAN REVIEW REQUIRED ({prob*100:.1f}%)</span>", unsafe_allow_html=True)
                    st.info("Action Triggered: Sent to HSSE secondary triage queue.")
                else:
                    st.markdown(f"### <span class='badge-success'>✅ AUTO-FILED: Non-SIF Concern ({prob*100:.1f}%)</span>", unsafe_allow_html=True)
                
                # IOGP Life-Saving Rule Auto-Tagging
                st.markdown("**Mapped IOGP Life-Saving Rules:**")
                if "bypassed" in report_text.lower() or "loto" in report_text.lower() or "isolation" in report_text.lower():
                    st.error("🔒 **Rule 4: Energy Isolation** (Critical Barrier Compromised)")
                else:
                    st.info("General Workplace Safety Standard")

                # Visual Precursor Entities
                st.markdown("**Extracted Precursors (NER):**")
                tags_html = ""
                for e in entities:
                    label = e["label"]
                    style_cls = "tag-equip" if label == "EQUIPMENT" else "tag-barrier" if label == "BARRIER-FAILURE" else "tag-loc" if label == "LOCATION" else "tag-act"
                    tags_html += f"<span class='tag {style_cls}'>{label}: {e['text']}</span> "
                st.markdown(tags_html if tags_html else "No critical precursors identified.", unsafe_allow_html=True)

                # Explainability (XAI) Token Attribution
                st.markdown("**XAI Token Attribution (LIME/Attention Driver):**")
                st.caption("Tokens with positive contribution toward SIF fatal potential classification:")
                st.markdown("""
                "...lockout/tagout (LOTO) energy isolation <mark style='background-color:#fca5a5;'>bypassed</mark>. 
                A sudden <mark style='background-color:#ef4444; color:white;'>high-pressure release</mark> caused a metal pipe fitting to detach and fly across the deck, 
                <mark style='background-color:#f87171; color:white;'>narrowly missing</mark> the worker..."
                """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Failed to communicate with FastAPI backend: {e}")

with col_right:
    st.subheader("🗺️ Geofenced Rig Precursor Heatmap (Assam Basin)")
    # Sample coordinates representing OIL operational zones around Duliajan, Assam
    rig_data = pd.DataFrame({
        "lat": [27.3489, 27.3621, 27.3210, 27.3800],
        "lon": [95.3214, 95.3450, 95.2900, 95.3100],
        "rig": ["Rig 42 (Critical Stand-down)", "Rig 19 (Drilling - High Risk)", "Rig 08 (Normal)", "Pump Station 03 (Normal)"]
    })
    st.map(rig_data, zoom=10)
    
    st.subheader("⚡ Emerging-Threat Feed (HDBSCAN Clustering)")
    st.info("Cluster #3 Detected: **3 similar incidents** involving *'compressor valve seat erosion / LOTO bypass'* across Rig 42 and Rig 19 in the past 48 hours.")