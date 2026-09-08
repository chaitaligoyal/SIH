"""
SIF-Sentinel design system.

One place for tokens and reusable components, so screens compose primitives
instead of restyling themselves. Nothing here calls the API or holds state.

Visual direction: control-room instrumentation, not a SaaS dashboard.

  * Colour comes from ANSI Z535 / ISO 3864 safety signage — the legislated
    palette an HSE officer already reads fluently. Red is danger, amber is
    warning, green is safe, blue is mandatory action. Nothing is coloured
    decoratively; if something is red it is because the standard says red
    means that.
  * Ground is cool slate rather than white, so signal colour carries weight
    without needing saturation.
  * Panels have hairline rules and no shadow. Radius is reserved for things
    you can click, so shape itself tells you what is interactive.
  * Archivo for text (a grotesque with signage lineage), IBM Plex Mono for
    numeric readouts only — tabular figures matter when digits must align
    down a column.
  * All-caps is used only for the signal words DANGER / WARNING / SAFE, which
    is the actual convention of the standard, never as a decorative label.
"""

from __future__ import annotations

from typing import List, Optional

import streamlit as st

# --------------------------------------------------------------------------
# Tokens
# --------------------------------------------------------------------------

INK = "#10151B"          # primary text
INK_2 = "#4A5563"        # secondary text
INK_3 = "#79838F"        # tertiary / captions
RULE = "#D4D9DF"         # hairline borders
RULE_STRONG = "#B3BBC4"
GROUND = "#EDEFF2"       # page background
PANEL = "#FFFFFF"        # panel background
PANEL_SUNK = "#F5F7F9"   # inset areas

DANGER = "#B4232A"       # ANSI safety red
DANGER_SOFT = "#FBEAEA"
WARNING = "#C9761B"      # ANSI safety orange
WARNING_SOFT = "#FDF1E3"
SAFE = "#1F7A4C"         # ANSI safety green
SAFE_SOFT = "#E7F3ED"
MANDATE = "#1D4E8F"      # ANSI safety blue
MANDATE_SOFT = "#E8EFF7"

# Operating points from the held-out threshold sweep in train.py.
# Kept here so the gauge draws the same setpoints the backend triages on.
T_REVIEW = 0.08
T_ESCALATE = 0.30


def inject_css() -> None:
    """Load fonts and the full stylesheet. Call once, first thing."""
    st.markdown(
        f"""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<style>
@import url('https://fonts.googleapis.com/css2?family=Archivo:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

:root {{
  --ink: {INK}; --ink2: {INK_2}; --ink3: {INK_3};
  --rule: {RULE}; --rule-strong: {RULE_STRONG};
  --ground: {GROUND}; --panel: {PANEL}; --sunk: {PANEL_SUNK};
  --danger: {DANGER}; --warning: {WARNING}; --safe: {SAFE}; --mandate: {MANDATE};
}}

/* ---- base ---------------------------------------------------------- */
html, body, [class*="css"], .stApp {{
  font-family: 'Archivo', system-ui, -apple-system, sans-serif;
  color: var(--ink);
}}
.stApp {{ background: var(--ground); }}
.block-container {{ padding: 0 2.25rem 4rem; max-width: 1500px; }}

/* Streamlit chrome we don't want in a demo */
#MainMenu, footer, header[data-testid="stHeader"] {{ display: none; }}

/* Keyboard focus must stay visible — quality floor, not decoration. */
*:focus-visible {{ outline: 2px solid var(--mandate); outline-offset: 2px; }}

@media (prefers-reduced-motion: reduce) {{
  * {{ animation: none !important; transition: none !important; }}
}}

/* ---- masthead ------------------------------------------------------ */
.masthead {{
  display: flex; align-items: baseline; gap: 14px;
  padding: 22px 0 12px; border-bottom: 2px solid var(--ink);
}}
.masthead h1 {{
  font-size: 25px; font-weight: 700; letter-spacing: -0.022em;
  margin: 0; line-height: 1;
}}
.masthead .sub {{
  font-size: 13.5px; color: var(--ink2); line-height: 1.3;
  border-left: 1px solid var(--rule-strong); padding-left: 14px;
}}

/* ---- status strip -------------------------------------------------- */
.statusbar {{
  display: flex; flex-wrap: wrap; gap: 0;
  border: 1px solid var(--rule); border-top: none;
  background: var(--panel); margin-bottom: 22px;
}}
.stat {{
  flex: 1 1 0; min-width: 150px; padding: 11px 16px 12px;
  border-right: 1px solid var(--rule);
}}
.stat:last-child {{ border-right: none; }}
.stat .k {{
  font-size: 10.5px; color: var(--ink3); letter-spacing: .05em;
  font-weight: 600; margin-bottom: 3px;
}}
.stat .v {{
  font-family: 'IBM Plex Mono', monospace; font-size: 19px;
  font-weight: 500; font-variant-numeric: tabular-nums;
  letter-spacing: -0.02em; line-height: 1.1;
}}
.stat .note {{ font-size: 11.5px; color: var(--ink3); margin-top: 2px; }}
.stat.live .v {{ color: var(--safe); }}
.stat.degraded .v {{ color: var(--warning); }}

/* ---- panels -------------------------------------------------------- */
.panel {{ background: var(--panel); border: 1px solid var(--rule); padding: 18px 20px 20px; }}
.panel + .panel {{ margin-top: 14px; }}
.panel-title {{
  font-size: 12.5px; font-weight: 700; letter-spacing: .01em;
  padding-bottom: 8px; margin-bottom: 14px;
  border-bottom: 1px solid var(--rule);
  display: flex; justify-content: space-between; align-items: baseline;
}}
.panel-title .aux {{ font-weight: 400; font-size: 11.5px; color: var(--ink3); }}

/* ---- verdict ------------------------------------------------------- */
.verdict {{ display: flex; align-items: stretch; border: 1px solid var(--rule); background: var(--panel); }}
.verdict .band {{ width: 9px; flex: none; }}
.verdict .body {{ padding: 16px 20px 18px; flex: 1; }}
.verdict .word {{
  font-size: 21px; font-weight: 700; letter-spacing: .04em; line-height: 1;
}}
.verdict .desc {{ font-size: 13px; color: var(--ink2); margin-top: 6px; max-width: 62ch; line-height: 1.45; }}
.v-danger .word {{ color: var(--danger); }}
.v-warning .word {{ color: var(--warning); }}
.v-safe .word {{ color: var(--safe); }}

/* ---- evidence chips ------------------------------------------------ */
.chiprow {{ display: flex; flex-wrap: wrap; gap: 6px; }}
.chip {{
  display: inline-flex; align-items: center; gap: 7px;
  font-size: 12px; padding: 5px 10px 5px 8px;
  border: 1px solid var(--rule-strong); background: var(--panel);
  max-width: 100%;
}}
.chip .dot {{ width: 7px; height: 7px; flex: none; }}
.chip .lab {{ color: var(--ink3); font-size: 10.5px; font-weight: 600; letter-spacing: .04em; }}
.chip .val {{ font-weight: 500; }}
.c-energy   {{ border-left: 3px solid var(--danger); }}
.c-energy .dot {{ background: var(--danger); }}
.c-barrier  {{ border-left: 3px solid var(--warning); }}
.c-barrier .dot {{ background: var(--warning); }}
.c-exposure {{ border-left: 3px solid #8A6A17; }}
.c-exposure .dot {{ background: #8A6A17; }}
.c-held     {{ border-left: 3px solid var(--safe); }}
.c-held .dot {{ background: var(--safe); }}

/* ---- rule rows ----------------------------------------------------- */
.rulerow {{ padding: 10px 0; border-bottom: 1px solid var(--rule); }}
.rulerow:last-child {{ border-bottom: none; }}
.rulerow .top {{ display: flex; justify-content: space-between; align-items: baseline; gap: 10px; }}
.rulerow .nm {{ font-size: 13.5px; font-weight: 600; }}
.rulerow .cf {{
  font-family: 'IBM Plex Mono', monospace; font-size: 12.5px;
  font-variant-numeric: tabular-nums; color: var(--ink2);
}}
.rulerow .bar {{ height: 3px; background: var(--sunk); margin: 7px 0 5px; }}
.rulerow .bar span {{ display: block; height: 100%; background: var(--mandate); }}
.rulerow .terms {{ font-size: 11.5px; color: var(--ink3); }}

/* ---- ranked bars --------------------------------------------------- */
.rank {{ display: grid; grid-template-columns: 1fr 62px; gap: 12px;
         align-items: center; padding: 9px 0; border-bottom: 1px solid var(--rule); }}
.rank:last-child {{ border-bottom: none; }}
.rank .nm {{ font-size: 13px; font-weight: 500; margin-bottom: 5px; }}
.rank .track {{ height: 8px; background: var(--sunk); }}
.rank .fill {{ height: 100%; }}
.rank .meta {{ font-size: 11px; color: var(--ink3); margin-top: 4px; }}
.rank .pct {{
  font-family: 'IBM Plex Mono', monospace; font-size: 16px; font-weight: 500;
  text-align: right; font-variant-numeric: tabular-nums;
}}

/* ---- empty / hint states ------------------------------------------- */
.empty {{
  border: 1px dashed var(--rule-strong); background: var(--panel);
  padding: 30px 26px; text-align: left;
}}
.empty h4 {{ font-size: 14.5px; font-weight: 600; margin: 0 0 6px; }}
.empty p {{ font-size: 13px; color: var(--ink2); margin: 0; max-width: 60ch; line-height: 1.5; }}

.note {{
  border-left: 3px solid var(--mandate); background: var(--panel);
  padding: 11px 14px; font-size: 12.5px; color: var(--ink2);
  line-height: 1.5; max-width: 78ch;
}}
.note.warn {{ border-left-color: var(--warning); }}

/* ---- streamlit widget overrides ------------------------------------ */
.stTabs [data-baseweb="tab-list"] {{ gap: 0; border-bottom: 1px solid var(--rule); }}
.stTabs [data-baseweb="tab"] {{
  height: 40px; padding: 0 18px; background: transparent;
  font-size: 13.5px; font-weight: 500; color: var(--ink2);
  border-bottom: 2px solid transparent; border-radius: 0;
}}
.stTabs [aria-selected="true"] {{ color: var(--ink); border-bottom-color: var(--danger); font-weight: 600; }}

.stButton > button {{
  border-radius: 3px; font-weight: 600; font-size: 13.5px;
  border: 1px solid var(--rule-strong); padding: 8px 18px;
  transition: background .12s ease, border-color .12s ease;
}}
.stButton > button[kind="primary"] {{
  background: var(--danger); border-color: var(--danger); color: #fff;
}}
.stButton > button[kind="primary"]:hover {{ background: #931B21; border-color: #931B21; }}

.stTextArea textarea, .stSelectbox div[data-baseweb="select"] > div {{
  border-radius: 3px !important; font-size: 13.5px !important;
  border-color: var(--rule-strong) !important;
}}
.stTextArea textarea {{ font-family: 'Archivo', sans-serif !important; line-height: 1.55 !important; }}

[data-testid="stFileUploaderDropzone"] {{
  background: var(--panel); border: 1px dashed var(--rule-strong); border-radius: 3px;
}}
div[data-testid="stDataFrame"] {{ border: 1px solid var(--rule); }}
hr {{ border-color: var(--rule); margin: 20px 0; }}
</style>
""",
        unsafe_allow_html=True,
    )


def _h(html: str) -> None:
    st.markdown(html, unsafe_allow_html=True)


# --------------------------------------------------------------------------
# Components
# --------------------------------------------------------------------------

def masthead(title: str, subtitle: str) -> None:
    _h(f"<div class='masthead'><h1>{title}</h1><div class='sub'>{subtitle}</div></div>")


def statusbar(items: List[dict]) -> None:
    """items: [{label, value, note?, state?}] — state in {'', 'live', 'degraded'}."""
    cells = ""
    for it in items:
        note = f"<div class='note'>{it['note']}</div>" if it.get("note") else ""
        cells += (
            f"<div class='stat {it.get('state','')}'>"
            f"<div class='k'>{it['label']}</div>"
            f"<div class='v'>{it['value']}</div>{note}</div>"
        )
    _h(f"<div class='statusbar'>{cells}</div>")


def panel_open(title: str, aux: str = "") -> None:
    aux_html = f"<span class='aux'>{aux}</span>" if aux else ""
    _h(f"<div class='panel'><div class='panel-title'>{title}{aux_html}</div>")


def panel_close() -> None:
    _h("</div>")


def empty_state(heading: str, body: str) -> None:
    _h(f"<div class='empty'><h4>{heading}</h4><p>{body}</p></div>")


def note(text: str, warn: bool = False) -> None:
    _h(f"<div class='note {'warn' if warn else ''}'>{text}</div>")


def verdict(action: str, prob: float) -> None:
    """The triage decision, stated as a signal word plus what it means to do."""
    if action.startswith("ESCALATE"):
        cls, colour, word = "v-danger", DANGER, "ESCALATE"
        desc = ("Fatal potential above the escalation setpoint. Route to the area "
                "supervisor now — do not queue this for the next review cycle.")
    elif action.startswith("HUMAN"):
        cls, colour, word = "v-warning", WARNING, "REVIEW"
        desc = ("Above the review setpoint. An HSE officer should read this before "
                "it is filed. Roughly half of reports at this level are genuine.")
    else:
        cls, colour, word = "v-safe", SAFE, "FILE"
        desc = ("Below both setpoints. No high-energy hazard with a failed control "
                "was found. File without review.")
    _h(
        f"<div class='verdict {cls}'><div class='band' style='background:{colour}'></div>"
        f"<div class='body'><div class='word'>{word}</div>"
        f"<div class='desc'>{desc}</div></div></div>"
    )


def gauge(prob: float) -> None:
    """
    Fatal-potential reading against its two alarm setpoints.

    Drawn as an instrument scale rather than a progress bar, because the number
    on its own means nothing: what matters is which side of 0.08 and 0.30 it
    falls on. Those setpoints came from the held-out threshold sweep, so the
    gauge shows the calibration and the reading in one glance.
    """
    W, H = 560, 92
    x0, x1 = 8, W - 8
    span = x1 - x0
    y = 44

    def px(v: float) -> float:
        return x0 + span * max(0.0, min(1.0, v))

    fill = DANGER if prob >= T_ESCALATE else (WARNING if prob >= T_REVIEW else SAFE)

    ticks = ""
    for v in (0.0, 0.25, 0.5, 0.75, 1.0):
        ticks += (
            f"<line x1='{px(v):.1f}' y1='{y+13}' x2='{px(v):.1f}' y2='{y+18}' "
            f"stroke='{RULE_STRONG}' stroke-width='1'/>"
            f"<text x='{px(v):.1f}' y='{y+31}' font-family='IBM Plex Mono, monospace' "
            f"font-size='10' fill='{INK_3}' text-anchor='middle'>{v:.2f}</text>"
        )

    setpoints = ""
    for v, lab, col in ((T_REVIEW, "review", WARNING), (T_ESCALATE, "escalate", DANGER)):
        setpoints += (
            f"<line x1='{px(v):.1f}' y1='{y-16}' x2='{px(v):.1f}' y2='{y+13}' "
            f"stroke='{col}' stroke-width='1.5' stroke-dasharray='3 2'/>"
            f"<text x='{px(v)+4:.1f}' y='{y-19}' font-family='Archivo, sans-serif' "
            f"font-size='10.5' font-weight='600' fill='{col}'>{lab} {v:.2f}</text>"
        )

    _h(
        f"""<svg viewBox="0 0 {W} {H}" width="100%" role="img"
     aria-label="Fatal potential {prob:.0%}, review setpoint {T_REVIEW}, escalate setpoint {T_ESCALATE}">
  <rect x="{x0}" y="{y}" width="{span}" height="13" fill="{PANEL_SUNK}"/>
  <rect x="{x0}" y="{y}" width="{max(2, px(prob)-x0):.1f}" height="13" fill="{fill}"/>
  {ticks}{setpoints}
  <polygon points="{px(prob):.1f},{y-4} {px(prob)-5:.1f},{y-12} {px(prob)+5:.1f},{y-12}" fill="{INK}"/>
  <line x1="{px(prob):.1f}" y1="{y-4}" x2="{px(prob):.1f}" y2="{y+13}" stroke="{INK}" stroke-width="2"/>
</svg>"""
    )


def barrier_chain(assessment: dict) -> None:
    """
    The three-gate model the classifier is built on, drawn the way HSE
    engineers draw it: hazard on the left, person on the right, controls in
    between. A precursor is the case where all three gates are open.

    This is the one place the interface raises its voice, because it is the
    one idea a judge needs to leave with.
    """
    energy = assessment.get("energy_sources", {}) or {}
    failed = assessment.get("barrier_failures", []) or []
    held = assessment.get("barrier_intact_evidence", []) or []
    exposed = assessment.get("exposure_evidence", []) or []
    ruled_out = assessment.get("exposure_ruled_out", []) or []

    if energy:
        g1 = (DANGER, "PRESENT", ", ".join(list(energy.keys())[:2]))
    else:
        g1 = (SAFE, "NONE FOUND", "no high-energy source in this narrative")

    if failed:
        g2 = (DANGER, "FAILED", ", ".join(failed[:2]))
    elif held:
        g2 = (SAFE, "HELD", ", ".join(held[:2]))
    else:
        g2 = (INK_3, "NOT STATED", "narrative does not describe the control")

    if exposed:
        g3 = (DANGER, "IN LINE OF FIRE", ", ".join(exposed[:2]))
    elif ruled_out:
        g3 = (SAFE, "CLEAR", ", ".join(ruled_out[:2]))
    else:
        g3 = (INK_3, "NOT STATED", "narrative does not place a person")

    open_gates = sum(1 for c, _, _ in (g1, g2, g3) if c == DANGER)

    W, H = 900, 168
    gw, gap = 246, 42
    xs = [30, 30 + gw + gap, 30 + 2 * (gw + gap)]
    gy = 46

    def esc(s: str) -> str:
        return (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))[:52]

    blocks = ""
    for x, (colour, state, detail), name in zip(
        xs, (g1, g2, g3), ("ENERGY", "CONTROL", "EXPOSURE")
    ):
        openg = colour == DANGER
        blocks += f"""
  <rect x="{x}" y="{gy}" width="{gw}" height="86" fill="{PANEL}"
        stroke="{colour}" stroke-width="{2 if openg else 1}"
        stroke-dasharray="{'0' if openg else '4 3'}"/>
  <rect x="{x}" y="{gy}" width="{gw}" height="4" fill="{colour}"/>
  <text x="{x+16}" y="{gy+30}" font-family="Archivo" font-size="11"
        font-weight="600" letter-spacing="0.06em" fill="{INK_3}">{name}</text>
  <text x="{x+16}" y="{gy+52}" font-family="Archivo" font-size="14.5"
        font-weight="700" fill="{colour}">{state}</text>
  <text x="{x+16}" y="{gy+71}" font-family="Archivo" font-size="11.5"
        fill="{INK_2}">{esc(detail)}</text>"""

    arrows = ""
    for i in (0, 1):
        ax = xs[i] + gw
        lit = (g1, g2, g3)[i][0] == DANGER
        col = DANGER if lit else RULE_STRONG
        arrows += (
            f"<line x1='{ax+8}' y1='{gy+43}' x2='{ax+gap-14}' y2='{gy+43}' "
            f"stroke='{col}' stroke-width='{2 if lit else 1}'/>"
            f"<polygon points='{ax+gap-14},{gy+43} {ax+gap-21},{gy+39} "
            f"{ax+gap-21},{gy+47}' fill='{col}'/>"
        )

    if open_gates == 3:
        caption = "All three gates open — this is a SIF precursor."
        cap_col = DANGER
    elif open_gates == 2:
        caption = "Two of three gates open — partial precursor pattern."
        cap_col = WARNING
    else:
        caption = f"{open_gates} of 3 gates open — the chain is broken, so harm was not available."
        cap_col = SAFE

    _h(
        f"""<svg viewBox="0 0 {W} {H}" width="100%" role="img"
     aria-label="Barrier chain: energy {g1[1]}, control {g2[1]}, exposure {g3[1]}">
  <text x="30" y="26" font-family="Archivo" font-size="12" font-weight="600"
        fill="{INK}">Hazard released</text>
  <text x="{xs[2]+gw}" y="26" text-anchor="end" font-family="Archivo" font-size="12"
        font-weight="600" fill="{INK}">Person harmed</text>
  {blocks}{arrows}
  <text x="30" y="{gy+112}" font-family="Archivo" font-size="12.5"
        font-weight="600" fill="{cap_col}">{caption}</text>
</svg>"""
    )


def chips(assessment: dict) -> None:
    """Every phrase that moved the score, labelled by what role it played."""
    rows = []
    for src, hits in (assessment.get("energy_sources") or {}).items():
        rows.append(("c-energy", "ENERGY", f"{src} — {', '.join(hits[:3])}"))
    for b in (assessment.get("barrier_failures") or [])[:6]:
        rows.append(("c-barrier", "CONTROL FAILED", b))
    for e in (assessment.get("exposure_evidence") or [])[:4]:
        rows.append(("c-exposure", "EXPOSURE", e))
    for hgood in (assessment.get("barrier_intact_evidence") or [])[:3]:
        rows.append(("c-held", "CONTROL HELD", hgood))

    if not rows:
        _h("<p style='font-size:12.5px;color:var(--ink3);margin:0'>"
           "No structural evidence extracted from this narrative.</p>")
        return

    html = "".join(
        f"<span class='chip {c}'><span class='dot'></span>"
        f"<span class='lab'>{lab}</span><span class='val'>{val}</span></span>"
        for c, lab, val in rows
    )
    _h(f"<div class='chiprow'>{html}</div>")


def rule_rows(rules: List[dict]) -> None:
    if not rules:
        _h("<p style='font-size:12.5px;color:var(--ink3);margin:0'>"
           "No Life-Saving Rule matched above threshold.</p>")
        return
    html = ""
    for r in rules:
        conf = r.get("confidence", 0)
        terms = ", ".join(r.get("matched_terms", [])[:4])
        basis = "keyword and meaning" if r.get("matched_terms") else "meaning only"
        html += (
            f"<div class='rulerow'><div class='top'>"
            f"<span class='nm'>{r['rule']}</span>"
            f"<span class='cf'>{conf:.2f}</span></div>"
            f"<div class='bar'><span style='width:{conf*100:.0f}%'></span></div>"
            f"<div class='terms'>{basis}{' · ' + terms if terms else ''}</div></div>"
        )
    _h(html)


def rank_bars(rows: List[dict], limit: int = 8, unit: str = "reports") -> None:
    """Ranked density list. Colour follows the same setpoints as the gauge."""
    if not rows:
        _h("<p style='font-size:12.5px;color:var(--ink3);margin:0'>Nothing to rank yet.</p>")
        return
    html = ""
    for r in rows[:limit]:
        d = r.get("sif_density", 0)
        colour = DANGER if d >= 0.5 else (WARNING if d >= 0.25 else MANDATE)
        html += (
            f"<div class='rank'><div>"
            f"<div class='nm'>{r['name']}</div>"
            f"<div class='track'><div class='fill' style='width:{max(d*100,1.2):.1f}%;"
            f"background:{colour}'></div></div>"
            f"<div class='meta'>{r['sif_flagged']} flagged of {r['total_reports']} {unit}"
            f" · peak {r.get('max_sif_probability',0):.2f}</div>"
            f"</div><div class='pct' style='color:{colour}'>{d*100:.0f}%</div></div>"
        )
    _h(html)


def distribution(probs: List[float]) -> None:
    """Where a batch falls across the scale, with the setpoints drawn in."""
    if not probs:
        return
    bins = [0] * 20
    for p in probs:
        bins[min(19, int(p * 20))] += 1
    peak = max(bins) or 1

    W, H = 560, 120
    x0, bw = 8, (W - 16) / 20
    base = 92

    bars = ""
    for i, c in enumerate(bins):
        mid = (i + 0.5) / 20
        colour = DANGER if mid >= T_ESCALATE else (WARNING if mid >= T_REVIEW else SAFE)
        h = (c / peak) * 72
        bars += (f"<rect x='{x0 + i*bw + 1:.1f}' y='{base - h:.1f}' "
                 f"width='{bw - 2:.1f}' height='{max(h, 0.6):.1f}' fill='{colour}'/>")

    lines = ""
    for v, lab, col in ((T_REVIEW, "review", WARNING), (T_ESCALATE, "escalate", DANGER)):
        x = x0 + (W - 16) * v
        lines += (
            f"<line x1='{x:.1f}' y1='12' x2='{x:.1f}' y2='{base}' stroke='{col}' "
            f"stroke-width='1.5' stroke-dasharray='3 2'/>"
            f"<text x='{x+4:.1f}' y='20' font-family='Archivo' font-size='10.5' "
            f"font-weight='600' fill='{col}'>{lab}</text>"
        )

    _h(
        f"""<svg viewBox="0 0 {W} {H}" width="100%" role="img"
     aria-label="Distribution of fatal-potential scores across the batch">
  {bars}<line x1="{x0}" y1="{base}" x2="{W-8}" y2="{base}" stroke="{RULE_STRONG}"/>
  {lines}
  <text x="{x0}" y="{base+18}" font-family="IBM Plex Mono, monospace" font-size="10"
        fill="{INK_3}">0.00</text>
  <text x="{W-8}" y="{base+18}" text-anchor="end" font-family="IBM Plex Mono, monospace"
        font-size="10" fill="{INK_3}">1.00</text>
</svg>"""
    )
