"""
AI Resume Screener — Streamlit App v4.0
UI ported from Google Stitch output:
  - EB Garamond headlines, Manrope body, Material Symbols
  - Warm terracotta palette (Material Design 3 variant)
  - Bento grid, SVG score ring, layered gradient button
  - All backend: LangChain structured output, security module
"""

import os
import streamlit.components.v1 as components
import streamlit as st
import pandas as pd
import textwrap
from dotenv import load_dotenv

from parser import extract_text_from_pdf
from llm_engine import analyze_resume
from scoring import compute_score
from utils import highlight_keywords, escape_html
from security import (
    validate_api_key, validate_jd, validate_resume_batch,
    sanitize_text_input, sanitize_filename,
    check_rate_limit, record_api_calls, get_rate_usage,
)

load_dotenv(override=True)
_RAW_KEY = os.getenv("OPENAI_API_KEY", "").strip()

st.set_page_config(
    page_title="AI Resume Screener — Precision Analyst",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed",   # we draw our own sidebar via HTML
)

# ─── Inject Stitch fonts + Material Symbols + CSS ────────────────
st.html("""
<link href="https://fonts.googleapis.com/css2?family=EB+Garamond:wght@400;500;700&family=Manrope:wght@400;500;600;700&display=swap" rel="stylesheet"/>
<link href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200&display=swap" rel="stylesheet"/>
<style>
/* ── Stitch design tokens ── */
:root {
  --on-background:              #3a302a;
  --on-surface:                 #3a302a;
  --outline-variant:            #d8d0c8;
  --surface-variant:            #ece6dc;
  --surface-dim:                #dcd6cc;
  --on-primary-container:       #fbe8d8;
  --on-surface-variant:         #605850;
  --error:                      #c0392b;
  --on-primary:                 #ffffff;
  --surface-container-high:     #ece6dc;
  --error-container:            #fce4e0;
  --surface:                    #faf5ee;
  --surface-container-lowest:   #ffffff;
  --primary:                    #c2652a;
  --primary-container:          #e08850;
  --on-primary-fixed:           #401a08;
  --on-error:                   #ffffff;
  --surface-container:          #f2ece4;
  --primary-fixed-dim:          #f0a878;
  --inverse-on-surface:         #faf5ee;
  --on-secondary:               #ffffff;
  --primary-fixed:              #fbe8d8;
  --surface-container-low:      #f6f0e8;
  --on-surface-sub:             #605850;
  --surface-bright:             #faf5ee;
  --surface-container-highest:  #e6e0d6;
  --on-secondary-container:     #605850;
  --surface-tint:               #c2652a;
  --outline:                    #9a9088;
  --tertiary:                   #8c3c3c;
  --on-error-container:         #7a1a10;
  --background:                 #faf5ee;
  --secondary:                  #78706a;
  --on-tertiary-fixed:          #2e1515;
  --tertiary-container:         #d47070;
  --secondary-container:        #eae2da;
  --secondary-fixed:            #eae2da;
  --tertiary-fixed:             #fce0e0;
  --on-tertiary-container:      #3a2020;

  --gradient: linear-gradient(135deg, #005bbf 0%, #1a73e8 100%);
  --shadow-sm:  0 4px 20px rgba(23,28,32,.03);
  --shadow-md:  0 8px 24px rgba(23,28,32,.08);
  --shadow-lg:  0 12px 32px rgba(23,28,32,.12);
  --radius-xl:  1.25rem;
  --radius-lg:  0.75rem;
  --radius-md:  0.5rem;
  --radius-sm:  0.375rem;
  --radius-full:9999px;
}

/* ── Global ── */
html, body, [data-testid="stApp"] {
  background: var(--background) !important;
  font-family: 'Manrope', -apple-system, sans-serif !important;
  color: var(--on-surface) !important;
}
.stApp { background: var(--background) !important; }
.block-container { padding: 0 !important; max-width: 100% !important; }

/* hide streamlit chrome */
#MainMenu, footer, header[data-testid="stHeader"] { display: none !important; }
section[data-testid="stSidebar"] { display: none !important; }

/* Material Symbols */
.material-symbols-outlined {
  font-family: 'Material Symbols Outlined';
  font-style: normal;
  font-size: 20px;
  line-height: 1;
  letter-spacing: normal;
  text-transform: none;
  white-space: nowrap;
  word-wrap: normal;
  direction: ltr;
  -webkit-font-feature-settings: 'liga';
  -webkit-font-smoothing: antialiased;
  font-variation-settings: 'FILL' 0, 'wght' 400, 'GRAD' 0, 'opsz' 24;
  vertical-align: middle;
}
.fill-icon { font-variation-settings: 'FILL' 1, 'wght' 400, 'GRAD' 0, 'opsz' 24; }

/* ── Streamlit widget overrides (within our layout) ── */
.stTextArea textarea {
  background: var(--surface-container-highest) !important;
  border: none !important;
  border-bottom: 2px solid var(--outline-variant) !important;
  border-radius: var(--radius-lg) !important;
  color: var(--on-surface) !important;
  font-family: 'Manrope', sans-serif !important;
  font-size: 0.875rem !important;
  line-height: 1.65 !important;
  padding: 1rem !important;
  transition: border-color .2s !important;
}
.stTextArea textarea:focus {
  border-bottom-color: var(--primary) !important;
  box-shadow: none !important;
}
label { color: var(--on-surface-variant) !important; font-family: 'Manrope', sans-serif !important; font-size: 0.7rem !important; font-weight: 700 !important; letter-spacing: 1.4px !important; text-transform: uppercase !important; }
[data-testid="stFileUploader"] {
  background: var(--surface) !important;
  border: 2px dashed rgba(216,208,200,.5) !important;
  border-radius: var(--radius-lg) !important;
  padding: 1rem !important;
  transition: background .2s, border-color .2s !important;
}
[data-testid="stFileUploader"]:hover {
  background: var(--surface-container-highest) !important;
  border-color: var(--primary) !important;
}
.stButton > button {
  background: var(--gradient) !important;
  color: #fff !important;
  border: none !important;
  border-radius: var(--radius-full) !important;
  padding: 0.85rem 2rem !important;
  font-family: 'EB Garamond', Georgia, serif !important;
  font-weight: 700 !important;
  font-size: 1.1rem !important;
  box-shadow: 0 8px 24px rgba(0,91,191,.25) !important;
  transition: opacity .2s, transform .15s, box-shadow .2s !important;
  width: 100% !important;
}
.stButton > button:hover {
  opacity: .94 !important;
  transform: translateY(-1px) !important;
  box-shadow: 0 12px 28px rgba(0,91,191,.3) !important;
}
.stButton > button:disabled {
  background: var(--surface-container-high) !important;
  color: var(--outline) !important;
  box-shadow: none !important;
  transform: none !important;
}
.stDownloadButton > button {
  background: transparent !important;
  color: var(--primary) !important;
  border: 1.5px solid var(--primary) !important;
  border-radius: var(--radius-full) !important;
  font-weight: 600 !important;
  font-size: 0.85rem !important;
  box-shadow: none !important;
  padding: 0.5rem 1.25rem !important;
}
.stDownloadButton > button:hover {
  background: var(--primary-fixed) !important;
  transform: none !important;
}
.stProgress > div > div {
  background: var(--gradient) !important;
  border-radius: 100px !important;
}
.stProgress > div {
  background: var(--surface-container-high) !important;
  border-radius: 100px !important;
  height: 5px !important;
}
[data-testid="stDataFrame"] {
  border-radius: var(--radius-lg) !important;
  border: 1px solid var(--outline-variant) !important;
  box-shadow: var(--shadow-sm) !important;
  overflow: hidden !important;
}
.streamlit-expanderHeader {
  background: var(--surface-container-lowest) !important;
  border: 1px solid var(--outline-variant) !important;
  border-radius: var(--radius-md) !important;
  color: var(--on-surface) !important;
  font-family: 'EB Garamond', serif !important;
  font-weight: 700 !important;
  font-size: 1.05rem !important;
  box-shadow: var(--shadow-sm) !important;
}
.streamlit-expanderContent {
  background: var(--surface-container-lowest) !important;
  border: 1px solid var(--outline-variant) !important;
  border-top: none !important;
  border-radius: 0 0 var(--radius-md) var(--radius-md) !important;
}
.stMarkdown p { color: var(--on-surface-variant) !important; font-size: .84rem !important; }
h1,h2,h3,h4 {
  font-family: 'EB Garamond', Georgia, serif !important;
  color: var(--on-surface) !important;
}

/* ── Custom HTML components ── */
.stitch-top-bar {
  display: flex; align-items: center; gap: 1rem;
  background: rgba(250,245,238,.85);
  backdrop-filter: blur(12px);
  padding: 1rem 2rem;
  border-bottom: 1px solid var(--outline-variant);
  position: sticky; top: 0; z-index: 100;
}
.stitch-icon-box {
  width: 48px; height: 48px;
  background: var(--surface-container-low);
  border-radius: 14px;
  display: flex; align-items: center; justify-content: center;
  font-size: 1.5rem;
}
.stitch-title {
  font-family: 'EB Garamond', serif;
  font-size: 1.5rem; font-weight: 700;
  color: var(--on-surface); line-height: 1.1;
}
.stitch-sub {
  font-size: .75rem; color: var(--on-surface-variant); margin-top: 2px;
}
.stitch-chips { margin-left: auto; display: flex; gap: 8px; flex-wrap: wrap; }
.stitch-chip {
  display: inline-flex; align-items: center; gap: 6px;
  background: var(--surface-container-high);
  color: var(--on-surface-variant);
  border-radius: var(--radius-full);
  font-size: .7rem; font-weight: 600;
  padding: 4px 12px; white-space: nowrap;
}

/* Bento section wrapper */
.bento-wrap {
  display: flex; gap: 2rem;
  padding: 1.5rem 2rem;
  align-items: flex-start;
}
.bento-left  { flex: 0 0 340px; display: flex; flex-direction: column; gap: 1rem; }
.bento-right { flex: 1; display: flex; flex-direction: column; gap: 1.5rem; }

/* Input card */
.input-card {
  background: var(--surface-container-low);
  border-radius: var(--radius-xl);
  padding: 1.5rem;
}
.field-label {
  font-size: .68rem; font-weight: 700; letter-spacing: 1.5px;
  text-transform: uppercase; color: var(--on-surface-variant);
  margin-bottom: .5rem; display: block;
}

/* Metrics bar */
.metrics-bar { display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem; }
.metric-tile {
  background: var(--surface-container-lowest);
  border-radius: var(--radius-lg);
  padding: 1.1rem 1rem;
  box-shadow: var(--shadow-sm);
}
.metric-tile.accent { border-left: 4px solid var(--tertiary-container); }
.metric-label { font-size: .7rem; font-weight: 500; color: var(--on-surface-variant); margin-bottom: .25rem; }
.metric-value {
  font-family: 'EB Garamond', serif;
  font-size: 2rem; font-weight: 700; color: var(--on-surface); line-height: 1;
}
.metric-unit { font-size: 1.1rem; color: var(--outline); }

/* Candidate detail card */
.cand-card {
  background: var(--surface-container-lowest);
  border-radius: var(--radius-xl);
  box-shadow: var(--shadow-lg);
  overflow: hidden;
}
.cand-card-header {
  padding: 1.5rem;
  border-bottom: 1px solid var(--surface-container-low);
  background: var(--surface-bright);
  display: flex; align-items: flex-start; justify-content: space-between;
}
.cand-card-body { padding: 1.5rem; display: flex; flex-direction: column; gap: 1.25rem; }

/* SVG score ring */
.score-ring-wrap {
  position: relative; width: 60px; height: 60px;
  flex-shrink: 0;
}
.score-ring-wrap svg { transform: rotate(-90deg); }
.score-num {
  position: absolute; inset: 0;
  display: flex; align-items: center; justify-content: center;
  font-family: 'EB Garamond', serif;
  font-size: 1.15rem; font-weight: 700;
}
.score-num.high { color: var(--primary); }
.score-num.mid  { color: #b45309; }
.score-num.low  { color: var(--error); }

/* AI tldr quote */
.tldr-block {
  background: rgba(194,101,42,.06);
  border-left: 3px solid var(--primary);
  border-radius: 0 var(--radius-sm) var(--radius-sm) 0;
  padding: .75rem 1rem;
  font-size: .83rem; line-height: 1.55;
  color: var(--on-surface);
}
.tldr-label { font-weight: 700; color: var(--primary); margin-right: 4px; }
.culture-block {
  background: rgba(140,60,60,.05);
  border-left: 3px solid var(--tertiary);
  border-radius: 0 var(--radius-sm) var(--radius-sm) 0;
  padding: .65rem .9rem;
  font-size: .78rem; color: var(--tertiary);
  line-height: 1.5;
}

/* Skill chips */
.chip-row { display: flex; flex-wrap: wrap; gap: 5px; margin-top: 5px; }
.chip {
  display: inline-flex; align-items: center; gap: 4px;
  padding: 3px 10px; border-radius: var(--radius-sm);
  font-size: .72rem; font-weight: 500;
}
.chip-green  { background: #f0fdf4; color: #166534; }
.chip-red    { background: var(--error-container); color: var(--on-error-container); }
.chip-blue   { background: #eff6ff; color: #1e40af; }
.chip-yellow { background: #fefce8; color: #854d0e; }
.chip-purple { background: #faf5ff; color: #6b21a8; }
.chip-tint   { background: var(--tertiary-fixed); color: var(--on-tertiary-container); font-weight: 700; }
.chip-surface{ background: var(--surface-container-high); color: var(--on-surface-variant); }

/* Section sub-label */
.sub-label {
  font-size: .67rem; font-weight: 700;
  text-transform: uppercase; letter-spacing: 1.3px;
  color: var(--on-surface-variant);
  display: flex; align-items: center; gap: 5px;
  margin-bottom: .35rem;
}
.sub-label .material-symbols-outlined { font-size: 14px; }

/* Interview question list */
.q-item {
  background: var(--surface-container-low);
  border-radius: var(--radius-md);
  padding: .75rem 1rem;
  font-size: .82rem; color: var(--on-surface);
  display: flex; gap: .6rem; line-height: 1.5;
}
.q-num { font-weight: 700; color: var(--secondary); flex-shrink: 0; }

/* Runner-up pool */
.pool-card {
  background: var(--surface-container-low);
  border-radius: var(--radius-xl);
  padding: 1rem;
}
.runner-tile {
  background: var(--surface-container-lowest);
  border-radius: var(--radius-lg);
  padding: .85rem 1rem;
  display: flex; align-items: center; justify-content: space-between;
  transition: box-shadow .2s; margin-bottom: .5rem;
}
.runner-tile:last-child { margin-bottom: 0; }
.runner-tile:hover { box-shadow: var(--shadow-md); }
.runner-avatar {
  width: 40px; height: 40px; border-radius: 50%;
  background: var(--surface-container);
  display: flex; align-items: center; justify-content: center;
  font-family: 'EB Garamond', serif;
  font-weight: 700; color: var(--primary); font-size: .9rem;
  flex-shrink: 0;
}
.runner-name {
  font-family: 'EB Garamond', serif;
  font-weight: 700; font-size: .95rem; color: var(--on-surface);
}
.runner-sub { font-size: .73rem; color: var(--on-surface-variant); }
.dot-row { display: flex; gap: 5px; }
.dot { width: 9px; height: 9px; border-radius: 50%; }
.dot-on  { background: #22c55e; }
.dot-off { background: var(--surface-dim); }

/* Banners */
.banner {
  display: flex; align-items: flex-start; gap: 8px;
  padding: 9px 13px; border-radius: var(--radius-sm);
  font-size: .81rem; line-height: 1.5; margin: 5px 0;
  border: 1px solid transparent;
}
.banner-ok   { background: #f0fdf4; border-color: rgba(22,101,52,.2); color: #166534; }
.banner-warn { background: #fefce8; border-color: rgba(133,77,14,.2); color: #854d0e; }
.banner-err  { background: var(--error-container); border-color: rgba(192,57,43,.2); color: var(--on-error-container); }
.banner-info { background: var(--primary-fixed); border-color: rgba(194,101,42,.25); color: var(--primary); }

/* Progress label */
.prog-label { display: flex; align-items: center; gap: 8px; font-size: .83rem; font-weight: 500; color: var(--primary); padding: 4px 0; }
.prog-dot { width: 8px; height: 8px; border-radius: 50%; background: var(--primary); animation: blink 1.1s ease-in-out infinite; }
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:.2} }

/* Stats row (results) */
.stats-bar { display: grid; grid-template-columns: repeat(5,1fr); gap: .75rem; margin: .5rem 0 1.25rem; }
.stat-tile {
  background: var(--surface-container-lowest);
  border-radius: var(--radius-lg);
  padding: 1rem;
  text-align: center;
  box-shadow: var(--shadow-sm);
  transition: box-shadow .2s, transform .2s;
}
.stat-tile:hover { box-shadow: var(--shadow-md); transform: translateY(-2px); }
.stat-val {
  font-family: 'EB Garamond', serif;
  font-size: 1.9rem; font-weight: 700; line-height: 1; color: var(--on-surface);
}
.stat-key { font-size: .65rem; font-weight: 700; text-transform: uppercase; letter-spacing: 1.2px; color: var(--on-surface-variant); margin-top: 4px; }

/* Resume preview */
.resume-box {
  background: var(--surface-container);
  border-radius: var(--radius-lg);
  padding: 1rem 1.1rem;
  font-family: 'Manrope', monospace;
  font-size: .75rem;
  color: var(--on-surface-variant);
  max-height: 200px; overflow-y: auto;
  line-height: 1.75; white-space: pre-wrap; word-break: break-word;
}
.resume-box::-webkit-scrollbar { width: 6px; }
.resume-box::-webkit-scrollbar-thumb { background: var(--outline-variant); border-radius: 4px; }
mark { background: #fcd34d; color: #451a03; padding: 1px 4px; border-radius: 3px; }

.divider { height: 1px; background: var(--outline-variant); margin: 1.25rem 0; }

/* Empty state */
.empty-state {
  background: var(--surface-container-lowest);
  border-radius: var(--radius-xl);
  box-shadow: var(--shadow-sm);
  text-align: center; padding: 4rem 2rem;
}
.empty-ring {
  width: 72px; height: 72px; border-radius: 50%;
  background: var(--surface-container-low);
  display: flex; align-items: center; justify-content: center;
  font-size: 2rem; margin: 0 auto 1.1rem;
}
.empty-title { font-family: 'EB Garamond', serif; font-size: 1.3rem; font-weight: 700; color: var(--on-surface); }
.empty-sub { font-size: .83rem; color: var(--on-surface-variant); margin-top: .4rem; line-height: 1.6; }

/* Footer */
.footer { text-align: center; font-size: .7rem; color: var(--outline); padding: 1.5rem 2rem; border-top: 1px solid var(--outline-variant); }

/* Word count hint */
.wc-hint { font-size: .72rem; margin-top: 4px; padding: 4px 8px; border-radius: var(--radius-sm); display: inline-block; }
.wc-ok   { background: #f0fdf4; color: #166534; }
.wc-warn { background: #fefce8; color: #854d0e; }

/* Analysis section label */
.section-hed {
  font-family: 'EB Garamond', serif;
  font-size: 1.2rem; font-weight: 700; color: var(--on-surface);
  margin: 0 0 .2rem;
}
.section-sub { font-size: .73rem; color: var(--on-surface-variant); margin: 0; }

/* Error type pill */
.err-pill { 
  display: inline-block; font-size: .63rem; font-weight: 700; letter-spacing: .8px;
  text-transform: uppercase; padding: 2px 8px; border-radius: var(--radius-full);
  background: var(--error-container); color: var(--on-error-container);
}

@media (max-width: 900px) {
  .bento-wrap { flex-direction: column; }
  .bento-left { flex: none; width: 100%; }
  .metrics-bar { grid-template-columns: repeat(2,1fr); }
  .stats-bar   { grid-template-columns: repeat(3,1fr); }
  .stitch-chips { display: none; }
}
</style>
""")


# ───────────────────────────────────────────────────────────────
# TOP APP BAR
# ───────────────────────────────────────────────────────────────
api_key_valid, api_key_err = validate_api_key(_RAW_KEY)
used_calls, max_calls = get_rate_usage()

st.markdown("""
<div class="stitch-top-bar">
  <div class="stitch-icon-box">🎯</div>
  <div>
    <div class="stitch-title">Precision Analyst</div>
    <div class="stitch-sub">AI Talent Intelligence · LangChain · GPT-4o mini</div>
  </div>
  <div class="stitch-chips">
    <span class="stitch-chip"><span class="material-symbols-outlined" style="font-size:13px">lock</span> Security Hardened</span>
    <span class="stitch-chip"><span class="material-symbols-outlined" style="font-size:13px">psychology</span> Structured AI</span>
    <span class="stitch-chip"><span class="material-symbols-outlined" style="font-size:13px">rule</span> 5-Factor Score</span>
    <span class="stitch-chip"><span class="material-symbols-outlined" style="font-size:13px">csv</span> CSV Export</span>
  </div>
</div>
""", unsafe_allow_html=True)


# ───────────────────────────────────────────────────────────────
# BENTO LAYOUT — left inputs, right results
# ───────────────────────────────────────────────────────────────
st.markdown('<div class="bento-wrap">', unsafe_allow_html=True)

# ── LEFT COLUMN (inputs) rendered via a hidden streamlit col ──
# We use a 1-col layout trick: render inputs inline, wrap in bento-left div
st.markdown("""
<div class="bento-left" id="bento-left-anchor"></div>
""", unsafe_allow_html=True)

# We need to use st.columns to place widgets, then style with CSS
# Streamlit can't nest elements inside custom HTML divs directly,
# so we use a single-column layout and use margin to simulate the bento grid

st.markdown('</div>', unsafe_allow_html=True)  # close bento-wrap

# ── Actual layout using st.columns ──
st.markdown('<div style="padding: 0 2rem;">', unsafe_allow_html=True)

col_left, col_right = st.columns([5, 7], gap="large")

with col_left:
    # Page title inline (matches Stitch page header)
    st.markdown("""
    <div style="margin-bottom:1.25rem;">
      <div style="display:flex;align-items:center;gap:.75rem;margin-bottom:.6rem;">
        <div style="width:44px;height:44px;background:var(--surface-container-low);border-radius:14px;display:flex;align-items:center;justify-content:center;font-size:1.4rem;">🎯</div>
        <div>
          <div style="font-family:'EB Garamond',serif;font-size:1.5rem;font-weight:700;color:var(--on-surface);line-height:1.1;">AI Resume Screener</div>
          <div style="font-size:.74rem;color:var(--on-surface-variant);margin-top:1px;">Configure criteria and ingest candidate documentation for automated scoring.</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Input card
    st.markdown('<div class="input-card">', unsafe_allow_html=True)

    st.markdown('<span class="field-label">Job Description</span>', unsafe_allow_html=True)
    jd_raw = st.text_area(
        "jd",
        height=180,
        placeholder="Paste the comprehensive job description, including required skills, years of experience, and cultural fit attributes here...",
        label_visibility="collapsed",
        key="jd_input",
    )
    jd_valid, jd_err = validate_jd(jd_raw)
    if jd_raw.strip():
        wc = len(jd_raw.split())
        if jd_valid:
            cls = "wc-ok" if wc >= 50 else "wc-warn"
            note = f"✓ {wc} words — good detail" if wc >= 50 else f"⚠ {wc} words — add more detail"
        else:
            cls = "wc-warn"
            note = f"⚠ {jd_err}"
        st.markdown(f'<span class="wc-hint {cls}">{note}</span>', unsafe_allow_html=True)

    st.markdown('<div style="height:.75rem;"></div>', unsafe_allow_html=True)
    st.markdown('<span class="field-label">Upload Resumes</span>', unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        "resumes",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
        key="resume_upload",
    )

    if uploaded_files:
        valid_files, file_errors = validate_resume_batch(uploaded_files)
        n_valid = len(valid_files)
        for fe in file_errors:
            st.markdown(f'<div class="banner banner-warn">⚠ {escape_html(fe)}</div>', unsafe_allow_html=True)
        if n_valid == 0:
            st.markdown('<div class="banner banner-err">✗ No valid PDFs to analyze.</div>', unsafe_allow_html=True)
        elif n_valid == 1:
            st.markdown('<div class="banner banner-info">📄 1 resume ready</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="banner banner-ok">✓ {n_valid} resumes validated</div>', unsafe_allow_html=True)
        for f in valid_files:
            sname = sanitize_filename(f.name)
            kb = round(f.size / 1024, 1) if hasattr(f, "size") else "?"
            st.markdown(f'<div style="font-size:.72rem;color:var(--on-surface-variant);padding:2px 0;">📄 {escape_html(sname)} <span style="color:var(--outline);">({kb} KB)</span></div>', unsafe_allow_html=True)
    else:
        valid_files, n_valid = [], 0

    st.markdown('</div>', unsafe_allow_html=True)  # /input-card

    # Analyze button
    st.markdown('<div style="margin-top:1rem;">', unsafe_allow_html=True)
    rate_ok, rate_err_msg = check_rate_limit(n_valid)
    ready = jd_valid and n_valid >= 1 and rate_ok
    analyze_clicked = st.button(
        "⬤  Analyze Resumes",
        disabled=not ready,
        use_container_width=True,
    )
    if not jd_raw.strip():
        st.markdown('<div style="font-size:.72rem;color:var(--outline);text-align:center;margin-top:.35rem;">Paste a job description to continue</div>', unsafe_allow_html=True)
    elif not ready and rate_ok and jd_valid:
        st.markdown('<div style="font-size:.72rem;color:var(--outline);text-align:center;margin-top:.35rem;">Upload at least 1 PDF resume</div>', unsafe_allow_html=True)
    elif not rate_ok:
        st.markdown(f'<div style="font-size:.72rem;color:var(--error);text-align:center;margin-top:.35rem;">{escape_html(rate_err_msg)}</div>', unsafe_allow_html=True)
    else:
        est = n_valid * 0.001
        st.markdown(f'<div style="font-size:.72rem;color:var(--outline);text-align:center;margin-top:.35rem;">Processing ~{n_valid} doc{"s" if n_valid>1 else ""} · est. &lt;${est:.3f}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # API key status (below button)
    st.markdown('<div style="margin-top:1rem;">', unsafe_allow_html=True)
    if api_key_valid:
        masked = _RAW_KEY[:8] + "·····" + _RAW_KEY[-4:]
        st.markdown(f'<div class="banner banner-ok" style="font-size:.74rem;">✓ <strong>OpenRouter key loaded</strong> · <code style="font-size:.7rem;">{masked}</code></div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="banner banner-err" style="font-size:.74rem;">✗ <strong>Key missing:</strong> {escape_html(api_key_err)}</div>', unsafe_allow_html=True)

    # Rate usage mini-bar
    rate_pct = int((used_calls / max_calls) * 100)
    bar_col = "#c2652a" if rate_pct < 50 else "#b45309" if rate_pct < 80 else "#c0392b"
    st.markdown(f"""
    <div style="margin-top:.5rem; font-size:.71rem; color:var(--on-surface-variant);">
      Session usage: <strong style="color:var(--on-surface);">{used_calls}/{max_calls}</strong> analyses
      <div style="height:4px;background:var(--surface-container-high);border-radius:100px;margin-top:5px;">
        <div style="width:{rate_pct}%;height:100%;border-radius:100px;background:{bar_col};transition:width .4s;"></div>
      </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Scoring weights (mini card)
    st.markdown("""
    <div style="margin-top:1.25rem; background:var(--surface-container-low); border-radius:var(--radius-lg); padding:1rem;">
      <div class="sub-label" style="margin-bottom:.6rem;">Scoring Weights</div>
      <div style="display:grid; grid-template-columns:1fr auto; gap:2px 12px; font-size:.75rem;">
        <span style="color:var(--on-surface-variant);">Skills match</span>     <span style="font-weight:700;color:#166534;">35%</span>
        <span style="color:var(--on-surface-variant);">Experience</span>        <span style="font-weight:700;color:#1e40af;">30%</span>
        <span style="color:var(--on-surface-variant);">Keywords</span>          <span style="font-weight:700;color:#6b21a8;">20%</span>
        <span style="color:var(--on-surface-variant);">Gap penalty</span>       <span style="font-weight:700;color:#9b1c1c;">10%</span>
        <span style="color:var(--on-surface-variant);">Seniority</span>         <span style="font-weight:700;color:#134e4a;">5%</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Clear results
    if "results" in st.session_state and st.session_state["results"]:
        st.markdown('<div style="margin-top:.75rem;">', unsafe_allow_html=True)
        if st.button("🗑  Clear Results", use_container_width=True):
            del st.session_state["results"]
            st.session_state.pop("analysis_errors", None)
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# RIGHT COLUMN — analysis pipeline + results
# ─────────────────────────────────────────────────────────────
with col_right:

    # ── API gate ──
    if not api_key_valid:
        st.markdown(f"""
        <div class="banner banner-err">
          <span class="material-symbols-outlined" style="font-size:18px;flex-shrink:0;">error</span>
          <div><strong>OpenRouter API Key Invalid</strong><br>{escape_html(api_key_err)}<br>
          Add <code>OPENAI_API_KEY=sk-or-v1-...</code> to your <code>.env</code> file and restart.</div>
        </div>
        """, unsafe_allow_html=True)

    # ── Run pipeline ──
    if analyze_clicked and ready and api_key_valid:
        results = []
        analysis_errors = []
        successful_calls = 0

        prog_bar  = st.progress(0)
        status_ph = st.empty()
        jd_clean  = sanitize_text_input(jd_raw, max_chars=8000)

        for idx, pdf_file in enumerate(valid_files):
            sname = sanitize_filename(pdf_file.name)
            prog_bar.progress(idx / n_valid)
            status_ph.markdown(
                f'<div class="prog-label"><span class="prog-dot"></span> Analyzing {idx+1}/{n_valid}: <strong>{escape_html(sname)}</strong></div>',
                unsafe_allow_html=True,
            )

            _, resume_raw = extract_text_from_pdf(pdf_file)
            resume_clean = sanitize_text_input(resume_raw, max_chars=5000) if resume_raw else ""
            parse_failed = not bool(resume_clean.strip())

            if parse_failed:
                analysis_errors.append(f"**{sname}**: could not extract text — image-based or encrypted PDF.")
                result_dict, llm_ok, err_type = {}, False, "parse"
                score, rec = 0, "❌ Parse Error"
            else:
                result = analyze_resume(jd_clean, resume_clean, _RAW_KEY)
                if result.success:
                    successful_calls += 1
                else:
                    labels = {"auth":"Authentication failed","quota":"Quota/rate limit","timeout":"Request timed out","network":"Network error","parse":"Output error","unknown":"Unexpected error"}
                    analysis_errors.append(f"**{sname}** [{labels.get(result.error_type,'Error')}]: {result.error_message}")
                result_dict = result.analysis
                score, rec = compute_score(result_dict)
                llm_ok, err_type = result.success, result.error_type

            candidate_name = (
                sname.replace(".pdf","").replace(".PDF","")
                .replace("_"," ").replace("-"," ").replace("  "," ").strip().title()
            )
            results.append({
                "candidate_name": candidate_name, "filename": sname,
                "score": score, "recommendation": rec,
                "analysis": result_dict, "resume_text": resume_clean,
                "parse_error": parse_failed, "llm_success": llm_ok, "error_type": err_type,
            })

        prog_bar.progress(1.0)
        status_ph.markdown(
            f'<div class="prog-label">✓ Done — {successful_calls}/{n_valid} analyzed successfully</div>',
            unsafe_allow_html=True,
        )
        record_api_calls(successful_calls)
        results.sort(key=lambda r: r["score"], reverse=True)
        st.session_state["results"] = results
        st.session_state["analysis_errors"] = analysis_errors

    # ── Show results ──
    if "results" in st.session_state and st.session_state["results"]:
        results = st.session_state["results"]
        errors  = st.session_state.get("analysis_errors", [])

        for err in errors:
            st.markdown(f'<div class="banner banner-warn">⚠ {err}</div>', unsafe_allow_html=True)

        # Metrics bar (Stitch style — 4 tiles)
        ok_r   = [r for r in results if r["llm_success"]]
        avg_sc = round(sum(r["score"] for r in ok_r) / len(ok_r)) if ok_r else 0
        strong = sum(1 for r in ok_r if r["score"] >= 70)
        kw_miss_avg = round(sum(len(r["analysis"].get("keywords_missing") or []) for r in ok_r) / len(ok_r)) if ok_r else 0

        st.markdown(f"""
        <div class="metrics-bar">
          <div class="metric-tile">
            <div class="metric-label">Total Scanned</div>
            <div class="metric-value">{len(results)}</div>
          </div>
          <div class="metric-tile accent">
            <div class="metric-label">Strong Fit (&ge;70)</div>
            <div class="metric-value">{strong}</div>
          </div>
          <div class="metric-tile">
            <div class="metric-label">Avg. Match Score</div>
            <div class="metric-value">{avg_sc}<span class="metric-unit">%</span></div>
          </div>
          <div class="metric-tile">
            <div class="metric-label">Avg Missing KW</div>
            <div class="metric-value">{kw_miss_avg}<span class="metric-unit"></span></div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Top candidate deep-dive (Stitch hero card)
        top = results[0]
        ta  = top["analysis"]
        top_score = top["score"]
        dash_val  = int(top_score * 100 / 100)  # already 0-100
        sc_col = "var(--primary)" if top_score >= 70 else ("#b45309" if top_score >= 45 else "var(--error)")
        sc_track_col = "var(--primary-fixed-dim)"

        tldr    = ta.get("tldr", "") or ""
        culture = ta.get("culture_fit_notes", "") or ""
        skills  = ta.get("skills_match") or []
        missing = ta.get("missing_skills") or []
        questions = ta.get("interview_questions") or []

        skills_html  = "".join(f'<span class="chip chip-green">✓ {escape_html(s)}</span>' for s in skills[:4])
        missing_html = "".join(f'<span class="chip chip-red">{escape_html(s)}</span>' for s in missing[:4])
        q_html = "".join(f'<div class="q-item"><span class="q-num">Q{i+1}.</span><span style="font-style:italic;">{escape_html(q)}</span></div>' for i, q in enumerate(questions))

        st.html(f"""
        <div class="cand-card">
          <div class="cand-card-header">
            <div style="display:flex;align-items:center;gap:1rem;">
              <div class="score-ring-wrap">
                <svg viewBox="0 0 36 36" style="width:60px;height:60px;">
                  <path fill="none" stroke="{sc_track_col}" stroke-width="3"
                    d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"
                    stroke-dasharray="100,100"/>
                  <path fill="none" stroke="{sc_col}" stroke-width="4" stroke-linecap="round"
                    d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"
                    stroke-dasharray="{top_score},100"
                    style="transform:rotate(-90deg);transform-origin:50% 50%;"/>
                </svg>
                <div class="score-num" style="color:{sc_col};font-size:1.05rem;">{top_score}</div>
              </div>
              <div>
                <div style="font-family:'EB Garamond',serif;font-size:1.25rem;font-weight:700;color:var(--on-surface);">
                  🥇 {escape_html(top["candidate_name"])}
                </div>
                <div style="font-size:.76rem;color:var(--on-surface-variant);margin-top:2px;">
                  {top["recommendation"]} · {ta.get("experience_relevance","—")} Experience · Seniority: {ta.get("seniority_match","—")}
                </div>
              </div>
            </div>
          </div>
          <div class="cand-card-body">
            {"" if not tldr or tldr=="Analysis failed" else f'<div class="tldr-block"><span class="tldr-label">AI Summary:</span>{escape_html(tldr)}</div>'}
            {"" if not culture or culture=="No signals detected." else f'<div class="culture-block">🎭 {escape_html(culture)}</div>'}
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:1.25rem;">
              <div>
                <div class="sub-label"><span class="material-symbols-outlined fill-icon" style="color:#166534;">check_circle</span> Verified Skills</div>
                <div class="chip-row">{skills_html or '<span style="font-size:.75rem;color:var(--outline);">None identified</span>'}</div>
              </div>
              <div>
                <div class="sub-label"><span class="material-symbols-outlined" style="color:var(--error);">warning</span> Gaps</div>
                <div class="chip-row">{missing_html or '<span style="font-size:.75rem;color:var(--outline);">None identified</span>'}</div>
              </div>
            </div>
            {"" if not questions else f'<div style="padding-top:.5rem;border-top:1px dashed var(--outline-variant);"><div class="sub-label" style="margin-bottom:.5rem;"><span class="material-symbols-outlined">forum</span> Suggested Interview Probes</div>{q_html}</div>'}
          </div>
        </div>
        """)

        # Runner-up pool
        if len(results) > 1:
            runners = results[1:]
            runner_tiles = ""
            for i, r in enumerate(runners[:4]):
                rs = r["score"]
                medal = "🥈" if i == 0 else "🥉" if i == 1 else f"#{i+2}"
                exp = r["analysis"].get("experience_relevance","—") if r["analysis"] else "—"
                # dot indicators: skills-match / exp-high / llm-ok
                d1 = "dot-on" if rs >= 70 else "dot-off"
                d2 = "dot-on" if exp == "High" else "dot-off"
                d3 = "dot-on" if r["llm_success"] else "dot-off"
                runner_tiles += textwrap.dedent(f"""
                <div class="runner-tile">
                  <div style="display:flex;align-items:center;gap:.75rem;">
                    <div class="runner-avatar">{rs}</div>
                    <div>
                      <div class="runner-name">{medal} {escape_html(r["candidate_name"])}</div>
                      <div class="runner-sub">{r["recommendation"]} · {exp} Exp</div>
                    </div>
                  </div>
                  <div class="dot-row">
                    <div class="dot {d1}" title="Fit ≥70"></div>
                    <div class="dot {d2}" title="High experience"></div>
                    <div class="dot {d3}" title="Analysis OK"></div>
                  </div>
                </div>""")

            st.html(f"""
            <div class="pool-card" style="margin-top:.5rem;">
              <div style="font-family:'EB Garamond',serif;font-weight:700;font-size:1rem;color:var(--on-surface);padding:.25rem .5rem .75rem;">
                Runner Up Pool
              </div>
              {runner_tiles}
            </div>
            """)

        # Full comparison table + CSV
        st.markdown('<div style="margin-top:1.5rem;">', unsafe_allow_html=True)
        st.markdown('<div class="sub-label" style="margin-bottom:.5rem;">Full Comparison Table</div>', unsafe_allow_html=True)
        rows = []
        for r in results:
            a = r["analysis"]
            rows.append({
                "Rank":        results.index(r)+1,
                "Candidate":   r["candidate_name"],
                "Score":       r["score"],
                "Fit":         r["recommendation"].replace("✅ ","").replace("⚠️ ","").replace("❌ ",""),
                "Experience":  a.get("experience_relevance","—"),
                "Seniority":   a.get("seniority_match","—"),
                "Skills ✓":    len(a.get("skills_match") or []),
                "KW ✓":        len(a.get("keywords_matched") or []),
                "Top Strength":((a.get("strengths") or ["—"])[0])[:55],
            })
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True, column_config={
            "Rank":  st.column_config.NumberColumn("🏅", width="small"),
            "Score": st.column_config.ProgressColumn("Score /100", min_value=0, max_value=100, format="%d"),
            "Skills ✓": st.column_config.NumberColumn("Skills ✓", width="small"),
            "KW ✓":     st.column_config.NumberColumn("KW ✓", width="small"),
        })
        st.download_button("📥  Download CSV", data=df.to_csv(index=False), file_name="screening_results.csv", mime="text/csv")
        st.markdown('</div>', unsafe_allow_html=True)

        # All candidate expanders
        if len(results) > 1:
            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
            st.markdown('<div class="sub-label">All Candidate Profiles</div>', unsafe_allow_html=True)
            for rank, r in enumerate(results, 1):
                sc = r["score"]
                a  = r["analysis"]
                sc_col_ = "sr-high" if sc >= 70 else "sr-mid" if sc >= 45 else "sr-low"

                medal = "🥇" if rank==1 else "🥈" if rank==2 else "🥉" if rank==3 else f"#{rank}"
                with st.expander(f"{medal}  {r['candidate_name']}  ·  {r['recommendation']}  ·  {sc}/100", expanded=False):
                    lc, rc2 = st.columns([1, 3])
                    with lc:
                        sc_c = "var(--primary)" if sc>=70 else ("#b45309" if sc>=45 else "var(--error)")
                        sc_t = "var(--primary-fixed-dim)"
                        st.html(f"""
                        <div style="text-align:center;padding:.5rem 0;">
                          <div style="position:relative;width:64px;height:64px;margin:0 auto 6px;">
                            <svg viewBox="0 0 36 36" style="width:64px;height:64px;transform:rotate(-90deg);">
                              <path fill="none" stroke="{sc_t}" stroke-width="3"
                                d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"
                                stroke-dasharray="100,100"/>
                              <path fill="none" stroke="{sc_c}" stroke-width="4" stroke-linecap="round"
                                d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"
                                stroke-dasharray="{sc},100"
                                style="transform:rotate(-90deg);transform-origin:50% 50%;"/>
                            </svg>
                            <div style="position:absolute;inset:0;display:flex;align-items:center;justify-content:center;font-family:'EB Garamond',serif;font-size:1.1rem;font-weight:700;color:{sc_c};">{sc}</div>
                          </div>
                          <div style="font-size:.65rem;color:var(--outline);margin-bottom:5px;">Fit Score</div>
                          <span class="chip chip-surface">{a.get("experience_relevance","—")} Exp</span>
                          <span class="chip chip-surface">{a.get("seniority_match","—")}</span>
                          {"" if not r.get('parse_error') else '<br><span class="err-pill">parse error</span>'}
                        </div>
                        """)
                        t = a.get("tldr","")
                        if t and t != "Analysis failed":
                            st.markdown(f'<div class="tldr-block" style="font-size:.76rem;">{escape_html(t)}</div>', unsafe_allow_html=True)

                    with rc2:
                        sk  = a.get("skills_match") or []
                        mis = a.get("missing_skills") or []
                        kh  = a.get("keywords_matched") or []
                        km  = a.get("keywords_missing") or []
                        str_ = a.get("strengths") or []
                        gap_ = a.get("gaps") or []
                        qs   = a.get("interview_questions") or []

                        c1, c2 = st.columns(2)
                        with c1:
                            if sk:
                                tags = "".join(f'<span class="chip chip-green">✓ {escape_html(s)}</span>' for s in sk)
                                st.markdown(f'<div class="sub-label">Skills Matched</div><div class="chip-row">{tags}</div>', unsafe_allow_html=True)
                        with c2:
                            if mis:
                                tags = "".join(f'<span class="chip chip-red">{escape_html(s)}</span>' for s in mis)
                                st.markdown(f'<div class="sub-label">Missing Skills</div><div class="chip-row">{tags}</div>', unsafe_allow_html=True)

                        c3, c4 = st.columns(2)
                        with c3:
                            if kh:
                                tags = "".join(f'<span class="chip chip-blue">{escape_html(k)}</span>' for k in kh)
                                st.markdown(f'<div class="sub-label">Keywords Found</div><div class="chip-row">{tags}</div>', unsafe_allow_html=True)
                        with c4:
                            if km:
                                tags = "".join(f'<span class="chip chip-yellow">{escape_html(k)}</span>' for k in km)
                                st.markdown(f'<div class="sub-label">Keywords Missing</div><div class="chip-row">{tags}</div>', unsafe_allow_html=True)

                        c5, c6 = st.columns(2)
                        with c5:
                            if str_:
                                items = "".join(f'<div style="font-size:.8rem;padding:4px 0;color:var(--on-surface-variant);border-bottom:1px solid var(--outline-variant);">✅ {escape_html(s)}</div>' for s in str_)
                                st.markdown(f'<div class="sub-label">Strengths</div>{items}', unsafe_allow_html=True)
                        with c6:
                            if gap_:
                                items = "".join(f'<div style="font-size:.8rem;padding:4px 0;color:var(--on-surface-variant);border-bottom:1px solid var(--outline-variant);">⚠️ {escape_html(g)}</div>' for g in gap_)
                                st.markdown(f'<div class="sub-label">Gaps</div>{items}', unsafe_allow_html=True)

                        if qs:
                            q_h2 = "".join(f'<div class="q-item"><span class="q-num">Q{i+1}.</span><span style="font-style:italic;">{escape_html(q)}</span></div>' for i,q in enumerate(qs))
                            st.markdown(f'<div class="sub-label" style="margin-top:.75rem;color:var(--secondary);">Interview Probes</div>{q_h2}', unsafe_allow_html=True)

                    if r["resume_text"]:
                        st.markdown('<div class="divider" style="margin:.75rem 0;"></div>', unsafe_allow_html=True)
                        st.markdown('<div class="sub-label">Resume Preview</div>', unsafe_allow_html=True)
                        all_kw = list(set((kh or []) + (sk or [])))
                        hl = highlight_keywords(r["resume_text"][:2000], all_kw)
                        st.markdown(f'<div class="resume-box">{hl}</div>', unsafe_allow_html=True)

    else:
        # Empty state
        st.markdown("""
        <div class="empty-state">
          <div class="empty-ring">🎯</div>
          <div class="empty-title">Ready to screen candidates</div>
          <div class="empty-sub">Paste a job description, upload PDF resumes,<br>and click <strong>Analyze Resumes</strong> to get ranked results instantly.</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)  # /padding wrapper

# Footer
st.markdown("""
<div class="footer">
  Precision Analyst · AI Resume Screener v4.1 · LangChain · OpenRouter · pdfplumber · Streamlit
</div>
""", unsafe_allow_html=True)
