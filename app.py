"""
AI Resume Screener — Streamlit App v4.0
UI ported from Google Stitch output:
  - EB Garamond headlines, Manrope body, Material Symbols
  - Warm terracotta palette (Material Design 3 variant)
  - Bento grid, SVG score ring, layered gradient button
  - All backend: LangChain structured output, security module
"""

import io
import os
import re
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

# ─── Wizard state init ─────────────────────────────────────────
_WIZARD_DEFAULTS = {
    "wizard_step": 1,
    "jd_text": "",
    "cached_resumes": [],          # list of {"name", "size", "bytes"}
    "cached_file_errors": [],
    "results": None,
    "analysis_errors": [],
    "active_candidate_id": 0,
}
for _k, _v in _WIZARD_DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

STEP_LABELS = ["Job Description", "Upload Resumes", "Review & Analyze", "Results"]


def _goto(step: int):
    st.session_state["wizard_step"] = max(1, min(4, step))
    st.rerun()


def _reset_to_step1(clear_results: bool = True):
    st.session_state["wizard_step"] = 1
    if clear_results:
        st.session_state["results"] = None
        st.session_state["analysis_errors"] = []

# ─── Inject Stitch fonts + Material Symbols + CSS ────────────────
st.html("""
<link rel="preconnect" href="https://fonts.googleapis.com"/>
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin/>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet"/>
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

  --gradient: linear-gradient(135deg, var(--primary) 0%, var(--primary-container) 100%);
  --shadow-sm:  0 1px 2px rgba(23,28,32,.04), 0 4px 12px rgba(23,28,32,.04);
  --shadow-md:  0 2px 6px rgba(23,28,32,.05), 0 12px 28px rgba(23,28,32,.08);
  --shadow-lg:  0 4px 12px rgba(23,28,32,.06), 0 20px 40px rgba(23,28,32,.10);
  --radius-xl:  1.25rem;
  --radius-lg:  0.875rem;
  --radius-md:  0.5rem;
  --radius-sm:  0.375rem;
  --radius-full:9999px;

  /* Apple system font stack — uses true SF Pro on Apple devices, Inter elsewhere */
  --font-display: -apple-system, BlinkMacSystemFont, "SF Pro Display", "SF Pro Text", "Inter", system-ui, "Segoe UI", Roboto, sans-serif;
  --font-text:    -apple-system, BlinkMacSystemFont, "SF Pro Text", "Inter", system-ui, "Segoe UI", Roboto, sans-serif;
  --font-mono:    ui-monospace, "SF Mono", Menlo, Consolas, "Roboto Mono", monospace;
}

/* ── Global ── */
html, body, [data-testid="stApp"] {
  background: var(--background) !important;
  font-family: var(--font-text) !important;
  color: var(--on-surface) !important;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  text-rendering: optimizeLegibility;
  font-feature-settings: "ss01", "cv11";   /* SF Pro / Inter stylistic alternates */
  letter-spacing: -0.01em;
}
.stApp { background: var(--background) !important; }
.block-container {
  padding: 0 1.5rem 4rem !important;
  max-width: 880px !important;
  margin: 0 auto !important;
}

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
  background: #ffffff !important;
  border: 1.5px solid var(--outline-variant) !important;
  border-radius: var(--radius-md) !important;
  color: var(--on-surface) !important;
  font-family: var(--font-text) !important;
  font-size: 0.9rem !important;
  line-height: 1.7 !important;
  padding: 1rem 1.1rem !important;
  transition: border-color .18s, box-shadow .18s !important;
  box-shadow: 0 1px 3px rgba(0,0,0,.06) inset !important;
}
.stTextArea textarea::placeholder {
  color: var(--outline) !important;
  opacity: 1 !important;
}
.stTextArea textarea:focus {
  border-color: var(--primary) !important;
  box-shadow: 0 0 0 3px rgba(194,101,42,.14) !important;
  outline: none !important;
}
label { color: var(--on-surface-variant) !important; font-family: var(--font-text) !important; font-size: 0.7rem !important; font-weight: 700 !important; letter-spacing: 1.4px !important; text-transform: uppercase !important; }
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
  font-family: var(--font-display) !important;
  font-weight: 700 !important;
  font-size: 1.1rem !important;
  box-shadow: 0 8px 24px rgba(194,101,42,.25) !important;
  transition: opacity .2s, transform .15s, box-shadow .2s !important;
  width: 100% !important;
}
.stButton > button:hover {
  opacity: .94 !important;
  transform: translateY(-1px) !important;
  box-shadow: 0 12px 28px rgba(194,101,42,.3) !important;
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
  font-family: var(--font-display) !important;
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
  font-family: var(--font-display) !important;
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
  font-family: var(--font-display);
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
  font-family: var(--font-display);
  font-size: 2rem; font-weight: 700; color: var(--on-surface); line-height: 1;
  letter-spacing: -0.025em;
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
  font-family: var(--font-display);
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
  font-family: var(--font-display);
  font-weight: 700; color: var(--primary); font-size: .9rem;
  flex-shrink: 0;
}
.runner-name {
  font-family: var(--font-display);
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
  font-family: var(--font-display);
  font-size: 1.9rem; font-weight: 700; line-height: 1; color: var(--on-surface);
  letter-spacing: -0.025em;
}
.stat-key { font-size: .65rem; font-weight: 700; text-transform: uppercase; letter-spacing: 1.2px; color: var(--on-surface-variant); margin-top: 4px; }

/* Resume preview */
.resume-box {
  background: var(--surface-container);
  border-radius: var(--radius-lg);
  padding: 1rem 1.1rem;
  font-family: var(--font-mono);
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
.empty-title { font-family: var(--font-display); font-size: 1.3rem; font-weight: 700; color: var(--on-surface); }
.empty-sub { font-size: .83rem; color: var(--on-surface-variant); margin-top: .4rem; line-height: 1.6; }

/* Footer */
.footer { text-align: center; font-size: .7rem; color: var(--outline); padding: 1.5rem 2rem; border-top: 1px solid var(--outline-variant); }

/* Word count hint */
.wc-hint { font-size: .72rem; margin-top: 4px; padding: 4px 8px; border-radius: var(--radius-sm); display: inline-block; }
.wc-ok   { background: #f0fdf4; color: #166534; }
.wc-warn { background: #fefce8; color: #854d0e; }

/* Analysis section label */
.section-hed {
  font-family: var(--font-display);
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

/* ── Wizard shell ── */
.wizard-shell {
  margin: 1.5rem 0 3rem;
  padding: 0;
}
.wizard-shell.is-results { margin: 1.5rem 0; }

/* Slim top bar — iOS frosted-glass style */
.top-bar {
  display: flex; align-items: center; gap: .85rem;
  background: rgba(250,245,238,.78);
  backdrop-filter: saturate(180%) blur(24px);
  -webkit-backdrop-filter: saturate(180%) blur(24px);
  padding: .85rem 1.5rem;
  border-bottom: 0.5px solid var(--outline-variant);
  position: sticky; top: 0; z-index: 100;
}
.top-bar-icon {
  width: 38px; height: 38px;
  background: var(--primary-fixed);
  border-radius: 11px;
  display: flex; align-items: center; justify-content: center;
  font-size: 1.1rem;
}
.top-bar-title {
  font-family: var(--font-display);
  font-size: 1.15rem; font-weight: 700;
  color: var(--on-surface); line-height: 1.05;
  letter-spacing: -0.022em;
}
.top-bar-sub {
  font-size: .68rem; color: var(--on-surface-variant); margin-top: 2px;
}
.top-bar-status {
  margin-left: auto;
  display: flex; gap: .4rem; align-items: center; flex-wrap: wrap;
}
.tb-pill {
  display: inline-flex; align-items: center; gap: 5px;
  font-size: .68rem; font-weight: 700;
  letter-spacing: .8px; text-transform: uppercase;
  padding: 4px 10px; border-radius: var(--radius-full);
  border: 1px solid transparent;
}
.tb-pill.ok   { background: #f0fdf4; color: #166534; border-color: rgba(22,101,52,.15); }
.tb-pill.err  { background: var(--error-container); color: var(--on-error-container); border-color: rgba(192,57,43,.2); }
.tb-pill.info { background: var(--surface-container-high); color: var(--on-surface-variant); }

/* Stepper */
.stepper {
  display: flex; align-items: flex-start; justify-content: center;
  gap: 0; margin: 1.25rem 0 1.5rem;
  padding: 0 .5rem;
}
.step-node {
  display: flex; flex-direction: column;
  align-items: center; gap: 6px;
  flex: 0 0 auto; min-width: 96px;
}
.step-node-disc {
  width: 30px; height: 30px;
  border-radius: 50%;
  display: flex; align-items: center; justify-content: center;
  background: var(--surface-container-lowest);
  border: 1.5px solid var(--outline-variant);
  font-family: var(--font-display);
  font-weight: 700; font-size: .9rem;
  color: var(--on-surface-variant);
  transition: all .25s ease;
}
.step-node.is-active .step-node-disc {
  background: var(--primary); border-color: var(--primary);
  color: var(--on-primary);
  box-shadow: 0 0 0 5px rgba(194,101,42,.16);
}
.step-node.is-done .step-node-disc {
  background: var(--primary); border-color: var(--primary);
  color: var(--on-primary);
}
.step-node-label {
  font-size: .67rem; font-weight: 700;
  text-transform: uppercase; letter-spacing: 1.2px;
  color: var(--on-surface-variant);
  white-space: nowrap;
}
.step-node.is-active .step-node-label,
.step-node.is-done   .step-node-label { color: var(--on-surface); }
.step-connector {
  flex: 1 1 auto; height: 2px;
  background: var(--outline-variant);
  margin: 15px -8px 0;
  max-width: 70px;
  transition: background .25s ease;
}
.step-connector.is-done { background: var(--primary); }

/* Step card */
.step-card {
  background: var(--surface-container-lowest);
  border: 1px solid var(--outline-variant);
  border-radius: var(--radius-xl);
  box-shadow: var(--shadow-sm);
  padding: 2.25rem 2.25rem 1.5rem;
  animation: fadeIn .25s ease-out;
}
@keyframes fadeIn {
  from { opacity: 0; transform: translateY(8px); }
  to   { opacity: 1; transform: none; }
}
.step-badge {
  display: inline-block;
  font-size: .65rem; font-weight: 700;
  letter-spacing: 1.4px;
  text-transform: uppercase;
  color: var(--primary);
  background: var(--primary-fixed);
  padding: 4px 10px;
  border-radius: var(--radius-full);
  margin-bottom: .9rem;
}
.step-title {
  font-family: var(--font-display);
  font-size: 1.85rem;
  font-weight: 700;
  color: var(--on-surface);
  line-height: 1.12;
  letter-spacing: -0.024em;
  margin: 0 0 .35rem;
}
.step-sub {
  font-size: .9rem;
  color: var(--on-surface-variant);
  line-height: 1.55;
  margin: 0 0 1.5rem;
}
.step-divider {
  height: 1px;
  background: var(--outline-variant);
  margin: 1.5rem 0 1.25rem;
}

/* Secondary (ghost) button — used for Back, Re-run, Edit inputs */
.stButton > button[kind="secondary"] {
  background: transparent !important;
  color: var(--on-surface-variant) !important;
  border: 1.5px solid var(--outline-variant) !important;
  box-shadow: none !important;
  font-family: var(--font-text) !important;
  font-weight: 600 !important;
  font-size: .9rem !important;
  padding: 0.7rem 1.5rem !important;
}
.stButton > button[kind="secondary"]:hover {
  background: var(--surface-container-low) !important;
  border-color: var(--primary) !important;
  color: var(--primary) !important;
  transform: none !important;
  box-shadow: none !important;
}

/* Step 3 review tiles */
.review-row {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 1rem;
  margin-bottom: 1.25rem;
}
.review-tile {
  background: var(--surface-container-low);
  border-radius: var(--radius-lg);
  padding: 1rem 1.1rem;
}
.review-tile-label {
  font-size: .65rem; font-weight: 700;
  text-transform: uppercase; letter-spacing: 1.3px;
  color: var(--on-surface-variant);
  margin-bottom: .4rem;
}
.review-tile-value {
  font-family: var(--font-display);
  font-weight: 700;
  font-size: 1.15rem;
  color: var(--on-surface);
  line-height: 1.25;
}
.review-jd {
  background: var(--surface-container-low);
  border-radius: var(--radius-lg);
  padding: 1rem 1.1rem;
  font-size: .82rem;
  line-height: 1.55;
  color: var(--on-surface-variant);
  max-height: 140px; overflow-y: auto;
  border-left: 3px solid var(--primary);
  margin-bottom: 1.25rem;
}

/* File pill list (step 2 & 3) */
.file-pill-row { display: flex; flex-wrap: wrap; gap: .5rem; margin-top: .5rem; }
.file-pill {
  display: inline-flex; align-items: center; gap: 8px;
  background: var(--surface-container-low);
  border: 1px solid var(--outline-variant);
  border-radius: var(--radius-full);
  padding: 5px 12px;
  font-size: .76rem;
  color: var(--on-surface);
}
.file-pill-size { color: var(--outline); font-size: .7rem; }

/* Results header bar (step 4) */
.results-header {
  display: flex; align-items: center; justify-content: space-between;
  background: var(--surface-container-lowest);
  border: 1px solid var(--outline-variant);
  border-radius: var(--radius-xl);
  padding: 1.1rem 1.5rem;
  margin-bottom: 1.5rem;
  flex-wrap: wrap; gap: 1rem;
  box-shadow: var(--shadow-sm);
}
.results-header-title {
  font-family: var(--font-display);
  font-size: 1.45rem; font-weight: 700;
  color: var(--on-surface); line-height: 1.1;
  letter-spacing: -0.022em;
}
.results-header-sub {
  font-size: .76rem; color: var(--on-surface-variant); margin-top: 3px;
}

/* Wizard responsive */
@media (max-width: 720px) {
  .wizard-shell { padding: 0 .75rem; margin: 1rem auto 2rem; }
  .step-card { padding: 1.5rem 1.25rem 1.25rem; }
  .step-title { font-size: 1.45rem; }
  .step-sub { font-size: .82rem; }
  .stepper { flex-wrap: wrap; gap: .35rem; }
  .step-connector { display: none; }
  .step-node { min-width: 70px; }
  .step-node-label { font-size: .58rem; }
  .review-row { grid-template-columns: 1fr; }
  .results-header { flex-direction: column; align-items: flex-start; }
  .top-bar { padding: .6rem 1rem; }
  .top-bar-sub { display: none; }
}
</style>
""")

# ─── God-mode design system v5 — high-contrast professional ────
st.html("""
<style>
/* ══════════════════════════════════════════════════════════════════
   DESIGN SYSTEM v5  ·  High-contrast professional overhaul
   All tokens redefined here; legacy aliases kept for results HTML.
══════════════════════════════════════════════════════════════════ */
:root {
  /* ── Surfaces ── */
  --bg:             #e5e1d8;
  --surface:        #ffffff;
  --surface-low:    #f5f2ec;
  --surface-mid:    #edeae3;
  --surface-high:   #e3dfd7;

  /* ── Text — 7:1+ contrast on white ── */
  --text-1: #0d0c0b;
  --text-2: #3b3630;
  --text-3: #6b6358;
  --text-4: #a69e94;

  /* ── Brand (deeper, more saturated terracotta) ── */
  --brand:       #be4218;
  --brand-dark:  #952e0d;
  --brand-mid:   #d35228;
  --brand-light: #fde8dc;
  --brand-glow:  rgba(190,66,24,.15);
  --brand-shadow:rgba(190,66,24,.30);

  /* ── Gradient ── */
  --gradient: linear-gradient(140deg, #be4218 0%, #d35228 55%, #e07030 100%);
  --gradient-btn-shadow: 0 4px 14px rgba(190,66,24,.35);

  /* ── Semantic ── */
  --ok:    #14532d;  --ok-bg:   #f0fdf4;  --ok-ring:   rgba(20,83,45,.2);
  --warn:  #92400e;  --warn-bg: #fffbeb;  --warn-ring: rgba(146,64,14,.2);
  --err:   #991b1b;  --err-bg:  #fef2f2;  --err-ring:  rgba(153,27,27,.2);

  /* ── Borders ── */
  --border:        #ccc5ba;
  --border-strong: #b0a89c;

  /* ── Shadows (heavier = more depth) ── */
  --shadow-xs: 0 1px 2px rgba(13,12,11,.08);
  --shadow-sm: 0 1px 3px rgba(13,12,11,.10), 0 4px 14px rgba(13,12,11,.07);
  --shadow-md: 0 4px 12px rgba(13,12,11,.11), 0 2px 4px rgba(13,12,11,.07);
  --shadow-lg: 0 8px 30px rgba(13,12,11,.13), 0 4px 8px rgba(13,12,11,.07);
  --shadow-xl: 0 20px 50px rgba(13,12,11,.15);

  /* ── Radii ── */
  --r-sm:  6px; --r-md: 10px; --r-lg: 16px;
  --r-xl:  22px; --r-full: 9999px;
  --radius-xl: 22px; --radius-lg: 16px;
  --radius-md: 10px; --radius-sm: 6px;
  --radius-full: 9999px;

  /* ── Fonts ── */
  --font-display: -apple-system,BlinkMacSystemFont,"SF Pro Display","Inter",system-ui,"Segoe UI",sans-serif;
  --font-text:    -apple-system,BlinkMacSystemFont,"SF Pro Text","Inter",system-ui,"Segoe UI",sans-serif;
  --font-mono:    ui-monospace,"SF Mono",Menlo,Consolas,"Roboto Mono",monospace;

  /* ── Legacy aliases (keep results rendering working) ── */
  --primary:                  var(--brand);
  --primary-container:        var(--brand-mid);
  --primary-fixed:            var(--brand-light);
  --primary-fixed-dim:        #f5bf9e;
  --on-primary:               #ffffff;
  --on-primary-fixed:         #3a1606;
  --on-primary-container:     var(--brand-light);
  --on-surface:               var(--text-1);
  --on-surface-variant:       var(--text-3);
  --on-surface-sub:           var(--text-3);
  --surface-container-lowest: #ffffff;
  --surface-container-low:    var(--surface-low);
  --surface-container:        var(--surface-mid);
  --surface-container-high:   var(--surface-high);
  --surface-container-highest:var(--border);
  --surface-bright:           #ffffff;
  --surface-variant:          var(--surface-mid);
  --surface-dim:              var(--surface-high);
  --background:               var(--bg);
  --on-background:            var(--text-1);
  --outline:                  var(--text-4);
  --outline-variant:          var(--border);
  --error:                    var(--err);
  --error-container:          var(--err-bg);
  --on-error:                 #ffffff;
  --on-error-container:       #7f1d1d;
  --secondary:                var(--text-3);
  --secondary-container:      var(--surface-mid);
  --on-secondary:             #ffffff;
  --on-secondary-container:   var(--text-3);
  --tertiary:                 #7a3c3c;
  --tertiary-container:       #d07070;
  --tertiary-fixed:           #fce0e0;
  --on-tertiary-container:    #4a2020;
  --on-tertiary-fixed:        #2e1515;
  --surface-tint:             var(--brand);
  --shadow-sm: 0 1px 3px rgba(13,12,11,.10), 0 4px 14px rgba(13,12,11,.07);
  --shadow-md: 0 4px 12px rgba(13,12,11,.11), 0 2px 4px rgba(13,12,11,.07);
  --shadow-lg: 0 8px 30px rgba(13,12,11,.13), 0 4px 8px rgba(13,12,11,.07);
}

/* ── BASE ── */
html, body, [data-testid="stApp"] {
  background: var(--bg) !important;
  font-family: var(--font-text) !important;
  color: var(--text-2) !important;
}
.stApp { background: var(--bg) !important; }
.block-container {
  padding: 0 1.5rem 4rem !important;
  max-width: 880px !important;
  margin: 0 auto !important;
}

/* ── WIDGETS ── */
.stTextArea textarea {
  background: #ffffff !important;
  border: 2px solid var(--border-strong) !important;
  border-radius: var(--r-md) !important;
  color: var(--text-1) !important;
  font-family: var(--font-text) !important;
  font-size: 0.9375rem !important;
  line-height: 1.7 !important;
  padding: 1rem 1.125rem !important;
  box-shadow: none !important;
  transition: border-color .15s, box-shadow .15s !important;
}
.stTextArea textarea::placeholder {
  color: var(--text-4) !important;
  opacity: 1 !important;
}
.stTextArea textarea:focus {
  border-color: var(--brand) !important;
  box-shadow: 0 0 0 4px var(--brand-glow) !important;
  outline: none !important;
}
label {
  color: var(--text-3) !important;
  font-size: 0.6875rem !important;
  font-weight: 700 !important;
  letter-spacing: 0.08em !important;
  text-transform: uppercase !important;
}
[data-testid="stFileUploader"] {
  background: #ffffff !important;
  border: 2px dashed var(--border-strong) !important;
  border-radius: var(--r-lg) !important;
  padding: 1.5rem !important;
  transition: border-color .18s, background .18s !important;
}
[data-testid="stFileUploader"]:hover {
  background: var(--brand-light) !important;
  border-color: var(--brand) !important;
}

/* ── BUTTONS ── */
.stButton > button {
  background: var(--gradient) !important;
  color: #fff !important;
  border: none !important;
  border-radius: var(--r-full) !important;
  padding: 0.8rem 2rem !important;
  font-family: var(--font-display) !important;
  font-weight: 700 !important;
  font-size: 0.9375rem !important;
  letter-spacing: -0.01em !important;
  box-shadow: var(--gradient-btn-shadow) !important;
  transition: opacity .15s, transform .12s, box-shadow .15s !important;
  width: 100% !important;
}
.stButton > button:hover {
  opacity: .92 !important;
  transform: translateY(-2px) !important;
  box-shadow: 0 10px 28px var(--brand-shadow) !important;
}
.stButton > button:active { transform: translateY(0) !important; }
.stButton > button:disabled {
  background: var(--surface-high) !important;
  color: var(--text-4) !important;
  box-shadow: none !important;
  transform: none !important;
  opacity: 1 !important;
}
.stButton > button[kind="secondary"] {
  background: #ffffff !important;
  color: var(--text-2) !important;
  border: 1.5px solid var(--border-strong) !important;
  box-shadow: var(--shadow-xs) !important;
  font-weight: 600 !important;
  font-size: 0.875rem !important;
  padding: 0.75rem 1.5rem !important;
  letter-spacing: 0 !important;
}
.stButton > button[kind="secondary"]:hover {
  background: var(--surface-low) !important;
  border-color: var(--brand) !important;
  color: var(--brand) !important;
  transform: none !important;
  box-shadow: var(--shadow-xs) !important;
}
.stDownloadButton > button {
  background: transparent !important;
  color: var(--brand) !important;
  border: 1.5px solid var(--brand) !important;
  border-radius: var(--r-full) !important;
  font-weight: 600 !important;
  font-size: 0.8125rem !important;
  box-shadow: none !important;
  padding: 0.5rem 1.25rem !important;
}
.stDownloadButton > button:hover {
  background: var(--brand-light) !important;
  transform: none !important;
}
.stProgress > div > div { background: var(--gradient) !important; border-radius: 100px !important; }
.stProgress > div { background: var(--border) !important; border-radius: 100px !important; height: 5px !important; }
[data-testid="stDataFrame"] { border-radius: var(--r-lg) !important; border: 1px solid var(--border) !important; box-shadow: var(--shadow-sm) !important; overflow: hidden !important; }
.streamlit-expanderHeader {
  background: #ffffff !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--r-md) !important;
  color: var(--text-1) !important;
  font-family: var(--font-display) !important;
  font-weight: 700 !important;
  font-size: 1rem !important;
  box-shadow: var(--shadow-sm) !important;
}
.streamlit-expanderContent {
  background: #ffffff !important;
  border: 1px solid var(--border) !important;
  border-top: none !important;
  border-radius: 0 0 var(--r-md) var(--r-md) !important;
}
.stMarkdown p { color: var(--text-3) !important; font-size: .875rem !important; }
h1,h2,h3,h4 { font-family: var(--font-display) !important; color: var(--text-1) !important; }

/* ══ TOP BAR ══ */
.top-bar {
  display: flex; align-items: center; gap: 0.875rem;
  background: rgba(229,225,216,.93);
  backdrop-filter: saturate(200%) blur(24px);
  -webkit-backdrop-filter: saturate(200%) blur(24px);
  padding: 0.875rem 1.5rem;
  border-bottom: 1px solid rgba(204,197,186,.75);
  position: sticky; top: 0; z-index: 100;
  box-shadow: 0 1px 16px rgba(13,12,11,.07);
}
.top-bar-icon {
  width: 36px; height: 36px;
  background: var(--gradient);
  border-radius: 10px;
  display: flex; align-items: center; justify-content: center;
  font-size: 1rem;
  box-shadow: 0 2px 10px var(--brand-shadow);
  flex-shrink: 0;
}
.top-bar-title {
  font-family: var(--font-display);
  font-size: 1rem; font-weight: 700;
  color: var(--text-1); letter-spacing: -0.018em;
}
.top-bar-sub { display: none; }
.top-bar-status { margin-left: auto; display: flex; gap: 0.5rem; align-items: center; }
.tb-pill {
  display: inline-flex; align-items: center; gap: 5px;
  font-size: 0.6875rem; font-weight: 700;
  letter-spacing: 0.06em; text-transform: uppercase;
  padding: 4px 10px; border-radius: var(--r-full);
  border: 1px solid transparent;
}
.tb-pill.ok   { background: var(--ok-bg);   color: var(--ok);   border-color: var(--ok-ring); }
.tb-pill.err  { background: var(--err-bg);  color: var(--err);  border-color: var(--err-ring); }
.tb-pill.info { background: #ffffff; color: var(--text-3); border-color: var(--border); }

/* ══ STEPPER ══ */
.stepper {
  display: flex; align-items: flex-start; justify-content: center;
  gap: 0; margin: 2rem 0 1.75rem; padding: 0;
}
.step-node {
  display: flex; flex-direction: column;
  align-items: center; gap: 8px;
  flex: 0 0 auto; min-width: 90px;
}
.step-node-disc {
  width: 32px; height: 32px; border-radius: 50%;
  display: flex; align-items: center; justify-content: center;
  background: #ffffff; border: 2px solid var(--border-strong);
  font-family: var(--font-display);
  font-weight: 700; font-size: 0.875rem; color: var(--text-3);
  transition: all .22s ease;
}
.step-node.is-active .step-node-disc {
  background: var(--brand); border-color: var(--brand); color: #ffffff;
  box-shadow: 0 0 0 5px var(--brand-glow), 0 4px 14px var(--brand-shadow);
}
.step-node.is-done .step-node-disc { background: var(--brand); border-color: var(--brand); color: #ffffff; }
.step-node-label {
  font-size: 0.6rem; font-weight: 700; text-transform: uppercase;
  letter-spacing: 0.1em; color: var(--text-4); white-space: nowrap; text-align: center;
}
.step-node.is-active .step-node-label { color: var(--text-1); }
.step-node.is-done   .step-node-label { color: var(--brand); }
.step-connector {
  flex: 1 1 auto; height: 2px; background: var(--border);
  margin: 16px -6px 0; max-width: 80px; transition: background .3s;
}
.step-connector.is-done { background: var(--brand); }

/* ══ STEP CARD ══ */
.step-card {
  background: #ffffff;
  border: 1px solid var(--border);
  border-radius: var(--r-xl);
  box-shadow: var(--shadow-lg);
  padding: 2.5rem 2.5rem 2rem;
  animation: fadeUp .24s ease-out;
}
@keyframes fadeUp {
  from { opacity: 0; transform: translateY(10px); }
  to   { opacity: 1; transform: none; }
}
.step-badge {
  display: inline-flex; align-items: center; gap: 6px;
  font-size: 0.6875rem; font-weight: 700;
  letter-spacing: 0.1em; text-transform: uppercase;
  color: var(--brand); background: var(--brand-light);
  padding: 4px 12px; border-radius: var(--r-full);
  margin-bottom: 1rem;
  border: 1px solid rgba(190,66,24,.18);
}
.step-title {
  font-family: var(--font-display);
  font-size: 2.25rem; font-weight: 800;
  color: var(--text-1); line-height: 1.08;
  letter-spacing: -0.032em; margin: 0 0 0.5rem;
}
.step-sub {
  font-size: 0.9375rem; color: var(--text-3);
  line-height: 1.6; margin: 0 0 1.75rem;
}
.step-divider { height: 1px; background: var(--border); margin: 2rem 0 1.5rem; }

/* ══ METRICS / TILES ══ */
.metrics-bar { display: grid; grid-template-columns: repeat(4,1fr); gap: 1rem; }
.metric-tile {
  background: #ffffff; border: 1px solid var(--border);
  border-radius: var(--r-lg); padding: 1.25rem 1rem;
  box-shadow: var(--shadow-xs);
  transition: box-shadow .18s, transform .18s;
}
.metric-tile:hover { box-shadow: var(--shadow-md); transform: translateY(-1px); }
.metric-tile.accent { border-left: 3px solid var(--brand); }
.metric-label { font-size: 0.6875rem; font-weight: 700; letter-spacing: 0.07em; text-transform: uppercase; color: var(--text-3); margin-bottom: 0.4rem; }
.metric-value { font-family: var(--font-display); font-size: 2.25rem; font-weight: 800; color: var(--text-1); line-height: 1; letter-spacing: -0.03em; }
.metric-unit { font-size: 1.1rem; color: var(--text-4); }

/* ══ CANDIDATE CARD ══ */
.cand-card { background: #ffffff; border: 1px solid var(--border); border-radius: var(--r-xl); box-shadow: var(--shadow-lg); overflow: hidden; }
.cand-card-header { padding: 1.75rem; border-bottom: 1px solid var(--border); background: var(--surface-low); display: flex; align-items: flex-start; justify-content: space-between; }
.cand-card-body { padding: 1.75rem; display: flex; flex-direction: column; gap: 1.25rem; }
.score-ring-wrap { position: relative; width: 60px; height: 60px; flex-shrink: 0; }
.score-ring-wrap svg { transform: rotate(-90deg); }
.score-num { position: absolute; inset: 0; display: flex; align-items: center; justify-content: center; font-family: var(--font-display); font-size: 1.15rem; font-weight: 800; }
.score-num.high { color: var(--brand); }
.score-num.mid  { color: #b45309; }
.score-num.low  { color: var(--err); }
.tldr-block { background: var(--brand-light); border-left: 3px solid var(--brand); border-radius: 0 var(--r-sm) var(--r-sm) 0; padding: 0.875rem 1.125rem; font-size: 0.875rem; line-height: 1.6; color: var(--text-2); }
.tldr-label { font-weight: 700; color: var(--brand); margin-right: 6px; }
.culture-block { background: rgba(122,60,60,.07); border-left: 3px solid var(--tertiary); border-radius: 0 var(--r-sm) var(--r-sm) 0; padding: 0.75rem 1rem; font-size: 0.8125rem; color: var(--tertiary); line-height: 1.5; }

/* ══ CHIPS ══ */
.chip-row { display: flex; flex-wrap: wrap; gap: 5px; margin-top: 5px; }
.chip { display: inline-flex; align-items: center; gap: 4px; padding: 3px 10px; border-radius: var(--r-sm); font-size: 0.75rem; font-weight: 600; }
.chip-green  { background: #f0fdf4; color: #14532d; border: 1px solid rgba(21,128,61,.2); }
.chip-red    { background: var(--err-bg); color: #991b1b; border: 1px solid var(--err-ring); }
.chip-blue   { background: #eff6ff; color: #1e3a8a; border: 1px solid rgba(37,99,235,.2); }
.chip-yellow { background: #fffbeb; color: #78350f; border: 1px solid rgba(146,64,14,.2); }
.chip-purple { background: #faf5ff; color: #581c87; border: 1px solid rgba(109,40,217,.2); }
.chip-tint   { background: var(--brand-light); color: var(--brand-dark); border: 1px solid rgba(190,66,24,.2); font-weight: 700; }
.chip-surface{ background: var(--surface-mid); color: var(--text-3); border: 1px solid var(--border); }
.sub-label { font-size: 0.6875rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.1em; color: var(--text-3); display: flex; align-items: center; gap: 5px; margin-bottom: 0.4rem; }
.sub-label .material-symbols-outlined { font-size: 14px; }

/* ══ BANNERS ══ */
.banner { display: flex; align-items: flex-start; gap: 8px; padding: 10px 14px; border-radius: var(--r-sm); font-size: 0.8125rem; line-height: 1.5; margin: 6px 0; border: 1px solid transparent; }
.banner-ok   { background: var(--ok-bg);   border-color: var(--ok-ring);   color: var(--ok); }
.banner-warn { background: var(--warn-bg); border-color: var(--warn-ring); color: var(--warn); }
.banner-err  { background: var(--err-bg);  border-color: var(--err-ring);  color: var(--err); }
.banner-info { background: var(--brand-light); border-color: rgba(190,66,24,.2); color: var(--brand-dark); }

/* ══ MISC ══ */
.q-item { background: var(--surface-low); border: 1px solid var(--border); border-radius: var(--r-md); padding: 0.875rem 1.125rem; font-size: 0.875rem; color: var(--text-2); display: flex; gap: 0.75rem; line-height: 1.55; }
.q-num { font-weight: 700; color: var(--text-3); flex-shrink: 0; }
.pool-card { background: #ffffff; border: 1px solid var(--border); border-radius: var(--r-xl); box-shadow: var(--shadow-sm); padding: 1.25rem; }
.runner-tile { background: var(--surface-low); border: 1px solid var(--border); border-radius: var(--r-lg); padding: 0.875rem 1.125rem; display: flex; align-items: center; justify-content: space-between; transition: box-shadow .18s, border-color .18s; margin-bottom: 0.625rem; }
.runner-tile:last-child { margin-bottom: 0; }
.runner-tile:hover { box-shadow: var(--shadow-md); border-color: var(--brand); }
.runner-avatar { width: 40px; height: 40px; border-radius: 50%; background: var(--brand-light); border: 2px solid rgba(190,66,24,.2); display: flex; align-items: center; justify-content: center; font-family: var(--font-display); font-weight: 800; color: var(--brand); font-size: 0.9rem; flex-shrink: 0; }
.runner-name { font-family: var(--font-display); font-weight: 700; font-size: 0.9375rem; color: var(--text-1); }
.runner-sub  { font-size: 0.75rem; color: var(--text-3); }
.dot-row { display: flex; gap: 5px; }
.dot { width: 9px; height: 9px; border-radius: 50%; }
.dot-on  { background: #22c55e; }
.dot-off { background: var(--border); }
.prog-label { display: flex; align-items: center; gap: 8px; font-size: 0.875rem; font-weight: 600; color: var(--brand); padding: 4px 0; }
.prog-dot { width: 8px; height: 8px; border-radius: 50%; background: var(--brand); animation: blink 1.1s ease-in-out infinite; }
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:.25} }
.stats-bar { display: grid; grid-template-columns: repeat(5,1fr); gap: .75rem; margin: .5rem 0 1.25rem; }
.stat-tile { background: #ffffff; border: 1px solid var(--border); border-radius: var(--r-lg); padding: 1rem; text-align: center; box-shadow: var(--shadow-xs); transition: box-shadow .18s, transform .18s; }
.stat-tile:hover { box-shadow: var(--shadow-md); transform: translateY(-2px); }
.stat-val { font-family: var(--font-display); font-size: 1.9rem; font-weight: 800; line-height: 1; color: var(--text-1); letter-spacing: -0.03em; }
.stat-key { font-size: .65rem; font-weight: 700; text-transform: uppercase; letter-spacing: .1em; color: var(--text-3); margin-top: 4px; }
.resume-box { background: var(--surface-low); border: 1px solid var(--border); border-radius: var(--r-lg); padding: 1rem 1.125rem; font-family: var(--font-mono); font-size: .8125rem; color: var(--text-3); max-height: 200px; overflow-y: auto; line-height: 1.75; white-space: pre-wrap; word-break: break-word; }
.resume-box::-webkit-scrollbar { width: 6px; }
.resume-box::-webkit-scrollbar-thumb { background: var(--border-strong); border-radius: 4px; }
mark { background: #fcd34d; color: #451a03; padding: 1px 4px; border-radius: 3px; }
.divider { height: 1px; background: var(--border); margin: 1.25rem 0; }
.footer { text-align: center; font-size: .6875rem; color: var(--text-4); padding: 2rem 1.5rem; border-top: 1px solid var(--border); letter-spacing: .03em; }
.wc-hint { font-size: .75rem; margin-top: 6px; padding: 5px 10px; border-radius: var(--r-sm); display: inline-block; font-weight: 500; }
.wc-ok   { background: var(--ok-bg);   color: var(--ok);   border: 1px solid var(--ok-ring); }
.wc-warn { background: var(--warn-bg); color: var(--warn); border: 1px solid var(--warn-ring); }
.section-hed { font-family: var(--font-display); font-size: 1.25rem; font-weight: 700; color: var(--text-1); margin: 0 0 .2rem; }
.section-sub { font-size: .75rem; color: var(--text-3); margin: 0; }
.err-pill { display: inline-block; font-size: .6875rem; font-weight: 700; letter-spacing: .06em; text-transform: uppercase; padding: 2px 8px; border-radius: var(--r-full); background: var(--err-bg); color: var(--err); border: 1px solid var(--err-ring); }
.input-card { background: #ffffff; border: 1px solid var(--border); border-radius: var(--r-xl); box-shadow: var(--shadow-sm); padding: 1.75rem; }
.field-label { font-size: .6875rem; font-weight: 700; letter-spacing: .1em; text-transform: uppercase; color: var(--text-3); margin-bottom: .5rem; display: block; }
.review-row { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-bottom: 1.25rem; }
.review-tile { background: var(--surface-low); border: 1px solid var(--border); border-radius: var(--r-lg); padding: 1.125rem; }
.review-tile-label { font-size: .6875rem; font-weight: 700; text-transform: uppercase; letter-spacing: .1em; color: var(--text-3); margin-bottom: .4rem; }
.review-tile-value { font-family: var(--font-display); font-weight: 800; font-size: 1.25rem; color: var(--text-1); line-height: 1.2; }
.review-jd { background: var(--brand-light); border: 1px solid rgba(190,66,24,.2); border-left: 3px solid var(--brand); border-radius: var(--r-lg); padding: 1rem 1.125rem; font-size: .875rem; line-height: 1.6; color: var(--text-2); max-height: 140px; overflow-y: auto; margin-bottom: 1.25rem; }
.file-pill-row { display: flex; flex-wrap: wrap; gap: .5rem; margin-top: .5rem; }
.file-pill { display: inline-flex; align-items: center; gap: 8px; background: #ffffff; border: 1.5px solid var(--border-strong); border-radius: var(--r-full); padding: 5px 12px; font-size: .8125rem; font-weight: 500; color: var(--text-2); }
.file-pill-size { color: var(--text-4); font-size: .75rem; }
.results-header { display: flex; align-items: center; justify-content: space-between; background: #ffffff; border: 1px solid var(--border); border-radius: var(--r-xl); padding: 1.25rem 1.75rem; margin-bottom: 1.5rem; flex-wrap: wrap; gap: 1rem; box-shadow: var(--shadow-sm); }
.results-header-title { font-family: var(--font-display); font-size: 1.5rem; font-weight: 800; color: var(--text-1); line-height: 1.1; letter-spacing: -.022em; }
.results-header-sub { font-size: .8125rem; color: var(--text-3); margin-top: 3px; }
.wizard-shell { margin: 1.5rem 0 3rem; padding: 0; }
.wizard-shell.is-results { margin: 1.5rem 0; }
.bento-wrap { display: flex; gap: 2rem; padding: 1.5rem 0; align-items: flex-start; }
.bento-left { flex: 0 0 340px; display: flex; flex-direction: column; gap: 1rem; }
.bento-right { flex: 1; display: flex; flex-direction: column; gap: 1.5rem; }

@media (max-width: 720px) {
  .block-container { padding: 0 .75rem 2rem !important; }
  .step-card { padding: 1.75rem 1.25rem 1.5rem; }
  .step-title { font-size: 1.75rem; }
  .stepper { flex-wrap: wrap; gap: .4rem; }
  .step-connector { display: none; }
  .step-node { min-width: 70px; }
  .review-row { grid-template-columns: 1fr; }
  .results-header { flex-direction: column; align-items: flex-start; }
  .metrics-bar { grid-template-columns: repeat(2,1fr); }
  .stats-bar   { grid-template-columns: repeat(3,1fr); }
  .top-bar { padding: .75rem 1rem; }
}
</style>
""")


# ─── Editorial design system v6 — warm-paper, serif-led ────────
st.html("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Newsreader:ital,opsz,wght@0,6..72,400;0,6..72,500;0,6..72,600;0,6..72,700;1,6..72,400;1,6..72,500&family=Inter+Tight:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
<style>
/* ════════════════════════════════════════════════════════════
   DESIGN SYSTEM v6 — Editorial warm-paper
   Overrides v5 tokens; wins by source order.
   ════════════════════════════════════════════════════════════ */
:root {
  --paper:        #f4f1ea;
  --paper-2:      #ebe7dd;
  --card:         #fbfaf6;
  --card-2:       #ffffff;
  --ink:          #161513;
  --ink-2:        #2c2a26;
  --ink-3:        #5b574f;
  --ink-4:        #8a857a;
  --rule:         #d9d3c5;
  --rule-2:       #e6e1d4;

  --signal:       #c2410c;
  --signal-soft:  #fbe8d8;
  --signal-deep:  #9a3308;

  --pos:    #2f6b3a;  --pos-bg:  #e7eee0;
  --neg:    #8a3a2e;  --neg-bg:  #f3e2dc;
  --warn-e: #7a5a1a;  --warn-eb: #f1e9d2;

  --serif: "Newsreader", ui-serif, Georgia, "Times New Roman", serif;
  --sans:  "Inter Tight", ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif;
  --mono:  "JetBrains Mono", ui-monospace, "SF Mono", Menlo, monospace;

  --bg:                       var(--paper);
  --background:               var(--paper);
  --surface:                  var(--card);
  --surface-low:              var(--paper-2);
  --surface-mid:              var(--paper-2);
  --surface-high:             var(--rule-2);
  --text-1:                   var(--ink);
  --text-2:                   var(--ink-2);
  --text-3:                   var(--ink-3);
  --text-4:                   var(--ink-4);
  --brand:                    var(--signal);
  --brand-dark:               var(--signal-deep);
  --brand-mid:                var(--signal);
  --brand-light:              var(--signal-soft);
  --brand-glow:               rgba(194,65,12,.12);
  --brand-shadow:             rgba(194,65,12,.22);
  --gradient:                 var(--signal);
  --gradient-btn-shadow:      0 1px 0 rgba(0,0,0,.06);
  --border:                   var(--rule);
  --border-strong:            var(--ink-4);
  --primary:                  var(--signal);
  --primary-container:        var(--signal);
  --primary-fixed:            var(--signal-soft);
  --primary-fixed-dim:        #f0d4ba;
  --on-primary:               #ffffff;
  --on-primary-fixed:         var(--signal-deep);
  --on-surface:               var(--ink);
  --on-surface-variant:       var(--ink-3);
  --on-surface-sub:           var(--ink-3);
  --surface-container-lowest: var(--card-2);
  --surface-container-low:    var(--card);
  --surface-container:        var(--paper-2);
  --surface-container-high:   var(--rule-2);
  --surface-container-highest:var(--rule);
  --surface-bright:           var(--card-2);
  --surface-variant:          var(--paper-2);
  --surface-dim:              var(--rule-2);
  --outline:                  var(--ink-4);
  --outline-variant:          var(--rule);
  --tertiary:                 var(--ink-3);
  --error:                    var(--neg);
  --error-container:          var(--neg-bg);
  --on-error-container:       var(--neg);
  --font-display:             var(--serif);
  --font-text:                var(--sans);
  --shadow-sm: 0 1px 0 rgba(22,21,19,.04), 0 1px 2px rgba(22,21,19,.04);
  --shadow-md: 0 1px 0 rgba(22,21,19,.04), 0 12px 28px -16px rgba(22,21,19,.18);
}

html, body, [data-testid="stApp"] {
  background: var(--paper) !important;
  font-family: var(--sans) !important;
  color: var(--ink) !important;
  font-feature-settings: "ss01","cv11";
  letter-spacing: -0.005em;
}
.stApp { background: var(--paper) !important; }
.block-container {
  padding: 0 1.25rem 3rem !important;
  max-width: 1180px !important;
}
::selection { background: var(--signal); color:#ffffff; }

[data-testid="stApp"]::before {
  content:""; position:fixed; inset:0; pointer-events:none; z-index:0;
  background-image:
    radial-gradient(rgba(22,21,19,0.025) 1px, transparent 1px),
    radial-gradient(rgba(22,21,19,0.018) 1px, transparent 1px);
  background-size: 3px 3px, 7px 7px;
  background-position: 0 0, 1px 1px;
  opacity:.55;
}
[data-testid="stApp"] > * { position:relative; z-index:1; }

.top-bar {
  background: rgba(244,241,234,.78) !important;
  backdrop-filter: saturate(180%) blur(20px);
  border-bottom: 1px solid var(--rule) !important;
  padding: 14px 24px !important;
  gap: 14px !important;
}
.top-bar-icon {
  width: 38px !important; height: 38px !important;
  background: var(--ink) !important;
  border-radius: 10px !important;
  color: var(--paper) !important;
  font-family: var(--serif) !important;
  font-style: italic !important;
  font-size: 22px !important;
  display: grid !important; place-items: center !important;
  position: relative !important;
}
.top-bar-icon::after {
  content:""; position:absolute; right:-3px; bottom:-3px;
  width:10px; height:10px; border-radius:50%;
  background: var(--signal);
  box-shadow:0 0 0 2px var(--paper);
}
.top-bar-title {
  font-family: var(--serif) !important;
  font-weight: 400 !important;
  font-size: 22px !important;
  letter-spacing: -.01em !important;
}
.top-bar-sub { display: none !important; }
.tb-pill {
  font-family: var(--mono) !important;
  font-size: 10.5px !important;
  font-weight: 500 !important;
  letter-spacing: .06em !important;
  border-radius: 999px !important;
  padding: 5px 10px !important;
  background: var(--card) !important;
  border: 1px solid var(--rule) !important;
  color: var(--ink-3) !important;
}
.tb-pill.ok { color: var(--pos) !important; background: var(--pos-bg) !important; border-color: rgba(47,107,58,.2) !important; }
.tb-pill.err { color: var(--neg) !important; background: var(--neg-bg) !important; border-color: rgba(138,58,46,.2) !important; }

h1, h2, h3, h4, .step-title, .results-header-title {
  font-family: var(--serif) !important;
  font-weight: 400 !important;
  letter-spacing: -.022em !important;
}

.step-node-disc {
  background: var(--card-2) !important;
  border: 1.5px solid var(--rule) !important;
  color: var(--ink-3) !important;
  font-family: var(--mono) !important;
  font-weight: 500 !important;
}
.step-node.is-active .step-node-disc,
.step-node.is-done .step-node-disc {
  background: var(--ink) !important;
  border-color: var(--ink) !important;
  color: var(--paper) !important;
  box-shadow: 0 0 0 4px rgba(22,21,19,.08) !important;
}
.step-node.is-done .step-node-disc { background: var(--signal) !important; border-color: var(--signal) !important; }
.step-node-label {
  font-family: var(--mono) !important;
  font-size: 10px !important;
  letter-spacing: .14em !important;
  font-weight: 500 !important;
}
.step-connector { background: var(--rule) !important; }
.step-connector.is-done { background: var(--signal) !important; }

.step-card {
  background: var(--card) !important;
  border: 1px solid var(--rule) !important;
  border-radius: 16px !important;
  box-shadow: var(--shadow-sm) !important;
}
.step-badge {
  background: var(--paper-2) !important;
  color: var(--signal) !important;
  font-family: var(--mono) !important;
  letter-spacing: .14em !important;
  border-radius: 6px !important;
}
.step-title {
  font-family: var(--serif) !important;
  font-size: 2rem !important;
  letter-spacing: -.025em !important;
  font-weight: 400 !important;
}
.step-title em { font-style: italic; color: var(--ink-3); }
.step-sub { color: var(--ink-3) !important; }

.stButton > button {
  background: var(--ink) !important;
  color: var(--paper) !important;
  border: 1px solid var(--ink) !important;
  border-radius: 8px !important;
  padding: 10px 16px !important;
  font-family: var(--sans) !important;
  font-weight: 500 !important;
  font-size: 13px !important;
  box-shadow: none !important;
  letter-spacing: -.005em !important;
  transition: background .15s, transform .08s !important;
}
.stButton > button:hover {
  background: var(--ink-2) !important;
  border-color: var(--ink-2) !important;
  transform: none !important;
  opacity: 1 !important;
  box-shadow: none !important;
}
.stButton > button:active { transform: translateY(1px) !important; }
.stButton > button[kind="secondary"] {
  background: var(--card-2) !important;
  color: var(--ink) !important;
  border: 1px solid var(--rule) !important;
}
.stButton > button[kind="secondary"]:hover {
  background: var(--paper-2) !important;
  border-color: var(--ink-4) !important;
  color: var(--ink) !important;
}
.stButton > button:disabled {
  background: var(--paper-2) !important;
  color: var(--ink-4) !important;
  border-color: var(--rule) !important;
}
.stDownloadButton > button {
  background: var(--signal) !important;
  color: #fff !important;
  border: 1px solid var(--signal) !important;
  border-radius: 8px !important;
  font-family: var(--sans) !important;
  font-weight: 500 !important;
  font-size: 13px !important;
}
.stDownloadButton > button:hover {
  background: var(--signal-deep) !important;
  border-color: var(--signal-deep) !important;
}

.stTextArea textarea {
  background: var(--card-2) !important;
  border: 1px solid var(--rule) !important;
  border-radius: 10px !important;
  font-family: var(--sans) !important;
  font-size: 13.5px !important;
  color: var(--ink) !important;
  box-shadow: none !important;
}
.stTextArea textarea:focus {
  border-color: var(--ink) !important;
  box-shadow: 0 0 0 3px rgba(22,21,19,.06) !important;
}
[data-testid="stFileUploader"] {
  background: var(--card-2) !important;
  border: 1.5px dashed var(--rule) !important;
  border-radius: 12px !important;
}
[data-testid="stFileUploader"]:hover {
  background: var(--paper-2) !important;
  border-color: var(--ink-4) !important;
}
label {
  font-family: var(--mono) !important;
  font-size: 10px !important;
  letter-spacing: .14em !important;
  font-weight: 500 !important;
  color: var(--ink-4) !important;
}

.stProgress > div > div { background: var(--ink) !important; }
.stProgress > div { background: var(--paper-2) !important; height: 4px !important; }

.banner {
  font-family: var(--sans) !important;
  border-radius: 10px !important;
  font-size: 13px !important;
  border: 1px solid var(--rule) !important;
  background: var(--card-2) !important;
  color: var(--ink-2) !important;
}
.banner-ok   { background: var(--pos-bg) !important; border-color: rgba(47,107,58,.2) !important; color: var(--pos) !important; }
.banner-warn { background: var(--warn-eb) !important; border-color: rgba(122,90,26,.2) !important; color: var(--warn-e) !important; }
.banner-err  { background: var(--neg-bg) !important; border-color: rgba(138,58,46,.2) !important; color: var(--neg) !important; }

[data-testid="stDataFrame"] {
  border: 1px solid var(--rule) !important;
  border-radius: 12px !important;
  box-shadow: var(--shadow-sm) !important;
  background: var(--card-2) !important;
}

.streamlit-expanderHeader {
  background: var(--card) !important;
  border: 1px solid var(--rule) !important;
  border-radius: 10px !important;
  font-family: var(--serif) !important;
  font-weight: 500 !important;
  color: var(--ink) !important;
  box-shadow: none !important;
}
.streamlit-expanderContent {
  background: var(--card-2) !important;
  border: 1px solid var(--rule) !important;
  border-top: none !important;
}

/* ════════════════════════════════════════════════════════════
   RESULTS DASHBOARD — editorial bento
   ════════════════════════════════════════════════════════════ */

.ed-pagehead {
  display: flex; align-items: flex-end; justify-content: space-between;
  gap: 48px; margin: 8px 0 28px;
}
.ed-pagehead h1 {
  font-family: var(--serif); font-weight: 400;
  font-size: 56px; line-height: .98; letter-spacing: -.025em;
  margin: 0; color: var(--ink); max-width: 780px;
}
.ed-pagehead h1 em { font-style: italic; color: var(--ink-3); }
.ed-pagehead h1 .slash { color: var(--signal); font-style: italic; font-weight: 500; }
.ed-lede {
  font-size: 13px; color: var(--ink-3); line-height: 1.55;
  max-width: 320px; text-align: right;
  flex: 0 0 auto; padding-bottom: 6px;
}
.ed-lede .ts {
  display: block; font-family: var(--mono); font-size: 10px;
  text-transform: uppercase; letter-spacing: .14em;
  color: var(--ink-4); margin-bottom: 6px;
}
.ed-lede .summary {
  display: block; margin-top: 4px;
  color: var(--ink); font-family: var(--serif); font-size: 18px;
  line-height: 1.4; letter-spacing: -.01em;
}
.ed-lede .summary em { font-style: italic; }
.ed-lede .summary .maybe { font-style: italic; color: var(--ink-3); }
.ed-lede .meta {
  display: block; margin-top: 6px;
  font-size: 12px; color: var(--ink-4); font-family: var(--mono);
  letter-spacing: .04em;
}

.ed-jdstrip {
  display: grid; grid-template-columns: 1fr auto;
  align-items: center; gap: 18px 24px;
  background: var(--card); border: 1px solid var(--rule);
  border-radius: 14px; padding: 18px 22px;
  margin-bottom: 18px; box-shadow: var(--shadow-sm);
}
.ed-jd-main { min-width: 0; display: flex; flex-direction: column; gap: 10px; }
.ed-jd-row1 { display: flex; align-items: baseline; gap: 14px; min-width: 0; flex-wrap: wrap; }
.ed-jd-role {
  font-family: var(--serif); font-size: 22px; font-weight: 500;
  color: var(--ink); letter-spacing: -.02em; line-height: 1.1;
}
.ed-jd-meta {
  font-family: var(--sans); font-size: 12.5px; color: var(--ink-3);
  display: flex; align-items: center; gap: 10px;
}
.ed-jd-meta .sep { width: 3px; height: 3px; border-radius: 50%; background: var(--ink-4); }
.ed-jd-musts { display: flex; align-items: center; gap: 6px; flex-wrap: wrap; }
.ed-jd-musts-k {
  font-family: var(--mono); font-size: 9.5px; letter-spacing: .14em;
  text-transform: uppercase; color: var(--ink-4); margin-right: 4px;
}
.ed-jd-must {
  font-family: var(--sans); font-size: 11.5px; color: var(--ink);
  background: var(--paper-2); border: 1px solid var(--rule);
  border-radius: 6px; padding: 3px 8px; font-weight: 500;
}
.ed-jd-corner {
  font-family: var(--mono); font-size: 10px;
  letter-spacing: .14em; text-transform: uppercase;
  color: var(--ink-4);
}
.ed-jd-corner b {
  display: block; font-family: var(--serif); font-style: italic;
  font-weight: 400; font-size: 28px; color: var(--ink);
  letter-spacing: -.02em; margin-top: 4px;
}

.ed-bento {
  display: grid; grid-template-columns: repeat(12, 1fr);
  gap: 14px; margin-bottom: 14px;
}
.ed-tile {
  background: var(--card); border: 1px solid var(--rule);
  border-radius: 16px; padding: 18px;
  box-shadow: var(--shadow-sm);
  display: flex; flex-direction: column; min-width: 0;
}
.ed-tile.dark { background: var(--ink); color: var(--paper); border-color: var(--ink); }
.ed-tile.signal { background: var(--signal-soft); border-color: #f0d4ba; }
.ed-tile-head {
  display: flex; align-items: center; justify-content: space-between;
  margin-bottom: 14px;
}
.ed-tile-label {
  font-family: var(--mono); font-size: 10px; font-weight: 500;
  letter-spacing: .16em; text-transform: uppercase;
  color: var(--ink-4);
}
.ed-tile.dark .ed-tile-label { color: #a5a09a; }
.ed-tile.signal .ed-tile-label { color: var(--signal); opacity: .85; }
.ed-c4 { grid-column: span 4; } .ed-c8 { grid-column: span 8; }
.ed-c3 { grid-column: span 3; } .ed-c6 { grid-column: span 6; }
.ed-c12 { grid-column: span 12; }

.ed-stat-big {
  font-family: var(--serif); font-weight: 400;
  font-size: 52px; line-height: 1; letter-spacing: -.025em;
  margin: 14px 0 8px;
  display: flex; align-items: baseline; gap: 4px;
  color: var(--ink);
}
.ed-stat-big .unit {
  font-family: var(--serif); font-style: italic;
  font-size: 20px; color: var(--ink-3); font-weight: 400;
}
.ed-tile.signal .ed-stat-big { color: var(--signal); }
.ed-tile.signal .ed-stat-big .unit { color: var(--signal-deep); }
.ed-stat-sub { font-size: 12px; color: var(--ink-3); line-height: 1.4; }
.ed-tile.signal .ed-stat-sub { color: var(--signal-deep); }

.ed-hero { overflow: hidden; }
.ed-hero-top {
  display: flex; align-items: flex-start; justify-content: space-between;
  gap: 24px; margin-bottom: 18px;
}
.ed-hero-id {
  font-family: var(--mono); font-size: 10px; letter-spacing: .16em;
  text-transform: uppercase; color: var(--ink-4); margin-bottom: 8px;
}
.ed-hero-id .rank { color: var(--signal); margin-right: 10px; font-weight: 600; }
.ed-hero-name {
  font-family: var(--serif); font-weight: 400;
  font-size: 46px; line-height: 1; letter-spacing: -.02em;
  margin: 0; color: var(--ink);
}
.ed-hero-name em { font-style: italic; color: var(--ink-3); }
.ed-hero-meta {
  display: flex; align-items: center; gap: 14px; margin-top: 10px;
  font-size: 13px; color: var(--ink-3); flex-wrap: wrap;
}
.ed-hero-meta .dot { width: 3px; height: 3px; border-radius: 50%; background: var(--ink-4); }
.ed-hero-tldr {
  font-family: var(--serif); font-style: italic; font-weight: 400;
  font-size: 22px; line-height: 1.35; letter-spacing: -.005em;
  color: var(--ink-2);
  padding: 18px 0 20px;
  border-top: 1px solid var(--rule);
  border-bottom: 1px solid var(--rule);
  margin: 6px 0 18px;
  position: relative; padding-left: 22px;
}
.ed-hero-tldr::before {
  content: "\\201C"; font-family: var(--serif); font-style: normal;
  position: absolute; left: -2px; top: 6px;
  font-size: 48px; color: var(--signal); line-height: 1;
}
.ed-hero-grid {
  display: grid; grid-template-columns: 1.4fr 1fr 1fr; gap: 0;
  border: 1px solid var(--rule); border-radius: 12px; overflow: hidden;
  background: var(--card-2);
}
.ed-hg-cell { padding: 14px 16px; border-right: 1px solid var(--rule); }
.ed-hg-cell:last-child { border-right: none; }
.ed-hg-k {
  font-family: var(--mono); font-size: 9.5px; letter-spacing: .14em;
  text-transform: uppercase; color: var(--ink-4); margin-bottom: 6px;
}
.ed-hg-v {
  font-family: var(--serif); font-size: 22px; line-height: 1.1;
  letter-spacing: -.01em; color: var(--ink);
}
.ed-hg-v.small { font-family: var(--sans); font-size: 13px; font-weight: 500; line-height: 1.4; color: var(--ink-3); }

.ed-ringwrap { flex: 0 0 auto; position: relative; width: 156px; height: 156px; }
.ed-ringwrap svg { transform: rotate(-90deg); }
.ed-score-text {
  position: absolute; inset: 0;
  display: flex; flex-direction: column; align-items: center; justify-content: center;
  font-family: var(--serif); line-height: 1;
}
.ed-score-num { font-size: 44px; letter-spacing: -.02em; color: var(--ink); }
.ed-score-of {
  font-family: var(--mono); font-size: 9.5px; letter-spacing: .14em;
  color: var(--ink-4); margin-top: 5px; text-transform: uppercase;
}
.ed-score-grade {
  margin-top: 8px;
  font-family: var(--mono); font-size: 10px; letter-spacing: .14em;
  text-transform: uppercase; padding: 3px 8px; border-radius: 4px;
  background: var(--signal-soft); color: var(--signal); font-weight: 600;
}
.ed-score-grade.mid { background: var(--warn-eb); color: var(--warn-e); }
.ed-score-grade.low { background: var(--neg-bg); color: var(--neg); }

.ed-chips { display: flex; flex-wrap: wrap; gap: 6px; }
.ed-chip {
  font-family: var(--mono); font-size: 11px; font-weight: 500;
  padding: 4px 9px; border-radius: 6px;
  border: 1px solid var(--rule); background: var(--card-2);
  color: var(--ink-2);
  display: inline-flex; align-items: center; gap: 4px;
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
  max-width: 200px;
}
.ed-chip.match { background: #e7eee0; border-color: #cfdcc1; color: #2f5a1f; }
.ed-chip.miss { background: #f3e2dc; border-color: #e2c5bb; color: #7a2e1f; text-decoration: line-through; text-decoration-thickness: 1px; text-decoration-color: #b66a55; }
.ed-chip.match::before { content: "+"; font-weight: 700; }
.ed-chip.miss::before { content: "\\2212"; font-weight: 700; }

.ed-qlist { display: flex; flex-direction: column; gap: 10px; flex: 1; }
.ed-qitem {
  display: flex; gap: 14px;
  padding: 14px 16px;
  border: 1px solid var(--rule); border-radius: 12px;
  background: var(--card-2);
}
.ed-qnum {
  flex: 0 0 auto;
  font-family: var(--serif); font-style: italic;
  font-size: 24px; line-height: 1; color: var(--signal);
  background: var(--signal-soft); width: 34px; height: 34px;
  border-radius: 8px; display: grid; place-items: center;
  letter-spacing: -.03em;
}
.ed-qbody {
  font-family: var(--serif); font-size: 16px; line-height: 1.4;
  color: var(--ink); letter-spacing: -.005em;
}

.ed-clist {
  display: flex; flex-direction: column; flex: 1; overflow: hidden;
  border-radius: 12px; border: 1px solid var(--rule);
}
.ed-crow {
  display: grid;
  grid-template-columns: 32px 28px 1fr 90px 110px 18px;
  align-items: center; gap: 12px;
  padding: 12px 14px;
  border-bottom: 1px solid var(--rule-2);
  background: var(--card-2);
}
.ed-crow:last-child { border-bottom: none; }
.ed-crow.active { background: var(--ink); color: var(--paper); }
.ed-crow.active .ed-cmeta { color: #a5a09a; }
.ed-crow.active .ed-cscore-num { color: var(--paper); }
.ed-crow.active .ed-cscore-bar { background: rgba(255,255,255,.18); }
.ed-crow.active .ed-cscore-fill { background: var(--signal); }
.ed-crow.active .ed-carrow { color: var(--signal); }
.ed-crank {
  font-family: var(--mono); font-size: 11px; color: var(--ink-4); font-weight: 500;
}
.ed-crow.active .ed-crank { color: #a5a09a; }
.ed-cavatar {
  width: 28px; height: 28px; border-radius: 8px;
  background: var(--paper-2); border: 1px solid var(--rule);
  display: grid; place-items: center;
  font-family: var(--serif); font-size: 14px; line-height: 1;
  color: var(--ink-2);
}
.ed-crow.active .ed-cavatar { background: var(--signal); border-color: var(--signal); color: #ffffff; }
.ed-cinfo { min-width: 0; }
.ed-cname {
  font-size: 13.5px; font-weight: 500;
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
  letter-spacing: -.005em;
}
.ed-cmeta {
  font-family: var(--mono); font-size: 10px;
  color: var(--ink-4); letter-spacing: .04em; margin-top: 2px;
}
.ed-cscore-cell { display: flex; flex-direction: column; gap: 5px; align-items: flex-end; }
.ed-cscore-num {
  font-family: var(--mono); font-size: 13px; font-weight: 600;
  color: var(--ink); letter-spacing: -.01em;
}
.ed-cscore-bar {
  width: 90px; height: 4px; border-radius: 2px;
  background: var(--paper-2); overflow: hidden;
}
.ed-cscore-fill { height: 100%; background: var(--ink); border-radius: 2px; }
.ed-ctag {
  font-family: var(--mono); font-size: 9.5px; font-weight: 500;
  padding: 3px 7px; border-radius: 4px;
  letter-spacing: .08em; text-transform: uppercase; text-align: center;
}
.ed-ctag.strong { background: var(--signal); color: #ffffff; }
.ed-ctag.match  { background: var(--pos-bg); color: var(--pos); }
.ed-ctag.maybe  { background: var(--warn-eb); color: var(--warn-e); }
.ed-ctag.weak   { background: var(--neg-bg); color: var(--neg); }
.ed-carrow {
  font-family: var(--mono); font-size: 14px; color: var(--ink-4); text-align: right;
}

.ed-insight-quote {
  font-family: var(--serif); font-style: italic; font-weight: 400;
  font-size: 28px; line-height: 1.18; letter-spacing: -.015em;
  color: var(--paper); margin: 6px 0 auto;
}
.ed-insight-quote em { font-style: italic; color: var(--signal); }
.ed-insight-foot {
  font-family: var(--mono); font-size: 10px; letter-spacing: .14em;
  text-transform: uppercase; color: #a5a09a; margin-top: 14px;
  display: flex; align-items: center; justify-content: space-between;
}

.ed-dist { display: flex; flex-direction: column; gap: 11px; flex: 1; }
.ed-dist-row { display: flex; align-items: center; gap: 10px; font-size: 12px; }
.ed-dist-label {
  flex: 0 0 86px; color: var(--ink-3);
  font-family: var(--mono); font-size: 10px;
  letter-spacing: .08em; text-transform: uppercase;
}
.ed-dist-bar {
  flex: 1; height: 8px; border-radius: 4px;
  background: var(--paper-2); position: relative; overflow: hidden;
}
.ed-dist-fill { position: absolute; inset: 0 auto 0 0; background: var(--ink); border-radius: 4px; }
.ed-dist-fill.signal { background: var(--signal); }
.ed-dist-fill.warn { background: #c89020; }
.ed-dist-fill.neg { background: #a3543e; }
.ed-dist-n {
  flex: 0 0 32px; text-align: right;
  font-family: var(--mono); font-size: 11px; color: var(--ink-2); font-weight: 500;
}

.ed-sech {
  font-family: var(--serif); font-weight: 400;
  font-size: 28px; line-height: 1.1; letter-spacing: -.022em;
  color: var(--ink); margin: 30px 0 14px;
}
.ed-sech em { font-style: italic; color: var(--ink-3); }

.resume-box {
  background: var(--card-2) !important;
  border: 1px solid var(--rule) !important;
  border-radius: 10px !important;
  font-family: var(--mono) !important;
  color: var(--ink-3) !important;
}
mark { background: var(--signal-soft) !important; color: var(--signal-deep) !important; padding: 1px 4px; border-radius: 3px; }

.streamlit-expanderHeader p { font-family: var(--serif) !important; font-size: 16px !important; }

@media (max-width: 1180px) {
  .ed-c4 { grid-column: span 6; } .ed-c8 { grid-column: span 12; }
  .ed-c3 { grid-column: span 6; }
  .ed-pagehead h1 { font-size: 42px; }
  .ed-pagehead { flex-direction: column; align-items: flex-start; }
  .ed-lede { text-align: left; }
}
@media (max-width: 760px) {
  .ed-c3, .ed-c4, .ed-c6, .ed-c8 { grid-column: span 12; }
  .ed-hero-name { font-size: 36px; }
  .ed-hero-top { flex-direction: column; }
  .ed-hero-grid { grid-template-columns: 1fr; }
  .ed-hg-cell { border-right: none; border-bottom: 1px solid var(--rule); }
  .ed-hg-cell:last-child { border-bottom: none; }
}
</style>
""")

# Final layout + spacing polish — loads last, wins everything
st.html("""<style>
/* ── Width: use nearly full viewport, no wasted margins ── */
.block-container {
  max-width: min(1440px, 97vw) !important;
  padding: 0 1rem 3rem !important;
  margin: 0 auto !important;
}

/* ── Wizard step cards: centered at 760px, breathe without being wide ── */
.step-card {
  max-width: 760px !important;
  margin: 0 auto !important;
  padding: 2rem 2.25rem 1.75rem !important;
}
.stepper { max-width: 760px; margin: 0 auto; }

/* ── Results bento: tighter gaps, less padding per tile ── */
.ed-bento { gap: 10px !important; margin-bottom: 10px !important; }
.ed-tile { padding: 14px 16px !important; border-radius: 14px !important; }
.ed-tile-head { margin-bottom: 10px !important; }

/* ── Page head: less vertical air ── */
.ed-pagehead { margin: 4px 0 18px !important; gap: 24px !important; }
.ed-pagehead h1 { font-size: 48px !important; max-width: 100% !important; }
.ed-lede { max-width: 260px !important; }

/* ── JD strip: tighter ── */
.ed-jdstrip { padding: 14px 18px !important; margin-bottom: 12px !important; border-radius: 12px !important; }

/* ── Score stat: reduce whitespace ── */
.ed-stat-big { font-size: 44px !important; margin: 8px 0 5px !important; }

/* ── Hero name: a touch smaller so it fits without overflow ── */
.ed-hero-name { font-size: 38px !important; }
.ed-hero-tldr {
  font-size: 18px !important;
  padding: 12px 0 14px !important;
  margin: 4px 0 12px !important;
}
.ed-hero-tldr::before { font-size: 38px !important; top: 2px !important; }
.ed-hg-cell { padding: 10px 14px !important; }
.ed-hg-v { font-size: 18px !important; }

/* ── Candidate list rows: tighter ── */
.ed-crow { padding: 9px 12px !important; gap: 10px !important; }
.ed-clist { border-radius: 10px !important; }

/* ── Question tiles: compact ── */
.ed-qitem { padding: 10px 12px !important; border-radius: 10px !important; gap: 10px !important; }
.ed-qnum { width: 28px !important; height: 28px !important; font-size: 19px !important; }
.ed-qbody { font-size: 14px !important; }
.ed-qlist { gap: 7px !important; }

/* ── Insight / distribution tiles ── */
.ed-insight-quote { font-size: 22px !important; }
.ed-dist { gap: 8px !important; }

/* ── Section headings ── */
.ed-sech { font-size: 24px !important; margin: 20px 0 10px !important; }

/* ── Button shape: rectangle over pill ── */
.stButton > button,
[data-testid="stButton"] > button,
[data-testid^="stBaseButton"] {
  border-radius: 8px !important;
}

/* ── JD preview: neutral border, no red left accent ── */
.review-jd {
  border: 1px solid #d9d3c5 !important;
  border-left: 1px solid #d9d3c5 !important;
  background: #f9f7f2 !important;
}

/* ── Step header typography (used in steps 1–3) ── */
.step-hd {
  margin-bottom: 1.25rem;
}
.step-hd .step-badge {
  font-family: 'Inter Tight', 'Inter', sans-serif;
  font-size: 0.68rem;
  font-weight: 700;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: var(--signal, #c2410c);
  margin-bottom: 0.4rem;
}
.step-hd .step-title {
  font-family: 'Newsreader', 'Georgia', serif;
  font-size: 2rem;
  font-weight: 600;
  line-height: 1.15;
  color: var(--ink, #161513);
  margin: 0 0 0.5rem;
}
.step-hd .step-sub {
  font-family: 'Inter Tight', 'Inter', sans-serif;
  font-size: 0.875rem;
  color: var(--ink-3, #6b6760);
  line-height: 1.5;
  margin: 0;
}

/* ── Word-count hint chip ── */
.wc-hint {
  display: inline-block;
  font-family: 'Inter Tight', sans-serif;
  font-size: 0.75rem;
  font-weight: 500;
  padding: 2px 8px;
  border-radius: 4px;
  margin-bottom: 0.75rem;
}
.wc-ok   { background: #dcfce7; color: #166534; }
.wc-warn { background: #fef9c3; color: #854d0e; }

/* ── Review tiles (step 3) ── */
.review-row {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 10px;
  margin-bottom: 1.1rem;
}
.review-tile {
  background: var(--surface-container-low, #f4f1ea);
  border: 1px solid var(--rule, #d9d3c5);
  border-radius: 10px;
  padding: 0.8rem 1rem;
}
.review-tile-label {
  font-family: 'Inter Tight', sans-serif;
  font-size: 0.7rem;
  font-weight: 700;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: var(--ink-3, #6b6760);
  margin-bottom: 0.3rem;
}
.review-tile-value {
  font-family: 'Newsreader', serif;
  font-size: 1.75rem;
  font-weight: 600;
  color: var(--ink, #161513);
  line-height: 1;
}
.review-jd {
  font-family: 'Inter Tight', sans-serif;
  font-size: 0.82rem;
  color: var(--ink-3, #6b6760);
  background: var(--surface-container-low, #f4f1ea);
  border: 1px solid var(--rule, #d9d3c5);
  border-radius: 10px;
  padding: 0.9rem 1rem;
  white-space: pre-wrap;
  margin-bottom: 1.1rem;
  max-height: 140px;
  overflow-y: auto;
}
.step-divider {
  height: 1px;
  background: var(--rule, #d9d3c5);
  margin: 1rem 0;
}
</style>""")


# ───────────────────────────────────────────────────────────────
# CHROME — slim top bar + stepper
# ───────────────────────────────────────────────────────────────
api_key_valid, api_key_err = validate_api_key(_RAW_KEY)
used_calls, max_calls = get_rate_usage()


def _render_top_bar():
    if api_key_valid:
        api_pill = (
            '<span class="tb-pill ok">'
            '<span class="material-symbols-outlined" style="font-size:13px">verified</span> API Ready'
            '</span>'
        )
    else:
        api_pill = (
            '<span class="tb-pill err">'
            '<span class="material-symbols-outlined" style="font-size:13px">error</span> API Missing'
            '</span>'
        )
    rate_pill = f'<span class="tb-pill info">{used_calls}/{max_calls} runs</span>'
    st.markdown(f"""
    <div class="top-bar">
      <div class="top-bar-icon">R</div>
      <div>
        <div class="top-bar-title">Résumé <em style="font-style:italic;color:var(--ink-3);">Screener</em></div>
      </div>
      <div class="top-bar-status">
        {api_pill}
        {rate_pill}
      </div>
    </div>
    """, unsafe_allow_html=True)


def _render_stepper(active: int):
    parts = []
    for i, label in enumerate(STEP_LABELS, start=1):
        node_cls = "is-active" if i == active else ("is-done" if i < active else "")
        glyph = "✓" if i < active else str(i)
        parts.append(
            f'<div class="step-node {node_cls}">'
            f'<div class="step-node-disc">{glyph}</div>'
            f'<div class="step-node-label">{label}</div>'
            f'</div>'
        )
        if i < len(STEP_LABELS):
            conn_cls = " is-done" if i < active else ""
            parts.append(f'<div class="step-connector{conn_cls}"></div>')
    st.markdown(f'<div class="stepper">{"".join(parts)}</div>', unsafe_allow_html=True)


_render_top_bar()


# ───────────────────────────────────────────────────────────────
# WIZARD DISPATCH
# ───────────────────────────────────────────────────────────────
_step = st.session_state["wizard_step"]
_shell_cls = "wizard-shell is-results" if _step == 4 else "wizard-shell"
st.markdown(f'<div class="{_shell_cls}">', unsafe_allow_html=True)

# For steps 1–3: constrain the block-container to a centered 820 px column.
# This is the only reliable way to constrain Streamlit native widgets (text_area,
# file_uploader, buttons) since they cannot be contained inside arbitrary HTML divs.
# Step 4 overrides this back to full-width inside _render_step_4().
if _step != 4:
    st.html("""<style>
    .block-container {
      max-width: 820px !important;
      padding: 1.5rem 2rem 4rem !important;
      margin: 0 auto !important;
    }
    .stepper { max-width: 100% !important; margin: 0 0 1.5rem !important; }
    </style>""")

_render_stepper(_step)


# ─── Step 1 — Job Description ──────────────────────────────────
def _render_step_1():
    st.html("""
    <div class="step-hd">
      <div class="step-badge">STEP 1 OF 4</div>
      <div class="step-title">Define the role</div>
      <div class="step-sub">Paste the job description you want candidates screened against.
      Include required skills, years of experience, and responsibilities for best results.</div>
    </div>""")

    jd_raw = st.text_area(
        "jd",
        value=st.session_state.get("jd_text", ""),
        height=240,
        placeholder="Paste the full job description — skills, experience level, responsibilities…",
        label_visibility="collapsed",
        key="jd_input_w1",
    )
    if jd_raw != st.session_state.get("jd_text", ""):
        st.session_state["jd_text"] = jd_raw

    jd_valid, jd_err = validate_jd(jd_raw)
    if jd_raw.strip():
        wc = len(jd_raw.split())
        if jd_valid:
            cls = "wc-ok" if wc >= 50 else "wc-warn"
            note = f"✓ {wc} words — looks good" if wc >= 50 else f"⚠ {wc} words — aim for 50+ for accurate scoring"
        else:
            cls = "wc-warn"
            note = f"⚠ {jd_err}"
        st.markdown(f'<span class="wc-hint {cls}">{note}</span>', unsafe_allow_html=True)

    _, c_btn = st.columns([3, 1])
    with c_btn:
        if st.button("Continue →", type="primary", use_container_width=True,
                     disabled=not jd_valid, key="step1_continue"):
            _goto(2)


# ─── Step 2 — Upload Resumes ───────────────────────────────────
def _render_step_2():
    st.html("""
    <div class="step-hd">
      <div class="step-badge">STEP 2 OF 4</div>
      <div class="step-title">Upload candidate resumes</div>
      <div class="step-sub">Drop in PDF resumes — up to 12 files, 10 MB each.
      Files are validated for format and integrity before processing.</div>
    </div>""")

    uploaded = st.file_uploader(
        "resumes", type=["pdf"], accept_multiple_files=True,
        label_visibility="collapsed", key="resumes_w2",
    )

    if uploaded:
        valid_files, file_errors = validate_resume_batch(uploaded)
        st.session_state["cached_resumes"] = [
            {"name": f.name, "size": getattr(f, "size", 0), "bytes": f.getvalue()}
            for f in valid_files
        ]
        st.session_state["cached_file_errors"] = file_errors

    cached = st.session_state.get("cached_resumes", [])
    cached_errors = st.session_state.get("cached_file_errors", [])
    n_valid = len(cached)

    for fe in cached_errors:
        st.markdown(f'<div class="banner banner-warn">⚠ {escape_html(fe)}</div>', unsafe_allow_html=True)

    if n_valid == 0:
        st.markdown('<div class="banner banner-info">Upload PDFs above to get started.</div>', unsafe_allow_html=True)
    elif n_valid == 1:
        st.markdown('<div class="banner banner-ok">✓ 1 resume ready to screen</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="banner banner-ok">✓ {n_valid} resumes ready to screen</div>', unsafe_allow_html=True)

    if cached:
        pills = "".join(
            f'<span class="file-pill">📄 {escape_html(sanitize_filename(c["name"]))}'
            f'<span class="file-pill-size">{round(c["size"] / 1024, 1)} KB</span></span>'
            for c in cached
        )
        st.markdown(f'<div class="file-pill-row" style="margin-top:.5rem;">{pills}</div>',
                    unsafe_allow_html=True)

    c_back, c_continue = st.columns([1, 1])
    with c_back:
        if st.button("← Back", type="secondary", use_container_width=True, key="step2_back"):
            _goto(1)
    with c_continue:
        if st.button("Continue →", type="primary", use_container_width=True,
                     disabled=(n_valid < 1), key="step2_continue"):
            _goto(3)


# ─── Step 3 — Review & Analyze ─────────────────────────────────
def _run_analysis():
    """Read JD + cached resumes, run pipeline, store results, advance to step 4."""
    jd_text = st.session_state.get("jd_text", "")
    cached = st.session_state.get("cached_resumes", [])
    if not cached or not jd_text:
        return

    results = []
    analysis_errors = []
    successful_calls = 0

    prog_bar = st.progress(0)
    status_ph = st.empty()
    jd_clean = sanitize_text_input(jd_text, max_chars=8000)

    n = len(cached)
    for idx, c in enumerate(cached):
        sname = sanitize_filename(c["name"])
        prog_bar.progress(idx / n)
        status_ph.markdown(
            f'<div class="prog-label"><span class="prog-dot"></span> Analyzing {idx+1}/{n}: <strong>{escape_html(sname)}</strong></div>',
            unsafe_allow_html=True,
        )

        bio = io.BytesIO(c["bytes"])
        bio.name = c["name"]
        _, resume_raw = extract_text_from_pdf(bio)
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
                labels = {"auth": "Authentication failed", "quota": "Quota/rate limit",
                          "timeout": "Request timed out", "network": "Network error",
                          "parse": "Output error", "unknown": "Unexpected error"}
                analysis_errors.append(f"**{sname}** [{labels.get(result.error_type, 'Error')}]: {result.error_message}")
            result_dict = result.analysis
            score, rec = compute_score(result_dict)
            llm_ok, err_type = result.success, result.error_type

        candidate_name = (
            sname.replace(".pdf", "").replace(".PDF", "")
            .replace("_", " ").replace("-", " ").replace("  ", " ").strip()
        )
        candidate_name = re.sub(r'\s*\(\d+\)\s*$', '', candidate_name).strip().title()
        results.append({
            "candidate_name": candidate_name, "filename": sname,
            "score": score, "recommendation": rec,
            "analysis": result_dict, "resume_text": resume_clean,
            "parse_error": parse_failed, "llm_success": llm_ok, "error_type": err_type,
        })

    prog_bar.progress(1.0)
    status_ph.markdown(
        f'<div class="prog-label">✓ Done — {successful_calls}/{n} analyzed successfully</div>',
        unsafe_allow_html=True,
    )
    record_api_calls(successful_calls)
    results.sort(key=lambda r: r["score"], reverse=True)
    st.session_state["results"] = results
    st.session_state["analysis_errors"] = analysis_errors
    _goto(4)


def _render_step_3():
    jd_text = st.session_state.get("jd_text", "")
    cached = st.session_state.get("cached_resumes", [])
    n_valid = len(cached)
    rate_ok, rate_err_msg = check_rate_limit(n_valid) if n_valid else (False, "No resumes to analyze.")
    ready = bool(jd_text.strip()) and n_valid >= 1 and rate_ok and api_key_valid
    est = n_valid * 0.001

    st.html("""
    <div class="step-hd">
      <div class="step-badge">STEP 3 OF 4</div>
      <div class="step-title">Review and analyze</div>
      <div class="step-sub">Confirm everything looks right, then start scoring.
      Results are deterministic — the same inputs always produce the same scores.</div>
    </div>""")

    # Stat tiles
    st.markdown(f"""
    <div class="review-row">
      <div class="review-tile">
        <div class="review-tile-label">Resumes queued</div>
        <div class="review-tile-value">{n_valid}</div>
      </div>
      <div class="review-tile">
        <div class="review-tile-label">Estimated API cost</div>
        <div class="review-tile-value">~${est:.3f}</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # File pills
    if cached:
        pills = "".join(
            f'<span class="file-pill">📄 {escape_html(sanitize_filename(c["name"]))}'
            f'<span class="file-pill-size">{round(c["size"] / 1024, 1)} KB</span></span>'
            for c in cached
        )
        st.markdown(
            f'<div class="sub-label" style="margin-bottom:.4rem;">Candidates</div>'
            f'<div class="file-pill-row" style="margin-bottom:1rem;">{pills}</div>',
            unsafe_allow_html=True,
        )

    # JD preview
    jd_preview = (jd_text[:400] + "…") if len(jd_text) > 400 else jd_text
    st.markdown(
        f'<div class="sub-label" style="margin-bottom:.4rem;">Job description</div>'
        f'<div class="review-jd">{escape_html(jd_preview)}</div>',
        unsafe_allow_html=True,
    )

    # Scoring weights card
    st.markdown("""
    <div style="background:#f4f1ea;border:1px solid #d9d3c5;border-radius:10px;
                padding:.85rem 1rem;margin-bottom:1rem;">
      <div class="sub-label" style="margin-bottom:.55rem;">Scoring weights</div>
      <div style="display:grid;grid-template-columns:1fr auto;gap:4px 16px;
                  font-family:'Inter Tight',sans-serif;font-size:.8rem;">
        <span style="color:#6b6760;">Skills match</span>  <span style="font-weight:700;color:#166534;">35%</span>
        <span style="color:#6b6760;">Experience</span>    <span style="font-weight:700;color:#1e40af;">30%</span>
        <span style="color:#6b6760;">Keywords</span>      <span style="font-weight:700;color:#6b21a8;">20%</span>
        <span style="color:#6b6760;">Gap penalty</span>   <span style="font-weight:700;color:#9b1c1c;">10%</span>
        <span style="color:#6b6760;">Seniority fit</span> <span style="font-weight:700;color:#134e4a;">5%</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # API & rate banners
    if api_key_valid:
        masked = _RAW_KEY[:8] + "·····" + _RAW_KEY[-4:]
        st.markdown(
            f'<div class="banner banner-ok" style="font-size:.75rem;margin-bottom:.5rem;">'
            f'✓ <strong>API key ready</strong> · <code style="font-size:.7rem;">{masked}</code></div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="banner banner-err" style="font-size:.75rem;margin-bottom:.5rem;">'
            f'✗ <strong>API key missing:</strong> {escape_html(api_key_err)}</div>',
            unsafe_allow_html=True,
        )

    if not rate_ok and n_valid > 0:
        st.markdown(
            f'<div class="banner banner-warn" style="font-size:.75rem;margin-bottom:.5rem;">'
            f'⚠ {escape_html(rate_err_msg)}</div>',
            unsafe_allow_html=True,
        )

    st.markdown('<div class="step-divider"></div>', unsafe_allow_html=True)

    c_back, c_continue = st.columns([1, 3])
    with c_back:
        if st.button("← Back", type="secondary", use_container_width=True, key="step3_back"):
            _goto(2)
    with c_continue:
        cta_label = f"Analyze {n_valid} resume{'s' if n_valid != 1 else ''}  ·  ~${est:.3f}"
        if st.button(cta_label, type="primary", use_container_width=True,
                     disabled=not ready, key="step3_run"):
            _run_analysis()


# ─── Step 4 — Results ──────────────────────────────────────────
def _initials(name: str) -> str:
    parts = [p for p in (name or "—").strip().split() if p]
    if not parts:
        return "—"
    if len(parts) == 1:
        return parts[0][:2].upper()
    return (parts[0][0] + parts[-1][0]).upper()


def _split_name(name: str):
    parts = (name or "").strip().split()
    if not parts:
        return "—", ""
    return parts[0], " ".join(parts[1:])


def _grade(score: int):
    if score >= 85: return ("Strong", "")
    if score >= 70: return ("Match", "")
    if score >= 55: return ("Maybe", "mid")
    return ("Weak", "low")


def _ctag(score: int):
    if score >= 85: return ("STRONG", "strong")
    if score >= 70: return ("MATCH", "match")
    if score >= 55: return ("MAYBE", "maybe")
    return ("WEAK", "weak")


def _infer_jd_role(jd_text: str) -> str:
    import re as _re
    if not jd_text:
        return "Open Role"
    _SKIP = {
        "requirements", "requirement", "responsibilities", "responsibility",
        "qualifications", "qualification", "overview", "description", "about",
        "skills", "experience", "benefits", "compensation", "location", "notes",
        "introduction", "summary", "details", "information", "core", "basic",
        "key", "position", "role", "job", "hiring", "we are", "the team",
    }
    for line in jd_text.splitlines():
        line = line.strip(" \t#*•-—_:|")
        if not line:
            continue
        words = line.split()
        if 2 <= len(words) <= 10 and len(line) <= 80 and not line.endswith(":"):
            lower_words = {w.lower().rstrip("s") for w in words}
            if not (lower_words & _SKIP):
                return line
    # Fallback: extract from "looking for a X" / "hiring a X" pattern
    m = _re.search(
        r'(?:looking for|hiring|seeking) (?:an? )([\w\s]{5,60}?)(?:\s+with|\s+who|\s+to|\s*[,\.])',
        jd_text, _re.I,
    )
    if m:
        return m.group(1).strip().title()
    return "Open Role"


def _fit_display(sc: int):
    """Returns (label, css_class) consistent with _ctag() thresholds."""
    if sc >= 85: return ("Strong Fit", "strong")
    if sc >= 70: return ("Match", "match")
    if sc >= 55: return ("Maybe", "maybe")
    return ("Not a Fit", "weak")


def _infer_must_haves(results) -> list:
    from collections import Counter
    bag = Counter()
    for r in results:
        for s in (r["analysis"].get("skills_match") or []):
            bag[s] += 1
        for s in (r["analysis"].get("keywords_matched") or []):
            bag[s] += 1
    return [s for s, _ in bag.most_common(6)]


def _verdict_sentence(results, strong: int, maybe: int) -> str:
    n = len(results)
    if n == 0:
        return "No candidates parsed."
    if strong == 0 and maybe == 0:
        return f"<em>None</em> of the {n} applicants cleared the bar — widen the funnel."
    if strong == 0:
        return f"Of {n} applicants, <em>{maybe}</em> are worth a screen call. None are interview-ready out of the gate."
    return (
        f"Of {n} applicants, <em>{strong}</em> warrant interviews this week. "
        f"The rest split between stack mismatch and seniority gap."
    )


def _render_step_4():
    st.markdown("""<style>
.block-container { max-width: min(1440px, 97vw) !important; padding: 0 1rem 3rem !important; }
</style>""", unsafe_allow_html=True)

    results = st.session_state.get("results")
    if not results:
        st.session_state["wizard_step"] = 1
        st.rerun()
        return

    errors    = st.session_state.get("analysis_errors", [])
    ok_r      = [r for r in results if r["llm_success"]]
    avg_sc    = round(sum(r["score"] for r in ok_r) / len(ok_r)) if ok_r else 0
    top_score = max((r["score"] for r in results), default=0)
    median    = sorted(r["score"] for r in results)[len(results)//2] if results else 0
    strong    = sum(1 for r in ok_r if r["score"] >= 85)
    matches   = sum(1 for r in ok_r if 70 <= r["score"] < 85)
    maybes    = sum(1 for r in ok_r if 55 <= r["score"] < 70)

    jd_text   = st.session_state.get("jd_text", "") or ""
    role      = _infer_jd_role(jd_text)
    must_haves= _infer_must_haves(ok_r)

    from datetime import datetime
    today = datetime.now().strftime("%b %d")
    if strong + matches > 0:
        slash_count = strong + matches
        head_tail = f"worth your<br/>calendar this week."
    else:
        slash_count = len(results)
        head_tail = f"in the funnel.<br/>None ready yet."
    st.html(f"""
    <div class="ed-pagehead">
      <h1>
        <em>{len(results)} résumés in.</em><br/>
        <span class="slash">{slash_count}</span> {head_tail}
      </h1>
      <div class="ed-lede">
        <span class="ts">Run summary · {today}</span>
        <span class="summary">
          <em>{strong} strong</em>, <span class="maybe">{matches + maybes} maybe</span>, the rest a pass.
        </span>
        <span class="meta">Median {median} · top {top_score} · {len(results)} parsed</span>
      </div>
    </div>
    """)

    a1, a2, a3, _ = st.columns([1, 1, 1, 2])
    with a1:
        if st.button("← Edit", type="secondary", use_container_width=True, key="r_edit"):
            _goto(1)
    with a2:
        if st.button("↺ Re-run", type="secondary", use_container_width=True, key="r_rerun"):
            _goto(3)
    with a3:
        if st.button("Clear", type="secondary", use_container_width=True, key="r_clear"):
            st.session_state["results"] = None
            st.session_state["analysis_errors"] = []
            _goto(1)

    for err in errors:
        st.markdown(f'<div class="banner banner-warn">⚠ {err}</div>', unsafe_allow_html=True)

    # ─── JD strip ──────────────────────────────────────────────
    musts_html = "".join(f'<span class="ed-jd-must">{escape_html(m)}</span>' for m in must_haves) \
                 or '<span class="ed-jd-must" style="color:var(--ink-4);">— no must-haves inferred —</span>'
    st.html(f"""
    <div class="ed-jdstrip">
      <div class="ed-jd-main">
        <div class="ed-jd-row1">
          <div class="ed-jd-role">{escape_html(role)}</div>
          <div class="ed-jd-meta">
            <span>{len(results)} applicants</span>
            <span class="sep"></span>
            <span>{len(jd_text.split())} words in JD</span>
          </div>
        </div>
        <div class="ed-jd-musts">
          <span class="ed-jd-musts-k">Top signals</span>
          {musts_html}
        </div>
      </div>
      <div class="ed-jd-corner">
        Median<b>{median}<span style="font-family:var(--mono);font-size:14px;font-style:normal;color:var(--ink-4);">/100</span></b>
      </div>
    </div>
    """)

    # ─── Stat row (4 tiles) ────────────────────────────────────
    kw_miss_avg = round(sum(len(r["analysis"].get("keywords_missing") or []) for r in ok_r) / len(ok_r)) if ok_r else 0
    st.html(f"""
    <div class="ed-bento">
      <div class="ed-tile ed-c3">
        <div class="ed-tile-label">Total screened</div>
        <div class="ed-stat-big">{len(results)}</div>
        <div class="ed-stat-sub">résumés parsed and scored</div>
      </div>
      <div class="ed-tile ed-c3 signal">
        <div class="ed-tile-label">Strong matches</div>
        <div class="ed-stat-big">{strong}</div>
        <div class="ed-stat-sub">≥ 85 score · interview-ready</div>
      </div>
      <div class="ed-tile ed-c3">
        <div class="ed-tile-label">Average score</div>
        <div class="ed-stat-big">{avg_sc}<span class="unit">/100</span></div>
        <div class="ed-stat-sub">across the parsed pool</div>
      </div>
      <div class="ed-tile ed-c3">
        <div class="ed-tile-label">Avg. gaps</div>
        <div class="ed-stat-big">{kw_miss_avg}<span class="unit">missing</span></div>
        <div class="ed-stat-sub">keywords missing per résumé</div>
      </div>
    </div>
    """)

    # ─── Active candidate selector (state) ─────────────────────
    if "active_candidate_id" not in st.session_state:
        st.session_state["active_candidate_id"] = 0
    active_id = st.session_state["active_candidate_id"]
    if active_id >= len(results):
        active_id = 0
        st.session_state["active_candidate_id"] = 0
    active = results[active_id]
    aa     = active["analysis"]

    # ─── Hero candidate + Interview probes (bento row) ─────────
    sc      = active["score"]
    sc_grade, sc_grade_cls = _grade(sc)
    sc_color = "var(--signal)" if sc >= 85 else ("var(--ink)" if sc >= 70 else ("#c89020" if sc >= 55 else "#a3543e"))
    fit_lbl, fit_cls = _fit_display(sc)
    first, last = _split_name(active["candidate_name"])
    tldr    = aa.get("tldr", "") or ""
    if tldr in ("", "Analysis failed"):
        tldr = "No summary available — review the resume manually."
    skills  = aa.get("skills_match") or []
    missing = aa.get("missing_skills") or []
    questions = aa.get("interview_questions") or []
    exp     = aa.get("experience_relevance", "—") or "—"
    sen     = aa.get("seniority_match", "—") or "—"

    def _chip_label(s, maxlen=26):
        return escape_html(s[:maxlen] + "…") if len(s) > maxlen else escape_html(s)
    skills_chips = "".join(f'<span class="ed-chip match" title="{escape_html(s)}">{_chip_label(s)}</span>' for s in skills[:5]) + \
                   "".join(f'<span class="ed-chip miss" title="{escape_html(s)}">{_chip_label(s)}</span>' for s in missing[:3])
    if not skills_chips:
        skills_chips = '<span style="font-size:11px;color:var(--ink-4);font-family:var(--mono);">No skills extracted</span>'

    import math
    ring_size = 156
    ring_r = (ring_size - 16) / 2
    ring_c = 2 * math.pi * ring_r
    ring_off = ring_c - (sc / 100) * ring_c

    questions_html = ""
    for i, q in enumerate(questions[:4]):
        questions_html += f"""
        <div class="ed-qitem">
          <div class="ed-qnum">{i+1}</div>
          <div><div class="ed-qbody">{escape_html(q)}</div></div>
        </div>"""
    if not questions_html:
        questions_html = '<div style="font-size:12px;color:var(--ink-4);font-family:var(--mono);">No interview questions generated.</div>'

    st.html(f"""
    <div class="ed-bento">
      <div class="ed-tile ed-hero ed-c8">
        <div class="ed-tile-head">
          <div class="ed-tile-label">Top candidate</div>
          <div class="ed-ctag {fit_cls}" style="font-size:10px;padding:3px 8px;">{fit_lbl.upper()}</div>
        </div>

        <div class="ed-hero-top">
          <div style="flex:1;min-width:0;">
            <div class="ed-hero-id">
              <span class="rank">#{active_id + 1}</span>
              <span style="color:var(--ink-4);font-size:11px;">{escape_html(role)}</span>
            </div>
            <h2 class="ed-hero-name">{escape_html(first)} <em>{escape_html(last)}</em></h2>
            <div class="ed-hero-meta">
              <span>{exp} experience</span>
              <span class="dot"></span>
              <span>Seniority: {sen}</span>
            </div>
          </div>
          <div class="ed-ringwrap">
            <svg width="{ring_size}" height="{ring_size}">
              <circle cx="{ring_size/2}" cy="{ring_size/2}" r="{ring_r}" fill="none" stroke="var(--paper-2)" stroke-width="10"/>
              <circle cx="{ring_size/2}" cy="{ring_size/2}" r="{ring_r}" fill="none"
                stroke="{sc_color}" stroke-width="10"
                stroke-dasharray="{ring_c:.2f}" stroke-dashoffset="{ring_off:.2f}"
                stroke-linecap="round"/>
            </svg>
            <div class="ed-score-text">
              <div class="ed-score-num">{sc}</div>
              <div class="ed-score-of">of 100</div>
              <div class="ed-score-grade {sc_grade_cls}">{sc_grade}</div>
            </div>
          </div>
        </div>

        <div class="ed-hero-tldr">{escape_html(tldr)}</div>

        <div class="ed-hero-grid">
          <div class="ed-hg-cell">
            <div class="ed-hg-k">Skills · {len(skills)} matched, {len(missing)} missing</div>
            <div class="ed-chips" style="margin-top:6px;">{skills_chips}</div>
          </div>
          <div class="ed-hg-cell">
            <div class="ed-hg-k">Experience</div>
            <div class="ed-hg-v">{exp}</div>
            <div class="ed-hg-v small">relevance to this role</div>
          </div>
          <div class="ed-hg-cell">
            <div class="ed-hg-k">Seniority</div>
            <div class="ed-hg-v">{sen}</div>
            <div class="ed-hg-v small">vs. required level</div>
          </div>
        </div>
      </div>

      <div class="ed-tile ed-c4">
        <div class="ed-tile-head">
          <div class="ed-tile-label">Interview probes</div>
        </div>
        <div style="font-size:12px;color:var(--ink-3);line-height:1.5;margin-bottom:14px;">
          Tailored to <span style="color:var(--ink);font-weight:500;">{escape_html(first)}</span>'s gaps and strengths.
        </div>
        <div class="ed-qlist">{questions_html}</div>
      </div>
    </div>
    """)

    # ─── Candidate list + side rail ────────────────────────────
    lcol, rcol = st.columns([2, 1], gap="small")
    with lcol:
        st.markdown('<div class="ed-tile-label" style="margin-bottom:12px;letter-spacing:.16em;text-transform:uppercase;font-family:var(--mono);font-size:10px;color:var(--ink-4);">Ranked candidates</div>', unsafe_allow_html=True)
        rows_html = ""
        for i, r in enumerate(results):
            rs = r["score"]
            tag_label, tag_cls = _ctag(rs)
            exp_i = (r["analysis"].get("experience_relevance", "—") or "—")
            sen_i = (r["analysis"].get("seniority_match", "—") or "—")
            initials = _initials(r["candidate_name"])
            active_cls = "active" if i == active_id else ""
            rows_html += f"""
            <div class="ed-crow {active_cls}">
              <div class="ed-crank">{str(i+1).zfill(2)}</div>
              <div class="ed-cavatar">{initials}</div>
              <div class="ed-cinfo">
                <div class="ed-cname">{escape_html(r['candidate_name'])}</div>
                <div class="ed-cmeta">{exp_i.upper()} EXP · {sen_i.upper()}</div>
              </div>
              <div class="ed-cscore-cell">
                <div class="ed-cscore-num">{rs}</div>
                <div class="ed-cscore-bar"><div class="ed-cscore-fill" style="width:{rs}%;"></div></div>
              </div>
              <div class="ed-ctag {tag_cls}">{tag_label}</div>
              <div class="ed-carrow">→</div>
            </div>"""
        st.html(f'<div class="ed-clist">{rows_html}</div>')

        st.markdown('<div style="margin-top:14px;font-family:var(--mono);font-size:10px;letter-spacing:.14em;text-transform:uppercase;color:var(--ink-4);margin-bottom:8px;">Inspect candidate</div>', unsafe_allow_html=True)
        n_btns = min(len(results), 6)
        bcols = st.columns(n_btns)
        for i, col in enumerate(bcols):
            if i >= len(results):
                break
            with col:
                fname = (results[i]["candidate_name"] or "—").split()[0][:12]
                sc_i = results[i]["score"]
                active_marker = " ▸" if i == active_id else ""
                label = f"#{i+1} {fname}{active_marker}"
                if st.button(label, key=f"pick_{i}", use_container_width=True,
                             type="primary" if i == active_id else "secondary"):
                    st.session_state["active_candidate_id"] = i
                    st.rerun()
        if len(results) > 6:
            st.caption(f"Showing first 6 · {len(results) - 6} more in the comparison table below")

    with rcol:
        buckets = [
            ("90–100", 90, 100, "signal"),
            ("75–89",  75, 89,  ""),
            ("60–74",  60, 74,  "warn"),
            ("0–59",    0, 59,  "neg"),
        ]
        bmax = max(1, max(sum(1 for r in results if b[1] <= r["score"] <= b[2]) for b in buckets))
        dist_rows = ""
        for label, lo, hi, cls in buckets:
            n = sum(1 for r in results if lo <= r["score"] <= hi)
            dist_rows += f"""
            <div class="ed-dist-row">
              <div class="ed-dist-label">{label}</div>
              <div class="ed-dist-bar"><div class="ed-dist-fill {cls}" style="width:{(n/bmax)*100:.0f}%;"></div></div>
              <div class="ed-dist-n">{n}</div>
            </div>"""
        st.html(f"""
        <div class="ed-tile dark">
          <div class="ed-tile-head"><div class="ed-tile-label">Recruiter brief</div></div>
          <div class="ed-insight-quote">{_verdict_sentence(results, strong, matches + maybes)}</div>
          <div class="ed-insight-foot">
            <span>Updated just now</span>
            <span>{len(results)} parsed</span>
          </div>
        </div>
        <div class="ed-tile" style="margin-top:14px;">
          <div class="ed-tile-head"><div class="ed-tile-label">Score distribution</div></div>
          <div class="ed-dist">{dist_rows}</div>
        </div>
        """)

    # ─── Full comparison table + drilldown ─────────────────────
    st.markdown('<div class="ed-sech"><em>Full</em> comparison</div>', unsafe_allow_html=True)
    rows = []
    for r in results:
        a = r["analysis"]
        fl, _ = _fit_display(r["score"])
        rows.append({
            "Rank":        results.index(r) + 1,
            "Candidate":   r["candidate_name"],
            "Score":       r["score"],
            "Fit":         fl,
            "Experience":  a.get("experience_relevance", "—"),
            "Seniority":   a.get("seniority_match", "—"),
            "Skills ✓":    len(a.get("skills_match") or []),
            "KW ✓":        len(a.get("keywords_matched") or []),
            "Top Strength": ((a.get("strengths") or ["—"])[0])[:55],
        })
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True, column_config={
        "Rank":  st.column_config.NumberColumn("#", width="small"),
        "Score": st.column_config.ProgressColumn("Score /100", min_value=0, max_value=100, format="%d"),
        "Skills ✓": st.column_config.NumberColumn("Skills", width="small"),
        "KW ✓":     st.column_config.NumberColumn("Keywords", width="small"),
    })
    st.download_button(
        "Download CSV",
        data=df.to_csv(index=False),
        file_name="screening_results.csv",
        mime="text/csv",
    )
    st.markdown('</div>', unsafe_allow_html=True)

    if len(results) > 1:
        st.markdown('<div class="ed-sech"><em>All</em> candidates</div>', unsafe_allow_html=True)
        for rank, r in enumerate(results, 1):
            sc = r["score"]
            a = r["analysis"]
            fl, _ = _fit_display(sc)
            with st.expander(f"#{str(rank).zfill(2)}  ·  {r['candidate_name']}  ·  {fl}  ·  {sc}/100", expanded=False):
                lc, rc2 = st.columns([1, 3])
                with lc:
                    sc_c = "var(--primary)" if sc >= 70 else ("#b45309" if sc >= 45 else "var(--error)")
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
                        <div style="position:absolute;inset:0;display:flex;align-items:center;justify-content:center;font-family:var(--font-display);font-size:1.1rem;font-weight:700;color:{sc_c};letter-spacing:-0.02em;">{sc}</div>
                      </div>
                      <div style="font-size:.65rem;color:var(--outline);margin-bottom:5px;">Fit Score</div>
                      <span class="chip chip-surface">{a.get("experience_relevance","—")} Exp</span>
                      <span class="chip chip-surface">{a.get("seniority_match","—")}</span>
                      {"" if not r.get('parse_error') else '<br><span class="err-pill">parse error</span>'}
                    </div>
                    """)
                    t = a.get("tldr", "")
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
                        q_h2 = "".join(f'<div class="q-item"><span class="q-num">Q{i+1}.</span><span style="font-style:italic;">{escape_html(q)}</span></div>' for i, q in enumerate(qs))
                        st.markdown(f'<div class="sub-label" style="margin-top:.75rem;color:var(--secondary);">Interview Probes</div>{q_h2}', unsafe_allow_html=True)

                if r["resume_text"]:
                    st.markdown('<div class="divider" style="margin:.75rem 0;"></div>', unsafe_allow_html=True)
                    st.markdown('<div class="sub-label">Resume Preview</div>', unsafe_allow_html=True)
                    all_kw = list(set((kh or []) + (sk or [])))
                    hl = highlight_keywords(r["resume_text"][:2000], all_kw)
                    st.markdown(f'<div class="resume-box">{hl}</div>', unsafe_allow_html=True)


# Dispatch
if _step == 1:
    _render_step_1()
elif _step == 2:
    _render_step_2()
elif _step == 3:
    _render_step_3()
else:
    _render_step_4()


st.markdown('</div>', unsafe_allow_html=True)  # /wizard-shell


# Footer
st.markdown("""
<div class="footer">AI Resume Screener</div>
""", unsafe_allow_html=True)