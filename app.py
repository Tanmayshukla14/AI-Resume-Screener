"""
AI Resume Screening System — Streamlit App
Optimized for minimal API usage (≤1 call per resume).
"""

import os
import time
import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from parser import extract_text_from_pdf
from llm_engine import analyze_resume
from scoring import compute_score
from utils import highlight_keywords, escape_html

# Load API keys from .env
load_dotenv(override=True)
API_KEY = os.getenv("OPENAI_API_KEY", "")

# ─── Page Config ────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Resume Screener",
    page_icon="📄",
    layout="wide",
)

# ─── Clean Dark Theme CSS ───────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root {
    --bg: #0e1117;
    --surface: #161b22;
    --border: #30363d;
    --text: #e6edf3;
    --text-muted: #8b949e;
    --accent: #58a6ff;
    --green: #3fb950;
    --yellow: #d29922;
    --red: #f85149;
}

* { font-family: 'Inter', -apple-system, sans-serif; }

.stApp { background: var(--bg); }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: var(--surface);
    border-right: 1px solid var(--border);
}

/* Inputs */
.stTextInput > div > div > input,
.stTextArea textarea,
.stSelectbox > div > div {
    background: var(--bg) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    border-radius: 8px !important;
}
.stTextInput > div > div > input:focus,
.stTextArea textarea:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 1px var(--accent) !important;
}

/* Labels */
label { color: var(--text-muted) !important; font-weight: 500 !important; font-size: 0.85rem !important; }
.stMarkdown p { color: var(--text); }

/* Headings */
h1, h2, h3 { color: var(--text) !important; font-family: 'Inter', sans-serif !important; }
h1 { font-weight: 800 !important; font-size: 1.8rem !important; }
h2 { font-weight: 700 !important; font-size: 1.3rem !important; }

/* Page header */
.page-header {
    padding: 12px 0 4px;
    border-bottom: 1px solid var(--border);
    margin-bottom: 24px;
}
.page-header h1 { margin: 0; font-size: 1.6rem; color: var(--text); }
.page-header-sub { color: var(--text-muted); font-size: 0.85rem; margin-top: 2px; }

/* Stats row */
.stats-row {
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 10px;
    margin: 16px 0 20px;
}
.stat-box {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 16px;
    text-align: center;
}
.stat-box .stat-num {
    font-size: 1.6rem;
    font-weight: 700;
    color: var(--text);
}
.stat-box .stat-lbl {
    font-size: 0.7rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-top: 2px;
}

/* Tags */
.tag {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 100px;
    font-size: 0.75rem;
    font-weight: 500;
    margin: 2px;
}
.tag-green { background: rgba(63,185,80,0.12); color: var(--green); border: 1px solid rgba(63,185,80,0.25); }
.tag-red { background: rgba(248,81,73,0.12); color: var(--red); border: 1px solid rgba(248,81,73,0.25); }
.tag-blue { background: rgba(88,166,255,0.12); color: var(--accent); border: 1px solid rgba(88,166,255,0.25); }
.tag-yellow { background: rgba(210,153,34,0.12); color: var(--yellow); border: 1px solid rgba(210,153,34,0.25); }

/* Score circle */
.score-circle {
    width: 72px;
    height: 72px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.5rem;
    font-weight: 700;
    margin: 0 auto 6px;
}
.sc-high { background: rgba(63,185,80,0.12); color: var(--green); border: 2px solid rgba(63,185,80,0.35); }
.sc-mid { background: rgba(210,153,34,0.12); color: var(--yellow); border: 2px solid rgba(210,153,34,0.35); }
.sc-low { background: rgba(248,81,73,0.12); color: var(--red); border: 2px solid rgba(248,81,73,0.35); }

/* Section label */
.sec-label {
    color: var(--text-muted);
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    font-weight: 600;
    margin: 12px 0 4px;
}

/* Resume preview */
.resume-box {
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 14px;
    font-size: 0.8rem;
    color: var(--text-muted);
    max-height: 260px;
    overflow-y: auto;
    line-height: 1.65;
}

/* Highlighted keywords */
mark {
    background: rgba(88,166,255,0.2);
    color: var(--accent);
    padding: 1px 4px;
    border-radius: 3px;
}

/* Buttons */
.stButton > button {
    background: var(--accent) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 10px 24px !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
}
.stButton > button:hover {
    opacity: 0.9;
}
.stButton > button:disabled {
    background: var(--border) !important;
    color: var(--text-muted) !important;
}
.stDownloadButton > button {
    background: transparent !important;
    color: var(--accent) !important;
    border: 1px solid var(--accent) !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
}

/* Divider */
.hr { height: 1px; background: var(--border); margin: 20px 0; }

/* Progress text */
.prog-text { color: var(--accent); font-size: 0.85rem; padding: 6px 0; font-weight: 500; }

/* Expander */
.streamlit-expanderHeader {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text) !important;
    font-weight: 600 !important;
}

/* Empty state */
.empty-state { text-align: center; padding: 60px 20px; }
.empty-icon { font-size: 2.5rem; margin-bottom: 12px; opacity: 0.4; }
.empty-title { color: var(--text-muted); font-size: 1rem; }
.empty-sub { color: #484f58; font-size: 0.8rem; margin-top: 4px; }

/* Sidebar section */
.sb-section {
    color: var(--text-muted);
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    font-weight: 600;
    margin: 14px 0 6px;
    padding-bottom: 4px;
    border-bottom: 1px solid var(--border);
}
</style>
""", unsafe_allow_html=True)

# ─── Header ─────────────────────────────────────────────────────
st.markdown("""
<div class="page-header">
    <h1>📄 AI Resume Screener</h1>
    <div class="page-header-sub">Paste a JD, upload resumes, get ranked candidates instantly</div>
</div>
""", unsafe_allow_html=True)

# ─── Sidebar (Settings) ─────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    
    st.markdown(
        "**API Configuration**\n\n"
        "This app runs natively on **OpenAI**.\n"
        "Ensure your `.env` file contains:\n"
        "`OPENAI_API_KEY=sk-proj-...`\n\n"
        "Model used: `gpt-4o-mini`",
        help="Delays and fallback logic have been removed for a clean, fast experience natively on OpenAI."
    )

# ─── Main Content ───────────────────────────────────────────────
if not API_KEY:
    st.error("⚠️ **OpenAI API Key Missing**\nPlease add `OPENAI_API_KEY=\"sk-proj-...\"` to your `.env` file to enable analysis.")

jd_text = st.text_area(
    "📝 Job Description",
    height=150,
    placeholder="Paste the job description here...",
)

st.markdown("#### 📎 Upload Resumes (3–6 PDFs)")
uploaded_files = st.file_uploader(
    "Upload resumes",
    type=["pdf"],
    accept_multiple_files=True,
    label_visibility="collapsed",
)

file_count = len(uploaded_files) if uploaded_files else 0
if uploaded_files:
    if file_count < 3:
        st.warning(f"Upload at least 3 resumes ({file_count} uploaded)")
    elif file_count > 6:
        st.warning(f"Maximum 6 resumes ({file_count} uploaded)")
    else:
        st.success(f"✓ {file_count} resumes ready")

# ─── Analyze ────────────────────────────────────────────────────
ready = API_KEY and jd_text.strip() and uploaded_files and 3 <= file_count <= 6

if st.button("Analyze Resumes", disabled=not ready, use_container_width=True):
    results = []
    progress_bar = st.progress(0)
    status_text = st.empty()

    for idx, pdf_file in enumerate(uploaded_files):
        progress_bar.progress(idx / file_count)
        status_text.markdown(
            f'<div class="prog-text">Analyzing {idx + 1}/{file_count}: {pdf_file.name}</div>',
            unsafe_allow_html=True,
        )

        filename, resume_text = extract_text_from_pdf(pdf_file)
        analysis = analyze_resume(jd_text, resume_text, API_KEY)

        score, recommendation = compute_score(analysis)
        candidate_name = filename.replace(".pdf", "").replace("_", " ").replace("-", " ").title()

        results.append({
            "candidate_name": candidate_name,
            "filename": filename,
            "score": score,
            "recommendation": recommendation,
            "analysis": analysis,
            "resume_text": resume_text,
        })

    progress_bar.progress(1.0)
    status_text.markdown('<div class="prog-text">✓ Analysis complete</div>', unsafe_allow_html=True)
    results.sort(key=lambda r: r["score"], reverse=True)
    st.session_state["results"] = results

# ─── Results ────────────────────────────────────────────────────
if "results" in st.session_state and st.session_state["results"]:
    results = st.session_state["results"]

    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)
    st.markdown("## Screening Results")

    # Stats
    avg_score = sum(r["score"] for r in results) / len(results) if results else 0
    strong = sum(1 for r in results if r["score"] >= 70)
    moderate = sum(1 for r in results if 45 <= r["score"] < 70)
    not_fit = sum(1 for r in results if r["score"] < 45)

    st.markdown(f"""
    <div class="stats-row">
        <div class="stat-box"><div class="stat-num">{len(results)}</div><div class="stat-lbl">Candidates</div></div>
        <div class="stat-box"><div class="stat-num">{avg_score:.0f}</div><div class="stat-lbl">Avg Score</div></div>
        <div class="stat-box"><div class="stat-num" style="color:var(--green)">{strong}</div><div class="stat-lbl">Strong</div></div>
        <div class="stat-box"><div class="stat-num" style="color:var(--yellow)">{moderate}</div><div class="stat-lbl">Moderate</div></div>
        <div class="stat-box"><div class="stat-num" style="color:var(--red)">{not_fit}</div><div class="stat-lbl">Not Fit</div></div>
    </div>
    """, unsafe_allow_html=True)

    # Table
    table_data = []
    for r in results:
        table_data.append({
            "Candidate": r["candidate_name"],
            "Score": r["score"],
            "Strengths": ", ".join(r["analysis"].get("strengths", [])[:3]),
            "Gaps": ", ".join(r["analysis"].get("gaps", [])[:3]),
            "Fit": r["recommendation"],
        })

    df = pd.DataFrame(table_data)
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Score": st.column_config.ProgressColumn("Score", min_value=0, max_value=100, format="%d"),
        },
    )

    csv = df.to_csv(index=False)
    st.download_button("📥 Download CSV", data=csv, file_name="screening_results.csv", mime="text/csv")

    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)
    st.markdown("## Candidate Details")

    for r in results:
        score = r["score"]
        analysis = r["analysis"]
        sc_class = "sc-high" if score >= 70 else "sc-mid" if score >= 45 else "sc-low"

        with st.expander(f"{r['candidate_name']}  —  {r['recommendation']}  ({score}/100)"):
            c1, c2 = st.columns([1, 3])

            with c1:
                exp_raw = str(analysis.get("experience_relevance", "Low"))
                st.markdown(f"""
                <div style="text-align:center">
                    <div class="score-circle {sc_class}">{score}</div>
                    <div style="color:var(--text-muted);font-size:0.75rem;margin-top:6px">Experience</div>
                    <div style="color:var(--text);font-weight:600;font-size:0.85rem">{exp_raw}</div>
                </div>
                """, unsafe_allow_html=True)

                tldr = analysis.get("tldr", "")
                if tldr and tldr != "Analysis failed":
                    st.markdown(f"""
                    <div style="margin-top: 24px; padding: 14px; background: rgba(88,166,255,0.06); border-radius: 8px; border: 1px solid rgba(88,166,255,0.15); text-align: center; font-size: 0.8rem; font-style: italic; color: var(--text); line-height: 1.4;">
                        "{escape_html(tldr)}"
                    </div>
                    """, unsafe_allow_html=True)

            with c2:
                skills = analysis.get("skills_match", [])
                missing = analysis.get("missing_skills", [])
                kw_matched = analysis.get("keywords_matched", [])
                kw_missing = analysis.get("keywords_missing", [])

                if skills:
                    tags = "".join(f'<span class="tag tag-green">{escape_html(s)}</span>' for s in skills)
                    st.markdown(f'<div class="sec-label">Skills Matched</div>{tags}', unsafe_allow_html=True)

                if missing:
                    tags = "".join(f'<span class="tag tag-red">{escape_html(s)}</span>' for s in missing)
                    st.markdown(f'<div class="sec-label">Missing Skills</div>{tags}', unsafe_allow_html=True)

                strengths = analysis.get("strengths", [])
                if strengths:
                    st.markdown('<div class="sec-label">Strengths</div>', unsafe_allow_html=True)
                    for s in strengths:
                        st.markdown(f"✅ {s}")

                gaps = analysis.get("gaps", [])
                if gaps:
                    st.markdown('<div class="sec-label">Gaps</div>', unsafe_allow_html=True)
                    for g in gaps:
                        st.markdown(f"⚠️ {g}")

                if kw_matched:
                    tags = "".join(f'<span class="tag tag-blue">{escape_html(k)}</span>' for k in kw_matched)
                    st.markdown(f'<div class="sec-label">Keywords Matched</div>{tags}', unsafe_allow_html=True)

                if kw_missing:
                    tags = "".join(f'<span class="tag tag-yellow">{escape_html(k)}</span>' for k in kw_missing)
                    st.markdown(f'<div class="sec-label">Keywords Missing</div>{tags}', unsafe_allow_html=True)

                questions = analysis.get("interview_questions", [])
                if questions:
                    st.markdown('<div class="sec-label" style="color:var(--accent); margin-top: 18px;">💡 Suggested Interview Questions</div>', unsafe_allow_html=True)
                    for q in questions:
                        st.markdown(f"❓ <span style='font-size:0.85rem; color:var(--text);'>*{escape_html(q)}*</span>", unsafe_allow_html=True)

            if r["resume_text"]:
                st.markdown('<div class="hr"></div>', unsafe_allow_html=True)
                st.markdown('<div class="sec-label">Resume Preview</div>', unsafe_allow_html=True)
                all_kw = kw_matched + skills
                highlighted = highlight_keywords(r["resume_text"][:2000], all_kw)
                st.markdown(f'<div class="resume-box">{highlighted}</div>', unsafe_allow_html=True)

# ─── Empty State ────────────────────────────────────────────────
if "results" not in st.session_state and not uploaded_files:
    st.markdown("""
    <div class="empty-state">
        <div class="empty-icon">📄</div>
        <div class="empty-title">Paste a job description and upload resumes to begin</div>
        <div class="empty-sub">Fast screening · 1 API call per resume · Local scoring · CSV export</div>
    </div>
    """, unsafe_allow_html=True)
