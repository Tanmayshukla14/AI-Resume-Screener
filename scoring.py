"""
Local scoring engine — computes candidate scores WITHOUT any additional LLM calls.
Uses all fields from the ResumeAnalysis Pydantic model for a richer score.
"""


def compute_score(analysis: dict) -> tuple[int, str]:
    """
    Compute a 0–100 score from the validated LLM analysis dict.

    Weights:
        Skills match:         35%
        Experience relevance: 30%
        Keyword coverage:     20%
        Gap penalty:          10%
        Seniority match:       5%

    Returns:
        (score, recommendation)  "✅ Strong Fit" / "⚠️ Moderate Fit" / "❌ Not Fit"
    """
    if not analysis:
        return 0, "❌ Not Fit"

    # ── Skills match (35%) ──────────────────────────────────────
    matched = len(analysis.get("skills_match") or [])
    missing = len(analysis.get("missing_skills") or [])
    total_skills = matched + missing
    # Neutral (0.5) when the LLM returned no skill data at all
    skills_ratio = (matched / total_skills) if total_skills > 0 else 0.5

    # ── Experience relevance (30%) ───────────────────────────────
    exp_map = {"high": 1.0, "medium": 0.6, "low": 0.2}
    exp_raw = str(analysis.get("experience_relevance") or "low").lower().strip()
    exp_score = exp_map.get(exp_raw, 0.2)

    # ── Keyword coverage (20%) ───────────────────────────────────
    kw_hit  = len(analysis.get("keywords_matched") or [])
    kw_miss = len(analysis.get("keywords_missing") or [])
    total_kw = kw_hit + kw_miss
    kw_ratio = (kw_hit / total_kw) if total_kw > 0 else 0.5

    # ── Gap penalty (10%) ────────────────────────────────────────
    # 0 gaps → full 10 pts; 3 gaps (LLM max) → 0 pts
    gaps = len(analysis.get("gaps") or [])
    gap_score = max(0.0, 1.0 - (gaps / 3.0))

    # ── Seniority match bonus/penalty (5%) ──────────────────────
    seniority_map = {
        "match":        1.0,
        "unclear":      0.7,   # give benefit of the doubt
        "overqualified":0.5,   # might leave early
        "below":        0.0,
    }
    seniority_raw = str(analysis.get("seniority_match") or "unclear").lower().strip()
    seniority_score = seniority_map.get(seniority_raw, 0.7)

    # ── Weighted total ───────────────────────────────────────────
    score = (
        skills_ratio    * 35
        + exp_score     * 30
        + kw_ratio      * 20
        + gap_score     * 10
        + seniority_score * 5
    )
    score = round(min(100, max(0, score)))

    # ── Recommendation thresholds ────────────────────────────────
    if score >= 70:
        recommendation = "✅ Strong Fit"
    elif score >= 45:
        recommendation = "⚠️ Moderate Fit"
    else:
        recommendation = "❌ Not Fit"

    return score, recommendation
