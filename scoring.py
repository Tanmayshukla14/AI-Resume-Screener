"""
Local scoring engine — computes candidate scores WITHOUT any LLM calls.
"""


def compute_score(analysis: dict) -> tuple[int, str]:
    """
    Compute a 0–100 score from the LLM analysis dict.

    Weights:
        Skills match:         40%
        Experience relevance: 30%
        Keyword match:        20%
        Gap penalty:          10%

    Returns:
        (score, recommendation) where recommendation is
        "Strong Fit" / "Moderate Fit" / "Weak Fit".
    """
    # --- Skills match (40%) ---
    matched = len(analysis.get("skills_match", []))
    missing = len(analysis.get("missing_skills", []))
    total_skills = matched + missing
    skills_ratio = matched / total_skills if total_skills > 0 else 0.0

    # --- Experience relevance (30%) ---
    exp_map = {"high": 1.0, "medium": 0.6, "low": 0.2}
    exp_raw = str(analysis.get("experience_relevance", "low")).lower().strip()
    exp_score = exp_map.get(exp_raw, 0.2)

    # --- Keyword match (20%) ---
    kw_matched = len(analysis.get("keywords_matched", []))
    kw_missing = len(analysis.get("keywords_missing", []))
    total_kw = kw_matched + kw_missing
    kw_ratio = kw_matched / total_kw if total_kw > 0 else 0.0

    # --- Gap penalty (10%) ---
    gaps = len(analysis.get("gaps", []))
    gap_score = max(0.0, 1.0 - (gaps / 5.0))

    # --- Weighted total ---
    score = (
        skills_ratio * 40
        + exp_score * 30
        + kw_ratio * 20
        + gap_score * 10
    )
    score = round(min(100, max(0, score)))

    # --- Recommendation ---
    if score >= 70:
        recommendation = "✅ Strong Fit"
    elif score >= 45:
        recommendation = "⚠️ Moderate Fit"
    else:
        recommendation = "❌ Not Fit"

    return score, recommendation
