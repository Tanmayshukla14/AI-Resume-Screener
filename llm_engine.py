"""
LLM engine — single OpenAI API call per resume for structured analysis.
Works natively with standard OpenAI API keys.
"""

import requests
from utils import safe_parse_json, truncate_text

PROMPT_TEMPLATE = """You are an AI recruitment evaluator.

Analyze the resume against the job description.

Return ONLY valid JSON. No explanation. No markdown.

Format:
{{"tldr": "One sentence summary of candidate's value prop", "skills_match": ["..."], "missing_skills": ["..."], "experience_relevance": "High/Medium/Low", "strengths": ["..."], "gaps": ["..."], "interview_questions": ["..."], "keywords_matched": ["..."], "keywords_missing": ["..."]}}

Rules:
- Exactly 2-3 items for 'strengths'
- Exactly 2-3 items for 'gaps'
- Exactly 2 custom 'interview_questions' designed to drill into their specific gaps or verify their biggest strength
- Max 5 items for skills lists
- Be concise and punchy
- Do not hallucinate
- Use only information present in resume

Job Description:
{jd}

Resume:
{resume}"""

EMPTY_ANALYSIS = {
    "tldr": "Analysis failed",
    "skills_match": [],
    "missing_skills": [],
    "experience_relevance": "Low",
    "strengths": [],
    "gaps": ["Analysis failed"],
    "interview_questions": [],
    "keywords_matched": [],
    "keywords_missing": [],
}

REQUIRED_KEYS = {"strengths", "gaps", "interview_questions", "tldr"}


def _validate_analysis(result: dict) -> dict:
    """Ensure all required keys exist with correct types."""
    validated = dict(EMPTY_ANALYSIS)
    for key, default_val in EMPTY_ANALYSIS.items():
        val = result.get(key, default_val)
        if isinstance(default_val, list):
            if isinstance(val, list):
                validated[key] = [str(item) for item in val[:5]]
            else:
                validated[key] = default_val
        elif isinstance(default_val, str):
            validated[key] = str(val) if val else default_val
    return validated


def analyze_resume(jd_text: str, resume_text: str, api_key: str) -> dict:
    """
    Single API call to analyze a resume against a JD via OpenAI.
    Uses gpt-4o-mini for fast, consistent JSON responses.
    """
    if not api_key or not api_key.strip():
        return {**EMPTY_ANALYSIS, "gaps": ["No OpenAI API key provided"]}
    if not resume_text or not resume_text.strip():
        return {**EMPTY_ANALYSIS, "gaps": ["Empty or unreadable resume"]}

    jd_text = truncate_text(jd_text, max_chars=1500)
    resume_text = truncate_text(resume_text, max_chars=3500)
    prompt = PROMPT_TEMPLATE.format(jd=jd_text, resume=resume_text)

    try:
        resp = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.1,
                "max_tokens": 1000,
            },
            timeout=30,
        )

        if resp.status_code == 401:
            return {**EMPTY_ANALYSIS, "gaps": ["Invalid OpenAI API key."]}
        if resp.status_code == 429:
            return {**EMPTY_ANALYSIS, "gaps": ["OpenAI Rate limit or Insufficient Quota."]}
        
        resp.raise_for_status()
        data = resp.json()

        if "choices" not in data or not data["choices"]:
            return {**EMPTY_ANALYSIS, "gaps": ["OpenAI returned an empty response."]}

        content = data["choices"][0].get("message", {}).get("content", "")
        result = safe_parse_json(content)

        if result and isinstance(result, dict):
            if any(k in result for k in REQUIRED_KEYS):
                return _validate_analysis(result)

        return {**EMPTY_ANALYSIS, "gaps": ["OpenAI returned invalid JSON format."]}

    except requests.exceptions.Timeout:
        return {**EMPTY_ANALYSIS, "gaps": ["Request to OpenAI timed out."]}
    except requests.exceptions.ConnectionError:
        return {**EMPTY_ANALYSIS, "gaps": ["Cannot connect to OpenAI. Check internet."]}
    except Exception as e:
        error_msg = str(e)[:120]
        return {**EMPTY_ANALYSIS, "gaps": [f"API error: {error_msg}"]}
