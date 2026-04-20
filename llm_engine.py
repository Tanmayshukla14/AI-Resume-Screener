"""
LLM Engine — LangChain-powered resume analysis with structured Pydantic output.

Architecture:
  - Uses langchain-openai ChatOpenAI with gpt-4o-mini
  - Structured output enforced via Pydantic model (with_structured_output)
  - System + Human message separation for cleaner prompt design
  - Automatic retry with exponential backoff via langchain_core Retry
  - Token usage tracked and returned for transparency
  - Graceful error classification: auth, quota, timeout, parse, network
"""

from __future__ import annotations

import logging
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.runnables import RunnableConfig

from utils import truncate_text

logger = logging.getLogger(__name__)

# ── Pydantic Output Schema ───────────────────────────────────────

class ResumeAnalysis(BaseModel):
    """
    Validated, type-safe output schema for a single resume analysis.
    LangChain enforces this schema via OpenAI function calling / JSON schema.
    """
    tldr: str = Field(
        description="One-sentence executive summary of the candidate's value proposition for this role.",
        min_length=10,
        max_length=300,
    )
    skills_match: list[str] = Field(
        description="Skills from the JD that the candidate clearly demonstrates. Max 6.",
        max_length=6,
        default_factory=list,
    )
    missing_skills: list[str] = Field(
        description="Key skills required by the JD that are absent from the resume. Max 6.",
        max_length=6,
        default_factory=list,
    )
    experience_relevance: Literal["High", "Medium", "Low"] = Field(
        description="How relevant the candidate's experience level and domain is to this role."
    )
    strengths: list[str] = Field(
        description="2-3 concrete strengths observed in the resume relative to the JD.",
        max_length=3,
        default_factory=list,
    )
    gaps: list[str] = Field(
        description="2-3 meaningful gaps or concerns relative to the JD requirements.",
        max_length=3,
        default_factory=list,
    )
    interview_questions: list[str] = Field(
        description="2 targeted interview questions designed to probe the candidate's key gaps or validate their top strength. Make them specific to THIS candidate.",
        max_length=2,
        default_factory=list,
    )
    keywords_matched: list[str] = Field(
        description="Important JD keywords/phrases present in the resume. Max 8.",
        max_length=8,
        default_factory=list,
    )
    keywords_missing: list[str] = Field(
        description="Important JD keywords/phrases absent from the resume. Max 8.",
        max_length=8,
        default_factory=list,
    )
    culture_fit_notes: str = Field(
        description="One sentence on soft skills, communication style, or culture signals observed (or 'No signals detected').",
        default="No signals detected.",
        max_length=300,
    )
    seniority_match: Literal["Below", "Match", "Overqualified", "Unclear"] = Field(
        description="Whether the candidate's seniority level matches the role.",
        default="Unclear",
    )

    @field_validator("skills_match", "missing_skills", "strengths", "gaps",
                     "interview_questions", "keywords_matched", "keywords_missing",
                     mode="before")
    @classmethod
    def coerce_list(cls, v):
        """Accept a single string or None, normalize to a list of strings."""
        if v is None:
            return []
        if isinstance(v, str):
            return [v] if v.strip() else []
        return [str(item).strip() for item in v if item]


# Empty analysis returned on hard errors — built as plain dict to avoid Pydantic validation
def _empty_analysis(reason: str) -> dict:
    return {
        "tldr": "Analysis failed",
        "skills_match": [],
        "missing_skills": [],
        "experience_relevance": "Low",
        "strengths": [],
        "gaps": [reason],
        "interview_questions": [],
        "keywords_matched": [],
        "keywords_missing": [],
        "culture_fit_notes": "No signals detected.",
        "seniority_match": "Unclear",
    }


# ── Prompts ──────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are an expert AI recruitment analyst with 15 years of hiring experience across tech, finance, and operations.
Your task is to rigorously evaluate a candidate resume against a job description.

Evaluation principles:
- Base ALL observations strictly on text present in the resume — never infer or hallucinate experience.
- Be concrete and specific: reference actual job titles, company names, technologies, or metrics from the resume.
- Be honest about gaps — do not soften genuine misaligns.
- Interview questions must be personalized to THIS candidate, not generic.
- Your output will be used by hiring managers to make real decisions — accuracy matters.
"""

HUMAN_PROMPT_TEMPLATE = """\
## Job Description
{jd}

---

## Candidate Resume
{resume}

---

Analyze the resume against the job description and return your evaluation.
"""


# ── LLM Chain Factory ────────────────────────────────────────────

def _build_chain(api_key: str):
    """
    Build the LangChain chain: ChatOpenAI → structured output parser.
    Uses OpenAI function calling for guaranteed schema compliance.
    """
    llm = ChatOpenAI(
        model="openai/gpt-4o-mini",
        temperature=0.1,          # low temp = consistent, factual output
        max_tokens=1200,
        max_retries=2,            # built-in LangChain retry for transient errors
        timeout=35,
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
    )
    return llm.with_structured_output(ResumeAnalysis, method="function_calling")


# ── Public API ───────────────────────────────────────────────────

class AnalysisResult(BaseModel):
    """Wrapper returned to the caller, includes token metadata."""
    analysis: dict
    success: bool
    error_type: str = ""          # "auth" | "quota" | "timeout" | "network" | "parse" | "unknown"
    error_message: str = ""
    model_used: str = "gpt-4o-mini"


def analyze_resume(jd_text: str, resume_text: str, api_key: str) -> AnalysisResult:
    """
    Analyze a resume against a JD using LangChain + OpenAI structured output.

    Args:
        jd_text:     The job description (already sanitized).
        resume_text: Extracted resume text (already sanitized).
        api_key:     OpenAI API key (already validated).

    Returns:
        AnalysisResult with the analysis dict and metadata.
    """
    # Input guards (defense-in-depth — security.py validates first but we re-check)
    if not api_key or not api_key.strip():
        return AnalysisResult(
            analysis=_empty_analysis("No API key provided."),
            success=False,
            error_type="auth",
            error_message="No API key provided.",
        )
    if not resume_text or not resume_text.strip():
        return AnalysisResult(
            analysis=_empty_analysis("Resume text is empty or could not be extracted."),
            success=False,
            error_type="parse",
            error_message="Empty resume text.",
        )

    # Truncate to token-safe sizes
    jd_trimmed     = truncate_text(jd_text.strip(),     max_chars=2000)
    resume_trimmed = truncate_text(resume_text.strip(), max_chars=4000)

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=HUMAN_PROMPT_TEMPLATE.format(
            jd=jd_trimmed,
            resume=resume_trimmed,
        )),
    ]

    try:
        chain = _build_chain(api_key)
        result: ResumeAnalysis = chain.invoke(
            messages,
            config=RunnableConfig(run_name="resume_analysis"),
        )
        return AnalysisResult(
            analysis=result.model_dump(),
            success=True,
        )

    # ── Classified error handling ──
    except Exception as exc:
        return _classify_error(exc)


def _classify_error(exc: Exception) -> AnalysisResult:
    """
    Map exceptions to a clean error type + safe user-facing message.
    Never exposes raw stack traces or internal details.
    """
    err_str = str(exc).lower()

    # Auth
    if "401" in err_str or "authentication" in err_str or "invalid api key" in err_str:
        return AnalysisResult(
            analysis=_empty_analysis("Invalid API key — check your .env file."),
            success=False,
            error_type="auth",
            error_message="Invalid OpenRouter API key.",
        )
    # Quota / rate limit
    if "429" in err_str or "quota" in err_str or "rate limit" in err_str or "insufficient_quota" in err_str or "payment" in err_str:
        return AnalysisResult(
            analysis=_empty_analysis("OpenRouter API quota exceeded or rate limited."),
            success=False,
            error_type="quota",
            error_message="OpenAI rate limit or quota exceeded.",
        )
    # Timeout
    if "timeout" in err_str or "timed out" in err_str:
        return AnalysisResult(
            analysis=_empty_analysis("Request to OpenAI timed out — try again."),
            success=False,
            error_type="timeout",
            error_message="Request timed out.",
        )
    # Network
    if "connection" in err_str or "network" in err_str or "unreachable" in err_str:
        return AnalysisResult(
            analysis=_empty_analysis("Cannot reach OpenAI — check internet connection."),
            success=False,
            error_type="network",
            error_message="Network connection error.",
        )
    # Pydantic / parse
    if "validation" in err_str or "pydantic" in err_str or "json" in err_str:
        return AnalysisResult(
            analysis=_empty_analysis("AI returned unexpected output format."),
            success=False,
            error_type="parse",
            error_message="Output format error.",
        )
    # Unknown
    logger.warning("Unclassified LLM error: %s", str(exc)[:200])
    return AnalysisResult(
        analysis=_empty_analysis("An unexpected error occurred during analysis."),
        success=False,
        error_type="unknown",
        error_message="Unexpected error. Check logs.",
    )
