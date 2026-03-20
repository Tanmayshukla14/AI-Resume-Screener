"""
Utility functions for safe JSON parsing and keyword highlighting.
"""

import json
import re


def safe_parse_json(text: str) -> dict | None:
    """
    Attempt to parse JSON from LLM output.
    Handles markdown code fences, stray text, and list wrapping.
    Returns parsed dict or None on failure.
    """
    if not text:
        return None

    # Strip markdown code fences (```json ... ``` or ``` ... ```)
    cleaned = re.sub(r"```(?:json)?\s*", "", text)
    cleaned = cleaned.strip()

    # Try direct parse first
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed
        if isinstance(parsed, list) and parsed and isinstance(parsed[0], dict):
            return parsed[0]
    except json.JSONDecodeError:
        pass

    # Fallback: extract first { ... } block
    match = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", cleaned, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group())
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

    return None


def highlight_keywords(text: str, keywords: list[str]) -> str:
    """
    Wrap matched keywords in HTML <mark> tags for display.
    Case-insensitive. Escapes HTML in text first to prevent XSS.
    Skips empty or very short keywords to avoid noisy highlighting.
    """
    if not text or not keywords:
        return escape_html(text) if text else text

    text = escape_html(text)

    for kw in keywords:
        if not kw or len(kw) < 2:
            continue
        pattern = re.compile(re.escape(kw), re.IGNORECASE)
        text = pattern.sub(lambda m: f"<mark>{m.group()}</mark>", text)

    return text


def truncate_text(text: str, max_chars: int = 3500) -> str:
    """Truncate text to max_chars, breaking at last space if possible."""
    if len(text) <= max_chars:
        return text
    truncated = text[:max_chars]
    last_space = truncated.rfind(" ")
    if last_space > max_chars * 0.8:
        return truncated[:last_space]
    return truncated


def clean_whitespace(text: str) -> str:
    """Collapse multiple whitespace characters into single spaces."""
    return re.sub(r"\s+", " ", text).strip()


def escape_html(text: str) -> str:
    """Escape HTML special characters to prevent XSS."""
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )
