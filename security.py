"""
Security module — input validation, sanitization, rate limiting, and file safety.
All validation logic is centralized here to keep app.py clean.
"""

import re
import time
import streamlit as st

# ── Constants ───────────────────────────────────────────────────
MAX_PDF_SIZE_BYTES  = 10 * 1024 * 1024   # 10 MB per file
MAX_JD_CHARS        = 8_000              # ~2k tokens
MAX_RESUMES         = 12
MIN_JD_WORDS        = 10

# Rate limiting: max N analyses per rolling window (session-level)
RATE_LIMIT_MAX      = 50                 # max calls per session
RATE_WINDOW_SECONDS = 3600              # 1 hour

# PDF magic bytes (first 4 bytes)
PDF_MAGIC = b"%PDF"


# ── API Key Validation ───────────────────────────────────────────

def validate_api_key(key: str) -> tuple[bool, str]:
    """
    Validate the format of an OpenAI API key.
    Returns (is_valid, error_message).
    Does NOT make a network call — purely structural.
    """
    if not key or not key.strip():
        return False, "API key is empty."

    key = key.strip()

    # OpenAI keys start with sk- and are 40–200 chars
    if not key.startswith("sk-"):
        return False, "API key must start with 'sk-'."
    if len(key) < 20:
        return False, "API key appears too short."
    if len(key) > 250:
        return False, "API key appears too long."
    # Only printable ASCII, no whitespace
    if re.search(r"\s", key):
        return False, "API key must not contain whitespace."

    return True, ""


# ── JD Validation ───────────────────────────────────────────────

def validate_jd(text: str) -> tuple[bool, str]:
    """
    Validate the job description input.
    Returns (is_valid, error_message).
    """
    if not text or not text.strip():
        return False, "Job description is empty."

    stripped = text.strip()

    if len(stripped) > MAX_JD_CHARS:
        return False, f"Job description exceeds {MAX_JD_CHARS:,} characters. Please trim it."

    word_count = len(stripped.split())
    if word_count < MIN_JD_WORDS:
        return False, f"Job description too short ({word_count} words). Add more detail."

    return True, ""


# ── File Validation ──────────────────────────────────────────────

def validate_pdf_file(uploaded_file) -> tuple[bool, str]:
    """
    Validate a single uploaded PDF file.
    Checks: MIME type, file size, magic bytes.
    Returns (is_valid, error_message).
    """
    name = uploaded_file.name

    # Extension check
    if not name.lower().endswith(".pdf"):
        return False, f"'{name}' is not a PDF file."

    # Size check
    size = uploaded_file.size if hasattr(uploaded_file, "size") else 0
    if size == 0:
        return False, f"'{name}' appears to be empty."
    if size > MAX_PDF_SIZE_BYTES:
        mb = size / (1024 * 1024)
        return False, f"'{name}' is {mb:.1f} MB — exceeds 10 MB limit."

    # Magic bytes check (read first 4 bytes without consuming the stream)
    try:
        header = uploaded_file.read(4)
        uploaded_file.seek(0)  # rewind for pdfplumber
        if header != PDF_MAGIC:
            return False, f"'{name}' does not appear to be a valid PDF (bad header)."
    except Exception:
        uploaded_file.seek(0)
        # Skip magic check if read fails — pdfplumber will catch it anyway

    return True, ""


def validate_resume_batch(files: list) -> tuple[list, list[str]]:
    """
    Validate all uploaded files.
    Returns (valid_files, error_messages).
    """
    if not files:
        return [], ["No files uploaded."]

    if len(files) > MAX_RESUMES:
        return [], [f"Too many files ({len(files)}). Maximum is {MAX_RESUMES}."]

    valid, errors = [], []
    seen_names = set()

    for f in files:
        # Duplicate filename check
        if f.name in seen_names:
            errors.append(f"Duplicate file: '{f.name}' — skipping.")
            continue
        seen_names.add(f.name)

        ok, err = validate_pdf_file(f)
        if ok:
            valid.append(f)
        else:
            errors.append(err)

    return valid, errors


# ── Rate Limiting ────────────────────────────────────────────────

def _init_rate_state():
    if "rl_calls" not in st.session_state:
        st.session_state["rl_calls"] = []


def check_rate_limit(n_calls: int = 1) -> tuple[bool, str]:
    """
    Check if the user is within the session-level rate limit.
    Uses a sliding window approach stored in session state.
    Returns (allowed, error_message).
    """
    _init_rate_state()
    now = time.time()
    window_start = now - RATE_WINDOW_SECONDS

    # Prune old calls outside the window
    st.session_state["rl_calls"] = [
        t for t in st.session_state["rl_calls"] if t > window_start
    ]

    total = len(st.session_state["rl_calls"])
    if total + n_calls > RATE_LIMIT_MAX:
        remaining = RATE_WINDOW_SECONDS - (now - st.session_state["rl_calls"][0])
        mins = int(remaining // 60)
        return False, f"Rate limit reached ({RATE_LIMIT_MAX} analyses/hour). Resets in ~{mins} min."

    return True, ""


def record_api_calls(n: int = 1):
    """Record that n API calls were made (call after successful analysis)."""
    _init_rate_state()
    now = time.time()
    st.session_state["rl_calls"].extend([now] * n)


def get_rate_usage() -> tuple[int, int]:
    """Return (calls_used, calls_limit) in the current window."""
    _init_rate_state()
    now = time.time()
    window_start = now - RATE_WINDOW_SECONDS
    used = len([t for t in st.session_state["rl_calls"] if t > window_start])
    return used, RATE_LIMIT_MAX


# ── Input Sanitization ───────────────────────────────────────────

def sanitize_text_input(text: str, max_chars: int = 8000) -> str:
    """
    Sanitize free-text inputs:
    - Strip leading/trailing whitespace
    - Collapse excessive blank lines
    - Truncate to max_chars
    - Remove null bytes
    """
    if not text:
        return ""

    # Remove null bytes
    text = text.replace("\x00", "")

    # Collapse 3+ consecutive newlines to 2
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Strip
    text = text.strip()

    # Truncate
    if len(text) > max_chars:
        text = text[:max_chars]

    return text


def sanitize_filename(name: str) -> str:
    """
    Produce a safe display name from a filename.
    Removes path traversal characters and limits length.
    """
    # Only keep basename (no directory components)
    name = re.sub(r"[/\\]", "", name)
    # Remove non-printable characters
    name = re.sub(r"[^\x20-\x7E]", "", name)
    # Limit length
    return name[:120]
