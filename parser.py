"""
PDF resume parser — extracts and cleans text from uploaded PDF files.
"""

import pdfplumber
from utils import clean_whitespace, truncate_text


def extract_text_from_pdf(pdf_file) -> tuple[str, str]:
    """
    Extract text from an uploaded PDF file.

    Args:
        pdf_file: Streamlit UploadedFile object.

    Returns:
        (filename, cleaned_text) — text is truncated to ~3500 chars.
        Returns (filename, "") on failure.
    """
    filename = pdf_file.name

    try:
        with pdfplumber.open(pdf_file) as pdf:
            pages_text = []
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    pages_text.append(page_text)

            if not pages_text:
                return filename, ""

            full_text = "\n".join(pages_text)
            full_text = clean_whitespace(full_text)
            full_text = truncate_text(full_text, max_chars=3500)
            return filename, full_text

    except Exception:
        return filename, ""
