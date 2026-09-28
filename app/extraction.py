"""Plain-text extraction from uploaded files (.txt / .md / .pdf)."""

from __future__ import annotations

import io
from pathlib import Path

SUPPORTED_EXTENSIONS = {".txt", ".md", ".markdown", ".pdf"}


class UnsupportedFormatError(ValueError):
    pass


def extract_text(filename: str, data: bytes) -> str:
    """Extract UTF-8 text from raw upload bytes based on file extension."""
    ext = Path(filename).suffix.lower()
    if ext in {".txt", ".md", ".markdown"}:
        return data.decode("utf-8", errors="replace")
    if ext == ".pdf":
        return _extract_pdf(data)
    raise UnsupportedFormatError(
        f"Unsupported file type {ext!r} for {filename!r}. "
        f"Supported: {sorted(SUPPORTED_EXTENSIONS)}"
    )


def _extract_pdf(data: bytes) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise UnsupportedFormatError(
            "PDF support requires the 'pypdf' package, which is not installed."
        ) from exc
    reader = io.BytesIO(data)
    pdf = PdfReader(reader)
    pages = [page.extract_text() or "" for page in pdf.pages]
    return "\n\n".join(pages)
