"""Unit tests for text extraction (no pypdf needed for txt/md paths)."""

import pytest

from app.extraction import SUPPORTED_EXTENSIONS, UnsupportedFormatError, extract_text


def test_extract_txt():
    assert extract_text("notes.txt", b"hello\nworld") == "hello\nworld"


def test_extract_md():
    assert extract_text("doc.md", b"# Title\n\nbody") == "# Title\n\nbody"


def test_extract_extension_case_insensitive():
    assert extract_text("DOC.MD", b"hi") == "hi"


def test_unsupported_extension_raises():
    with pytest.raises(UnsupportedFormatError):
        extract_text("image.png", b"\x89PNG")


def test_supported_extensions_cover_contract():
    assert {".txt", ".md", ".pdf"} <= SUPPORTED_EXTENSIONS


def test_pdf_without_pypdf_raises_clear_error(monkeypatch):
    # Simulate pypdf missing: extraction must fail loudly, not silently.
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "pypdf":
            raise ImportError("No module named 'pypdf'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(UnsupportedFormatError, match="pypdf"):
        extract_text("doc.pdf", b"%PDF-1.4 fake")
