# doc_to_concept/llamaparse_client.py
# ─────────────────────────────────────────────────────────────────────
# Wraps the LlamaIndex LlamaParse API for PDF-to-Markdown extraction.
#
# Why LlamaParse instead of a local PDF parser?
#   LlamaParse is purpose-built for complex document layouts including
#   multi-column text, tables, and — critically for this system —
#   mathematical notation. It preserves LaTeX delimiters through the
#   extraction step, satisfying Req 12.4 (no loss of equation syntax).
#
# Math-optimised settings used here:
#   result_type="markdown"  — retains heading hierarchy and equation
#                             blocks better than plain text output.
#   parsing_instruction     — explicit directive to the LlamaParse model
#                             to keep LaTeX delimiters intact rather than
#                             converting them to Unicode approximations.
#
# Requirements satisfied: 8.2 (PDF → Markdown), 8.9 (error → 422),
#                         12.4 (LaTeX preservation).
# ─────────────────────────────────────────────────────────────────────
import os
import pathlib
import tempfile

from llama_parse import LlamaParse


def parse_pdf_to_markdown(pdf_bytes: bytes, filename: str) -> str:
    """
    Send a PDF to LlamaParse and return the extracted Markdown string.

    The function writes the raw PDF bytes to a temporary file because
    LlamaParse's Python client accepts a file path rather than raw bytes.
    The temp file is always deleted in the ``finally`` block regardless
    of success or failure, preventing disk leaks.

    Math-optimised settings:
    - ``result_type="markdown"`` preserves equation structure (headings,
      code fences, inline math) better than plain-text output.
    - ``parsing_instruction`` explicitly instructs the LlamaParse model
      to retain LaTeX delimiters (``$...$`` and ``$$...$$``) so that
      downstream chunking and formula extraction receive intact notation.

    Args:
        pdf_bytes: Raw bytes of the uploaded PDF file.
        filename:  Original filename used in error messages for clarity.

    Returns:
        A single Markdown string joining all pages/sections returned by
        LlamaParse, separated by double newlines.

    Raises:
        ValueError: If LlamaParse returns no content (empty document or
                    unsupported file) or if the API call itself fails.
                    The router catches ValueError and re-raises it as
                    HTTP 422 (Req 8.9).
    """
    # ── Build the LlamaParse client ───────────────────────────────────
    # api_key is read from the environment — never hardcoded (Req 2.4).
    # verbose=False suppresses LlamaParse's own progress output so our
    # structured logs remain readable.
    parser = LlamaParse(
        api_key=os.environ["LLAMA_PARSE_API_KEY"],
        result_type="markdown",
        # This instruction is passed to the LlamaParse model at parse time.
        # It tells the model to treat mathematical expressions as first-class
        # content and preserve their LaTeX representation exactly (Req 12.4).
        parsing_instruction=(
            "Preserve all mathematical formulas and equations in LaTeX notation. "
            "Do not convert equations to plain text or Unicode approximations. "
            "Use $...$ for inline math and $$...$$ for display/block math. "
            "Retain all fraction, summation, integral, and Greek letter notation."
        ),
        verbose=False,
    )

    # ── Write bytes to a temporary file ──────────────────────────────
    # LlamaParse's load_data() accepts a file path, not raw bytes.
    # NamedTemporaryFile with delete=False lets us close the file handle
    # before passing the path to load_data() (required on Windows; safe
    # on all platforms).
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(pdf_bytes)
        tmp_path = tmp.name

    try:
        # ── Call LlamaParse ───────────────────────────────────────────
        # load_data() returns a list of LlamaIndex Document objects.
        # Each Document has a .text attribute containing the extracted
        # Markdown for one logical section of the PDF.
        documents = parser.load_data(tmp_path)

        if not documents:
            # An empty list means LlamaParse processed the file but found
            # no extractable content — likely a scanned image PDF without
            # OCR, a password-protected file, or a corrupt upload.
            raise ValueError(
                f"LlamaParse returned no content for '{filename}'. "
                "The file may be a scanned image PDF, password-protected, "
                "or otherwise unreadable."
            )

        # Join all document sections with a blank line so downstream
        # chunking treats section boundaries as natural split points.
        return "\n\n".join(doc.text for doc in documents)

    except ValueError:
        # Re-raise our own descriptive ValueErrors unchanged so the
        # router can surface them as HTTP 422 with the right message.
        raise

    except Exception as exc:
        # Wrap unexpected LlamaParse API errors (network failures,
        # authentication errors, quota exceeded, etc.) in a ValueError
        # so the router's single except-ValueError handler catches them.
        raise ValueError(
            f"LlamaParse failed to parse '{filename}': {exc}"
        ) from exc

    finally:
        # Always delete the temp file — even if an exception was raised.
        # missing_ok=True prevents a secondary FileNotFoundError if the
        # file was somehow already removed.
        pathlib.Path(tmp_path).unlink(missing_ok=True)
