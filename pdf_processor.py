
"""
pdf_processor.py
----------------
Handles PDF file validation, text extraction using PyPDF,
and text chunking using LangChain's RecursiveCharacterTextSplitter.

Each chunk retains metadata:
    - source: original filename
    - page: page number (1-indexed)
    - chunk_id: unique chunk identifier
"""

import logging
from typing import List, Tuple
from io import BytesIO

from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
MIN_CHUNK_LENGTH = 50


# ---------------------------------------------------------------------------
# Data structure
# ---------------------------------------------------------------------------

class DocumentChunk:
    """Represents a single text chunk with its associated metadata."""

    def __init__(self, text: str, source: str, page: int, chunk_id: str):
        self.text = text
        self.source = source
        self.page = page
        self.chunk_id = chunk_id

    def __repr__(self) -> str:
        return (
            f"DocumentChunk(source={self.source!r}, page={self.page}, "
            f"chunk_id={self.chunk_id!r}, len={len(self.text)})"
        )


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def is_valid_pdf(file_bytes: bytes) -> bool:
    """
    Check whether the provided bytes represent a PDF file.

    A PDF normally starts with the %PDF- magic header.
    """

    return file_bytes[:5] == b"%PDF-"


# ---------------------------------------------------------------------------
# PDF text extraction
# ---------------------------------------------------------------------------

def extract_text_from_pdf(
    file_bytes: bytes,
    filename: str
) -> Tuple[List[Tuple[int, str]], List[str]]:
    """
    Extract text from each page of the PDF.

    Returns:
        pages: List of (page_number, page_text) tuples.
        warnings: Warnings for pages with no extractable text.

    Raises:
        ValueError: If the PDF contains zero pages.
        RuntimeError: If the PDF cannot be opened or decrypted.
    """

    warnings: List[str] = []
    pages: List[Tuple[int, str]] = []

    try:
        reader = PdfReader(BytesIO(file_bytes))

        # Handle encrypted PDFs.
        if reader.is_encrypted:
            try:
                result = reader.decrypt("")

                if result == 0:
                    raise RuntimeError(
                        "The PDF is password-protected."
                    )

            except Exception as exc:
                raise RuntimeError(
                    f"The PDF is password-protected or could not be "
                    f"decrypted. Details: {exc}"
                ) from exc

    except Exception as exc:
        raise RuntimeError(
            f"Cannot open '{filename}' as a PDF. "
            f"The file may be corrupted or password-protected. "
            f"Details: {exc}"
        ) from exc

    total_pages = len(reader.pages)

    if total_pages == 0:
        raise ValueError(f"'{filename}' contains no pages.")

    for page_num, page in enumerate(reader.pages, start=1):

        try:
            text = page.extract_text() or ""
            text = text.strip()

        except Exception as exc:
            warnings.append(
                f"Page {page_num} of '{filename}' could not be read: {exc}"
            )
            continue

        if not text:
            warnings.append(
                f"Page {page_num} of '{filename}' has no extractable text "
                f"(may be a scanned image). This page will be skipped."
            )

        else:
            pages.append((page_num, text))

    return pages, warnings


# ---------------------------------------------------------------------------
# Text chunking
# ---------------------------------------------------------------------------

def chunk_pages(
    pages: List[Tuple[int, str]],
    filename: str
) -> List[DocumentChunk]:
    """
    Split page texts into overlapping chunks.

    Each chunk retains:
        - source
        - page
        - chunk_id
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks: List[DocumentChunk] = []

    for page_num, page_text in pages:

        raw_chunks = splitter.split_text(page_text)

        for idx, chunk_text in enumerate(raw_chunks):

            chunk_text = chunk_text.strip()

            # Discard very short chunks.
            if len(chunk_text) < MIN_CHUNK_LENGTH:
                continue

            chunk_id = f"{filename}_p{page_num}_c{idx}"

            chunks.append(
                DocumentChunk(
                    text=chunk_text,
                    source=filename,
                    page=page_num,
                    chunk_id=chunk_id,
                )
            )

    return chunks


# ---------------------------------------------------------------------------
# Top-level processor
# ---------------------------------------------------------------------------

def process_uploaded_pdf(
    file_bytes: bytes,
    filename: str
) -> Tuple[List[DocumentChunk], List[str], List[str]]:
    """
    Process one uploaded PDF.

    Steps:
        1. Validate the PDF.
        2. Extract text page by page.
        3. Split text into overlapping chunks.
        4. Attach metadata to each chunk.

    Returns:
        chunks, warnings, errors
    """

    errors: List[str] = []
    warnings: List[str] = []
    chunks: List[DocumentChunk] = []

    # Step 1: Validate.
    if not file_bytes:
        errors.append(f"'{filename}' appears to be empty.")
        return chunks, warnings, errors

    if not is_valid_pdf(file_bytes):
        errors.append(
            f"'{filename}' is not a valid PDF file. "
            "Please upload a genuine PDF file."
        )
        return chunks, warnings, errors

    # Step 2: Extract text.
    try:
        pages, page_warnings = extract_text_from_pdf(
            file_bytes,
            filename
        )

        warnings.extend(page_warnings)

    except (ValueError, RuntimeError) as exc:
        errors.append(str(exc))
        return chunks, warnings, errors

    except Exception as exc:
        errors.append(
            f"Unexpected error while reading '{filename}': {exc}"
        )
        return chunks, warnings, errors

    if not pages:
        errors.append(
            f"'{filename}' was opened successfully, but no text could be "
            "extracted from any page. The document may consist entirely "
            "of scanned images. Please use a PDF with selectable text."
        )
        return chunks, warnings, errors

    # Step 3: Chunk extracted text.
    try:
        chunks = chunk_pages(pages, filename)

    except Exception as exc:
        errors.append(
            f"Error while splitting '{filename}' into chunks: {exc}"
        )
        return chunks, warnings, errors

    if not chunks:
        errors.append(
            f"'{filename}' was processed but produced no usable text chunks. "
            "The document may contain only very short text snippets."
        )

    logger.info(
        "Processed '%s': %d pages with text, %d chunks produced.",
        filename,
        len(pages),
        len(chunks),
    )

    return chunks, warnings, errors


# ---------------------------------------------------------------------------
# Batch processor
# ---------------------------------------------------------------------------

def process_multiple_pdfs(
    files: List[Tuple[bytes, str]]
) -> Tuple[List[DocumentChunk], List[str], List[str]]:
    """
    Process multiple PDFs.

    Args:
        files: List of (file_bytes, filename) pairs.

    Returns:
        all_chunks, all_warnings, all_errors
    """

    all_chunks: List[DocumentChunk] = []
    all_warnings: List[str] = []
    all_errors: List[str] = []

    for file_bytes, filename in files:

        chunks, warnings, errors = process_uploaded_pdf(
            file_bytes,
            filename
        )

        all_chunks.extend(chunks)
        all_warnings.extend(warnings)
        all_errors.extend(errors)

    return all_chunks, all_warnings, all_errors
