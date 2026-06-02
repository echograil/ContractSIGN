from __future__ import annotations

from dataclasses import asdict, dataclass
from io import BytesIO
import json
import logging
from pathlib import Path
import re
from typing import BinaryIO

from pypdf import PdfReader

logger = logging.getLogger(__name__)

VALID_DOC_TYPES = frozenset({"contract", "quote", "faq", "product", "unknown"})
SOURCE_BYTES_INPUT = "<bytes_input>"
SCAN_PROBE_MIN_CHARS = 20
MIN_LAYOUT_LENGTH_RATIO = 0.75
DEFAULT_INPUT_DIR = "input"
DEFAULT_OUTPUT_DIR = "output"
CHUNKS_FILENAME = "chunks.json"
MANIFEST_FILENAME = "manifest.json"


@dataclass(frozen=True)
class TextChunk:
    text: str
    source_file: str
    location: str
    doc_type: str
    chunk_index: int
    clause_id: str | None = None


@dataclass(frozen=True)
class BatchIngestResult:
    source_file: str
    output_file: str
    chunk_count: int
    status: str


def ingest(file_path: str | bytes, hint_doc_type: str | None = None) -> list[TextChunk]:
    """Parse a text PDF into page-level chunks.

    Business-facing failures return an empty list by contract. Details are
    logged at warning level for operators and tests.
    """
    try:
        pdf_input, source_file = _open_pdf_input(file_path)
        if pdf_input is None:
            return []

        with pdf_input:
            reader = PdfReader(pdf_input)
            if reader.is_encrypted:
                logger.warning("Cannot ingest encrypted PDF: %s", source_file)
                return []

            page_texts = [_normalize_text(_extract_page_text(page)) for page in reader.pages]
    except Exception as exc:
        logger.warning("Cannot ingest PDF: %s", exc)
        return []

    if _looks_like_scanned_pdf(page_texts):
        logger.warning("Cannot ingest scanned or image-only PDF: %s", source_file)
        return []

    doc_type = _detect_doc_type(page_texts, hint_doc_type, source_file)
    chunks: list[TextChunk] = []
    for page_number, text in enumerate(page_texts, start=1):
        if not text:
            continue
        chunks.append(
            TextChunk(
                text=text,
                source_file=source_file,
                location=f"p.{page_number}",
                doc_type=doc_type,
                chunk_index=len(chunks),
                clause_id=None,
            )
        )

    return chunks


def ingest_directory(
    input_dir: str | Path = DEFAULT_INPUT_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    hint_doc_type: str | None = None,
) -> list[BatchIngestResult]:
    """Ingest all PDFs from input_dir and write normalized JSON to output_dir."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if not input_path.exists() or not input_path.is_dir():
        logger.warning("Input directory does not exist: %s", input_path)
        _write_json(output_path / MANIFEST_FILENAME, [])
        return []

    results: list[BatchIngestResult] = []
    all_chunks: list[dict[str, object]] = []

    for pdf_path in sorted(input_path.glob("*.pdf")):
        chunks = ingest(str(pdf_path), hint_doc_type=hint_doc_type)
        chunk_payload = [asdict(chunk) for chunk in chunks]
        output_file = f"{pdf_path.stem}.chunks.json"

        _write_json(output_path / output_file, chunk_payload)
        all_chunks.extend(chunk_payload)
        results.append(
            BatchIngestResult(
                source_file=pdf_path.name,
                output_file=output_file,
                chunk_count=len(chunks),
                status="ok" if chunks else "empty",
            )
        )

    _write_json(output_path / CHUNKS_FILENAME, all_chunks)
    _write_json(output_path / MANIFEST_FILENAME, [asdict(result) for result in results])
    return results


def _open_pdf_input(file_path: str | bytes) -> tuple[BinaryIO | None, str]:
    if isinstance(file_path, bytes):
        if not file_path:
            logger.warning("Cannot ingest empty bytes input")
            return None, SOURCE_BYTES_INPUT
        return BytesIO(file_path), SOURCE_BYTES_INPUT

    path = Path(file_path)
    if not path.exists() or path.stat().st_size == 0:
        logger.warning("Cannot ingest missing or empty file: %s", path)
        return None, path.name

    return path.open("rb"), path.name


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _extract_page_text(page: object) -> str:
    plain_text = page.extract_text() or ""
    try:
        layout_text = page.extract_text(extraction_mode="layout") or ""
    except TypeError:
        return plain_text

    plain_norm = _normalize_text(plain_text)
    layout_norm = _normalize_text(layout_text)
    if len(layout_norm) >= len(plain_norm):
        return layout_text
    if (
        len(layout_norm) >= len(plain_norm) * MIN_LAYOUT_LENGTH_RATIO
        and _glued_token_count(layout_text) < _glued_token_count(plain_text)
    ):
        return layout_text
    return plain_text


def _glued_token_count(text: str) -> int:
    return len(re.findall(r"[a-z][A-Z]|[A-Za-z]\d|\d[A-Za-z]", text))


def _looks_like_scanned_pdf(page_texts: list[str]) -> bool:
    return bool(page_texts) and len(page_texts[0]) < SCAN_PROBE_MIN_CHARS


def _detect_doc_type(
    page_texts: list[str], hint_doc_type: str | None, source_file: str
) -> str:
    if hint_doc_type in VALID_DOC_TYPES:
        return hint_doc_type

    combined_text = " ".join(page_texts).lower()
    name = source_file.lower()

    contract_markers = ("contract", "agreement", "party a", "party b")
    quote_markers = ("quote", "quotation")
    faq_markers = ("faq", "frequently asked")
    product_markers = ("product", "sku", "specification")

    if _contains_any(combined_text, contract_markers) or _contains_any(name, contract_markers):
        return "contract"
    if _contains_any(combined_text, quote_markers) or _contains_any(name, quote_markers):
        return "quote"
    if _contains_any(combined_text, faq_markers) or _contains_any(name, faq_markers):
        return "faq"
    if _contains_any(combined_text, product_markers) or _contains_any(name, product_markers):
        return "product"

    return "unknown"


def _contains_any(value: str, markers: tuple[str, ...]) -> bool:
    return any(marker in value for marker in markers)


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
