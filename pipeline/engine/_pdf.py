"""PDF text extraction and overlapping-chunk splitting."""

from pathlib import Path

from pypdf import PdfReader

from engine._config import Chunk, logger
from engine._text import clean_markdown_artifacts, normalize_whitespace

try:
    import pymupdf4llm

    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False


def extract_text_from_pdf(pdf_path: Path) -> str:
    """Extract text from a PDF, preferring pymupdf4llm when available."""
    if PYMUPDF_AVAILABLE:
        try:
            return clean_markdown_artifacts(str(pymupdf4llm.to_markdown(str(pdf_path))))
        except Exception as exc:
            logger.warning("Fallo pymupdf4llm en '%s': %s", pdf_path.name, exc)

    try:
        reader = PdfReader(str(pdf_path))
        texts = []
        for page_idx, page in enumerate(reader.pages):
            try:
                page_text = page.extract_text() or ""
            except Exception as page_exc:
                logger.warning(
                    "Fallo extract_text en '%s' pagina %s: %s",
                    pdf_path.name,
                    page_idx,
                    page_exc,
                )
                continue
            if page_text.strip():
                texts.append(page_text)
        return clean_markdown_artifacts("\n\n".join(texts))
    except Exception as exc:
        logger.error("No se pudo extraer texto de '%s': %s", pdf_path.name, exc)
        return ""


def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Split text into overlapping character chunks."""
    cleaned = normalize_whitespace(text)
    if not cleaned:
        return []

    chunks: list[str] = []
    step = max(1, chunk_size - chunk_overlap)
    for start in range(0, len(cleaned), step):
        end = min(len(cleaned), start + chunk_size)
        if end < len(cleaned) and cleaned[end - 1].isalnum() and cleaned[end : end + 1].isalnum():
            partial = cleaned[start:end]
            min_boundary = max(step, int(chunk_size * 0.6))
            boundary = max(partial.rfind("\n\n"), partial.rfind("\n"), partial.rfind(" "))
            if boundary >= min_boundary:
                end = start + boundary
        chunk = cleaned[start:end].strip()
        if len(chunk) >= 120:
            chunks.append(chunk)
        if start + chunk_size >= len(cleaned):
            break
    return chunks


def build_chunks_from_text(
    raw_text: str,
    document_name: str,
    document_stem: str,
    chunk_size: int,
    chunk_overlap: int,
    max_chunks: int | None,
) -> list[Chunk]:
    """Chunk already-extracted text into Chunk objects."""
    chunks: list[Chunk] = []
    part_chunks = chunk_text(raw_text, chunk_size, chunk_overlap)
    for index, text in enumerate(part_chunks):
        chunk_id = f"{document_stem}-chunk-{index:04d}"
        chunks.append(Chunk(chunk_id=chunk_id, document=document_name, text=text))
        if max_chunks is not None and len(chunks) >= max_chunks:
            break
    return chunks
