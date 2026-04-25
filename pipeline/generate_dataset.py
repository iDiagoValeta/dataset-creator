"""
DatasetCreator -- Topic-aware synthetic dataset generation from PDF documents.

The pipeline is optimized for local execution with Ollama and long-context models:
    1) Extract text from PDFs (text only; no images/tables processing)
    2) Build a global topic map per document using large context windows
    3) Generate Q/A items per topic while avoiding duplicates
    4) Export structured JSONL datasets and train/val/test splits

Usage:
    python pipeline/generate_dataset.py
    python pipeline/generate_dataset.py --model gemma4:e2b --questions-per-topic 8
    python pipeline/generate_dataset.py --source-dir pipeline/input --output pipeline/output/tesis.jsonl

Dependencies:
    - ollama
    - pymupdf4llm (optional, preferred PDF extraction)
    - pypdf (fallback extraction)
"""

# ----------------------------------------------------------------------
# MODULE MAP -- Section index
# ----------------------------------------------------------------------
#
#  CONFIGURATION
#  +-- 1. Imports
#  +-- 2. Optional dependencies
#  +-- 3. Global config and defaults
#
#  BUSINESS LOGIC
#  +-- 4. Utilities
#  +-- 5. PDF extraction and chunking
#  +-- 6. Topic mapping and generation prompts
#  +-- 7. Ollama generation pipeline
#  +-- 8. Export and splitting
#
#  ENTRY
#  +-- 9. CLI main()
#
# ----------------------------------------------------------------------


# ----------------------------------------------------------------------
# SECTION 1: IMPORTS
# ----------------------------------------------------------------------

import argparse
import json
import logging
import os
import platform
import random
import re
import subprocess
import time
import unicodedata
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any

import ollama
from pypdf import PdfReader

# ----------------------------------------------------------------------
# SECTION 2: OPTIONAL DEPENDENCIES
# ----------------------------------------------------------------------

try:
    import pymupdf4llm

    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False


# ----------------------------------------------------------------------
# SECTION 3: GLOBAL CONFIG
# ----------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_SOURCE_DIR = BASE_DIR / "input"
DEFAULT_OUTPUT_PATH = BASE_DIR / "output" / "dataset.jsonl"
DEFAULT_DEBUG_DIR = BASE_DIR / "run_logs"

DEFAULT_MODEL = os.getenv("OLLAMA_DATASET_MODEL", os.getenv("OLLAMA_RAG_MODEL", "gemma4:e2b"))
DEFAULT_LANGUAGE = os.getenv("DATASET_LANGUAGE", "auto")
DEFAULT_CHUNK_SIZE = int(os.getenv("DATASET_CHUNK_SIZE", "3500"))
DEFAULT_CHUNK_OVERLAP = int(os.getenv("DATASET_CHUNK_OVERLAP", "350"))
DEFAULT_NUM_TOPICS = int(os.getenv("DATASET_NUM_TOPICS", "8"))
DEFAULT_QUESTIONS_PER_TOPIC = int(os.getenv("DATASET_QUESTIONS_PER_TOPIC", "6"))
DEFAULT_SPLIT = os.getenv("DATASET_SPLIT", "0.8,0.1,0.1")
DEFAULT_TEMPERATURE = float(os.getenv("DATASET_TEMPERATURE", "0.2"))
DEFAULT_SEED = int(os.getenv("DATASET_SEED", "42"))
DEFAULT_MAX_DOC_CONTEXT_CHARS = int(os.getenv("DATASET_MAX_DOC_CONTEXT_CHARS", "110000"))
DEFAULT_MAX_TOPIC_CONTEXT_CHARS = int(os.getenv("DATASET_MAX_TOPIC_CONTEXT_CHARS", "24000"))
DEFAULT_QUALITY_GATE = os.getenv("DATASET_QUALITY_GATE", "strict")
DEFAULT_OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT_SECS", "300"))
MIN_TOPIC_CONTEXT_CHARS = 9000
VALID_TYPES = {"factual", "conceptual", "inference", "compare", "definition"}
VALID_DIFFICULTIES = {"easy", "medium", "hard"}
VALID_QUALITY_GATES = {"strict", "balanced", "off"}
DEFAULT_OLLAMA_RETRIES = int(os.getenv("OLLAMA_MAX_RETRIES", "3"))
DEFAULT_OLLAMA_RETRY_BACKOFF = float(os.getenv("OLLAMA_RETRY_BACKOFF_SECS", "2.0"))
STOPWORDS: frozenset = frozenset({
    # ES
    "cual", "como", "para", "este", "esta", "estos", "estas", "sobre",
    "desde", "hasta", "entre", "donde", "cuando", "que", "del", "las",
    "los", "una", "uno", "unos", "unas", "con", "sin", "por",
    # EN
    "what", "which", "how", "does", "that", "with", "from", "have", "has",
    "this", "these", "those", "the", "and", "for",
})
LANGUAGE_MARKERS: dict[str, set[str]] = {
    "en": {
        "the", "and", "of", "to", "in", "is", "are", "that", "with", "for",
        "from", "this", "these", "which", "system", "software", "computer",
    },
    "es": {
        "el", "la", "los", "las", "de", "del", "que", "en", "es", "son",
        "para", "con", "por", "como", "sistema", "software", "computadora",
    },
    "ca": {
        "el", "la", "els", "les", "de", "del", "que", "en", "es", "son",
        "per", "amb", "com", "sistema", "programari", "ordinador",
    },
    "fr": {
        "le", "la", "les", "des", "de", "du", "que", "dans", "est", "sont",
        "pour", "avec", "par", "comme", "systeme", "logiciel", "ordinateur",
    },
    "pt": {
        "o", "a", "os", "as", "de", "do", "da", "que", "em", "e", "sao",
        "para", "com", "por", "como", "sistema", "software", "computador",
    },
}
LANGUAGE_NAMES: dict[str, str] = {
    "en": "English",
    "es": "Spanish",
    "ca": "Catalan",
    "fr": "French",
    "pt": "Portuguese",
}
# Back-compat alias (kept for any external importers).
_STOPWORDS_SET = STOPWORDS

logging.basicConfig(
    level=logging.getLevelName(os.getenv("DATASET_LOG_LEVEL", "INFO")),
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("dataset_creator")
logging.getLogger("httpx").setLevel(logging.WARNING)


@dataclass(frozen=True)
class Chunk:
    """Represents a text chunk extracted from a PDF."""

    chunk_id: str
    document: str
    text: str


@dataclass(frozen=True)
class Topic:
    """Represents a document topic inferred by the model."""

    topic_id: str
    name: str
    summary: str
    keywords: list[str]


# ----------------------------------------------------------------------
# SECTION 4: UTILITIES
# ----------------------------------------------------------------------

def parse_split(split_text: str) -> tuple[float, float, float]:
    """Parse and validate split ratios in 'train,val,test' format."""
    parts = [p.strip() for p in split_text.split(",")]
    if len(parts) != 3:
        raise ValueError("Split debe tener 3 valores: train,val,test")

    train_ratio, val_ratio, test_ratio = (float(p) for p in parts)
    if min(train_ratio, val_ratio, test_ratio) < 0:
        raise ValueError("Los ratios de split no pueden ser negativos")
    total = train_ratio + val_ratio + test_ratio
    if total <= 0:
        raise ValueError("La suma del split debe ser > 0")

    return train_ratio / total, val_ratio / total, test_ratio / total


def list_pdf_files(source_dir: Path) -> list[Path]:
    """Return sorted PDF files from source directory."""
    if not source_dir.exists():
        raise FileNotFoundError(f"No existe carpeta de PDFs: {source_dir}")
    files = sorted(p for p in source_dir.iterdir() if p.is_file() and p.suffix.lower() == ".pdf")
    return files


def sanitize_question(text: str) -> str:
    """Normalize question text for deduplication."""
    clean = re.sub(r"\s+", " ", text.strip().lower())
    return re.sub(r"[^\w\s]", "", clean)


def normalize_whitespace(text: str) -> str:
    """Collapse noisy line breaks and spacing."""
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def strip_accents_ascii(text: str) -> str:
    """Lowercase and remove accents for simple language heuristics."""
    decomposed = unicodedata.normalize("NFKD", text.lower())
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def clean_markdown_artifacts(text: str) -> str:
    """Remove markdown/wiki artifacts that hurt topic and QA generation quality."""
    cleaned = text
    cleaned = cleaned.replace("Ã¢â‚¬â€", " - ").replace("Ã¢â‚¬â€œ", " - ")
    cleaned = cleaned.replace("Ã¢â‚¬Ëœ", "'").replace("Ã¢â‚¬â„¢", "'")
    cleaned = cleaned.replace("Ã¢â‚¬Å“", '"').replace("Ã¢â‚¬Â", '"')
    cleaned = re.sub(r"==\s*picture\s+\d+\s+x\s+\d+\s+intentionally omitted\s*<==", " ", cleaned, flags=re.I)
    cleaned = re.sub(r"\[([^\]]+)\]\((https?://[^\)]+)\)", r"\1", cleaned)
    cleaned = re.sub(r"https?://\S+", " ", cleaned)
    cleaned = re.sub(r"\[\[\d+\]\]", " ", cleaned)
    # Strip isolated 1-3 digit numbers (page/section artifacts).
    # Preserve: decimals (0.749), percentages (60%), compound adjectives (10-year, 30-minute).
    cleaned = re.sub(r"(?<![A-Za-z/.\d-])\b\d{1,3}\b(?![A-Za-z/.\d%-])", " ", cleaned)
    cleaned = re.sub(r"[~`*_#>{}\[\]|]+", " ", cleaned)
    # Slash-separated table cells: "Word / Word / Word"
    cleaned = re.sub(r"(?:\b\w+\b\s*/\s*){2,}\w+\b", " ", cleaned)
    # Pure numeric table rows: four or more numbers on a single line
    cleaned = re.sub(r"^\s*[-+]?\d*\.?\d+(?:\s+[-+]?\d*\.?\d+){3,}\s*$", " ", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    return normalize_whitespace(cleaned)


def clean_generated_text(text: str) -> str:
    """Clean extraction artifacts that leaked into generated questions or answers."""
    cleaned = text
    cleaned = cleaned.replace("Ã¢â‚¬â€", " - ").replace("Ã¢â‚¬â€œ", " - ")
    cleaned = cleaned.replace("Ã¢â‚¬Ëœ", "'").replace("Ã¢â‚¬â„¢", "'")
    cleaned = cleaned.replace("Ã¢â‚¬Å“", '"').replace("Ã¢â‚¬Â", '"')
    cleaned = re.sub(r"==\s*picture\s+\d+\s+x\s+\d+\s+intentionally omitted\s*<==", " ", cleaned, flags=re.I)
    cleaned = re.sub(r"(?<![A-Za-z/.\d-])\b\d{2,3}\b(?![A-Za-z/.\d%-])", " ", cleaned)
    cleaned = normalize_whitespace(cleaned)
    if cleaned:
        cleaned = cleaned[0].upper() + cleaned[1:]
    return cleaned


def now_iso() -> str:
    """Return UTC timestamp in ISO format."""
    return datetime.now(timezone.utc).isoformat()


def _sample_existing_questions(existing: Sequence[str], limit: int) -> list[str]:
    """Blend last-N with a deterministic sample of older questions to avoid recency bias."""
    if len(existing) <= limit:
        return list(existing)
    recent_share = max(1, limit // 2)
    recent = list(existing[-recent_share:])
    older_pool = list(existing[:-recent_share])
    older_budget = limit - len(recent)
    if older_budget <= 0 or not older_pool:
        return recent
    rng = random.Random(len(existing))  # deterministic per call-site growth
    older = rng.sample(older_pool, min(older_budget, len(older_pool)))
    return older + recent


def deduplicate_preserve_order(values: Sequence[str]) -> list[str]:
    """Remove duplicates preserving first appearance."""
    seen = set()
    out: list[str] = []
    for value in values:
        clean = value.strip()
        if not clean:
            continue
        low = clean.lower()
        if low in seen:
            continue
        seen.add(low)
        out.append(clean)
    return out


def detect_document_language(text: str) -> tuple[str, str, dict[str, int]]:
    """Infer the dominant document language from extracted text."""
    sample = normalize_whitespace(text[:60000]).lower()
    raw_words = re.findall(r"[a-zÀ-ÿ]{2,}", sample, flags=re.IGNORECASE)
    normalized_words = [strip_accents_ascii(word) for word in raw_words]
    scores = {
        code: sum(1 for word in normalized_words if word in markers)
        for code, markers in LANGUAGE_MARKERS.items()
    }

    if "ñ" in sample or "¿" in sample or "¡" in sample:
        scores["es"] += 8
    if any(token in sample for token in (" l'", " d'", " qu'", "à", "è", "ç")):
        scores["fr"] += 4
        scores["ca"] += 2
    if any(token in sample for token in ("ção", "ões", "ã", "õ")):
        scores["pt"] += 6

    best_code = max(scores, key=scores.get) if scores else "en"
    if scores.get(best_code, 0) <= 0:
        best_code = "en"
    return best_code, LANGUAGE_NAMES.get(best_code, best_code), scores


def resolve_generation_language(language_arg: str, document_text: str) -> tuple[str, str, dict[str, int]]:
    """Use explicit --language unless it is auto, otherwise detect from document text."""
    requested = (language_arg or "auto").strip()
    if requested.lower() != "auto":
        return requested, requested, {}
    code, name, scores = detect_document_language(document_text)
    prompt_language = f"{name} ({code}), the same language as the source document"
    return prompt_language, code, scores


# ----------------------------------------------------------------------
# SECTION 5: PDF EXTRACTION AND CHUNKING
# ----------------------------------------------------------------------

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
        chunk = cleaned[start : start + chunk_size].strip()
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


GENERIC_TOPIC_TERMS = {
    "concept",
    "concepts",
    "theory",
    "references",
    "literature",
    "history and literature",
    "structure and concepts",
}


def extract_section_headings(text: str) -> list[str]:
    """Extract short, useful section headings from PDF text."""
    headings: list[str] = []
    for raw_line in text.splitlines():
        line = normalize_whitespace(raw_line)
        line = line.replace("Histor y", "History")
        if not (3 <= len(line) <= 80):
            continue
        if re.search(r"https?://|ISBN|\[\d+\]|\d{4}|@", line):
            continue
        # Affiliation lines start with a digit directly glued to a letter ("2Department")
        if re.match(r"^\d[A-Za-z]", line):
            continue
        words = line.split()
        if len(words) > 8:
            continue
        letters = re.sub(r"[^A-Za-z ]", "", line)
        if not letters.strip():
            continue
        if line.endswith(".") or line.endswith(","):
            continue
        # Skip institution/affiliation lines
        if re.search(r"\b(?:University|Department|Institute|Laboratory|College|School)\b", line, re.I):
            continue
        lowercase_words = sum(
            1
            for word in words
            if word[:1].islower() and word.lower() not in {"and", "or", "of", "in", "for", "to", "the"}
        )
        if lowercase_words > max(1, len(words) // 2):
            continue
        headings.append(line)
    return deduplicate_preserve_order(headings)


def build_section_topics_from_text(text: str, num_topics: int) -> list[Topic]:
    """Build deterministic topics from document section headings."""
    headings = extract_section_headings(text)
    blocked = {"operating system", "components", "history", "popular operating systems", "references"}
    preferred = [
        heading for heading in headings
        if heading.lower() not in blocked and not heading.lower().startswith("list of ")
    ]
    if len(preferred) <= num_topics:
        selected = preferred
    else:
        # Spread uniformly across the full heading list so results/conclusions are included.
        indices = [int(i * len(preferred) / num_topics) for i in range(num_topics)]
        selected = [preferred[idx] for idx in indices]
    topics: list[Topic] = []
    for idx, heading in enumerate(selected):
        topics.append(
            Topic(
                topic_id=f"topic-{idx:02d}",
                name=heading,
                summary=f"Questions grounded in the '{heading}' section of the document.",
                keywords=extract_keywords(heading),
            )
        )
    return topics


def topics_are_too_generic(topics: Sequence[Topic]) -> bool:
    """Return True when the model topic map is mostly broad labels, not document sections."""
    if not topics:
        return True
    generic = 0
    for topic in topics:
        name = topic.name.lower()
        if any(term in name for term in GENERIC_TOPIC_TERMS):
            generic += 1
    return generic >= max(2, len(topics) // 2)


def topics_mostly_invalid(topics: Sequence[Topic]) -> bool:
    """Return True when most topics fail the is_valid_topic_name check."""
    if not topics:
        return True
    invalid = sum(1 for t in topics if not is_valid_topic_name(t.name))
    return invalid >= max(2, len(topics) // 2)


def is_valid_topic_name(name: str) -> bool:
    """Return True if name looks like a real topic, not a text fragment or table artifact."""
    if not name or len(name) < 5:
        return False
    # Accept section-style headings: "1 Introduction", "2.1 Background", "A. Appendix"
    if re.match(r"^[\dA-Z][\d.]*\s+[A-Za-z]", name):
        return len(re.findall(r"[A-Za-zÀ-ÿ]{3,}", name)) >= 1
    if not name[0].isalpha():
        return False
    real_words = re.findall(r"[A-Za-zÀ-ÿ]{3,}", name)
    if not real_words:
        return False
    if len(real_words) < 2:
        # Single-word topic: only valid when the word itself is substantial (≥6 chars).
        # This accepts "Abstract", "Introduction", "Methodology" but rejects "Topic", "Rf".
        if not re.search(r"[A-Za-zÀ-ÿ]{6,}", name):
            return False
    if name.count("/") > 2:
        return False
    return True


# ----------------------------------------------------------------------
# SECTION 6: TOPIC MAPPING AND PROMPTS
# ----------------------------------------------------------------------

def truncate_text(text: str, max_chars: int) -> str:
    """Trim text to max chars preserving sentence boundaries when possible."""
    if len(text) <= max_chars:
        return text
    partial = text[:max_chars]
    last_break = max(partial.rfind(". "), partial.rfind("\n"))
    if last_break > int(max_chars * 0.6):
        return partial[: last_break + 1]
    return partial


def build_topic_map_messages(document: str, full_text: str, language: str, num_topics: int) -> list[dict[str, str]]:
    """Create prompt messages to infer a non-overlapping topic map."""
    system_prompt = (
        "You are an expert dataset designer for educational and technical corpora. "
        "Return strict JSON only. No markdown. No extra keys."
    )

    user_prompt = f"""
Analyze this full document and extract a topic map.
Language for topic names and summaries: {language}.
Use that language for every generated field, even if the prompt instructions are in English.
Generate up to {num_topics} high-level topics with minimal overlap.

Output format (strict JSON object):
{{
  "topics": [
    {{
      "name": "string",
      "summary": "string",
      "keywords": ["string", "string"]
    }}
  ]
}}

Rules:
- Topics must represent different parts of the document.
- Avoid duplicated or near-duplicated topics.
- Keep summaries short (1-2 sentences).
- Keywords must be specific, not generic.

Document: {document}
Text:
\"\"\"
{full_text}
\"\"\"
""".strip()
    return [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]


def build_representative_document_context(chunks: Sequence[Chunk], max_chars: int) -> str:
    """Build a compact document sample spread across the whole source."""
    if not chunks or max_chars <= 0:
        return ""

    if len(chunks) <= 3:
        selected_indices = list(range(len(chunks)))
    else:
        mid = len(chunks) // 2
        selected_indices = deduplicate_preserve_order(
            [str(i) for i in [0, 1, mid, max(0, len(chunks) - 2), len(chunks) - 1]]
        )
        selected_indices = [int(i) for i in selected_indices]

    blocks: list[str] = []
    total = 0
    per_chunk_budget = max(1200, max_chars // max(1, len(selected_indices)))
    for chunk_idx in selected_indices:
        chunk = chunks[chunk_idx]
        excerpt = truncate_text(chunk.text, per_chunk_budget)
        block = f"[{chunk.chunk_id}]\n{excerpt}"
        if total + len(block) > max_chars:
            remaining = max_chars - total
            if remaining < 500:
                break
            block = truncate_text(block, remaining)
        blocks.append(block)
        total += len(block)
        if total >= max_chars:
            break
    return "\n\n".join(blocks)


def build_topic_map_messages_compact(
    document: str,
    sampled_text: str,
    language: str,
    num_topics: int,
) -> list[dict[str, str]]:
    """Create a shorter topic-map prompt for models with smaller context windows."""
    system_prompt = "Return only one valid JSON object. No markdown. No prose."
    user_prompt = f"""
Create up to {num_topics} non-overlapping document topics in {language}.
Use that language for every generated field, even if the prompt instructions are in English.

Text excerpts:
\"\"\"
{sampled_text}
\"\"\"

Return exactly this JSON shape:
{{
  "topics": [
    {{"name": "string", "summary": "string", "keywords": ["string"]}}
  ]
}}

Rules:
- Use only the excerpts.
- Make topic names readable, not copied mid-sentence fragments.
- No markdown or comments outside JSON.
""".strip()
    return [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]


def build_topic_generation_messages(
    document: str,
    topic: Topic,
    topic_context: str,
    language: str,
    questions_per_topic: int,
    existing_questions: Sequence[str],
) -> list[dict[str, str]]:
    """Create prompt messages to generate topic-specific non-repeated Q/A items."""
    system_prompt = (
        "You generate high-quality supervised Q/A datasets from source text. "
        "Return strict JSON only. No markdown. No extra keys."
    )

    max_existing_in_prompt = max(80, questions_per_topic * 10)
    sampled_existing = _sample_existing_questions(existing_questions, max_existing_in_prompt)
    existing_block = "\n".join(f"- {q}" for q in sampled_existing)
    user_prompt = f"""
Generate exactly {questions_per_topic} Q/A items for the topic below.
Language for question and answer: {language}.
Use that language for every question and answer, even if the prompt instructions are in English.

Output format (strict JSON object):
{{
  "items": [
    {{
      "question": "string",
      "answer": "string",
      "type": "factual|conceptual|inference|compare|definition",
      "difficulty": "easy|medium|hard",
      "context_source": "verbatim fragment from topic context that directly supports the answer (max 200 chars)"
    }}
  ]
}}

Rules:
- Questions must be answerable only from topic context.
- Questions must cover different sub-points inside the same topic.
- Never repeat semantics from Existing dataset questions.
- Answers must be argumentable: cite or paraphrase the specific passage that supports the answer.
- Answers should be 2-4 sentences: state the claim, then explain the evidence from the text.
- If context lacks enough information for a question, skip that question - do not hallucinate.
- Avoid asking about figures, images, or tables.
- Do not hallucinate details not present in context.
- Do not include citations, markdown, or XML tags.
- context_source must be a literal substring from topic context, not a paraphrase.

Question types:
- factual: specific fact stated in the text
- conceptual: explain a concept or mechanism
- inference: conclusion deducible from text though not explicit
- compare: contrast two elements from the same passage
- definition: meaning of a term according to the text
- Distribute: include at least one factual, one conceptual and one inference item.

Document: {document}
Topic: {topic.name}
Topic summary: {topic.summary}
Topic keywords: {", ".join(topic.keywords)}

Existing dataset questions (do not repeat):
{existing_block if existing_block else "- (none yet)"}

Topic context:
\"\"\"
{topic_context}
\"\"\"
""".strip()
    return [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]


def build_topic_generation_messages_compact(
    document: str,
    topic: Topic,
    topic_context: str,
    language: str,
    questions_per_topic: int,
) -> list[dict[str, str]]:
    """Compact fallback prompt to maximize schema adherence."""
    system_prompt = "Return strict JSON object only."
    user_prompt = f"""
Create {questions_per_topic} question-answer pairs in {language} from the topic context.
Use that language for every question and answer, even if the prompt instructions are in English.

Output:
{{
  "items": [
    {{
      "question": "string",
      "answer": "string",
      "type": "factual|conceptual|inference|compare|definition",
      "difficulty": "easy|medium|hard",
      "context_source": "literal supporting fragment from context"
    }}
  ]
}}

Constraints:
- Use only the context.
- No markdown.
- No prose outside JSON.
- Questions must be distinct.
- Prefer balanced types with at least one factual, conceptual and inference.

Document: {document}
Topic: {topic.name}
Context:
{truncate_text(topic_context, 7000)}
""".strip()
    return [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]


def try_parse_json_payload(payload: str) -> dict[str, Any]:
    """Parse model output into dict with an 'items' list."""
    stripped = (payload or "").strip()
    try:
        obj = json.loads(stripped)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    brace_start = stripped.find("{")
    if brace_start != -1:
        depth = 0
        for idx, ch in enumerate(stripped[brace_start:], start=brace_start):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = stripped[brace_start : idx + 1]
                    try:
                        obj = json.loads(candidate)
                        if isinstance(obj, dict):
                            return obj
                    except json.JSONDecodeError:
                        break

    # Third attempt: extract completed topic objects from a truncated JSON array.
    partial_topics: list[dict[str, Any]] = []
    for m in re.finditer(r'\{[^{}]*?"name"\s*:\s*"([^"]+)"[^{}]*?\}', stripped, re.S):
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, dict) and obj.get("name"):
                partial_topics.append(obj)
        except json.JSONDecodeError:
            pass
    if partial_topics:
        logger.debug("JSON truncado; se rescataron %d topicos parciales.", len(partial_topics))
        return {"topics": partial_topics}

    logger.debug("No se pudo parsear JSON del modelo; se retorna payload vacio.")
    return {"items": []}


# ----------------------------------------------------------------------
# SECTION 7: OLLAMA GENERATION PIPELINE
# ----------------------------------------------------------------------

def _ollama_list_models_entries(listing: Any) -> list[Any]:
    """Normalize Client.list(): dict JSON vs ollama>=0.4 ListResponse with .models."""
    if isinstance(listing, dict):
        raw = listing.get("models", [])
        return list(raw) if isinstance(raw, (list, tuple)) else []
    models = getattr(listing, "models", None)
    if isinstance(models, (list, tuple)):
        return list(models)
    return []


def _ollama_entry_model_name(entry: Any) -> str | None:
    if isinstance(entry, dict):
        name = entry.get("model") or entry.get("name")
    else:
        name = getattr(entry, "model", None) or getattr(entry, "name", None)
    if name is None:
        return None
    return str(name)


def verify_ollama_model(model: str) -> None:
    """Abort early if Ollama is unreachable or the requested model is missing."""
    try:
        client = ollama.Client(timeout=DEFAULT_OLLAMA_TIMEOUT)
        listing = client.list()
    except Exception as exc:
        raise RuntimeError(
            f"No se pudo conectar con Ollama. ¿Está el servicio corriendo? Detalle: {exc}"
        ) from exc

    models_payload = _ollama_list_models_entries(listing)
    available: set[str] = set()
    for entry in models_payload:
        name = _ollama_entry_model_name(entry)
        if name:
            available.add(name)
            available.add(name.split(":", 1)[0])

    if model in available or model.split(":", 1)[0] in available:
        return

    hint = ", ".join(sorted(available)) or "(ninguno)"
    raise RuntimeError(
        f"Modelo '{model}' no disponible en Ollama. Modelos detectados: {hint}. "
        f"Instálalo con: ollama pull {model}"
    )


def call_ollama_json(
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    seed: int | None = None,
    max_retries: int = DEFAULT_OLLAMA_RETRIES,
    backoff_secs: float = DEFAULT_OLLAMA_RETRY_BACKOFF,
) -> tuple[dict[str, Any], str]:
    """Execute Ollama chat call expecting a JSON object, with retry on transient errors."""
    options: dict[str, Any] = {"temperature": temperature}
    if seed is not None:
        options["seed"] = seed

    last_error: BaseException | None = None
    for attempt in range(1, max_retries + 1):
        call_start = time.time()
        try:
            client = ollama.Client(timeout=DEFAULT_OLLAMA_TIMEOUT)
            response = client.chat(
                model=model,
                messages=messages,
                format="json",
                options=options,
                think=False,
            )
            elapsed = time.time() - call_start
            logger.debug("Ollama '%s' OK en %.2fs (intento %s)", model, elapsed, attempt)
            content = response.get("message", {}).get("content", "")
            return try_parse_json_payload(content), content
        except ollama.ResponseError as exc:
            # 4xx-like errors are usually not transient (bad prompt, missing model).
            logger.error("Ollama ResponseError en modelo '%s' (intento %s/%s): %s",
                         model, attempt, max_retries, exc)
            return {"items": []}, ""
        except Exception as exc:
            elapsed = time.time() - call_start
            last_error = exc
            logger.warning(
                "Error transitorio llamando Ollama '%s' (intento %s/%s tras %.2fs): %s",
                model, attempt, max_retries, elapsed, exc,
            )
            if attempt < max_retries:
                time.sleep(backoff_secs * attempt)

    logger.error("Ollama falló tras %s intentos: %s", max_retries, last_error)
    return {"items": []}, ""


def _extract_topics_candidates(payload: dict[str, Any]) -> list[Any]:
    """Find topic candidates under common keys used by different models."""
    if not isinstance(payload, dict):
        return []

    keys = [
        "topics",
        "temas",
        "topicos",
        "items",
        "sections",
        "data",
    ]
    for key in keys:
        value = payload.get(key)
        if isinstance(value, list):
            return value

    # Fallback: first list value in dict.
    for value in payload.values():
        if isinstance(value, list):
            return value

    # Some models return {"Topic name": {"summary": ..., "keywords": [...]}}.
    object_topics: list[dict[str, Any]] = []
    for key, value in payload.items():
        if key in {"items", "questions", "data", "results"}:
            continue
        if isinstance(value, dict):
            topic_obj = dict(value)
            topic_obj.setdefault("name", key)
            object_topics.append(topic_obj)
        elif isinstance(value, str) and len(value.strip()) >= 8:
            object_topics.append({"name": key, "summary": value.strip(), "keywords": []})
    if object_topics:
        return object_topics
    return []


def parse_topics(payload: dict[str, Any]) -> list[Topic]:
    """Normalize topic payload into typed topics."""
    topics: list[Topic] = []
    raw_topics = _extract_topics_candidates(payload)
    if not raw_topics:
        return []

    seen_names = set()
    for idx, raw in enumerate(raw_topics):
        name = ""
        summary = ""
        keywords_raw: Any = []

        if isinstance(raw, str):
            name = raw.strip()
            summary = raw.strip()
        elif isinstance(raw, dict):
            name = str(
                raw.get("name")
                or raw.get("title")
                or raw.get("topic")
                or raw.get("tema")
                or ""
            ).strip()
            summary = str(
                raw.get("summary")
                or raw.get("description")
                or raw.get("resumen")
                or raw.get("Definition")
                or raw.get("definition")
                or ""
            ).strip()
            keywords_raw = (
                raw.get("keywords")
                or raw.get("tags")
                or raw.get("palabras_clave")
                or raw.get("Key Topics Covered")
                or raw.get("key_topics")
                or []
            )
        else:
            continue

        if not name:
            continue

        name_key = name.lower()
        if name_key in seen_names:
            continue
        seen_names.add(name_key)

        if isinstance(keywords_raw, list):
            keywords = [str(k).strip() for k in keywords_raw if str(k).strip()]
        else:
            keywords = [k.strip() for k in str(keywords_raw).split(",") if k.strip()]
        keywords = deduplicate_preserve_order(keywords)[:12]
        if not keywords:
            keywords = extract_keywords(f"{name} {summary}", max_keywords=8)

        topics.append(
            Topic(
                topic_id=f"topic-{idx:02d}",
                name=name,
                summary=summary or name,
                keywords=keywords,
            )
        )
    return topics


def extract_keywords(text: str, max_keywords: int = 8) -> list[str]:
    """Extract naive keywords from a text snippet without external deps."""
    words = re.findall(r"[A-Za-zÀ-ÿ][A-Za-zÀ-ÿ0-9_-]{3,}", text.lower())
    counts: dict[str, int] = {}
    for w in words:
        if w in STOPWORDS:
            continue
        counts[w] = counts.get(w, 0) + 1
    ordered = sorted(counts.items(), key=lambda item: item[1], reverse=True)
    return [word for word, _ in ordered[:max_keywords]]


def infer_topic_name_from_chunk(chunk_text: str, fallback_index: int) -> str:
    """Build a readable topic name from first meaningful line."""
    cleaned = normalize_whitespace(chunk_text)
    keyword_title = " / ".join(extract_keywords(cleaned, max_keywords=4)).title()
    for line in chunk_text.splitlines():
        line = line.strip().strip("-*#")
        if len(line) < 8:
            continue
        # Prefer short heading-like lines that look like real topics.
        if len(line) <= 90 and is_valid_topic_name(line):
            return line
        break

    sentence_candidates = re.split(r"(?<=[.!?])\s+", cleaned)
    for sentence in sentence_candidates:
        sentence = sentence.strip()
        words = re.findall(r"[\w-]+", sentence, flags=re.UNICODE)
        if len(words) < 4:
            continue
        if sentence[:1].islower() and keyword_title:
            return keyword_title
        if len(sentence) > 90:
            sentence = " ".join(words[:10])
        return truncate_text(sentence, 90).rstrip(" .,:;")

    words = re.findall(r"[\w-]+", cleaned, flags=re.UNICODE)[:8]
    if words:
        return " ".join(words)
    return f"Topico {fallback_index + 1}"


def build_fallback_topics_from_chunks(chunks: Sequence[Chunk], num_topics: int) -> list[Topic]:
    """Create deterministic topics from chunk slices when LLM topic map fails."""
    if not chunks:
        return []
    count = max(1, min(num_topics, len(chunks)))
    step = max(1, len(chunks) // count)
    selected_indices = list(range(0, len(chunks), step))[:count]
    topics: list[Topic] = []
    for i, chunk_idx in enumerate(selected_indices):
        chunk = chunks[chunk_idx]
        name = infer_topic_name_from_chunk(chunk.text, i)
        summary = truncate_text(chunk.text, 220).replace("\n", " ")
        keywords = extract_keywords(chunk.text, max_keywords=8)
        topics.append(
            Topic(
                topic_id=f"topic-{i:02d}",
                name=name,
                summary=summary,
                keywords=keywords,
            )
        )
    return topics


def score_chunk_for_topic(chunk: Chunk, topic: Topic) -> int:
    """Simple lexical scoring for chunk-topic alignment."""
    haystack = f"{chunk.text.lower()} {chunk.document.lower()}"
    score = 0
    for token in [topic.name, topic.summary, *topic.keywords]:
        token_low = token.lower().strip()
        if not token_low:
            continue
        if token_low in haystack:
            score += 4
        for word in token_low.split():
            if len(word) > 3 and word in haystack:
                score += 1
    return score


def build_topic_context(
    chunks: Sequence[Chunk],
    topic: Topic,
    max_topic_context_chars: int,
) -> str:
    """Select and concatenate best chunks for a given topic."""
    topic_name = topic.name.strip().lower()
    if topic_name:
        for index, chunk in enumerate(chunks):
            if topic_name in chunk.text.lower():
                selected_blocks: list[str] = []
                total = 0
                for section_chunk in chunks[index : min(len(chunks), index + 2)]:
                    block = f"[{section_chunk.chunk_id}] {section_chunk.text}"
                    if total + len(block) > max_topic_context_chars:
                        remaining = max_topic_context_chars - total
                        if remaining >= 500:
                            selected_blocks.append(truncate_text(block, remaining))
                        break
                    selected_blocks.append(block)
                    total += len(block)
                if selected_blocks:
                    return "\n\n".join(selected_blocks)

    scored = [(score_chunk_for_topic(chunk, topic), chunk) for chunk in chunks]
    scored.sort(key=lambda item: item[0], reverse=True)

    selected: list[str] = []
    total = 0
    fallback_mode = all(score <= 0 for score, _ in scored)

    for score, chunk in scored:
        if not fallback_mode and score <= 0:
            continue
        block = f"[{chunk.chunk_id}] {chunk.text}"
        if total + len(block) > max_topic_context_chars:
            break
        selected.append(block)
        total += len(block)
        if total >= max_topic_context_chars:
            break

    return "\n\n".join(selected)


def _content_words(text: str) -> list[str]:
    """Return normalized content words for lightweight support checks."""
    return [
        word
        for word in re.findall(r"[A-Za-zÀ-ÿ]{4,}", strip_accents_ascii(text))
        if word not in STOPWORDS
    ]


def find_verified_context_source(raw_source: str, answer: str, topic_context: str) -> tuple[str, bool]:
    """Find a literal context fragment that supports an answer."""
    normalized_context = normalize_whitespace(topic_context)
    source = normalize_whitespace(raw_source)[:300]
    if source and source.lower() in normalized_context.lower():
        start = normalized_context.lower().find(source.lower())
        return normalized_context[start : start + len(source)], True

    answer_words = _content_words(answer)
    if len(answer_words) < 3:
        return "", False
    answer_terms = set(answer_words)
    best_sentence = ""
    best_score = 0.0
    for sentence in re.split(r"(?<=[.!?])\s+", normalized_context):
        clean_sentence = sentence.strip()
        if not (40 <= len(clean_sentence) <= 450):
            continue
        sentence_terms = set(_content_words(clean_sentence))
        if not sentence_terms:
            continue
        score = len(answer_terms & sentence_terms) / max(1, len(answer_terms))
        if score > best_score:
            best_sentence = clean_sentence
            best_score = score

    if best_sentence and best_score >= 0.6:
        return best_sentence[:300], True
    return "", False


def source_chunk_ids_for_fragment(topic_context: str, fragment: str) -> list[str]:
    """Infer chunk ids that contain or precede a source fragment."""
    chunk_ids = re.findall(r"\[([^\]]+-chunk-\d{4})\]", topic_context)
    if not fragment:
        return chunk_ids[:1]
    fragment_pos = topic_context.lower().find(fragment.lower()[:80])
    if fragment_pos < 0:
        return chunk_ids[:1]
    prefix = topic_context[:fragment_pos]
    preceding = re.findall(r"\[([^\]]+-chunk-\d{4})\]", prefix)
    return preceding[-1:] if preceding else chunk_ids[:1]


def has_quality_artifact(item: dict[str, Any]) -> bool:
    """Detect extraction artifacts that should not reach strict output."""
    text = " ".join(
        str(item.get(key, ""))
        for key in ("question", "answer", "context_source", "context_excerpt", "topic")
    ).lower()
    if "intentionally omitted" in text or "== picture" in text:
        return True
    if re.search(r"\b\d{2,3}\s+(computer|medical|process|memory|device|system)\b", text):
        return True
    if str(item.get("topic", "")).strip().lower().endswith("references"):
        return True
    return False


def is_circular_answer(question: str, answer: str) -> bool:
    """True when the answer adds fewer than 3 new content tokens (words or numbers) vs. the question."""
    q_words = set(_content_words(question))
    a_words = _content_words(answer)
    if not a_words:
        return True
    new_words = [w for w in a_words if w not in q_words]
    # Numbers/percentages count as new content too (e.g. "14%–15%" in the answer)
    q_nums = set(re.findall(r"\d+(?:[.,]\d+)?%?", question))
    a_nums = set(re.findall(r"\d+(?:[.,]\d+)?%?", answer))
    new_nums = a_nums - q_nums
    return len(new_words) + len(new_nums) < 3


def generate_items_for_topic(
    model: str,
    document: str,
    topic: Topic,
    topic_context: str,
    language: str,
    questions_per_topic: int,
    temperature: float,
    existing_questions: Sequence[str],
    seed: int | None = None,
    document_language: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Generate normalized Q/A items for one topic."""
    debug_attempts: list[dict[str, Any]] = []
    item_document_language = document_language or language

    # Strip leading sentence fragment caused by chunk boundary cuts.
    # Only strips when the chunk text starts with ≤3 lowercase chars then a space
    # (e.g. "y environments..." or "nd of..."), avoiding longer mid-word cuts like "icient...".
    topic_context = re.sub(
        r"^(\[[\w.-]+-chunk-\d{4,}\]\s+)[a-z]{1,3}\s[^\n.!?]*[.!?]\s*",
        r"\1",
        topic_context,
    ).strip()

    def _extract_raw_items(parsed_obj: dict[str, Any]) -> list[Any]:
        raw_local = parsed_obj.get("items", [])
        if not isinstance(raw_local, list):
            for alt_key in ("questions", "preguntas", "data", "results"):
                alt_value = parsed_obj.get(alt_key)
                if isinstance(alt_value, list):
                    raw_local = alt_value
                    break
        return raw_local if isinstance(raw_local, list) else []

    def _normalize_items(raw_items_local: list[Any]) -> list[dict[str, Any]]:
        normalized_local: list[dict[str, Any]] = []
        for idx, item in enumerate(raw_items_local[:questions_per_topic]):
            if not isinstance(item, dict):
                continue
            question = clean_generated_text(str(item.get("question", "")).strip())
            answer = clean_generated_text(str(item.get("answer", "")).strip())
            if not question or not answer:
                continue
            raw_type = str(item.get("type", "factual")).strip().lower()
            item_type = raw_type if raw_type in VALID_TYPES else "factual"
            raw_difficulty = str(item.get("difficulty", "medium")).strip().lower()
            difficulty = raw_difficulty if raw_difficulty in VALID_DIFFICULTIES else "medium"

            raw_source = clean_generated_text(str(item.get("context_source", "")).strip())[:300]
            # Strip chunk-id markup the model may have copied from the prompt (e.g. "[doc-chunk-0001] text")
            raw_source = re.sub(r"^\[[\w.-]+-chunk-\d{4,}\]\s*", "", raw_source).strip()
            context_source, context_verified = find_verified_context_source(raw_source, answer, topic_context)
            if not context_source:
                context_source = normalize_whitespace(topic_context[:300])
            # Never let internal chunk-id markup reach the output field
            context_source = re.sub(r"^\[[\w.-]+-chunk-\d{4,}\]\s*", "", context_source).strip()

            normalized_local.append(
                {
                    "id": f"{document}-{topic.topic_id}-qa-{idx:02d}",
                    "question": question,
                    "answer": answer,
                    "type": item_type,
                    "difficulty": difficulty,
                    "context_source": context_source,
                    "context_source_verified": context_verified,
                    "topic": topic.name,
                    "topic_id": topic.topic_id,
                    "topic_keywords": topic.keywords,
                    "document": document,
                    "document_language": item_document_language,
                    "source_chunk_ids": source_chunk_ids_for_fragment(topic_context, context_source),
                    "created_at": now_iso(),
                    "context_excerpt": normalize_whitespace(topic_context[:500]),
                }
            )
        return normalized_local

    # Attempt 1: full prompt.
    messages = build_topic_generation_messages(
        document=document,
        topic=topic,
        topic_context=topic_context,
        language=language,
        questions_per_topic=questions_per_topic,
        existing_questions=existing_questions,
    )
    parsed, raw_content = call_ollama_json(model=model, messages=messages, temperature=temperature, seed=seed)
    raw_items = _extract_raw_items(parsed)
    debug_attempts.append({"attempt": 1, "raw_content": raw_content, "parsed": parsed, "raw_items_count": len(raw_items)})

    # Attempt 2: compact prompt with shorter context.
    if not raw_items:
        compact_messages = build_topic_generation_messages_compact(
            document=document,
            topic=topic,
            topic_context=topic_context,
            language=language,
            questions_per_topic=questions_per_topic,
        )
        parsed2, raw_content2 = call_ollama_json(model=model, messages=compact_messages, temperature=temperature, seed=seed)
        raw_items = _extract_raw_items(parsed2)
        debug_attempts.append(
            {"attempt": 2, "raw_content": raw_content2, "parsed": parsed2, "raw_items_count": len(raw_items)}
        )

    # Attempt 3: salvage from prose summary -> JSON items.
    if not raw_items and raw_content.strip():
        salvage_messages = [
            {
                "role": "system",
                "content": "Convert content into strict JSON object with key 'items'. No prose outside JSON.",
            },
            {
                "role": "user",
                "content": (
                    f"From this text, create {questions_per_topic} Q/A items in {language}.\n"
                    "Use keys: question, answer, type, difficulty.\n"
                    "Text:\n"
                    f"{truncate_text(raw_content, 5000)}"
                ),
            },
        ]
        parsed3, raw_content3 = call_ollama_json(model=model, messages=salvage_messages, temperature=temperature, seed=seed)
        raw_items = _extract_raw_items(parsed3)
        debug_attempts.append(
            {"attempt": 3, "raw_content": raw_content3, "parsed": parsed3, "raw_items_count": len(raw_items)}
        )

    normalized = _normalize_items(raw_items)
    if not normalized:
        last_raw = debug_attempts[-1]["raw_content"] if debug_attempts else ""
        logger.warning(
            "Items no parseables para '%s' / '%s'. Raw (300 chars): %s",
            document,
            topic.name,
            last_raw[:300] if last_raw else "(vacio)",
        )
    return normalized, {"raw_items": raw_items, "attempts": debug_attempts, "topic": topic.name}


def _question_bigrams(text: str) -> frozenset:
    """Build word-bigrams signature for semantic near-duplicate detection."""
    words = [
        w
        for w in re.findall(r"[A-Za-zÀ-ÿ]{4,}", text.lower())
        if w not in STOPWORDS
    ]
    if len(words) >= 2:
        return frozenset(zip(words, words[1:], strict=False))
    return frozenset(words)


def deduplicate_items(items: Iterable[dict[str, Any]], semantic_threshold: float = 0.85) -> list[dict[str, Any]]:
    """Drop exact and near-semantic duplicated questions."""
    seen_exact = set()
    seen_answers = set()
    seen_bigrams: list[frozenset] = []
    unique: list[dict[str, Any]] = []
    for item in items:
        question = str(item.get("question", ""))
        signature = sanitize_question(question)
        answer_signature = sanitize_question(str(item.get("answer", "")))
        if (
            not signature
            or signature in seen_exact
            or (len(answer_signature) >= 12 and answer_signature in seen_answers)
        ):
            continue

        q_bigrams = _question_bigrams(question)
        duplicated_semantic = False
        if q_bigrams:
            for existing in seen_bigrams:
                if not existing:
                    continue
                overlap = len(q_bigrams & existing) / min(len(q_bigrams), len(existing))
                if overlap >= semantic_threshold:
                    duplicated_semantic = True
                    break
        if duplicated_semantic:
            continue

        seen_exact.add(signature)
        if answer_signature:
            seen_answers.add(answer_signature)
        seen_bigrams.append(q_bigrams)
        unique.append(item)
    return unique


def apply_quality_gate(
    items: Sequence[dict[str, Any]],
    gate: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Filter generated items and return accepted rows, rejected rows, and quality stats."""
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    reason_counts: dict[str, int] = {}

    def reject(item: dict[str, Any], reason: str) -> None:
        row = dict(item)
        row["rejection_reason"] = reason
        rejected.append(row)
        reason_counts[reason] = reason_counts.get(reason, 0) + 1

    for item in items:
        if gate == "off":
            accepted.append(item)
            continue
        if has_quality_artifact(item):
            reject(item, "artifact_or_reference_noise")
            continue
        if gate in ("strict", "balanced") and is_circular_answer(
            str(item.get("question", "")), str(item.get("answer", ""))
        ):
            reject(item, "circular_answer")
            continue
        if gate == "strict" and not item.get("context_source_verified"):
            reject(item, "unverified_context_source")
            continue
        accepted.append(item)

    verified = sum(1 for item in accepted if item.get("context_source_verified"))
    return accepted, rejected, {
        "gate": gate,
        "accepted_items": len(accepted),
        "rejected_items": len(rejected),
        "verified_items": verified,
        "verified_ratio": round(verified / len(accepted), 4) if accepted else 0.0,
        "rejection_reasons": reason_counts,
    }


# ----------------------------------------------------------------------
# SECTION 8: EXPORT AND SPLITTING
# ----------------------------------------------------------------------

def write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    """Write rows as JSON Lines."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def split_rows(
    rows: Sequence[dict[str, Any]],
    split: tuple[float, float, float],
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Split rows into train/val/test using deterministic shuffle."""
    train_ratio, val_ratio, test_ratio = split
    shuffled = list(rows)
    random.Random(seed).shuffle(shuffled)

    total = len(shuffled)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    train_rows = shuffled[:train_end]
    val_rows = shuffled[train_end:val_end]
    test_rows = shuffled[val_end:]
    return train_rows, val_rows, test_rows


def write_metadata(path: Path, metadata: dict[str, Any]) -> None:
    """Write metadata as pretty JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


def _assert_safe_cleanup_root(path: Path) -> Path:
    """Resolve and validate cleanup roots before deleting generated artifacts."""
    resolved = path.resolve()
    allowed_roots = [
        (BASE_DIR / "output").resolve(),
        (BASE_DIR / "run_logs").resolve(),
    ]
    if not any(resolved == root or root in resolved.parents for root in allowed_roots):
        allowed = ", ".join(str(root) for root in allowed_roots)
        raise RuntimeError(f"Ruta de limpieza no permitida: {resolved}. Permitidas: {allowed}")
    return resolved


def _remove_empty_dirs(root: Path) -> int:
    """Remove empty directories below root, deepest first."""
    removed = 0
    if not root.exists():
        return removed
    dirs = [p for p in root.rglob("*") if p.is_dir()]
    for directory in sorted(dirs, key=lambda p: len(p.parts), reverse=True):
        try:
            directory.rmdir()
            removed += 1
        except OSError:
            continue
    return removed


def clean_generated_artifacts(output_path: Path, debug_dir: Path, dry_run: bool = False) -> dict[str, int]:
    """Clean generated JSON/JSONL artifacts from output and run-log directories."""
    roots = deduplicate_preserve_order([str(output_path.parent), str(debug_dir)])
    files: list[Path] = []
    for root_text in roots:
        root = _assert_safe_cleanup_root(Path(root_text))
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file() or path.name == ".gitkeep":
                continue
            if path.suffix.lower() in {".json", ".jsonl"}:
                files.append(path)

    files = sorted(set(files))
    if dry_run:
        for path in files:
            print(f"[DRY-RUN] borraria {path}")
        return {"files": len(files), "dirs": 0}

    for path in files:
        path.unlink()

    removed_dirs = 0
    for root_text in roots:
        root = _assert_safe_cleanup_root(Path(root_text))
        removed_dirs += _remove_empty_dirs(root)
    return {"files": len(files), "dirs": removed_dirs}


def _package_version(name: str) -> str | None:
    """Return installed version for a package, or None if not available."""
    try:
        return importlib_metadata.version(name)
    except importlib_metadata.PackageNotFoundError:
        return None


def _current_git_commit() -> str | None:
    """Return short git commit for the repo, or None if git is unavailable."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(BASE_DIR),
            stderr=subprocess.DEVNULL,
            timeout=3,
        )
        return out.decode().strip() or None
    except Exception:
        return None


def build_reproducibility_info() -> dict[str, Any]:
    """Collect runtime/version info for metadata reproducibility."""
    return {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "packages": {
            "ollama": _package_version("ollama"),
            "pypdf": _package_version("pypdf"),
            "pymupdf4llm": _package_version("pymupdf4llm"),
        },
        "git_commit": _current_git_commit(),
    }


# ----------------------------------------------------------------------
# SECTION 9: ENTRY
# ----------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    parser = argparse.ArgumentParser(description="Generate synthetic QA datasets from PDFs using Ollama.")
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR, help="Folder with PDF files.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH, help="Output JSONL path.")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Ollama model to generate QA pairs.")
    parser.add_argument(
        "--language",
        type=str,
        default=DEFAULT_LANGUAGE,
        help="Output language. Default: auto (detect per PDF). Use es/en/ca/etc. to force one.",
    )
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE, help="Chunk size in characters.")
    parser.add_argument("--chunk-overlap", type=int, default=DEFAULT_CHUNK_OVERLAP, help="Chunk overlap in chars.")
    parser.add_argument(
        "--num-topics",
        type=int,
        default=DEFAULT_NUM_TOPICS,
        help="Maximum number of topics to generate per document.",
    )
    parser.add_argument(
        "--questions-per-topic",
        type=int,
        default=DEFAULT_QUESTIONS_PER_TOPIC,
        help="How many Q/A items to request for each topic.",
    )
    parser.add_argument("--split", type=str, default=DEFAULT_SPLIT, help="train,val,test ratios, e.g. 0.8,0.1,0.1")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed for splitting.")
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE, help="Sampling temperature.")
    parser.add_argument("--max-chunks", type=int, default=None, help="Optional max chunks to process.")
    parser.add_argument(
        "--max-doc-context-chars",
        type=int,
        default=DEFAULT_MAX_DOC_CONTEXT_CHARS,
        help="Max chars to send as full-document context for topic mapping.",
    )
    parser.add_argument(
        "--max-topic-context-chars",
        type=int,
        default=DEFAULT_MAX_TOPIC_CONTEXT_CHARS,
        help="Max chars to send as per-topic context for question generation.",
    )
    parser.add_argument(
        "--debug-dir",
        type=Path,
        default=DEFAULT_DEBUG_DIR,
        help="Folder to store raw model outputs by document/topic.",
    )
    parser.add_argument(
        "--quality-gate",
        choices=sorted(VALID_QUALITY_GATES),
        default=DEFAULT_QUALITY_GATE,
        help="Quality filtering before writing final JSONL. Default: strict.",
    )
    parser.add_argument(
        "--only-doc",
        type=str,
        default=None,
        help="Process only the PDFs whose filename/stem match this value "
             "(comma-separated list for multiple).",
    )
    parser.add_argument(
        "--skip-model-check",
        action="store_true",
        help="Skip verifying that the Ollama model is available before starting.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip documents that already have a non-empty checkpoint file in --debug-dir.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Extract and chunk PDFs, print stats, and exit without calling Ollama.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Delete generated JSON/JSONL files from --output parent and --debug-dir, then exit.",
    )
    parser.add_argument(
        "--clean-dry-run",
        action="store_true",
        help="Show which generated files --clean would delete, without deleting them.",
    )
    return parser


def validate_args(args: argparse.Namespace) -> None:
    """Validate CLI args before starting expensive operations."""
    errors: list[str] = []
    if args.chunk_overlap >= args.chunk_size:
        errors.append(
            f"--chunk-overlap ({args.chunk_overlap}) debe ser < --chunk-size ({args.chunk_size})"
        )
    if args.num_topics < 1:
        errors.append("--num-topics debe ser >= 1")
    if args.questions_per_topic < 1:
        errors.append("--questions-per-topic debe ser >= 1")
    if not (0.0 <= args.temperature <= 2.0):
        errors.append(f"--temperature debe estar en [0.0, 2.0], recibido: {args.temperature}")
    if args.quality_gate not in VALID_QUALITY_GATES:
        errors.append(
            f"--quality-gate debe ser uno de {sorted(VALID_QUALITY_GATES)}, recibido: {args.quality_gate}"
        )

    if errors:
        for err in errors:
            print(f"[ERROR] {err}")
        raise SystemExit(1)


def _checkpoint_path(debug_dir: Path, pdf_path: Path) -> Path:
    """Return the per-document checkpoint JSONL path."""
    return debug_dir / f"{pdf_path.stem}.items.jsonl"


def load_checkpoint_items(debug_dir: Path, pdf_path: Path) -> list[dict[str, Any]]:
    """Load previously generated items for a document, or empty list if absent."""
    path = _checkpoint_path(debug_dir, pdf_path)
    if not path.exists() or path.stat().st_size == 0:
        return []
    items: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError:
                logger.warning("Checkpoint con linea corrupta en %s", path)
    return items


def save_checkpoint_items(debug_dir: Path, pdf_path: Path, items: Sequence[dict[str, Any]]) -> None:
    """Persist the items generated for a document (idempotent overwrite)."""
    path = _checkpoint_path(debug_dir, pdf_path)
    write_jsonl(path, list(items))


def filter_pdfs_by_only_doc(pdfs: Sequence[Path], only_doc: str | None) -> list[Path]:
    """Keep only the PDFs matching `only_doc` (comma-separated filenames or stems, case-insensitive)."""
    if not only_doc:
        return list(pdfs)
    targets = [t.strip().lower() for t in only_doc.split(",") if t.strip()]
    if not targets:
        return list(pdfs)

    matched: list[Path] = []
    seen: set = set()
    missing: list[str] = []
    for target in targets:
        hits = [p for p in pdfs if p.name.lower() == target or p.stem.lower() == target]
        if not hits:
            missing.append(target)
            continue
        for hit in hits:
            if hit not in seen:
                seen.add(hit)
                matched.append(hit)

    if missing:
        available = ", ".join(p.name for p in pdfs) or "(ninguno)"
        raise RuntimeError(
            f"--only-doc: no se encontraron {missing}. Disponibles: {available}"
        )
    return matched


def _run_dry_run(args: argparse.Namespace, pdf_files: Sequence[Path]) -> None:
    """Print how much work a real run would do, without calling Ollama."""
    print(f"[DRY-RUN] {len(pdf_files)} PDF(s) a procesar con modelo '{args.model}'")
    total_chunks = 0
    total_chars = 0
    for pdf_path in pdf_files:
        raw_text = extract_text_from_pdf(pdf_path)
        chunks = build_chunks_from_text(
            raw_text=raw_text,
            document_name=pdf_path.name,
            document_stem=pdf_path.stem,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            max_chunks=args.max_chunks,
        )
        total_chunks += len(chunks)
        total_chars += len(raw_text)
        _, detected_language, language_scores = detect_document_language(raw_text)
        print(
            f"  - {pdf_path.name}: {len(raw_text):>8} chars -> {len(chunks):>3} chunks "
            f"| idioma detectado: {detected_language} {language_scores}"
        )
    estimated_topics = len(pdf_files) * args.num_topics
    estimated_calls = len(pdf_files) + estimated_topics  # 1 topic-map + N topic-gen per doc
    estimated_items = estimated_topics * args.questions_per_topic
    print("")
    print(f"Totales: {total_chars} chars, {total_chunks} chunks")
    print(
        f"Estimacion maxima: ~{estimated_topics} topicos, ~{estimated_items} items, "
        f"~{estimated_calls} llamadas a Ollama."
    )


def main() -> None:
    """Run end-to-end dataset generation pipeline."""
    start_time = time.time()
    args = build_arg_parser().parse_args()
    validate_args(args)

    if args.clean or args.clean_dry_run:
        stats = clean_generated_artifacts(
            output_path=args.output,
            debug_dir=args.debug_dir,
            dry_run=args.clean_dry_run,
        )
        action = "Archivos que se borrarian" if args.clean_dry_run else "Archivos borrados"
        print(f"{action}: {stats['files']}")
        if not args.clean_dry_run:
            print(f"Directorios vacios borrados: {stats['dirs']}")
        return

    split = parse_split(args.split)

    pdf_files = list_pdf_files(args.source_dir)
    if not pdf_files:
        raise RuntimeError(f"No se encontraron PDFs en {args.source_dir}")

    pdf_files = filter_pdfs_by_only_doc(pdf_files, args.only_doc)

    if args.dry_run:
        _run_dry_run(args, pdf_files)
        return

    if not args.skip_model_check:
        verify_ollama_model(args.model)

    args.debug_dir.mkdir(parents=True, exist_ok=True)

    generated: list[dict[str, Any]] = []
    total_chunks = 0
    total_topics = 0
    resumed_docs = 0
    document_languages: dict[str, str] = {}

    for pdf_index, pdf_path in enumerate(pdf_files, start=1):
        print(f"[{pdf_index}/{len(pdf_files)}] Procesando documento: {pdf_path.name}")

        if args.resume:
            cached = load_checkpoint_items(args.debug_dir, pdf_path)
            if cached:
                generated.extend(cached)
                cached_language = str(cached[0].get("document_language", "unknown"))
                document_languages[pdf_path.name] = cached_language
                resumed_docs += 1
                print(f"  - Resume: {len(cached)} items cargados desde checkpoint; salto generacion.")
                continue

        doc_items: list[dict[str, Any]] = []
        raw_text = extract_text_from_pdf(pdf_path)
        full_context = truncate_text(raw_text, args.max_doc_context_chars)
        generation_language, document_language, language_scores = resolve_generation_language(args.language, raw_text)
        document_languages[pdf_path.name] = document_language

        chunks = build_chunks_from_text(
            raw_text=raw_text,
            document_name=pdf_path.name,
            document_stem=pdf_path.stem,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            max_chunks=args.max_chunks,
        )
        if not chunks:
            print(f"  - Saltado: sin chunks validos en {pdf_path.name}")
            continue
        total_chunks += len(chunks)
        print(f"  - Idioma: {document_language}")

        topic_messages = build_topic_map_messages(
            document=pdf_path.name,
            full_text=full_context,
            language=generation_language,
            num_topics=args.num_topics,
        )
        topics_payload, topics_raw_content = call_ollama_json(
            model=args.model,
            messages=topic_messages,
            temperature=args.temperature,
            seed=args.seed,
        )
        topics = parse_topics(topics_payload)[: args.num_topics]

        existing_questions = [str(item["question"]) for item in generated if item.get("document") == pdf_path.name]

        doc_debug = {
            "document": pdf_path.name,
            "language": document_language,
            "language_prompt": generation_language,
            "language_scores": language_scores,
            "topic_map_payload": topics_payload,
            "topic_map_raw_content": topics_raw_content,
            "topic_map_strategy": "llm",
            "topic_map_attempts": [
                {
                    "attempt": 1,
                    "strategy": "full_document",
                    "context_chars": len(full_context),
                    "raw_content": topics_raw_content,
                    "parsed_topics": len(topics),
                }
            ],
            "topics": [],
        }

        if not topics:
            compact_context = build_representative_document_context(
                chunks=chunks,
                max_chars=min(args.max_doc_context_chars, 18000),
            )
            compact_messages = build_topic_map_messages_compact(
                document=pdf_path.name,
                sampled_text=compact_context,
                language=generation_language,
                num_topics=args.num_topics,
            )
            topics_payload2, topics_raw_content2 = call_ollama_json(
                model=args.model,
                messages=compact_messages,
                temperature=args.temperature,
                seed=args.seed,
            )
            topics = parse_topics(topics_payload2)[: args.num_topics]
            doc_debug["topic_map_attempts"].append(
                {
                    "attempt": 2,
                    "strategy": "representative_excerpts",
                    "context_chars": len(compact_context),
                    "raw_content": topics_raw_content2,
                    "parsed_topics": len(topics),
                }
            )
            if topics:
                doc_debug["topic_map_strategy"] = "llm_compact"
                doc_debug["topic_map_payload"] = topics_payload2
                doc_debug["topic_map_raw_content"] = topics_raw_content2

        if not topics:
            logger.warning(
                "Topicos no parseables para '%s'. Raw (300 chars): %s",
                pdf_path.name,
                doc_debug["topic_map_attempts"][-1]["raw_content"][:300]
                if doc_debug["topic_map_attempts"][-1]["raw_content"]
                else "(vacio)",
            )
            chunk_topics = build_fallback_topics_from_chunks(chunks=chunks, num_topics=args.num_topics)
            section_topics = build_section_topics_from_text(raw_text, args.num_topics)
            if section_topics:
                topics = section_topics
                doc_debug["topic_map_strategy"] = "fallback_section_headings"
                print(
                    f"  - Aviso: el modelo no devolvio topicos parseables para {pdf_path.name}; "
                    f"usando secciones del documento ({len(topics)} topicos)."
                )
            else:
                topics = chunk_topics
                doc_debug["topic_map_strategy"] = "fallback_from_chunks"
                print(
                    f"  - Aviso: el modelo no devolvio topicos parseables para {pdf_path.name}; "
                    f"usando fallback de chunks ({len(topics)} topicos)."
                )
            doc_debug["topic_map_payload"] = {
                "topics": [
                    {"name": t.name, "summary": t.summary, "keywords": t.keywords}
                    for t in topics
                ]
            }
        elif topics_are_too_generic(topics) or topics_mostly_invalid(topics):
            section_topics = build_section_topics_from_text(raw_text, args.num_topics)
            if section_topics:
                doc_debug["llm_topics_replaced"] = [
                    {"name": t.name, "summary": t.summary, "keywords": t.keywords}
                    for t in topics
                ]
                topics = section_topics
                doc_debug["topic_map_strategy"] = f"{doc_debug['topic_map_strategy']}_section_refined"
                doc_debug["topic_map_payload"] = {
                    "topics": [
                        {"name": t.name, "summary": t.summary, "keywords": t.keywords}
                        for t in topics
                    ]
                }
                print("  - Aviso: topicos LLM invalidos o genericos; usando secciones del documento.")

        if not topics:
            print(f"  - Saltado: no se pudieron construir topicos para {pdf_path.name}")
            debug_path = args.debug_dir / f"{pdf_path.stem}.json"
            write_metadata(debug_path, doc_debug)
            continue

        total_topics += len(topics)
        print(f"  - Topicos detectados: {len(topics)}")
        for topic_idx, topic in enumerate(topics, start=1):
            print(f"    [{topic_idx}/{len(topics)}] {topic.name[:60]}", end="", flush=True)
            if not is_valid_topic_name(topic.name):
                print(" -> 0 items (nombre de topico invalido, saltado)")
                continue
            topic_context = build_topic_context(
                chunks=chunks,
                topic=topic,
                max_topic_context_chars=args.max_topic_context_chars,
            )
            if not topic_context.strip():
                print(" -> 0 items (sin contexto)")
                continue
            if (
                args.quality_gate != "strict"
                and len(topic_context) < MIN_TOPIC_CONTEXT_CHARS
                and len(full_context) > len(topic_context)
            ):
                topic_context = truncate_text(full_context, min(args.max_topic_context_chars, len(full_context)))

            topic_t0 = time.time()
            topic_items, topic_debug = generate_items_for_topic(
                model=args.model,
                document=pdf_path.name,
                topic=topic,
                topic_context=topic_context,
                language=generation_language,
                questions_per_topic=args.questions_per_topic,
                temperature=args.temperature,
                existing_questions=existing_questions,
                seed=args.seed,
                document_language=document_language,
            )
            topic_elapsed = time.time() - topic_t0
            generated.extend(topic_items)
            doc_items.extend(topic_items)
            existing_questions.extend(item["question"] for item in topic_items)
            print(f" -> {len(topic_items)} items ({topic_elapsed:.1f}s)")
            doc_debug["topics"].append(
                {
                    "topic_id": topic.topic_id,
                    "topic_name": topic.name,
                    "topic_keywords": topic.keywords,
                    "topic_context_chars": len(topic_context),
                    "generated_items": len(topic_items),
                    "raw": topic_debug,
                }
            )

        debug_path = args.debug_dir / f"{pdf_path.stem}.json"
        write_metadata(debug_path, doc_debug)
        save_checkpoint_items(args.debug_dir, pdf_path, doc_items)

    quality_candidates, rejected_rows, quality_stats = apply_quality_gate(generated, args.quality_gate)
    deduped = deduplicate_items(quality_candidates)
    quality_stats["accepted_items_before_dedup"] = len(quality_candidates)
    quality_stats["verified_items_before_dedup"] = quality_stats["verified_items"]
    quality_stats["duplicate_items_removed_after_quality"] = len(quality_candidates) - len(deduped)
    quality_stats["accepted_items"] = len(deduped)
    quality_stats["verified_items"] = sum(1 for item in deduped if item.get("context_source_verified"))
    train_rows, val_rows, test_rows = split_rows(deduped, split=split, seed=args.seed)

    write_jsonl(args.output, deduped)
    output_base = args.output.with_suffix("")
    write_jsonl(Path(f"{output_base}_train.jsonl"), train_rows)
    write_jsonl(Path(f"{output_base}_val.jsonl"), val_rows)
    write_jsonl(Path(f"{output_base}_test.jsonl"), test_rows)
    write_jsonl(Path(f"{output_base}.rejected.jsonl"), rejected_rows)

    verified_count = sum(1 for item in deduped if item.get("context_source_verified"))
    metadata = {
        "created_at": now_iso(),
        "model": args.model,
        "language": args.language,
        "document_languages": document_languages,
        "source_dir": str(args.source_dir),
        "output": str(args.output),
        "pdf_count": len(pdf_files),
        "chunk_count": total_chunks,
        "topic_count": total_topics,
        "generated_items": len(generated),
        "deduplicated_items": len(deduped),
        "accepted_items": len(deduped),
        "rejected_items": len(rejected_rows),
        "context_source_verified_items": verified_count,
        "quality": quality_stats,
        "resumed_documents": resumed_docs,
        "split": {"train": len(train_rows), "val": len(val_rows), "test": len(test_rows)},
        "params": {
            "chunk_size": args.chunk_size,
            "chunk_overlap": args.chunk_overlap,
            "num_topics": args.num_topics,
            "questions_per_topic": args.questions_per_topic,
            "temperature": args.temperature,
            "seed": args.seed,
            "max_doc_context_chars": args.max_doc_context_chars,
            "max_topic_context_chars": args.max_topic_context_chars,
            "quality_gate": args.quality_gate,
            "only_doc": args.only_doc,
            "resume": args.resume,
        },
        "runtime": build_reproducibility_info(),
    }
    write_metadata(args.output.with_suffix(".meta.json"), metadata)

    print("")
    if not deduped:
        print("[AVISO] El dataset final quedo vacio. Revisar:")
        print("  - Logs en run_logs/ por cada PDF (topic_map_strategy, attempts).")
        print("  - Que el modelo Ollama devuelve JSON valido para este idioma.")
        print("  - Que los PDFs contienen texto extraible (no solo imagenes).")
        print(f"  - Items generados: {len(generated)} | aceptados pre-dedup: {len(quality_candidates)}")
    else:
        print("Dataset generado correctamente.")
    print(f"- Total items: {len(deduped)}")
    print(f"- Items rechazados por quality gate: {len(rejected_rows)}")
    print(f"- Archivo principal: {args.output}")
    print(f"- Rechazados: {Path(f'{output_base}.rejected.jsonl')}")
    print(f"- Splits: train={len(train_rows)} val={len(val_rows)} test={len(test_rows)}")
    print(f"- context_source verificado (substring literal): {verified_count}/{len(deduped)}")
    if args.resume:
        print(f"- Documentos reanudados desde checkpoint: {resumed_docs}")
    print(f"- Metadata: {args.output.with_suffix('.meta.json')}")
    print(f"- Tiempo total: {time.time() - start_time:.1f}s")


if __name__ == "__main__":
    main()
