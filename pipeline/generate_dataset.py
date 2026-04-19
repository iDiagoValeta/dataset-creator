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
import random
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

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
DEFAULT_LANGUAGE = os.getenv("DATASET_LANGUAGE", "es")
DEFAULT_CHUNK_SIZE = int(os.getenv("DATASET_CHUNK_SIZE", "3500"))
DEFAULT_CHUNK_OVERLAP = int(os.getenv("DATASET_CHUNK_OVERLAP", "350"))
DEFAULT_NUM_TOPICS = int(os.getenv("DATASET_NUM_TOPICS", "8"))
DEFAULT_QUESTIONS_PER_TOPIC = int(os.getenv("DATASET_QUESTIONS_PER_TOPIC", "6"))
DEFAULT_SPLIT = os.getenv("DATASET_SPLIT", "0.8,0.1,0.1")
DEFAULT_TEMPERATURE = float(os.getenv("DATASET_TEMPERATURE", "0.2"))
DEFAULT_SEED = int(os.getenv("DATASET_SEED", "42"))
DEFAULT_MAX_DOC_CONTEXT_CHARS = int(os.getenv("DATASET_MAX_DOC_CONTEXT_CHARS", "110000"))
DEFAULT_MAX_TOPIC_CONTEXT_CHARS = int(os.getenv("DATASET_MAX_TOPIC_CONTEXT_CHARS", "24000"))
DEFAULT_OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT_SECS", "300"))
MIN_TOPIC_CONTEXT_CHARS = 9000
VALID_TYPES = {"factual", "conceptual", "inference", "compare", "definition", "reasoning"}
_STOPWORDS_SET = {
    "cual",
    "como",
    "para",
    "este",
    "esta",
    "estos",
    "estas",
    "sobre",
    "desde",
    "hasta",
    "entre",
    "donde",
    "cuando",
    "que",
    "del",
    "las",
    "los",
    "una",
    "uno",
    "unos",
    "unas",
    "con",
    "sin",
    "por",
    "what",
    "which",
    "how",
    "does",
    "that",
    "with",
    "from",
    "have",
    "has",
    "this",
    "these",
    "those",
}

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
    keywords: List[str]


# ----------------------------------------------------------------------
# SECTION 4: UTILITIES
# ----------------------------------------------------------------------

def parse_split(split_text: str) -> Tuple[float, float, float]:
    """Parse and validate split ratios in 'train,val,test' format."""
    parts = [p.strip() for p in split_text.split(",")]
    if len(parts) != 3:
        raise ValueError("Split debe tener 3 valores: train,val,test")

    train_ratio, val_ratio, test_ratio = (float(p) for p in parts)
    total = train_ratio + val_ratio + test_ratio
    if total <= 0:
        raise ValueError("La suma del split debe ser > 0")

    return train_ratio / total, val_ratio / total, test_ratio / total


def list_pdf_files(source_dir: Path) -> List[Path]:
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


def clean_markdown_artifacts(text: str) -> str:
    """Remove markdown/wiki artifacts that hurt topic and QA generation quality."""
    cleaned = text
    cleaned = re.sub(r"\[([^\]]+)\]\((https?://[^\)]+)\)", r"\1", cleaned)
    cleaned = re.sub(r"https?://\S+", " ", cleaned)
    cleaned = re.sub(r"\[\[\d+\]\]", " ", cleaned)
    cleaned = re.sub(r"[~`*_#>{}\[\]|]+", " ", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    return normalize_whitespace(cleaned)


def now_iso() -> str:
    """Return UTC timestamp in ISO format."""
    return datetime.now(timezone.utc).isoformat()


def deduplicate_preserve_order(values: Sequence[str]) -> List[str]:
    """Remove duplicates preserving first appearance."""
    seen = set()
    out: List[str] = []
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


def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """Split text into overlapping character chunks."""
    cleaned = normalize_whitespace(text)
    if not cleaned:
        return []

    chunks: List[str] = []
    step = max(1, chunk_size - chunk_overlap)
    for start in range(0, len(cleaned), step):
        chunk = cleaned[start : start + chunk_size].strip()
        if len(chunk) >= 120:
            chunks.append(chunk)
        if start + chunk_size >= len(cleaned):
            break
    return chunks


def build_chunks_from_pdfs(
    pdf_paths: Sequence[Path],
    chunk_size: int,
    chunk_overlap: int,
    max_chunks: int | None,
) -> List[Chunk]:
    """Extract and chunk all PDFs into a flat list of Chunk objects."""
    chunks: List[Chunk] = []
    for pdf_path in pdf_paths:
        raw_text = extract_text_from_pdf(pdf_path)
        part_chunks = chunk_text(raw_text, chunk_size, chunk_overlap)
        for index, text in enumerate(part_chunks):
            chunk_id = f"{pdf_path.stem}-chunk-{index:04d}"
            chunks.append(Chunk(chunk_id=chunk_id, document=pdf_path.name, text=text))
            if max_chunks is not None and len(chunks) >= max_chunks:
                return chunks
    return chunks


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


def build_topic_map_messages(document: str, full_text: str, language: str, num_topics: int) -> List[Dict[str, str]]:
    """Create prompt messages to infer a non-overlapping topic map."""
    system_prompt = (
        "You are an expert dataset designer for educational and technical corpora. "
        "Return strict JSON only. No markdown. No extra keys."
    )

    user_prompt = f"""
Analyze this full document and extract a topic map.
Language for topic names and summaries: {language}.
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


def build_topic_generation_messages(
    document: str,
    topic: Topic,
    topic_context: str,
    language: str,
    questions_per_topic: int,
    existing_questions: Sequence[str],
) -> List[Dict[str, str]]:
    """Create prompt messages to generate topic-specific non-repeated Q/A items."""
    system_prompt = (
        "You generate high-quality supervised Q/A datasets from source text. "
        "Return strict JSON only. No markdown. No extra keys."
    )

    max_existing_in_prompt = max(80, questions_per_topic * 10)
    existing_block = "\n".join(f"- {q}" for q in existing_questions[-max_existing_in_prompt:])
    user_prompt = f"""
Generate exactly {questions_per_topic} Q/A items for the topic below.
Language for question and answer: {language}.

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
) -> List[Dict[str, str]]:
    """Compact fallback prompt to maximize schema adherence."""
    system_prompt = "Return strict JSON object only."
    user_prompt = f"""
Create {questions_per_topic} question-answer pairs in {language} from the topic context.

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


def try_parse_json_payload(payload: str) -> Dict[str, Any]:
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

    logger.warning("No se pudo parsear JSON del modelo; se retorna payload vacio.")
    return {"items": []}


# ----------------------------------------------------------------------
# SECTION 7: OLLAMA GENERATION PIPELINE
# ----------------------------------------------------------------------

def call_ollama_json(
    model: str,
    messages: List[Dict[str, str]],
    temperature: float,
) -> Tuple[Dict[str, Any], str]:
    """Execute Ollama chat call expecting a JSON object."""
    try:
        client = ollama.Client(timeout=DEFAULT_OLLAMA_TIMEOUT)
        response = client.chat(
            model=model,
            messages=messages,
            format="json",
            options={"temperature": temperature},
            think=False,
        )
        content = response.get("message", {}).get("content", "")
        return try_parse_json_payload(content), content
    except ollama.ResponseError as exc:
        logger.error("Ollama ResponseError en modelo '%s': %s", model, exc)
    except Exception as exc:
        logger.error("Error de red/timeout al llamar Ollama '%s': %s", model, exc)
    return {"items": []}, ""


def _extract_topics_candidates(payload: Dict[str, Any]) -> List[Any]:
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
    return []


def parse_topics(payload: Dict[str, Any]) -> List[Topic]:
    """Normalize topic payload into typed topics."""
    topics: List[Topic] = []
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
            summary = str(raw.get("summary") or raw.get("description") or raw.get("resumen") or "").strip()
            keywords_raw = raw.get("keywords") or raw.get("tags") or raw.get("palabras_clave") or []
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

        topics.append(
            Topic(
                topic_id=f"topic-{idx:02d}",
                name=name,
                summary=summary or name,
                keywords=keywords,
            )
        )
    return topics


def extract_keywords(text: str, max_keywords: int = 8) -> List[str]:
    """Extract naive keywords from a text snippet without external deps."""
    stopwords = {
        "que",
        "para",
        "como",
        "este",
        "esta",
        "estos",
        "estas",
        "sobre",
        "desde",
        "hasta",
        "entre",
        "donde",
        "cuando",
        "cual",
        "del",
        "las",
        "los",
        "una",
        "uno",
        "unos",
        "unas",
        "con",
        "sin",
        "por",
        "the",
        "and",
        "for",
        "that",
        "from",
        "with",
        "this",
        "these",
    }
    words = re.findall(r"[A-Za-zÀ-ÿ][A-Za-zÀ-ÿ0-9_-]{3,}", text.lower())
    counts: Dict[str, int] = {}
    for w in words:
        if w in stopwords:
            continue
        counts[w] = counts.get(w, 0) + 1
    ordered = sorted(counts.items(), key=lambda item: item[1], reverse=True)
    return [word for word, _ in ordered[:max_keywords]]


def infer_topic_name_from_chunk(chunk_text: str, fallback_index: int) -> str:
    """Build a readable topic name from first meaningful line."""
    for line in chunk_text.splitlines():
        line = line.strip().strip("-*#")
        if len(line) < 8:
            continue
        # Prefer short heading-like lines.
        if len(line) <= 90:
            return line
        break
    words = re.findall(r"[A-Za-zÀ-ÿ0-9_-]+", chunk_text)[:8]
    if words:
        return " ".join(words)
    return f"Topico {fallback_index + 1}"


def build_fallback_topics_from_chunks(chunks: Sequence[Chunk], num_topics: int) -> List[Topic]:
    """Create deterministic topics from chunk slices when LLM topic map fails."""
    if not chunks:
        return []
    count = max(1, min(num_topics, len(chunks)))
    step = max(1, len(chunks) // count)
    selected_indices = list(range(0, len(chunks), step))[:count]
    topics: List[Topic] = []
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
    scored = [(score_chunk_for_topic(chunk, topic), chunk) for chunk in chunks]
    scored.sort(key=lambda item: item[0], reverse=True)

    selected: List[str] = []
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


def generate_items_for_topic(
    model: str,
    document: str,
    topic: Topic,
    topic_context: str,
    language: str,
    questions_per_topic: int,
    temperature: float,
    existing_questions: Sequence[str],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Generate normalized Q/A items for one topic."""
    debug_attempts: List[Dict[str, Any]] = []

    def _extract_raw_items(parsed_obj: Dict[str, Any]) -> List[Any]:
        raw_local = parsed_obj.get("items", [])
        if not isinstance(raw_local, list):
            for alt_key in ("questions", "preguntas", "data", "results"):
                alt_value = parsed_obj.get(alt_key)
                if isinstance(alt_value, list):
                    raw_local = alt_value
                    break
        return raw_local if isinstance(raw_local, list) else []

    def _normalize_items(raw_items_local: List[Any]) -> List[Dict[str, Any]]:
        normalized_local: List[Dict[str, Any]] = []
        for idx, item in enumerate(raw_items_local):
            if not isinstance(item, dict):
                continue
            question = str(item.get("question", "")).strip()
            answer = str(item.get("answer", "")).strip()
            if not question or not answer:
                continue
            raw_type = str(item.get("type", "factual")).strip().lower()
            item_type = raw_type if raw_type in VALID_TYPES else "factual"
            context_source = str(item.get("context_source", "")).strip()[:300] or topic_context[:300]
            normalized_local.append(
                {
                    "id": f"{document}-{topic.topic_id}-qa-{idx:02d}",
                    "question": question,
                    "answer": answer,
                    "type": item_type,
                    "difficulty": str(item.get("difficulty", "medium")).strip().lower(),
                    "context_source": context_source,
                    "topic": topic.name,
                    "topic_id": topic.topic_id,
                    "topic_keywords": topic.keywords,
                    "document": document,
                    "created_at": now_iso(),
                    "context_excerpt": topic_context[:500],
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
    parsed, raw_content = call_ollama_json(model=model, messages=messages, temperature=temperature)
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
        parsed2, raw_content2 = call_ollama_json(model=model, messages=compact_messages, temperature=temperature)
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
        parsed3, raw_content3 = call_ollama_json(model=model, messages=salvage_messages, temperature=temperature)
        raw_items = _extract_raw_items(parsed3)
        debug_attempts.append(
            {"attempt": 3, "raw_content": raw_content3, "parsed": parsed3, "raw_items_count": len(raw_items)}
        )

    normalized = _normalize_items(raw_items)
    return normalized, {"raw_items": raw_items, "attempts": debug_attempts, "topic": topic.name}


def _question_bigrams(text: str) -> frozenset:
    """Build word-bigrams signature for semantic near-duplicate detection."""
    words = [
        w
        for w in re.findall(r"[A-Za-zÀ-ÿ]{4,}", text.lower())
        if w not in _STOPWORDS_SET
    ]
    if len(words) >= 2:
        return frozenset(zip(words, words[1:]))
    return frozenset(words)


def deduplicate_items(items: Iterable[Dict[str, Any]], semantic_threshold: float = 0.6) -> List[Dict[str, Any]]:
    """Drop exact and near-semantic duplicated questions."""
    seen_exact = set()
    seen_bigrams: List[frozenset] = []
    unique: List[Dict[str, Any]] = []
    for item in items:
        question = str(item.get("question", ""))
        signature = sanitize_question(question)
        if not signature or signature in seen_exact:
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
        seen_bigrams.append(q_bigrams)
        unique.append(item)
    return unique


# ----------------------------------------------------------------------
# SECTION 8: EXPORT AND SPLITTING
# ----------------------------------------------------------------------

def write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    """Write rows as JSON Lines."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def split_rows(
    rows: Sequence[Dict[str, Any]],
    split: Tuple[float, float, float],
    seed: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
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


def write_metadata(path: Path, metadata: Dict[str, Any]) -> None:
    """Write metadata as pretty JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


# ----------------------------------------------------------------------
# SECTION 9: ENTRY
# ----------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    parser = argparse.ArgumentParser(description="Generate synthetic QA datasets from PDFs using Ollama.")
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR, help="Folder with PDF files.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH, help="Output JSONL path.")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Ollama model to generate QA pairs.")
    parser.add_argument("--language", type=str, default=DEFAULT_LANGUAGE, help="Output language, e.g. es/en/ca.")
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
    return parser


def validate_args(args: argparse.Namespace) -> None:
    """Validate CLI args before starting expensive operations."""
    errors: List[str] = []
    if args.chunk_overlap >= args.chunk_size:
        errors.append(
            f"--chunk-overlap ({args.chunk_overlap}) debe ser < --chunk-size ({args.chunk_size})"
        )
    if args.num_topics < 1:
        errors.append("--num-topics debe ser >= 1")
    if args.questions_per_topic < 1:
        errors.append("--questions-per-topic debe ser >= 1")
    if not (0.0 < args.temperature <= 2.0):
        errors.append(f"--temperature debe estar en (0.0, 2.0], recibido: {args.temperature}")

    if errors:
        for err in errors:
            print(f"[ERROR] {err}")
        raise SystemExit(1)


def main() -> None:
    """Run end-to-end dataset generation pipeline."""
    start_time = time.time()
    args = build_arg_parser().parse_args()
    validate_args(args)
    split = parse_split(args.split)

    pdf_files = list_pdf_files(args.source_dir)
    if not pdf_files:
        raise RuntimeError(f"No se encontraron PDFs en {args.source_dir}")

    args.debug_dir.mkdir(parents=True, exist_ok=True)

    generated: List[Dict[str, Any]] = []
    total_chunks = 0
    total_topics = 0

    for pdf_index, pdf_path in enumerate(pdf_files, start=1):
        print(f"[{pdf_index}/{len(pdf_files)}] Procesando documento: {pdf_path.name}")
        raw_text = extract_text_from_pdf(pdf_path)
        full_context = truncate_text(raw_text, args.max_doc_context_chars)

        chunks = build_chunks_from_pdfs(
            pdf_paths=[pdf_path],
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            max_chunks=args.max_chunks,
        )
        if not chunks:
            print(f"  - Saltado: sin chunks validos en {pdf_path.name}")
            continue
        total_chunks += len(chunks)

        topic_messages = build_topic_map_messages(
            document=pdf_path.name,
            full_text=full_context,
            language=args.language,
            num_topics=args.num_topics,
        )
        topics_payload, topics_raw_content = call_ollama_json(
            model=args.model,
            messages=topic_messages,
            temperature=args.temperature,
        )
        topics = parse_topics(topics_payload)

        existing_questions = [str(item["question"]) for item in generated if item.get("document") == pdf_path.name]

        doc_debug = {
            "document": pdf_path.name,
            "topic_map_payload": topics_payload,
            "topic_map_raw_content": topics_raw_content,
            "topic_map_strategy": "llm",
            "topics": [],
        }

        if not topics:
            logger.warning(
                "Topicos no parseables para '%s'. Raw (300 chars): %s",
                pdf_path.name,
                topics_raw_content[:300] if topics_raw_content else "(vacio)",
            )
            topics = build_fallback_topics_from_chunks(chunks=chunks, num_topics=args.num_topics)
            doc_debug["topic_map_strategy"] = "fallback_from_chunks"
            doc_debug["topic_map_payload"] = {
                "topics": [
                    {"name": t.name, "summary": t.summary, "keywords": t.keywords}
                    for t in topics
                ]
            }
            print(
                f"  - Aviso: el modelo no devolvio topicos parseables para {pdf_path.name}; "
                f"usando fallback ({len(topics)} topicos)."
            )

        if not topics:
            print(f"  - Saltado: no se pudieron construir topicos para {pdf_path.name}")
            debug_path = args.debug_dir / f"{pdf_path.stem}.json"
            write_metadata(debug_path, doc_debug)
            continue

        total_topics += len(topics)
        print(f"  - Topicos detectados: {len(topics)}")
        for topic_idx, topic in enumerate(topics, start=1):
            print(f"    [{topic_idx}/{len(topics)}] {topic.name[:60]}", end="", flush=True)
            topic_context = build_topic_context(
                chunks=chunks,
                topic=topic,
                max_topic_context_chars=args.max_topic_context_chars,
            )
            if not topic_context.strip():
                print(" -> 0 items (sin contexto)")
                continue
            if len(topic_context) < MIN_TOPIC_CONTEXT_CHARS and len(full_context) > len(topic_context):
                topic_context = truncate_text(full_context, min(args.max_topic_context_chars, len(full_context)))

            topic_items, topic_debug = generate_items_for_topic(
                model=args.model,
                document=pdf_path.name,
                topic=topic,
                topic_context=topic_context,
                language=args.language,
                questions_per_topic=args.questions_per_topic,
                temperature=args.temperature,
                existing_questions=existing_questions,
            )
            generated.extend(topic_items)
            existing_questions.extend(item["question"] for item in topic_items)
            print(f" -> {len(topic_items)} items")
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

    deduped = deduplicate_items(generated)
    train_rows, val_rows, test_rows = split_rows(deduped, split=split, seed=args.seed)

    write_jsonl(args.output, deduped)
    output_base = args.output.with_suffix("")
    write_jsonl(Path(f"{output_base}_train.jsonl"), train_rows)
    write_jsonl(Path(f"{output_base}_val.jsonl"), val_rows)
    write_jsonl(Path(f"{output_base}_test.jsonl"), test_rows)

    metadata = {
        "created_at": now_iso(),
        "model": args.model,
        "language": args.language,
        "source_dir": str(args.source_dir),
        "output": str(args.output),
        "pdf_count": len(pdf_files),
        "chunk_count": total_chunks,
        "topic_count": total_topics,
        "generated_items": len(generated),
        "deduplicated_items": len(deduped),
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
        },
    }
    write_metadata(args.output.with_suffix(".meta.json"), metadata)

    print("")
    print("Dataset generado correctamente.")
    print(f"- Total items: {len(deduped)}")
    print(f"- Archivo principal: {args.output}")
    print(f"- Splits: train={len(train_rows)} val={len(val_rows)} test={len(test_rows)}")
    print(f"- Metadata: {args.output.with_suffix('.meta.json')}")
    print(f"- Tiempo total: {time.time() - start_time:.1f}s")


if __name__ == "__main__":
    main()
