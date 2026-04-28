"""Topic parsing, validation, fallback construction, and chunk retrieval."""

import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from engine._config import Chunk, Topic, logger
from engine._ollama import get_topic_embedding, score_chunk_for_topic_semantic
from engine._text import (
    deduplicate_preserve_order,
    extract_keywords,
    normalize_whitespace,
    truncate_text,
)

try:
    import yaml as _yaml_lib

    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

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
        if not re.search(r"[A-Za-zÀ-ÿ]{6,}", name):
            return False
    if name.count("/") > 2:
        return False
    return True


def _extract_topics_candidates(payload: dict[str, Any]) -> list[Any]:
    """Find topic candidates under common keys used by different models."""
    if not isinstance(payload, dict):
        return []

    keys = ["topics", "temas", "topicos", "items", "sections", "data"]
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


def load_topics_file(path: Path) -> list[Topic]:
    """Load user-defined topics from a YAML (.yaml/.yml) or plain-text file."""
    suffix = path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        if not YAML_AVAILABLE:
            raise RuntimeError(
                "pyyaml es necesario para leer archivos YAML. "
                "Instálalo con: pip install pyyaml  o usa un archivo .txt"
            )
        with path.open("r", encoding="utf-8") as fh:
            data = _yaml_lib.safe_load(fh)
        raw_topics = data.get("topics", []) if isinstance(data, dict) else []
        topics: list[Topic] = []
        for idx, item in enumerate(raw_topics):
            if not isinstance(item, dict):
                continue
            name = str(item.get("name", "")).strip()
            if not name:
                continue
            summary = str(item.get("summary", name)).strip() or name
            raw_kw = item.get("keywords", None)
            keywords: list[str] = (
                [str(k) for k in raw_kw if k]
                if isinstance(raw_kw, list) and raw_kw
                else extract_keywords(name, max_keywords=6)
            )
            topics.append(Topic(topic_id=f"user-{idx:02d}", name=name, summary=summary, keywords=keywords))
    else:
        topics = []
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                topics.append(
                    Topic(
                        topic_id=f"user-{len(topics):02d}",
                        name=line,
                        summary=line,
                        keywords=extract_keywords(line, max_keywords=6),
                    )
                )
    if not topics:
        raise ValueError(f"--topics-file: no se encontraron temas válidos en {path}")
    return topics


def load_questions_file(path: Path) -> list[str]:
    """Load seed questions from a plain-text file (one per line)."""
    questions: list[str] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            questions.append(line)
    if not questions:
        raise ValueError(f"--questions-file: no se encontraron preguntas en {path}")
    return questions


def questions_to_topics(questions: list[str]) -> list[Topic]:
    """Convert seed questions into synthetic Topic objects for the existing pipeline."""
    return [
        Topic(
            topic_id=f"seed-{idx:02d}",
            name=q,
            summary=q,
            keywords=extract_keywords(q, max_keywords=6),
        )
        for idx, q in enumerate(questions)
    ]


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
    retrieval_mode: str = "lexical",
    embedding_model: str = "",
    embedding_cache: dict[str, list[float]] | None = None,
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

    if retrieval_mode == "semantic":
        cache: dict[str, list[float]] = embedding_cache if embedding_cache is not None else {}
        topic_emb = get_topic_embedding(topic, embedding_model)
        if not topic_emb:
            logger.warning("Embedding vacío para topic '%s'; usando scoring léxico.", topic.name)
            scored = [(float(score_chunk_for_topic(chunk, topic)), chunk) for chunk in chunks]
        else:
            scored = [
                (score_chunk_for_topic_semantic(chunk, topic, topic_emb, embedding_model, cache), chunk)
                for chunk in chunks
            ]
    else:
        scored = [(float(score_chunk_for_topic(chunk, topic)), chunk) for chunk in chunks]

    scored.sort(key=lambda item: item[0], reverse=True)

    selected: list[str] = []
    total = 0
    fallback_mode = all(score <= 0.0 for score, _ in scored)

    for score, chunk in scored:
        if not fallback_mode and score <= 0.0:
            continue
        block = f"[{chunk.chunk_id}] {chunk.text}"
        if total + len(block) > max_topic_context_chars:
            break
        selected.append(block)
        total += len(block)
        if total >= max_topic_context_chars:
            break

    return "\n\n".join(selected)
