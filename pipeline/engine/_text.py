"""Text transformation utilities: cleaning, language detection, keywords, math helpers."""

import math
import random
import re
import unicodedata
from collections.abc import Sequence
from datetime import datetime, timezone

from engine._config import LANGUAGE_MARKERS, LANGUAGE_NAMES, STOPWORDS


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
    cleaned = cleaned.replace("Ã¢â‚¬â€", " - ").replace("Ã¢â‚¬â€œ", " - ")
    cleaned = cleaned.replace("Ã¢â‚¬Ëœ", "'").replace("Ã¢â‚¬â„¢", "'")
    cleaned = cleaned.replace('Ã¢â‚¬Å"', '"').replace("Ã¢â‚¬Â", '"')
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
    cleaned = cleaned.replace("Ã¢â‚¬â€", " - ").replace("Ã¢â‚¬â€œ", " - ")
    cleaned = cleaned.replace("Ã¢â‚¬Ëœ", "'").replace("Ã¢â‚¬â„¢", "'")
    cleaned = cleaned.replace('Ã¢â‚¬Å"', '"').replace("Ã¢â‚¬Â", '"')
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


def truncate_text(text: str, max_chars: int) -> str:
    """Trim text to max chars preserving sentence boundaries when possible."""
    if len(text) <= max_chars:
        return text
    partial = text[:max_chars]
    last_break = max(partial.rfind(". "), partial.rfind("\n"))
    if last_break > int(max_chars * 0.6):
        return partial[: last_break + 1]
    return partial


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


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two vectors; returns 0.0 if either norm is zero."""
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)
