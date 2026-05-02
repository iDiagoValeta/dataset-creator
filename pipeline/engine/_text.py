"""Text transformation utilities: cleaning, language detection, keywords, math helpers."""

import math
import random
import re
import unicodedata
from collections.abc import Sequence
from datetime import datetime, timezone

from engine._config import LANGUAGE_MARKERS, LANGUAGE_NAMES, STOPWORDS

try:
    import ftfy as _ftfy

    _FTFY_AVAILABLE = True
except ImportError:
    _FTFY_AVAILABLE = False

# ---------------------------------------------------------------------------
# Encoding normalisation (mojibake fix)
# ---------------------------------------------------------------------------
# When pymupdf4llm reads PDFs whose internal fonts encode text as Windows-1252
# but do not declare the encoding, the raw bytes are returned without conversion.
# pymupdf4llm writes them as a UTF-8 markdown string, producing double-encoded
# sequences known as "mojibake".  For example, U+2013 EN DASH (UTF-8: E2 80 93)
# whose bytes are then read as three Windows-1252 codepoints:
#   0xE2 -> U+00E2 (â), 0x80 -> U+20AC (€), 0x93 -> U+201C (")
# so the en-dash appears as the three-char sequence "â€“".
#
# All characters below are spelled out as \u escapes to avoid any source-file
# encoding ambiguity — Python 3 always treats \u escapes as exact codepoints.

_MOJIBAKE_MAP: list[tuple[str, str]] = [
    # ---- Dashes ----
    # EN DASH U+2013: UTF-8 E2 80 93 -> Win-1252: 0xE2=U+00E2, 0x80=U+20AC, 0x93=U+201C
    ("â€“", "–"),   # â€" -> en-dash  (0x93=left-dbl-quote in Win-1252)
    # EM DASH U+2014: UTF-8 E2 80 94 -> Win-1252: 0x94=U+201D
    ("â€”", "—"),   # â€" -> em-dash
    # ---- Directional double quotes ----
    # LEFT U+201C: UTF-8 E2 80 9C -> Win-1252: 0x9C=U+0153
    ("â€œ", "“"),   # â€œ -> left double quote
    # RIGHT U+201D: UTF-8 E2 80 9D -> Win-1252: 0x9D=U+017D
    ("â€Ž", "”"),   # â€ -> right double quote
    # ---- Directional single quotes ----
    # LEFT U+2018: UTF-8 E2 80 98 -> Win-1252: 0x98=U+02DC (small tilde)
    ("â€˜", "‘"),   # â€˜ -> left single quote
    # RIGHT U+2019 / APOSTROPHE: UTF-8 E2 80 99 -> Win-1252: 0x99=U+2122 (trade mark)
    ("â€™", "’"),   # â€™ -> right single quote / apostrophe
    # ---- Greek letters (UTF-8 2-byte sequences read as Latin-1) ----
    # theta U+03B8: UTF-8 CE B8 -> 0xCE=U+00CE (Î), 0xB8=U+00B8 (¸)
    ("Î¸", "θ"),         # Î¸ -> theta
    # alpha U+03B1: UTF-8 CE B1 -> 0xB1=U+00B1 (±)
    ("Î±", "α"),         # Î± -> alpha
    # beta U+03B2: UTF-8 CE B2 -> 0xB2=U+00B2 (²)
    ("Î²", "β"),         # Î² -> beta
    # gamma U+03B3: UTF-8 CE B3 -> 0xB3=U+00B3 (³)
    ("Î³", "γ"),         # Î³ -> gamma
    # delta U+03B4: UTF-8 CE B4 -> 0xB4=U+00B4 (´)
    ("Î´", "δ"),         # Î´ -> delta
    # sigma U+03C3: UTF-8 CF 83 -> 0xCF=U+00CF (Ï), 0x83=U+0192 (ƒ)
    ("Ïƒ", "σ"),         # Ïƒ -> sigma
    # mu U+03BC: UTF-8 CE BC -> 0xBC=U+00BC (¼)
    ("Î¼", "μ"),         # Î¼ -> mu
    # lambda U+03BB: UTF-8 CE BB -> 0xBB=U+00BB (»)
    ("Î»", "λ"),         # Î» -> lambda
    # ---- Math symbols (UTF-8 3-byte sequences read as Latin-1/Win-1252) ----
    # ELEMENT OF U+2208: UTF-8 E2 88 88 -> 0x88=U+02C6 (ˆ)
    ("âˆˆ", "∈"),   # âˆˆ -> element-of (∈)
    # N-ARY SUMMATION U+2211: UTF-8 E2 88 91 -> 0x91=U+2018 (left single quote)
    ("âˆ‘", "∑"),   # âˆ' -> summation (∑)
    # SQUARE ROOT U+221A: UTF-8 E2 88 9A -> 0x9A=U+0161 (s with caron)
    ("âˆš", "√"),   # âˆš -> square root (√)
    # LESS-THAN-OR-EQUAL U+2264: UTF-8 E2 89 A4 -> 0x89=U+2030 (‰), 0xA4=U+00A4 (¤)
    ("â‰¤", "≤"),   # â‰¤ -> ≤
    # GREATER-THAN-OR-EQUAL U+2265: UTF-8 E2 89 A5 -> 0xA5=U+00A5 (¥)
    ("â‰¥", "≥"),   # â‰¥ -> ≥
    # PLUS-MINUS U+00B1: UTF-8 C2 B1 -> 0xC2=U+00C2 (Â), 0xB1=U+00B1 (±)
    ("Â±", "±"),         # Â± -> ±
    # MIDDLE DOT U+00B7: UTF-8 C2 B7 -> 0xB7=U+00B7 (·)
    ("Â·", "·"),         # Â· -> middle dot
    # ---- Whitespace ----
    ("\xa0", " "),   # NO-BREAK SPACE -> regular space
    ("\x00", ""),    # NULL byte -> remove
]

# Pattern to detect surviving mojibake as a quality-gate safety net.
# The sequence U+00E2 U+20AC is the universal mojibake prefix for anything
# whose UTF-8 starts with E2 80 xx (covers dashes, quotes, bullets, etc.).
# U+00CE followed by common bytes covers Greek letter mojibake.
# All written as \u escapes to avoid source-encoding ambiguity.
MOJIBAKE_PATTERN = re.compile(
    "â€"               # â€ prefix (covers â€™ â€" â€" etc.)
    "|Î[¸±²³´µ¼»]"  # Î¸ Î± Î² etc.
    "|Ïƒ"              # Ïƒ (sigma mojibake)
    "|âˆˆ"        # âˆˆ (element-of mojibake)
    "|Ã¢"              # Ã¢ (another common prefix)
)


def normalize_encoding(text: str) -> str:
    """Fix double-encoded UTF-8 (mojibake) on raw extracted PDF text.

    Uses ftfy when installed; falls back to a curated replacement map.
    Call this on raw text *before* any chunking or markdown cleaning so that
    downstream checks receive clean Unicode.
    """
    if _FTFY_AVAILABLE:
        return _ftfy.fix_text(text)
    cleaned = text
    for bad, good in _MOJIBAKE_MAP:
        cleaned = cleaned.replace(bad, good)
    # Strip ASCII control characters except tab (0x09) and newlines (0x0A, 0x0D).
    cleaned = re.sub(r"[\x01-\x08\x0b\x0c\x0e-\x1f\x7f]", "", cleaned)
    return cleaned


# ---------------------------------------------------------------------------
# Remaining utilities (unchanged from original)
# ---------------------------------------------------------------------------


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
    cleaned = re.sub(r"<\s*br\b\s*/?\s*>?", " ", cleaned, flags=re.I)
    cleaned = re.sub(r"<[^>]+>", " ", cleaned)
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
    cleaned = normalize_domain_terms(cleaned)
    cleaned = strip_non_content_tail_sections(cleaned)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    return normalize_whitespace(cleaned)


def clean_generated_text(text: str) -> str:
    """Clean extraction artifacts that leaked into generated questions or answers."""
    cleaned = text
    cleaned = cleaned.replace("Ã¢â‚¬â€", " - ").replace("Ã¢â‚¬â€œ", " - ")
    cleaned = cleaned.replace("Ã¢â‚¬Ëœ", "'").replace("Ã¢â‚¬â„¢", "'")
    cleaned = cleaned.replace('Ã¢â‚¬Å"', '"').replace("Ã¢â‚¬Â", '"')
    cleaned = re.sub(r"==\s*picture\s+\d+\s+x\s+\d+\s+intentionally omitted\s*<==", " ", cleaned, flags=re.I)
    cleaned = re.sub(r"<\s*br\b\s*/?\s*>?", " ", cleaned, flags=re.I)
    cleaned = re.sub(r"<[^>]+>", " ", cleaned)
    cleaned = re.sub(r"(\w)-\s+(\w)", r"\1\2", cleaned)
    cleaned = re.sub(r"(?<![A-Za-z/.\d-])\b\d{2,3}\b(?![A-Za-z/.\d%-])", " ", cleaned)
    cleaned = normalize_domain_terms(cleaned)
    cleaned = normalize_whitespace(cleaned)
    if cleaned:
        cleaned = cleaned[0].upper() + cleaned[1:]
    return cleaned


def normalize_domain_terms(text: str) -> str:
    """Repair common PDF word-join artifacts in finance/ML papers."""
    replacements = {
        r"\briskadjusted\b": "risk-adjusted",
        r"\briskreward\b": "risk-reward",
        r"\bgainsloss\b": "gains-loss",
        r"\bhighfrequency\b": "high-frequency",
        r"\boutofsample\b": "out-of-sample",
        r"\binsample\b": "in-sample",
        r"\bpricebased\b": "price-based",
        r"\btimeseries\b": "time-series",
    }
    cleaned = text
    for pattern, replacement in replacements.items():
        cleaned = re.sub(pattern, replacement, cleaned, flags=re.I)
    return cleaned


def strip_non_content_tail_sections(text: str) -> str:
    """Drop bibliography/code tail sections that should not seed QA generation."""
    headings = (
        "references",
        "bibliography",
        "code availability",
        "data availability",
        "acknowledgements",
        "acknowledgments",
    )
    pattern = r"(?im)^\s*(?:" + "|".join(re.escape(heading) for heading in headings) + r")\s*$"
    for match in re.finditer(pattern, text):
        if match.start() > len(text) * 0.2:
            return text[: match.start()]
    return text


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

    if "ñ" in sample or "¿" in sample or "¡" in sample:  # ñ ¿ ¡
        scores["es"] += 8
    if any(token in sample for token in (" l'", " d'", " qu'", "à", "è", "ç")):  # à è ç
        scores["fr"] += 4
        scores["ca"] += 2
    if any(token in sample for token in ("ção", "ões", "ã", "õ")):  # ção ões ã õ
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
