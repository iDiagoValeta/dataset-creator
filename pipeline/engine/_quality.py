"""Context verification, quality gate filtering, and item deduplication."""

import re
from collections.abc import Iterable, Sequence
from typing import Any

from engine._config import STOPWORDS
from engine._text import (
    normalize_whitespace,
    sanitize_question,
    strip_accents_ascii,
    truncate_text,
)

_CHUNK_MARKER_RE = re.compile(r"\[([^\]]+-chunk-\d{4})\]")

_DOMAIN_TERMS: dict[str, set[str]] = {
    "technical": {
        "technical", "analysis", "indicator", "indicators", "bollinger", "ema", "cci",
        "historical", "price-based", "trend", "trends", "sentiment", "trader", "traders",
        "emh", "efficient", "hypothesis", "noise", "mispricing", "mispricings",
    },
    "ml": {
        "machine", "learning", "neural", "network", "networks", "lstm", "transformer",
        "transformers", "gradient", "boosting", "reinforcement", "model", "models",
        "adaptive", "framework", "frameworks", "algorithm", "algorithms",
    },
    "hft": {
        "high-frequency", "frequency", "microstructure", "microstructures", "latency",
        "minute", "order", "flow", "trading", "out-of-sample", "in-sample",
        "buy-and-hold", "tail", "volatility", "risk", "profitability",
    },
}


def clean_context_artifacts(text: str) -> str:
    """Remove inline extraction artifacts while preserving readable evidence text."""
    cleaned = str(text)
    cleaned = re.sub(r"<\s*br\b\s*/?\s*>?", " ", cleaned, flags=re.I)
    cleaned = re.sub(r"<[^>]+>", " ", cleaned)
    cleaned = re.sub(r"==\s*picture\b.*?intentionally omitted\s*<==", " ", cleaned, flags=re.I)
    cleaned = re.sub(r"-----\s*Start of picture text\s*-----", " ", cleaned, flags=re.I)
    cleaned = re.sub(r"-----\s*End of picture text\s*-----", " ", cleaned, flags=re.I)
    cleaned = re.sub(r"(\w)-\s+(\w)", r"\1\2", cleaned)
    return normalize_whitespace(cleaned)


def _content_words(text: str) -> list[str]:
    """Return normalized content words for lightweight support checks."""
    return [
        word
        for word in re.findall(r"[A-Za-zÀ-ÿ]{4,}", strip_accents_ascii(text))
        if word not in STOPWORDS
    ]


def _normalized_terms(text: str) -> set[str]:
    """Return normalized content terms plus common hyphenated phrases."""
    normalized = strip_accents_ascii(text)
    words = {
        word
        for word in re.findall(r"[A-Za-z][A-Za-z-]{3,}", normalized)
        if word not in STOPWORDS
    }
    phrases = set(re.findall(r"[A-Za-z]+(?:-[A-Za-z]+)+", normalized))
    return words | phrases


def find_verified_context_source(raw_source: str, answer: str, topic_context: str) -> tuple[str, bool]:
    """Find a literal context fragment that supports an answer."""
    normalized_context = clean_context_artifacts(topic_context)
    source = _truncate_excerpt(clean_context_artifacts(raw_source), 300)
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
        return _truncate_excerpt(best_sentence, 300), True
    return "", False


def source_chunk_ids_for_fragment(topic_context: str, fragment: str) -> list[str]:
    """Infer chunk ids that contain or precede a source fragment."""
    normalized_context = clean_context_artifacts(topic_context)
    chunk_ids = _CHUNK_MARKER_RE.findall(normalized_context)
    if not fragment:
        return chunk_ids[:1]
    fragment_pos = normalized_context.lower().find(normalize_whitespace(fragment).lower()[:80])
    if fragment_pos < 0:
        return chunk_ids[:1]
    prefix = normalized_context[:fragment_pos]
    preceding = _CHUNK_MARKER_RE.findall(prefix)
    return preceding[-1:] if preceding else chunk_ids[:1]


def _truncate_excerpt(text: str, max_chars: int) -> str:
    """Trim an excerpt without ending in the middle of a word when possible."""
    excerpt = clean_context_artifacts(text)
    if len(excerpt) <= max_chars:
        return excerpt
    shortened = truncate_text(excerpt, max_chars).strip()
    if len(shortened) <= max_chars and not re.search(r"\w$", shortened):
        return shortened.rstrip(" ,;:")
    partial = excerpt[:max_chars]
    last_space = partial.rfind(" ")
    if last_space > int(max_chars * 0.6):
        return partial[:last_space].rstrip(" ,;:")
    return partial.rstrip(" ,;:")


def context_excerpt_for_fragment(topic_context: str, fragment: str, max_chars: int = 500) -> str:
    """Return an excerpt anchored to the chunk that contains a supporting fragment."""
    normalized_context = clean_context_artifacts(topic_context)
    if not normalized_context or max_chars <= 0:
        return ""

    normalized_fragment = clean_context_artifacts(fragment)
    fragment_pos = -1
    if normalized_fragment:
        fragment_pos = normalized_context.lower().find(normalized_fragment.lower()[:80])

    if fragment_pos < 0:
        return _truncate_excerpt(normalized_context, max_chars)

    marker_start = 0
    marker_end = 0
    for match in _CHUNK_MARKER_RE.finditer(normalized_context):
        if match.start() <= fragment_pos:
            marker_start = match.start()
            marker_end = match.end()
        else:
            break

    marker = normalized_context[marker_start:marker_end]
    chunk_body_start = marker_end
    if normalized_context[chunk_body_start : chunk_body_start + 1] == " ":
        chunk_body_start += 1

    body_budget = max_chars - len(marker) - 1
    if body_budget <= 0:
        return _truncate_excerpt(marker, max_chars)

    fragment_offset = max(0, fragment_pos - chunk_body_start)
    body = normalized_context[chunk_body_start:]
    body_start = 0
    if fragment_offset > max(80, body_budget // 4):
        body_start = max(0, fragment_offset - max(60, body_budget // 5))
        boundary = body.rfind(" ", 0, body_start)
        if boundary > 0:
            body_start = boundary + 1

    prefix = f"{marker} "
    if body_start > 0:
        prefix = f"{marker} ... "
        body_budget = max_chars - len(prefix)
    return prefix + _truncate_excerpt(body[body_start:], body_budget)


def has_quality_artifact(item: dict[str, Any]) -> bool:
    """Detect extraction artifacts that should not reach strict output."""
    text = " ".join(
        str(item.get(key, ""))
        for key in ("question", "answer", "context_source", "context_excerpt", "topic")
    ).lower()
    if "intentionally omitted" in text or "== picture" in text:
        return True
    if re.search(r"<\s*br\b|<[^>]+>", text):
        return True
    if "fuctuation" in text or "fuctuations" in text:
        return True
    if "text cuts off" in text or "[the text" in text:
        return True
    if "riskadjusted" in text:
        return True
    if re.search(r"\b(?:references|bibliography)\b", str(item.get("context_excerpt", "")).lower()):
        return True
    if "forthcoming" in str(item.get("context_source", "")).lower():
        return True
    if re.search(r"\b\d{2,3}\s+(computer|medical|process|memory|device|system)\b", text):
        return True
    if str(item.get("topic", "")).strip().lower().endswith("references"):
        return True
    return False


def has_truncated_context_source(item: dict[str, Any]) -> bool:
    """Detect source fragments that start inside a cut word."""
    source = normalize_whitespace(str(item.get("context_source", "")))
    if not source:
        return False
    first_words = source.split()[:2]
    if len(first_words) >= 2 and first_words[0].islower() and first_words[1][:1].isupper():
        return True
    return bool(re.match(r"^[a-z]{4,}\s+[A-Z][A-Za-z-]+", source))


def has_first_person_answer(item: dict[str, Any]) -> bool:
    """Return True when the answer is phrased from the paper authors' point of view."""
    answer = normalize_whitespace(str(item.get("answer", ""))).lower()
    return bool(re.match(r"^(?:we|our)\b", answer))


def _domain_scores(text: str) -> dict[str, int]:
    terms = _normalized_terms(text)
    return {domain: len(terms & domain_terms) for domain, domain_terms in _DOMAIN_TERMS.items()}


def _topic_domains(item: dict[str, Any]) -> set[str]:
    topic_text = " ".join(
        [
            str(item.get("topic", "")),
            " ".join(str(k) for k in item.get("topic_keywords", []) if k),
        ]
    )
    scores = _domain_scores(topic_text)
    return {domain for domain, score in scores.items() if score > 0}


def has_topic_mismatch(item: dict[str, Any]) -> bool:
    """Detect clear cases where an item belongs to another discovered topic."""
    topic_domains = _topic_domains(item)
    if not topic_domains:
        return False

    item_text = " ".join(
        str(item.get(key, ""))
        for key in ("question", "answer", "context_source")
    )
    item_scores = _domain_scores(item_text)
    current_score = max((item_scores[domain] for domain in topic_domains), default=0)
    other_scores = [score for domain, score in item_scores.items() if domain not in topic_domains]
    strongest_other = max(other_scores, default=0)

    if current_score == 0 and strongest_other >= 2:
        return True
    if current_score == 0 and strongest_other >= current_score + 3:
        return True

    return False


def has_insufficient_context_support(item: dict[str, Any]) -> bool:
    """Reject answers that add unsupported detail beyond the cited source."""
    item_type = str(item.get("type", "")).lower()
    if item_type not in {"compare", "inference"}:
        return False

    answer_terms = set(_content_words(str(item.get("answer", ""))))
    if len(answer_terms) < 6:
        return False
    source_terms = set(_content_words(str(item.get("context_source", ""))))
    if not source_terms:
        return True

    coverage = len(answer_terms & source_terms) / max(1, len(answer_terms))
    threshold = 0.5 if item_type == "compare" else 0.35
    return coverage <= threshold


def audit_item_quality(item: dict[str, Any]) -> str | None:
    """Return a rejection reason for deterministic item-level quality issues."""
    if has_quality_artifact(item):
        return "artifact_or_reference_noise"
    if has_truncated_context_source(item):
        return "truncated_context_source"
    if has_first_person_answer(item):
        return "first_person_answer"
    if has_insufficient_context_support(item):
        return "insufficient_context_support"
    if has_topic_mismatch(item):
        return "topic_mismatch"
    return None


def is_circular_answer(question: str, answer: str) -> bool:
    """True when the answer adds fewer than 3 new content tokens vs. the question."""
    q_words = set(_content_words(question))
    a_words = _content_words(answer)
    if not a_words:
        return True
    new_words = [w for w in a_words if w not in q_words]
    q_nums = set(re.findall(r"\d+(?:[.,]\d+)?%?", question))
    a_nums = set(re.findall(r"\d+(?:[.,]\d+)?%?", answer))
    new_nums = a_nums - q_nums
    return len(new_words) + len(new_nums) < 3


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


def _semantic_terms(text: str) -> set[str]:
    """Content-word set for coarse cross-topic deduplication."""
    return set(_content_words(text))


def deduplicate_items(items: Iterable[dict[str, Any]], semantic_threshold: float = 0.85) -> list[dict[str, Any]]:
    """Drop exact and near-semantic duplicated questions."""
    seen_exact = set()
    seen_answers = set()
    seen_bigrams: list[frozenset] = []
    seen_question_terms: list[set[str]] = []
    seen_qa_terms: list[set[str]] = []
    unique: list[dict[str, Any]] = []
    for item in items:
        question = str(item.get("question", ""))
        answer = str(item.get("answer", ""))
        signature = sanitize_question(question)
        answer_signature = sanitize_question(answer)
        if (
            not signature
            or signature in seen_exact
            or (len(answer_signature) >= 12 and answer_signature in seen_answers)
        ):
            continue

        q_bigrams = _question_bigrams(question)
        q_terms = _semantic_terms(question)
        qa_terms = _semantic_terms(f"{question} {answer}")
        duplicated_semantic = False
        if q_bigrams:
            for existing in seen_bigrams:
                if not existing:
                    continue
                overlap = len(q_bigrams & existing) / min(len(q_bigrams), len(existing))
                if overlap >= semantic_threshold:
                    duplicated_semantic = True
                    break
        if not duplicated_semantic and q_terms:
            for existing in seen_question_terms:
                if len(q_terms & existing) / max(1, min(len(q_terms), len(existing))) >= 0.75:
                    duplicated_semantic = True
                    break
        if not duplicated_semantic and qa_terms:
            for existing in seen_qa_terms:
                if len(qa_terms & existing) / max(1, min(len(qa_terms), len(existing))) >= 0.8:
                    duplicated_semantic = True
                    break
        if duplicated_semantic:
            continue

        seen_exact.add(signature)
        if answer_signature:
            seen_answers.add(answer_signature)
        seen_bigrams.append(q_bigrams)
        seen_question_terms.append(q_terms)
        seen_qa_terms.append(qa_terms)
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
        audit_reason = audit_item_quality(item)
        if audit_reason:
            reject(item, audit_reason)
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
