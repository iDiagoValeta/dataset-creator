"""Context verification, quality gate filtering, and item deduplication."""

import re
from collections.abc import Iterable, Sequence
from typing import Any

from engine._config import STOPWORDS
from engine._text import (
    MOJIBAKE_PATTERN,
    normalize_whitespace,
    sanitize_question,
    strip_accents_ascii,
    truncate_text,
)

_CHUNK_MARKER_RE = re.compile(r"\[([^\]]+-chunk-\d{4})\]")
_SENTENCE_END_RE = re.compile(r"[.!?](?:\s+|$)")
_BAD_CONTEXT_ENDINGS: frozenset[str] = frozenset({
    "a", "an", "and", "as", "at", "because", "by", "for", "from", "in", "into",
    "of", "offering", "on", "or", "that", "the", "to", "with",
})

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

_GENERIC_TOPIC_TERMS: frozenset[str] = frozenset({
    "approach", "approaches", "analysis", "applications", "based", "chapter",
    "concept", "concepts", "design", "document", "framework", "general",
    "implementation", "introduction", "method", "methods", "overview", "paper",
    "section", "study", "system", "systems", "topic", "topics", "using",
})


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


def find_verified_context_source(raw_source: str, answer: str, topic_context: str) -> tuple[str, bool]:  # noqa: ARG001
    """Return a context fragment only when it is a literal source substring."""
    normalized_context = normalize_whitespace(topic_context)
    source = normalize_whitespace(str(raw_source))
    if source and source.lower() in normalized_context.lower():
        start = normalized_context.lower().find(source.lower())
        end = start + len(source)
        return _expand_literal_to_sentence_window(normalized_context, start, end), True
    return "", False


def _expand_literal_to_sentence_window(context: str, start: int, end: int, max_chars: int = 700) -> str:
    """Expand a verified literal match to complete surrounding sentence boundaries."""
    left_floor = max(0, start - max_chars)
    left_region = context[left_floor:start]
    sentence_starts = [left_floor + m.end() for m in _SENTENCE_END_RE.finditer(left_region)]
    chunk_starts = [m.end() + 1 for m in _CHUNK_MARKER_RE.finditer(context[left_floor:start])]
    expanded_start = max(sentence_starts + chunk_starts, default=left_floor)

    right_ceiling = min(len(context), end + max_chars)
    right_region = context[end:right_ceiling]
    right_match = _SENTENCE_END_RE.search(right_region)
    expanded_end = end + right_match.end() if right_match else end

    if expanded_end - expanded_start > max_chars:
        expanded_start = start
        right_match = _SENTENCE_END_RE.search(context[end:min(len(context), start + max_chars)])
        expanded_end = end + right_match.end() if right_match else end
    return context[expanded_start:expanded_end].strip()


def fallback_context_source(topic_context: str, max_chars: int = 700) -> str:
    """Return a readable context fallback without arbitrary mid-sentence clipping."""
    normalized = normalize_whitespace(topic_context)
    normalized = re.sub(r"^\[[\w.-]+-chunk-\d{4,}\]\s*", "", normalized).strip()
    if not normalized or len(normalized) <= max_chars:
        return normalized

    for match in _SENTENCE_END_RE.finditer(normalized[:max_chars]):
        if match.end() >= 80:
            return normalized[:match.end()].strip()

    partial = normalized[:max_chars]
    last_space = partial.rfind(" ")
    if last_space > int(max_chars * 0.6):
        return partial[:last_space].rstrip(" ,;:")
    return partial.rstrip(" ,;:")


def source_chunk_ids_for_fragment(topic_context: str, fragment: str) -> list[str]:
    """Return all chunk IDs whose text range overlaps with the given source fragment.

    When a fragment spans multiple chunks the list contains every involved chunk ID
    in document order, making source_chunk_ids accurate for cross-chunk evidence.
    """
    normalized_context = clean_context_artifacts(topic_context)
    chunk_ids = _CHUNK_MARKER_RE.findall(normalized_context)
    if not fragment:
        return chunk_ids[:1]
    norm_fragment = normalize_whitespace(fragment)
    fragment_pos = normalized_context.lower().find(norm_fragment.lower()[:80])
    if fragment_pos < 0:
        return chunk_ids[:1]
    fragment_end = fragment_pos + len(norm_fragment)

    # Build a list of (marker_start, chunk_id) pairs from left to right.
    marker_positions: list[tuple[int, str]] = [
        (m.start(), m.group(1)) for m in _CHUNK_MARKER_RE.finditer(normalized_context)
    ]
    if not marker_positions:
        return chunk_ids[:1]

    # Each chunk "owns" the text from its marker up to the next marker (or end of context).
    involved: list[str] = []
    for i, (pos, chunk_id) in enumerate(marker_positions):
        next_pos = marker_positions[i + 1][0] if i + 1 < len(marker_positions) else len(normalized_context)
        # Include this chunk when its range overlaps [fragment_pos, fragment_end).
        if pos < fragment_end and next_pos > fragment_pos and chunk_id not in involved:
            involved.append(chunk_id)

    if not involved:
        prefix = normalized_context[:fragment_pos]
        preceding = _CHUNK_MARKER_RE.findall(prefix)
        return preceding[-1:] if preceding else chunk_ids[:1]
    return involved


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
    if re.search(r"(?:^|\s)(?:---\s*){2,}", str(item.get("context_source", ""))):
        return True
    if "forthcoming" in str(item.get("context_source", "")).lower():
        return True
    if re.search(r"\b\d{2,3}\s+(computer|medical|process|memory|device|system)\b", text):
        return True
    if "\ufffd" in text:
        return True
    if re.search(r"\(\s*dof\s*\)", text):
        return True
    if re.search(r"(?:\u2212|-|\?)\s*=\s*dof\b", text):
        return True
    if re.search(r"\bsection\s*\)", text):
        return True
    if re.search(r"\bfigure\s*,", text):
        return True
    if re.search(r"\bfig(?:ure)?\.?\s*:", text):
        return True
    if re.search(r"\b(?:as\s+)?(?:summari[sz]ed|shown|reported|displayed|illustrated)\s+in figure\s+\d+\b", text):
        return True
    if re.search(r"[a-z]\?{2,}", text):
        return True
    if re.search(r"\?\s*=\s*(?:dof|[a-z])\b", text):
        return True
    if str(item.get("topic", "")).strip().lower().endswith("references"):
        return True
    # Detect surviving mojibake in context_source or answer (safety net if normalize_encoding missed it).
    raw_text = " ".join(str(item.get(k, "")) for k in ("context_source", "answer", "question"))
    if MOJIBAKE_PATTERN.search(raw_text):
        return True
    # Detect chunk-boundary markers anywhere inside context_source.
    # A marker in the interior means the fragment crosses chunk boundaries.
    if re.search(r"\[[\w.-]+-chunk-\d{4,}\]", str(item.get("context_source", ""))):
        return True
    # Dangling section letters at the end ("... C.") indicate a section header
    # was glued to the extracted source. Trigger on a single-letter sentence
    # following at least one full sentence, which is the shape extraction
    # leaves behind when a heading is tacked on.
    for field in ("context_source", "context_excerpt"):
        text_field = normalize_whitespace(str(item.get(field, "")))
        if not text_field or len(text_field) < 20:
            continue
        if re.search(r"\b[A-Z]\.\s*$", text_field) and re.search(r"\.\s+[A-Z]\.\s*$", text_field):
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
    if re.match(r"^[a-z]{3,}\b", source):
        return True
    if len(source) > 120 and source[-1] not in ".;:!?)]":
        return True
    last_word_match = re.search(r"([A-Za-z]+)[^\w]*$", source.lower())
    if len(source) > 40 and last_word_match and last_word_match.group(1) in _BAD_CONTEXT_ENDINGS:
        return True
    return bool(re.match(r"^[a-z]{4,}\s+[A-Z][A-Za-z-]+", source))


def has_first_person_answer(item: dict[str, Any]) -> bool:
    """Return True when the answer is phrased from the paper authors' point of view."""
    answer = normalize_whitespace(str(item.get("answer", ""))).lower()
    return bool(re.match(r"^(?:we|our)\b", answer))


def _answer_bigrams(text: str) -> frozenset:
    """Word bigrams over tokens ≥3 chars for verbatim-copy detection."""
    words = re.findall(r"[A-Za-zÀ-ÿ0-9]{3,}", text.lower())
    if len(words) < 2:
        return frozenset(words)
    return frozenset(zip(words, words[1:], strict=False))


def has_verbatim_answer(item: dict[str, Any], threshold: float = 0.70) -> bool:
    """Return True when the answer is a near-verbatim copy of context_source.

    Uses bigram coverage of the answer over context_source. Short answers can
    still be over-extractive when they copy the exact evidence phrase.
    """
    answer = normalize_whitespace(str(item.get("answer", "")))
    context = normalize_whitespace(str(item.get("context_source", "")))
    answer_words = answer.split()
    context_words = context.split()
    if len(answer_words) < 6 or len(context_words) < 10:
        return False
    ab = _answer_bigrams(answer)
    cb = _answer_bigrams(context)
    if not ab:
        return False
    return len(ab & cb) / len(ab) >= threshold


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


def _topic_alignment_terms(item: dict[str, Any]) -> set[str]:
    topic_text = " ".join(
        [
            str(item.get("topic", "")),
            str(item.get("topic_summary", "")),
            " ".join(str(k) for k in item.get("topic_keywords", []) if k),
        ]
    )
    return {
        term
        for term in _normalized_terms(topic_text)
        if term not in _GENERIC_TOPIC_TERMS and len(term) >= 4
    }


def _topic_alignment_terms_from_topic(topic: dict[str, Any]) -> set[str]:
    """Like ``_topic_alignment_terms`` but accepts a Topic-shaped dict directly."""
    topic_text = " ".join(
        [
            str(topic.get("name", "")),
            str(topic.get("summary", "")),
            " ".join(str(k) for k in topic.get("keywords", []) if k),
        ]
    )
    return {
        term
        for term in _normalized_terms(topic_text)
        if term not in _GENERIC_TOPIC_TERMS and len(term) >= 4
    }


def score_item_against_topic(item: dict[str, Any], topic: dict[str, Any]) -> float:
    """Return a coarse alignment score between an item and a candidate topic.

    Combines term overlap (alignment terms ∩ item content terms), keyword
    overlap, and a small bonus when the topic name appears in the question
    or answer. The score is comparable across sibling topics of the same
    document but not normalized globally.
    """
    topic_terms = _topic_alignment_terms_from_topic(topic)
    if not topic_terms:
        return 0.0
    item_terms = _normalized_terms(
        " ".join(str(item.get(key, "")) for key in ("question", "answer", "context_source"))
    )
    if not item_terms:
        return 0.0

    overlap = topic_terms & item_terms
    score = float(len(overlap))

    keyword_terms = {
        str(k).strip().lower()
        for k in topic.get("keywords", [])
        if str(k).strip()
    }
    if keyword_terms:
        item_text = " ".join(
            str(item.get(key, "")) for key in ("question", "answer", "context_source")
        ).lower()
        kw_hits = sum(1 for kw in keyword_terms if kw and kw in item_text)
        score += 0.5 * kw_hits

    name = str(topic.get("name", "")).strip().lower()
    if name and len(name) >= 5:
        qa_text = " ".join(
            str(item.get(key, "")) for key in ("question", "answer")
        ).lower()
        if name in qa_text:
            score += 1.0
    return score


def reassign_or_reject_topic(
    item: dict[str, Any],
    candidate_topics: Sequence[dict[str, Any]],
    margin: float = 1.5,
    min_match_score: float = 1.0,
) -> tuple[dict[str, Any], str | None]:
    """Move an item to a better-fitting topic from the same document, or reject.

    Returns a tuple ``(possibly_updated_item, action)`` where ``action`` is one
    of: ``None`` (item kept as-is), ``"reassigned"`` (item moved to a stronger
    topic), or ``"topic_mismatch_reassign"`` (no candidate had enough overlap).

    ``margin`` is the minimum score advantage that another topic needs over the
    current one to trigger a reassignment. ``min_match_score`` is the floor for
    the best candidate; below it the item is rejected.
    """
    if not candidate_topics:
        return item, None

    current_topic_id = str(item.get("topic_id", ""))
    scored: list[tuple[float, dict[str, Any]]] = [
        (score_item_against_topic(item, topic), topic) for topic in candidate_topics
    ]
    if not scored:
        return item, None

    scored.sort(key=lambda pair: pair[0], reverse=True)
    best_score, best_topic = scored[0]

    if best_score < min_match_score:
        return item, "topic_mismatch_reassign"

    current_score = next(
        (score for score, topic in scored if str(topic.get("topic_id", "")) == current_topic_id),
        None,
    )

    if current_score is not None and (best_score - current_score) < margin:
        return item, None

    if str(best_topic.get("topic_id", "")) == current_topic_id:
        return item, None

    updated = dict(item)
    updated["reassigned_from"] = {
        "topic_id": current_topic_id,
        "topic": str(item.get("topic", "")),
    }
    updated["topic_id"] = str(best_topic.get("topic_id", ""))
    updated["topic"] = str(best_topic.get("name", ""))
    updated["topic_summary"] = str(best_topic.get("summary", ""))
    updated["topic_keywords"] = list(best_topic.get("keywords", []))
    return updated, "reassigned"


def apply_topic_reassignment(
    items: Sequence[dict[str, Any]],
    topics_by_document: dict[str, Sequence[dict[str, Any]]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    """Apply ``reassign_or_reject_topic`` over a stream of items.

    Returns ``(kept_items, rejected_items, stats)``. Items rejected here carry
    ``rejection_reason="topic_mismatch_reassign"``; the dedicated reason makes
    it visible in ``quality.rejection_reasons`` separately from the
    deterministic ``topic_mismatch`` rule.
    """
    kept: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    stats = {"reassigned": 0, "rejected_no_match": 0, "kept_as_is": 0}
    for item in items:
        document = str(item.get("document", ""))
        candidates = list(topics_by_document.get(document, []))
        updated, action = reassign_or_reject_topic(item, candidates)
        if action == "topic_mismatch_reassign":
            row = dict(updated)
            row["rejection_reason"] = "topic_mismatch_reassign"
            rejected.append(row)
            stats["rejected_no_match"] += 1
            continue
        if action == "reassigned":
            stats["reassigned"] += 1
        else:
            stats["kept_as_is"] += 1
        kept.append(updated)
    return kept, rejected, stats


def has_topic_mismatch(item: dict[str, Any]) -> bool:
    """Detect clear cases where an item belongs to another discovered topic."""
    topic_terms = _topic_alignment_terms(item)
    has_keywords = bool(item.get("topic_keywords"))
    if len(topic_terms) >= 2 and (has_keywords or len(topic_terms) >= 4):
        item_terms = _normalized_terms(
            " ".join(str(item.get(key, "")) for key in ("question", "answer", "context_source"))
        )
        shared = topic_terms & item_terms
        if not shared:
            return True

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
    if item_type not in {"compare", "definition", "factual", "inference"}:
        return False

    answer_terms = set(_content_words(str(item.get("answer", ""))))
    if len(answer_terms) < 6:
        return False
    source_terms = set(_content_words(str(item.get("context_source", ""))))
    if not source_terms:
        return True

    coverage = len(answer_terms & source_terms) / max(1, len(answer_terms))
    if item_type == "inference":
        threshold = 0.35
        max_new_terms = 7
    elif item_type == "compare":
        threshold = 0.58
        max_new_terms = 5
    else:
        threshold = 0.68
        max_new_terms = 4
    new_terms = answer_terms - source_terms
    return coverage <= threshold or len(new_terms) >= max_new_terms


def audit_item_quality(item: dict[str, Any]) -> str | None:
    """Return a rejection reason for deterministic item-level quality issues."""
    # Cross-chunk marker: structurally invalid — check before artifact detection.
    if re.search(r"\[[\w.-]+-chunk-\d{4,}\]", str(item.get("context_source", ""))):
        return "cross_chunk_context"
    if has_quality_artifact(item):
        return "artifact_or_reference_noise"
    if has_truncated_context_source(item):
        return "truncated_context_source"
    if has_first_person_answer(item):
        return "first_person_answer"
    if has_topic_mismatch(item):
        return "topic_mismatch"
    # Verbatim check before lexical support so copy-paste defects are visible.
    if has_verbatim_answer(item):
        return "verbatim_answer"
    if has_insufficient_context_support(item):
        return "insufficient_context_support"
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


def _answer_terms(text: str) -> set[str]:
    """Content words used to catch duplicated answers with small punctuation drift."""
    return set(_content_words(text))


def deduplicate_items(
    items: Iterable[dict[str, Any]],
    semantic_threshold: float = 0.78,
    return_stats: bool = False,
) -> list[dict[str, Any]] | tuple[list[dict[str, Any]], dict[str, int]]:
    """Drop exact and near-semantic duplicated questions.

    When ``return_stats=True`` returns ``(unique, breakdown)`` where
    ``breakdown`` distinguishes ``duplicate_exact``,
    ``duplicate_semantic_question``, ``duplicate_semantic_qa`` and
    ``duplicate_semantic_answer`` counts. The exact-question and
    duplicate-answer-string checks are reported under
    ``duplicate_exact`` and ``duplicate_semantic_answer`` respectively
    so the totals match the historical aggregate.
    """
    seen_exact = set()
    seen_answers = set()
    seen_bigrams: list[frozenset] = []
    seen_question_terms: list[set[str]] = []
    seen_qa_terms: list[set[str]] = []
    seen_answer_terms: list[set[str]] = []
    unique: list[dict[str, Any]] = []
    breakdown = {
        "duplicate_exact": 0,
        "duplicate_semantic_question": 0,
        "duplicate_semantic_qa": 0,
        "duplicate_semantic_answer": 0,
    }
    for item in items:
        question = str(item.get("question", ""))
        answer = str(item.get("answer", ""))
        signature = sanitize_question(question)
        answer_signature = sanitize_question(answer)
        if not signature:
            breakdown["duplicate_exact"] += 1
            continue
        if signature in seen_exact:
            breakdown["duplicate_exact"] += 1
            continue
        if len(answer_signature) >= 12 and answer_signature in seen_answers:
            breakdown["duplicate_semantic_answer"] += 1
            continue

        q_bigrams = _question_bigrams(question)
        q_terms = _semantic_terms(question)
        qa_terms = _semantic_terms(f"{question} {answer}")
        answer_terms = _answer_terms(answer)
        duplicated_reason: str | None = None
        if q_bigrams:
            for existing in seen_bigrams:
                if not existing:
                    continue
                overlap = len(q_bigrams & existing) / min(len(q_bigrams), len(existing))
                if overlap >= semantic_threshold:
                    duplicated_reason = "duplicate_semantic_question"
                    break
        if duplicated_reason is None and q_terms:
            for existing in seen_question_terms:
                if len(q_terms & existing) / max(1, min(len(q_terms), len(existing))) >= 0.75:
                    duplicated_reason = "duplicate_semantic_question"
                    break
        if duplicated_reason is None and qa_terms:
            for existing in seen_qa_terms:
                if len(qa_terms & existing) / max(1, min(len(qa_terms), len(existing))) >= 0.8:
                    duplicated_reason = "duplicate_semantic_qa"
                    break
        if duplicated_reason is None and answer_terms:
            for existing in seen_answer_terms:
                if len(answer_terms & existing) / max(1, min(len(answer_terms), len(existing))) >= 0.78:
                    duplicated_reason = "duplicate_semantic_answer"
                    break
        if duplicated_reason is not None:
            breakdown[duplicated_reason] += 1
            continue

        seen_exact.add(signature)
        if answer_signature:
            seen_answers.add(answer_signature)
        seen_bigrams.append(q_bigrams)
        seen_question_terms.append(q_terms)
        seen_qa_terms.append(qa_terms)
        seen_answer_terms.append(answer_terms)
        unique.append(item)
    if return_stats:
        return unique, breakdown
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
