"""Literal evidence extraction and ranking for evidence-first QA generation.

MODULE MAP:
- ``EvidenceWindow``: immutable evidence record used by generation.
- Topic-context parsing: split marker-prefixed context into chunk bodies.
- Sentence windows: build 1-3 sentence literal evidence candidates.
- Filtering/ranking: remove noisy windows and rank by topic affinity.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from engine._config import Chunk, Topic
from engine._quality import has_quality_artifact, has_truncated_context_source
from engine._text import MOJIBAKE_PATTERN, normalize_whitespace, sanitize_question
from engine._topics import score_chunk_for_topic

_CHUNK_MARKER_RE = re.compile(r"\[([^\]]+-chunk-\d{4,})\]")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[\"'(\[]?[A-Z0-9])")
_ALPHA_WORD_RE = re.compile(r"[A-Za-zÀ-ÿ]{3,}")


@dataclass(frozen=True)
class EvidenceWindow:
    """A complete, literal evidence window from one source chunk."""

    evidence_id: str
    text: str
    chunk_id: str | None
    chunk_index: int
    sentence_start: int
    sentence_count: int
    score: float


def _chunk_blocks(topic_context: str) -> list[tuple[str | None, str]]:
    """Return ``(chunk_id, body)`` blocks from marker-prefixed topic context."""
    context = normalize_whitespace(topic_context)
    if not context:
        return []

    markers = list(_CHUNK_MARKER_RE.finditer(context))
    if not markers:
        return [(None, context)]

    blocks: list[tuple[str | None, str]] = []
    for idx, marker in enumerate(markers):
        next_start = markers[idx + 1].start() if idx + 1 < len(markers) else len(context)
        body = normalize_whitespace(context[marker.end():next_start])
        if body:
            blocks.append((marker.group(1), body))
    return blocks


def _sentences(text: str) -> list[str]:
    """Split normalized text into sentence-like complete units."""
    compact = normalize_whitespace(text).replace("\n", " ")
    if not compact:
        return []
    parts = [part.strip() for part in _SENTENCE_SPLIT_RE.split(compact) if part.strip()]
    return [part for part in parts if part[-1:] in {".", "!", "?", ")"}]


def _has_reference_or_layout_noise(text: str) -> bool:
    """Detect evidence windows dominated by references, figures, or layout noise."""
    lowered = text.lower()
    if re.search(r"\b(?:references|bibliography|acknowledg(?:e)?ments)\b", lowered):
        return True
    if re.search(r"\b(?:fig|figure|table)\.?\s*\d+\b", lowered):
        return True
    if re.search(r"\b(?:isbn|doi|arxiv|copyright|permission from)\b", lowered):
        return True
    if re.search(r"(?:^|\s)(?:---\s*){2,}", text):
        return True
    if len(re.findall(r"\b[\w.-]+@[\w.-]+\b", text)) >= 1:
        return True
    alpha_chars = sum(ch.isalpha() for ch in text)
    return len(text) >= 120 and alpha_chars / max(1, len(text)) < 0.45


def _is_usable_evidence(text: str, topic: Topic, min_words: int = 10, min_chars: int = 60) -> bool:
    """Return True when a window is suitable as a literal RAG context_source."""
    source = normalize_whitespace(text)
    if len(source) < min_chars:
        return False
    if len(_ALPHA_WORD_RE.findall(source)) < min_words:
        return False
    if MOJIBAKE_PATTERN.search(source):
        return False
    if _CHUNK_MARKER_RE.search(source):
        return False
    if _has_reference_or_layout_noise(source):
        return False
    if has_truncated_context_source({"context_source": source}):
        return False
    if has_quality_artifact(
        {
            "question": "placeholder question",
            "answer": "placeholder answer",
            "context_source": source,
            "context_excerpt": source,
            "topic": topic.name,
        }
    ):
        return False
    return True


def _window_score(
    text: str,
    topic: Topic,
    all_topics: Sequence[Topic] | None,
    chunk_id: str | None,
) -> float:
    """Score a candidate evidence window against its topic."""
    chunk = Chunk(chunk_id=chunk_id or "__evidence__", document="", text=text)
    lexical = float(score_chunk_for_topic(chunk, topic, all_topics=all_topics))
    topic_terms = {
        sanitize_question(term)
        for term in [topic.name, topic.summary, *topic.keywords]
        if str(term).strip()
    }
    lowered = sanitize_question(text)
    direct_hits = sum(1 for term in topic_terms if term and term in lowered)
    return lexical + (0.25 * direct_hits)


def collect_evidence_windows(
    topic_context: str,
    topic: Topic,
    all_topics: Sequence[Topic] | None = None,
    excluded_texts: set[str] | None = None,
    max_windows: int = 80,
) -> tuple[list[EvidenceWindow], dict[str, int]]:
    """Extract ranked, literal evidence windows plus collection stats."""
    excluded = {normalize_whitespace(text).lower() for text in (excluded_texts or set()) if text}
    windows: list[EvidenceWindow] = []
    discarded = 0
    raw_windows = 0

    for chunk_index, (chunk_id, body) in enumerate(_chunk_blocks(topic_context)):
        sent_list = _sentences(body)
        for start in range(len(sent_list)):
            for count in (1, 2, 3):
                selected = sent_list[start:start + count]
                if len(selected) != count:
                    continue
                raw_windows += 1
                text = normalize_whitespace(" ".join(selected))
                if normalize_whitespace(text).lower() in excluded:
                    discarded += 1
                    continue
                if not _is_usable_evidence(text, topic):
                    discarded += 1
                    continue
                score = _window_score(text, topic, all_topics, chunk_id)
                evidence_id = f"{chunk_id or 'context'}-sent-{start:04d}-{count}"
                windows.append(
                    EvidenceWindow(
                        evidence_id=evidence_id,
                        text=text,
                        chunk_id=chunk_id,
                        chunk_index=chunk_index,
                        sentence_start=start,
                        sentence_count=count,
                        score=score,
                    )
                )

    windows.sort(
        key=lambda window: (
            -window.score,
            window.chunk_index,
            window.sentence_start,
            window.sentence_count,
        )
    )
    if max_windows > 0:
        windows = windows[:max_windows]

    return windows, {
        "raw_windows": raw_windows,
        "candidate_windows": len(windows),
        "discarded_windows": discarded,
    }
