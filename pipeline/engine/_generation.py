"""Evidence-first QA item generation for a single topic.

MODULE MAP:
- Paper-voice normalization and response parsing helpers.
- Evidence-bound item normalization.
- Generation-time quality/repair loop.
- ``generate_items_for_topic`` public entry point.
"""

import re
from collections.abc import Sequence
from typing import Any

from engine._config import VALID_DIFFICULTIES, VALID_TYPES, Topic, logger
from engine._evidence import EvidenceWindow, collect_evidence_windows
from engine._ollama import call_ollama_json
from engine._prompts import (
    build_evidence_generation_messages,
    build_evidence_seed_question_messages,
)
from engine._quality import (
    audit_item_quality,
    context_excerpt_for_fragment,
    is_circular_answer,
    source_chunk_ids_for_fragment,
)
from engine._text import clean_generated_text, normalize_whitespace, now_iso, sanitize_question


def _normalize_paper_voice(answer: str) -> str:
    """Avoid first-person paper phrasing in dataset answers."""
    possessive_replacements = {
        "Our findings ": "The paper's findings ",
        "Our results ": "The paper's results ",
        "Our study ": "The study ",
        "Our paper ": "The paper ",
    }
    for prefix, replacement in possessive_replacements.items():
        if answer.startswith(prefix):
            return f"{replacement}{answer[len(prefix):]}"

    replacements = {
        "assess": "assesses",
        "analyze": "analyzes",
        "analyse": "analyses",
        "evaluate": "evaluates",
        "examine": "examines",
        "find": "finds",
        "show": "shows",
        "propose": "proposes",
        "present": "presents",
    }
    for verb, replacement in replacements.items():
        prefix = f"We {verb} "
        if answer.startswith(prefix):
            return f"The paper {replacement} {answer[len(prefix):]}"
    return answer


def _extract_raw_items(parsed_obj: dict[str, Any]) -> list[Any]:
    """Return item-shaped payloads from common JSON response shapes."""
    raw_local: Any = parsed_obj.get("items", [])
    if isinstance(raw_local, dict):
        raw_local = [raw_local]
    if not isinstance(raw_local, list):
        for alt_key in ("item", "questions", "preguntas", "data", "results"):
            alt_value = parsed_obj.get(alt_key)
            if isinstance(alt_value, dict):
                raw_local = [alt_value]
                break
            if isinstance(alt_value, list):
                raw_local = alt_value
                break
    return raw_local if isinstance(raw_local, list) else []


def _build_item_from_evidence(
    raw_item: dict[str, Any],
    *,
    document: str,
    topic: Topic,
    topic_context: str,
    evidence: EvidenceWindow,
    idx: int,
    id_offset: int,
    document_language: str,
    seed_question: str | None = None,
) -> dict[str, Any] | None:
    """Normalize one raw model item while keeping evidence literal and verified."""
    raw_question = seed_question if seed_question is not None else raw_item.get("question", "")
    question = clean_generated_text(str(raw_question).strip())
    answer = _normalize_paper_voice(clean_generated_text(str(raw_item.get("answer", "")).strip()))
    if not question or not answer:
        return None

    raw_type = str(raw_item.get("type", "factual")).strip().lower()
    item_type = raw_type if raw_type in VALID_TYPES else "factual"
    raw_difficulty = str(raw_item.get("difficulty", "medium")).strip().lower()
    difficulty = raw_difficulty if raw_difficulty in VALID_DIFFICULTIES else "medium"

    normalized_context = normalize_whitespace(topic_context)
    context_source = normalize_whitespace(evidence.text)
    if not context_source or context_source.lower() not in normalized_context.lower():
        return None
    context_source = re.sub(r"\[[\w.-]+-chunk-\d{4,}\]\s*", "", context_source).strip()
    source_chunk_ids = [evidence.chunk_id] if evidence.chunk_id else source_chunk_ids_for_fragment(
        topic_context,
        context_source,
    )
    context_excerpt = context_excerpt_for_fragment(topic_context, context_source)

    return {
        "id": f"{document}-{topic.topic_id}-qa-{idx + id_offset:02d}",
        "question": question,
        "answer": answer,
        "type": item_type,
        "difficulty": difficulty,
        "context_source": context_source,
        "context_source_verified": True,
        "topic": topic.name,
        "topic_id": topic.topic_id,
        "topic_keywords": topic.keywords,
        "document": document,
        "document_language": document_language,
        "source_chunk_ids": source_chunk_ids,
        "created_at": now_iso(),
        "context_excerpt": context_excerpt,
    }


def _candidate_rejection_reason(
    item: dict[str, Any],
    existing_question_signatures: set[str],
) -> str | None:
    """Return a generation-time rejection reason, mirroring the strict quality gate."""
    signature = sanitize_question(str(item.get("question", "")))
    if signature and signature in existing_question_signatures:
        return "duplicate_existing_question"
    audit_reason = audit_item_quality(item)
    if audit_reason:
        return audit_reason
    if is_circular_answer(str(item.get("question", "")), str(item.get("answer", ""))):
        return "circular_answer"
    if not item.get("context_source_verified"):
        return "unverified_context_source"
    return None


def _repair_feedback(reason: str) -> str:
    """Create concise feedback for a repair attempt."""
    if reason == "verbatim_answer":
        return "Rewrite the answer in your own words; do not copy the evidence wording."
    if reason == "insufficient_context_support":
        return "Remove every term or claim that is not explicitly supported by the evidence."
    if reason == "circular_answer":
        return "Make the answer add concrete information from the evidence beyond the question wording."
    if reason == "duplicate_existing_question":
        return "Ask about a different fact or concept from the same evidence."
    if reason == "first_person_answer":
        return "Do not answer from the paper authors' first-person perspective."
    return "Create a narrower factual Q/A pair that is fully supported by the evidence."


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
    seed_question_mode: bool = False,
    id_offset: int = 0,
    all_topics: Sequence[Topic] | None = None,
    used_context_sources: set[str] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Generate normalized Q/A items from deterministic literal evidence windows."""
    debug_attempts: list[dict[str, Any]] = []
    item_document_language = document_language or language

    # Strip leading sentence fragment caused by chunk boundary cuts.
    topic_context = re.sub(
        r"^(\[[\w.-]+-chunk-\d{4,}\]\s+)[a-z]{1,3}\s[^\n.!?]*[.!?]\s*",
        r"\1",
        topic_context,
    ).strip()

    evidence_windows, evidence_stats = collect_evidence_windows(
        topic_context=topic_context,
        topic=topic,
        all_topics=all_topics,
        excluded_texts=used_context_sources,
    )
    stats = {
        "candidate_windows": evidence_stats["candidate_windows"],
        "attempted_windows": 0,
        "accepted_from_evidence": 0,
        "repair_attempts": 0,
        "discarded_windows": evidence_stats["discarded_windows"],
        "evidence_exhausted_topics": 0,
    }
    raw_items_debug: list[Any] = []
    normalized: list[dict[str, Any]] = []
    existing_signatures = {sanitize_question(q) for q in existing_questions if str(q).strip()}
    used_sources = {normalize_whitespace(text).lower() for text in (used_context_sources or set()) if text}
    max_repairs = 2
    max_attempted_windows = min(len(evidence_windows), max(questions_per_topic * 5, 16))

    for evidence in evidence_windows[:max_attempted_windows]:
        if len(normalized) >= questions_per_topic:
            break
        evidence_key = normalize_whitespace(evidence.text).lower()
        if evidence_key in used_sources:
            continue
        stats["attempted_windows"] += 1
        accepted_for_window = False
        repair_feedback = ""
        last_reason = "not_attempted"

        for attempt_idx in range(max_repairs + 1):
            if seed_question_mode:
                messages = build_evidence_seed_question_messages(
                    document=document,
                    question=topic.name,
                    evidence=evidence.text,
                    language=language,
                    repair_feedback=repair_feedback,
                )
            else:
                messages = build_evidence_generation_messages(
                    document=document,
                    topic=topic,
                    evidence=evidence.text,
                    language=language,
                    existing_questions=[
                        *existing_questions,
                        *[item["question"] for item in normalized],
                    ],
                    repair_feedback=repair_feedback,
                )

            parsed, raw_content = call_ollama_json(
                model=model,
                messages=messages,
                temperature=temperature,
                seed=None if seed is None else seed + stats["attempted_windows"] + attempt_idx,
            )
            raw_items = _extract_raw_items(parsed)
            raw_items_debug.extend(raw_items)
            raw_item = next((item for item in raw_items if isinstance(item, dict)), None)
            candidate = (
                _build_item_from_evidence(
                    raw_item,
                    document=document,
                    topic=topic,
                    topic_context=topic_context,
                    evidence=evidence,
                    idx=len(normalized),
                    id_offset=id_offset,
                    document_language=item_document_language,
                    seed_question=topic.name if seed_question_mode else None,
                )
                if raw_item
                else None
            )
            last_reason = "parse_error" if candidate is None else (
                _candidate_rejection_reason(candidate, existing_signatures) or ""
            )
            debug_attempts.append(
                {
                    "attempt": len(debug_attempts) + 1,
                    "evidence_id": evidence.evidence_id,
                    "chunk_id": evidence.chunk_id,
                    "evidence_score": evidence.score,
                    "repair_attempt": attempt_idx,
                    "rejection_reason": last_reason or None,
                    "raw_content": raw_content,
                    "parsed": parsed,
                    "raw_items_count": len(raw_items),
                }
            )

            if candidate is not None and not last_reason:
                normalized.append(candidate)
                existing_signatures.add(sanitize_question(str(candidate.get("question", ""))))
                used_sources.add(evidence_key)
                stats["accepted_from_evidence"] += 1
                accepted_for_window = True
                print(
                    f"      evidencia {stats['attempted_windows']}/{max_attempted_windows}: "
                    f"{len(normalized)}/{questions_per_topic} aceptados",
                    flush=True,
                )
                break

            if attempt_idx < max_repairs:
                stats["repair_attempts"] += 1
                repair_feedback = _repair_feedback(last_reason)

        if not accepted_for_window:
            stats["discarded_windows"] += 1
            if stats["attempted_windows"] % 5 == 0:
                print(
                    f"      evidencia {stats['attempted_windows']}/{max_attempted_windows}: "
                    f"{len(normalized)}/{questions_per_topic} aceptados "
                    f"(ultimo rechazo: {last_reason})",
                    flush=True,
                )

    if len(normalized) < questions_per_topic and stats["attempted_windows"] >= max_attempted_windows:
        stats["evidence_exhausted_topics"] = 1

    if not normalized:
        last_raw = debug_attempts[-1]["raw_content"] if debug_attempts else ""
        logger.warning(
            "No se generaron items evidence-first para '%s' / '%s'. Raw (300 chars): %s",
            document,
            topic.name,
            last_raw[:300] if last_raw else "(vacio)",
        )
    return normalized, {
        "raw_items": raw_items_debug,
        "attempts": debug_attempts,
        "topic": topic.name,
        "evidence_first": stats,
        "attempted_context_sources": [
            evidence.text
            for evidence in evidence_windows
            if any(attempt.get("evidence_id") == evidence.evidence_id for attempt in debug_attempts)
        ],
        "evidence_windows": [
            {
                "evidence_id": evidence.evidence_id,
                "chunk_id": evidence.chunk_id,
                "score": evidence.score,
                "text": evidence.text,
            }
            for evidence in evidence_windows[:20]
        ],
    }
