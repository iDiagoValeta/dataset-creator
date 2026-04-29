"""QA item generation for a single topic."""

import re
from collections.abc import Sequence
from typing import Any

from engine._config import VALID_DIFFICULTIES, VALID_TYPES, Topic, logger
from engine._ollama import call_ollama_json
from engine._prompts import (
    build_seed_question_messages,
    build_topic_generation_messages,
    build_topic_generation_messages_compact,
    truncate_text,
)
from engine._quality import (
    context_excerpt_for_fragment,
    find_verified_context_source,
    source_chunk_ids_for_fragment,
)
from engine._text import clean_generated_text, normalize_whitespace, now_iso


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
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Generate normalized Q/A items for one topic."""
    debug_attempts: list[dict[str, Any]] = []
    item_document_language = document_language or language

    # Strip leading sentence fragment caused by chunk boundary cuts.
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

            raw_source = clean_generated_text(str(item.get("context_source", "")).strip())
            # Strip chunk-id markup the model may have copied from the prompt
            raw_source = re.sub(r"^\[[\w.-]+-chunk-\d{4,}\]\s*", "", raw_source).strip()
            context_source, context_verified = find_verified_context_source(raw_source, answer, topic_context)
            if not context_source:
                context_source = normalize_whitespace(topic_context[:300])
            context_source = re.sub(r"^\[[\w.-]+-chunk-\d{4,}\]\s*", "", context_source).strip()
            source_chunk_ids = source_chunk_ids_for_fragment(topic_context, context_source)
            context_excerpt = context_excerpt_for_fragment(topic_context, context_source)

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
                    "source_chunk_ids": source_chunk_ids,
                    "created_at": now_iso(),
                    "context_excerpt": context_excerpt,
                }
            )
        return normalized_local

    # Attempt 1: full prompt (seed-question mode uses a fixed-question prompt).
    if seed_question_mode:
        messages = build_seed_question_messages(
            document=document,
            question=topic.name,
            topic_context=topic_context,
            language=language,
        )
    else:
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
