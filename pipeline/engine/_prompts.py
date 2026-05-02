"""LLM prompt builders and JSON payload parser."""

import json
import re
from collections.abc import Sequence
from typing import Any

from engine._config import Topic, logger
from engine._text import _sample_existing_questions, deduplicate_preserve_order, truncate_text


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


def build_representative_document_context(chunks: Sequence[Any], max_chars: int) -> str:
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
      "context_source": "one to three complete sentences copied verbatim from topic context"
    }}
  ]
}}

Rules:
- Questions must be answerable only from the returned context_source.
- Questions must cover different sub-points inside the same topic.
- Use varied question openers; do not start more than two questions with the same first word.
- Never repeat semantics from Existing dataset questions.
- Answers must be brief, natural, and strictly deducible from context_source.
- Answers should be 1-2 sentences and must not introduce terms or claims absent from context_source.
- For compare or inference questions, context_source must include the complete evidence needed for every claim in the answer.
- If context lacks enough information for a question, skip that question - do not hallucinate.
- Avoid asking about figures, images, or tables.
- Avoid questions about mathematical notation or formulas when the extracted text is visually degraded.
- Do not use context fragments that start mid-sentence or end mid-sentence.
- Do not hallucinate details not present in context.
- Do not include citations, markdown, or XML tags.
- context_source must be a literal substring from topic context, not a paraphrase.
- context_source must contain complete sentence(s), not a cut fragment.
- Answers must reformulate evidence in your own words. Do not copy context_source verbatim. The answer must be a concise factual synthesis, not a direct quote.

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


def _compact_topic_context(topic_context: str, max_chars: int) -> str:
    """Keep several retrieved chunks visible in compact fallback prompts."""
    if len(topic_context) <= max_chars:
        return topic_context

    marker_re = re.compile(r"(?=\[[^\]]+-chunk-\d{4}\])")
    blocks = [block.strip() for block in marker_re.split(topic_context) if block.strip()]
    if len(blocks) <= 1:
        return truncate_text(topic_context, max_chars)

    per_block_budget = max(800, max_chars // len(blocks))
    selected: list[str] = []
    total = 0
    for block in blocks:
        remaining = max_chars - total
        if remaining < 500:
            break
        excerpt = truncate_text(block, min(per_block_budget, remaining))
        selected.append(excerpt)
        total += len(excerpt) + 2
    return "\n\n".join(selected)


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
      "context_source": "one to three complete sentences copied verbatim from context"
    }}
  ]
}}

Constraints:
- Use only the context.
- No markdown.
- No prose outside JSON.
- Questions must be distinct.
- Use varied question openers.
- Prefer balanced types with at least one factual, conceptual and inference.
- Answers must be brief, natural, and strictly deducible from context_source.
- For compare or inference questions, include all supporting evidence in context_source.
- context_source must contain complete sentence(s), not a cut fragment.
- Avoid figures, tables, images, broken formulas, and degraded mathematical notation.
- Reformulate answers in your own words; do not copy context_source verbatim.

Document: {document}
Topic: {topic.name}
Context:
{_compact_topic_context(topic_context, 7000)}
""".strip()
    return [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]


def build_seed_question_messages(
    document: str,
    question: str,
    topic_context: str,
    language: str,
) -> list[dict[str, str]]:
    """Prompt to generate one answered item for a user-supplied seed question."""
    system_prompt = (
        "You generate high-quality supervised Q/A datasets from source text. "
        "Return strict JSON only. No markdown. No extra keys."
    )
    user_prompt = f"""
Answer the following question using only the provided context.
Language for the answer: {language}.
The question field must reproduce the exact question given — do not rephrase it.

Output format (strict JSON object):
{{
  "items": [
    {{
      "question": "the exact question reproduced verbatim",
      "answer": "string",
      "type": "factual|conceptual|inference|compare|definition",
      "difficulty": "easy|medium|hard",
      "context_source": "one to three complete sentences copied verbatim from context"
    }}
  ]
}}

Rules:
- answer must be brief, natural, and strictly deducible from context_source; do not hallucinate.
- context_source must be a literal substring from context, not a paraphrase.
- context_source must contain complete sentence(s), not a cut fragment.
- Avoid figures, tables, images, broken formulas, and degraded mathematical notation.
- If the context lacks enough information, answer from available text.
- Reformulate the answer in your own words; do not copy context_source verbatim.

Document: {document}
Question: {question}

Context:
\"\"\"
{truncate_text(topic_context, 7000)}
\"\"\"
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
