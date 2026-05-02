"""LLM-based audit judge for generated QA items."""

import json
from collections import Counter
from collections.abc import Sequence
from typing import Any

from engine._ollama import call_ollama_json
from engine._text import normalize_whitespace, truncate_text

VALID_JUDGE_DECISIONS: frozenset[str] = frozenset({"pass", "review", "fail"})
KNOWN_JUDGE_REASONS: frozenset[str] = frozenset({
    "factual",
    "unsupported_detail",
    "contradiction",
    "unclear_question",
    "weak_context",
    "extraction_artifact",
    "duplicate_like",
    "overly_literal",
    "judge_error",
})


def build_judge_messages(item: dict[str, Any]) -> list[dict[str, str]]:
    """Build a factuality-focused judge prompt for one QA item."""
    payload = {
        "question": item.get("question", ""),
        "answer": item.get("answer", ""),
        "context_source": item.get("context_source", ""),
        "context_excerpt": truncate_text(str(item.get("context_excerpt", "")), 1800),
        "topic": item.get("topic", ""),
        "type": item.get("type", ""),
        "difficulty": item.get("difficulty", ""),
        "document_language": item.get("document_language", ""),
    }
    return [
        {
            "role": "system",
            "content": (
                "You are a strict factuality judge for supervised QA datasets. "
                "Evaluate whether the answer is fully supported by the supplied source context. "
                "Return strict JSON only."
            ),
        },
        {
            "role": "user",
            "content": (
                "Judge this RAG QA triple. Prioritize factual support over style.\n"
                "Use these decisions:\n"
                "- pass: factual, clear, and usable.\n"
                "- review: probably useful but has minor issues or weak context.\n"
                "- fail: unsupported, contradictory, ambiguous, artifacted, or misleading.\n\n"
                "The context_source must be treated as the only allowed evidence. "
                "The answer must be factually deducible from that literal context, not from outside knowledge "
                "or broader paper context. Check for unsupported details, contradictions, unclear questions, "
                "weak or vague evidence, visible extraction artifacts, broken formulas, and overly literal "
                "copy-paste answers that do not actually answer the question.\n\n"
                "Return JSON object with keys:\n"
                "{"
                "\"score\": number from 0.0 to 1.0, "
                "\"decision\": \"pass\"|\"review\"|\"fail\", "
                "\"reasons\": [short reason strings], "
                "\"explanation\": short explanation"
                "}\n\n"
                f"QA item:\n{json.dumps(payload, ensure_ascii=False, indent=2)}"
            ),
        },
    ]


def _coerce_score(value: Any) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, round(score, 4)))


def _normalize_reason(value: Any) -> str:
    reason = normalize_whitespace(str(value)).lower().replace(" ", "_").replace("-", "_")
    if reason in KNOWN_JUDGE_REASONS:
        return reason
    if not reason:
        return "judge_error"
    if "supported" in reason and "not" not in reason and "un" not in reason:
        return "factual"
    if "unsupported" in reason or "not_supported" in reason or "halluc" in reason:
        return "unsupported_detail"
    if "contrad" in reason:
        return "contradiction"
    if "artifact" in reason or "formula" in reason or "symbol" in reason:
        return "extraction_artifact"
    if "context" in reason or "evidence" in reason:
        return "weak_context"
    if "unclear" in reason or "ambig" in reason:
        return "unclear_question"
    return reason[:40]


def normalize_judge_result(payload: dict[str, Any], model: str) -> dict[str, Any]:
    """Normalize arbitrary judge JSON to the public judged item fields."""
    decision = str(payload.get("decision", "")).strip().lower()
    score = _coerce_score(payload.get("score"))
    if decision not in VALID_JUDGE_DECISIONS:
        if score >= 0.8:
            decision = "pass"
        elif score >= 0.45:
            decision = "review"
        elif score > 0:
            decision = "fail"
        else:
            decision = "review"

    raw_reasons = payload.get("reasons", [])
    if isinstance(raw_reasons, str):
        raw_reasons = [raw_reasons]
    if not isinstance(raw_reasons, list):
        raw_reasons = []
    reasons = []
    for reason in raw_reasons:
        normalized = _normalize_reason(reason)
        if normalized not in reasons:
            reasons.append(normalized)
    if not reasons:
        reasons = ["factual"] if decision == "pass" else ["judge_error"]

    explanation = normalize_whitespace(str(payload.get("explanation", "")))
    if not explanation:
        explanation = "Judge did not provide an explanation."

    return {
        "judge_score": score,
        "judge_decision": decision,
        "judge_reasons": reasons[:5],
        "judge_explanation": truncate_text(explanation, 280),
        "judge_model": model,
    }


def judge_error_result(model: str, message: str = "Judge call failed or returned invalid JSON.") -> dict[str, Any]:
    """Return a non-destructive review result when judge evaluation fails."""
    return {
        "judge_score": 0.0,
        "judge_decision": "review",
        "judge_reasons": ["judge_error"],
        "judge_explanation": truncate_text(message, 280),
        "judge_model": model,
    }


def judge_item(
    item: dict[str, Any],
    model: str,
    temperature: float,
    seed: int | None = None,
) -> dict[str, Any]:
    """Return a copy of one item annotated with judge fields."""
    judged = dict(item)
    parsed, raw_content = call_ollama_json(
        model=model,
        messages=build_judge_messages(item),
        temperature=temperature,
        seed=seed,
    )
    if not parsed or parsed == {"items": []}:
        judged.update(judge_error_result(model))
        judged["judge_raw_content"] = raw_content
        return judged
    judged.update(normalize_judge_result(parsed, model))
    return judged


def build_judge_stats(mode: str, model: str, judged_items: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate judge decisions and reasons for dataset metadata."""
    decisions = Counter(str(item.get("judge_decision", "unknown")) for item in judged_items)
    reasons: Counter[str] = Counter()
    scores: list[float] = []
    for item in judged_items:
        scores.append(_coerce_score(item.get("judge_score")))
        for reason in item.get("judge_reasons", []):
            reasons[str(reason)] += 1

    return {
        "mode": mode,
        "model": model,
        "judged_items": len(judged_items),
        "decision_counts": dict(sorted(decisions.items())),
        "average_score": round(sum(scores) / len(scores), 4) if scores else 0.0,
        "reason_counts": dict(sorted(reasons.items())),
    }


def audit_items_with_judge(
    items: Sequence[dict[str, Any]],
    model: str,
    temperature: float,
    seed: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Annotate final accepted rows with judge scores without filtering them."""
    judged: list[dict[str, Any]] = []
    for idx, item in enumerate(items):
        item_seed = None if seed is None else seed + idx
        judged.append(judge_item(item=item, model=model, temperature=temperature, seed=item_seed))
    return judged, build_judge_stats("audit", model, judged)
