"""LLM-based audit judge for generated QA items."""

import json
import re
from collections import Counter
from collections.abc import Sequence
from typing import Any

from engine._ollama import call_ollama_json
from engine._text import MOJIBAKE_PATTERN, normalize_whitespace, truncate_text

VALID_JUDGE_DECISIONS: frozenset[str] = frozenset({"pass", "review", "fail"})
KNOWN_JUDGE_REASONS: frozenset[str] = frozenset({
    "cross_chunk_context",
    "factual",
    "extraction_artifact",
    "judge_error",
    "overly_extractive",
    "truncated_context",
    "unclear_question",
    "unsupported_detail",
    "weak_context",
})
# Any of these reasons forces the item to fail regardless of LLM scores.
BLOCKING_REASONS: frozenset[str] = frozenset({
    "cross_chunk_context",
    "extraction_artifact",
    "overly_extractive",
    "truncated_context",
    "unsupported_detail",
    "weak_context",
})

_CHUNK_MARKER_RE = re.compile(r"\[[\w.-]+-chunk-\d{4,}\]")


def _is_verbatim_answer(answer: str, context_source: str, threshold: float = 0.80) -> bool:
    """Bigram-Jaccard verbatim check used by the judge pre-checks."""
    aw = re.findall(r"[A-Za-zÀ-ÿ0-9]{3,}", answer.lower())
    cw = re.findall(r"[A-Za-zÀ-ÿ0-9]{3,}", context_source.lower())
    if len(aw) < 6 or len(cw) < 8:
        return False
    ab = frozenset(zip(aw, aw[1:], strict=False))
    cb = frozenset(zip(cw, cw[1:], strict=False))
    if not ab:
        return False
    return len(ab & cb) / len(ab) >= threshold


def _deterministic_judge_prechecks(item: dict[str, Any]) -> dict[str, Any] | None:
    """Run fast deterministic checks before calling the LLM judge.

    Returns a raw judge payload dict when a defect is found, or None to proceed
    with the LLM.  Using pre-checks keeps token usage low for clear-cut failures.
    """
    context_source = str(item.get("context_source", ""))
    answer = str(item.get("answer", ""))

    # Check 1: mojibake in context_source or answer.
    if MOJIBAKE_PATTERN.search(f"{context_source} {answer}"):
        return {
            "context_quality": 0.1,
            "answer_support": 0.5,
            "question_quality": 0.5,
            "overall_score": 0.1,
            "decision": "fail",
            "reasons": ["extraction_artifact"],
            "explanation": "Mojibake or encoding artifact detected in context_source or answer.",
        }

    # Check 2: chunk-boundary marker inside context_source.
    if _CHUNK_MARKER_RE.search(context_source):
        return {
            "context_quality": 0.1,
            "answer_support": 0.5,
            "question_quality": 0.5,
            "overall_score": 0.1,
            "decision": "fail",
            "reasons": ["extraction_artifact"],
            "explanation": "Chunk boundary marker found inside context_source.",
        }

    # Check 3: answer is a near-verbatim copy of context_source.
    if _is_verbatim_answer(answer, context_source, threshold=0.80):
        return {
            "context_quality": 0.8,
            "answer_support": 0.9,
            "question_quality": 0.4,
            "overall_score": 0.55,
            "decision": "review",
            "reasons": ["overly_extractive"],
            "explanation": "Answer is a near-verbatim copy of context_source with no reformulation.",
        }

    return None


def build_judge_messages(item: dict[str, Any]) -> list[dict[str, str]]:
    """Build a factuality-focused judge prompt for one QA item."""
    payload = {
        "question": item.get("question", ""),
        "answer": item.get("answer", ""),
        "context_source": item.get("context_source", ""),
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
                "Judge this RAG QA triple strictly.\n\n"
                "SCORING RUBRIC — apply rigorously:\n"
                "1.0 = perfect: answer fully and unambiguously supported by context_source.\n"
                "0.9-0.8 = minor issue: mostly supported but one unsupported term or slight overstatement.\n"
                "0.5-0.7 = serious issue: answer introduces key terms or conclusions absent from context_source.\n"
                "< 0.5 = fail: answer contradicts context, relies on outside knowledge, or context is degraded.\n\n"
                "FAIL INDICATORS — assign decision=fail if ANY apply:\n"
                "- Answer uses terms, numbers, or conclusions not present in context_source.\n"
                "- context_source contains visible encoding artifacts (â€™, Î¸, âˆˆ, broken formulas).\n"
                "- context_source contains a chunk boundary marker [doc-chunk-NNNN].\n"
                "- context_source starts or ends mid-sentence (truncated).\n"
                "- Answer directly contradicts the context.\n\n"
                "REVIEW INDICATORS — assign decision=review (unless already fail) if ANY apply:\n"
                "- Answer requires slight inference beyond what context_source states explicitly.\n"
                "- Answer is a near-verbatim copy of context_source with no reformulation.\n"
                "- Question is ambiguous but still answerable from context.\n\n"
                "PASS: answer is factually deducible from context_source alone, clearly stated, "
                "and reformulated in the respondent's own words (not copied verbatim).\n\n"
                "Return strict JSON only:\n"
                "{"
                "\"context_quality\": 0.0-1.0, "
                "\"answer_support\": 0.0-1.0, "
                "\"question_quality\": 0.0-1.0, "
                "\"overall_score\": 0.0-1.0, "
                "\"decision\": \"pass\"|\"review\"|\"fail\", "
                "\"reasons\": [from: factual, weak_context, unsupported_detail, "
                "truncated_context, extraction_artifact, unclear_question, overly_extractive, cross_chunk_context], "
                "\"explanation\": \"short explanation\""
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
    if "trunc" in reason or "cut" in reason or "incomplete" in reason:
        return "truncated_context"
    if "contrad" in reason:
        return "unsupported_detail"
    if "artifact" in reason or "formula" in reason or "symbol" in reason:
        return "extraction_artifact"
    if "extractive" in reason or "verbatim" in reason or "copy" in reason:
        return "overly_extractive"
    if "context" in reason or "evidence" in reason:
        return "weak_context"
    if "unclear" in reason or "ambig" in reason:
        return "unclear_question"
    return "weak_context"


def _score_from_payload(payload: dict[str, Any], key: str, fallback: float) -> float:
    return _coerce_score(payload.get(key, fallback))


def normalize_judge_result(payload: dict[str, Any], model: str) -> dict[str, Any]:
    """Normalize arbitrary judge JSON to the public judged item fields."""
    decision = str(payload.get("decision", "")).strip().lower()
    context_quality = _score_from_payload(payload, "context_quality", payload.get("score", 0.0))
    answer_support = _score_from_payload(payload, "answer_support", payload.get("score", 0.0))
    question_quality = _score_from_payload(payload, "question_quality", payload.get("score", 0.0))
    score = _score_from_payload(payload, "overall_score", payload.get("score", 0.0))
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
    if (
        decision == "pass"
        and min(context_quality, answer_support, question_quality) >= 0.8
        and set(reasons).issubset({"factual", "judge_error", "weak_context"})
    ):
        reasons = ["factual"]
    if any(reason in BLOCKING_REASONS for reason in reasons):
        if decision in ("pass", "review"):
            decision = "fail"
            score = min(score, 0.35)
    if min(context_quality, answer_support, question_quality) < 0.6:
        decision = "fail"
        score = min(score, 0.39)
    elif min(context_quality, answer_support, question_quality) < 0.8 and decision == "pass":
        decision = "review"
        score = min(score, 0.79)

    explanation = normalize_whitespace(str(payload.get("explanation", "")))
    if not explanation:
        explanation = "Judge did not provide an explanation."

    return {
        "judge_score": score,
        "judge_context_quality": context_quality,
        "judge_answer_support": answer_support,
        "judge_question_quality": question_quality,
        "judge_decision": decision,
        "judge_reasons": reasons[:5],
        "judge_explanation": truncate_text(explanation, 280),
        "judge_model": model,
    }


def judge_error_result(model: str, message: str = "Judge call failed or returned invalid JSON.") -> dict[str, Any]:
    """Return a non-destructive review result when judge evaluation fails."""
    return {
        "judge_score": 0.0,
        "judge_context_quality": 0.0,
        "judge_answer_support": 0.0,
        "judge_question_quality": 0.0,
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
    # Fast deterministic checks — skip the LLM when the defect is obvious.
    precheck = _deterministic_judge_prechecks(item)
    if precheck is not None:
        judged.update(normalize_judge_result(precheck, model=f"{model}:precheck"))
        return judged
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
    context_scores: list[float] = []
    answer_scores: list[float] = []
    question_scores: list[float] = []
    for item in judged_items:
        scores.append(_coerce_score(item.get("judge_score")))
        context_scores.append(_coerce_score(item.get("judge_context_quality")))
        answer_scores.append(_coerce_score(item.get("judge_answer_support")))
        question_scores.append(_coerce_score(item.get("judge_question_quality")))
        for reason in item.get("judge_reasons", []):
            reasons[str(reason)] += 1

    return {
        "mode": mode,
        "model": model,
        "judged_items": len(judged_items),
        "decision_counts": dict(sorted(decisions.items())),
        "average_score": round(sum(scores) / len(scores), 4) if scores else 0.0,
        "average_context_quality": round(sum(context_scores) / len(context_scores), 4) if context_scores else 0.0,
        "average_answer_support": round(sum(answer_scores) / len(answer_scores), 4) if answer_scores else 0.0,
        "average_question_quality": round(sum(question_scores) / len(question_scores), 4) if question_scores else 0.0,
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
