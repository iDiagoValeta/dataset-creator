"""Unit tests for pure functions in generate_dataset."""
import json
import re
from pathlib import Path

import generate_dataset as gd
import pytest

# --- parse_split ------------------------------------------------------

def test_parse_split_normalizes_ratios():
    train, val, test = gd.parse_split("8,1,1")
    assert train == pytest.approx(0.8)
    assert val == pytest.approx(0.1)
    assert test == pytest.approx(0.1)


def test_parse_split_rejects_wrong_arity():
    with pytest.raises(ValueError):
        gd.parse_split("0.8,0.2")


def test_parse_split_rejects_zero_total():
    with pytest.raises(ValueError):
        gd.parse_split("0,0,0")


def test_parse_split_rejects_negative_component():
    with pytest.raises(ValueError, match="negativos"):
        gd.parse_split("0.8,-0.1,0.3")


# --- defaults ---------------------------------------------------------

def test_cli_defaults_use_hybrid_embeddinggemma_and_gemma4_e4b():
    args = gd.build_arg_parser().parse_args([])
    assert args.model == "gemma4:e4b"
    assert args.embedding_model == "embeddinggemma:latest"
    assert args.retrieval == "hybrid"
    assert args.judge == "off"
    assert args.judge_model == "gemma4:e4b"


def test_cli_accepts_judge_filter_mode():
    args = gd.build_arg_parser().parse_args(["--judge", "filter"])
    gd.validate_args(args)
    assert args.judge == "filter"


# --- chunk_text -------------------------------------------------------

def test_chunk_text_produces_overlapping_chunks():
    text = "abcdefghij" * 100  # 1000 chars
    chunks = gd.chunk_text(text, chunk_size=200, chunk_overlap=50)
    assert len(chunks) > 1
    assert all(len(c) <= 200 for c in chunks)


def test_chunk_text_filters_tiny_chunks():
    assert gd.chunk_text("short", chunk_size=200, chunk_overlap=50) == []


def test_chunk_text_empty_input():
    assert gd.chunk_text("", chunk_size=200, chunk_overlap=50) == []


def test_chunk_text_prefers_word_boundary_near_limit():
    text = "alpha " * 100
    chunks = gd.chunk_text(text, chunk_size=182, chunk_overlap=30)
    assert len(chunks) > 1
    assert all(len(c) <= 182 for c in chunks)
    assert chunks[0].split()[-1] == "alpha"


def test_clean_markdown_artifacts_removes_picture_and_ref_noise():
    text = "Virtual memory in the 80 computer. == picture 188 x 297 intentionally omitted <== OS/360 stayed."
    cleaned = gd.clean_markdown_artifacts(text)
    assert "picture" not in cleaned
    assert "80 computer" not in cleaned
    assert "OS/360" in cleaned


def test_clean_markdown_artifacts_normalizes_domain_terms_and_strips_tail_sections():
    text = (
        "The highfrequency setting rewards riskadjusted metrics.\n\n"
        "Conclusion\nUseful final content.\n\n"
        "Code Availability\nThe implementation code is available online.\n\n"
        "References\nA. Author."
    )
    cleaned = gd.clean_markdown_artifacts(text)
    assert "high-frequency" in cleaned
    assert "risk-adjusted" in cleaned
    assert "Useful final content" in cleaned
    assert "Code Availability" not in cleaned
    assert "References" not in cleaned


def test_clean_generated_text_capitalizes_and_removes_numeric_artifact():
    cleaned = gd.clean_generated_text("swapping helps the 80 computer avoid limits Ã¢â‚¬â€ safely.")
    assert cleaned.startswith("Swapping")
    assert "80 computer" not in cleaned
    assert "Ã¢" not in cleaned


def test_clean_generated_text_removes_html_breaks_and_rejoins_words():
    cleaned = gd.clean_generated_text("sortino focuses on down-<br side risk and <b>losses</b>.")
    assert "<br" not in cleaned
    assert "<b>" not in cleaned
    assert "downside risk" in cleaned.lower()


# --- try_parse_json_payload -------------------------------------------

def test_try_parse_json_payload_clean_object():
    assert gd.try_parse_json_payload('{"items": [1, 2]}') == {"items": [1, 2]}


def test_try_parse_json_payload_extracts_from_prose():
    payload = 'Here is the result:\n{"items": [{"q": "x"}]}\nThanks.'
    parsed = gd.try_parse_json_payload(payload)
    assert parsed == {"items": [{"q": "x"}]}


def test_try_parse_json_payload_invalid_returns_empty_items():
    assert gd.try_parse_json_payload("no json here") == {"items": []}


# --- parse_topics -----------------------------------------------------

def test_parse_topics_reads_topics_key():
    payload = {
        "topics": [
            {"name": "A", "summary": "s", "keywords": ["x", "y"]},
            {"name": "B", "summary": "s", "keywords": "k1, k2"},
        ]
    }
    topics = gd.parse_topics(payload)
    assert [t.name for t in topics] == ["A", "B"]
    assert topics[1].keywords == ["k1", "k2"]


def test_parse_topics_skips_duplicates_case_insensitive():
    payload = {"topics": [{"name": "Alpha"}, {"name": "alpha"}]}
    topics = gd.parse_topics(payload)
    assert len(topics) == 1


def test_parse_topics_accepts_alternative_keys():
    payload = {"temas": [{"title": "T1"}]}
    topics = gd.parse_topics(payload)
    assert topics[0].name == "T1"


def test_parse_topics_empty_payload():
    assert gd.parse_topics({}) == []


def test_section_topics_from_headings_skip_generic_reference_noise():
    text = """
Operating system
Definition and purpose
This paragraph explains what operating systems do.
Components
Kernel
Program execution
References
List of operating systems
Histor y
"""
    topics = gd.build_section_topics_from_text(text, num_topics=4)
    assert [t.name for t in topics] == [
        "Definition and purpose",
        "Kernel",
        "Program execution",
    ]


def test_build_topic_context_prefers_exact_section_chunk():
    chunks = [
        gd.Chunk("doc-chunk-0000", "doc.pdf", "Definition and purpose\nOperating systems manage hardware."),
        gd.Chunk("doc-chunk-0001", "doc.pdf", "Types of operating systems\nEmbedded systems are common."),
        gd.Chunk("doc-chunk-0002", "doc.pdf", "Kernel\nA kernel protects memory."),
    ]
    topic = gd.Topic("topic-00", "Types of operating systems", "s", ["operating", "systems"])
    context = gd.build_topic_context(chunks, topic, max_topic_context_chars=10_000)
    assert "doc-chunk-0001" in context
    assert "doc-chunk-0000" not in context


def test_topics_are_too_generic_detects_bad_llm_map():
    topics = [
        gd.Topic("topic-00", "Operating System Concepts", "s", []),
        gd.Topic("topic-01", "Operating System References", "s", []),
        gd.Topic("topic-02", "Kernel", "s", []),
    ]
    assert gd.topics_are_too_generic(topics) is True


# --- language detection -----------------------------------------------

def test_detect_document_language_identifies_english():
    text = "The operating system manages computer hardware and software resources."
    code, name, scores = gd.detect_document_language(text)
    assert code == "en"
    assert name == "English"
    assert scores["en"] > scores["es"]


def test_resolve_generation_language_auto_builds_prompt_language():
    prompt_language, document_language, scores = gd.resolve_generation_language(
        "auto",
        "El sistema operativo gestiona los recursos de la computadora.",
    )
    assert document_language == "es"
    assert "Spanish" in prompt_language
    assert scores["es"] > scores["en"]


# --- deduplicate_items ------------------------------------------------

def _item(question: str) -> dict:
    return {"question": question, "answer": "a"}


def test_deduplicate_items_drops_exact_duplicates():
    items = [_item("What is X?"), _item("what is x?")]
    assert len(gd.deduplicate_items(items)) == 1


def test_deduplicate_items_drops_near_duplicates():
    items = [
        _item("What is the main goal of the operating system?"),
        _item("What is the main goal of an operating system?"),
    ]
    unique = gd.deduplicate_items(items, semantic_threshold=0.5)
    assert len(unique) == 1


def test_deduplicate_items_keeps_distinct_questions():
    items = [_item("What is memory paging?"), _item("How does disk scheduling work?")]
    assert len(gd.deduplicate_items(items)) == 2


def test_deduplicate_items_drops_duplicate_answers():
    items = [
        {"question": "What is an OS?", "answer": "An OS manages hardware resources."},
        {"question": "How can an operating system be described?", "answer": "An OS manages hardware resources."},
    ]
    assert len(gd.deduplicate_items(items)) == 1


def test_deduplicate_items_drops_near_duplicate_answers_with_punctuation_drift():
    items = [
        {
            "question": "What action did Singapore take?",
            "answer": "Singapore appended an additional section (Section ) to its guidance on software as a medical device (SaMD) (HSA), 2022.",
        },
        {
            "question": "What guidance did Singapore append for SaMD?",
            "answer": "Singapore, appended an additional section (Section ) to its guidance on software as a medical device (SaMD) (HSA), 2022 .",
        },
    ]
    assert len(gd.deduplicate_items(items)) == 1


def test_deduplicate_items_drops_cross_topic_semantic_duplicates():
    items = [
        {
            "question": "What does the Efficient Market Hypothesis suggest about asset prices?",
            "answer": "Asset prices fully incorporate all available information.",
        },
        {
            "question": "What does the Efficient Market Hypothesis suggest about stock prices?",
            "answer": "Stock prices fully reflect all available information.",
        },
    ]
    assert len(gd.deduplicate_items(items)) == 1


# --- filter_pdfs_by_only_doc ------------------------------------------

def test_filter_pdfs_by_only_doc_matches_filename(tmp_path: Path):
    a = tmp_path / "Operating_system.pdf"
    b = tmp_path / "Networks.pdf"
    a.write_bytes(b"%PDF-1.4")
    b.write_bytes(b"%PDF-1.4")
    pdfs = [a, b]
    assert gd.filter_pdfs_by_only_doc(pdfs, "Operating_system.pdf") == [a]
    assert gd.filter_pdfs_by_only_doc(pdfs, "operating_system") == [a]


def test_filter_pdfs_by_only_doc_no_match_raises(tmp_path: Path):
    pdf = tmp_path / "A.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    with pytest.raises(RuntimeError):
        gd.filter_pdfs_by_only_doc([pdf], "Missing.pdf")


def test_filter_pdfs_by_only_doc_none_returns_all(tmp_path: Path):
    pdf = tmp_path / "A.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    assert gd.filter_pdfs_by_only_doc([pdf], None) == [pdf]


def test_filter_pdfs_by_only_doc_multiple(tmp_path: Path):
    a = tmp_path / "A.pdf"
    b = tmp_path / "B.pdf"
    c = tmp_path / "C.pdf"
    for p in (a, b, c):
        p.write_bytes(b"%PDF-1.4")
    selected = gd.filter_pdfs_by_only_doc([a, b, c], "a,c")
    assert selected == [a, c]


def test_filter_pdfs_by_only_doc_multiple_partial_miss_raises(tmp_path: Path):
    a = tmp_path / "A.pdf"
    a.write_bytes(b"%PDF-1.4")
    with pytest.raises(RuntimeError):
        gd.filter_pdfs_by_only_doc([a], "a,ghost")


# --- context_source validation ---------------------------------------

def test_context_source_verified_flag_when_substring(monkeypatch):
    """Items whose context_source is a literal substring of context keep the claim."""
    topic = gd.Topic(topic_id="topic-00", name="T", summary="s", keywords=["k"])
    context = "The kernel schedules processes and allocates memory fairly."

    def fake_call(*_, **__):
        return (
            {
                "items": [
                    {
                        "question": "q",
                        "answer": "a",
                        "type": "factual",
                        "difficulty": "hard",
                        "context_source": "schedules processes",
                    }
                ]
            },
            "",
        )

    monkeypatch.setattr("engine._generation.call_ollama_json", fake_call)
    items, _ = gd.generate_items_for_topic(
        model="m",
        document="doc.pdf",
        topic=topic,
        topic_context=context,
        language="es",
        questions_per_topic=1,
        temperature=0.0,
        existing_questions=[],
    )
    assert items[0]["context_source_verified"] is True
    assert items[0]["context_source"] == "The kernel schedules processes and allocates memory fairly."
    assert items[0]["document_language"] == "es"
    assert items[0]["difficulty"] == "hard"
    assert items[0]["source_chunk_ids"] == []


def test_generated_item_excerpt_uses_verified_source_chunk(monkeypatch):
    topic = gd.Topic(topic_id="topic-00", name="T", summary="s", keywords=["k"])
    context = (
        "[doc-chunk-0000] Abstract overview unrelated to the generated answer. "
        "[doc-chunk-0001] Results show consistently negative Sortino ratios and weak downside protection."
    )

    def fake_call(*_, **__):
        return (
            {
                "items": [
                    {
                        "question": "q",
                        "answer": "negative Sortino ratios",
                        "type": "factual",
                        "difficulty": "medium",
                        "context_source": "negative Sortino ratios and weak downside protection",
                    }
                ]
            },
            "",
        )

    monkeypatch.setattr("engine._generation.call_ollama_json", fake_call)
    items, _ = gd.generate_items_for_topic(
        model="m",
        document="doc.pdf",
        topic=topic,
        topic_context=context,
        language="en",
        questions_per_topic=1,
        temperature=0.0,
        existing_questions=[],
    )
    assert items[0]["source_chunk_ids"] == ["doc-chunk-0001"]
    assert items[0]["context_excerpt"].startswith("[doc-chunk-0001]")


def test_find_verified_context_source_does_not_repair_from_answer_overlap():
    context = (
        "[doc-chunk-0001] The kernel schedules processes and allocates memory fairly. "
        "File systems store persistent data."
    )
    source, verified = gd.find_verified_context_source(
        "not a literal source",
        "The kernel schedules processes and allocates memory fairly.",
        context,
    )
    assert verified is False
    assert source == ""


def test_context_excerpt_for_fragment_anchors_to_source_chunk():
    context = (
        "[doc-chunk-0000] Abstract text that is relevant to a different question. "
        "[doc-chunk-0001] Results show consistently negative Sortino ratios and weak downside protection."
    )
    excerpt = gd.context_excerpt_for_fragment(context, "negative Sortino ratios and weak downside", max_chars=160)
    assert excerpt.startswith("[doc-chunk-0001]")
    assert "doc-chunk-0000" not in excerpt
    assert "negative Sortino ratios" in excerpt


def test_context_excerpt_for_fragment_keeps_distant_source_visible():
    context = (
        "[doc-chunk-0004] "
        + ("background sentence. " * 40)
        + "The dataset used in this study consists of minute-level historical stock data."
    )
    source = "The dataset used in this study consists of minute-level historical stock data."
    excerpt = gd.context_excerpt_for_fragment(context, source, max_chars=180)
    assert excerpt.startswith("[doc-chunk-0004]")
    assert source in excerpt


def test_context_excerpt_for_fragment_fallback_keeps_word_boundary():
    context = "[doc-chunk-0000] " + ("alpha " * 100)
    excerpt = gd.context_excerpt_for_fragment(context, "missing fragment", max_chars=125)
    assert len(excerpt) <= 125
    assert excerpt.split()[-1] == "alpha"


def test_find_verified_context_source_expands_to_complete_sentence():
    context = "Introductory note. The kernel schedules processes and allocates memory fairly. Final note."
    source, verified = gd.find_verified_context_source("schedules processes", "It schedules processes.", context)
    assert verified is True
    assert source == "The kernel schedules processes and allocates memory fairly."


def test_find_verified_context_source_keeps_long_literal_without_mid_word_cut():
    long_source = " ".join(["alpha"] * 80)
    source, verified = gd.find_verified_context_source(long_source, "alpha alpha alpha", long_source)
    assert verified is True
    assert source == long_source


def test_find_verified_context_source_cleans_html_breaks_from_source():
    context = "[doc-chunk-0001] Sortino ratio improves on Sharpe by focusing only on downside risk."
    source, verified = gd.find_verified_context_source(
        "Sortino ratio improves on Sharpe by focusing only on down-<br side risk.",
        "It focuses only on downside risk.",
        context,
    )
    assert verified is False
    assert source == ""


def test_sample_existing_questions_respects_limit():
    qs = [f"q{i}" for i in range(100)]
    sampled = gd._sample_existing_questions(qs, limit=20)
    assert len(sampled) == 20
    # Recent half must be present in order at the tail.
    assert sampled[-10:] == qs[-10:]


def test_sample_existing_questions_no_sampling_when_under_limit():
    qs = ["a", "b", "c"]
    assert gd._sample_existing_questions(qs, limit=10) == qs


def test_checkpoint_roundtrip(tmp_path: Path):
    pdf = tmp_path / "Doc.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    items = [{"id": "x-00", "question": "q", "answer": "a"}]
    assert gd.load_checkpoint_items(tmp_path, pdf) == []
    gd.save_checkpoint_items(tmp_path, pdf, items)
    assert gd.load_checkpoint_items(tmp_path, pdf) == items


def test_infer_resume_counts_prefers_debug_metadata(tmp_path: Path):
    pdf = tmp_path / "Doc.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    (tmp_path / "Doc.json").write_text(
        json.dumps(
            {
                "chunk_count": 7,
                "topics": [
                    {"topic_id": "topic-00"},
                    {"topic_id": "topic-01"},
                ],
            }
        ),
        encoding="utf-8",
    )
    items = [
        {"topic_id": "topic-00", "source_chunk_ids": ["doc-chunk-0001"]},
        {"topic_id": "topic-01", "source_chunk_ids": ["doc-chunk-0002"]},
    ]

    assert gd.infer_resume_counts(tmp_path, pdf, items) == (7, 2)


def test_infer_resume_counts_falls_back_to_checkpoint_items(tmp_path: Path):
    pdf = tmp_path / "Doc.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    items = [
        {"topic_id": "topic-00", "source_chunk_ids": ["doc-chunk-0001", "doc-chunk-0002"]},
        {"topic_id": "topic-01", "source_chunk_ids": ["doc-chunk-0002"]},
    ]

    assert gd.infer_resume_counts(tmp_path, pdf, items) == (2, 2)


def test_infer_resume_counts_recomputes_chunk_count_when_configured(tmp_path: Path, monkeypatch):
    pdf = tmp_path / "Doc.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    items = [{"topic_id": "topic-00", "source_chunk_ids": ["doc-chunk-0001"]}]

    monkeypatch.setattr(gd, "extract_text_from_pdf", lambda path: "alpha " * 200)

    chunks = [
        gd.Chunk(chunk_id=f"doc-chunk-{idx:04d}", document="Doc.pdf", text="alpha")
        for idx in range(5)
    ]

    def fake_build_chunks(**kwargs):  # noqa: ARG001
        return chunks

    monkeypatch.setattr(gd, "build_chunks_from_text", fake_build_chunks)

    assert gd.infer_resume_counts(
        tmp_path,
        pdf,
        items,
        chunk_size=3500,
        chunk_overlap=350,
        max_chunks=None,
    ) == (5, 1)


def test_context_source_falls_back_when_not_substring(monkeypatch):
    topic = gd.Topic(topic_id="topic-00", name="T", summary="s", keywords=[])
    context = "Pagination avoids fragmentation."

    def fake_call(*_, **__):
        return (
            {
                "items": [
                    {
                        "question": "q",
                        "answer": "a",
                        "type": "bogus",
                        "difficulty": "extreme",
                        "context_source": "literal fabrication not in source",
                    }
                ]
            },
            "",
        )

    monkeypatch.setattr("engine._generation.call_ollama_json", fake_call)
    items, _ = gd.generate_items_for_topic(
        model="m",
        document="doc.pdf",
        topic=topic,
        topic_context=context,
        language="es",
        questions_per_topic=1,
        temperature=0.0,
        existing_questions=[],
    )
    assert items[0]["context_source_verified"] is False
    assert items[0]["type"] == "factual"          # bogus type normalized
    assert items[0]["difficulty"] == "medium"     # bogus difficulty normalized
    assert items[0]["context_source"].startswith("Pagination")


def test_apply_quality_gate_strict_rejects_unverified_and_artifacts():
    items = [
        {"id": "ok", "context_source_verified": True, "answer": "Kernel allocates memory.", "topic": "Kernel"},
        {"id": "bad-source", "context_source_verified": False, "answer": "Unsupported.", "topic": "Kernel"},
        {
            "id": "bad-artifact",
            "context_source_verified": True,
            "answer": "Virtual memory in the 80 computer.",
            "topic": "Kernel",
        },
    ]
    accepted, rejected, stats = gd.apply_quality_gate(items, "strict")
    assert [item["id"] for item in accepted] == ["ok"]
    assert {item["id"] for item in rejected} == {"bad-source", "bad-artifact"}
    assert stats["verified_ratio"] == 1.0


def test_apply_quality_gate_rejects_generation_and_reference_noise():
    items = [
        {
            "id": "cut-off",
            "context_source_verified": True,
            "answer": "The text cuts off before the full evidence is shown.",
            "topic": "Results",
        },
        {
            "id": "joined-word",
            "context_source_verified": True,
            "answer": "The model improves riskadjusted performance.",
            "topic": "Results",
        },
        {
            "id": "references-noise",
            "context_source_verified": True,
            "answer": "The paper uses the Rachev ratio.",
            "context_excerpt": "The paper uses the Rachev ratio. References Abrol et al.",
            "topic": "Results",
        },
    ]
    accepted, rejected, stats = gd.apply_quality_gate(items, "strict")
    assert accepted == []
    assert {item["id"] for item in rejected} == {"cut-off", "joined-word", "references-noise"}
    assert stats["rejection_reasons"] == {"artifact_or_reference_noise": 3}


def test_apply_quality_gate_rejects_html_formula_noise():
    item = {
        "id": "html-noise",
        "context_source_verified": True,
        "answer": "The Sortino ratio focuses on downside risk.",
        "context_source": "Sortino ratio E Rp-Rf <br sd <br downside risk.",
        "topic": "Risk metrics",
    }
    accepted, rejected, stats = gd.apply_quality_gate([item], "strict")
    assert accepted == []
    assert rejected[0]["rejection_reason"] == "artifact_or_reference_noise"
    assert stats["rejection_reasons"] == {"artifact_or_reference_noise": 1}


@pytest.mark.parametrize(
    "bad_text",
    [
        "Singapore appended an additional section (Section ) to its guidance.",
        "In Figure , the paper compares six tasks.",
        "J?? controls the position of the chosen arm joints.",
        "The remaining ? = DoF coordinates are solved with IK.",
        "The remaining ( DoF) coordinates are solved with IK.",
        "The remaining \u2212 = DoF coordinates are solved with IK.",
        "Fig. : reports the control ablation.",
        "Figure : reports the control ablation.",
        "The result is summarized in Figure 7.",
        "The answer contains a replacement character \ufffd.",
    ],
)
def test_apply_quality_gate_rejects_broken_extraction_artifacts(bad_text):
    item = {
        "id": "broken-artifact",
        "question": "What does the paper state?",
        "answer": bad_text,
        "context_source": bad_text,
        "context_source_verified": True,
        "topic": "Results",
    }
    accepted, rejected, stats = gd.apply_quality_gate([item], "strict")
    assert accepted == []
    assert rejected[0]["rejection_reason"] == "artifact_or_reference_noise"
    assert stats["rejection_reasons"] == {"artifact_or_reference_noise": 1}


def test_apply_quality_gate_rejects_compare_with_unsupported_answer_detail():
    item = {
        "id": "unsupported-compare",
        "question": "How do A and B differ?",
        "answer": "A reduces risk by using adaptive features, while B increases returns through raw prices.",
        "context_source": "A reduces risk by using adaptive features.",
        "context_source_verified": True,
        "type": "compare",
        "topic": "Model comparison",
    }
    accepted, rejected, stats = gd.apply_quality_gate([item], "strict")
    assert accepted == []
    assert rejected[0]["rejection_reason"] == "insufficient_context_support"
    assert stats["rejection_reasons"] == {"insufficient_context_support": 1}


def test_apply_quality_gate_rejects_factual_answer_with_unsupported_terms():
    item = {
        "id": "unsupported-factual",
        "question": "What did the policy require?",
        "answer": "The policy required encrypted cloud storage, biometric review, and monthly audits.",
        "context_source": "The policy required encrypted cloud storage.",
        "context_source_verified": True,
        "type": "factual",
        "topic": "Data governance",
    }
    accepted, rejected, stats = gd.apply_quality_gate([item], "strict")
    assert accepted == []
    assert rejected[0]["rejection_reason"] == "insufficient_context_support"
    assert stats["rejection_reasons"] == {"insufficient_context_support": 1}


def test_apply_quality_gate_rejects_topic_mismatch():
    item = {
        "id": "wrong-topic",
        "question": "What deep learning architectures should future studies explore?",
        "answer": "Future studies should explore LSTM networks and Transformer-based architectures.",
        "context_source": "Future studies should explore models such as Long Short-Term Memory (LSTM) networks and Transformer-based architectures.",
        "context_source_verified": True,
        "topic": "Technical Analysis and Market Prediction",
        "topic_keywords": ["Technical Analysis", "Market Prediction", "Financial Forecasting", "Technical Indicators"],
    }
    accepted, rejected, stats = gd.apply_quality_gate([item], "strict")
    assert accepted == []
    assert rejected[0]["rejection_reason"] == "topic_mismatch"
    assert stats["rejection_reasons"] == {"topic_mismatch": 1}


def test_apply_quality_gate_rejects_local_topic_mismatch_without_domain_terms():
    item = {
        "id": "wrong-local-topic",
        "question": "What should organizations do regarding their energy consumption and carbon footprint?",
        "answer": "They should publish their energy consumption and carbon footprint.",
        "context_source": "They should also publish their energy consumption and carbon footprint.",
        "context_source_verified": True,
        "type": "factual",
        "topic": "Semiconductor and Hardware Design",
        "topic_keywords": ["semiconductor", "hardware design", "AI infrastructure"],
    }
    accepted, rejected, stats = gd.apply_quality_gate([item], "strict")
    assert accepted == []
    assert rejected[0]["rejection_reason"] == "topic_mismatch"
    assert stats["rejection_reasons"] == {"topic_mismatch": 1}


def test_apply_quality_gate_keeps_shared_domain_feature_importance_item():
    item = {
        "id": "shared-domain",
        "question": "What feature importance split did the Random Forest analysis find?",
        "answer": "Primary price-based features accounted for over 60% while technical indicators accounted for 14% to 15%.",
        "context_source": "A feature importance analysis demonstrates that primary price-based features dominate the predictions made by the model, accounting for over 60% of the importance, while established technical indicators account for only 14% to 15%.",
        "context_source_verified": True,
        "topic": "Random Forest Models in High-Frequency Trading",
        "topic_keywords": ["Random Forest Regression", "High-Frequency Trading", "Stock Price Prediction"],
    }
    accepted, rejected, _ = gd.apply_quality_gate([item], "strict")
    assert accepted == [item]
    assert rejected == []


def test_apply_quality_gate_keeps_related_cross_domain_hft_item():
    item = {
        "id": "related-topic",
        "question": "What did the findings suggest about traditional technical indicators in high-frequency markets?",
        "answer": "They may have lower predictive value in modern high-frequency markets.",
        "context_source": "These findings suggest that traditional technical indicators may have diminishing predictive value in modern high-frequency markets.",
        "context_source_verified": True,
        "topic": "High-Frequency Trading and Market Microstructure",
        "topic_keywords": ["High-Frequency Trading", "Market Microstructure", "Order Flow", "Latency"],
    }
    accepted, rejected, _ = gd.apply_quality_gate([item], "strict")
    assert accepted == [item]
    assert rejected == []


def test_apply_quality_gate_rejects_truncated_context_source():
    item = {
        "id": "truncated",
        "question": "What does EMH suggest about asset prices?",
        "answer": "The EMH suggests that asset prices incorporate all available information.",
        "context_source": "icient Market Hypothesis (EMH) suggests that asset prices fully incorporate all available information.",
        "context_source_verified": True,
        "topic": "Technical Analysis and Market Prediction",
        "topic_keywords": ["Technical Analysis", "Market Prediction", "Financial Forecasting", "Technical Indicators"],
    }
    accepted, rejected, _ = gd.apply_quality_gate([item], "strict")
    assert accepted == []
    assert rejected[0]["rejection_reason"] == "truncated_context_source"


@pytest.mark.parametrize(
    "context_source",
    [
        "ult, the joint space controller produced lower tracking error than the task-space controller.",
        "The task-space controller produced lower tracking error by",
        "The benchmark reports stronger performance across all tasks, offering",
    ],
)
def test_apply_quality_gate_rejects_cut_context_boundaries(context_source):
    item = {
        "id": "cut-boundary",
        "question": "What does the benchmark report?",
        "answer": "The benchmark reports stronger performance.",
        "context_source": context_source,
        "context_source_verified": True,
        "topic": "Robot learning",
    }
    accepted, rejected, _ = gd.apply_quality_gate([item], "strict")
    assert accepted == []
    assert rejected[0]["rejection_reason"] == "truncated_context_source"


def test_apply_quality_gate_rejects_first_person_answer():
    item = {
        "id": "first-person",
        "question": "What do the findings support?",
        "answer": "Our findings provide empirical support for the semi-strong form of the EMH.",
        "context_source": "The findings provide empirical support for the semi-strong form of the EMH.",
        "context_source_verified": True,
        "topic": "High-Frequency Trading and Market Microstructure",
        "topic_keywords": ["High-Frequency Trading", "Market Microstructure", "Order Flow", "Latency"],
    }
    accepted, rejected, _ = gd.apply_quality_gate([item], "strict")
    assert accepted == []
    assert rejected[0]["rejection_reason"] == "first_person_answer"


# --- split/audit ------------------------------------------------------

def test_split_rows_spreads_small_splits_across_topics():
    rows = (
        [{"id": f"a{i}", "topic_id": "topic-a"} for i in range(5)]
        + [{"id": f"b{i}", "topic_id": "topic-b"} for i in range(4)]
        + [{"id": f"c{i}", "topic_id": "topic-c"} for i in range(3)]
    )
    train, val, test = gd.split_rows(rows, split=(0.7, 0.15, 0.15), seed=42)
    assert len(train) == 8
    assert len(val) == 1
    assert len(test) == 3
    assert len({row["topic_id"] for row in test}) == 3


def test_build_dataset_audit_reports_topic_and_split_warnings():
    # Keys are now composite "{document}::{topic_id}".
    doc = "paper.pdf"
    accepted = [
        {"document": doc, "topic_id": "topic-a"},
        {"document": doc, "topic_id": "topic-a"},
        {"document": doc, "topic_id": "topic-b"},
    ]
    rejected = [{"document": doc, "topic_id": "topic-c", "rejection_reason": "artifact_or_reference_noise"}]
    audit = gd.build_dataset_audit(
        accepted,
        rejected,
        train_rows=accepted,
        val_rows=[{"document": doc, "topic_id": "topic-a"}],
        test_rows=[],
        expected_topic_ids=["topic-a", "topic-b", "topic-c"],
    )
    assert audit["accepted_by_topic"] == {"paper.pdf::topic-a": 2, "paper.pdf::topic-b": 1}
    assert audit["rejected_by_topic"] == {"paper.pdf::topic-c": 1}
    assert audit["topics_without_accepted"] == ["paper.pdf::topic-c"]
    assert "paper.pdf::topic-b" in audit["low_accepted_topics"]
    assert "val_split_has_single_topic" in audit["warnings"]


def test_build_dataset_audit_includes_expected_composite_topics_without_rows():
    audit = gd.build_dataset_audit(
        accepted_rows=[{"document": "doc-a.pdf", "topic_id": "topic-00"}],
        rejected_rows=[],
        train_rows=[],
        val_rows=[],
        test_rows=[],
        expected_topic_ids=["doc-a.pdf::topic-00", "doc-b.pdf::topic-00"],
    )
    assert "doc-b.pdf::topic-00" in audit["topics_without_accepted"]


# --- judge audit ------------------------------------------------------

def test_build_judge_messages_payload_contains_only_essential_fields():
    item = {
        "question": "Q?",
        "answer": "A.",
        "context_source": "Ctx.",
        "topic": "T",
        "type": "factual",
        "difficulty": "easy",
        "document_language": "en",
        "context_excerpt": "Extra.",
    }
    messages = gd.build_judge_messages(item)
    user_content = messages[-1]["content"]
    payload = json.loads(user_content.split("QA item:\n", 1)[1])

    assert set(payload.keys()) == {"question", "answer", "context_source"}
    assert re.search(r'"(topic|type|difficulty|document_language|context_excerpt)"', user_content) is None


def test_audit_items_per_document_produces_same_judged_rows(monkeypatch):
    items = [
        {"id": "a1", "document": "doc-a.pdf", "question": "q1", "answer": "a1", "context_source": "ctx1"},
        {"id": "b1", "document": "doc-b.pdf", "question": "q2", "answer": "a2", "context_source": "ctx2"},
        {"id": "a2", "document": "doc-a.pdf", "question": "q3", "answer": "a3", "context_source": "ctx3"},
    ]
    calls = []

    def fake_audit(batch, model, temperature, seed):  # noqa: ARG001
        calls.append([item["id"] for item in batch])
        judged = [
            {
                **item,
                "judge_score": 1.0,
                "judge_context_quality": 1.0,
                "judge_answer_support": 1.0,
                "judge_question_quality": 1.0,
                "judge_decision": "pass",
                "judge_reasons": ["factual"],
                "judge_explanation": "ok",
                "judge_model": model,
            }
            for item in batch
        ]
        return judged, gd.build_judge_stats("audit", model, judged)

    monkeypatch.setattr(gd, "audit_items_with_judge", fake_audit)

    all_at_once, _ = fake_audit(items, model="judge-model", temperature=0.0, seed=1)
    calls.clear()
    by_document, stats = gd.audit_items_with_judge_by_document(
        items,
        model="judge-model",
        temperature=0.0,
        seed=1,
    )

    assert calls == [["a1", "a2"], ["b1"]]
    assert {item["id"] for item in by_document} == {item["id"] for item in all_at_once}
    assert stats["judged_items"] == 3
    assert stats["decision_counts"] == {"pass": 3}


def test_audit_items_per_document_uses_requested_stats_mode(monkeypatch):
    def fake_audit(batch, model, temperature, seed):  # noqa: ARG001
        judged = [
            {
                **item,
                "judge_score": 1.0,
                "judge_context_quality": 1.0,
                "judge_answer_support": 1.0,
                "judge_question_quality": 1.0,
                "judge_decision": "pass",
                "judge_reasons": ["factual"],
                "judge_explanation": "ok",
                "judge_model": model,
            }
            for item in batch
        ]
        return judged, gd.build_judge_stats("audit", model, judged)

    monkeypatch.setattr(gd, "audit_items_with_judge", fake_audit)
    _, stats = gd.audit_items_with_judge_by_document(
        [{"id": "a", "document": "doc.pdf"}],
        model="judge-model",
        temperature=0.0,
        mode="filter",
    )
    assert stats["mode"] == "filter"


def test_filter_rows_by_judge_keeps_only_passed_rows():
    rows = [
        {"id": "pass", "question": "q1"},
        {"id": "fail", "question": "q2"},
        {"id": "review", "question": "q3"},
    ]
    judged = [
        {**rows[0], "judge_decision": "pass", "judge_reasons": ["factual"]},
        {**rows[1], "judge_decision": "fail", "judge_reasons": ["unsupported_detail"]},
        {**rows[2], "judge_decision": "review", "judge_reasons": ["overly_extractive"]},
    ]
    passed, rejected = gd.filter_rows_by_judge(rows, judged)
    assert passed == [rows[0]]
    assert [row["id"] for row in rejected] == ["fail", "review"]
    assert rejected[0]["rejection_reason"] == "judge_fail:unsupported_detail"
    assert rejected[1]["rejection_reason"] == "judge_fail:overly_extractive"


def test_normalize_judge_result_coerces_valid_public_fields():
    result = gd.normalize_judge_result(
        {
            "context_quality": "0.92",
            "answer_support": 0.94,
            "question_quality": 0.9,
            "overall_score": "0.91",
            "decision": "PASS",
            "reasons": ["Supported by context"],
            "explanation": "The answer is supported by the cited source.",
        },
        model="judge-model",
    )
    assert result["judge_score"] == pytest.approx(0.91)
    assert result["judge_context_quality"] == pytest.approx(0.92)
    assert result["judge_answer_support"] == pytest.approx(0.94)
    assert result["judge_question_quality"] == pytest.approx(0.9)
    assert result["judge_decision"] == "pass"
    assert result["judge_reasons"] == ["factual"]
    assert result["judge_model"] == "judge-model"


@pytest.mark.parametrize("reason", ["unsupported_detail", "truncated_context", "extraction_artifact"])
def test_normalize_judge_result_blocking_reasons_force_fail(reason):
    result = gd.normalize_judge_result(
        {
            "context_quality": 0.95,
            "answer_support": 0.95,
            "question_quality": 0.95,
            "overall_score": 0.95,
            "decision": "pass",
            "reasons": [reason],
            "explanation": "The item has a blocking issue.",
        },
        model="judge-model",
    )
    assert result["judge_decision"] == "fail"
    assert result["judge_score"] <= 0.35


@pytest.mark.parametrize("reason", ["weak_context", "judge_error"])
def test_normalize_judge_result_reconciles_noisy_pass_reasons(reason):
    result = gd.normalize_judge_result(
        {
            "context_quality": 1.0,
            "answer_support": 1.0,
            "question_quality": 1.0,
            "overall_score": 1.0,
            "decision": "pass",
            "reasons": [reason],
            "explanation": "The answer is fully supported by the context.",
        },
        model="judge-model",
    )
    assert result["judge_decision"] == "pass"
    assert result["judge_reasons"] == ["factual"]


def test_normalize_judge_result_weak_context_still_fails_without_pass_confidence():
    result = gd.normalize_judge_result(
        {
            "context_quality": 0.7,
            "answer_support": 0.7,
            "question_quality": 0.7,
            "overall_score": 0.7,
            "decision": "review",
            "reasons": ["weak_context"],
            "explanation": "The context is too weak.",
        },
        model="judge-model",
    )
    assert result["judge_decision"] == "fail"
    assert result["judge_score"] <= 0.35


def test_normalize_judge_result_fails_low_component_score():
    result = gd.normalize_judge_result(
        {
            "context_quality": 0.9,
            "answer_support": 0.45,
            "question_quality": 0.9,
            "overall_score": 0.7,
            "decision": "pass",
            "reasons": ["factual"],
            "explanation": "Answer support is too weak.",
        },
        model="judge-model",
    )
    assert result["judge_decision"] == "fail"
    assert result["judge_score"] <= 0.39


def test_judge_item_marks_review_on_invalid_judge_payload(monkeypatch):
    monkeypatch.setattr("engine._judge.call_ollama_json", lambda **kw: ({"items": []}, "not-json"))
    judged = gd.judge_item(
        {
            "id": "qa-1",
            "question": "What is supported?",
            "answer": "A supported answer.",
            "context_source": "A supported answer.",
        },
        model="judge-model",
        temperature=0.0,
        seed=42,
    )
    assert judged["judge_decision"] == "review"
    assert judged["judge_reasons"] == ["judge_error"]
    assert judged["judge_raw_content"] == "not-json"


def test_audit_items_with_judge_builds_stats(monkeypatch):
    responses = [
        (
            {
                "context_quality": 0.9,
                "answer_support": 1.0,
                "question_quality": 0.95,
                "overall_score": 0.95,
                "decision": "pass",
                "reasons": ["factual"],
                "explanation": "ok",
            },
            "",
        ),
        (
            {
                "context_quality": 0.4,
                "answer_support": 0.3,
                "question_quality": 0.35,
                "overall_score": 0.35,
                "decision": "fail",
                "reasons": ["unsupported detail"],
                "explanation": "bad",
            },
            "",
        ),
    ]

    def fake_call(**kw):  # noqa: ARG001
        return responses.pop(0)

    monkeypatch.setattr("engine._judge.call_ollama_json", fake_call)
    items = [
        {"id": "a", "question": "q1", "answer": "a1", "context_source": "a1"},
        {"id": "b", "question": "q2", "answer": "a2", "context_source": "source"},
    ]
    judged, stats = gd.audit_items_with_judge(items, model="judge-model", temperature=0.0, seed=1)
    assert [item["judge_decision"] for item in judged] == ["pass", "fail"]
    assert stats["mode"] == "audit"
    assert stats["model"] == "judge-model"
    assert stats["judged_items"] == 2
    assert stats["decision_counts"] == {"fail": 1, "pass": 1}
    assert stats["average_score"] == pytest.approx(0.65)
    assert stats["average_context_quality"] == pytest.approx(0.65)
    assert stats["average_answer_support"] == pytest.approx(0.65)
    assert stats["average_question_quality"] == pytest.approx(0.65)
    assert stats["reason_counts"] == {"factual": 1, "unsupported_detail": 1}


# --- cosine_similarity ------------------------------------------------

def test_cosine_similarity_identical_vectors():
    assert gd.cosine_similarity([1.0, 0.0, 0.0], [1.0, 0.0, 0.0]) == pytest.approx(1.0)


def test_cosine_similarity_orthogonal_vectors():
    assert gd.cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_cosine_similarity_zero_vector():
    assert gd.cosine_similarity([0.0, 0.0], [1.0, 0.0]) == 0.0


# --- load_topics_file -------------------------------------------------

def test_load_topics_file_plaintext(tmp_path):
    f = tmp_path / "topics.txt"
    f.write_text("Métodos estadísticos\n# comentario ignorado\n\nResultados experimentales\n", encoding="utf-8")
    topics = gd.load_topics_file(f)
    assert len(topics) == 2
    assert topics[0].name == "Métodos estadísticos"
    assert topics[1].name == "Resultados experimentales"
    assert topics[0].topic_id == "user-00"
    assert topics[1].topic_id == "user-01"


def test_load_topics_file_yaml(tmp_path):
    pytest.importorskip("yaml")
    f = tmp_path / "topics.yaml"
    f.write_text(
        "topics:\n"
        "  - name: Introducción\n"
        "    summary: Descripción general\n"
        "    keywords: [resumen, contexto]\n"
        "  - name: Metodología\n",
        encoding="utf-8",
    )
    topics = gd.load_topics_file(f)
    assert len(topics) == 2
    assert topics[0].summary == "Descripción general"
    assert topics[0].keywords == ["resumen", "contexto"]
    assert topics[1].summary == "Metodología"
    assert len(topics[1].keywords) > 0


def test_load_topics_file_empty_raises(tmp_path):
    f = tmp_path / "empty.txt"
    f.write_text("# solo comentarios\n\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no se encontraron temas"):
        gd.load_topics_file(f)


# --- load_questions_file ----------------------------------------------

def test_load_questions_file_filters_blanks(tmp_path):
    f = tmp_path / "qs.txt"
    f.write_text("¿Qué es el EMH?\n# ignorado\n\n¿Cómo funciona el modelo?\n", encoding="utf-8")
    qs = gd.load_questions_file(f)
    assert qs == ["¿Qué es el EMH?", "¿Cómo funciona el modelo?"]


def test_load_questions_file_empty_raises(tmp_path):
    f = tmp_path / "empty.txt"
    f.write_text("\n\n# nada\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no se encontraron preguntas"):
        gd.load_questions_file(f)


# --- questions_to_topics ----------------------------------------------

def test_questions_to_topics_structure():
    qs = ["¿Cuál es la hipótesis?", "¿Qué métodos se usaron?", "¿Cuáles son los resultados?"]
    topics = gd.questions_to_topics(qs)
    assert len(topics) == 3
    for i, (t, q) in enumerate(zip(topics, qs, strict=False)):
        assert t.topic_id == f"seed-{i:02d}"
        assert t.name == q
        assert t.summary == q
        assert isinstance(t.keywords, list)


# --- get_chunk_embedding (caching) ------------------------------------

def test_get_chunk_embedding_caches_result(monkeypatch):
    call_count = {"n": 0}

    class FakeResp:
        embedding = [0.1, 0.2, 0.3]

    class FakeClient:
        def embeddings(self, model, prompt):  # noqa: ARG002
            call_count["n"] += 1
            return FakeResp()

    monkeypatch.setattr(gd.ollama, "Client", lambda **kw: FakeClient())  # noqa: ARG005
    chunk = gd.Chunk(chunk_id="doc-chunk-0001", document="doc.pdf", text="Texto de prueba.")
    cache: dict = {}
    result1 = gd.get_chunk_embedding(chunk, "gemma4:e4b", cache)
    result2 = gd.get_chunk_embedding(chunk, "gemma4:e4b", cache)
    assert result1 == [0.1, 0.2, 0.3]
    assert result1 == result2
    assert call_count["n"] == 1  # computed only once


def test_get_chunk_embedding_fallback_on_error(monkeypatch):
    class BrokenClient:
        def embeddings(self, **kw):  # noqa: ARG002
            raise RuntimeError("Ollama unreachable")

    monkeypatch.setattr(gd.ollama, "Client", lambda **kw: BrokenClient())  # noqa: ARG005
    chunk = gd.Chunk(chunk_id="doc-chunk-0002", document="doc.pdf", text="Texto.")
    result = gd.get_chunk_embedding(chunk, "gemma4:e4b", {})
    assert result == []


# --- build_topic_context with semantic mode ---------------------------

def test_build_topic_context_semantic_falls_back_on_empty_embedding(monkeypatch):
    monkeypatch.setattr("engine._topics.get_topic_embedding", lambda topic, model: [])
    chunks = [
        gd.Chunk(chunk_id="d-chunk-0001", document="d.pdf", text="Alpha beta gamma"),
        gd.Chunk(chunk_id="d-chunk-0002", document="d.pdf", text="Delta epsilon zeta"),
    ]
    topic = gd.Topic(topic_id="t-00", name="Inexistente", summary="resumen", keywords=[])
    ctx = gd.build_topic_context(chunks, topic, 50000, retrieval_mode="semantic")
    assert ctx  # no error, returns something


def test_build_topic_context_semantic_prefers_similar_chunk(monkeypatch):
    def fake_topic_emb(topic, model):
        return [1.0, 0.0]

    def fake_chunk_emb(chunk, model, cache):
        if "0001" in chunk.chunk_id:
            return [1.0, 0.0]  # high cosine with topic
        return [0.0, 1.0]  # low cosine with topic

    monkeypatch.setattr("engine._topics.get_topic_embedding", fake_topic_emb)
    monkeypatch.setattr("engine._ollama.get_chunk_embedding", fake_chunk_emb)
    chunks = [
        gd.Chunk(chunk_id="d-chunk-0001", document="d.pdf", text="Contenido relevante del tema"),
        gd.Chunk(chunk_id="d-chunk-0002", document="d.pdf", text="Contenido irrelevante completamente"),
    ]
    topic = gd.Topic(topic_id="t-00", name="Tema de prueba", summary="resumen", keywords=[])
    ctx = gd.build_topic_context(chunks, topic, 50000, retrieval_mode="semantic")
    assert "Contenido relevante" in ctx


def test_build_topic_context_hybrid_combines_lexical_and_semantic(monkeypatch):
    monkeypatch.setattr("engine._topics.get_topic_embedding", lambda topic, model: [1.0, 0.0])

    def fake_chunk_emb(chunk, model, cache):
        if "0002" in chunk.chunk_id:
            return [1.0, 0.0]
        return [0.0, 1.0]

    monkeypatch.setattr("engine._ollama.get_chunk_embedding", fake_chunk_emb)
    chunks = [
        gd.Chunk(chunk_id="d-chunk-0001", document="d.pdf", text="rarelex token appears here."),
        gd.Chunk(chunk_id="d-chunk-0002", document="d.pdf", text="Semantically closest passage."),
    ]
    topic = gd.Topic(topic_id="t-00", name="Missing topic", summary="resumen", keywords=["rarelex"])
    ctx = gd.build_topic_context(chunks, topic, 50000, retrieval_mode="hybrid")
    assert ctx.index("d-chunk-0002") < ctx.index("d-chunk-0001")


def test_build_topic_context_hybrid_falls_back_to_lexical_on_empty_embedding(monkeypatch):
    monkeypatch.setattr("engine._topics.get_topic_embedding", lambda topic, model: [])
    chunks = [
        gd.Chunk(chunk_id="d-chunk-0001", document="d.pdf", text="unrelated text"),
        gd.Chunk(chunk_id="d-chunk-0002", document="d.pdf", text="rarelex token appears here."),
    ]
    topic = gd.Topic(topic_id="t-00", name="Missing topic", summary="resumen", keywords=["rarelex"])
    ctx = gd.build_topic_context(chunks, topic, 50000, retrieval_mode="hybrid")
    assert ctx.startswith("[d-chunk-0002]")


# --- validate_args: mutual exclusion ----------------------------------

def test_validate_args_rejects_mutually_exclusive_flags(tmp_path):
    tf = tmp_path / "topics.txt"
    qf = tmp_path / "qs.txt"
    tf.write_text("Topic A\n")
    qf.write_text("¿Pregunta?\n")
    import argparse
    args = argparse.Namespace(
        chunk_overlap=350, chunk_size=3500, num_topics=8, questions_per_topic=6,
        temperature=0.2, quality_gate="strict",
        topics_file=tf, questions_file=qf,
        retrieval="lexical", embedding_model="gemma4:e4b",
    )
    with pytest.raises(SystemExit):
        gd.validate_args(args)


def test_validate_args_rejects_missing_topics_file(tmp_path):
    import argparse
    args = argparse.Namespace(
        chunk_overlap=350, chunk_size=3500, num_topics=8, questions_per_topic=6,
        temperature=0.2, quality_gate="strict",
        topics_file=tmp_path / "nonexistent.txt", questions_file=None,
        retrieval="lexical", embedding_model="gemma4:e4b",
    )
    with pytest.raises(SystemExit):
        gd.validate_args(args)


def test_validate_args_rejects_empty_embedding_model_for_hybrid(tmp_path):
    import argparse
    args = argparse.Namespace(
        chunk_overlap=350, chunk_size=3500, num_topics=8, questions_per_topic=6,
        temperature=0.2, quality_gate="strict",
        topics_file=None, questions_file=None,
        retrieval="hybrid", embedding_model="",
    )
    with pytest.raises(SystemExit):
        gd.validate_args(args)


# =============================================================================
# New tests: encoding normalisation, cross-chunk detection, judge pre-checks,
# verbatim answer detection, and composite audit keys.
# =============================================================================

# ---------------------------------------------------------------------------
# Group A — encoding
# ---------------------------------------------------------------------------

def test_normalize_encoding_fixes_mojibake_map_fallback(monkeypatch):
    """normalize_encoding corrects apostrophe mojibake via the fallback map."""
    monkeypatch.setattr("engine._text._FTFY_AVAILABLE", False)
    # â€™ = U+00E2 + U+20AC + U+2122 (TM sign used as 0x99 in Windows-1252)
    mojibake_apostrophe = "â€™"
    dirty = f"The paper{mojibake_apostrophe}s findings"
    result = gd.normalize_encoding(dirty)
    assert mojibake_apostrophe not in result
    assert "paper" in result  # surrounding text preserved


def test_has_quality_artifact_rejects_mojibake_in_context_source():
    """has_quality_artifact returns True for text containing the â€ mojibake prefix."""
    # â€ = U+00E2 + U+20AC — the universal prefix for most mojibake sequences.
    mojibake = "â€™"  # â€™ variant
    item = {
        "question": "What does the ratio measure?",
        "answer": "Downside risk only.",
        "context_source": f"The Sortino {mojibake}s formula penalises losses.",
        "topic": "Risk",
    }
    assert gd.has_quality_artifact(item) is True


def test_has_quality_artifact_clean_item_passes():
    """has_quality_artifact returns False for clean ASCII/UTF-8 text."""
    item = {
        "question": "What is the Sharpe ratio?",
        "answer": "A measure of risk-adjusted return.",
        "context_source": "The Sharpe ratio divides excess return by the standard deviation.",
        "topic": "Risk metrics",
    }
    assert gd.has_quality_artifact(item) is False


# ---------------------------------------------------------------------------
# Group B — cross-chunk detection
# ---------------------------------------------------------------------------

def test_has_quality_artifact_rejects_internal_chunk_marker():
    """has_quality_artifact returns True when context_source has a chunk marker."""
    item = {
        "question": "q",
        "answer": "a",
        "context_source": "First sentence. [doc-chunk-0002] Second sentence from another chunk.",
        "topic": "T",
    }
    assert gd.has_quality_artifact(item) is True


def test_audit_item_quality_returns_cross_chunk_context():
    """audit_item_quality returns 'cross_chunk_context' for marker inside context_source."""
    item = {
        "question": "q",
        "answer": "a",
        "context_source": "First. [paper-chunk-0003] Second.",
        "context_source_verified": True,
        "topic": "T",
    }
    reason = gd.audit_item_quality(item)
    assert reason == "cross_chunk_context"


def test_source_chunk_ids_for_fragment_detects_multispan():
    """source_chunk_ids_for_fragment returns both chunk IDs when the fragment contains a marker.

    Literal-substring search cannot cross a chunk marker unless the marker text is part of the
    fragment itself (i.e. the LLM copied the marker into context_source).  That fragment is
    already rejected by has_quality_artifact / cross_chunk_context, but the ID attribution must
    still cover both chunks so rejected-row metadata is correct.
    """
    context = (
        "[doc-chunk-0001] First chunk content. "
        "[doc-chunk-0002] Second chunk content."
    )
    # Fragment that includes the marker spans both chunk ranges.
    fragment = "First chunk content. [doc-chunk-0002] Second chunk content."
    ids = gd.source_chunk_ids_for_fragment(context, fragment)
    assert "doc-chunk-0001" in ids
    assert "doc-chunk-0002" in ids


def test_source_chunk_ids_for_fragment_single_chunk():
    """source_chunk_ids_for_fragment returns one ID when fragment is in one chunk."""
    context = (
        "[doc-chunk-0001] First chunk content. "
        "[doc-chunk-0002] Second chunk with risk metrics here."
    )
    fragment = "risk metrics"
    ids = gd.source_chunk_ids_for_fragment(context, fragment)
    assert ids == ["doc-chunk-0002"]


# ---------------------------------------------------------------------------
# Group C — judge pre-checks and threshold
# ---------------------------------------------------------------------------

def test_normalize_judge_result_min_score_threshold_is_0_6():
    """answer_support = 0.55 (< 0.6) must force decision to fail."""
    result = gd.normalize_judge_result(
        {
            "context_quality": 0.9,
            "answer_support": 0.55,
            "question_quality": 0.9,
            "overall_score": 0.78,
            "decision": "pass",
            "reasons": ["factual"],
            "explanation": "Borderline answer support.",
        },
        model="judge",
    )
    assert result["judge_decision"] == "fail"
    assert result["judge_score"] <= 0.39


def test_judge_item_precheck_fails_mojibake_without_llm_call(monkeypatch):
    """A context_source with mojibake triggers fail in pre-checks without calling Ollama."""
    llm_calls = {"n": 0}

    def counting_fake_ollama(**kw):
        llm_calls["n"] += 1
        return ({}, "")

    monkeypatch.setattr("engine._judge.call_ollama_json", counting_fake_ollama)
    mojibake = "â€™"  # â€™
    result = gd.judge_item(
        {
            "question": "What is measured?",
            "answer": "Downside risk.",
            "context_source": f"The ratio{mojibake}s approach penalises losses.",
            "topic": "Risk",
        },
        model="judge-model",
        temperature=0.0,
    )
    assert llm_calls["n"] == 0, "LLM should not be called when pre-check triggers"
    assert result["judge_decision"] == "fail"
    assert "extraction_artifact" in result["judge_reasons"]


def test_judge_item_precheck_flags_internal_chunk_marker(monkeypatch):
    """A chunk-boundary marker inside context_source causes judge pre-check to fail."""
    monkeypatch.setattr("engine._judge.call_ollama_json", lambda **kw: ({}, ""))
    result = gd.judge_item(
        {
            "question": "q",
            "answer": "a",
            "context_source": "Text. [doc-chunk-0002] More text from next chunk.",
            "topic": "T",
        },
        model="judge-model",
        temperature=0.0,
    )
    assert result["judge_decision"] == "fail"


# ---------------------------------------------------------------------------
# Group D — verbatim answer detection
# ---------------------------------------------------------------------------

def test_has_verbatim_answer_detects_copy_paste():
    """has_verbatim_answer returns True when answer nearly copies context_source."""
    context = (
        "The Sortino ratio focuses exclusively on downside deviation, "
        "penalising only negative returns below a target threshold."
    )
    item = {
        "question": "What does the Sortino ratio focus on?",
        "answer": "The Sortino ratio focuses exclusively on downside deviation, "
                  "penalising only negative returns below a target threshold.",
        "context_source": context,
    }
    assert gd.has_verbatim_answer(item) is True


def test_has_verbatim_answer_accepts_short_reformulated_answer():
    """has_verbatim_answer returns False for a short reformulated answer."""
    context = (
        "These findings suggest that traditional technical indicators may have "
        "diminishing predictive value in modern high-frequency markets."
    )
    item = {
        "question": "What did findings suggest about traditional indicators?",
        "answer": "They may lose predictive power in fast modern markets.",
        "context_source": context,
    }
    assert gd.has_verbatim_answer(item) is False


def test_has_verbatim_answer_detects_short_copied_phrase():
    context = (
        "The trade-off between controllability and efficiency leads to the question: "
        "does an ideal action space which strikes a balance between these aspects exist?"
    )
    item = {
        "question": "What is the primary trade-off?",
        "answer": "The trade-off is between controllability and efficiency.",
        "context_source": context,
    }
    assert gd.has_verbatim_answer(item) is True


def test_apply_quality_gate_rejects_verbatim_answer():
    """apply_quality_gate rejects a near-verbatim answer with 'verbatim_answer' reason."""
    context = (
        "Technical indicators such as Bollinger Bands provide dynamic support "
        "and resistance levels that adapt to recent market volatility patterns."
    )
    item = {
        "id": "verbatim-1",
        "question": "What do Bollinger Bands provide?",
        "answer": (
            "Technical indicators such as Bollinger Bands provide dynamic support "
            "and resistance levels that adapt to recent market volatility patterns."
        ),
        "context_source": context,
        "context_source_verified": True,
        "topic": "Technical Analysis",
        "type": "factual",
    }
    accepted, rejected, stats = gd.apply_quality_gate([item], "strict")
    assert accepted == []
    assert rejected[0]["rejection_reason"] == "verbatim_answer"


# ---------------------------------------------------------------------------
# Group E — composite audit keys
# ---------------------------------------------------------------------------

def test_build_dataset_audit_uses_doc_topic_composite_key():
    """accepted_by_topic uses '{document}::{topic_id}' as key to prevent cross-doc collisions."""
    accepted = [
        {"document": "doc_a.pdf", "topic_id": "topic-00"},
        {"document": "doc_b.pdf", "topic_id": "topic-00"},  # same topic_id, different doc
    ]
    audit = gd.build_dataset_audit(
        accepted,
        [],
        train_rows=accepted,
        val_rows=[],
        test_rows=[],
    )
    assert "doc_a.pdf::topic-00" in audit["accepted_by_topic"]
    assert "doc_b.pdf::topic-00" in audit["accepted_by_topic"]
    assert audit["accepted_by_topic"]["doc_a.pdf::topic-00"] == 1
    assert audit["accepted_by_topic"]["doc_b.pdf::topic-00"] == 1
