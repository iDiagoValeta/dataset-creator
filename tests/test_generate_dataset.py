"""Unit tests for pure functions in generate_dataset."""
from pathlib import Path

import pytest

import generate_dataset as gd


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

    monkeypatch.setattr(gd, "call_ollama_json", fake_call)
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
    assert items[0]["context_source"] == "schedules processes"
    assert items[0]["difficulty"] == "hard"


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

    monkeypatch.setattr(gd, "call_ollama_json", fake_call)
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
