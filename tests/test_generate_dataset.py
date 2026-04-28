"""Unit tests for pure functions in generate_dataset."""
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


def test_clean_markdown_artifacts_removes_picture_and_ref_noise():
    text = "Virtual memory in the 80 computer. == picture 188 x 297 intentionally omitted <== OS/360 stayed."
    cleaned = gd.clean_markdown_artifacts(text)
    assert "picture" not in cleaned
    assert "80 computer" not in cleaned
    assert "OS/360" in cleaned


def test_clean_generated_text_capitalizes_and_removes_numeric_artifact():
    cleaned = gd.clean_generated_text("swapping helps the 80 computer avoid limits Ã¢â‚¬â€ safely.")
    assert cleaned.startswith("Swapping")
    assert "80 computer" not in cleaned
    assert "Ã¢" not in cleaned


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
    assert items[0]["context_source"] == "schedules processes"
    assert items[0]["document_language"] == "es"
    assert items[0]["difficulty"] == "hard"
    assert items[0]["source_chunk_ids"] == []


def test_find_verified_context_source_repairs_from_answer_overlap():
    context = (
        "[doc-chunk-0001] The kernel schedules processes and allocates memory fairly. "
        "File systems store persistent data."
    )
    source, verified = gd.find_verified_context_source(
        "not a literal source",
        "The kernel schedules processes and allocates memory fairly.",
        context,
    )
    assert verified is True
    assert source == "[doc-chunk-0001] The kernel schedules processes and allocates memory fairly."


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
