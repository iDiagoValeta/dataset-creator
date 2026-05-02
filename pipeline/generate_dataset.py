"""
DatasetCreator -- Topic-aware synthetic dataset generation from PDF documents.

The pipeline is optimized for local execution with Ollama and long-context models:
    1) Extract text from PDFs (text only; no images/tables processing)
    2) Build a global topic map per document using large context windows
    3) Generate Q/A items per topic while avoiding duplicates
    4) Export structured JSONL datasets and train/val/test splits

Usage:
    python pipeline/generate_dataset.py
    python pipeline/generate_dataset.py --model gemma4:e4b --questions-per-topic 8
    python pipeline/generate_dataset.py --source-dir pipeline/input --output pipeline/output/tesis.jsonl

Dependencies:
    - ollama
    - pymupdf4llm (optional, preferred PDF extraction)
    - pypdf (fallback extraction)
"""

# Re-export all public symbols from engine submodules so that callers and tests
# can continue to use `import generate_dataset as gd; gd.Chunk`, `gd.parse_split`, etc.
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import ollama  # noqa: F401 — exposed as gd.ollama for test monkeypatching
from engine._cli import (  # noqa: F401
    _checkpoint_path,
    _run_dry_run,
    build_arg_parser,
    filter_pdfs_by_only_doc,
    list_pdf_files,
    load_checkpoint_items,
    save_checkpoint_items,
    validate_args,
)
from engine._config import (  # noqa: F401
    _STOPWORDS_SET,
    BASE_DIR,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_DEBUG_DIR,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_JUDGE_MODE,
    DEFAULT_JUDGE_MODEL,
    DEFAULT_LANGUAGE,
    DEFAULT_MAX_DOC_CONTEXT_CHARS,
    DEFAULT_MAX_TOPIC_CONTEXT_CHARS,
    DEFAULT_MODEL,
    DEFAULT_NUM_TOPICS,
    DEFAULT_OLLAMA_RETRIES,
    DEFAULT_OLLAMA_RETRY_BACKOFF,
    DEFAULT_OLLAMA_TIMEOUT,
    DEFAULT_OUTPUT_PATH,
    DEFAULT_QUALITY_GATE,
    DEFAULT_QUESTIONS_PER_TOPIC,
    DEFAULT_RETRIEVAL_MODE,
    DEFAULT_SEED,
    DEFAULT_SOURCE_DIR,
    DEFAULT_SPLIT,
    DEFAULT_TEMPERATURE,
    LANGUAGE_MARKERS,
    LANGUAGE_NAMES,
    MIN_TOPIC_CONTEXT_CHARS,
    STOPWORDS,
    VALID_DIFFICULTIES,
    VALID_JUDGE_MODES,
    VALID_QUALITY_GATES,
    VALID_TYPES,
    Chunk,
    Topic,
    logger,
)
from engine._export import (  # noqa: F401
    _assert_safe_cleanup_root,
    _current_git_commit,
    _package_version,
    _remove_empty_dirs,
    build_dataset_audit,
    build_reproducibility_info,
    clean_generated_artifacts,
    split_rows,
    write_jsonl,
    write_metadata,
)
from engine._generation import generate_items_for_topic  # noqa: F401
from engine._judge import (  # noqa: F401
    KNOWN_JUDGE_REASONS,
    VALID_JUDGE_DECISIONS,
    audit_items_with_judge,
    build_judge_messages,
    build_judge_stats,
    judge_error_result,
    judge_item,
    normalize_judge_result,
)
from engine._ollama import (  # noqa: F401
    _ollama_entry_model_name,
    _ollama_list_models_entries,
    call_ollama_json,
    get_chunk_embedding,
    get_topic_embedding,
    score_chunk_for_topic_semantic,
    verify_ollama_model,
)
from engine._pdf import (  # noqa: F401
    PYMUPDF_AVAILABLE,
    build_chunks_from_text,
    chunk_text,
    extract_text_from_pdf,
)
from engine._prompts import (  # noqa: F401
    build_representative_document_context,
    build_seed_question_messages,
    build_topic_generation_messages,
    build_topic_generation_messages_compact,
    build_topic_map_messages,
    build_topic_map_messages_compact,
    try_parse_json_payload,
)
from engine._quality import (  # noqa: F401
    _content_words,
    _question_bigrams,
    apply_quality_gate,
    audit_item_quality,
    clean_context_artifacts,
    context_excerpt_for_fragment,
    deduplicate_items,
    find_verified_context_source,
    has_insufficient_context_support,
    has_quality_artifact,
    has_topic_mismatch,
    has_verbatim_answer,
    is_circular_answer,
    source_chunk_ids_for_fragment,
)
from engine._text import (  # noqa: F401
    _sample_existing_questions,
    clean_generated_text,
    clean_markdown_artifacts,
    cosine_similarity,
    deduplicate_preserve_order,
    detect_document_language,
    extract_keywords,
    normalize_domain_terms,
    normalize_encoding,
    normalize_whitespace,
    now_iso,
    parse_split,
    resolve_generation_language,
    sanitize_question,
    strip_accents_ascii,
    strip_non_content_tail_sections,
    truncate_text,
)
from engine._topics import (  # noqa: F401
    GENERIC_TOPIC_TERMS,
    YAML_AVAILABLE,
    _extract_topics_candidates,
    build_fallback_topics_from_chunks,
    build_section_topics_from_text,
    build_topic_context,
    extract_section_headings,
    infer_topic_name_from_chunk,
    is_valid_topic_name,
    load_questions_file,
    load_topics_file,
    parse_topics,
    questions_to_topics,
    score_chunk_for_topic,
    topics_are_too_generic,
    topics_mostly_invalid,
)


def audit_items_with_judge_by_document(
    items: list[dict[str, Any]],
    model: str,
    temperature: float,
    seed: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Audit accepted rows one document at a time and aggregate judge stats."""
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in items:
        groups[str(row.get("document", "__unknown__"))].append(row)

    judged_rows: list[dict[str, Any]] = []
    for document, document_items in groups.items():
        print(f"[JUDGE] {document}: auditando {len(document_items)} item(s) con {model}...")
        document_judged, _ = audit_items_with_judge(
            document_items,
            model=model,
            temperature=temperature,
            seed=seed,
        )
        judged_rows.extend(document_judged)

    return judged_rows, build_judge_stats("audit", model, judged_rows)


def infer_resume_counts(
    debug_dir: Path,
    pdf_path: Path,
    checkpoint_items: list[dict[str, Any]],
) -> tuple[int, int]:
    """Infer chunk/topic counts from per-document debug metadata during resume."""
    debug_path = debug_dir / f"{pdf_path.stem}.json"
    debug_payload: dict[str, Any] = {}
    if debug_path.exists():
        try:
            debug_payload = json.loads(debug_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            debug_payload = {}

    chunk_count = int(debug_payload.get("chunk_count") or 0)
    if chunk_count <= 0:
        chunk_ids = {
            str(chunk_id)
            for item in checkpoint_items
            for chunk_id in item.get("source_chunk_ids", [])
            if str(chunk_id).strip()
        }
        chunk_count = len(chunk_ids)

    debug_topics = debug_payload.get("topics", [])
    topic_count = len(debug_topics) if isinstance(debug_topics, list) else 0
    if topic_count <= 0:
        topic_count = len({
            str(item.get("topic_id", ""))
            for item in checkpoint_items
            if str(item.get("topic_id", "")).strip()
        })

    return chunk_count, topic_count


def main() -> None:
    """Run end-to-end dataset generation pipeline."""
    start_time = time.time()
    args = build_arg_parser().parse_args()
    validate_args(args)

    user_topics: list[Topic] | None = None
    seed_question_mode_active = False
    if getattr(args, "topics_file", None):
        user_topics = load_topics_file(args.topics_file)
        print(f"[INFO] {len(user_topics)} topic(s) cargados desde {args.topics_file}")
    elif getattr(args, "questions_file", None):
        seed_qs = load_questions_file(args.questions_file)
        user_topics = questions_to_topics(seed_qs)
        seed_question_mode_active = True
        print(f"[INFO] Modo seed-questions: {len(user_topics)} pregunta(s) desde {args.questions_file}")

    if args.clean or args.clean_dry_run:
        stats = clean_generated_artifacts(
            output_path=args.output,
            debug_dir=args.debug_dir,
            dry_run=args.clean_dry_run,
        )
        action = "Archivos que se borrarian" if args.clean_dry_run else "Archivos borrados"
        print(f"{action}: {stats['files']}")
        if not args.clean_dry_run:
            print(f"Directorios vacios borrados: {stats['dirs']}")
        return

    split = parse_split(args.split)

    pdf_files = list_pdf_files(args.source_dir)
    if not pdf_files:
        raise RuntimeError(f"No se encontraron PDFs en {args.source_dir}")

    pdf_files = filter_pdfs_by_only_doc(pdf_files, args.only_doc)

    if args.dry_run:
        _run_dry_run(args, pdf_files, user_topics=user_topics, seed_question_mode=seed_question_mode_active)
        return

    if not args.skip_model_check:
        verify_ollama_model(args.model)
        if getattr(args, "retrieval", "lexical") in {"semantic", "hybrid"}:
            verify_ollama_model(args.embedding_model)
        if getattr(args, "judge", "off") == "audit":
            verify_ollama_model(args.judge_model)

    args.debug_dir.mkdir(parents=True, exist_ok=True)

    generated: list[dict[str, Any]] = []
    topic_generation_records: list[dict[str, Any]] = []
    total_chunks = 0
    total_topics = 0
    resumed_docs = 0
    document_languages: dict[str, str] = {}

    for pdf_index, pdf_path in enumerate(pdf_files, start=1):
        print(f"[{pdf_index}/{len(pdf_files)}] Procesando documento: {pdf_path.name}")

        if args.resume:
            cached = load_checkpoint_items(args.debug_dir, pdf_path)
            if cached:
                generated.extend(cached)
                cached_language = str(cached[0].get("document_language", "unknown"))
                document_languages[pdf_path.name] = cached_language
                resumed_chunk_count, resumed_topic_count = infer_resume_counts(args.debug_dir, pdf_path, cached)
                total_chunks += resumed_chunk_count
                total_topics += resumed_topic_count
                resumed_docs += 1
                print(f"  - Resume: {len(cached)} items cargados desde checkpoint; salto generacion.")
                continue

        doc_items: list[dict[str, Any]] = []
        raw_text = extract_text_from_pdf(pdf_path)
        full_context = truncate_text(raw_text, args.max_doc_context_chars)
        generation_language, document_language, language_scores = resolve_generation_language(args.language, raw_text)
        document_languages[pdf_path.name] = document_language

        chunks = build_chunks_from_text(
            raw_text=raw_text,
            document_name=pdf_path.name,
            document_stem=pdf_path.stem,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            max_chunks=args.max_chunks,
        )
        if not chunks:
            print(f"  - Saltado: sin chunks validos en {pdf_path.name}")
            continue
        total_chunks += len(chunks)
        print(f"  - Idioma: {document_language}")

        existing_questions = [str(item["question"]) for item in generated if item.get("document") == pdf_path.name]

        if user_topics is not None:
            topics = user_topics
            doc_debug = {
                "document": pdf_path.name,
                "language": document_language,
                "language_prompt": generation_language,
                "language_scores": language_scores,
                "chunk_count": len(chunks),
                "topic_map_strategy": "user_file",
                "topic_map_payload": {
                    "topics": [
                        {"name": t.name, "summary": t.summary, "keywords": t.keywords}
                        for t in topics
                    ]
                },
                "topic_map_raw_content": None,
                "topic_map_attempts": [],
                "topics": [],
            }
        else:
            topic_messages = build_topic_map_messages(
                document=pdf_path.name,
                full_text=full_context,
                language=generation_language,
                num_topics=args.num_topics,
            )
            topics_payload, topics_raw_content = call_ollama_json(
                model=args.model,
                messages=topic_messages,
                temperature=args.temperature,
                seed=args.seed,
            )
            topics = parse_topics(topics_payload)[: args.num_topics]

            doc_debug = {
                "document": pdf_path.name,
                "language": document_language,
                "language_prompt": generation_language,
                "language_scores": language_scores,
                "chunk_count": len(chunks),
                "topic_map_payload": topics_payload,
                "topic_map_raw_content": topics_raw_content,
                "topic_map_strategy": "llm",
                "topic_map_attempts": [
                    {
                        "attempt": 1,
                        "strategy": "full_document",
                        "context_chars": len(full_context),
                        "raw_content": topics_raw_content,
                        "parsed_topics": len(topics),
                    }
                ],
                "topics": [],
            }

            if not topics:
                compact_context = build_representative_document_context(
                    chunks=chunks,
                    max_chars=min(args.max_doc_context_chars, 18000),
                )
                compact_messages = build_topic_map_messages_compact(
                    document=pdf_path.name,
                    sampled_text=compact_context,
                    language=generation_language,
                    num_topics=args.num_topics,
                )
                topics_payload2, topics_raw_content2 = call_ollama_json(
                    model=args.model,
                    messages=compact_messages,
                    temperature=args.temperature,
                    seed=args.seed,
                )
                topics = parse_topics(topics_payload2)[: args.num_topics]
                doc_debug["topic_map_attempts"].append(
                    {
                        "attempt": 2,
                        "strategy": "representative_excerpts",
                        "context_chars": len(compact_context),
                        "raw_content": topics_raw_content2,
                        "parsed_topics": len(topics),
                    }
                )
                if topics:
                    doc_debug["topic_map_strategy"] = "llm_compact"
                    doc_debug["topic_map_payload"] = topics_payload2
                    doc_debug["topic_map_raw_content"] = topics_raw_content2

            if not topics:
                logger.warning(
                    "Topicos no parseables para '%s'. Raw (300 chars): %s",
                    pdf_path.name,
                    doc_debug["topic_map_attempts"][-1]["raw_content"][:300]
                    if doc_debug["topic_map_attempts"][-1]["raw_content"]
                    else "(vacio)",
                )
                chunk_topics = build_fallback_topics_from_chunks(chunks=chunks, num_topics=args.num_topics)
                section_topics = build_section_topics_from_text(raw_text, args.num_topics)
                if section_topics:
                    topics = section_topics
                    doc_debug["topic_map_strategy"] = "fallback_section_headings"
                    print(
                        f"  - Aviso: el modelo no devolvio topicos parseables para {pdf_path.name}; "
                        f"usando secciones del documento ({len(topics)} topicos)."
                    )
                else:
                    topics = chunk_topics
                    doc_debug["topic_map_strategy"] = "fallback_from_chunks"
                    print(
                        f"  - Aviso: el modelo no devolvio topicos parseables para {pdf_path.name}; "
                        f"usando fallback de chunks ({len(topics)} topicos)."
                    )
                doc_debug["topic_map_payload"] = {
                    "topics": [
                        {"name": t.name, "summary": t.summary, "keywords": t.keywords}
                        for t in topics
                    ]
                }
            elif topics_are_too_generic(topics) or topics_mostly_invalid(topics):
                section_topics = build_section_topics_from_text(raw_text, args.num_topics)
                if section_topics:
                    doc_debug["llm_topics_replaced"] = [
                        {"name": t.name, "summary": t.summary, "keywords": t.keywords}
                        for t in topics
                    ]
                    topics = section_topics
                    doc_debug["topic_map_strategy"] = f"{doc_debug['topic_map_strategy']}_section_refined"
                    doc_debug["topic_map_payload"] = {
                        "topics": [
                            {"name": t.name, "summary": t.summary, "keywords": t.keywords}
                            for t in topics
                        ]
                    }
                    print("  - Aviso: topicos LLM invalidos o genericos; usando secciones del documento.")

            if not topics:
                print(f"  - Saltado: no se pudieron construir topicos para {pdf_path.name}")
                debug_path = args.debug_dir / f"{pdf_path.stem}.json"
                write_metadata(debug_path, doc_debug)
                continue

        total_topics += len(topics)
        print(f"  - Topicos detectados: {len(topics)}")
        chunk_embedding_cache: dict[str, list[float]] = {}
        for topic_idx, topic in enumerate(topics, start=1):
            print(f"    [{topic_idx}/{len(topics)}] {topic.name[:60]}", end="", flush=True)
            if not seed_question_mode_active and not is_valid_topic_name(topic.name):
                print(" -> 0 items (nombre de topico invalido, saltado)")
                continue
            topic_context = build_topic_context(
                chunks=chunks,
                topic=topic,
                max_topic_context_chars=args.max_topic_context_chars,
                retrieval_mode=getattr(args, "retrieval", "lexical"),
                embedding_model=getattr(args, "embedding_model", DEFAULT_EMBEDDING_MODEL),
                embedding_cache=chunk_embedding_cache,
            )
            if not topic_context.strip():
                print(" -> 0 items (sin contexto)")
                continue
            if (
                args.quality_gate != "strict"
                and len(topic_context) < MIN_TOPIC_CONTEXT_CHARS
                and len(full_context) > len(topic_context)
            ):
                topic_context = truncate_text(full_context, min(args.max_topic_context_chars, len(full_context)))

            topic_t0 = time.time()
            topic_items, topic_debug = generate_items_for_topic(
                model=args.model,
                document=pdf_path.name,
                topic=topic,
                topic_context=topic_context,
                language=generation_language,
                questions_per_topic=args.questions_per_topic,
                temperature=args.temperature,
                existing_questions=existing_questions,
                seed=args.seed,
                document_language=document_language,
                seed_question_mode=seed_question_mode_active,
            )
            topic_elapsed = time.time() - topic_t0
            generated.extend(topic_items)
            doc_items.extend(topic_items)
            existing_questions.extend(item["question"] for item in topic_items)
            topic_generation_records.append(
                {
                    "document": pdf_path.name,
                    "pdf_path": pdf_path,
                    "topic": topic,
                    "topic_context": topic_context,
                    "language": generation_language,
                    "document_language": document_language,
                }
            )
            print(f" -> {len(topic_items)} items ({topic_elapsed:.1f}s)")
            doc_debug["topics"].append(
                {
                    "topic_id": topic.topic_id,
                    "topic_name": topic.name,
                    "topic_keywords": topic.keywords,
                    "topic_context_chars": len(topic_context),
                    "generated_items": len(topic_items),
                    "raw": topic_debug,
                }
            )

        debug_path = args.debug_dir / f"{pdf_path.stem}.json"
        write_metadata(debug_path, doc_debug)
        save_checkpoint_items(args.debug_dir, pdf_path, doc_items)

    quality_candidates, rejected_rows, quality_stats = apply_quality_gate(generated, args.quality_gate)
    if not seed_question_mode_active and topic_generation_records:
        min_items_per_topic = 2
        accepted_by_topic: dict[str, int] = {}
        for item in quality_candidates:
            topic_id = str(item.get("topic_id", ""))
            accepted_by_topic[topic_id] = accepted_by_topic.get(topic_id, 0) + 1

        backfill_items: list[dict[str, Any]] = []
        for record_idx, record in enumerate(topic_generation_records):
            topic = record["topic"]
            accepted_count = accepted_by_topic.get(topic.topic_id, 0)
            if accepted_count >= min_items_per_topic:
                continue
            needed = min(args.questions_per_topic, min_items_per_topic - accepted_count + 2)
            existing_questions = [
                str(item.get("question", ""))
                for item in generated + backfill_items
                if item.get("document") == record["document"]
            ]
            print(
                f"[BACKFILL] {record['document']} / {topic.name[:60]}: "
                f"{accepted_count} aceptados; generando {needed}."
            )
            extra_items, _ = generate_items_for_topic(
                model=args.model,
                document=record["document"],
                topic=topic,
                topic_context=record["topic_context"],
                language=record["language"],
                questions_per_topic=needed,
                temperature=args.temperature,
                existing_questions=existing_questions,
                seed=args.seed + 1000 + record_idx,
                document_language=record["document_language"],
                id_offset=args.questions_per_topic,
            )
            backfill_items.extend(extra_items)

        quality_stats["backfill_generated_items"] = len(backfill_items)
        if backfill_items:
            generated.extend(backfill_items)
            affected_pdf_paths = {
                record["pdf_path"]
                for record in topic_generation_records
                if any(item.get("document") == record["document"] for item in backfill_items)
            }
            for affected_pdf_path in affected_pdf_paths:
                save_checkpoint_items(
                    args.debug_dir,
                    affected_pdf_path,
                    [item for item in generated if item.get("document") == affected_pdf_path.name],
                )
            quality_candidates, rejected_rows, quality_stats = apply_quality_gate(generated, args.quality_gate)
            quality_stats["backfill_generated_items"] = len(backfill_items)

    deduped = deduplicate_items(quality_candidates)
    quality_stats["accepted_items_before_dedup"] = len(quality_candidates)
    quality_stats["verified_items_before_dedup"] = quality_stats["verified_items"]
    quality_stats["duplicate_items_removed_after_quality"] = len(quality_candidates) - len(deduped)
    quality_stats["accepted_items"] = len(deduped)
    quality_stats["verified_items"] = sum(1 for item in deduped if item.get("context_source_verified"))
    train_rows, val_rows, test_rows = split_rows(deduped, split=split, seed=args.seed)
    expected_topic_ids = [str(record["topic"].topic_id) for record in topic_generation_records]
    dataset_audit = build_dataset_audit(
        deduped,
        rejected_rows,
        train_rows,
        val_rows,
        test_rows,
        expected_topic_ids=expected_topic_ids,
    )

    write_jsonl(args.output, deduped)
    output_base = args.output.with_suffix("")
    write_jsonl(Path(f"{output_base}_train.jsonl"), train_rows)
    write_jsonl(Path(f"{output_base}_val.jsonl"), val_rows)
    write_jsonl(Path(f"{output_base}_test.jsonl"), test_rows)
    write_jsonl(Path(f"{output_base}.rejected.jsonl"), rejected_rows)

    judge_stats = build_judge_stats(getattr(args, "judge", "off"), args.judge_model, [])
    if getattr(args, "judge", "off") == "audit":
        judged_rows, judge_stats = audit_items_with_judge_by_document(
            deduped,
            model=args.judge_model,
            temperature=0.0,
            seed=args.seed,
        )
        write_jsonl(Path(f"{output_base}.judged.jsonl"), judged_rows)

    verified_count = sum(1 for item in deduped if item.get("context_source_verified"))
    metadata = {
        "created_at": now_iso(),
        "model": args.model,
        "language": args.language,
        "document_languages": document_languages,
        "source_dir": str(args.source_dir),
        "output": str(args.output),
        "pdf_count": len(pdf_files),
        "chunk_count": total_chunks,
        "topic_count": total_topics,
        "generated_items": len(generated),
        "deduplicated_items": len(deduped),
        "accepted_items": len(deduped),
        "rejected_items": len(rejected_rows),
        "context_source_verified_items": verified_count,
        "quality": quality_stats,
        "judge": judge_stats,
        "audit": dataset_audit,
        "resumed_documents": resumed_docs,
        "split": {"train": len(train_rows), "val": len(val_rows), "test": len(test_rows)},
        "params": {
            "chunk_size": args.chunk_size,
            "chunk_overlap": args.chunk_overlap,
            "num_topics": args.num_topics,
            "questions_per_topic": args.questions_per_topic,
            "temperature": args.temperature,
            "seed": args.seed,
            "max_doc_context_chars": args.max_doc_context_chars,
            "max_topic_context_chars": args.max_topic_context_chars,
            "quality_gate": args.quality_gate,
            "only_doc": args.only_doc,
            "resume": args.resume,
            "retrieval": args.retrieval,
            "embedding_model": args.embedding_model,
            "judge": args.judge,
            "judge_model": args.judge_model,
        },
        "runtime": build_reproducibility_info(),
    }
    write_metadata(args.output.with_suffix(".meta.json"), metadata)

    print("")
    if not deduped:
        print("[AVISO] El dataset final quedo vacio. Revisar:")
        print("  - Logs en run_logs/ por cada PDF (topic_map_strategy, attempts).")
        print("  - Que el modelo Ollama devuelve JSON valido para este idioma.")
        print("  - Que los PDFs contienen texto extraible (no solo imagenes).")
        print(f"  - Items generados: {len(generated)} | aceptados pre-dedup: {len(quality_candidates)}")
    else:
        print("Dataset generado correctamente.")
    print(f"- Total items: {len(deduped)}")
    print(f"- Items rechazados por quality gate: {len(rejected_rows)}")
    print(f"- Archivo principal: {args.output}")
    print(f"- Rechazados: {Path(f'{output_base}.rejected.jsonl')}")
    if getattr(args, "judge", "off") == "audit":
        print(f"- Auditados por juez: {Path(f'{output_base}.judged.jsonl')}")
    print(f"- Splits: train={len(train_rows)} val={len(val_rows)} test={len(test_rows)}")
    print(f"- context_source verificado (substring literal): {verified_count}/{len(deduped)}")
    if dataset_audit["warnings"]:
        print(f"- Avisos de auditoria: {len(dataset_audit['warnings'])}")
    if args.resume:
        print(f"- Documentos reanudados desde checkpoint: {resumed_docs}")
    print(f"- Metadata: {args.output.with_suffix('.meta.json')}")
    print(f"- Tiempo total: {time.time() - start_time:.1f}s")


if __name__ == "__main__":
    main()
