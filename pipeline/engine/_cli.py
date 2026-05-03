"""CLI argument parser, validation, PDF filtering, checkpoints, and dry-run."""

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from engine._config import (
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
    DEFAULT_OUTPUT_PATH,
    DEFAULT_QUALITY_GATE,
    DEFAULT_QUESTIONS_PER_TOPIC,
    DEFAULT_RETRIEVAL_MODE,
    DEFAULT_SEED,
    DEFAULT_SOURCE_DIR,
    DEFAULT_SPLIT,
    DEFAULT_TEMPERATURE,
    VALID_JUDGE_MODES,
    VALID_QUALITY_GATES,
    Topic,
    logger,
)
from engine._export import write_jsonl
from engine._pdf import build_chunks_from_text, extract_text_from_pdf
from engine._text import detect_document_language


def list_pdf_files(source_dir: Path) -> list[Path]:
    """Return sorted PDF files from source directory."""
    if not source_dir.exists():
        raise FileNotFoundError(f"No existe carpeta de PDFs: {source_dir}")
    files = sorted(p for p in source_dir.iterdir() if p.is_file() and p.suffix.lower() == ".pdf")
    return files


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    parser = argparse.ArgumentParser(description="Generate synthetic QA datasets from PDFs using Ollama.")
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR, help="Folder with PDF files.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH, help="Output JSONL path.")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Ollama model to generate QA pairs.")
    parser.add_argument(
        "--language",
        type=str,
        default=DEFAULT_LANGUAGE,
        help="Output language. Default: auto (detect per PDF). Use es/en/ca/etc. to force one.",
    )
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE, help="Chunk size in characters.")
    parser.add_argument("--chunk-overlap", type=int, default=DEFAULT_CHUNK_OVERLAP, help="Chunk overlap in chars.")
    parser.add_argument(
        "--num-topics",
        type=int,
        default=DEFAULT_NUM_TOPICS,
        help="Maximum number of topics to generate per document.",
    )
    parser.add_argument(
        "--questions-per-topic",
        type=int,
        default=DEFAULT_QUESTIONS_PER_TOPIC,
        help="How many Q/A items to request for each topic.",
    )
    parser.add_argument("--split", type=str, default=DEFAULT_SPLIT, help="train,val,test ratios, e.g. 0.8,0.1,0.1")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed for splitting.")
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE, help="Sampling temperature.")
    parser.add_argument("--max-chunks", type=int, default=None, help="Optional max chunks to process.")
    parser.add_argument(
        "--max-doc-context-chars",
        type=int,
        default=DEFAULT_MAX_DOC_CONTEXT_CHARS,
        help="Max chars to send as full-document context for topic mapping.",
    )
    parser.add_argument(
        "--max-topic-context-chars",
        type=int,
        default=DEFAULT_MAX_TOPIC_CONTEXT_CHARS,
        help="Max chars to send as per-topic context for question generation.",
    )
    parser.add_argument(
        "--debug-dir",
        type=Path,
        default=DEFAULT_DEBUG_DIR,
        help="Folder to store raw model outputs by document/topic.",
    )
    parser.add_argument(
        "--quality-gate",
        choices=sorted(VALID_QUALITY_GATES),
        default=DEFAULT_QUALITY_GATE,
        help="Quality filtering before writing final JSONL. Default: strict.",
    )
    parser.add_argument(
        "--only-doc",
        type=str,
        default=None,
        help="Process only the PDFs whose filename/stem match this value "
             "(comma-separated list for multiple).",
    )
    parser.add_argument(
        "--skip-model-check",
        action="store_true",
        help="Skip verifying that the Ollama model is available before starting.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip documents that already have a non-empty checkpoint file in --debug-dir.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Extract and chunk PDFs, print stats, and exit without calling Ollama.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Delete generated JSON/JSONL files from --output parent and --debug-dir, then exit.",
    )
    parser.add_argument(
        "--clean-dry-run",
        action="store_true",
        help="Show which generated files --clean would delete, without deleting them.",
    )
    parser.add_argument(
        "--retrieval",
        choices=["lexical", "semantic", "hybrid"],
        default=DEFAULT_RETRIEVAL_MODE,
        help=(
            "Método de selección de chunks por topic. "
            "'hybrid' combina scoring léxico y embeddings Ollama. Default: hybrid."
        ),
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default=DEFAULT_EMBEDDING_MODEL,
        help="Modelo Ollama para embeddings (con --retrieval semantic o hybrid). Default: embeddinggemma:latest.",
    )
    parser.add_argument(
        "--judge",
        choices=sorted(VALID_JUDGE_MODES),
        default=DEFAULT_JUDGE_MODE,
        help="Judge final accepted QA items: off, audit, or filter failed/review rows. Default: off.",
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default=DEFAULT_JUDGE_MODEL,
        help="Ollama model used by --judge audit/filter. Default: gemma4:e4b.",
    )
    g = parser.add_argument_group("User-supplied topics / questions")
    g.add_argument(
        "--topics-file",
        type=Path,
        default=None,
        help="Archivo YAML o texto plano con topics definidos por el usuario. Reemplaza el topic mapping automático.",
    )
    g.add_argument(
        "--questions-file",
        type=Path,
        default=None,
        help=(
            "Texto plano con una pregunta seed por línea. "
            "Salta el topic mapping; genera respuestas para cada pregunta. "
            "Mutuamente excluyente con --topics-file."
        ),
    )
    return parser


def validate_args(args: argparse.Namespace) -> None:
    """Validate CLI args before starting expensive operations."""
    errors: list[str] = []
    if args.chunk_overlap >= args.chunk_size:
        errors.append(
            f"--chunk-overlap ({args.chunk_overlap}) debe ser < --chunk-size ({args.chunk_size})"
        )
    if args.num_topics < 1:
        errors.append("--num-topics debe ser >= 1")
    if args.questions_per_topic < 1:
        errors.append("--questions-per-topic debe ser >= 1")
    if not (0.0 <= args.temperature <= 2.0):
        errors.append(f"--temperature debe estar en [0.0, 2.0], recibido: {args.temperature}")
    if args.quality_gate not in VALID_QUALITY_GATES:
        errors.append(
            f"--quality-gate debe ser uno de {sorted(VALID_QUALITY_GATES)}, recibido: {args.quality_gate}"
        )
    if getattr(args, "topics_file", None) and getattr(args, "questions_file", None):
        errors.append("--topics-file y --questions-file son mutuamente excluyentes")
    if getattr(args, "topics_file", None) and not args.topics_file.exists():
        errors.append(f"--topics-file: archivo no encontrado: {args.topics_file}")
    if getattr(args, "questions_file", None) and not args.questions_file.exists():
        errors.append(f"--questions-file: archivo no encontrado: {args.questions_file}")
    if getattr(args, "retrieval", "lexical") in {"semantic", "hybrid"} and not getattr(
        args, "embedding_model", ""
    ).strip():
        errors.append("--embedding-model no puede estar vacío cuando --retrieval es 'semantic' o 'hybrid'")

    if getattr(args, "judge", "off") not in VALID_JUDGE_MODES:
        errors.append(f"--judge debe ser uno de {sorted(VALID_JUDGE_MODES)}, recibido: {args.judge}")
    if getattr(args, "judge", "off") in {"audit", "filter"} and not getattr(args, "judge_model", "").strip():
        errors.append("--judge-model no puede estar vacio cuando --judge es 'audit' o 'filter'")

    if errors:
        for err in errors:
            print(f"[ERROR] {err}")
        raise SystemExit(1)


def _checkpoint_path(debug_dir: Path, pdf_path: Path) -> Path:
    """Return the per-document checkpoint JSONL path."""
    return debug_dir / f"{pdf_path.stem}.items.jsonl"


def load_checkpoint_items(debug_dir: Path, pdf_path: Path) -> list[dict[str, Any]]:
    """Load previously generated items for a document, or empty list if absent."""
    path = _checkpoint_path(debug_dir, pdf_path)
    if not path.exists() or path.stat().st_size == 0:
        return []
    items: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError:
                logger.warning("Checkpoint con linea corrupta en %s", path)
    return items


def save_checkpoint_items(debug_dir: Path, pdf_path: Path, items: Sequence[dict[str, Any]]) -> None:
    """Persist the items generated for a document (idempotent overwrite)."""
    path = _checkpoint_path(debug_dir, pdf_path)
    write_jsonl(path, list(items))


def filter_pdfs_by_only_doc(pdfs: Sequence[Path], only_doc: str | None) -> list[Path]:
    """Keep only the PDFs matching `only_doc` (comma-separated filenames or stems, case-insensitive)."""
    if not only_doc:
        return list(pdfs)
    targets = [t.strip().lower() for t in only_doc.split(",") if t.strip()]
    if not targets:
        return list(pdfs)

    matched: list[Path] = []
    seen: set = set()
    missing: list[str] = []
    for target in targets:
        hits = [p for p in pdfs if p.name.lower() == target or p.stem.lower() == target]
        if not hits:
            missing.append(target)
            continue
        for hit in hits:
            if hit not in seen:
                seen.add(hit)
                matched.append(hit)

    if missing:
        available = ", ".join(p.name for p in pdfs) or "(ninguno)"
        raise RuntimeError(
            f"--only-doc: no se encontraron {missing}. Disponibles: {available}"
        )
    return matched


def _run_dry_run(
    args: argparse.Namespace,
    pdf_files: Sequence[Path],
    user_topics: list[Topic] | None = None,
    seed_question_mode: bool = False,
) -> None:
    """Print how much work a real run would do, without calling Ollama."""
    print(f"[DRY-RUN] {len(pdf_files)} PDF(s) a procesar con modelo '{args.model}'")
    if user_topics is not None:
        mode = "seed-questions" if seed_question_mode else "topics-file"
        print(f"[DRY-RUN] Modo {mode}: {len(user_topics)} topic(s) definidos por el usuario")
        for t in user_topics[:5]:
            print(f"  - {t.name[:80]}")
        if len(user_topics) > 5:
            print(f"  ... y {len(user_topics) - 5} más")
    total_chunks = 0
    total_chars = 0
    for pdf_path in pdf_files:
        raw_text = extract_text_from_pdf(pdf_path)
        chunks = build_chunks_from_text(
            raw_text=raw_text,
            document_name=pdf_path.name,
            document_stem=pdf_path.stem,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            max_chunks=args.max_chunks,
        )
        total_chunks += len(chunks)
        total_chars += len(raw_text)
        _, detected_language, language_scores = detect_document_language(raw_text)
        print(
            f"  - {pdf_path.name}: {len(raw_text):>8} chars -> {len(chunks):>3} chunks "
            f"| idioma detectado: {detected_language} {language_scores}"
        )
    if user_topics is not None:
        estimated_topics = len(user_topics) * len(pdf_files)
        estimated_calls = estimated_topics  # no topic-map call per doc
    else:
        estimated_topics = len(pdf_files) * args.num_topics
        estimated_calls = len(pdf_files) + estimated_topics  # 1 topic-map + N topic-gen per doc
    estimated_items = estimated_topics * args.questions_per_topic
    print("")
    print(f"Totales: {total_chars} chars, {total_chunks} chunks")
    print(
        f"Estimacion maxima: ~{estimated_topics} topicos, ~{estimated_items} items, "
        f"~{estimated_calls} llamadas a Ollama."
    )
