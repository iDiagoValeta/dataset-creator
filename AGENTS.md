---
description:
alwaysApply: true
---

# AGENTS.md

Working guide for this repository.

## 1. Project context

This repo generates synthetic supervised datasets from PDFs using local models served by Ollama.
The architectural reference is `localOllamaRAG`, but the goal here is dataset creation, not an end-user RAG interface.
The primary use case is producing QA corpora in the same language as each source PDF. Spanish (Castilian) and Valencian are important use cases, but default generation should follow per-document language detection unless the user explicitly forces `--language`.

## 2. Operating rules

1. Reply to the user in Spanish (user-facing language), even though this file is in English.
2. Do not commit or push without explicit user approval.
3. Keep the script-first style with a clear `main()` entry point.
4. Prioritize reproducible artifacts (JSONL + metadata JSON), not just stdout output.
5. Current flow is text-only. Do not add OCR, image, or table handling.
6. **Keep `README.md` and `AGENTS.md` in sync with the code.** Whenever you change any of the following, update both files in the same turn:
   - Environment variables (names, defaults, meaning).
   - CLI flags or their defaults.
   - Item schema in the output JSONL (field names, allowed values).
   - Keys written to `dataset.meta.json` or to per-document debug/checkpoint files.
   - Directory layout or default input/output paths.
   - Install / test instructions.
   Before closing a task that touches the pipeline, re-read the relevant sections of both docs and reconcile any drift; do not defer doc updates to a later commit.

## 3. Expected structure

```text
pipeline/
  generate_dataset.py    <- orchestrator and CLI entry point
  engine/
    _config.py           <- constants, dataclasses, logging
    _text.py             <- text transforms, language detection
    _pdf.py              <- PDF extraction and chunking
    _prompts.py          <- LLM prompt builders and JSON parser
    _ollama.py           <- Ollama client wrapper
    _topics.py           <- topic parsing, validation, retrieval
    _quality.py          <- quality gate and deduplication
    _judge.py            <- optional factuality judge audit
    _generation.py       <- QA generation per topic
    _export.py           <- JSONL I/O, splits, metadata
    _cli.py              <- argparse, validation, dry-run
  requirements.txt
  requirements-dev.txt
  input/
  output/
  run_logs/
tests/
.github/workflows/ci.yml
pyproject.toml
.env.example
```

## 4. Code conventions

- MODULE MAP at the top of non-trivial modules.
- Clear section separators.
- Imports ordered: stdlib -> third-party -> local.
- Environment variables for important defaults.
- Optional dependencies guarded with `try/except` + boolean flag.
- Docstrings in English; user-facing messages in Spanish.
- `print()` for visible progress; `logging` for technical diagnostics.

## 5. Functional flow

1. Load PDFs from `pipeline/input/`.
2. Extract text (`pymupdf4llm` preferred, `pypdf` defensive fallback).
3. Split text into overlapping chunks.
4. Detect the document language unless `--language` forces a specific output language.
5. Generate a topic map per document using wide context, with compact retry, section-heading refinement for overly generic maps, and chunk fallback.
6. Generate questions per topic without semantic repetition in the document language.
7. Apply the quality gate before deduplication. In `strict` mode, keep only clean items with verified source context, topic alignment, and enough source support; write rejected rows to `dataset.rejected.jsonl`.
8. Backfill topics with fewer than 2 accepted rows, then rerun quality filtering.
9. Deduplicate accepted items (exact + semantic via bigrams, including duplicate answers).
10. Optionally run `--judge audit` over final accepted rows and write `dataset.judged.jsonl` without changing `dataset.jsonl`.
11. Export the main JSONL and topic-aware train/val/test splits.
12. Persist metadata, audit stats, per-document/topic run logs, plus per-document checkpoints for `--resume`.

## 6. Critical variables

| Variable | Default | Usage |
|---|---|---|
| `OLLAMA_DATASET_MODEL` | `gemma4:e4b` | Primary generator model |
| `OLLAMA_TIMEOUT_SECS` | `300` | Ollama call timeout |
| `OLLAMA_MAX_RETRIES` | `3` | Retries on transient Ollama errors |
| `OLLAMA_RETRY_BACKOFF_SECS` | `2.0` | Linear backoff between retries |
| `OLLAMA_EMBEDDING_MODEL` | `embeddinggemma:latest` | Ollama model for semantic/hybrid embeddings |
| `OLLAMA_JUDGE_MODEL` | `gemma4:e4b` | Ollama model for `--judge audit` |
| `DATASET_RETRIEVAL` | `hybrid` | Chunk retrieval mode (`lexical`, `semantic`, or `hybrid`) |
| `DATASET_JUDGE` | `off` | Judge mode (`off` or `audit`) |
| `DATASET_LOG_LEVEL` | `INFO` | Logging level (`DEBUG/INFO/WARNING/ERROR`) |
| `DATASET_LANGUAGE` | `auto` | Detect language per PDF; use `es`, `en`, `ca`, etc. to force output |
| `DATASET_CHUNK_SIZE` | `3500` | Chunk size |
| `DATASET_CHUNK_OVERLAP` | `350` | Chunk overlap |
| `DATASET_NUM_TOPICS` | `8` | Max topics per document |
| `DATASET_QUESTIONS_PER_TOPIC` | `6` | QA items per topic |
| `DATASET_SPLIT` | `0.8,0.1,0.1` | Train/val/test split ratios |
| `DATASET_TEMPERATURE` | `0.2` | Sampling temperature |
| `DATASET_SEED` | `42` | Random seed for splits and generation |
| `DATASET_MAX_DOC_CONTEXT_CHARS` | `110000` | Max context for topic mapping |
| `DATASET_MAX_TOPIC_CONTEXT_CHARS` | `24000` | Max context for QA generation |
| `DATASET_QUALITY_GATE` | `strict` | Final quality filter (`strict`, `balanced`, `off`) |

## 7. Current CLI notes

- `--language auto` is the default and should generate topics, questions, and answers in the detected source-document language.
- `--language <code>` forces one language for all PDFs.
- `--quality-gate strict|balanced|off` controls final dataset filtering. Default `strict` rejects unverified items, circular answers, common extraction artifacts, truncated context sources, first-person paper voice, clear topic mismatches, and insufficiently supported compare/inference answers; rejected rows are written to `dataset.rejected.jsonl`.
- `context_source_verified=true` means `context_source` is a literal substring of the extracted document context after whitespace normalization. Do not infer or repair source context from answer overlap; this project is primarily for RAG-style triples where the answer must be deducible from literal source context.
- `balanced` keeps unverified context sources but still applies deterministic quality checks and circular-answer filtering.
- `--judge off|audit` controls optional LLM factuality auditing after quality filtering and deduplication. Default `off` keeps current cost and output behavior.
- `--judge-model MODEL` sets the Ollama model used by `--judge audit`. Defaults to `OLLAMA_JUDGE_MODEL`, whose default is `gemma4:e4b`.
- `--retrieval lexical|semantic|hybrid` selects the chunk retrieval strategy for topic context building. Default `hybrid` combines lexical scoring and Ollama embeddings; embeddings are cached per chunk within a document. Requires the embedding model to be available in Ollama.
- `--embedding-model MODEL` sets the Ollama model used for embeddings when `--retrieval semantic` or `--retrieval hybrid`. Defaults to `embeddinggemma:latest`.
- `--topics-file PATH` loads a YAML (requires `pyyaml`) or plain-text file with user-defined topics. Skips the LLM topic mapping step entirely. Topics get `topic_id` with prefix `user-`.
- `--questions-file PATH` loads a plain-text file with one seed question per line. Skips topic mapping; generates one answer per question. Items get `topic_id` with prefix `seed-`. Mutually exclusive with `--topics-file`. Both flags are compatible with `--only-doc`, `--resume`, and `--quality-gate`.
- `--dry-run` extracts/chunks PDFs and reports detected language, estimated topics/items, and estimated Ollama calls without model generation.
- `--clean-dry-run` lists generated JSON/JSONL artifacts that would be removed.
- `--clean` removes generated `.json` and `.jsonl` files from `pipeline/output` and `pipeline/run_logs`, preserving `.gitkeep` and input PDFs.

## 8. Output schema and metadata notes

Each item includes `document_language` and `source_chunk_ids` in addition to the QA fields, topic fields, source document, timestamps, and context traceability. The core RAG triple is `question`, literal `context_source`, and `answer`; the answer should be factually deducible from that context alone. When `--judge audit` is enabled, `dataset.judged.jsonl` adds `judge_score`, `judge_decision` (`pass`, `review`, or `fail`), `judge_reasons`, `judge_explanation`, and `judge_model` to each accepted item.

`dataset.meta.json` includes `document_languages`, a mapping from PDF filename to the detected or forced language, plus counts, params, split sizes, runtime info, judge stats, audit stats, and reproducibility metadata.

Important metadata counts distinguish generation stages: `generated_items`, `deduplicated_items`, `accepted_items`, `rejected_items`, and `context_source_verified_items`. The `quality` block records the active gate, accepted/rejected counts, verified ratio, rejection reasons, backfill count, accepted items before dedupe, verified items before dedupe, and duplicates removed after quality filtering. The `judge` block records mode, model, judged item count, decision counts, average score, and reason counts. The `audit` block records accepted/rejected counts by topic, split coverage by topic, topics with no accepted rows, low-coverage topics, and audit warnings.

The standard output set includes:

- `pipeline/output/dataset.jsonl`
- `pipeline/output/dataset_train.jsonl`
- `pipeline/output/dataset_val.jsonl`
- `pipeline/output/dataset_test.jsonl`
- `pipeline/output/dataset.meta.json`
- `pipeline/output/dataset.rejected.jsonl`
- `pipeline/output/dataset.judged.jsonl` (only with `--judge audit`)
