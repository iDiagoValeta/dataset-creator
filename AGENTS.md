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
    _evidence.py         <- literal evidence-window extraction/ranking
    _quality.py          <- quality gate and deduplication
    _judge.py            <- optional factuality judge audit/filter
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
6. Select ranked literal evidence windows per topic (1-3 complete sentences, one source chunk), then ask Ollama to generate only question/answer/type/difficulty from each fixed evidence window.
7. Apply evidence-level repair when generated answers are copied too closely, circular, or insufficiently supported, then apply the quality gate before deduplication. In `strict` mode, keep only clean items with verified source context, topic alignment, and enough source support; write rejected rows to `dataset.rejected.jsonl`.
8. Backfill topics below the requested item count by trying unused evidence windows first, then an alternate retrieval context; topics with fewer than 2 final rows are recorded in `quality.unfillable_topics`.
9. Deduplicate accepted items (exact + semantic via bigrams, including duplicate answers).
10. Optionally run `--judge audit` over final accepted rows, grouped by document for progress and smaller judge batches, and write one combined `dataset.judged.jsonl` without changing `dataset.jsonl`; or run `--judge filter` to remove judge `review`/`fail` rows from the final dataset and append them to rejected rows.
11. Export the main JSONL and topic-aware train/val/test splits.
12. Persist metadata, audit stats, per-document/topic run logs, plus per-document checkpoints for `--resume`.

## 6. Critical variables

| Variable | Default | Usage |
|---|---|---|
| `OLLAMA_CONTEXT_LENGTH` | `32768` | Recommended Ollama server context window for long-document topic mapping; restart Ollama after changing it |
| `OLLAMA_DATASET_MODEL` | `gemma4:e4b` | Primary generator model |
| `OLLAMA_TIMEOUT_SECS` | `300` | Ollama call timeout |
| `OLLAMA_MAX_RETRIES` | `3` | Retries on transient Ollama errors |
| `OLLAMA_RETRY_BACKOFF_SECS` | `2.0` | Linear backoff between retries |
| `OLLAMA_EMBEDDING_MODEL` | `embeddinggemma:latest` | Ollama model for semantic/hybrid embeddings |
| `OLLAMA_JUDGE_MODEL` | `gemma4:e4b` | Ollama model for `--judge audit` / `--judge filter` |
| `DATASET_RETRIEVAL` | `hybrid` | Chunk retrieval mode (`lexical`, `semantic`, or `hybrid`) |
| `DATASET_JUDGE` | `off` | Judge mode (`off`, `audit`, or `filter`) |
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
- `--quality-gate strict|balanced|off` controls final dataset filtering. Default `strict` rejects unverified items, circular answers, common extraction artifacts, truncated context sources, first-person paper voice, clear topic mismatches, insufficiently supported factual/definition/compare/inference answers, cross-chunk contexts, and verbatim answers; rejected rows are written to `dataset.rejected.jsonl`.
- Generation is evidence-first by default: code selects literal `context_source` windows from the extracted topic context and Ollama generates only `question`, `answer`, `type`, and `difficulty` from that fixed evidence. `context_source_verified=true` means `context_source` is a literal substring of the extracted document context after whitespace normalization. Do not infer or repair source context from answer overlap; this project is primarily for RAG-style triples where the answer must be deducible from literal source context.
- Deterministic quality checks in `strict` mode also reject: **mojibake** (double-encoded UTF-8 sequences such as `â€™` or `Î¸`) detected in any Q/A field; **cross-chunk context** (`cross_chunk_context`) where `context_source` contains an internal chunk marker; and **verbatim answers** (`verbatim_answer`) with >= 70 % answer-bigram overlap with `context_source`, including short copied evidence phrases. Encoding is normalized at PDF extraction time using `ftfy` when available, or a built-in replacement map as fallback. Chunk markers are stripped from all generated text before quality checks.
- `balanced` keeps unverified context sources but still applies deterministic quality checks and circular-answer filtering.
- `--judge off|audit|filter` controls optional LLM factuality auditing after quality filtering and deduplication. Default `off` keeps current cost and output behavior. `audit` writes judge scores without changing `dataset.jsonl`; `filter` keeps only `judge_decision=pass` rows in `dataset.jsonl` and moves `review`/`fail` rows to `dataset.rejected.jsonl` with `rejection_reason="judge_fail:<reason>"`.
- `--judge-model MODEL` sets the Ollama model used by `--judge audit` or `--judge filter`. Defaults to `OLLAMA_JUDGE_MODEL`, whose default is `gemma4:e4b`.
- The judge audits accepted rows one document at a time, then writes one combined `dataset.judged.jsonl`. It applies deterministic pre-checks (mojibake, internal chunk markers, near-verbatim answers) before calling the LLM. Blocking reasons (`cross_chunk_context`, `extraction_artifact`, `overly_extractive`, `truncated_context`, `unsupported_detail`) force `fail` regardless of LLM output. A noisy `weak_context` or `judge_error` reason is reconciled to `factual` only when the LLM also returns `decision=pass` and all component scores are at least 0.8. The minimum passing component score (context quality, answer support, question quality) is 0.6. Topic audit keys in `dataset.meta.json` use composite `"{document}::{topic_id}"` format to prevent cross-document collisions.
- `--retrieval lexical|semantic|hybrid` selects the chunk retrieval strategy for topic context building. Default `hybrid` combines lexical scoring and Ollama embeddings; embeddings are cached per chunk within a document. Requires the embedding model to be available in Ollama. Evidence windows are then extracted and ranked from the retrieved context before Q/A generation.
- `--embedding-model MODEL` sets the Ollama model used for embeddings when `--retrieval semantic` or `--retrieval hybrid`. Defaults to `embeddinggemma:latest`.
- `--topics-file PATH` loads a YAML (requires `pyyaml`) or plain-text file with user-defined topics. Skips the LLM topic mapping step entirely. Topics get `topic_id` with prefix `user-`.
- `--questions-file PATH` loads a plain-text file with one seed question per line. Skips topic mapping; generates one answer per question. Items get `topic_id` with prefix `seed-`. Mutually exclusive with `--topics-file`. Both flags are compatible with `--only-doc`, `--resume`, and `--quality-gate`.
- `--dry-run` extracts/chunks PDFs and reports detected language, estimated topics/items, and estimated Ollama calls without model generation.
- `--clean-dry-run` lists generated JSON/JSONL artifacts that would be removed.
- `--clean` removes generated `.json` and `.jsonl` files from `pipeline/output` and `pipeline/run_logs`, preserving `.gitkeep` and input PDFs.

## 8. Output schema and metadata notes

Each item includes `document_language` and `source_chunk_ids` in addition to the QA fields, topic fields, source document, timestamps, and context traceability. The core RAG triple is `question`, literal `context_source`, and `answer`; the answer should be factually deducible from that context alone. When `--judge audit` or `--judge filter` is enabled, `dataset.judged.jsonl` adds `judge_score`, `judge_context_quality`, `judge_answer_support`, `judge_question_quality`, `judge_decision` (`pass`, `review`, or `fail`), `judge_reasons`, `judge_explanation`, and `judge_model` to each accepted item before optional judge filtering.

`dataset.meta.json` includes `document_languages`, a mapping from PDF filename to the detected or forced language, plus counts, params, split sizes, runtime info, judge stats, audit stats, and reproducibility metadata.

Important metadata counts distinguish generation stages: `generated_items`, `deduplicated_items`, `accepted_items`, `rejected_items`, and `context_source_verified_items`. The `quality` block records the active gate, accepted/rejected counts, verified ratio, rejection reasons, backfill count, accepted items before dedupe, verified items before dedupe, duplicates removed after quality filtering, evidence-first counters (`candidate_windows`, `attempted_windows`, `accepted_from_evidence`, `repair_attempts`, `discarded_windows`, `evidence_exhausted_topics`), and when `--judge filter` is active, `judge_filtered_items` plus `final_items_after_judge`. The `judge` block records mode, model, judged item count, decision counts, average score, average component scores (`context_quality`, `answer_support`, `question_quality`), and reason counts. The `audit` block records accepted/rejected counts by topic, split coverage by topic, topics with no accepted rows, low-coverage topics, and audit warnings.

Per-document debug JSON files in `pipeline/run_logs` include language fields, topic-map attempts, `chunk_count`, topic records, selected evidence windows, raw generation attempts, and parsed payloads. Resume mode uses these debug files plus per-document `*.items.jsonl` checkpoints to infer `chunk_count` and `topic_count` in `dataset.meta.json` without regenerating the document.

The standard output set includes:

- `pipeline/output/dataset.jsonl`
- `pipeline/output/dataset_train.jsonl`
- `pipeline/output/dataset_val.jsonl`
- `pipeline/output/dataset_test.jsonl`
- `pipeline/output/dataset.meta.json`
- `pipeline/output/dataset.rejected.jsonl`
- `pipeline/output/dataset.judged.jsonl` (only with `--judge audit` or `--judge filter`)
