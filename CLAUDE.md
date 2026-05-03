---
description:
alwaysApply: true
---

# CLAUDE.md

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
6. **Keep `README.md` and `CLAUDE.md` in sync with the code.** Whenever you change any of the following, update both files in the same turn:
   - Environment variables (names, defaults, meaning).
   - CLI flags or their defaults.
   - Item schema in the output JSONL (field names, allowed values).
   - Keys written to `dataset.meta.json` or to per-document debug/checkpoint files.
   - Directory layout or default input/output paths.
   - Install / test instructions.
   Before closing a task that touches the pipeline, re-read the relevant sections of both docs and reconcile any drift — do not defer doc updates to a later commit.

## 3. Expected structure

```text
pipeline/
  generate_dataset.py    ← orchestrator and CLI entry point
  engine/
    _config.py           ← constants, dataclasses, logging
    _text.py             ← text transforms, language detection
    _pdf.py              ← PDF extraction and chunking
    _prompts.py          ← LLM prompt builders and JSON parser
    _ollama.py           ← Ollama client wrapper
    _topics.py           ← topic parsing, validation, retrieval
    _evidence.py         ← literal evidence-window extraction/ranking
    _quality.py          ← quality gate and deduplication
    _judge.py            ← optional factuality judge audit/filter
    _generation.py       ← QA generation per topic
    _export.py           ← JSONL I/O, splits, metadata
    _cli.py              ← argparse, validation, dry-run
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
7. Reassign each generated item to the best-fitting topic of its document, or reject with `topic_mismatch_reassign` when no topic matches.
8. Apply evidence-level repair when generated answers are copied too closely, circular, or insufficiently supported, then apply the quality gate before deduplication. In `strict` mode, keep only clean items with verified source context, topic alignment, and enough source support; write rejected rows to `dataset.rejected.jsonl`.
9. Backfill topics below the requested item count by trying unused evidence windows first, then an alternate retrieval context; topics with fewer than 2 final rows are recorded in `quality.unfillable_topics`.
10. Deduplicate accepted items (exact + semantic via bigrams, plus question-, QA-, and answer-term overlap), reporting separated counts in `quality.dedup_breakdown`.
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
- Generation is evidence-first by default: code selects literal `context_source` windows from the extracted topic context and Ollama generates only `question`, `answer`, `type`, and `difficulty` from that fixed evidence. `context_source_verified=true` means `context_source` is a literal substring of the extracted document context after whitespace normalization.
- `--quality-gate strict|balanced|off` controls final dataset filtering. Default `strict` rejects unverified items, circular answers, common extraction artifacts, truncated context sources, first-person paper voice, clear topic mismatches, and insufficiently supported compare/inference answers; rejected rows are written to `dataset.rejected.jsonl`.
- `balanced` keeps unverified context sources but still applies deterministic quality checks and circular-answer filtering.
- `--judge off|audit|filter` controls optional LLM factuality auditing after quality filtering and deduplication. Default `off` keeps current cost and output behavior. `audit` writes judge scores without changing `dataset.jsonl`; `filter` keeps only `judge_decision=pass` rows in `dataset.jsonl` and moves `review`/`fail` rows to `dataset.rejected.jsonl`.
- `--judge-model MODEL` sets the Ollama model used by `--judge audit` or `--judge filter`. Defaults to `OLLAMA_JUDGE_MODEL`, whose default is `gemma4:e4b`.
- `--retrieval lexical|semantic|hybrid` selects the chunk retrieval strategy for topic context building. Default `hybrid` combines lexical scoring and Ollama embeddings; embeddings are cached per chunk within a document. Requires the embedding model to be available in Ollama. Evidence windows are then extracted and ranked from the retrieved context before Q/A generation.
- `--embedding-model MODEL` sets the Ollama model used for embeddings when `--retrieval semantic` or `--retrieval hybrid`. Defaults to `embeddinggemma:latest`.
- `--topics-file PATH` loads a YAML (requires `pyyaml`) or plain-text file with user-defined topics. Skips the LLM topic mapping step entirely. Topics get `topic_id` with prefix `user-`.
- `--questions-file PATH` loads a plain-text file with one seed question per line. Skips topic mapping; generates one answer per question. Items get `topic_id` with prefix `seed-`. Mutually exclusive with `--topics-file`. Both flags are compatible with `--only-doc`, `--resume`, and `--quality-gate`.
- `--dry-run` extracts/chunks PDFs and reports detected language, estimated topics/items, and estimated Ollama calls without model generation.
- `--clean-dry-run` lists generated JSON/JSONL artifacts that would be removed.
- `--clean` removes generated `.json` and `.jsonl` files from `pipeline/output` and `pipeline/run_logs`, preserving `.gitkeep` and input PDFs.

## 8. Output schema and metadata notes

Each item includes `document_language` and `source_chunk_ids` in addition to the QA fields, topic fields, source document, timestamps, and context traceability. Items moved to a different topic by post-generation reassignment also include `reassigned_from = {topic_id, topic}` recording the original assignment.

`dataset.meta.json` includes `document_languages`, a mapping from PDF filename to the detected or forced language, plus counts, params, split sizes, runtime info, audit stats, and reproducibility metadata.

Important metadata counts distinguish generation stages: `generated_items`, `deduplicated_items`, `accepted_items`, `rejected_items`, and `context_source_verified_items`. The `quality` block records the active gate, accepted/rejected counts, verified ratio, rejection reasons (including `topic_mismatch_reassign`), backfill count, accepted items before dedupe, verified items before dedupe, duplicates removed after quality filtering, plus:

- `quality.dedup_breakdown` — separated counts for `duplicate_exact`, `duplicate_semantic_question`, `duplicate_semantic_qa`, and `duplicate_semantic_answer`.
- `quality.evidence_first` — counters for `candidate_windows`, `attempted_windows`, `accepted_from_evidence`, `repair_attempts`, `discarded_windows`, and `evidence_exhausted_topics`.
- `quality.topic_reassignments` — `{reassigned, rejected_no_match, kept_as_is}` counts produced by the reassignment pass.
- `quality.backfill_attempts` — list of attempts (`primary` or `alt:<retrieval>`) per topic key.
- `quality.unfillable_topics` — list of `{document, topic_id, topic_name, attempts, accepted}` for topics that did not reach the minimum after backfill.

The `audit` block records accepted/rejected counts by topic, split coverage by topic, topics with no accepted rows, low-coverage topics, audit warnings, and additionally:

- `audit.coverage_ratio` — fraction of expected topics with at least one accepted item.
- `audit.multi_doc_in_splits` — `{train, val, test}` booleans indicating whether each split contains rows from more than one document.
- `audit.pipeline_success` — `True` when topic coverage clears the soft threshold (≤ 20 % of topics missing accepted items, with a minimum tolerance of one topic for small documents).

Per-document debug JSON files in `pipeline/run_logs` include selected evidence windows, evidence-first generation attempts, and parsed payloads.

The standard output set includes:

- `pipeline/output/dataset.jsonl`
- `pipeline/output/dataset_train.jsonl`
- `pipeline/output/dataset_val.jsonl`
- `pipeline/output/dataset_test.jsonl`
- `pipeline/output/dataset.meta.json`
- `pipeline/output/dataset.rejected.jsonl`
