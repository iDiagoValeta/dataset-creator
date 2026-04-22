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
  generate_dataset.py
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
6. Generate questions per topic without semantic repetition (`think=False`) in the document language.
7. Apply the quality gate before deduplication. In `strict` mode, keep only clean items with verified source context and write rejected rows to `dataset.rejected.jsonl`.
8. Deduplicate accepted items (exact + semantic via bigrams, including duplicate answers).
9. Export the main JSONL and train/val/test splits.
10. Persist metadata and per-document/topic run logs, plus per-document checkpoints for `--resume`.

## 6. Critical variables

| Variable | Default | Usage |
|---|---|---|
| `OLLAMA_DATASET_MODEL` | `gemma4:e2b` | Primary generator model |
| `OLLAMA_TIMEOUT_SECS` | `300` | Ollama call timeout |
| `OLLAMA_MAX_RETRIES` | `3` | Retries on transient Ollama errors |
| `OLLAMA_RETRY_BACKOFF_SECS` | `2.0` | Linear backoff between retries |
| `DATASET_LOG_LEVEL` | `INFO` | Logging level (`DEBUG/INFO/WARNING/ERROR`) |
| `DATASET_LANGUAGE` | `auto` | Detect language per PDF; use `es`, `en`, `ca`, etc. to force output |
| `DATASET_CHUNK_SIZE` | `3500` | Chunk size |
| `DATASET_CHUNK_OVERLAP` | `350` | Chunk overlap |
| `DATASET_NUM_TOPICS` | `8` | Max topics per document |
| `DATASET_QUESTIONS_PER_TOPIC` | `6` | QA items per topic |
| `DATASET_MAX_DOC_CONTEXT_CHARS` | `110000` | Max context for topic mapping |
| `DATASET_MAX_TOPIC_CONTEXT_CHARS` | `24000` | Max context for QA generation |
| `DATASET_QUALITY_GATE` | `strict` | Final quality filter (`strict`, `balanced`, `off`) |

## 7. Current CLI notes

- `--language auto` is the default and should generate topics, questions, and answers in the detected source-document language.
- `--language <code>` forces one language for all PDFs.
- `--quality-gate strict|balanced|off` controls final dataset filtering. Default `strict` rejects unverified items and common extraction artifacts; rejected rows are written to `dataset.rejected.jsonl`.
- `--dry-run` extracts/chunks PDFs and reports detected language, estimated topics/items, and estimated Ollama calls without model generation.
- `--clean-dry-run` lists generated JSON/JSONL artifacts that would be removed.
- `--clean` removes generated `.json` and `.jsonl` files from `pipeline/output` and `pipeline/run_logs`, preserving `.gitkeep` and input PDFs.

## 8. Output schema and metadata notes

Each item includes `document_language` and `source_chunk_ids` in addition to the QA fields, topic fields, source document, timestamps, and context traceability.

`dataset.meta.json` includes `document_languages`, a mapping from PDF filename to the detected or forced language, plus counts, params, split sizes, runtime info, and reproducibility metadata.

Important metadata counts distinguish generation stages: `generated_items`, `deduplicated_items`, `accepted_items`, `rejected_items`, and `context_source_verified_items`. The `quality` block records the active gate, accepted/rejected counts, verified ratio, rejection reasons, accepted items before dedupe, verified items before dedupe, and duplicates removed after quality filtering.

The standard output set includes:

- `pipeline/output/dataset.jsonl`
- `pipeline/output/dataset_train.jsonl`
- `pipeline/output/dataset_val.jsonl`
- `pipeline/output/dataset_test.jsonl`
- `pipeline/output/dataset.meta.json`
- `pipeline/output/dataset.rejected.jsonl`
