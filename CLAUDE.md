---
description:
alwaysApply: true
---

# CLAUDE.md

Working guide for this repository.

## 1. Project context

This repo generates synthetic supervised datasets from PDFs using local models served by Ollama.
The architectural reference is `localOllamaRAG`, but the goal here is dataset creation, not an end-user RAG interface.
The primary use case is producing QA corpora in Spanish (Castilian) and Valencian; English is supported for completeness.

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
4. Generate a topic map per document using wide context.
5. Generate questions per topic without semantic repetition (`think=False`).
6. Deduplicate (exact + semantic via bigrams).
7. Export the main JSONL and train/val/test splits.
8. Persist metadata and per-document/topic run logs, plus per-document checkpoints for `--resume`.

## 6. Critical variables

| Variable | Default | Usage |
|---|---|---|
| `OLLAMA_DATASET_MODEL` | `gemma4:e2b` | Primary generator model |
| `OLLAMA_TIMEOUT_SECS` | `300` | Ollama call timeout |
| `OLLAMA_MAX_RETRIES` | `3` | Retries on transient Ollama errors |
| `OLLAMA_RETRY_BACKOFF_SECS` | `2.0` | Linear backoff between retries |
| `DATASET_LOG_LEVEL` | `INFO` | Logging level (`DEBUG/INFO/WARNING/ERROR`) |
| `DATASET_LANGUAGE` | `es` | Output language |
| `DATASET_CHUNK_SIZE` | `3500` | Chunk size |
| `DATASET_CHUNK_OVERLAP` | `350` | Chunk overlap |
| `DATASET_NUM_TOPICS` | `8` | Max topics per document |
| `DATASET_QUESTIONS_PER_TOPIC` | `6` | QA items per topic |
| `DATASET_MAX_DOC_CONTEXT_CHARS` | `110000` | Max context for topic mapping |
| `DATASET_MAX_TOPIC_CONTEXT_CHARS` | `24000` | Max context for QA generation |
