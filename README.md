# dataset-creator

Generate supervised QA datasets from PDFs using local Ollama models.

## Why this repo exists

> Still a work in progress.

I needed high-quality QA datasets for my own projects, especially in **Spanish (Castilian)** and **Valencian**, and the pipelines I had available didn't cover that workflow well. Until now, I was building those datasets **manually with NotebookLM** — slow, non-reproducible, and hard to iterate on.

This repository automates that workflow end-to-end: extract text from PDFs, detect the document language, map topics, generate grounded QA pairs with a local Ollama model, deduplicate, and export ready-to-train JSONL splits — all while keeping traceability back to the source context.

It is a personal tool first, so expect ongoing changes.

## Goal

Given one or more PDFs in `pipeline/input/`, the pipeline produces a structured JSONL with questions answerable from the text, argumentable answers, and per-item context traceability.

## Structure

```text
dataset-creator/
├── CLAUDE.md
├── README.md
├── pyproject.toml
├── .env.example
├── .github/
│   └── workflows/ci.yml
├── pipeline/
│   ├── generate_dataset.py
│   ├── requirements.txt
│   ├── requirements-dev.txt
│   ├── input/
│   │   └── .gitkeep
│   ├── output/
│   │   └── .gitkeep
│   └── run_logs/
│       └── .gitkeep
└── tests/
    └── test_generate_dataset.py
```

## Requirements

- Python 3.10+
- Ollama running locally
- A model pulled in Ollama (recommended default: `gemma4:e2b`)

## Installation

```bash
# Option 1: requirements files
pip install -r pipeline/requirements.txt
pip install -r pipeline/requirements-dev.txt   # adds pytest

# Option 2: pyproject
pip install -e .
pip install -e ".[dev]"                         # adds pytest
```

Copy `.env.example` to `.env` (or export the variables in your shell) and tweak as needed.

## Environment variables

- `OLLAMA_DATASET_MODEL` (fallback: `OLLAMA_RAG_MODEL`, default `gemma4:e2b`)
- `OLLAMA_TIMEOUT_SECS` (default `300`)
- `OLLAMA_MAX_RETRIES` (default `3`) — retries on transient Ollama errors
- `OLLAMA_RETRY_BACKOFF_SECS` (default `2.0`) — linear backoff between retries
- `DATASET_LANGUAGE` (default `auto`; detects each PDF language. Use `es`, `ca`, `en`, etc. to force one)
- `DATASET_CHUNK_SIZE` (default `3500`)
- `DATASET_CHUNK_OVERLAP` (default `350`)
- `DATASET_NUM_TOPICS` (default `8`)
- `DATASET_QUESTIONS_PER_TOPIC` (default `6`)
- `DATASET_SPLIT` (default `0.8,0.1,0.1`)
- `DATASET_TEMPERATURE` (default `0.2`; `0.0` allowed for greedy decoding)
- `DATASET_SEED` (default `42`; forwarded to Ollama for determinism)
- `DATASET_MAX_DOC_CONTEXT_CHARS` (default `110000`)
- `DATASET_MAX_TOPIC_CONTEXT_CHARS` (default `24000`)
- `DATASET_LOG_LEVEL` (default `INFO`)

## Quick usage

1. Drop PDFs into `pipeline/input/`.
2. Run:

```bash
python pipeline/generate_dataset.py
```

Output:

- `pipeline/output/dataset.jsonl`
- `pipeline/output/dataset_train.jsonl`
- `pipeline/output/dataset_val.jsonl`
- `pipeline/output/dataset_test.jsonl`
- `pipeline/output/dataset.meta.json`
- `pipeline/run_logs/<pdf>.json` (raw model outputs per document/topic)
- `pipeline/run_logs/<pdf>.items.jsonl` (per-document checkpoint, used by `--resume`)

## CLI examples

```bash
python pipeline/generate_dataset.py --model gemma4:e2b --num-topics 10 --questions-per-topic 8
python pipeline/generate_dataset.py --split 0.7,0.15,0.15
python pipeline/generate_dataset.py --language es   # force Spanish output instead of auto-detecting
python pipeline/generate_dataset.py --language ca   # force Valencian / Catalan output
python pipeline/generate_dataset.py --source-dir pipeline/input --output pipeline/output/thesis_es.jsonl
python pipeline/generate_dataset.py --only-doc Operating_system.pdf
python pipeline/generate_dataset.py --only-doc a.pdf,b.pdf
python pipeline/generate_dataset.py --temperature 0.0 --skip-model-check
python pipeline/generate_dataset.py --resume
python pipeline/generate_dataset.py --dry-run
python pipeline/generate_dataset.py --clean-dry-run
python pipeline/generate_dataset.py --clean
```

Notable flags:

- `--language auto`: detect the language per PDF and generate topics/questions/answers in that source language. This is the default.
- `--language <code>`: force a language for all PDFs (`es`, `en`, `ca`, etc.).
- `--only-doc <names>`: process only the matching PDFs (filename or stem, case-insensitive). Comma-separated for multiple (e.g. `a.pdf,b.pdf`).
- `--skip-model-check`: skip the initial `ollama.list()` availability check.
- `--resume`: skip documents that already have a non-empty checkpoint file in `--debug-dir` (`<stem>.items.jsonl`).
- `--dry-run`: extract and chunk PDFs, print stats (chunks, estimated Ollama calls and items), and exit without calling the model. Useful for sizing a run.
- `--clean-dry-run`: print generated JSON/JSONL files that would be deleted from the output/log folders.
- `--clean`: delete generated JSON/JSONL files from the output folder and debug/log folder, preserving `.gitkeep` and input PDFs.

## Language detection

By default, the pipeline detects each PDF's language from the extracted text and asks the model to generate topics, questions, and answers in that same language.

The detector is local and heuristic: it counts frequent marker words for English, Spanish, Catalan, French, and Portuguese, with small bonuses for distinctive characters such as `ñ`, `¿`, `ç`, `ã`, etc. The detected language is stored in per-item `document_language`, per-document debug logs, and `dataset.meta.json`.

## Checkpointing

Every document writes its generated items to `pipeline/run_logs/<stem>.items.jsonl`. If a run is interrupted, resume it with:

```bash
python pipeline/generate_dataset.py --resume
```

Documents with a checkpoint are loaded from disk without calling the model again; the remaining ones are processed normally.

## Item format

Each JSONL line contains:

- `id`
- `question`
- `answer`
- `type` (`factual|conceptual|inference|compare|definition`)
- `difficulty` (`easy|medium|hard`, normalized)
- `context_source` (literal fragment from the context supporting the answer)
- `context_source_verified` (`true` if the fragment appears verbatim in the context)
- `topic`
- `topic_id`
- `topic_keywords`
- `document`
- `document_language`
- `created_at`
- `context_excerpt`

## Reproducible metadata

`dataset.meta.json` includes:

- Counts (`pdf_count`, `chunk_count`, `topic_count`, `generated_items`, `deduplicated_items`, `context_source_verified_items`, `resumed_documents`).
- `document_languages` mapping each PDF filename to its detected or forced language.
- `params` with every hyperparameter used (including `only_doc`, `resume`).
- `runtime` with `python_version`, `platform`, installed versions of `ollama` / `pypdf` / `pymupdf4llm`, and `git_commit` (when executed inside a repo).

## Robustness and validation

- Defensive handling of PDF read errors (corrupt PDF or failing pages).
- Timeout + backoff retries for Ollama calls.
- Initial model availability check via `ollama.list()` (skippable with `--skip-model-check`).
- Argument validation at startup (`chunk_overlap < chunk_size`, parameter ranges).
- Per-document language detection by default, with manual override through `--language`.
- Chunk-based topic fallback when the model fails to return parseable topics.
- Compact topic-map retry when the full-document topic map is not parseable.
- Exact + semantic (bigram) deduplication to reduce repetition.
- `type` and `difficulty` normalization against out-of-schema model outputs.
- Substring verification of `context_source` (exposed as `context_source_verified`).

## Tests and lint

```bash
pip install -r pipeline/requirements-dev.txt
pytest           # uses config in pyproject.toml
ruff check .
```

Continuous integration runs both on `push` and `pull_request` against `main` via `.github/workflows/ci.yml` on Python 3.10 / 3.11 / 3.12.
