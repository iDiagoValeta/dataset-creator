# dataset-creator

Generate supervised QA datasets from PDFs using local Ollama models.

## Why this repo exists

I needed high-quality QA datasets for my own projects, especially in **Spanish (Castilian)** and **Valencian**, and the pipelines I had available didn't cover that workflow well. Until now, I was building those datasets **manually with NotebookLM** — slow, non-reproducible, and hard to iterate on.

This repository automates that workflow end-to-end: extract text from PDFs, detect the document language, map topics, generate grounded QA pairs with a local Ollama model, deduplicate, and export ready-to-train JSONL splits — all while keeping traceability back to the source context.

## Example

The `examples/` folder contains a complete generation run on a research paper about Random Forest models in high-frequency trading ([arXiv 2412.15448](https://arxiv.org/abs/2412.15448)):

```
examples/
  2412.15448v2.pdf       — source document (8 chunks in this run)
  dataset.jsonl          — 21 QA items, all context-verified
  dataset_train.jsonl    — 16 items (train split)
  dataset_val.jsonl      —  2 items (val split)
  dataset_test.jsonl     —  3 items (test split)
  dataset.meta.json      — run metadata, quality stats, and audit stats
  dataset.rejected.jsonl — 11 items rejected by the quality gate
```

Generated with `gemma4:e4b`, strict quality gate, language auto-detected as English, and default `--retrieval hybrid` using `embeddinggemma:latest`. The run produced 42 raw items, 21 accepted items after quality filtering and deduplication, 11 rejected rows, 7 accepted topics, and no audit warnings. See `dataset.meta.json` for the full parameter set.

## How it works

The pipeline runs sequentially per PDF:

**1. Extract and chunk**  
Text is extracted with `pymupdf4llm` (markdown-aware) and falls back to `pypdf` on failure. The text is split into overlapping chunks (`chunk_size` characters with `chunk_overlap` overlap).

**2. Detect language**  
The extracted text is scanned for language-marker words (English, Spanish, Catalan, French, Portuguese) plus distinctive characters (`ñ`, `¿`, `ç`, `ã`, …). The detected language is stored per document and used for all model prompts unless `--language` forces a specific code.

**3. Build a topic map**  
The model receives a wide context window (up to `max_doc_context_chars` characters) and is asked to return a JSON list of topics with name, summary, and keywords. If the model output is not parseable, a compact retry is attempted. If topics are still too generic or garbage, the pipeline falls back to section headings extracted directly from the text. Topics are distributed uniformly across the document so that results, discussion, and conclusions sections are covered, not just the introduction.

**4. Generate QA pairs per topic**  
For each topic, the pipeline selects the most relevant chunks (`hybrid` retrieval by default, combining lexical scoring and embedding-based cosine similarity) and sends them to the model with a generation prompt in the detected language. The model returns up to `questions_per_topic` JSON items, each with `question`, `answer`, `type`, `difficulty`, and `context_source`.

**5. Verify context source**  
Each item's `context_source` is checked against the topic context by substring match. If the model's suggested source is not found verbatim, the pipeline searches the context for the answer words and returns the best matching fragment. The result is stored as `context_source_verified: true/false`.

**6. Apply the quality gate**  
Before deduplication, the quality gate filters items:
- `strict` (default): rejects items with unverified `context_source`, circular answers (answer adds no new information over the question), PDF extraction artifacts, first-person paper voice, truncated source fragments, clear topic mismatches, and weakly supported compare/inference answers.
- `balanced`: applies deterministic quality checks and circular-answer filtering, but keeps unverified context sources.
- `off`: no filtering.  

Rejected items are written to `dataset.rejected.jsonl` with a `rejection_reason` field.

**7. Backfill low-coverage topics**  
If a discovered topic has too few accepted items, the pipeline makes a deterministic second generation pass for that topic before final deduplication.

**8. Deduplicate**  
Exact duplicates and near-duplicates (bigram overlap on questions and answers) are removed from the accepted set.

**9. Split and export**  
The deduplicated items are shuffled deterministically (using `seed`) and split into train/val/test according to `--split`. When rows contain `topic_id`, the splitter keeps small validation/test splits more topic-aware. Each split is written to its own JSONL file plus the combined `dataset.jsonl`.

**10. Write metadata**  
`dataset.meta.json` records all counts, quality stats, audit stats, parameters, detected languages, and runtime information for reproducibility.

## Structure

```text
dataset-creator/
├── CLAUDE.md
├── README.md
├── pyproject.toml
├── .env.example
├── examples/                  — sample PDF and generated dataset
├── .github/
│   └── workflows/ci.yml
├── pipeline/
│   ├── generate_dataset.py    — orchestrator and CLI entry point
│   ├── engine/                — pipeline submodules
│   │   ├── _config.py         — constants, dataclasses, logging
│   │   ├── _text.py           — text transforms, language detection
│   │   ├── _pdf.py            — PDF extraction and chunking
│   │   ├── _prompts.py        — LLM prompt builders and JSON parser
│   │   ├── _ollama.py         — Ollama client wrapper
│   │   ├── _topics.py         — topic parsing, validation, retrieval
│   │   ├── _quality.py        — quality gate and deduplication
│   │   ├── _generation.py     — QA generation per topic
│   │   ├── _export.py         — JSONL I/O, splits, metadata
│   │   └── _cli.py            — argparse, validation, dry-run
│   ├── requirements.txt
│   ├── requirements-dev.txt
│   ├── input/                 — drop PDFs here
│   ├── output/                — generated JSONL and metadata
│   └── run_logs/              — per-document debug logs and checkpoints
└── tests/
    └── test_generate_dataset.py
```

## Requirements

- Python 3.10+
- [Ollama](https://ollama.com/) running locally
- A model pulled in Ollama (recommended default: `gemma4:e4b`)

## Installation

```bash
# Clone and enter the repo
git clone https://github.com/nadiv/dataset-creator.git
cd dataset-creator

# Install dependencies
pip install -r pipeline/requirements.txt
pip install -r pipeline/requirements-dev.txt   # adds pytest

# Or with pyproject
pip install -e .
pip install -e ".[dev]"
```

Copy `.env.example` to `.env` (or export variables in your shell) and adjust as needed.

## Quick start

1. Start Ollama and pull a model:

```bash
ollama pull gemma4:e4b
```

2. Drop one or more PDFs into `pipeline/input/`.

3. Run:

```bash
python pipeline/generate_dataset.py
```

Output appears in `pipeline/output/`.

## CLI reference

### Basic generation

```bash
python pipeline/generate_dataset.py
```

Processes all PDFs in `pipeline/input/` with default settings.

### Common flags

| Flag | Default | Description |
|---|---|---|
| `--model MODEL` | `gemma4:e4b` | Ollama model to use |
| `--source-dir DIR` | `pipeline/input` | Folder with PDFs |
| `--output FILE` | `pipeline/output/dataset.jsonl` | Output JSONL path |
| `--num-topics N` | `8` | Max topics per document |
| `--questions-per-topic N` | `6` | QA items per topic |
| `--chunk-size N` | `3500` | Chunk size in characters |
| `--chunk-overlap N` | `350` | Overlap between chunks |
| `--split A,B,C` | `0.8,0.1,0.1` | Train/val/test proportions |
| `--temperature F` | `0.2` | Sampling temperature |
| `--seed N` | `42` | Random seed for reproducibility |
| `--language CODE` | `auto` | Output language (`auto`, `es`, `en`, `ca`, …) |
| `--quality-gate MODE` | `strict` | Quality filter mode (`strict`, `balanced`, `off`) |
| `--retrieval MODE` | `hybrid` | Chunk selection method: `lexical`, `semantic`, or `hybrid` |
| `--embedding-model MODEL` | `embeddinggemma:latest` | Ollama model used for embeddings when `--retrieval semantic` or `--retrieval hybrid` |
| `--topics-file PATH` | — | YAML or plain-text file with user-defined topics. Skips automatic topic mapping. |
| `--questions-file PATH` | — | Plain-text file with one seed question per line. Generates answers for each question. Mutually exclusive with `--topics-file`. |
| `--only-doc NAMES` | — | Process only these PDFs (comma-separated filename or stem) |
| `--resume` | off | Skip documents that already have a checkpoint |
| `--skip-model-check` | off | Skip the initial `ollama.list()` availability check |
| `--dry-run` | off | Estimate chunks and Ollama calls without generating |
| `--clean-dry-run` | off | List generated files that `--clean` would delete |
| `--clean` | off | Delete generated files from output and log folders |

### Examples

```bash
# Use a different model and generate more items
python pipeline/generate_dataset.py --model llama3.1:8b --num-topics 10 --questions-per-topic 8

# Force Spanish output for all PDFs
python pipeline/generate_dataset.py --language es

# Force Valencian / Catalan output
python pipeline/generate_dataset.py --language ca

# Process only one PDF from a multi-PDF input folder
python pipeline/generate_dataset.py --only-doc thesis.pdf

# Process two specific PDFs
python pipeline/generate_dataset.py --only-doc a.pdf,b.pdf

# Custom output path and split
python pipeline/generate_dataset.py \
  --output pipeline/output/thesis_es.jsonl \
  --split 0.7,0.15,0.15

# Estimate cost before committing to a full run
python pipeline/generate_dataset.py --dry-run

# Resume an interrupted run without reprocessing finished documents
python pipeline/generate_dataset.py --resume

# Apply a looser quality filter (keeps unverified items)
python pipeline/generate_dataset.py --quality-gate balanced

# Disable quality filtering entirely
python pipeline/generate_dataset.py --quality-gate off

# Preview which files --clean would delete, then clean
python pipeline/generate_dataset.py --clean-dry-run
python pipeline/generate_dataset.py --clean

# Greedy decoding, skip model check (useful in CI or scripted pipelines)
python pipeline/generate_dataset.py --temperature 0.0 --skip-model-check
```

### Quality gate modes

| Mode | What it keeps |
|---|---|
| `strict` | Only items with verified `context_source`, non-circular answers, clean extraction text, topic alignment, and enough source support. This is the default. |
| `balanced` | Keeps unverified context sources but still removes deterministic quality failures and circular answers. |
| `off` | Passes everything through. |

Rejected items are always written to `dataset.rejected.jsonl` with a `rejection_reason` field such as `unverified_context_source`, `circular_answer`, `artifact_or_reference_noise`, `truncated_context_source`, `first_person_answer`, `topic_mismatch`, or `insufficient_context_support`.

### Dataset audit

After quality filtering, backfill, and deduplication, `dataset.meta.json` includes an `audit` block with accepted/rejected counts by topic, split coverage by topic, topics with no accepted rows, topics below the minimum coverage threshold, and audit warnings. This makes small or uneven datasets easier to spot before training.

### Language detection

By default (`--language auto`) the pipeline detects each PDF's language from the extracted text and asks the model to generate topics, questions, and answers in that language. Pass an explicit code to override for all PDFs:

```bash
python pipeline/generate_dataset.py --language es   # Spanish
python pipeline/generate_dataset.py --language ca   # Catalan / Valencian
python pipeline/generate_dataset.py --language en   # English
python pipeline/generate_dataset.py --language fr   # French
python pipeline/generate_dataset.py --language pt   # Portuguese
```

### Checkpointing and resume

Every processed document writes its items to `pipeline/run_logs/<stem>.items.jsonl`. If a run is interrupted, restart with `--resume` and the pipeline will skip any document that already has a checkpoint, reloading its items from disk instead of calling the model again.

### Chunk retrieval

By default the pipeline uses `--retrieval hybrid`, combining lexical topic scoring with embedding-based cosine similarity. Use `lexical` for a faster no-embedding run, or `semantic` when you only want embedding similarity.

```bash
# Pull the default embedding model first
ollama pull embeddinggemma:latest

# Force lexical retrieval
python pipeline/generate_dataset.py --retrieval lexical

# Use a different model for embeddings
python pipeline/generate_dataset.py --retrieval hybrid --embedding-model nomic-embed-text
```

Chunk embeddings are cached in memory across topics within the same document, so each chunk is embedded only once per run. The lexical fast-path (exact topic-name match in chunk text) is always attempted first regardless of retrieval mode.

### User-supplied topics and seed questions

#### `--topics-file` — define topics manually

Provide a YAML or plain-text file to bypass automatic topic mapping entirely. The topics you specify are used for all processed PDFs.

**YAML format** (requires `pip install pyyaml`):

```yaml
topics:
  - name: "Métodos estadísticos"
    summary: "Análisis de regresión y tests de hipótesis"   # optional
    keywords: ["regresión", "hipótesis", "p-value"]          # optional
  - name: "Resultados del modelo"
```

**Plain text format** (one topic per line):

```
Métodos estadísticos
Resultados del modelo
Conclusiones
```

```bash
python pipeline/generate_dataset.py --topics-file pipeline/input/topics.yaml
python pipeline/generate_dataset.py --topics-file pipeline/input/topics.txt
```

Topics defined in the file get `topic_id` values prefixed with `user-` (e.g. `user-00`, `user-01`).

#### `--questions-file` — seed questions with generated answers

Provide a plain-text file with one question per line. The pipeline skips topic mapping entirely and generates an answer for each question from the most relevant document context.

```
¿Cuál es la hipótesis central del artículo?
¿Qué métodos de validación se utilizan?
¿Cuáles son las principales limitaciones del estudio?
```

```bash
python pipeline/generate_dataset.py --questions-file pipeline/input/preguntas.txt
```

Seed-question items have `topic_id` values prefixed with `seed-` (e.g. `seed-00`). All other fields — `context_source`, `context_source_verified`, quality gate, deduplication — work exactly as in normal mode.

`--topics-file` and `--questions-file` are mutually exclusive. Both are compatible with `--only-doc`, `--resume`, and `--quality-gate`.

## Environment variables

All flags have environment variable equivalents. Variables take effect when the corresponding flag is not supplied on the command line.

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_DATASET_MODEL` | `gemma4:e4b` | Primary generator model |
| `OLLAMA_TIMEOUT_SECS` | `300` | Ollama call timeout in seconds |
| `OLLAMA_MAX_RETRIES` | `3` | Retries on transient Ollama errors |
| `OLLAMA_RETRY_BACKOFF_SECS` | `2.0` | Linear backoff between retries |
| `OLLAMA_EMBEDDING_MODEL` | `embeddinggemma:latest` | Ollama model for semantic/hybrid embeddings |
| `DATASET_RETRIEVAL` | `hybrid` | Chunk retrieval mode (`lexical`, `semantic`, or `hybrid`) |
| `DATASET_LANGUAGE` | `auto` | Language code or `auto` |
| `DATASET_CHUNK_SIZE` | `3500` | Chunk size in characters |
| `DATASET_CHUNK_OVERLAP` | `350` | Overlap between chunks |
| `DATASET_NUM_TOPICS` | `8` | Max topics per document |
| `DATASET_QUESTIONS_PER_TOPIC` | `6` | QA items per topic |
| `DATASET_SPLIT` | `0.8,0.1,0.1` | Train/val/test split ratios |
| `DATASET_TEMPERATURE` | `0.2` | Sampling temperature |
| `DATASET_SEED` | `42` | Random seed |
| `DATASET_MAX_DOC_CONTEXT_CHARS` | `110000` | Max context for topic mapping |
| `DATASET_MAX_TOPIC_CONTEXT_CHARS` | `24000` | Max context for QA generation |
| `DATASET_QUALITY_GATE` | `strict` | Quality filter mode |
| `DATASET_LOG_LEVEL` | `INFO` | Logging level (`DEBUG`/`INFO`/`WARNING`/`ERROR`) |

## Output files

| File | Description |
|---|---|
| `dataset.jsonl` | All accepted items |
| `dataset_train.jsonl` | Train split |
| `dataset_val.jsonl` | Validation split |
| `dataset_test.jsonl` | Test split |
| `dataset.meta.json` | Run metadata, quality stats, audit stats, parameters, runtime |
| `dataset.rejected.jsonl` | Items filtered by the quality gate |
| `run_logs/<stem>.json` | Raw model outputs per document and topic |
| `run_logs/<stem>.items.jsonl` | Per-document checkpoint (used by `--resume`) |

## Item schema

Each line in the JSONL files is a JSON object with these fields:

| Field | Type | Description |
|---|---|---|
| `id` | string | Unique item identifier |
| `question` | string | Generated question |
| `answer` | string | Generated answer |
| `type` | string | `factual` / `conceptual` / `inference` / `compare` / `definition` |
| `difficulty` | string | `easy` / `medium` / `hard` |
| `context_source` | string | Literal fragment from the source context supporting the answer |
| `context_source_verified` | bool | `true` if the fragment appears verbatim in the context window |
| `topic` | string | Topic name the item was generated under |
| `topic_id` | string | Topic identifier (e.g. `topic-00`) |
| `topic_keywords` | array | Keywords associated with the topic |
| `document` | string | Source PDF filename |
| `document_language` | string | Detected or forced language code |
| `source_chunk_ids` | array | Chunk IDs where the answer source was found |
| `created_at` | string | ISO 8601 timestamp |
| `context_excerpt` | string | Diagnostic excerpt from the source chunk around the supporting fragment |

## Tests and lint

```bash
pip install -r pipeline/requirements-dev.txt
pytest
ruff check .
```

Continuous integration runs on every push and pull request against `main` via `.github/workflows/ci.yml`, testing Python 3.10, 3.11, and 3.12.
