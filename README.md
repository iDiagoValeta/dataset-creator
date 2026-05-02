# dataset-creator

Generate supervised QA datasets from PDFs using local Ollama models.

## Why This Repo Exists

I needed high-quality QA datasets for my own projects, especially in **Spanish (Castilian)** and **Valencian**, and the pipelines I had available didn't cover that workflow well. Until now, I was building those datasets **manually with NotebookLM**: slow, non-reproducible, and hard to iterate on.

This repository automates that workflow end-to-end: extract text from PDFs, detect the document language, map topics, generate grounded QA pairs with a local Ollama model, deduplicate, and export ready-to-train JSONL splits, all while keeping traceability back to the source context.

The pipeline extracts text, detects the document language, maps topics, generates grounded Q/A pairs, applies quality checks, deduplicates rows, and exports JSONL train/val/test splits with traceability back to source chunks. Optionally, `--judge audit` adds a final factuality audit over accepted rows without changing the main dataset.

## Quick Start

Requirements:

- Python 3.10+
- Ollama running locally
- Generator model: `gemma4:e4b`
- Embedding model for default hybrid retrieval: `embeddinggemma:latest`
- Optional judge model for `--judge audit`: `gemma4:e4b`

```bash
pip install -r pipeline/requirements.txt
pip install -r pipeline/requirements-dev.txt

ollama pull gemma4:e4b
ollama pull embeddinggemma:latest

python pipeline/generate_dataset.py
```

Put source PDFs in `pipeline/input/`. Generated files are written to `pipeline/output/`.

To regenerate from scratch:

```powershell
python pipeline/generate_dataset.py --clean; python pipeline/generate_dataset.py
```

## Example

`examples/` contains a complete run over three arXiv papers on ML carbon footprint, AI governance in healthcare, and redundancy-aware robot learning.

```text
examples/
  carbon_footprint_ml_training.pdf      source PDF 1
  global_ai_governance_healthcare.pdf   source PDF 2
  redundancy_aware_robot_learning.pdf   source PDF 3
  dataset.jsonl          24 accepted QA items
  dataset_train.jsonl    19 train items
  dataset_val.jsonl       2 validation items
  dataset_test.jsonl      3 test items
  dataset.meta.json      run metadata, quality stats, audit stats
  dataset.rejected.jsonl 48 rejected items
  dataset.judged.jsonl   24 items with judge scores
```

Generated with `gemma4:e4b`, `quality_gate=strict`, `retrieval=hybrid`, `embedding_model=embeddinggemma:latest`, `--judge audit`, and English auto-detection. All 24 accepted items have verified literal context (`context_source_verified=true`), zero mojibake, zero cross-chunk contexts, and zero verbatim answers. The judge found 12 pass and 12 fail among the accepted rows.

## Common Commands

```bash
# Default generation
python pipeline/generate_dataset.py

# Process only specific PDFs
python pipeline/generate_dataset.py --only-doc thesis.pdf
python pipeline/generate_dataset.py --only-doc a.pdf,b.pdf

# Force output language
python pipeline/generate_dataset.py --language es
python pipeline/generate_dataset.py --language ca

# Change model or volume
python pipeline/generate_dataset.py --model llama3.1:8b --num-topics 10 --questions-per-topic 8

# Retrieval modes
python pipeline/generate_dataset.py --retrieval lexical
python pipeline/generate_dataset.py --retrieval semantic
python pipeline/generate_dataset.py --retrieval hybrid --embedding-model nomic-embed-text

# Factuality judge audit (does not filter dataset.jsonl)
python pipeline/generate_dataset.py --judge audit
python pipeline/generate_dataset.py --quality-gate strict --judge audit
python pipeline/generate_dataset.py --judge audit --judge-model llama3.1:8b

# Resume or estimate
python pipeline/generate_dataset.py --resume
python pipeline/generate_dataset.py --dry-run

# Clean generated output
python pipeline/generate_dataset.py --clean-dry-run
python pipeline/generate_dataset.py --clean
```

## Main Options

| Flag | Default | Purpose |
|---|---:|---|
| `--model` | `gemma4:e4b` | Ollama generator model |
| `--source-dir` | `pipeline/input` | Folder containing PDFs |
| `--output` | `pipeline/output/dataset.jsonl` | Main JSONL output |
| `--language` | `auto` | Auto-detect, or force `es`, `ca`, `en`, etc. |
| `--num-topics` | `8` | Max topics per document |
| `--questions-per-topic` | `6` | Requested QA rows per topic |
| `--retrieval` | `hybrid` | `lexical`, `semantic`, or `hybrid` chunk selection |
| `--embedding-model` | `embeddinggemma:latest` | Ollama model for semantic/hybrid retrieval |
| `--quality-gate` | `strict` | `strict`, `balanced`, or `off` |
| `--judge` | `off` | `off` or `audit`; audits final accepted rows without filtering |
| `--judge-model` | `gemma4:e4b` | Ollama judge model; override with `OLLAMA_JUDGE_MODEL` or the CLI flag |
| `--split` | `0.8,0.1,0.1` | Train/val/test ratios |
| `--resume` | off | Reuse per-document checkpoints |
| `--clean` | off | Remove generated JSON/JSONL artifacts |

## Quality And Metadata

`strict` mode keeps only rows with verified `context_source`, non-circular answers, clean extracted text, topic alignment, and enough support for factual/definition/compare/inference answers. `context_source_verified=true` means the context is a literal substring of the extracted document context after whitespace normalization; the pipeline does not infer or repair source context from answer overlap. Verified context is expanded to complete sentence-bounded evidence when possible, so RAG rows favor complete literal support instead of clipped 300-character fragments. Rejected rows are written to `dataset.rejected.jsonl` with a `rejection_reason`.

The deterministic quality gate also rejects:

- **Mojibake** — double-encoded UTF-8 sequences (e.g. `â€™`, `Î¸`) in any Q/A field. Encoding is normalized at PDF extraction time using `ftfy` when available, or a built-in replacement map as fallback.
- **Cross-chunk context** (`cross_chunk_context`) — `context_source` that contains an internal chunk marker, meaning the LLM copied text spanning two source chunks. Markers are also stripped from generated text before quality checks.
- **Verbatim answers** (`verbatim_answer`) — answers with ≥ 75 % bigram overlap with `context_source` and length comparable to the context (a proxy for copy-paste without reformulation).
- Context that appears to start or end mid-sentence, broken figure references, degraded formula notation, replacement characters, and answers that add too many unsupported content terms for RAG-style factual rows.

`--judge audit` runs an Ollama judge after quality filtering and deduplication. It audits accepted rows one document at a time for clearer progress and smaller judge batches, then writes one combined `dataset.judged.jsonl`. It treats `context_source` as the only allowed evidence and checks whether the answer is factually deducible from that literal context. The judge applies deterministic pre-checks (mojibake, internal chunk markers, verbatim answers) before calling the LLM, and blocking reasons (`cross_chunk_context`, `extraction_artifact`, `overly_extractive`, `truncated_context`, `unsupported_detail`) force a `fail` decision regardless of the LLM score. A noisy `weak_context` or `judge_error` reason is reconciled to `factual` only when the LLM also returns `decision=pass` and all component scores are at least 0.8. The minimum passing score across all three components (context quality, answer support, question quality) is 0.6. The judge writes `dataset.judged.jsonl` with the original accepted rows plus `judge_score`, `judge_context_quality`, `judge_answer_support`, `judge_question_quality`, `judge_decision`, `judge_reasons`, `judge_explanation`, and `judge_model`. It does not modify `dataset.jsonl`.

The metadata file records:

- generation counts and split sizes
- detected document languages
- quality gate stats and rejection reasons
- judge audit stats when enabled, including overall and component score averages
- deduplication counts
- topic coverage audit and warnings
- runtime/package/git reproducibility info

Per-document debug files in `pipeline/run_logs/*.json` include the detected language, topic-map attempts, `chunk_count`, topic contexts, raw generation attempts, and parsed topic/item payloads. `--resume` reuses these debug files plus `*.items.jsonl` checkpoints to preserve metadata counts without regenerating Q/A rows.

## User-Supplied Topics Or Questions

Use `--topics-file` to bypass automatic topic mapping. The file can be YAML or plain text, one topic per line.

```bash
python pipeline/generate_dataset.py --topics-file pipeline/input/topics.txt
python pipeline/generate_dataset.py --topics-file pipeline/input/topics.yaml
```

Use `--questions-file` to answer fixed seed questions from the document context.

```bash
python pipeline/generate_dataset.py --questions-file pipeline/input/questions.txt
```

`--topics-file` and `--questions-file` are mutually exclusive.

## Output Schema

Each JSONL row contains:

```text
id, question, answer, type, difficulty,
context_source, context_source_verified, context_excerpt,
topic, topic_id, topic_keywords,
document, document_language, source_chunk_ids, created_at
```

For RAG-oriented use, the core supervised triple is `question`, literal `context_source`, and `answer`; the other fields are traceability and audit metadata.

When `--judge audit` is enabled, `dataset.judged.jsonl` adds:

```text
judge_score, judge_context_quality, judge_answer_support,
judge_question_quality, judge_decision, judge_reasons,
judge_explanation, judge_model
```

## Tests

```bash
pytest
ruff check .
```
