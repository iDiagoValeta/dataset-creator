# dataset-creator

Repositorio para generar datasets de preguntas/respuestas a partir de PDFs usando Ollama en local.

## Objetivo

Dado un PDF en `pipeline/input/`, el pipeline genera un JSONL estructurado con preguntas deducibles del texto, respuestas argumentables y trazabilidad de contexto.

## Estructura

```text
dataset-creator/
├── CLAUDE.md
├── README.md
└── pipeline/
    ├── generate_dataset.py
    ├── requirements.txt
    ├── input/
    │   └── .gitkeep
    ├── output/
    │   └── .gitkeep
    └── run_logs/
        └── .gitkeep
```

## Requisitos

- Python 3.10+
- Ollama levantado localmente
- Modelo descargado en Ollama (default recomendado: `gemma4:e2b`)

## Instalacion

```bash
pip install -r pipeline/requirements.txt
```

## Variables de entorno

- `OLLAMA_DATASET_MODEL` (fallback: `OLLAMA_RAG_MODEL`, default `gemma4:e2b`)
- `DATASET_LANGUAGE` (default `es`)
- `DATASET_CHUNK_SIZE` (default `3500`)
- `DATASET_CHUNK_OVERLAP` (default `350`)
- `DATASET_NUM_TOPICS` (default `8`)
- `DATASET_QUESTIONS_PER_TOPIC` (default `6`)
- `DATASET_SPLIT` (default `0.8,0.1,0.1`)
- `DATASET_TEMPERATURE` (default `0.2`)
- `DATASET_SEED` (default `42`)
- `DATASET_MAX_DOC_CONTEXT_CHARS` (default `110000`)
- `DATASET_MAX_TOPIC_CONTEXT_CHARS` (default `24000`)
- `OLLAMA_TIMEOUT_SECS` (default `300`)
- `DATASET_LOG_LEVEL` (default `INFO`)

## Uso rapido

1. Coloca PDFs en `pipeline/input/`.
2. Ejecuta:

```bash
python pipeline/generate_dataset.py
```

Salida:

- `pipeline/output/dataset.jsonl`
- `pipeline/output/dataset_train.jsonl`
- `pipeline/output/dataset_val.jsonl`
- `pipeline/output/dataset_test.jsonl`
- `pipeline/output/dataset.meta.json`
- `pipeline/run_logs/<pdf>.json`

## Ejemplos CLI

```bash
python pipeline/generate_dataset.py --model gemma4:e2b --num-topics 10 --questions-per-topic 8
python pipeline/generate_dataset.py --language en --split 0.7,0.15,0.15
python pipeline/generate_dataset.py --source-dir pipeline/input --output pipeline/output/tesis_es.jsonl
```

## Formato de cada item

Cada linea JSONL incluye:

- `id`
- `question`
- `answer`
- `type` (`factual|conceptual|inference|compare|definition`)
- `difficulty` (`easy|medium|hard`)
- `context_source`
- `topic`
- `topic_id`
- `topic_keywords`
- `document`
- `created_at`
- `context_excerpt`

## Robustez y validaciones

- Manejo defensivo de errores de lectura PDF (incluyendo PDF corrupto o pagina fallida).
- Timeout configurable para llamadas a Ollama.
- Validacion de argumentos al inicio (`chunk_overlap < chunk_size`, rangos de parametros).
- Fallback de topicos por chunks si el modelo no devuelve topicos parseables.
- Deduplicacion exacta y semantica (bigrams) para reducir repeticiones.

