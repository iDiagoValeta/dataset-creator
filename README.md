# dataset-creator

Repositorio para generar datasets de entrenamiento/evaluacion a partir de documentos PDF usando un agente local en Ollama.

Inspirado en la estructura de `localOllamaRAG`, pero enfocado en un flujo concreto:

1. Extraer texto de PDFs.
2. Construir mapa global de topicos por documento (con contexto amplio).
3. Generar preguntas por topico sin repeticion semantica.
4. Exportar dataset en JSONL y splits train/val/test.

## Estructura

```text
dataset-creator/
├── CLAUDE.md
├── README.md
└── rag/
    ├── generate_dataset.py
    ├── requirements.txt
    ├── pdfs/
    │   └── .gitkeep
    ├── datasets/
    │   └── .gitkeep
    └── debug_dataset/
        └── .gitkeep
```

## Requisitos

- Python 3.10+
- Ollama ejecutandose localmente
- Un modelo de generacion descargado en Ollama

## Instalacion

```bash
pip install -r rag/requirements.txt
```

## Configuracion por entorno

Variables opcionales:

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

## Uso rapido

1. Coloca tus PDFs en `rag/pdfs/`.
2. Ejecuta:

```bash
python rag/generate_dataset.py
```

Salida:

- `rag/datasets/dataset.jsonl`
- `rag/datasets/dataset_train.jsonl`
- `rag/datasets/dataset_val.jsonl`
- `rag/datasets/dataset_test.jsonl`
- `rag/datasets/dataset.meta.json`
- `rag/debug_dataset/*.json` (salida cruda por chunk para debug)

## Ejemplos CLI

```bash
python rag/generate_dataset.py --model gemma4:e2b --num-topics 10 --questions-per-topic 8
python rag/generate_dataset.py --language en --split 0.7,0.15,0.15
python rag/generate_dataset.py --source-dir rag/pdfs --output rag/datasets/tesis_es.jsonl
```

## Formato de cada item

Cada linea JSONL incluye:

- `id`
- `question`
- `answer`
- `type` (`factual|conceptual|reasoning`)
- `topic`
- `topic_id`
- `topic_keywords`
- `difficulty` (`easy|medium|hard`)
- `document`
- `created_at`
- `context_excerpt`

## Notas

- Si `pymupdf4llm` falla o no esta instalado, se usa fallback a `pypdf`.
- El script usa `think=False` en Ollama para priorizar salida util y estable.
- El flujo actual es solo texto: no usa imagenes ni tablas para generar preguntas.
- Primero se crea un mapa de topicos global por PDF y luego se generan preguntas separadas por topico.
- Si cambias de modelo o prompt, regenera el dataset completo para mantener consistencia.
