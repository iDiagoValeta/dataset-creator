---
description:
alwaysApply: true
---

# CLAUDE.md

Guia de trabajo para este repositorio.

## 1. Contexto del proyecto

Este repo genera datasets sinteticos desde PDFs con modelos locales servidos en Ollama.
La referencia arquitectonica es `localOllamaRAG`, pero aqui el objetivo es dataset creation, no interfaz RAG de usuario.

## 2. Reglas operativas

1. Responder en espanol.
2. No hacer commit/push sin aprobacion explicita del usuario.
3. Mantener estilo script-first con `main()` claro.
4. Priorizar artefactos reproducibles (JSONL y metadata JSON), no solo stdout.
5. Si se cambian rutas, parametros o formato de salida, actualizar tambien `README.md`.
6. Flujo actual: solo texto. No incluir OCR, imagenes ni tablas.

## 3. Estructura esperada

```text
rag/
  generate_dataset.py
  requirements.txt
  pdfs/
  datasets/
  debug_dataset/
```

## 4. Convenciones de codigo

- MODULE MAP al inicio de modulos no triviales.
- Secciones con separadores claros.
- Imports ordenados: stdlib -> third-party -> local.
- Variables de entorno para defaults importantes.
- Dependencias opcionales con `try/except` + flag booleano.
- Docstrings en ingles y mensajes de usuario en espanol.

## 5. Flujo funcional

1. Cargar PDFs desde `rag/pdfs/`.
2. Extraer texto (`pymupdf4llm` preferido, `pypdf` fallback).
3. Trocear texto en chunks con overlap.
4. Generar mapa de topicos por documento usando contexto amplio.
5. Generar preguntas por topico sin repeticion semantica (`think=False`).
6. Deduplicar por pregunta.
7. Exportar JSONL principal y splits train/val/test.
8. Guardar metadata y debug por documento/topico.
