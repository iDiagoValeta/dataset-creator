"""Ollama client wrapper: model verification, chat calls, and embedding functions."""

import time
from typing import Any

import ollama  # noqa: F401 — also exposed via generate_dataset.ollama for monkeypatching

from engine._config import (
    DEFAULT_OLLAMA_RETRIES,
    DEFAULT_OLLAMA_RETRY_BACKOFF,
    DEFAULT_OLLAMA_TIMEOUT,
    Chunk,
    Topic,
    logger,
)
from engine._prompts import try_parse_json_payload
from engine._text import cosine_similarity


def _ollama_list_models_entries(listing: Any) -> list[Any]:
    """Normalize Client.list(): dict JSON vs ollama>=0.4 ListResponse with .models."""
    if isinstance(listing, dict):
        raw = listing.get("models", [])
        return list(raw) if isinstance(raw, (list, tuple)) else []
    models = getattr(listing, "models", None)
    if isinstance(models, (list, tuple)):
        return list(models)
    return []


def _ollama_entry_model_name(entry: Any) -> str | None:
    if isinstance(entry, dict):
        name = entry.get("model") or entry.get("name")
    else:
        name = getattr(entry, "model", None) or getattr(entry, "name", None)
    if name is None:
        return None
    return str(name)


def verify_ollama_model(model: str) -> None:
    """Abort early if Ollama is unreachable or the requested model is missing."""
    try:
        client = ollama.Client(timeout=DEFAULT_OLLAMA_TIMEOUT)
        listing = client.list()
    except Exception as exc:
        raise RuntimeError(
            f"No se pudo conectar con Ollama. ¿Está el servicio corriendo? Detalle: {exc}"
        ) from exc

    models_payload = _ollama_list_models_entries(listing)
    available: set[str] = set()
    for entry in models_payload:
        name = _ollama_entry_model_name(entry)
        if name:
            available.add(name)
            available.add(name.split(":", 1)[0])

    if model in available or model.split(":", 1)[0] in available:
        return

    hint = ", ".join(sorted(available)) or "(ninguno)"
    raise RuntimeError(
        f"Modelo '{model}' no disponible en Ollama. Modelos detectados: {hint}. "
        f"Instálalo con: ollama pull {model}"
    )


def get_topic_embedding(topic: Topic, model: str) -> list[float]:
    """Compute an embedding for a topic description using Ollama."""
    prompt = f"{topic.name}. {topic.summary}. {' '.join(topic.keywords)}"
    try:
        client = ollama.Client(timeout=DEFAULT_OLLAMA_TIMEOUT)
        resp = client.embeddings(model=model, prompt=prompt)
        raw = resp.get("embedding", []) if isinstance(resp, dict) else resp.embedding
        return list(raw)
    except Exception as exc:
        logger.warning("Fallo al obtener embedding para topic '%s': %s", topic.name, exc)
        return []


def get_chunk_embedding(chunk: Chunk, model: str, cache: dict[str, list[float]]) -> list[float]:
    """Return the embedding for a chunk, computing and caching it on first access."""
    if chunk.chunk_id in cache:
        return cache[chunk.chunk_id]
    try:
        client = ollama.Client(timeout=DEFAULT_OLLAMA_TIMEOUT)
        resp = client.embeddings(model=model, prompt=chunk.text)
        raw = resp.get("embedding", []) if isinstance(resp, dict) else resp.embedding
        embedding = list(raw)
    except Exception as exc:
        logger.warning("Fallo al obtener embedding para chunk '%s': %s", chunk.chunk_id, exc)
        embedding = []
    cache[chunk.chunk_id] = embedding
    return embedding


def score_chunk_for_topic_semantic(
    chunk: Chunk,
    topic: Topic,
    topic_embedding: list[float],
    embedding_model: str,
    cache: dict[str, list[float]],
) -> float:
    """Cosine similarity between a topic embedding and a chunk embedding."""
    if not topic_embedding:
        return 0.0
    chunk_embedding = get_chunk_embedding(chunk, embedding_model, cache)
    return cosine_similarity(topic_embedding, chunk_embedding)


def call_ollama_json(
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    seed: int | None = None,
    max_retries: int = DEFAULT_OLLAMA_RETRIES,
    backoff_secs: float = DEFAULT_OLLAMA_RETRY_BACKOFF,
) -> tuple[dict[str, Any], str]:
    """Execute Ollama chat call expecting a JSON object, with retry on transient errors."""
    options: dict[str, Any] = {"temperature": temperature}
    if seed is not None:
        options["seed"] = seed

    last_error: BaseException | None = None
    for attempt in range(1, max_retries + 1):
        call_start = time.time()
        try:
            client = ollama.Client(timeout=DEFAULT_OLLAMA_TIMEOUT)
            response = client.chat(
                model=model,
                messages=messages,
                format="json",
                options=options,
                think=False,
            )
            elapsed = time.time() - call_start
            logger.debug("Ollama '%s' OK en %.2fs (intento %s)", model, elapsed, attempt)
            content = response.get("message", {}).get("content", "")
            return try_parse_json_payload(content), content
        except ollama.ResponseError as exc:
            # 4xx-like errors are usually not transient (bad prompt, missing model).
            logger.error("Ollama ResponseError en modelo '%s' (intento %s/%s): %s",
                         model, attempt, max_retries, exc)
            return {"items": []}, ""
        except Exception as exc:
            elapsed = time.time() - call_start
            last_error = exc
            logger.warning(
                "Error transitorio llamando Ollama '%s' (intento %s/%s tras %.2fs): %s",
                model, attempt, max_retries, elapsed, exc,
            )
            if attempt < max_retries:
                time.sleep(backoff_secs * attempt)

    logger.error("Ollama falló tras %s intentos: %s", max_retries, last_error)
    return {"items": []}, ""
