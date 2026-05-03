"""
Global configuration: dataclasses, constants, and logging setup.
No internal dependencies — safe to import from any other module.
"""

import logging
import os
from dataclasses import dataclass
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent  # pipeline/engine/_config.py → pipeline/
DEFAULT_SOURCE_DIR = BASE_DIR / "input"
DEFAULT_OUTPUT_PATH = BASE_DIR / "output" / "dataset.jsonl"
DEFAULT_DEBUG_DIR = BASE_DIR / "run_logs"

DEFAULT_MODEL = os.getenv("OLLAMA_DATASET_MODEL") or os.getenv("OLLAMA_RAG_MODEL") or "gemma4:e4b"
DEFAULT_LANGUAGE = os.getenv("DATASET_LANGUAGE", "auto")
DEFAULT_CHUNK_SIZE = int(os.getenv("DATASET_CHUNK_SIZE", "3500"))
DEFAULT_CHUNK_OVERLAP = int(os.getenv("DATASET_CHUNK_OVERLAP", "350"))
DEFAULT_NUM_TOPICS = int(os.getenv("DATASET_NUM_TOPICS", "8"))
DEFAULT_QUESTIONS_PER_TOPIC = int(os.getenv("DATASET_QUESTIONS_PER_TOPIC", "6"))
DEFAULT_SPLIT = os.getenv("DATASET_SPLIT", "0.8,0.1,0.1")
DEFAULT_TEMPERATURE = float(os.getenv("DATASET_TEMPERATURE", "0.2"))
DEFAULT_SEED = int(os.getenv("DATASET_SEED", "42"))
DEFAULT_MAX_DOC_CONTEXT_CHARS = int(os.getenv("DATASET_MAX_DOC_CONTEXT_CHARS", "110000"))
DEFAULT_MAX_TOPIC_CONTEXT_CHARS = int(os.getenv("DATASET_MAX_TOPIC_CONTEXT_CHARS", "24000"))
DEFAULT_QUALITY_GATE = os.getenv("DATASET_QUALITY_GATE", "strict")
DEFAULT_OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT_SECS", "300"))
DEFAULT_OLLAMA_RETRIES = int(os.getenv("OLLAMA_MAX_RETRIES", "3"))
DEFAULT_OLLAMA_RETRY_BACKOFF = float(os.getenv("OLLAMA_RETRY_BACKOFF_SECS", "2.0"))
DEFAULT_EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL") or "embeddinggemma:latest"
DEFAULT_RETRIEVAL_MODE = os.getenv("DATASET_RETRIEVAL", "hybrid")
DEFAULT_JUDGE_MODE = os.getenv("DATASET_JUDGE", "off")
DEFAULT_JUDGE_MODEL = os.getenv("OLLAMA_JUDGE_MODEL") or DEFAULT_MODEL

MIN_TOPIC_CONTEXT_CHARS = 9000
VALID_TYPES: frozenset[str] = frozenset({"factual", "conceptual", "inference", "compare", "definition"})
VALID_DIFFICULTIES: frozenset[str] = frozenset({"easy", "medium", "hard"})
VALID_QUALITY_GATES: frozenset[str] = frozenset({"strict", "balanced", "off"})
VALID_JUDGE_MODES: frozenset[str] = frozenset({"audit", "filter", "off"})

STOPWORDS: frozenset[str] = frozenset({
    "cual", "como", "para", "este", "esta", "estos", "estas", "sobre",
    "desde", "hasta", "entre", "donde", "cuando", "que", "del", "las",
    "los", "una", "uno", "unos", "unas", "con", "sin", "por",
    "what", "which", "how", "does", "that", "with", "from", "have", "has",
    "this", "these", "those", "the", "and", "for",
})
# Back-compat alias.
_STOPWORDS_SET = STOPWORDS

LANGUAGE_MARKERS: dict[str, set[str]] = {
    "en": {"the", "and", "of", "to", "in", "is", "are", "that", "with", "for",
           "from", "this", "these", "which", "system", "software", "computer"},
    "es": {"el", "la", "los", "las", "de", "del", "que", "en", "es", "son",
           "para", "con", "por", "como", "sistema", "software", "computadora"},
    "ca": {"el", "la", "els", "les", "de", "del", "que", "en", "es", "son",
           "per", "amb", "com", "sistema", "programari", "ordinador"},
    "fr": {"le", "la", "les", "des", "de", "du", "que", "dans", "est", "sont",
           "pour", "avec", "par", "comme", "systeme", "logiciel", "ordinateur"},
    "pt": {"o", "a", "os", "as", "de", "do", "da", "que", "em", "e", "sao",
           "para", "com", "por", "como", "sistema", "software", "computador"},
}
LANGUAGE_NAMES: dict[str, str] = {
    "en": "English", "es": "Spanish", "ca": "Catalan", "fr": "French", "pt": "Portuguese",
}

logging.basicConfig(
    level=logging.getLevelName(os.getenv("DATASET_LOG_LEVEL", "INFO")),
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("dataset_creator")
logging.getLogger("httpx").setLevel(logging.WARNING)


@dataclass(frozen=True)
class Chunk:
    """Represents a text chunk extracted from a PDF."""
    chunk_id: str
    document: str
    text: str


@dataclass(frozen=True)
class Topic:
    """Represents a document topic inferred by the model."""
    topic_id: str
    name: str
    summary: str
    keywords: list[str]
