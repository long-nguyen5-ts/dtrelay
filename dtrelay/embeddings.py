"""Local embeddings, served from the same daemon as the chat completions.

Claude Code cannot produce embedding vectors at any price -- there is no
interface for it -- so DeepTutor's knowledge bases would otherwise need an API
key purely for indexing. This runs a small ONNX model in-process instead, so
the whole system stays key-free.

BAAI/bge-small-en-v1.5: 384 dimensions, ~130MB, CPU-only. The model is loaded
lazily on the first request so starting the relay stays instant.
"""
from __future__ import annotations

import logging
import threading

log = logging.getLogger(__name__)

EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"
EMBED_MODEL_ID = "bge-small-en-v1.5"
EMBED_DIM = 384

_model = None
_lock = threading.Lock()


def _get_model():
    """Load once, on first use. Guarded because uvicorn serves from a pool."""
    global _model
    if _model is None:
        with _lock:
            if _model is None:
                from fastembed import TextEmbedding

                log.info("loading embedding model %s", EMBED_MODEL_NAME)
                _model = TextEmbedding(model_name=EMBED_MODEL_NAME)
                log.info("embedding model ready (%d dims)", EMBED_DIM)
    return _model


def normalize_input(value) -> list[str]:
    """Coerce the OpenAI `input` field to a list of strings.

    Pre-tokenized input (arrays of ints) is rejected rather than coerced: we
    have no detokenizer, and embedding the string "[1, 2, 3]" would silently
    produce a meaningless vector.
    """
    if isinstance(value, str):
        texts = [value]
    elif isinstance(value, list):
        texts = value
    else:
        raise ValueError("input must be a string or a list of strings")
    if not texts:
        raise ValueError("input must not be empty")
    for t in texts:
        if not isinstance(t, str):
            raise ValueError("token-array input is not supported; send text")
        if not t.strip():
            raise ValueError("input must not be empty")
    return texts


def embed_texts(texts: list[str]) -> list[list[float]]:
    return [vec.tolist() for vec in _get_model().embed(texts)]
