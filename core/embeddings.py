"""
Alfred AI - Embedding Engine
Local embedding model using sentence-transformers.
Runs entirely on-device — no API calls, no data leaves your machine.
"""

import time
import hashlib
import threading
from typing import Union
from collections import OrderedDict
from sentence_transformers import SentenceTransformer
from .config import config
from .logging import get_logger

logger = get_logger("embeddings")


class EmbeddingEngine:
    """Local embedding engine using nomic-embed-text with LRU caching."""

    def __init__(self, model_name: str = None, cache_size: int = 256):
        self.model_name = model_name or config.EMBEDDING_MODEL
        self._model = None
        self._lock = threading.Lock()  # SentenceTransformer.encode() is not thread-safe

        # LRU cache for query embeddings — avoids re-embedding identical queries
        self._cache: OrderedDict[str, list[float]] = OrderedDict()
        self._cache_size = cache_size
        self._cache_hits = 0
        self._cache_misses = 0

    @property
    def model(self) -> SentenceTransformer:
        """Lazy-load the model on first use."""
        if self._model is None:
            logger.info(f"Loading embedding model: {self.model_name}")
            start = time.time()
            self._model = SentenceTransformer(
                self.model_name,
                trust_remote_code=True,
                cache_folder=str(config.MODELS_CACHE_DIR),
            )
            elapsed = time.time() - start
            logger.info(f"Model loaded in {elapsed:.1f}s")
        return self._model

    @property
    def dimension(self) -> int:
        """Embedding vector dimension."""
        return config.EMBEDDING_DIMENSION

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of documents (for storage).

        Adds the 'search_document: ' prefix for nomic-embed-text
        which differentiates between documents and queries.
        """
        prefixed = [f"{config.EMBEDDING_PREFIX_DOCUMENT}{t}" for t in texts]
        with self._lock:
            embeddings = self.model.encode(prefixed, normalize_embeddings=True)
        return embeddings.tolist()

    def embed_query(self, query: str) -> list[float]:
        """Embed a single query (for search).

        Adds the 'search_query: ' prefix for nomic-embed-text
        which optimizes retrieval by differentiating query vs document.

        Uses LRU cache to avoid re-embedding identical queries.
        """
        cache_key = hashlib.sha256(query.encode()[:400]).hexdigest()[:16]

        if cache_key in self._cache:
            self._cache_hits += 1
            # Move to end (most recently used)
            self._cache.move_to_end(cache_key)
            return self._cache[cache_key]

        self._cache_misses += 1
        prefixed = f"{config.EMBEDDING_PREFIX_SEARCH}{query}"
        with self._lock:
            embedding = self.model.encode([prefixed], normalize_embeddings=True)
        result = embedding[0].tolist()

        # Store in cache, evict oldest if full
        self._cache[cache_key] = result
        if len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)

        return result

    def embed_raw(self, text: str) -> list[float]:
        """Embed text without any prefix. For custom use cases."""
        with self._lock:
            embedding = self.model.encode([text], normalize_embeddings=True)
        return embedding[0].tolist()

    @property
    def cache_stats(self) -> dict:
        """Get embedding cache statistics."""
        total = self._cache_hits + self._cache_misses
        return {
            "cache_size": len(self._cache),
            "max_size": self._cache_size,
            "hits": self._cache_hits,
            "misses": self._cache_misses,
            "hit_rate": f"{self._cache_hits / total:.0%}" if total > 0 else "0%",
        }


# Singleton instance
_engine = None

def get_embedding_engine() -> EmbeddingEngine:
    """Get or create the singleton embedding engine."""
    global _engine
    if _engine is None:
        _engine = EmbeddingEngine()
    return _engine
