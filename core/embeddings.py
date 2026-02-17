"""
Alfred AI - Embedding Engine
Local embedding model using sentence-transformers.
Runs entirely on-device — no API calls, no data leaves your machine.
"""

import time
from typing import Union
from sentence_transformers import SentenceTransformer
from .config import config
from .logging import get_logger

logger = get_logger("embeddings")


class EmbeddingEngine:
    """Local embedding engine using nomic-embed-text."""

    def __init__(self, model_name: str = None):
        self.model_name = model_name or config.EMBEDDING_MODEL
        self._model = None

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
        embeddings = self.model.encode(prefixed, normalize_embeddings=True)
        return embeddings.tolist()

    def embed_query(self, query: str) -> list[float]:
        """Embed a single query (for search).

        Adds the 'search_query: ' prefix for nomic-embed-text
        which optimizes retrieval by differentiating query vs document.
        """
        prefixed = f"{config.EMBEDDING_PREFIX_SEARCH}{query}"
        embedding = self.model.encode([prefixed], normalize_embeddings=True)
        return embedding[0].tolist()

    def embed_raw(self, text: str) -> list[float]:
        """Embed text without any prefix. For custom use cases."""
        embedding = self.model.encode([text], normalize_embeddings=True)
        return embedding[0].tolist()


# Singleton instance
_engine = None

def get_embedding_engine() -> EmbeddingEngine:
    """Get or create the singleton embedding engine."""
    global _engine
    if _engine is None:
        _engine = EmbeddingEngine()
    return _engine
