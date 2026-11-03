"""A frozen sentence encoder for text queries and text documents."""

from __future__ import annotations

import numpy as np


class SentenceEncoder:
    """Wraps a sentence-transformers model; returns L2-normalised float32.

    ``query_prefix`` is prepended to queries only. BGE v1.5 recommends
    "Represent this sentence for searching relevant passages: " for short
    queries against passages; it is a config choice, recorded with the run.
    """

    def __init__(self, model: str = "BAAI/bge-base-en-v1.5", device: str | None = None,
                 query_prefix: str = "", batch_size: int = 64):
        from sentence_transformers import SentenceTransformer

        self.name = model.rsplit("/", 1)[-1]
        self.model = SentenceTransformer(model, device=device)
        self.query_prefix = query_prefix
        self.batch_size = batch_size

    @property
    def dim(self) -> int:
        return self.model.get_sentence_embedding_dimension()

    def _encode(self, texts: list[str]) -> np.ndarray:
        emb = self.model.encode(texts, batch_size=self.batch_size, normalize_embeddings=True,
                                convert_to_numpy=True, show_progress_bar=False)
        return emb.astype(np.float32)

    def embed_texts(self, texts: list[str], *, as_query: bool = False) -> np.ndarray:
        if as_query and self.query_prefix:
            texts = [self.query_prefix + t for t in texts]
        return self._encode(texts)
