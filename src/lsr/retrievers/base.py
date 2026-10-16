"""The retriever protocol.

One protocol, N backends. The protocol exists so that a comparison can change
the speech-to-text bridge while holding the task, the pool and the metric fixed.

Two ways to be a retriever:

1. **In-process** — implement ``Retriever`` and return embeddings directly.
   Convenient, and only possible when the backend's dependencies can be
   installed alongside the core.
2. **Out-of-process** — write the cached-embedding format described in
   ``lsr.store`` from whatever environment or container the backend needs, and
   let the core score it. This is the contract that matters; see docs/DESIGN.md.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from lsr.tasks.base import Document, Query


@runtime_checkable
class Retriever(Protocol):
    """Embeds queries and documents into one shared space.

    Scoring is not the retriever's job. A retriever that cannot express itself
    as two embedding matrices and a dot product — a late-interaction scorer, for
    instance — implements ``ScoringRetriever`` instead.
    """

    name: str

    @property
    def dim(self) -> int:
        """Dimensionality of the shared space."""
        ...

    def embed_queries(self, queries: list[Query]) -> np.ndarray:
        """Return ``[len(queries), dim]``, L2-normalised, rows aligned to input.

        Normalising here rather than at scoring time keeps every backend's
        embeddings directly comparable and makes a dot product the cosine.
        """
        ...

    def embed_documents(self, documents: list[Document]) -> np.ndarray:
        """Return ``[len(documents), dim]``, L2-normalised, rows aligned to input."""
        ...


@runtime_checkable
class ScoringRetriever(Protocol):
    """For retrievers whose score is not a dot product of two vectors.

    Late interaction (max-sim over per-token or per-chunk embeddings) is the
    case this exists for. Implementations own their own scoring, and therefore
    their own efficiency.
    """

    name: str

    def score(self, queries: list[Query], documents: list[Document]) -> np.ndarray:
        """Return a ``[len(queries), len(documents)]`` score matrix, higher is better."""
        ...


def score_dense(query_emb: np.ndarray, document_emb: np.ndarray) -> np.ndarray:
    """Score L2-normalised embeddings into a ``[Nq, Nd]`` cosine matrix.

    One matmul. Not a Python loop with a ``.item()`` per cell — that pattern is
    what made the inherited ``build_similarity_matrix`` unusable past a few
    thousand rows, and the pool here is meant to grow.

    TODO(you): decide whether this blocks over documents for memory. At 2k x 2k
    it is irrelevant; at 100k documents x 4096 dims it is not.
    """
    raise NotImplementedError
