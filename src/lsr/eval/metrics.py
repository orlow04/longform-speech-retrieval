"""The metric core.

Every backend, every task and every perturbation condition reports through this
module, so its conventions are the ones that have to be written down:

- **Ranks are 1-based.** The gold document's rank is one plus the number of
  documents scored strictly above it. Strictly above, so ties are resolved
  optimistically — state it, because it changes the number.
- **Hits@k and Recall@k are not synonyms.** They coincide only when a query has
  exactly one relevant document. Take qrels as sets and compute each honestly.
- **A metric without its pool size is not a result.** Carry ``n_queries`` and
  ``n_documents`` in the output dict.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RankingResult:
    """Metrics plus the context needed to interpret them."""

    metrics: dict[str, float]
    ranks: np.ndarray
    n_queries: int
    n_documents: int


def ranks_from_scores(
    scores: np.ndarray,
    qrels: dict[str, set[str]],
    query_ids: list[str],
    document_ids: list[str],
) -> np.ndarray:
    """Rank of the best-ranked relevant document per query, 1-based.

    Note what this does *not* do: assume the gold document sits on the diagonal.
    The inherited implementation took ``similarity_matrix[i][i]`` as ground
    truth, which silently ties the metric to row order and makes a shuffled pool
    produce nonsense. Ground truth comes from qrels and ids.

    TODO(you): decide the policy for a query whose relevant documents are all
    outside the pool — drop it and report the count, or rank it last. Either is
    defensible; silently dropping is not.
    """
    raise NotImplementedError


def compute_metrics(ranks: np.ndarray, ks: tuple[int, ...] = (1, 5, 10, 50)) -> dict[str, float]:
    """Hits@k, MRR, MAP, mean and median rank from 1-based ranks.

    TODO(you): with one relevant document per query, AP = 1/rank so MAP = MRR.
    Decide whether to emit both (clear, redundant) or one with a note.
    """
    raise NotImplementedError


def bootstrap_ci(
    ranks: np.ndarray,
    metric: str = "Hits@1",
    n_resamples: int = 1000,
    confidence: float = 0.95,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap CI over the query pool.

    Cheap, needs no retraining, and turns every table from a point estimate into
    something a reviewer can assess. Resamples queries, never documents — the
    pool is fixed by the task.
    """
    raise NotImplementedError
