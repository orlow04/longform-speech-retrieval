"""The metric core.

Every backend, every task and every perturbation condition reports through this
module, so its conventions are the ones that have to be written down:

- **Ranks are 1-based.** A query's rank is the position of its best-ranked
  relevant document.
- **Ties are resolved pessimistically.** Every non-relevant document scoring
  *greater than or equal to* the best relevant score is counted above it. The
  inherited code counted only strictly-greater scores, under which a collapsed
  retriever — every embedding identical, every score tied — reports Hits@1 = 1.0.
  A broken pipeline must not be able to look perfect.
- **Only the best relevant rank is kept**, so the metrics here are those that
  depend on it alone: Hits@k, MRR, mean and median rank. MAP and Recall@k need
  every relevant document's rank and are not computed. Where a task has exactly
  one relevant document per query, MAP = MRR and Recall@k = Hits@k, and that is
  a property of the task, not something this module assumes.
- **A metric without its pool size is not a result.** ``RankingResult`` carries
  ``n_queries`` and ``n_documents``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

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
    on_missing: Literal["raise", "last"] = "raise",
) -> np.ndarray:
    """Rank of the best-ranked relevant document per query, 1-based.

    Note what this does *not* do: assume the gold document sits on the diagonal.
    The inherited implementation took ``similarity_matrix[i][i]`` as ground
    truth, which silently ties the metric to row order and makes a shuffled pool
    produce nonsense. Ground truth comes from qrels and ids.

    A query with no relevant document in the pool cannot be ranked. By default
    that raises, listing the offenders — ``check_task`` should have caught it.
    With ``on_missing="last"`` such a query gets rank ``n_documents + 1``: a miss
    at every cutoff, still counted in the denominator. Queries are never dropped.
    """
    scores = np.asarray(scores)
    if scores.shape != (len(query_ids), len(document_ids)):
        raise ValueError(
            f"scores has shape {scores.shape}, expected "
            f"({len(query_ids)}, {len(document_ids)}) from the id lists"
        )
    if not np.all(np.isfinite(scores)):
        raise ValueError("scores contain NaN or inf")

    doc_index = {d: i for i, d in enumerate(document_ids)}
    if len(doc_index) != len(document_ids):
        raise ValueError("document_ids contain duplicates")

    n_docs = len(document_ids)
    ranks = np.empty(len(query_ids), dtype=np.int64)
    missing: list[str] = []
    for qi, qid in enumerate(query_ids):
        relevant = [doc_index[d] for d in qrels.get(qid, ()) if d in doc_index]
        if not relevant:
            missing.append(qid)
            ranks[qi] = n_docs + 1
            continue
        row = scores[qi]
        is_relevant = np.zeros(n_docs, dtype=bool)
        is_relevant[relevant] = True
        best = row[is_relevant].max()
        ranks[qi] = 1 + int(np.count_nonzero((row >= best) & ~is_relevant))

    if missing and on_missing == "raise":
        shown = ", ".join(missing[:10]) + (" ..." if len(missing) > 10 else "")
        raise ValueError(f"{len(missing)} queries have no relevant document in the pool: {shown}")
    return ranks


_HITS = re.compile(r"Hits@(\d+)")


def _metric_values(ranks: np.ndarray, metric: str) -> np.ndarray:
    """Evaluate ``metric`` along the last axis of ``ranks``; works on 1-D or 2-D."""
    if m := _HITS.fullmatch(metric):
        return (ranks <= int(m.group(1))).mean(axis=-1)
    if metric == "MRR":
        return (1.0 / ranks).mean(axis=-1)
    if metric == "mean_rank":
        return ranks.mean(axis=-1)
    if metric == "median_rank":
        return np.median(ranks, axis=-1)
    raise ValueError(f"unknown metric {metric!r}")


def compute_metrics(ranks: np.ndarray, ks: tuple[int, ...] = (1, 5, 10, 50)) -> dict[str, float]:
    """Hits@k, MRR, mean and median rank from 1-based best-relevant ranks.

    MAP is deliberately absent; see the module docstring.
    """
    ranks = np.asarray(ranks)
    if ranks.ndim != 1 or ranks.size == 0:
        raise ValueError("ranks must be a non-empty 1-D array")
    if ranks.min() < 1:
        raise ValueError("ranks are 1-based; got a value below 1")
    names = [f"Hits@{k}" for k in ks] + ["MRR", "mean_rank", "median_rank"]
    return {name: float(_metric_values(ranks, name)) for name in names}


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
    ranks = np.asarray(ranks)
    if ranks.ndim != 1 or ranks.size == 0:
        raise ValueError("ranks must be a non-empty 1-D array")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be in (0, 1)")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, ranks.size, size=(n_resamples, ranks.size))
    values = _metric_values(ranks[idx], metric)
    alpha = (1 - confidence) / 2
    lo, hi = np.quantile(values, [alpha, 1 - alpha])
    return float(lo), float(hi)


def ranks_within_groups(
    scores: np.ndarray,
    qrels: dict[str, set[str]],
    query_ids: list[str],
    document_ids: list[str],
    query_group: dict[str, str],
    document_group: dict[str, str],
) -> np.ndarray:
    """Like ``ranks_from_scores``, but each query competes only within its group.

    For long-form audio the group is the recording: "find the answer in this
    recording" ranks a question against its own recording's segments, not the
    whole archive. Ranks come back in ``query_ids`` order, so the result feeds
    ``compute_metrics`` and ``bootstrap_ci`` unchanged. A query whose group has
    no relevant document raises, as in ``ranks_from_scores``.
    """
    scores = np.asarray(scores)
    docs_by_group: dict[str, list[int]] = {}
    for j, d in enumerate(document_ids):
        docs_by_group.setdefault(document_group[d], []).append(j)
    ranks = np.empty(len(query_ids), dtype=np.int64)
    for qi, q in enumerate(query_ids):
        cols = docs_by_group.get(query_group[q], [])
        if not cols:
            raise ValueError(f"query {q!r}: group {query_group[q]!r} has no documents")
        ranks[qi] = ranks_from_scores(
            scores[qi : qi + 1, cols], qrels, [q], [document_ids[j] for j in cols]
        )[0]
    return ranks
