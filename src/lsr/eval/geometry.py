"""Latent-space statistics: is a pooled embedding still telling documents apart?

Mean pooling over longer audio averages more content into one vector. If that
pulls every document toward a common direction, retrieval fails for a reason
visible in the embeddings alone, before any query is scored. These numbers
track that per condition:

- ``raw_norm``: norm before L2 normalisation (averaging shrinks it when the
  averaged parts disagree).
- ``pairwise_cos``: mean cosine between distinct documents; 1.0 is collapse.
- ``effective_rank``: exp(entropy) of the normalised singular-value spectrum
  of the centred, L2-normalised embeddings (Roy & Vetterli 2007); how many
  directions the documents actually use.
- ``top1_var``: share of variance on the first principal direction.

and, given queries with one gold document each, the score geometry:
``gold_cos``, ``best_negative_cos`` and their ``margin``.
"""

from __future__ import annotations

import numpy as np


def _l2n(x: np.ndarray) -> np.ndarray:
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-12, None)


def _spectrum(e: np.ndarray) -> tuple[float, float]:
    sv = np.linalg.svd(e - e.mean(axis=0), compute_uv=False)
    var = sv**2 / max((sv**2).sum(), 1e-12)
    p = sv / max(sv.sum(), 1e-12)
    p = p[p > 0]
    return float(np.exp(-(p * np.log(p)).sum())), float(var[0])


def embedding_stats(raw: np.ndarray, n_sub: int | None = None, n_draws: int = 50,
                    seed: int = 0) -> dict[str, float]:
    """``n_sub``: also report the spectrum on random subsets of that size.

    Effective rank is bounded by n - 1, so conditions with different document
    counts are only comparable at a common ``n_sub``.
    """
    raw = np.asarray(raw, dtype=np.float64)
    n = len(raw)
    e = _l2n(raw)
    s = e.sum(axis=0)
    eff, top1 = _spectrum(e)
    out = {
        "n": n,
        "raw_norm": float(np.linalg.norm(raw, axis=1).mean()),
        "pairwise_cos": float((s @ s - n) / (n * (n - 1))) if n > 1 else float("nan"),
        "effective_rank": eff,
        "top1_var": top1,
    }
    if n_sub is not None and 1 < n_sub <= n:
        rng = np.random.default_rng(seed)
        draws = [_spectrum(e[rng.choice(n, n_sub, replace=False)]) for _ in range(n_draws)]
        out["effective_rank_sub"] = float(np.mean([d[0] for d in draws]))
        out["top1_var_sub"] = float(np.mean([d[1] for d in draws]))
    return out


def score_stats(scores: np.ndarray, gold: np.ndarray) -> dict[str, float]:
    """``scores`` [Q, D] cosine; ``gold`` [Q] column index of each query's gold document."""
    scores = np.asarray(scores, dtype=np.float64)
    rows = np.arange(len(gold))
    g = scores[rows, gold]
    neg = scores.copy()
    neg[rows, gold] = -np.inf
    b = neg.max(axis=1)
    return {"gold_cos": float(g.mean()), "best_negative_cos": float(b.mean()),
            "margin": float((g - b).mean())}
