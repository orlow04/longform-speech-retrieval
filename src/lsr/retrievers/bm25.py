"""BM25 over text, numpy only.

The lexical baseline. On datasets whose questions were written while reading
the passage (SQuAD and everything built from it), word overlap is unusually
high and BM25 is strong; on questions written about spontaneous speech it
should not be. The gap is one of the measurements, not just a baseline.
"""

from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np

_TOKEN = re.compile(r"[a-z0-9']+")
_STOP = frozenset(
    "a an and are as at be by did do does for from had has have how i in is it its of on or "
    "that the their they this to was were what when where which who whom why with you".split())


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOP]


def bm25_scores(queries: list[str], documents: list[str], k1: float = 1.5, b: float = 0.75
                ) -> np.ndarray:
    """``[len(queries), len(documents)]`` BM25 scores, higher is better."""
    docs = [Counter(tokenize(d)) for d in documents]
    lengths = np.array([sum(d.values()) for d in docs], dtype=float)
    avg = lengths.mean() if len(docs) and lengths.mean() > 0 else 1.0
    df = Counter(t for d in docs for t in d)
    n = len(docs)
    idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}
    norm = k1 * (1 - b + b * lengths / avg)
    scores = np.zeros((len(queries), n))
    for i, q in enumerate(queries):
        for t in set(tokenize(q)):
            if t not in idf:
                continue
            tf = np.array([d.get(t, 0) for d in docs], dtype=float)
            scores[i] += idf[t] * tf * (k1 + 1) / (tf + norm)
    return scores
