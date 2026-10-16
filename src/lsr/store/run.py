"""Run manifests and the cached-embedding format.

This module is the boundary that lets four incompatible dependency stacks share
one metric path. A backend writes a run directory; the core reads it.

```
runs/<run_id>/
  embeddings.npz      queries [Nq, D] float32, documents [Nd, D] float32, L2-normalised
  ids.json            {"query_ids": [...], "document_ids": [...]}  order matches the arrays
  provenance.json     RunManifest, below
  metrics.json        written by the core after scoring
```

A run directory is self-describing on purpose: six months later the only thing
that distinguishes two numbers is what this file recorded.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class RunManifest:
    """Everything needed to interpret, trust or reproduce one run.

    Recorded *before* the metric, not after. A number whose manifest is
    incomplete is a number that cannot be defended in review.
    """

    run_id: str
    task: str
    retriever: str
    git_sha: str
    git_dirty: bool
    config_hash: str
    seed: int
    n_queries: int
    n_documents: int
    checkpoint_sha256: str | None = None
    perturbation: str | None = None
    hardware: str | None = None
    wall_clock_s: float | None = None
    notes: str | None = None


def current_manifest(**fields) -> RunManifest:
    """Build a manifest, filling git state and hardware from the environment.

    TODO(you): ``git_dirty`` matters more than it looks. A run from a dirty tree
    is not reproducible, and the honest options are to record the flag or to
    refuse the run outright. Decide which, and make it the same everywhere.
    """
    raise NotImplementedError


def write_embeddings(
    run_dir: Path,
    query_emb: np.ndarray,
    document_emb: np.ndarray,
    query_ids: list[str],
    document_ids: list[str],
) -> None:
    """Write the cached-embedding format.

    Validates before writing: row counts match the id lists, dtype is float32,
    no NaNs, and every row is unit norm to within tolerance. A backend that
    forgot to normalise should fail here, loudly, not three steps later as a
    quietly wrong metric.
    """
    raise NotImplementedError


def read_embeddings(run_dir: Path) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    """Read back what ``write_embeddings`` wrote, with the same validation."""
    raise NotImplementedError
