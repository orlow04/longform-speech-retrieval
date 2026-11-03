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

import json
import platform
import shutil
import subprocess
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


_NORM_TOL = 1e-3


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args], cwd=Path(__file__).resolve().parent,
            capture_output=True, text=True, check=True, timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip()


def _hardware() -> str:
    desc = f"{platform.node()} {platform.system()} {platform.machine()}"
    if shutil.which("nvidia-smi"):
        try:
            gpus = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True, text=True, check=True, timeout=10,
            ).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            gpus = ""
        if gpus:
            desc += " | " + ", ".join(gpus.splitlines())
    return desc


def current_manifest(**fields) -> RunManifest:
    """Build a manifest, filling git state and hardware from the environment.

    Git state is read from the repository containing this package — the code
    that actually ran — not from the working directory.

    A dirty tree is **recorded, not refused**: refusing would make every quick
    experiment a commit, and the flag is enough to disqualify a run from a
    results table. Untracked files count as dirty, since an uncommitted script
    may be the thing that ran. Outside a git checkout, ``git_sha`` is
    ``"unknown"`` and ``git_dirty`` is ``True``.

    Fields passed explicitly win, so a backend running in a container without
    the repo can supply the SHA it was built from.
    """
    if "git_sha" not in fields:
        fields["git_sha"] = _git("rev-parse", "HEAD") or "unknown"
    if "git_dirty" not in fields:
        status = _git("status", "--porcelain")
        fields["git_dirty"] = status is None or status != ""
    fields.setdefault("hardware", _hardware())
    return RunManifest(**fields)


def _validate(
    query_emb: np.ndarray,
    document_emb: np.ndarray,
    query_ids: list[str],
    document_ids: list[str],
) -> None:
    for label, emb, ids in (
        ("query", query_emb, query_ids),
        ("document", document_emb, document_ids),
    ):
        if emb.ndim != 2:
            raise ValueError(f"{label} embeddings must be 2-D, got shape {emb.shape}")
        if emb.dtype != np.float32:
            raise ValueError(f"{label} embeddings must be float32, got {emb.dtype}")
        if emb.shape[0] != len(ids):
            raise ValueError(f"{len(ids)} {label} ids but {emb.shape[0]} {label} rows")
        if len(set(ids)) != len(ids):
            raise ValueError(f"duplicate {label} ids")
        if not np.all(np.isfinite(emb)):
            raise ValueError(f"{label} embeddings contain NaN or inf")
        norms = np.linalg.norm(emb, axis=1)
        bad = np.flatnonzero(np.abs(norms - 1.0) > _NORM_TOL)
        if bad.size:
            raise ValueError(
                f"{bad.size} {label} rows are not unit norm (e.g. row {bad[0]}: "
                f"{norms[bad[0]]:.4f}); normalise in the backend"
            )
    if query_emb.shape[1] != document_emb.shape[1]:
        raise ValueError(
            f"dimension mismatch: queries {query_emb.shape[1]}, documents {document_emb.shape[1]}"
        )


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
    query_ids, document_ids = list(query_ids), list(document_ids)
    _validate(query_emb, document_emb, query_ids, document_ids)
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    np.savez(run_dir / "embeddings.npz", queries=query_emb, documents=document_emb)
    (run_dir / "ids.json").write_text(
        json.dumps({"query_ids": query_ids, "document_ids": document_ids})
    )


def read_embeddings(run_dir: Path) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    """Read back what ``write_embeddings`` wrote, with the same validation."""
    run_dir = Path(run_dir)
    with np.load(run_dir / "embeddings.npz", allow_pickle=False) as npz:
        query_emb, document_emb = npz["queries"], npz["documents"]
    ids = json.loads((run_dir / "ids.json").read_text())
    query_ids, document_ids = ids["query_ids"], ids["document_ids"]
    _validate(query_emb, document_emb, query_ids, document_ids)
    return query_emb, document_emb, query_ids, document_ids
