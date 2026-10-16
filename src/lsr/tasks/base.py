"""What a retrieval task is.

A task's identity includes its query side, its document side and its pool. Two
tasks over the same corpus that differ in direction are different tasks, and the
difference belongs in the name — see docs/DESIGN.md, "Tasks name their direction
explicitly".
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Literal, Protocol

Modality = Literal["text", "audio"]


@dataclass(frozen=True)
class Query:
    """One query.

    Exactly one of ``text`` or ``audio_path`` is set, matching the task's
    ``query_modality``.
    """

    id: str
    text: str | None = None
    audio_path: str | None = None


@dataclass(frozen=True)
class Document:
    """One retrievable unit.

    For a long-audio task a document may be a segment of a larger recording, in
    which case ``parent_id`` names the recording and ``start_s`` / ``end_s``
    locate it. Those three fields are what ``lsr.longaudio`` needs to score
    segment selection against a gold timestamp.
    """

    id: str
    text: str | None = None
    audio_path: str | None = None
    parent_id: str | None = None
    start_s: float | None = None
    end_s: float | None = None


class Task(Protocol):
    """A retrieval task.

    Implementations are responsible for being honest about ``name``: it is what
    appears in every result table, so it must distinguish a question->passage
    task from a passage->reading-of-itself task over the same corpus.
    """

    name: str
    query_modality: Modality
    document_modality: Modality

    def queries(self) -> Iterator[Query]:
        """Yield every query, in a stable order."""
        ...

    def documents(self) -> Iterator[Document]:
        """Yield every document in the candidate pool, in a stable order.

        The pool is the whole thing being ranked. A task that ranks against a
        sampled subset must say so in ``name`` and fix the sample with a seed,
        because a metric is meaningless without the pool size it was measured
        against.
        """
        ...

    def qrels(self) -> dict[str, set[str]]:
        """Map query id -> set of relevant document ids.

        A set, not a single id: do not bake in the one-relevant-per-query
        assumption. Where it does hold, Hits@k and Recall@k coincide and MAP
        equals MRR, but that is a property of the task, not of the metric.
        """
        ...


def check_task(task: Task) -> None:
    """Assert a task is internally consistent before anything is run on it.

    Intended to catch, at minimum: qrels referencing unknown query or document
    ids, queries whose populated field disagrees with ``query_modality``,
    duplicate ids on either side, and an empty pool.

    TODO(you): this is the cheapest bug-catcher in the repo. Write it before the
    first task.
    """
    raise NotImplementedError
