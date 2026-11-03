"""What a retrieval task is.

A task's identity includes its query side, its document side and its pool. Two
tasks over the same corpus that differ in direction are different tasks, and the
difference belongs in the name — see docs/DESIGN.md, "Tasks name their direction
explicitly".
"""

from __future__ import annotations

from collections import Counter
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

    Checks: a non-empty name; no duplicate ids on either side; every query and
    document populates exactly the field its modality calls for; a non-empty
    pool; every query has at least one relevant document; and qrels reference
    only known query and document ids.

    Collects every problem and raises one ``ValueError`` listing them, so a
    broken task is diagnosed in one pass rather than one fix at a time.
    """
    problems: list[str] = []
    if not getattr(task, "name", ""):
        problems.append("task has no name")

    queries = list(task.queries())
    documents = list(task.documents())
    qrels = task.qrels()

    def side(items, modality: Modality, label: str) -> set[str]:
        ids = [x.id for x in items]
        dupes = sorted(i for i, n in Counter(ids).items() if n > 1)
        if dupes:
            problems.append(f"duplicate {label} ids: {_show(dupes)}")
        wrong = [
            x.id for x in items
            if (x.text is not None) != (modality == "text")
            or (x.audio_path is not None) != (modality == "audio")
        ]
        if wrong:
            problems.append(
                f"{len(wrong)} {label}s do not set exactly the {modality!r} field: {_show(wrong)}"
            )
        return set(ids)

    query_ids = side(queries, task.query_modality, "query")
    doc_ids = side(documents, task.document_modality, "document")

    if not queries:
        problems.append("no queries")
    if not documents:
        problems.append("empty pool")

    unknown_q = sorted(set(qrels) - query_ids)
    if unknown_q:
        problems.append(f"qrels reference {len(unknown_q)} unknown query ids: {_show(unknown_q)}")
    unknown_d = sorted({d for rel in qrels.values() for d in rel} - doc_ids)
    if unknown_d:
        problems.append(
            f"qrels reference {len(unknown_d)} document ids outside the pool: {_show(unknown_d)}"
        )
    unjudged = sorted(q for q in query_ids if not qrels.get(q))
    if unjudged:
        problems.append(f"{len(unjudged)} queries have no relevant document: {_show(unjudged)}")

    if problems:
        raise ValueError(f"task {getattr(task, 'name', '?')!r} is inconsistent:\n  - "
                         + "\n  - ".join(problems))


def _show(ids: list[str], limit: int = 5) -> str:
    return ", ".join(ids[:limit]) + (" ..." if len(ids) > limit else "")
