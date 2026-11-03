"""Tasks read from the CLSR release's manifest format.

The release (``github.com/193746/CLSR``, audited at 1a03529) stores each split
as Kaldi-style files, one ``<row_id> <value>`` per line, rows aligned by id:

- ``CLSR_aqtc`` (LibriSQA): ``wav.scp`` · ``que.txt`` · ``con.txt``. The audio
  is the **context** — the LibriSpeech utterance whose transcript is
  ``con.txt``. Questions exist only as text. So, despite the variant's name, the
  task it trains and scores is *text question -> spoken context*.
- ``CLSR_aqac`` (SLUE-SQA-5): ``wav_que.scp`` · ``wav_con.scp`` ·
  ``text_que.txt`` · ``text_con.txt``. Both sides have audio and transcripts.

Every combination of sides the files support is a task here, with the direction
in the name: ``tq`` / ``aq`` for a text / audio question, ``tc`` / ``ac`` for a
text / audio context. The text-to-text variant is the clean-text ceiling (the
paper's "BGE" row).

**Pools.** One row per question means a context with several questions appears
several times. Two protocols are offered, and the pool name is part of the task
name so they can never be compared by accident:

- ``unique`` (default): one document per distinct context, keyed by its audio
  path or its text. Each question's qrel is that one document.
- ``rows``: one document per manifest row, and each question's only relevant
  document is its own row. This is what the release's ``evaluate()`` does
  (diagonal ground truth over the whole split), so it is the protocol under
  which the paper's numbers are comparable. A twin context scores identically
  to the gold one; under the pessimistic tie rule in ``lsr.eval`` it is counted
  above the gold, which makes ``rows`` slightly *stricter* than the release,
  whose ``torch.argsort`` breaks ties in unspecified order.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from lsr.tasks.base import Document, Modality, Query

Pool = Literal["unique", "rows"]

# Manifest file per (dataset, side, modality). A missing key means the release
# has no such file: LibriSQA has no question audio.
_FILES: dict[str, dict[tuple[str, Modality], str]] = {
    "librisqa": {
        ("query", "text"): "que.txt",
        ("doc", "text"): "con.txt",
        ("doc", "audio"): "wav.scp",
    },
    # Built by lsr.tasks.spoken_squad; same layout as the release's LibriSQA.
    "spoken_squad": {
        ("query", "text"): "que.txt",
        ("doc", "text"): "con.txt",
        ("doc", "audio"): "wav.scp",
    },
    "slue_sqa5": {
        ("query", "text"): "text_que.txt",
        ("query", "audio"): "wav_que.scp",
        ("doc", "text"): "text_con.txt",
        ("doc", "audio"): "wav_con.scp",
    },
}
_SHORT = {"text": "t", "audio": "a"}


def read_manifest(path: Path) -> dict[str, str]:
    """Read a ``<row_id> <value>`` file, preserving order. Duplicate ids raise."""
    rows: dict[str, str] = {}
    for n, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row_id, _, value = line.partition(" ")
        if row_id in rows:
            raise ValueError(f"{path}:{n}: duplicate row id {row_id!r}")
        rows[row_id] = value.strip()
    return rows


@dataclass
class ManifestTask:
    """A retrieval task over one split of a CLSR-format manifest directory.

    ``split_dir`` holds the manifest files. Relative audio paths are resolved
    against ``audio_root`` when given (the release writes them relative to its
    variant directory, e.g. ``CLSR_aqtc/``), and kept verbatim otherwise.
    """

    dataset: str
    split_dir: Path
    query_modality: Modality
    document_modality: Modality
    pool: Pool = "unique"
    audio_root: Path | None = None
    name: str = field(init=False)
    _queries: list[Query] = field(init=False, repr=False)
    _documents: list[Document] = field(init=False, repr=False)
    _qrels: dict[str, set[str]] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        files = _FILES.get(self.dataset)
        if files is None:
            raise ValueError(f"unknown dataset {self.dataset!r}; known: {sorted(_FILES)}")
        for side, modality in (("query", self.query_modality), ("doc", self.document_modality)):
            if (side, modality) not in files:
                raise ValueError(f"{self.dataset} has no {modality} on the {side} side")

        self.split_dir = Path(self.split_dir)
        split = self.split_dir.name
        self.name = (
            f"{self.dataset}_{_SHORT[self.query_modality]}q2{_SHORT[self.document_modality]}c"
            f"_{split}" + ("_rowpool" if self.pool == "rows" else "")
        )

        q_rows = read_manifest(self.split_dir / files[("query", self.query_modality)])
        d_rows = read_manifest(self.split_dir / files[("doc", self.document_modality)])
        if list(q_rows) != list(d_rows):
            raise ValueError(f"{self.split_dir}: query and context manifests list different rows")

        self._queries = [
            self._make(Query, rid, value, self.query_modality) for rid, value in q_rows.items()
        ]
        self._documents, self._qrels = [], {}
        doc_by_value: dict[str, str] = {}
        for rid, value in d_rows.items():
            if self.pool == "rows":
                doc_id = rid
            elif value in doc_by_value:
                self._qrels[rid] = {doc_by_value[value]}
                continue
            else:
                doc_id = doc_by_value[value] = f"c{len(doc_by_value):06d}"
            self._documents.append(self._make(Document, doc_id, value, self.document_modality))
            self._qrels[rid] = {doc_id}

    def _make(self, cls, item_id: str, value: str, modality: Modality):
        if modality == "text":
            return cls(id=item_id, text=value)
        path = Path(value)
        if self.audio_root is not None and not path.is_absolute():
            path = Path(self.audio_root) / path
        return cls(id=item_id, audio_path=str(path))

    def queries(self) -> Iterator[Query]:
        yield from self._queries

    def documents(self) -> Iterator[Document]:
        yield from self._documents

    def qrels(self) -> dict[str, set[str]]:
        return {k: set(v) for k, v in self._qrels.items()}
