"""Long spoken context: a question, a long recording, a gold time span.

Every long-audio dataset — native or assembled from short clips — is built into
the same directory format, so the retrievers, the metric and the diagnostics
cannot tell them apart except by what the data itself contains:

```
<build_dir>/
  recordings.jsonl   {"recording_id", "audio", "duration_s", "kind", "source",
                      "transcript"?, "pieces"?}
  questions.jsonl    {"query_id", "recording_id", "text", "gold_start_s", "gold_end_s"}
  transcripts/<recording_id>.jsonl   {"start_s", "end_s", "text"}   time-aligned
```

- ``kind`` is ``"native"`` (one continuous recording) or ``"concatenated"``
  (assembled from shorter clips). It is the variable under study, so it is
  recorded per recording rather than inferred.
- ``pieces`` lists the clips a concatenated recording was assembled from, with
  their spans, so variants (shuffled, smoothed, spliced) can re-assemble them.
- ``transcript`` is optional; with it, the clean-text ceiling works on windows.
- Paths are relative to the build directory.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                          encoding="utf-8")


@dataclass
class LongAudioTask:
    """Text question -> window of a long recording.

    Documents are the windows ``segmenter`` cuts from every recording. A window
    is relevant when it overlaps the question's gold span by at least
    ``min(min_overlap_s, gold duration)`` seconds.

    Two pools, never mixed: rank with ``lsr.eval.ranks_within_groups`` using
    ``query_group`` / ``document_group`` for "find it in this recording", or
    across every window for the archive. The archive is only meaningful when a
    question identifies one recording; ``archive_valid`` is False when the same
    question text is asked of several recordings (CORAAL-QA's "When was the
    speaker born?" has one answer per interview).

    ``gold_units`` keeps only questions whose gold span has that granularity
    (``questions.jsonl``'s ``gold_unit``), so that datasets are compared at
    matched target length; a question without the field counts as "span".
    ``gold_span="paragraph"`` scores every question that has a
    ``paragraph_span`` against it instead — the usual Spoken-SQuAD setup.
    """

    build_dir: Path
    segmenter: object  # lsr.longaudio.Segmenter
    min_overlap_s: float = 5.0
    gold_units: tuple[str, ...] | None = None  # e.g. ("sentence",); None keeps all
    gold_span: str = "gold"  # "gold", or "paragraph" to score against paragraph_span
    query_modality: str = "text"
    document_modality: str = "audio"
    name: str = field(init=False)

    def __post_init__(self) -> None:
        from lsr.longaudio import as_documents, overlap_s
        from lsr.tasks.base import Query

        self.build_dir = Path(self.build_dir)
        self.name = f"longform_tq2ac_{self.build_dir.name}_{self.segmenter.name}"
        self.recordings = {r["recording_id"]: r
                           for r in read_jsonl(self.build_dir / "recordings.jsonl")}
        questions = read_jsonl(self.build_dir / "questions.jsonl")
        if self.gold_units is not None:
            questions = [q for q in questions
                         if q.get("gold_unit", "span") in self.gold_units]
            self.name += "_gold-" + "+".join(self.gold_units)
        if self.gold_span == "paragraph":
            questions = [{**q, "gold_start_s": q["paragraph_span"][0],
                          "gold_end_s": q["paragraph_span"][1]}
                         for q in questions if q.get("paragraph_span")]
            self.name += "_span-paragraph"
        elif self.gold_span != "gold":
            raise ValueError(f"gold_span must be 'gold' or 'paragraph', not {self.gold_span!r}")

        self.segments = {}
        self.document_group: dict[str, str] = {}
        self._documents = []
        for rid, rec in self.recordings.items():
            segs = self.segmenter.segment(str(self.build_dir / rec["audio"]), rid,
                                          duration_s=rec["duration_s"])
            self.segments[rid] = segs
            self._documents += as_documents(segs)
            self.document_group.update({s.id: rid for s in segs})

        self.question_rows = {q["query_id"]: q for q in questions}
        self._queries, self._qrels = [], {}
        self.query_group: dict[str, str] = {}
        self.gold_span: dict[str, tuple[float, float]] = {}
        for q in questions:
            span = (q["gold_start_s"], q["gold_end_s"])
            need = min(self.min_overlap_s, span[1] - span[0])
            self._queries.append(Query(id=q["query_id"], text=q["text"]))
            self._qrels[q["query_id"]] = {
                s.id for s in self.segments[q["recording_id"]]
                if overlap_s(s.start_s, s.end_s, *span) >= need}
            self.query_group[q["query_id"]] = q["recording_id"]
            self.gold_span[q["query_id"]] = span

        recordings_per_text: dict[str, set[str]] = {}
        for q in questions:
            recordings_per_text.setdefault(_norm(q["text"]), set()).add(q["recording_id"])
        self.n_ambiguous_questions = sum(
            1 for q in questions if len(recordings_per_text[_norm(q["text"])]) > 1)
        self.archive_valid = self.n_ambiguous_questions == 0
        self._transcripts: dict[str, list[dict]] = {}

    def queries(self):
        yield from self._queries

    def documents(self):
        yield from self._documents

    def qrels(self) -> dict[str, set[str]]:
        return {k: set(v) for k, v in self._qrels.items()}

    def has_transcripts(self) -> bool:
        return all("transcript" in r for r in self.recordings.values())

    def transcript(self, recording_id: str) -> list[dict]:
        if recording_id not in self._transcripts:
            rel = self.recordings[recording_id]["transcript"]
            self._transcripts[recording_id] = read_jsonl(self.build_dir / rel)
        return self._transcripts[recording_id]

    def transcript_in_span(self, recording_id: str, start_s: float, end_s: float) -> str:
        """Gold text whose midpoint falls in the span — the clean-text ceiling."""
        return " ".join(u["text"] for u in self.transcript(recording_id)
                        if start_s <= (u["start_s"] + u["end_s"]) / 2 < end_s)


def _norm(text: str) -> str:
    return " ".join(text.lower().split())
