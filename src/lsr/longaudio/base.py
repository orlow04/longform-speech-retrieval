"""Retrieval in long-form audio: segmentation, selection, and what follows.

The direction this repository exists for. A long recording is segmented, the
segments are ranked against a query, the top ones become the context for a
downstream audio language model, and the whole chain is measured — retrieval
quality, answer quality, and latency.

Two things to get right from the start, because they are where published numbers
in this area go wrong:

- **Segment selection has its own ground truth.** If the gold answer occupies a
  known time span, then "did the selected segment contain it" is measurable
  independently of whether the downstream model then answered correctly. Measure
  both; a pipeline that answers correctly from the wrong segment is not working.
- **Downstream audio models truncate.** Qwen-Audio and Qwen2-Audio cap at 30 s
  (3,000 mel frames). A 40-second selected segment is silently cut, and a
  "no-retrieval" baseline fed a 30-minute recording sees its first 30 seconds —
  which makes that baseline a truncation baseline, not a long-context one. Record
  how many seconds actually reached the model in every arm.
"""

from __future__ import annotations

import wave
from dataclasses import dataclass
from typing import Protocol

from lsr.tasks.base import Document, Query


@dataclass(frozen=True)
class Segment:
    """A span of a longer recording, and a retrievable document in its own right."""

    id: str
    parent_id: str
    audio_path: str
    start_s: float
    end_s: float

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


class Segmenter(Protocol):
    """Cuts a recording into candidate segments.

    The interesting axis: fixed-width windows (with or without overlap), VAD
    boundaries, or semantic/topic boundaries. Fixed windows are the baseline
    every paper uses and the one most likely to split an answer in half, which
    is exactly why overlap is a parameter and not a constant.
    """

    name: str

    def segment(self, audio_path: str, parent_id: str) -> list[Segment]:
        """Return segments covering the recording, in time order."""
        ...


class Selector(Protocol):
    """Chooses which segments become context for the downstream model."""

    name: str

    def select(self, query: Query, segments: list[Segment], k: int = 1) -> list[Segment]:
        """Return the k segments to pass downstream, best first."""
        ...


def overlap_s(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    """Length of the intersection of two time spans, 0 if disjoint."""
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def localization_accuracy(
    selected: list[Segment],
    gold_start_s: float,
    gold_end_s: float,
    *,
    min_overlap_s: float = 0.0,
) -> bool:
    """Did any selected segment actually overlap the gold answer span?

    The retrieval-side ground truth for a long-audio pipeline, independent of
    the downstream answer.

    A segment counts when it intersects the gold span by *more than*
    ``min_overlap_s`` seconds; at the default 0.0 that is any non-empty
    intersection, so merely touching at an endpoint does not count. Strictness
    is the caller's choice and belongs in the task config; setting
    ``min_overlap_s`` to the gold span's length demands full containment.
    Only segments of the same recording are considered.
    """
    return any(
        overlap_s(s.start_s, s.end_s, gold_start_s, gold_end_s) > min_overlap_s
        for s in selected
    )


def as_documents(segments: list[Segment]) -> list[Document]:
    """Expose segments as retrievable documents, preserving parent and timing.

    This is what lets the ordinary retriever protocol and the ordinary metric
    core operate on long audio with no special casing. ``audio_path`` is the
    parent recording; the span is in ``start_s`` / ``end_s``, and a backend must
    honour it rather than embed the whole file.
    """
    return [
        Document(id=s.id, audio_path=s.audio_path, parent_id=s.parent_id,
                 start_s=s.start_s, end_s=s.end_s)
        for s in segments
    ]


def wav_duration_s(path: str) -> float:
    """Duration of a PCM WAV, read from its header."""
    with wave.open(path, "rb") as f:
        return f.getnframes() / f.getframerate()


@dataclass(frozen=True)
class FixedWindowSegmenter:
    """Fixed-width windows every ``hop_s`` seconds; the last is cut at the end.

    ``hop_s < window_s`` gives overlap, which is what keeps an answer from being
    split in half at a boundary. Windows never leave a gap, so the recording is
    covered end to end; the final window may be shorter than ``window_s``.
    """

    window_s: float
    hop_s: float | None = None

    @property
    def name(self) -> str:
        return f"fixed_w{self.window_s:g}_h{self.hop:g}"

    @property
    def hop(self) -> float:
        return self.window_s if self.hop_s is None else self.hop_s

    def segment(self, audio_path: str, parent_id: str, duration_s: float | None = None
                ) -> list[Segment]:
        if self.window_s <= 0 or not 0 < self.hop <= self.window_s:
            raise ValueError("need window_s > 0 and 0 < hop_s <= window_s (no gaps)")
        total = wav_duration_s(audio_path) if duration_s is None else duration_s
        segments, start, n = [], 0.0, 0
        while True:
            end = min(start + self.window_s, total)
            segments.append(Segment(id=f"{parent_id}#{n}", parent_id=parent_id,
                                    audio_path=audio_path, start_s=start, end_s=end))
            if end >= total:
                return segments
            n += 1
            start = n * self.hop
