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

    TODO(you): decide what counts as overlap. Any intersection is generous; the
    answer being wholly contained is strict. The choice changes the headline
    number, so it belongs in the task definition rather than in a default here.
    """
    raise NotImplementedError


def as_documents(segments: list[Segment]) -> list[Document]:
    """Expose segments as retrievable documents, preserving parent and timing.

    This is what lets the ordinary retriever protocol and the ordinary metric
    core operate on long audio with no special casing.
    """
    raise NotImplementedError
