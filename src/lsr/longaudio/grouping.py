"""Group consecutive transcript units of a recording into documents of a target length.

"Concatenating passages simply" at a chosen duration: units (sentences, for
Spoken-SQuAD) are never split, so every document boundary is a unit boundary,
and documents never cross recordings. Greedy: a unit joins the open document
unless stopping first would land closer to the target. A short tail (under
half the target) is merged into the previous document rather than left as a
runt that would be trivially distinct.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Group:
    recording_id: str
    index: int
    start_s: float
    end_s: float
    units: tuple[int, ...]

    @property
    def id(self) -> str:
        return f"{self.recording_id}@{self.index}"

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


def group_units(recording_id: str, units: list[dict], target_s: float) -> list[Group]:
    """``units``: time-ordered ``{"start_s", "end_s"}`` rows of one recording."""
    if target_s <= 0:
        raise ValueError("target_s must be positive")
    spans: list[list[int]] = []
    cur: list[int] = []
    for i, u in enumerate(units):
        if cur:
            have = units[cur[-1]]["end_s"] - units[cur[0]]["start_s"]
            with_u = u["end_s"] - units[cur[0]]["start_s"]
            if have >= target_s or abs(with_u - target_s) > abs(have - target_s):
                spans.append(cur)
                cur = []
        cur.append(i)
    if cur:
        tail = units[cur[-1]]["end_s"] - units[cur[0]]["start_s"]
        if spans and tail < target_s / 2:
            spans[-1].extend(cur)
        else:
            spans.append(cur)
    return [Group(recording_id, k, units[s[0]]["start_s"], units[s[-1]]["end_s"], tuple(s))
            for k, s in enumerate(spans)]


def group_containing(groups: list[Group], start_s: float, end_s: float,
                     tol_s: float = 0.05) -> Group | None:
    """The group wholly containing [start_s, end_s], or None if it straddles a boundary."""
    for g in groups:
        if g.start_s - tol_s <= start_s and end_s <= g.end_s + tol_s:
            return g
    return None
