"""Controlled variants of long-audio builds, for the native-vs-concatenated study.

Each variant re-assembles existing audio and changes one thing, so a change in
retrieval can be attributed to it:

- ``shuffled``  a concatenated recording's pieces in random order. Removes
                discourse continuity; keeps every boundary cue.
- ``smoothed``  pieces with leading/trailing silence trimmed and loudness
                levelled. Removes the acoustic boundary cues; keeps content
                and order.
- ``spliced``   pieces of a concatenated build inserted into the recordings of
                a native build — CLSR Table 5's construction. The host's own
                questions are kept (``origin`` "host") as a native control in
                the very same audio; the inserted ones are ``origin`` "spliced".

All three go through one mechanism: an output recording is a list of segments,
each a time range of some source recording, and every timestamp (gold spans,
transcripts) is mapped through that list. Output is 16-bit mono at the source
rate (``spliced``: the host's rate, pieces resampled).
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from lsr.audio import read_wav, resample, write_wav
from lsr.tasks.longform import read_jsonl, write_jsonl


@dataclass(frozen=True)
class Seg:
    """Copy ``[t0, t1)`` of a source recording, scaled by ``gain``."""

    build: Path
    recording_id: str
    audio: Path
    t0: float
    t1: float
    gain: float = 1.0


class _Source:
    def __init__(self, build_dir: Path):
        self.dir = Path(build_dir)
        self.recordings = {r["recording_id"]: r for r in read_jsonl(self.dir / "recordings.jsonl")}
        self.questions = read_jsonl(self.dir / "questions.jsonl")

    def audio(self, rid: str) -> Path:
        return self.dir / self.recordings[rid]["audio"]

    def seg(self, rid: str, t0: float, t1: float, gain: float = 1.0) -> Seg:
        return Seg(self.dir, rid, self.audio(rid), t0, t1, gain)

    def transcript(self, rid: str) -> list[dict]:
        rel = self.recordings[rid].get("transcript")
        return read_jsonl(self.dir / rel) if rel else []


def _assemble(segs: list[Seg], out_wav: Path, rate: int | None = None
              ) -> tuple[list[tuple[Seg, float, float]], int]:
    """Write the concatenation; return each segment's placement and the rate."""
    parts, placed, t = [], [], 0.0
    for seg in segs:
        x, sr = read_wav(seg.audio, seg.t0, seg.t1)
        rate = rate or sr
        x = resample(x, sr, rate) * seg.gain
        parts.append(x)
        placed.append((seg, t, t + len(x) / rate))
        t += len(x) / rate
    write_wav(out_wav, np.concatenate(parts) if parts else np.zeros(0, np.float32), rate)
    return placed, rate


def _mapper(placed: list[tuple[Seg, float, float]]):
    """Map (build, recording, source span) -> output span, or None if not copied."""
    def to_out(build: Path, rid: str, a: float, b: float):
        mid = (a + b) / 2
        for seg, o0, o1 in placed:
            if seg.build == build and seg.recording_id == rid and seg.t0 <= mid < seg.t1:
                lo = o0 + max(a, seg.t0) - seg.t0
                hi = o0 + min(b, seg.t1) - seg.t0
                return min(lo, o1), min(hi, o1)
        return None
    return to_out


def _emit(out_dir: Path, rid: str, segs: list[Seg], sources: list[_Source], kind: str,
          source_name: str, rate: int | None = None, origin_of=None) -> tuple[dict, list[dict]]:
    """Assemble one output recording; map questions and transcripts into it."""
    placed, rate = _assemble(segs, out_dir / "audio" / f"{rid}.wav", rate)
    to_out = _mapper(placed)
    used = {(s.build, s.recording_id) for s in segs}

    transcript, questions = [], []
    for src in sources:
        for srid in src.recordings:
            if (src.dir, srid) not in used:
                continue
            for u in src.transcript(srid):
                span = to_out(src.dir, srid, u["start_s"], u["end_s"])
                if span:
                    transcript.append({**u, "start_s": span[0], "end_s": span[1]})
            for q in src.questions:
                if q["recording_id"] != srid:
                    continue
                span = to_out(src.dir, srid, q["gold_start_s"], q["gold_end_s"])
                if span is None or span[1] <= span[0]:
                    continue
                row = {**q, "recording_id": rid, "gold_start_s": span[0], "gold_end_s": span[1]}
                if "paragraph_span" in q:
                    p = to_out(src.dir, srid, *q["paragraph_span"])
                    row["paragraph_span"] = list(p) if p else None
                if origin_of:
                    row["origin"] = origin_of(src)
                questions.append(row)
    transcript.sort(key=lambda u: u["start_s"])
    write_jsonl(out_dir / "transcripts" / f"{rid}.jsonl", transcript)
    rec = {"recording_id": rid, "audio": f"audio/{rid}.wav",
           "duration_s": placed[-1][2] if placed else 0.0, "kind": kind,
           "source": source_name, "transcript": f"transcripts/{rid}.jsonl",
           "pieces": [{"piece_id": f"{s.recording_id}@{s.t0:.2f}", "start_s": o0, "end_s": o1}
                      for s, o0, o1 in placed]}
    return rec, questions


def _write(out_dir: Path, recordings: list[dict], questions: list[dict]) -> dict[str, int]:
    write_jsonl(out_dir / "recordings.jsonl", recordings)
    write_jsonl(out_dir / "questions.jsonl", questions)
    return {"recordings": len(recordings), "questions": len(questions)}


def shuffled(src_dir: Path, out_dir: Path, seed: int = 0) -> dict[str, int]:
    """Each concatenated recording with its pieces in a seeded random order."""
    src, rng, out_dir = _Source(src_dir), random.Random(seed), Path(out_dir)
    recordings, questions = [], []
    for rid, rec in src.recordings.items():
        pieces = list(rec["pieces"])
        rng.shuffle(pieces)
        segs = [src.seg(rid, p["start_s"], p["end_s"]) for p in pieces]
        r, q = _emit(out_dir, rid, segs, [src], "concatenated", f"{rec['source']}+shuffled")
        recordings.append(r)
        questions += q
    return _write(out_dir, recordings, questions)


def _trim_bounds(x: np.ndarray, rate: int, threshold_db: float, frame_s: float = 0.02
                 ) -> tuple[float, float]:
    """Seconds of leading and trailing audio below ``threshold_db`` re the loudest frame."""
    n = max(1, int(frame_s * rate))
    frames = x[: len(x) // n * n].reshape(-1, n)
    if len(frames) == 0:
        return 0.0, len(x) / rate
    db = 20 * np.log10(np.sqrt(np.mean(frames**2, axis=1)) + 1e-10)
    loud = np.flatnonzero(db > db.max() + threshold_db)
    if len(loud) == 0:
        return 0.0, len(x) / rate
    return loud[0] * n / rate, min((loud[-1] + 1) * n / rate, len(x) / rate)


def smoothed(src_dir: Path, out_dir: Path, trim_db: float = -40.0, target_db: float = -23.0
             ) -> dict[str, int]:
    """Pieces trimmed of edge silence and levelled to ``target_db`` RMS, then joined."""
    src, out_dir = _Source(src_dir), Path(out_dir)
    recordings, questions = [], []
    for rid, rec in src.recordings.items():
        segs = []
        for p in rec["pieces"]:
            x, rate = read_wav(src.audio(rid), p["start_s"], p["end_s"])
            lead, tail = _trim_bounds(x, rate, trim_db)
            core = x[int(lead * rate): int(tail * rate)]
            rms = np.sqrt(np.mean(core**2)) if len(core) else 0.0
            gain = 10 ** (target_db / 20) / rms if rms > 0 else 1.0
            segs.append(src.seg(rid, p["start_s"] + lead, p["start_s"] + tail, gain))
        r, q = _emit(out_dir, rid, segs, [src], "concatenated", f"{rec['source']}+smoothed")
        recordings.append(r)
        questions += q
    return _write(out_dir, recordings, questions)


def spliced(src_dir: Path, host_dir: Path, out_dir: Path, pieces_per_host: int = 5,
            seed: int = 0) -> dict[str, int]:
    """Insert pieces of ``src_dir`` into each recording of ``host_dir``.

    Pieces are drawn without replacement; insertion points are host utterance
    ends when the host has a transcript (so no host word is cut), uniform
    otherwise. Every question of an inserted piece, and every host question,
    is carried over with its ``origin``.
    """
    src, host, out_dir = _Source(src_dir), _Source(host_dir), Path(out_dir)
    rng = random.Random(seed)
    pool = [(rid, p) for rid, rec in src.recordings.items() for p in rec["pieces"]]
    rng.shuffle(pool)
    recordings, questions = [], []
    for hid, hrec in host.recordings.items():
        chosen, pool = pool[:pieces_per_host], pool[pieces_per_host:]
        if not chosen:
            break
        ends = sorted({u["end_s"] for u in host.transcript(hid)}) or None
        dur = hrec["duration_s"]
        cuts = sorted(rng.sample(ends, len(chosen)) if ends and len(ends) >= len(chosen)
                      else [rng.uniform(0, dur) for _ in chosen])
        segs, t = [], 0.0
        for cut, (rid, p) in zip(cuts, chosen):
            segs.append(host.seg(hid, t, cut))
            segs.append(src.seg(rid, p["start_s"], p["end_s"]))
            t = cut
        segs.append(host.seg(hid, t, dur))
        segs = [s for s in segs if s.t1 > s.t0]
        _, host_rate = read_wav(host.audio(hid), 0, 0.01)
        r, q = _emit(out_dir, hid, segs, [host, src], "concatenated",
                     f"{hrec['source']}+spliced", rate=host_rate,
                     origin_of=lambda s: "host" if s is host else "spliced")
        recordings.append(r)
        questions += q
    return _write(out_dir, recordings, questions)
