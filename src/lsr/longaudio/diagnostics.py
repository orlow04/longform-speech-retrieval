"""Does a long-audio benchmark leak its answers through acoustics?

Concatenating clips leaves marks exactly where the clips meet: a pause, a jump
in loudness, a change of channel or voice. When a gold span is a whole clip,
its edges coincide with those marks, and a system can localise it without
understanding a word. Native recordings have no such marks at answer edges.

These diagnostics measure that directly, per recording, with energy alone:

- ``boundary_cues``: how unusual the pause and the loudness jump at each gold
  edge are, as percentiles of the same quantity at every pause / random point
  of that recording. ~0.5 means gold edges look like anywhere else; ~1.0 means
  they are the most marked moments in the recording. "Edge at a pause" comes
  with its base rate at random times, without which it means nothing.
- ``cue_only_ranks``: rank the windows of a recording by acoustic boundary
  strength alone — the question is never read. Scored with the ordinary metric
  core, it is a content-free baseline: any lift over its own chance level is
  information the benchmark gives away for free.
"""

from __future__ import annotations

import numpy as np

FRAME_S = 0.02


def frame_db(x: np.ndarray, rate: int, frame_s: float = FRAME_S) -> np.ndarray:
    n = max(1, int(frame_s * rate))
    frames = x[: len(x) // n * n].reshape(-1, n)
    return 20 * np.log10(np.sqrt(np.mean(frames**2, axis=1)) + 1e-10)


def pauses(db: np.ndarray, frame_s: float = FRAME_S, depth: float = 0.25,
           min_pause_s: float = 0.15) -> list[tuple[float, float]]:
    """Runs of quiet frames at least ``min_pause_s`` long.

    Quiet means below ``depth`` of the way from the noise floor (10th-percentile
    frame) to speech level (95th). Relative to each recording's own floor, so a
    noisy field interview and digitally silent TTS gaps are both handled; a
    fixed dB threshold finds no pauses at all in the former.
    """
    floor, speech = np.percentile(db, 10), np.percentile(db, 95)
    quiet = db < floor + depth * (speech - floor)
    out, start = [], None
    for i, q in enumerate(np.append(quiet, False)):
        if q and start is None:
            start = i
        elif not q and start is not None:
            if (i - start) * frame_s >= min_pause_s:
                out.append((start * frame_s, i * frame_s))
            start = None
    return out


def _pause_at(t: float, ps: list[tuple[float, float]], tol: float) -> float:
    return max((e - s for s, e in ps if s - tol <= t <= e + tol), default=0.0)


def _jump_db(db: np.ndarray, t: float, frame_s: float, ctx_s: float = 1.0) -> float:
    i, k = int(t / frame_s), int(ctx_s / frame_s)
    before, after = db[max(0, i - k): i], db[i: i + k]
    if len(before) == 0 or len(after) == 0:
        return 0.0
    return abs(float(np.median(after) - np.median(before)))


def _percentile(value: float, population: np.ndarray) -> float:
    if len(population) == 0:
        return float("nan")
    return float((np.sum(population < value) + 0.5 * np.sum(population == value))
                 / len(population))


def boundary_cues(x: np.ndarray, rate: int, edges: list[float], *, tol_s: float = 0.25,
                  n_random: int = 500, seed: int = 0) -> dict[str, float]:
    """Percentile of pause length and loudness jump at gold edges, within one recording."""
    db = frame_db(x, rate)
    ps = pauses(db)
    pause_pop = np.array([e - s for s, e in ps])
    dur = len(x) / rate
    rnd = np.random.default_rng(seed).uniform(0, dur, n_random)
    jump_pop = np.array([_jump_db(db, t, FRAME_S) for t in rnd])
    edge_pause = [_pause_at(t, ps, tol_s) for t in edges]
    edge_jump = [_jump_db(db, t, FRAME_S) for t in edges]
    nan = float("nan")
    return {
        "n_edges": len(edges),
        "edge_at_pause": float(np.mean([p > 0 for p in edge_pause])) if edges else nan,
        # Base rate for the line above: conversation is full of short pauses.
        "random_at_pause": float(np.mean([_pause_at(t, ps, tol_s) > 0 for t in rnd])),
        "pause_percentile": float(np.mean([_percentile(p, pause_pop) for p in edge_pause]))
        if edges and len(pause_pop) else nan,
        "jump_percentile": float(np.mean([_percentile(j, jump_pop) for j in edge_jump]))
        if edges else float("nan"),
        "n_pauses": len(ps),
    }


def window_cue_strength(x: np.ndarray, rate: int, windows: list[tuple[float, float]],
                        tol_s: float = 0.25) -> np.ndarray:
    """Content-free score per window: the strongest pause-plus-jump inside it."""
    db = frame_db(x, rate)
    ps = pauses(db)
    pause_mid = [((s + e) / 2, e - s) for s, e in ps]
    scores = np.zeros(len(windows))
    for j, (a, b) in enumerate(windows):
        inside = [(t, length) for t, length in pause_mid if a - tol_s <= t <= b + tol_s]
        scores[j] = max((length + _jump_db(db, t, FRAME_S) / 10 for t, length in inside),
                        default=0.0)
    return scores
