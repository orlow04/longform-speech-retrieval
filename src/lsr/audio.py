"""Minimal PCM WAV I/O on numpy, for the core.

Builders and diagnostics run in the core environment, so this uses the standard
library's ``wave`` plus numpy and nothing else. Resampling is linear
interpolation: adequate for splicing clips into a host recording and for
energy-based diagnostics, not for anything a model is trained on — backends
resample with their own libraries.
"""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

_DTYPES = {1: np.uint8, 2: np.int16, 4: np.int32}


def read_wav(path: str | Path, start_s: float | None = None, end_s: float | None = None
             ) -> tuple[np.ndarray, int]:
    """Read PCM WAV (or a slice of it) as mono float32 in [-1, 1]; channels averaged."""
    with wave.open(str(path), "rb") as f:
        width, channels, rate = f.getsampwidth(), f.getnchannels(), f.getframerate()
        first = 0 if start_s is None else min(int(round(start_s * rate)), f.getnframes())
        last = f.getnframes() if end_s is None else min(int(round(end_s * rate)), f.getnframes())
        f.setpos(first)
        raw = f.readframes(max(0, last - first))
    if width not in _DTYPES:
        raise ValueError(f"{path}: unsupported sample width {width * 8} bit")
    x = np.frombuffer(raw, dtype=_DTYPES[width]).astype(np.float32)
    if width == 1:
        x = x - 128.0
    x /= float(2 ** (8 * width - 1))
    if channels > 1:
        x = x.reshape(-1, channels).mean(axis=1)
    return x, rate


def write_wav(path: str | Path, x: np.ndarray, rate: int) -> None:
    """Write mono float audio as 16-bit PCM, clipping to [-1, 1]."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    pcm = (np.clip(x, -1.0, 1.0) * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(rate)
        f.writeframes(pcm.tobytes())


def resample(x: np.ndarray, rate: int, target: int) -> np.ndarray:
    """Linear-interpolation resampling; identity when the rates match."""
    if rate == target or len(x) == 0:
        return x
    n = int(round(len(x) * target / rate))
    return np.interp(np.arange(n) * (rate / target), np.arange(len(x)), x).astype(np.float32)


def rms_db(x: np.ndarray) -> float:
    return float(20 * np.log10(np.sqrt(np.mean(np.square(x))) + 1e-10))
