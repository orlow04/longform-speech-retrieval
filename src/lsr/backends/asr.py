"""Whisper ASR over whole recordings, with chunk-level timestamps.

One pass per recording; the timestamps are what let a long-form cascade cut its
transcript on the same windows a direct audio retriever would embed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

TARGET_SR = 16_000


@dataclass(frozen=True)
class Chunk:
    start_s: float
    end_s: float
    text: str


class WhisperASR:
    def __init__(self, model: str = "openai/whisper-large-v3-turbo", device: str = "cuda",
                 batch_size: int = 8, language: str = "en"):
        import torch
        from transformers import pipeline

        self.name = model.rsplit("/", 1)[-1]
        self.language = language
        self.pipe = pipeline(
            "automatic-speech-recognition", model=model, device=device,
            torch_dtype=torch.float16 if device.startswith("cuda") else torch.float32,
            chunk_length_s=30, batch_size=batch_size,
        )

    def transcribe(self, audio_path: str) -> list[Chunk]:
        """Transcribe one recording of any length; resampled to 16 kHz first."""
        import librosa

        wav, _ = librosa.load(audio_path, sr=TARGET_SR, mono=True)
        # English-only checkpoints (*.en) reject language/task arguments.
        kwargs = {} if self.name.endswith(".en") else {
            "generate_kwargs": {"language": self.language, "task": "transcribe"}}
        out = self.pipe({"raw": wav, "sampling_rate": TARGET_SR}, return_timestamps=True,
                        **kwargs)
        duration = len(wav) / TARGET_SR
        chunks = []
        for c in out["chunks"]:
            start, end = c["timestamp"]
            start = 0.0 if start is None else float(start)
            end = duration if end is None else float(end)  # Whisper omits the final end
            chunks.append(Chunk(start, end, c["text"].strip()))
        return chunks


def text_in_span(chunks: list[Chunk], start_s: float, end_s: float) -> str:
    """Concatenate the chunks whose midpoint falls in [start_s, end_s)."""
    return " ".join(c.text for c in chunks
                    if start_s <= (c.start_s + c.end_s) / 2 < end_s).strip()


def normalize_for_wer(text: str) -> str:
    """Lowercase, drop punctuation, collapse whitespace. Numbers are left as is."""
    return " ".join(re.sub(r"[^\w\s']", " ", text.lower()).split())
