"""Check lsr.backends.clasp against CLASP's own inference code on one recording.

    uv run --extra clasp python scripts/check_clasp_parity.py AUDIO.wav \
        [--clasp-src ~/dev/paper-audio/clasp/CLASP/src] [--checkpoint ...]

Compares per-stage outputs (HuBERT mean, EfficientNet logits, fused embedding)
on audio longer than one chunk, so the chunking and mean rule are exercised.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

DEFAULT_SRC = Path.home() / "dev/paper-audio/clasp/CLASP/src"
DEFAULT_CKPT = DEFAULT_SRC.parent / "models/checkpoints/CLASP_Concat_Final_Fusion_Encoder.pt"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("audio", type=Path)
    p.add_argument("--clasp-src", type=Path, default=DEFAULT_SRC)
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT)
    p.add_argument("--seconds", type=float, default=45.0)
    p.add_argument("--device", default="cpu")
    a = p.parse_args()

    import soundfile as sf
    import torch

    from lsr.backends.clasp import ClaspEncoder

    x, rate = sf.read(str(a.audio), always_2d=False)
    x = x[: int(a.seconds * rate)]
    enc = ClaspEncoder(str(a.checkpoint), device=a.device)
    ours = enc.chunk_features(x, rate)
    ours_fused = enc.pool([ours], "mean")[0]

    sys.path.insert(0, str(a.clasp_src))
    import librosa
    from clasp.inference.embed_audio import hubert_numpy_waveform
    from clasp.inference.pipeline import load_model
    from clasp.inference.spectrogram_image import efficientnet_embedding_from_waveform

    # CLASP's loader, applied to the same excerpt (it reads whole files).
    y = np.asarray(x, dtype=np.float32)
    y = librosa.resample(y / np.max(np.abs(y)), orig_sr=rate, target_sr=16_000)
    dev = torch.device(a.device)
    h = hubert_numpy_waveform(y, enc.processor, enc.hubert, dev, pooling="mean").cpu().numpy()
    i = efficientnet_embedding_from_waveform(y, enc.effnet, enc.effnet_preprocess, dev,
                                             pooling="mean").cpu().numpy()
    model = load_model(str(a.checkpoint), dev)
    with torch.no_grad():
        f = model(torch.from_numpy(h)[None].to(dev), torch.from_numpy(i)[None].to(dev))
    f = f[0].cpu().numpy()

    def report(name, u, v):
        cos = float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v)))
        print(f"{name:8s} max|diff| {np.max(np.abs(u - v)):.2e}  cosine {cos:.6f}")
        return cos

    print(f"{len(ours.hubert)} chunks of {enc.chunk_samples / 16000:.0f} s")
    worst = min(report("hubert", ours.hubert.mean(0), h),
                report("image", ours.image.mean(0), i),
                report("fused", ours_fused, f))
    print("PARITY OK" if worst > 0.9999 else "PARITY FAILED")
    sys.exit(0 if worst > 0.9999 else 1)


if __name__ == "__main__":
    main()
