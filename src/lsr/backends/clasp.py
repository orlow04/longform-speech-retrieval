"""CLASP (Abootorabi & Asgari, ECIR 2025): text <-> speech in one 768-d space.

Audio side: HuBERT-large (mean over frames) and EfficientNet-B7 logits on a
rendered log-spectrogram image, each computed per 20 s chunk, fused by the
trained ``HubertLabseConcat`` head. Text side: LaBSE. This reproduces
``clasp.inference`` (paper-audio/clasp) step for step, including the
matplotlib spectrogram rendering the image branch was trained on; check parity
with ``scripts/check_clasp_parity.py`` after touching anything here.

Long audio is split into ``chunk_s`` chunks and the per-chunk features are
kept, so pooling is a choice made afterwards, not baked into extraction:

    mean         CLASP's own long-audio rule: average HuBERT and image
                 features over chunks, then fuse once
    multivector  fuse every chunk separately ([C, 768]), for max-sim

The checkpoint is a whole pickled module saved from a notebook, so its class
must be importable as ``__main__.HubertLabseConcat`` when it is loaded.
"""

from __future__ import annotations

import io
import sys
from dataclasses import dataclass

import numpy as np

HUBERT_MODEL = "facebook/hubert-large-ls960-ft"
LABSE_MODEL = "sentence-transformers/LaBSE"
SR = 16_000
MIN_SAMPLES = SR  # HuBERT's conv stack wants at least ~1 s


def _fusion_class():
    from torch import nn

    class HubertLabseConcat(nn.Module):
        """Verbatim from clasp.models.fusion; only needed to unpickle the checkpoint."""

        def __init__(self, in_features_text, in_features_image, mode="joint"):
            super().__init__()
            self.mode = mode

            def branch(d_in):
                return nn.Sequential(
                    nn.Linear(d_in, 768), nn.BatchNorm1d(768), nn.ReLU(), nn.Dropout(p=0.15),
                    nn.Linear(768, 576), nn.BatchNorm1d(576), nn.LeakyReLU(), nn.Dropout(p=0.1),
                    nn.Linear(576, 768))

            self.image_seq = branch(in_features_image)
            self.audio_seq = branch(in_features_text)
            self.mix_seq = nn.Sequential(
                nn.Linear(2 * 768, 1024), nn.BatchNorm1d(1024), nn.LeakyReLU(),
                nn.Linear(1024, 800), nn.LeakyReLU(), nn.Linear(800, 768))

        def forward(self, x_audio, x_image):
            import torch

            x1 = self.audio_seq(x_audio)
            if self.mode == "audio":
                return x1
            x2 = self.image_seq(x_image)
            if self.mode == "image":
                return x2
            return self.mix_seq(torch.cat((x1, x2), dim=1))

    return HubertLabseConcat


def load_fusion(path, device, mode: str = "joint"):
    """A fusion head from any of CLASP's checkpoint forms, without the clasp package.

    The official checkpoint pickles the module as ``__main__.HubertLabseConcat``
    (saved from a notebook); ``scripts/train.py`` pickles it as
    ``clasp.models.fusion.HubertLabseConcat``; ``--init-checkpoint`` files may
    be plain state dicts. All three resolve to the vendored class. ``mode`` is
    used only for state dicts, which do not record it.
    """
    import pickle
    import types

    import torch

    cls = _fusion_class()
    setattr(sys.modules["__main__"], "HubertLabseConcat", cls)

    class _Unpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if name == "HubertLabseConcat":
                return cls
            return super().find_class(module, name)

    pickle_module = types.ModuleType("lsr_clasp_pickle")
    pickle_module.__dict__.update({k: v for k, v in vars(pickle).items()
                                   if not k.startswith("__")})
    pickle_module.Unpickler = _Unpickler
    obj = torch.load(path, map_location=device, weights_only=False, pickle_module=pickle_module)
    if isinstance(obj, dict):
        state = obj.get("state_dict", obj.get("model_state_dict", obj))
        model = cls(state["audio_seq.0.weight"].shape[1], state["image_seq.0.weight"].shape[1],
                    mode=mode)
        model.load_state_dict(state, strict=True)
    else:
        model = obj
        if not hasattr(model, "mode"):
            model.mode = "joint"
    return model.to(device).eval()


def prepare_waveform(x: np.ndarray, rate: int) -> np.ndarray:
    """CLASP's loader: mono, peak-normalised, resampled to 16 kHz, padded to 1 s."""
    import librosa

    y = np.asarray(x, dtype=np.float32).reshape(-1)
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    if peak > 0:
        y = y / peak
    if rate != SR:
        y = librosa.resample(y, orig_sr=rate, target_sr=SR)
    if y.size < MIN_SAMPLES:
        y = np.pad(y, (0, MIN_SAMPLES - y.size))
    return y.astype(np.float32)


def chunks_of(y: np.ndarray, chunk_samples: int) -> list[np.ndarray]:
    """Consecutive chunks; each padded to 1 s (as in CLASP, which also pads the last one)."""
    out = []
    for a in range(0, max(len(y), 1), chunk_samples):
        piece = y[a:a + chunk_samples]
        if piece.size < MIN_SAMPLES:
            piece = np.pad(piece, (0, MIN_SAMPLES - piece.size))
        out.append(piece)
    return out


def spectrogram_image(y: np.ndarray):
    """Log-STFT rendered by librosa.display.specshow into a 4x4 in. figure, as CLASP does."""
    import librosa
    import librosa.display
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    spec_db = librosa.amplitude_to_db(np.abs(librosa.stft(np.asarray(y, dtype=np.float32))))
    fig, ax = plt.subplots(figsize=(4, 4))
    librosa.display.specshow(spec_db, x_axis="time", y_axis="log", ax=ax)
    ax.set_xlabel("")
    ax.set_ylabel("")
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


@dataclass
class ChunkFeatures:
    """Per-chunk inputs to the fusion head for one piece of audio."""

    hubert: np.ndarray  # [C, 1024]
    image: np.ndarray   # [C, 1000]


class ClaspEncoder:
    def __init__(self, checkpoint: str, device: str | None = None,
                 hubert_model: str = HUBERT_MODEL, labse_model: str = LABSE_MODEL,
                 chunk_s: float = 20.0, chunk_batch_size: int = 4):
        import torch
        from sentence_transformers import SentenceTransformer
        from torchvision.models import EfficientNet_B7_Weights, efficientnet_b7
        from transformers import AutoProcessor, HubertModel

        if device is None:
            device = ("cuda" if torch.cuda.is_available()
                      else "mps" if torch.backends.mps.is_available() else "cpu")
        self.device = torch.device(device)
        self.name = "clasp"
        self.chunk_samples = int(chunk_s * SR)
        self.chunk_batch_size = chunk_batch_size
        self.fusion = load_fusion(checkpoint, self.device)
        self.processor = AutoProcessor.from_pretrained(hubert_model)
        self.hubert = HubertModel.from_pretrained(hubert_model).to(self.device).eval()
        weights = EfficientNet_B7_Weights.DEFAULT
        self.effnet = efficientnet_b7(weights=weights).to(self.device).eval()
        self.effnet_preprocess = weights.transforms()
        self.labse = SentenceTransformer(labse_model, device=str(self.device))

    def chunk_features(self, x: np.ndarray, rate: int) -> ChunkFeatures:
        import torch

        pieces = chunks_of(prepare_waveform(x, rate), self.chunk_samples)
        hub, img = [], []
        bs = self.chunk_batch_size
        with torch.no_grad():
            for i in range(0, len(pieces), bs):
                batch = pieces[i:i + bs]
                # HuBERT one chunk at a time, CLASP's default: padding a batch would
                # pull the short last chunk's frame mean toward the padding.
                for p in batch:
                    inputs = self.processor(p, sampling_rate=SR, return_tensors="pt")
                    inputs = inputs.to(self.device)
                    hub.append(self.hubert(**inputs).last_hidden_state.mean(dim=1)[0].cpu())
                images = torch.stack([self.effnet_preprocess(spectrogram_image(p))
                                      for p in batch])
                img.extend(self.effnet(images.to(self.device)).float().cpu())
        return ChunkFeatures(torch.stack(hub).numpy().astype(np.float32),
                             torch.stack(img).numpy().astype(np.float32))

    def fuse(self, hubert: np.ndarray, image: np.ndarray, fusion=None) -> np.ndarray:
        """Fusion head on [N, 1024] x [N, 1000] -> raw (unnormalised) [N, 768].

        ``fusion``: another head from ``load_fusion``. HuBERT, EfficientNet and
        LaBSE are frozen in CLASP, so chunk features are shared across checkpoints
        and only this step differs.
        """
        import torch

        with torch.no_grad():
            out = (fusion or self.fusion)(torch.from_numpy(hubert).to(self.device),
                                          torch.from_numpy(image).to(self.device))
        return out.float().cpu().numpy()

    def pool(self, feats: list[ChunkFeatures], how: str = "mean",
             fusion=None) -> list[np.ndarray]:
        """Raw fused embeddings: one [768] per item (mean) or [C, 768] per item (multivector)."""
        if how == "mean":
            h = np.stack([f.hubert.mean(axis=0) for f in feats])
            i = np.stack([f.image.mean(axis=0) for f in feats])
            return list(self.fuse(h, i, fusion))
        if how == "multivector":
            return [self.fuse(f.hubert, f.image, fusion) for f in feats]
        raise ValueError(f"pooling must be 'mean' or 'multivector', got {how!r}")

    def embed_texts(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        emb = self.labse.encode(texts, batch_size=batch_size, normalize_embeddings=True,
                                convert_to_numpy=True, show_progress_bar=False)
        return emb.astype(np.float32)


def l2n(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x / np.clip(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12, None)
