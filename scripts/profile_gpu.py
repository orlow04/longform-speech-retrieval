"""Predict whether the whole plan fits one GPU: measure each stage, extrapolate.

    uv pip install -e ".[profile]"
    uv run python scripts/profile_gpu.py --config configs/profile_gpu.yaml

Each heavy component of docs/PLAN.md is loaded on the GPU and run on a few
minutes of real audio, recording peak memory and throughput:

    asr            Whisper, long-form with timestamps     (cascade arm)
    text_encoder   sentence encoder over passages          (every arm)
    audio_encoder  HuBERT over fixed windows               (direct audio arm)
    image_encoder  EfficientNet-B7 over spectrogram images (CLASP's second branch)
    train_step     trainable speech encoder + frozen text encoder, contrastive,
                   batch-size sweep until OOM              (bridge training)
    lalm           audio LM answering from one 30 s window (long-form QA)

The measured rates are then multiplied out against the ``workload`` section of
the config — hours of audio per condition, number of perturbation conditions,
training samples and epochs, QA instances — into projected GPU-hours, VRAM
headroom and disk. Voice conversion (Seed-VC) runs out of tree, so its speed is
a config input, not a measurement.

Throughput on noise is not throughput on speech (Whisper decodes far less on
noise), so point ``sample_audio`` at a real recording — any built article WAV.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import yaml

SR = 16_000
REPO = Path(__file__).resolve().parents[1]


class Probe:
    def __init__(self, torch):
        self.torch = torch
        self.cuda = torch.cuda.is_available()
        self.device = "cuda" if self.cuda else "cpu"

    def sync(self):
        if self.cuda:
            self.torch.cuda.synchronize()

    def start(self):
        if self.cuda:
            self.torch.cuda.empty_cache()
            self.torch.cuda.reset_peak_memory_stats()
        self.sync()
        return time.perf_counter()

    def stop(self, t0) -> tuple[float, float | None]:
        self.sync()
        peak = self.torch.cuda.max_memory_allocated() / 2**30 if self.cuda else None
        return time.perf_counter() - t0, peak

    def free(self, *objs):
        for o in objs:
            del o
        import gc

        gc.collect()
        if self.cuda:
            self.torch.cuda.empty_cache()


def windows(wav: np.ndarray, seconds: float) -> list[np.ndarray]:
    n = int(seconds * SR)
    return [wav[i : i + n] for i in range(0, len(wav) - n + 1, n)] or [wav]


def fp(torch, p):
    return torch.float16 if p.cuda else torch.float32


# --- stages -------------------------------------------------------------------------


def stage_asr(cfg, wav, p, torch):
    from transformers import pipeline

    pipe = pipeline("automatic-speech-recognition", model=cfg["model"], device=p.device,
                    torch_dtype=fp(torch, p), chunk_length_s=30,
                    batch_size=cfg.get("batch_size", 8))
    kw = {} if cfg["model"].endswith(".en") else {
        "generate_kwargs": {"language": "en", "task": "transcribe"}}
    pipe({"raw": wav[: 30 * SR], "sampling_rate": SR}, **kw)  # warm-up
    t0 = p.start()
    out = pipe({"raw": wav, "sampling_rate": SR}, return_timestamps=True, **kw)
    dt, peak = p.stop(t0)
    p.free(pipe)
    return {"x_realtime": len(wav) / SR / dt, "peak_gb": peak,
            "words_out": len(out["text"].split())}


def stage_text_encoder(cfg, wav, p, torch):
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(cfg["model"], device=p.device)
    passage = " ".join(["the quick brown fox jumps over the lazy dog"] * 15)  # ~135 words
    texts = [passage] * cfg.get("n", 512)
    model.encode(texts[:32])
    t0 = p.start()
    model.encode(texts, batch_size=cfg.get("batch_size", 64))
    dt, peak = p.stop(t0)
    p.free(model)
    return {"texts_per_s": len(texts) / dt, "peak_gb": peak}


def stage_audio_encoder(cfg, wav, p, torch):
    from transformers import AutoModel

    model = AutoModel.from_pretrained(cfg["model"], torch_dtype=fp(torch, p)).to(p.device).eval()
    clips = windows(wav, cfg.get("window_s", 30))
    bs = cfg.get("batch_size", 4)
    batch = torch.tensor(np.stack(clips[:bs]), dtype=fp(torch, p), device=p.device)
    with torch.inference_mode():
        model(batch[:1])
        t0 = p.start()
        n = 0
        for i in range(0, len(clips), bs):
            x = torch.tensor(np.stack(clips[i : i + bs]), dtype=fp(torch, p), device=p.device)
            model(x).last_hidden_state.mean(1)
            n += x.shape[0]
        dt, peak = p.stop(t0)
    p.free(model)
    return {"x_realtime": n * cfg.get("window_s", 30) / dt, "peak_gb": peak}


def stage_image_encoder(cfg, wav, p, torch):
    import torchvision

    model = torchvision.models.efficientnet_b7(weights="DEFAULT").to(p.device, fp(torch, p))
    model.eval()
    size, bs, n_img = cfg.get("image_size", 600), cfg.get("batch_size", 8), cfg.get("n", 32)
    x = torch.randn(bs, 3, size, size, device=p.device, dtype=fp(torch, p))
    with torch.inference_mode():
        model(x[:1])
        t0 = p.start()
        for _ in range(max(1, n_img // bs)):
            model(x)
        dt, peak = p.stop(t0)
    p.free(model)
    images = max(1, n_img // bs) * bs
    # One image per window of window_s seconds (CLASP renders one per 20 s chunk).
    return {"images_per_s": images / dt, "x_realtime": images * cfg.get("window_s", 20) / dt,
            "peak_gb": peak}


def stage_train_step(cfg, wav, p, torch):
    """Bridge-training proxy: trainable speech encoder -> pooled -> linear, frozen
    text encoder, in-batch InfoNCE, AdamW, mixed precision. A HuBERT-large encoder
    (316M) is a conservative stand-in for Paraformer (220M trainable)."""
    from transformers import AutoModel, AutoTokenizer

    speech = AutoModel.from_pretrained(cfg["speech_model"]).to(p.device).train()
    if cfg.get("gradient_checkpointing"):
        speech.gradient_checkpointing_enable()
    text = AutoModel.from_pretrained(cfg["text_model"]).to(p.device).eval().requires_grad_(False)
    tok = AutoTokenizer.from_pretrained(cfg["text_model"])
    proj = torch.nn.Linear(speech.config.hidden_size, text.config.hidden_size).to(p.device)
    params = list(speech.parameters()) + list(proj.parameters())
    opt = torch.optim.AdamW(params, lr=5e-5)
    scaler = torch.amp.GradScaler(enabled=p.cuda)
    clip = windows(wav, cfg["clip_s"])[0]
    passage = " ".join(["the quick brown fox jumps over the lazy dog"] * 15)

    def step(bs):
        x = torch.tensor(np.stack([clip] * bs), device=p.device)
        t = tok([passage] * bs, padding=True, truncation=True, max_length=512,
                return_tensors="pt").to(p.device)
        with torch.autocast(p.device, enabled=p.cuda, dtype=torch.float16):
            a = torch.nn.functional.normalize(proj(speech(x).last_hidden_state.mean(1)), dim=-1)
            with torch.no_grad():
                e = torch.nn.functional.normalize(text(**t).last_hidden_state[:, 0], dim=-1)
            logits = a @ e.T / 0.05
            loss = torch.nn.functional.cross_entropy(logits, torch.arange(bs, device=p.device))
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()

    sweep = {}
    for bs in cfg["batch_sizes"]:
        try:
            step(bs)
            t0 = p.start()
            for _ in range(cfg.get("steps", 3)):
                step(bs)
            dt, peak = p.stop(t0)
            sweep[bs] = {"s_per_step": dt / cfg.get("steps", 3), "peak_gb": peak}
        except torch.OutOfMemoryError:
            sweep[bs] = "OOM"
            break
    p.free(speech, text, opt)
    fits = {b: v for b, v in sweep.items() if v != "OOM"}
    if not fits:
        return {"sweep": sweep, "error": "smallest batch does not fit"}
    best = max(fits)
    return {"sweep": sweep, "max_batch": best,
            "samples_per_s": best / fits[best]["s_per_step"], "peak_gb": fits[best]["peak_gb"]}


def stage_lalm(cfg, wav, p, torch):
    from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration

    processor = AutoProcessor.from_pretrained(cfg["model"])
    t0 = p.start()
    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        cfg["model"], torch_dtype=fp(torch, p), device_map=p.device)
    load_s, load_peak = p.stop(t0)
    clip = wav[: 30 * SR]
    conv = [{"role": "user", "content": [
        {"type": "audio", "audio_url": "clip.wav"},
        {"type": "text", "text": "What is the answer to: who won the game?"}]}]
    prompt = processor.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
    try:
        inputs = processor(text=prompt, audio=[clip], sampling_rate=SR, return_tensors="pt")
    except TypeError:  # older processors call it `audios`
        inputs = processor(text=prompt, audios=[clip], sampling_rate=SR, return_tensors="pt")
    inputs = inputs.to(p.device)
    n_new = cfg.get("max_new_tokens", 32)
    model.generate(**inputs, max_new_tokens=4)
    times = []
    for _ in range(cfg.get("repeats", 3)):
        t0 = p.start()
        model.generate(**inputs, max_new_tokens=n_new)
        dt, peak = p.stop(t0)
        times.append(dt)
    p.free(model)
    return {"s_per_instance": float(np.median(times)), "peak_gb": peak,
            "load_s": load_s, "weights_gb": load_peak}


STAGES = {
    "asr": stage_asr, "text_encoder": stage_text_encoder, "audio_encoder": stage_audio_encoder,
    "image_encoder": stage_image_encoder, "train_step": stage_train_step, "lalm": stage_lalm,
}


# --- projection ------------------------------------------------------------------------


def project(m: dict, w: dict) -> list[dict]:
    """Turn measured rates into GPU-hours for the configured workload."""
    rows = []

    def add(name, hours, stage, note):
        r = m.get(stage, {})
        rows.append({"job": name, "gpu_h": hours, "peak_gb": r.get("peak_gb"), "note": note})

    audio_h = w["audio_hours"] * w["conditions"]
    if "x_realtime" in m.get("asr", {}):
        add("ASR, every condition", audio_h / m["asr"]["x_realtime"], "asr",
            f"{audio_h:.0f} h audio = {w['audio_hours']} h x {w['conditions']} conditions")
    direct = [m[s]["x_realtime"] for s in ("audio_encoder", "image_encoder")
              if "x_realtime" in m.get(s, {})]
    if direct:
        per_h = sum(1 / x for x in direct)  # branches run one after the other
        add("direct audio embedding, every condition x window config",
            audio_h * w["window_configs"] * per_h, "audio_encoder",
            f"{len(direct)} branch(es), {w['window_configs']} window configs")
    if "texts_per_s" in m.get("text_encoder", {}):
        n = w["text_units"] * w["conditions"] * w["window_configs"]
        add("text embedding", n / m["text_encoder"]["texts_per_s"] / 3600, "text_encoder",
            f"{n:,} passages/windows")
    if "samples_per_s" in m.get("train_step", {}):
        n = w["train_samples"] * w["epochs"] * w["seeds"] * w["trained_arms"]
        add("bridge training", n / m["train_step"]["samples_per_s"] / 3600, "train_step",
            f"{w['train_samples']:,} samples x {w['epochs']} epochs x {w['seeds']} seeds "
            f"x {w['trained_arms']} arms, batch {m['train_step']['max_batch']}")
    if "s_per_instance" in m.get("lalm", {}):
        n = w["qa_instances"] * w["qa_arms"] * w["conditions_qa"]
        add("audio-LM QA", n * m["lalm"]["s_per_instance"] / 3600, "lalm",
            f"{n:,} instances")
    if w.get("vc_x_realtime"):
        vc_h = w["audio_hours"] * w["vc_voices"]
        rows.append({"job": "voice conversion (Seed-VC, configured rate)",
                     "gpu_h": vc_h / w["vc_x_realtime"], "peak_gb": None,
                     "note": f"{vc_h:.0f} h audio at {w['vc_x_realtime']}x realtime"})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, required=True)
    cfg = yaml.safe_load(parser.parse_args().config.read_text())

    import librosa
    import torch

    p = Probe(torch)
    if cfg.get("sample_audio") and Path(cfg["sample_audio"]).exists():
        wav, _ = librosa.load(cfg["sample_audio"], sr=SR, mono=True)
        wav = wav[: int(cfg.get("sample_seconds", 300) * SR)]
        source = cfg["sample_audio"]
    else:
        print("WARNING: no sample_audio; using noise. ASR throughput will be wrong.")
        wav = np.random.default_rng(0).normal(0, 0.1, int(cfg.get("sample_seconds", 300) * SR))
        source = "noise"
    wav = wav.astype(np.float32)

    total_gb = (torch.cuda.get_device_properties(0).total_memory / 2**30) if p.cuda else None
    device = torch.cuda.get_device_name() if p.cuda else "cpu"
    print(f"device: {device}, {total_gb and round(total_gb, 1)} GB; sample: {source}, "
          f"{len(wav) / SR:.0f} s")

    measured = {}
    for name, scfg in cfg["stages"].items():
        if not scfg.get("enabled", True):
            continue
        print(f"-- {name}", flush=True)
        try:
            measured[name] = STAGES[name](scfg, wav, p, torch)
        except Exception as e:  # one failing stage must not hide the others
            measured[name] = {"error": f"{type(e).__name__}: {e}"[:300]}
            p.free()
        print("   ", {k: (round(v, 2) if isinstance(v, float) else v)
                      for k, v in measured[name].items()}, flush=True)

    rows = project(measured, cfg["workload"])
    total = sum(r["gpu_h"] for r in rows)
    print(f"\nprojection on {device} ({cfg['workload']['label']}):")
    for r in rows:
        mem = "" if r["peak_gb"] is None else f"{r['peak_gb']:5.1f} GB"
        flag = " OVER VRAM" if total_gb and r["peak_gb"] and r["peak_gb"] > total_gb * 0.95 else ""
        print(f"  {r['gpu_h']:8.1f} GPU-h  {mem:>8}  {r['job']}{flag}\n{'':30}{r['note']}")
    print(f"  {total:8.1f} GPU-h total = {total / 24:.1f} days of one GPU, back to back")
    w = cfg["workload"]
    disk = w["audio_hours"] * w["conditions"] * 3600 * SR * 2 / 2**30
    print(f"  disk if every condition's audio is stored at 16 kHz/16-bit: {disk:,.0f} GB")

    out = REPO / "runs" / f"profile-{time.strftime('%Y%m%d-%H%M%S')}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "profile.json").write_text(json.dumps({
        "device": device, "total_gb": total_gb, "sample": source, "config": cfg,
        "measured": measured, "projection": rows, "total_gpu_h": total, "disk_gb": disk,
    }, indent=2, default=str))
    print(f"\nwrote {out / 'profile.json'}")


if __name__ == "__main__":
    main()
