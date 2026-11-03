"""First end-to-end try on one GPU: Spoken-SQuAD, ASR cascade vs. clean text.

    uv pip install -e ".[cascade]"
    uv run python scripts/smoke_spoken_squad.py --config configs/smoke_spoken_squad.yaml

For each voice condition in the config:

1. build the selected articles (skipped if the build already exists);
2. transcribe every article recording once with Whisper, keeping timestamps
   (cached next to the build);
3. score, each into its own ``runs/<run_id>/``:
   - ``text``      clean question -> clean paragraph text (the ceiling; once)
   - ``cascade``   question -> ASR transcript of each paragraph
   - ``longform``  question -> ASR transcript of each fixed window of the
                   article recording, ranked against the whole archive and
                   within the question's own recording.

There is no direct audio retriever yet; that arm comes next (docs/PLAN.md).
Every run records peak GPU memory and wall-clock, because the point of this
script is to find out what fits.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import string
import time
from pathlib import Path

import numpy as np
import yaml

from lsr.eval import bootstrap_ci, compute_metrics, ranks_from_scores, ranks_within_groups
from lsr.longaudio import FixedWindowSegmenter
from lsr.retrievers.base import score_dense
from lsr.store import current_manifest, write_embeddings, write_run
from lsr.tasks import ManifestTask, check_task
from lsr.tasks.spoken_squad import LongFormTask, build_spoken_squad

REPO = Path(__file__).resolve().parents[1]


def load_config(path: Path) -> dict:
    env = {"LSR_DATA": "data/datasets", **os.environ}

    def expand(v):
        if isinstance(v, str):
            return string.Template(v).substitute(env)
        if isinstance(v, dict):
            return {k: expand(x) for k, x in v.items()}
        if isinstance(v, list):
            return [expand(x) for x in v]
        return v

    return expand(yaml.safe_load(path.read_text()))


class Gpu:
    """Peak-memory and wall-clock bookkeeping, a no-op without CUDA."""

    def __init__(self):
        import torch

        self.torch = torch
        self.cuda = torch.cuda.is_available()

    def reset(self):
        if self.cuda:
            self.torch.cuda.reset_peak_memory_stats()
        return time.perf_counter()

    def stats(self, t0: float) -> dict:
        out = {"wall_clock_s": round(time.perf_counter() - t0, 2)}
        if self.cuda:
            out["gpu_peak_mem_gb"] = round(self.torch.cuda.max_memory_allocated() / 2**30, 2)
            out["gpu"] = self.torch.cuda.get_device_name()
        return out


def score(ranks: np.ndarray, seed: int) -> dict:
    m = compute_metrics(ranks, ks=(1, 5, 10))
    m["Hits@1_ci95"] = bootstrap_ci(ranks, "Hits@1", seed=seed)
    m["MRR_ci95"] = bootstrap_ci(ranks, "MRR", seed=seed)
    return m


def save(cfg, run_id, task_name, retriever, q_emb, d_emb, qids, dids, metrics, extra):
    run_dir = REPO / "runs" / run_id
    write_embeddings(run_dir, q_emb, d_emb, qids, dids)
    manifest = current_manifest(
        run_id=run_id, task=task_name, retriever=retriever,
        config_hash=cfg["_hash"], seed=cfg["seed"],
        n_queries=len(qids), n_documents=len(dids),
        wall_clock_s=extra.get("wall_clock_s"),
    )
    write_run(run_dir, manifest, {**metrics, **extra})
    shown = {k: round(v, 3) for k, v in metrics.items()
             if isinstance(v, float) and k.split("_")[-1] in ("Hits@1", "Hits@5", "MRR")}
    cost = {k: v for k, v in extra.items() if k in ("wall_clock_s", "gpu_peak_mem_gb")}
    print(f"  {run_id}: Nq={len(qids)} Nd={len(dids)} {shown} {cost}")


def transcribe_articles(asr, build_dir: Path, articles: list[dict]) -> dict[str, list]:
    """Article id -> chunks, cached as JSONL so a crashed run resumes."""
    from lsr.backends.asr import Chunk

    cache = build_dir / f"asr_{asr.name}.jsonl"
    done = {}
    if cache.exists():
        for line in cache.read_text().splitlines():
            row = json.loads(line)
            done[row["article_id"]] = [Chunk(**c) for c in row["chunks"]]
    with open(cache, "a") as f:
        for art in articles:
            if art["article_id"] in done:
                continue
            chunks = asr.transcribe(str(build_dir / art["audio"]))
            done[art["article_id"]] = chunks
            f.write(json.dumps({"article_id": art["article_id"],
                                "chunks": [c.__dict__ for c in chunks]}) + "\n")
            f.flush()
    return done


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, required=True)
    cfg = load_config(parser.parse_args().config)
    cfg["_hash"] = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:12]
    seed = cfg["seed"]
    stamp = time.strftime("%Y%m%d-%H%M%S")

    import jiwer
    import torch

    from lsr.backends.asr import WhisperASR, normalize_for_wer, text_in_span
    from lsr.backends.text import SentenceEncoder

    torch.manual_seed(seed)
    gpu = Gpu()
    device = "cuda" if gpu.cuda else "cpu"
    enc = SentenceEncoder(cfg["text_encoder"], device=device,
                          query_prefix=cfg.get("query_prefix", ""))
    asr = WhisperASR(cfg["asr"], device=device, batch_size=cfg.get("asr_batch_size", 8))
    segmenters = [FixedWindowSegmenter(**w) for w in cfg["windows"]]
    build_root = Path(cfg["build_root"])

    conditions = {}
    for cond, wav_dir in cfg["conditions"].items():
        if Path(wav_dir).is_dir():
            conditions[cond] = wav_dir
        else:
            print(f"WARNING: skipping {cond}: {wav_dir} does not exist")

    question_sets = {}
    for cond, wav_dir in conditions.items():
        print(f"[{cond}]")
        build_dir = build_root / cond
        if not (build_dir / "articles.jsonl").exists():
            t0 = time.perf_counter()
            counts = build_spoken_squad(Path(cfg["json"]), Path(wav_dir), build_dir,
                                        cfg["articles"])
            print(f"  built {counts} in {time.perf_counter() - t0:.1f}s")

        para_task = ManifestTask("spoken_squad", build_dir, "text", "audio",
                                 audio_root=build_dir)
        check_task(para_task)
        queries = list(para_task.queries())
        qids = [q.id for q in queries]
        question_sets[cond] = set(qids)
        q_emb = enc.embed_texts([q.text for q in queries], as_query=True)
        qrels = para_task.qrels()

        # Clean-text ceiling: identical for every voice, so run it once.
        if cond == next(iter(conditions)):
            text_task = ManifestTask("spoken_squad", build_dir, "text", "text")
            check_task(text_task)
            t0 = gpu.reset()
            docs = list(text_task.documents())
            d_emb = enc.embed_texts([d.text for d in docs])
            dids = [d.id for d in docs]
            ranks = ranks_from_scores(score_dense(q_emb, d_emb), text_task.qrels(), qids, dids)
            save(cfg, f"{stamp}_text", text_task.name, enc.name, q_emb, d_emb, qids, dids,
                 score(ranks, seed), gpu.stats(t0))

        # ASR once per article recording.
        articles = [json.loads(line) for line in
                    (build_dir / "articles.jsonl").read_text().splitlines()]
        t0 = gpu.reset()
        chunks = transcribe_articles(asr, build_dir, articles)
        audio_s = sum(a["duration_s"] for a in articles)
        asr_stats = {**gpu.stats(t0), "audio_s": round(audio_s, 1)}
        contexts = {}
        for line in (build_dir / "con.txt").read_text().splitlines():
            row_id, _, text = line.partition(" ")
            contexts[row_id] = text
        ref = [normalize_for_wer(" ".join(contexts[p["question_ids"][0]]
                                          for p in a["paragraphs"])) for a in articles]
        hyp = [normalize_for_wer(" ".join(c.text for c in chunks[a["article_id"]]))
               for a in articles]
        wer = jiwer.wer(ref, hyp)
        print(f"  ASR {asr.name}: WER {wer:.4f} over {audio_s / 60:.1f} min, {asr_stats}")

        # Paragraph-level cascade: a paragraph's transcript is the ASR text in its span.
        span_of = {}
        for a in articles:
            for p in a["paragraphs"]:
                span_of[p["paragraph_id"]] = (a["article_id"], p["start_s"], p["end_s"])
        t0 = gpu.reset()
        docs = list(para_task.documents())
        transcripts = []
        for d in docs:
            art, s, e = span_of[Path(d.audio_path).stem]
            transcripts.append(text_in_span(chunks[art], s, e))
        d_emb = enc.embed_texts(transcripts)
        dids = [d.id for d in docs]
        ranks = ranks_from_scores(score_dense(q_emb, d_emb), qrels, qids, dids)
        save(cfg, f"{stamp}_{cond}_cascade", para_task.name, f"{asr.name}+{enc.name}",
             q_emb, d_emb, qids, dids, {**score(ranks, seed), "WER": wer},
             {**gpu.stats(t0), "asr": asr_stats})

        # Long-form: windows of each article recording.
        for seg in segmenters:
            lf = LongFormTask(build_dir, seg, min_overlap_s=cfg["min_overlap_s"])
            check_task(lf)
            t0 = gpu.reset()
            lf_queries = list(lf.queries())
            lf_qids = [q.id for q in lf_queries]
            lf_q_emb = q_emb[[qids.index(q) for q in lf_qids]]
            docs = list(lf.documents())
            d_emb = enc.embed_texts([
                text_in_span(chunks[lf.document_group[d.id]], d.start_s, d.end_s) for d in docs
            ])
            dids = [d.id for d in docs]
            scores = score_dense(lf_q_emb, d_emb)
            lf_qrels = lf.qrels()
            archive = score(ranks_from_scores(scores, lf_qrels, lf_qids, dids), seed)
            within = score(ranks_within_groups(scores, lf_qrels, lf_qids, dids,
                                               lf.query_group, lf.document_group), seed)
            metrics = {f"archive_{k}": v for k, v in archive.items()}
            metrics |= {f"within_{k}": v for k, v in within.items()}
            metrics["WER"] = wer
            save(cfg, f"{stamp}_{cond}_longform_{seg.name}", lf.name, f"{asr.name}+{enc.name}",
                 lf_q_emb, d_emb, lf_qids, dids, metrics,
                 {**gpu.stats(t0), "windows_per_article": round(len(docs) / len(articles), 1)})

    first = next(iter(question_sets.values()))
    if any(s != first for s in question_sets.values()):
        print("WARNING: conditions cover different questions; do not compare them")


if __name__ == "__main__":
    main()
