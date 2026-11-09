"""Retrieval from long spoken context, native vs. concatenated.

    uv run python scripts/run_longaudio.py --config configs/longaudio.yaml

For every build x window configuration x arm, a text question is ranked against
the windows of long recordings and scored with the metric core:

    random       chance, averaged over 20 seeded permutations
    cue_only     acoustic boundary strength only; the question is never read
    bm25_clean   BM25 on the gold transcript of each window          (no GPU)
    dense_clean  sentence encoder on the gold transcript             (GPU helps)
    bm25_asr     BM25 on Whisper's transcript of each window         (GPU)
    dense_asr    sentence encoder on Whisper's transcript            (GPU)

Pools: within-recording always; the archive only where the task says it is
well-posed. Builds whose questions carry an ``origin`` (spliced) are also scored
per origin. Per build, it reports the diagnostics that bear on the question
"is this benchmark measuring long-audio retrieval or concatenation artifacts":
gold-edge acoustic cues, gold-span length and position, and the lexical overlap
between questions and their gold text.

Every run goes to ``runs/<run_id>/``; the summary table to
``runs/<stamp>_longaudio_summary.{json,tsv}``.
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

from lsr.audio import read_wav
from lsr.eval import bootstrap_ci, compute_metrics, ranks_from_scores, ranks_within_groups
from lsr.longaudio import FixedWindowSegmenter
from lsr.longaudio.diagnostics import boundary_cues, window_cue_strength
from lsr.retrievers.base import score_dense
from lsr.retrievers.bm25 import bm25_scores, tokenize
from lsr.store import current_manifest, write_embeddings, write_run
from lsr.tasks import check_task
from lsr.tasks.longform import LongAudioTask

REPO = Path(__file__).resolve().parents[1]
GPU_ARMS = {"dense_clean", "bm25_asr", "dense_asr"}


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


def metrics_for(ranks: np.ndarray, seed: int) -> dict:
    m = compute_metrics(ranks, ks=(1, 3, 5))
    m["Hits@1_ci95"] = bootstrap_ci(ranks, "Hits@1", seed=seed)
    m["MRR_ci95"] = bootstrap_ci(ranks, "MRR", seed=seed)
    return m


def score_all(task, scores, qids, dids, seed) -> dict:
    """Within-recording (and archive, if valid) metrics, plus per-origin splits."""
    qrels = task.qrels()
    within = ranks_within_groups(scores, qrels, qids, dids, task.query_group, task.document_group)
    out = {"within": metrics_for(within, seed)}
    if task.archive_valid:
        out["archive"] = metrics_for(ranks_from_scores(scores, qrels, qids, dids), seed)
    origins = {task.question_rows[q].get("origin") for q in qids}
    if origins != {None}:
        for o in sorted(x for x in origins if x):
            idx = [i for i, q in enumerate(qids) if task.question_rows[q].get("origin") == o]
            out[f"within_{o}"] = metrics_for(within[idx], seed)
    return out


def diagnostics(task: LongAudioTask, max_recordings: int) -> dict:
    """Gold-edge acoustic cues, gold length/position, question-gold lexical overlap."""
    rows = list(task.question_rows.values())
    lengths = [q["gold_end_s"] - q["gold_start_s"] for q in rows]
    rel_pos = [q["gold_start_s"] / task.recordings[q["recording_id"]]["duration_s"] for q in rows]
    durations = [r["duration_s"] for r in task.recordings.values()]
    out = {
        "recordings": len(durations), "questions": len(rows),
        "recording_min_median": float(np.median(durations) / 60),
        "gold_s_median": float(np.median(lengths)),
        "gold_rel_position_mean": float(np.mean(rel_pos)),
        "ambiguous_questions": task.n_ambiguous_questions,
    }
    if task.has_transcripts():
        overlap = []
        for q in rows:
            gold = set(tokenize(task.transcript_in_span(q["recording_id"], q["gold_start_s"],
                                                        q["gold_end_s"])))
            qt = set(tokenize(q["text"]))
            if qt:
                overlap.append(len(qt & gold) / len(qt))
        out["question_gold_token_overlap"] = float(np.mean(overlap))
    cues = []
    for rid in list(task.recordings)[:max_recordings]:
        edges = sorted({t for q in rows if q["recording_id"] == rid
                        for t in (q["gold_start_s"], q["gold_end_s"])})
        if edges:
            x, rate = read_wav(task.build_dir / task.recordings[rid]["audio"])
            cues.append(boundary_cues(x, rate, edges))
    if cues:
        w = np.array([c["n_edges"] for c in cues], dtype=float)
        for k in ("edge_at_pause", "random_at_pause", "pause_percentile", "jump_percentile"):
            v = np.array([c[k] for c in cues], dtype=float)
            ok = ~np.isnan(v)
            out[f"edge_{k}"] = float(np.average(v[ok], weights=w[ok])) if ok.any() else None
    return out


def transcribe(asr, task: LongAudioTask) -> dict[str, list]:
    """Recording id -> ASR chunks, cached next to the build."""
    from lsr.backends.asr import Chunk

    cache = task.build_dir / f"asr_{asr.name}.jsonl"
    done = {}
    if cache.exists():
        for line in cache.read_text().splitlines():
            row = json.loads(line)
            done[row["recording_id"]] = [Chunk(**c) for c in row["chunks"]]
    with open(cache, "a") as f:
        for rid, rec in task.recordings.items():
            if rid in done:
                continue
            done[rid] = asr.transcribe(str(task.build_dir / rec["audio"]))
            f.write(json.dumps({"recording_id": rid,
                                "chunks": [c.__dict__ for c in done[rid]]}) + "\n")
            f.flush()
    return done


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, required=True)
    cfg = load_config(parser.parse_args().config)
    cfg["_hash"] = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:12]
    seed, arms = cfg["seed"], cfg["arms"]
    stamp = time.strftime("%Y%m%d-%H%M%S")

    enc = asr = None
    if GPU_ARMS & set(arms):
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        if {"dense_clean", "dense_asr"} & set(arms):
            from lsr.backends.text import SentenceEncoder

            enc = SentenceEncoder(cfg["text_encoder"], device=device,
                                  query_prefix=cfg.get("query_prefix", ""))
        if {"bm25_asr", "dense_asr"} & set(arms):
            from lsr.backends.asr import WhisperASR

            asr = WhisperASR(cfg["asr"], device=device, batch_size=cfg.get("asr_batch_size", 8))

    summary = []
    for build_name, bcfg in cfg["builds"].items():
        build_dir = Path(bcfg["dir"])
        if not (build_dir / "recordings.jsonl").exists():
            print(f"WARNING: skipping {build_name}: no build at {build_dir}")
            continue
        gold_units = tuple(bcfg["gold_units"]) if bcfg.get("gold_units") else None
        chunks = None
        for wcfg in cfg["windows"]:
            seg = FixedWindowSegmenter(**wcfg)
            task = LongAudioTask(build_dir, seg, min_overlap_s=cfg["min_overlap_s"],
                                 gold_units=gold_units,
                                 gold_span=bcfg.get("gold_span", "gold"))
            try:
                check_task(task)
            except ValueError as e:
                print(f"WARNING: skipping {build_name} | {seg.name}: {e}")
                continue
            queries = list(task.queries())
            qids, qtext = [q.id for q in queries], [q.text for q in queries]
            docs = list(task.documents())
            dids = [d.id for d in docs]
            print(f"[{build_name} | {seg.name}] {len(qids)} questions, {len(dids)} windows, "
                  f"archive {'valid' if task.archive_valid else 'not well-posed'}")
            diag = diagnostics(task, cfg.get("diagnostic_recordings", 50)) \
                if wcfg is cfg["windows"][0] else None
            if diag:
                print("  diagnostics", {k: round(v, 3) if isinstance(v, float) else v
                                        for k, v in diag.items()})

            def text_of(kind):
                if kind == "clean":
                    return [task.transcript_in_span(task.document_group[d.id], d.start_s,
                                                    d.end_s) for d in docs]
                from lsr.backends.asr import text_in_span

                return [text_in_span(chunks[task.document_group[d.id]], d.start_s, d.end_s)
                        for d in docs]

            for arm in arms:
                if arm.endswith("_clean") and not task.has_transcripts():
                    continue
                if arm.endswith("_asr") and chunks is None:
                    t0 = time.perf_counter()
                    chunks = transcribe(asr, task)
                    print(f"  ASR done in {time.perf_counter() - t0:.0f}s")
                t0 = time.perf_counter()
                q_emb = d_emb = None
                if arm == "random":
                    rng = np.random.default_rng(seed)
                    runs = [score_all(task, rng.random((len(qids), len(dids))), qids, dids, seed)
                            for _ in range(20)]
                    result = {pool: {k: float(np.mean([r[pool][k] for r in runs]))
                                     for k in runs[0][pool] if not k.endswith("ci95")}
                              for pool in runs[0]}
                elif arm == "cue_only":
                    col = np.zeros(len(dids))
                    for rid, rec in task.recordings.items():
                        idx = [j for j, d in enumerate(docs) if task.document_group[d.id] == rid]
                        x, rate = read_wav(build_dir / rec["audio"])
                        col[idx] = window_cue_strength(
                            x, rate, [(docs[j].start_s, docs[j].end_s) for j in idx])
                    result = score_all(task, np.tile(col, (len(qids), 1)), qids, dids, seed)
                elif arm.startswith("bm25_"):
                    result = score_all(task, bm25_scores(qtext, text_of(arm[5:])),
                                       qids, dids, seed)
                else:  # dense_clean / dense_asr
                    q_emb = enc.embed_texts(qtext, as_query=True)
                    d_emb = enc.embed_texts(text_of(arm[6:]))
                    result = score_all(task, score_dense(q_emb, d_emb), qids, dids, seed)

                run_id = f"{stamp}_{build_name}_{seg.name}_{arm}"
                run_dir = REPO / "runs" / run_id
                if q_emb is not None:
                    write_embeddings(run_dir, q_emb, d_emb, qids, dids)
                manifest = current_manifest(
                    run_id=run_id, task=task.name, retriever=arm, config_hash=cfg["_hash"],
                    seed=seed, n_queries=len(qids), n_documents=len(dids),
                    wall_clock_s=round(time.perf_counter() - t0, 2))
                write_run(run_dir, manifest, {**result, "diagnostics": diag})
                for pool, m in result.items():
                    summary.append({"build": build_name, "windows": seg.name, "arm": arm,
                                    "pool": pool, "Hits@1": m["Hits@1"], "Hits@5": m["Hits@5"],
                                    "MRR": m["MRR"], "n_queries": len(qids),
                                    "windows_per_recording": len(dids) / len(task.recordings)})
                shown = {p: round(m["Hits@1"], 3) for p, m in result.items()}
                print(f"  {arm:12s} Hits@1 {shown}")

    out = REPO / "runs" / f"{stamp}_longaudio_summary"
    out.with_suffix(".json").write_text(json.dumps(summary, indent=2))
    cols = ["build", "windows", "arm", "pool", "Hits@1", "Hits@5", "MRR", "n_queries",
            "windows_per_recording"]
    lines = ["\t".join(cols)] + ["\t".join(f"{r[c]:.4f}" if isinstance(r[c], float) else str(r[c])
                                           for c in cols) for r in summary]
    out.with_suffix(".tsv").write_text("\n".join(lines) + "\n")
    print(f"\nsummary: {out.with_suffix('.tsv')}")


if __name__ == "__main__":
    main()
