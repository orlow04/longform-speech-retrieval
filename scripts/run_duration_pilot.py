"""Pilot: does CLASP retrieval degrade as the spoken document gets longer?

    uv run --extra clasp --extra cascade python scripts/run_duration_pilot.py \
        --config configs/duration_pilot.yaml

The Notion "Validar" card: CLASP with mean pooling on audio of 30 s, 1 min and
5 min, built by simple concatenation of passages, plus a look at the latent
space. Here the passages are Spoken-SQuAD sentence clips (one TTS clip per
sentence of the SQuAD text), joined in article order into documents of each
target duration (``lsr.longaudio.grouping``); ``paragraph`` uses the original
paragraphs, CLASP's training unit, as an anchor to its known numbers.

A text question retrieves the document holding the sentence that contains its
answer, from every document of that duration across all articles. Arms:

    clasp_mean       CLASP audio, one arm per entry of ``checkpoints``: HuBERT
                     + spectrogram image averaged over 20 s chunks, fused once
                     (CLASP's own long-audio rule); query = LaBSE(question)
    labse_text_mean  the same pooling rule on the gold text: mean of LaBSE
                     sentence embeddings. Separates "longer documents are
                     harder" from "the audio encoder degrades with length"
    bm25_text        BM25 on the document's gold text

Two confounds are controlled by construction:

- **same queries everywhere:** only questions whose gold sentence lies inside
  one document at *every* duration are scored;
- **pool size:** at 30 s there are ~10x more candidates than at 5 min, so
  besides the full pool every condition is also scored against a fixed pool of
  ``pool_size`` (gold + random negatives, exact expectation;
  ``lsr.eval.sampled_pool_values``).

Latent-space statistics (``lsr.eval.geometry``) are reported per condition.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import yaml

from lsr.audio import read_wav
from lsr.eval import (
    bootstrap_ci,
    compute_metrics,
    mean_ci,
    ranks_from_scores,
    sampled_pool_values,
)
from lsr.eval.geometry import embedding_stats, score_stats
from lsr.longaudio.grouping import Group, group_containing, group_units
from lsr.retrievers.bm25 import bm25_scores
from lsr.store import current_manifest, write_embeddings, write_run
from lsr.tasks.longform import read_jsonl

REPO = Path(__file__).resolve().parents[1]
KS = (1, 5, 10)


def load_build(build_dir: Path):
    recordings = {r["recording_id"]: r for r in read_jsonl(build_dir / "recordings.jsonl")}
    questions = read_jsonl(build_dir / "questions.jsonl")
    units = {rid: read_jsonl(build_dir / r["transcript"]) for rid, r in recordings.items()}
    return recordings, questions, units


def make_groups(cond, recordings, units) -> dict[str, list[Group]]:
    out = {}
    for rid, rec in recordings.items():
        if cond == "paragraph":
            out[rid] = [Group(rid, k, p["start_s"], p["end_s"],
                              tuple(i for i, u in enumerate(units[rid])
                                    if p["start_s"] - 0.05 <= u["start_s"] < p["end_s"] - 0.05))
                        for k, p in enumerate(rec["pieces"])]
        else:
            out[rid] = group_units(rid, units[rid], float(cond))
    return out


def chunk_features(enc, build_dir, recordings, groups_by_cond, cache: Path):
    """{cond: [ChunkFeatures per group]} with a per-condition cache on disk."""
    from lsr.backends.clasp import ChunkFeatures

    out, todo = {}, []
    for cond in groups_by_cond:
        f = cache / f"clasp_chunks_{cond}.npz"
        if f.exists():
            z = np.load(f)
            off = z["offsets"]
            out[cond] = [ChunkFeatures(z["hubert"][a:b], z["image"][a:b])
                         for a, b in zip(off[:-1], off[1:])]
        else:
            todo.append(cond)
    if not todo:
        return out
    feats = {c: [] for c in todo}
    t0 = time.perf_counter()
    for n, (rid, rec) in enumerate(recordings.items(), 1):
        x, rate = read_wav(build_dir / rec["audio"])
        for c in todo:
            for g in groups_by_cond[c][rid]:
                feats[c].append(enc.chunk_features(x[int(g.start_s * rate):int(g.end_s * rate)],
                                                   rate))
        print(f"  features: recording {n}/{len(recordings)} "
              f"({time.perf_counter() - t0:.0f}s)", flush=True)
    cache.mkdir(parents=True, exist_ok=True)
    for c in todo:
        off = np.cumsum([0] + [len(f.hubert) for f in feats[c]])
        np.savez(cache / f"clasp_chunks_{c}.npz", offsets=off,
                 hubert=np.concatenate([f.hubert for f in feats[c]]),
                 image=np.concatenate([f.image for f in feats[c]]))
        out[c] = feats[c]
    return out


def score_condition(scores, gold, qids, dids, pool_size, seed) -> dict:
    qrels = {q: {dids[g]} for q, g in zip(qids, gold)}
    ranks = ranks_from_scores(scores, qrels, qids, dids)
    full = compute_metrics(ranks, ks=KS)
    full["Hits@1_ci95"] = bootstrap_ci(ranks, "Hits@1", seed=seed)
    full["MRR_ci95"] = bootstrap_ci(ranks, "MRR", seed=seed)
    vals = sampled_pool_values(ranks, len(dids), pool_size, ks=KS)
    fixed = {k: float(v.mean()) for k, v in vals.items()}
    fixed["Hits@1_ci95"] = mean_ci(vals["Hits@1"], seed=seed)
    fixed["MRR_ci95"] = mean_ci(vals["MRR"], seed=seed)
    fixed["chance_Hits@1"] = 1 / pool_size
    return {"full_pool": full, "fixed_pool": fixed, "n_documents": len(dids),
            "pool_size": pool_size}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, required=True)
    cfg = yaml.safe_load(parser.parse_args().config.read_text())
    cfg_hash = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:12]
    seed, arms = cfg["seed"], cfg["arms"]
    build_dir = Path(cfg["build"])
    conds = [str(c) for c in cfg["durations"]]
    stamp = time.strftime("%Y%m%d-%H%M%S")

    recordings, questions, units = load_build(build_dir)
    groups = {c: make_groups(c, recordings, units) for c in conds}
    flat = {c: [g for rid in recordings for g in groups[c][rid]] for c in conds}

    # Gold group per condition; keep only questions resolvable in every condition.
    gold, dropped = {}, 0
    for q in questions:
        hit = {c: group_containing(groups[c][q["recording_id"]], q["gold_start_s"], q["gold_end_s"])
               for c in conds}
        if all(hit.values()):
            gold[q["query_id"]] = {c: hit[c].id for c in conds}
        else:
            dropped += 1
    qrows = [q for q in questions if q["query_id"] in gold]
    qids, qtext = [q["query_id"] for q in qrows], [q["text"] for q in qrows]
    pool_size = cfg.get("pool_size") or min(len(flat[c]) for c in conds)
    print(f"{len(recordings)} recordings, {len(qids)} questions "
          f"({dropped} dropped: gold straddles a document boundary somewhere), "
          f"fixed pool {pool_size}")
    for c in conds:
        d = [g.duration_s for g in flat[c]]
        print(f"  {c:>9}: {len(d)} documents, duration median {np.median(d):.1f}s "
              f"[{min(d):.1f}, {max(d):.1f}]")

    import torch  # noqa: F401  (backends below need it; fail early if missing)

    from lsr.backends.clasp import ClaspEncoder, l2n, load_fusion

    # One clasp_mean arm per checkpoint; the frozen encoders are shared, so the chunk
    # features are extracted once and only the fusion head changes.
    ckpts = {k: Path(v).expanduser() for k, v in cfg["checkpoints"].items()}
    for name, path in ckpts.items():
        if not path.exists():
            raise FileNotFoundError(f"checkpoint {name!r}: {path}")
    ckpt_sha = {k: hashlib.sha256(p.read_bytes()).hexdigest() for k, p in ckpts.items()}
    enc = ClaspEncoder(str(next(iter(ckpts.values()))), device=cfg.get("device"),
                       chunk_s=cfg.get("chunk_s", 20.0))
    fusions = {k: load_fusion(p, enc.device) for k, p in ckpts.items()}
    arms = [a for arm in arms
            for a in ([f"clasp_mean:{k}" for k in ckpts] if arm == "clasp_mean" else [arm])]
    q_emb = enc.embed_texts(qtext)
    feats = chunk_features(enc, build_dir, recordings, groups,
                           build_dir / cfg.get("cache", "cache")) \
        if any(a.startswith("clasp_mean") for a in arms) else {}
    unit_emb = {}
    if "labse_text_mean" in arms:
        for rid in recordings:
            unit_emb[rid] = enc.embed_texts([u["text"] for u in units[rid]])

    summary = []
    for c in conds:
        docs = flat[c]
        dids = [g.id for g in docs]
        col = {d: j for j, d in enumerate(dids)}
        gidx = np.array([col[gold[q][c]] for q in qids])
        text = [" ".join(units[g.recording_id][i]["text"] for i in g.units) for g in docs]
        for arm in arms:
            t0 = time.perf_counter()
            d_emb, geom = None, {}
            if arm.startswith("clasp_mean:"):
                raw = np.stack(enc.pool(feats[c], "mean", fusions[arm.split(":", 1)[1]]))
                d_emb = l2n(raw)
                geom = embedding_stats(raw, n_sub=pool_size, seed=seed)
            elif arm == "labse_text_mean":
                raw = np.stack([unit_emb[g.recording_id][list(g.units)].mean(axis=0)
                                for g in docs])
                d_emb = l2n(raw)
                geom = embedding_stats(raw, n_sub=pool_size, seed=seed)
            elif arm == "bm25_text":
                scores = bm25_scores(qtext, text)
            else:
                raise ValueError(f"unknown arm {arm!r}")
            if d_emb is not None:
                scores = q_emb @ d_emb.T
                geom.update(score_stats(scores, gidx))
            result = score_condition(scores, gidx, qids, dids, pool_size, seed)
            result["geometry"] = geom
            result["doc_duration_s_median"] = float(np.median([g.duration_s for g in docs]))
            result["dropped_questions"] = dropped

            run_id = f"{stamp}_duration_{c}_{arm.replace(':', '-')}"
            run_dir = REPO / "runs" / run_id
            if d_emb is not None:
                write_embeddings(run_dir, q_emb, d_emb, qids, dids)
            manifest = current_manifest(
                run_id=run_id, task=f"spoken_squad_duration_{c}_{build_dir.name}", retriever=arm,
                config_hash=cfg_hash, seed=seed, n_queries=len(qids), n_documents=len(dids),
                wall_clock_s=round(time.perf_counter() - t0, 2),
                **({"notes": f"checkpoint {ckpts[arm.split(':', 1)[1]]}",
                    "checkpoint_sha256": ckpt_sha[arm.split(":", 1)[1]]}
                   if arm.startswith("clasp_mean:") else {}))
            write_run(run_dir, manifest, result)
            row = {"duration": c, "arm": arm, "n_docs": len(dids), "n_queries": len(qids),
                   "full_Hits@1": result["full_pool"]["Hits@1"],
                   "full_Hits@10": result["full_pool"]["Hits@10"],
                   "full_MRR": result["full_pool"]["MRR"],
                   "fixed_Hits@1": result["fixed_pool"]["Hits@1"],
                   "fixed_Hits@1_lo": result["fixed_pool"]["Hits@1_ci95"][0],
                   "fixed_Hits@1_hi": result["fixed_pool"]["Hits@1_ci95"][1],
                   "fixed_MRR": result["fixed_pool"]["MRR"],
                   **{k: geom.get(k, float("nan")) for k in
                      ("raw_norm", "pairwise_cos", "effective_rank", "effective_rank_sub",
                       "top1_var", "top1_var_sub", "gold_cos",
                       "best_negative_cos", "margin")}}
            summary.append(row)
            print(f"[{c:>9} | {arm:22s}] full H@1 {row['full_Hits@1']:.3f}  "
                  f"fixed({pool_size}) H@1 {row['fixed_Hits@1']:.3f} "
                  f"[{row['fixed_Hits@1_lo']:.3f}, {row['fixed_Hits@1_hi']:.3f}]  "
                  f"pairwise_cos {row['pairwise_cos']:.3f}  "
                  f"eff_rank@{pool_size} {row['effective_rank_sub']:.1f}")

    out = REPO / "runs" / f"{stamp}_duration_pilot_summary"
    out.with_suffix(".json").write_text(json.dumps(
        {"config": cfg, "pool_size": pool_size, "dropped_questions": dropped, "rows": summary},
        indent=2))
    cols = list(summary[0])
    lines = ["\t".join(cols)] + ["\t".join(f"{r[k]:.4f}" if isinstance(r[k], float) else str(r[k])
                                           for k in cols) for r in summary]
    out.with_suffix(".tsv").write_text("\n".join(lines) + "\n")
    print(f"\nsummary: {out.with_suffix('.tsv')}")


if __name__ == "__main__":
    main()
