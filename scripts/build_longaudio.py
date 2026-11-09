"""Build every long-audio dataset and variant named in a config.

    uv run python scripts/build_longaudio.py --config configs/longaudio.yaml

Sources first (Spoken-SQuAD, CORAAL-QA), then variants derived from them
(shuffled, smoothed, spliced), in config order. An existing build is skipped;
delete its directory to rebuild. Each build leaves ``build.json`` with its
config, counts and git state.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from run_longaudio import load_config

from lsr.store import current_manifest
from lsr.tasks import variants
from lsr.tasks.coraal_qa import build_coraal_qa
from lsr.tasks.spoken_squad import build_spoken_squad


def build_one(name: str, spec: dict, dirs: dict[str, Path]) -> dict:
    out = Path(spec["dir"])
    kind = spec["build"]
    if kind == "spoken_squad":
        return build_spoken_squad(Path(spec["json"]), Path(spec["wav_dir"]), out,
                                  spec.get("articles"),
                                  Path(spec["text_json"]) if spec.get("text_json") else None)
    if kind == "coraal_qa":
        return build_coraal_qa(Path(spec["qa_dir"]), Path(spec["coraal_dir"]), out)
    if kind == "shuffled":
        return variants.shuffled(dirs[spec["from"]], out, seed=spec.get("seed", 0))
    if kind == "smoothed":
        return variants.smoothed(dirs[spec["from"]], out)
    if kind == "spliced":
        return variants.spliced(dirs[spec["from"]], dirs[spec["host"]], out,
                                pieces_per_host=spec.get("pieces_per_host", 5),
                                seed=spec.get("seed", 0))
    raise ValueError(f"{name}: unknown build kind {kind!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, required=True)
    cfg = load_config(parser.parse_args().config)
    dirs = {name: Path(spec["dir"]) for name, spec in cfg["builds"].items()}
    for name, spec in cfg["builds"].items():
        out = dirs[name]
        if spec["build"] == "alias":
            continue
        if (out / "recordings.jsonl").exists():
            print(f"[{name}] exists, skipped")
            continue
        missing = [d for d in (spec.get("from"), spec.get("host")) if d and
                   not (dirs[d] / "recordings.jsonl").exists()]
        if missing:
            print(f"[{name}] skipped: needs {missing} built first")
            continue
        counts = build_one(name, spec, dirs)
        manifest = current_manifest(run_id=f"build-{name}", task=name, retriever="none",
                                    config_hash="", seed=spec.get("seed", 0),
                                    n_queries=counts.get("questions", 0),
                                    n_documents=counts.get("recordings", counts.get("articles", 0)))
        (out / "build.json").write_text(json.dumps(
            {"spec": spec, "counts": counts, "provenance": asdict(manifest)}, indent=2))
        print(f"[{name}] {counts}")


if __name__ == "__main__":
    main()
