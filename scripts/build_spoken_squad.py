"""Build one Spoken-SQuAD split: paragraph manifests, article recordings, a log.

    uv run python scripts/build_spoken_squad.py --config configs/spoken_squad_dev.yaml

Writes into the configured ``out_dir`` and leaves ``build.json`` there with the
config, the counts and the git state, so the output explains itself.
"""

from __future__ import annotations

import argparse
import json
import os
import string
from dataclasses import asdict
from pathlib import Path

import yaml

from lsr.store import current_manifest
from lsr.tasks import ManifestTask, check_task
from lsr.tasks.spoken_squad import build_spoken_squad


def load_config(path: Path) -> dict[str, str]:
    env = {"LSR_DATA": "data", **os.environ}
    raw = yaml.safe_load(path.read_text())
    return {k: string.Template(str(v)).substitute(env) for k, v in raw.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, required=True)
    cfg = load_config(parser.parse_args().config)
    out_dir = Path(cfg["out_dir"])

    counts = build_spoken_squad(Path(cfg["json"]), Path(cfg["wav_dir"]), out_dir)
    task = ManifestTask("spoken_squad", out_dir, "text", "audio", audio_root=out_dir)
    check_task(task)

    manifest = current_manifest(
        run_id=f"build-{out_dir.name}", task=task.name, retriever="none",
        config_hash="", seed=0, n_queries=counts["questions"],
        n_documents=counts["paragraphs"],
    )
    (out_dir / "build.json").write_text(json.dumps(
        {"config": cfg, "counts": counts, "provenance": asdict(manifest)}, indent=2
    ))
    print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()
