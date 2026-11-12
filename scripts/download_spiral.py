"""Download SPIRAL (Lin et al., SpeechPrune, ICME 2025) into data/datasets/spiral/.

    uv run python scripts/download_spiral.py

1,012 synthetic recordings (GPT-4o scripts, StyleTTS2 voices; lectures,
meetings, conversations; ~70-100 s each, 24 kHz), one question per recording
about one "key sentence" inside it. Two sources:

- annotations: ``data.jsonl`` and ``data_h.jsonl`` (the hard subset) from
  github.com/linyueqian/SPIRAL_Dataset, pinned to a commit and checksummed;
- audio: ``spiral_dataset.tar.gz`` (3.5 GB) from the authors' Box share
  (duke.box.com/v/spiral-dataset). The download resumes if interrupted. The
  archive holds 1,050 WAVs; the 1,012 that ``data.jsonl`` names are the dataset.
  Its own copy of the annotations differs from GitHub's only in the authors'
  absolute paths.

Result: ``<root>/data.jsonl``, ``<root>/data_h.jsonl``, ``<root>/wavs/*.wav``.
Root directory: ``$LSR_DATA`` (default ``data/datasets``).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tarfile
import urllib.error
import urllib.request
from pathlib import Path

COMMIT = "8a8906f746624bb55047fad7e33ed8eef1b1d314"
JSON_URL = f"https://raw.githubusercontent.com/linyueqian/SPIRAL_Dataset/{COMMIT}/{{}}"
JSONS = {
    "data.jsonl": "485225b9ef52c98298ceb06745c7acf022dcc7c6436c610a17dbd11e8022519f",
    "data_h.jsonl": "a6c33e2a9ebf8f79c298f6aeb86a904ea136bc3bd845cc87c25fd6812750328c",
}
# Box refuses its direct-link form for this account; the shared-file endpoint works
# with the share's vanity name, and wants a browser User-Agent.
AUDIO_URL = ("https://duke.app.box.com/index.php?rm=box_download_shared_file"
             "&vanity_name=spiral-dataset&file_id=f_1718970573016")
AUDIO_SIZE = 3_481_512_068
AUDIO_SHA256 = "3001f10565b1b62ea5869f7bbfd3630531ad96ea40ab60dba3974a64d628c094"
UA = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
      "Chrome/120.0 Safari/537.36")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fetch(url: str, dest: Path, size: int | None = None, attempts: int = 20) -> None:
    """Download ``url`` to ``dest``, resuming a partial file with range requests."""
    part = dest.with_name(dest.name + ".part")
    for attempt in range(1, attempts + 1):
        have = part.stat().st_size if part.exists() else 0
        if size is not None and have >= size:
            break
        req = urllib.request.Request(url, headers={"User-Agent": UA, "Range": f"bytes={have}-"})
        try:
            with urllib.request.urlopen(req, timeout=120) as r, part.open("ab") as f:
                if have and r.status != 206:
                    raise OSError(f"server ignored the range request (HTTP {r.status})")
                while block := r.read(1 << 20):
                    f.write(block)
                    have += len(block)
                    if size and have % (256 << 20) < (1 << 20):
                        print(f"  {have / 1e9:.2f} / {size / 1e9:.2f} GB", flush=True)
            if size is None:
                break
        except (OSError, urllib.error.URLError) as e:
            print(f"  attempt {attempt}: {e}; resuming at {have} bytes", flush=True)
    got = part.stat().st_size if part.exists() else 0
    if size is not None and got != size:
        raise OSError(f"{dest.name}: got {got} bytes, expected {size}")
    part.replace(dest)


def extract_wavs(archive: Path, wav_dir: Path) -> int:
    """Copy out the archive's ``wavs/*.wav`` by base name; nothing else is written.

    The archive was made on macOS: every file has an AppleDouble twin
    (``._name.wav``, metadata, not audio), skipped here.
    """
    wav_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar:
            name = Path(member.name)
            if not (member.isfile() and name.suffix == ".wav" and name.parent.name == "wavs"
                    and not name.name.startswith("._")):
                continue
            with tar.extractfile(member) as src, (wav_dir / name.name).open("wb") as dst:
                shutil.copyfileobj(src, dst)
            n += 1
    return n


def missing_wavs(jsonl: Path, wav_dir: Path) -> list[str]:
    """Audio files ``data.jsonl`` names that are not in ``wav_dir``."""
    names = [name for line in jsonl.read_text(encoding="utf-8").splitlines() if line.strip()
             for name in json.loads(line)["audio_data"]["combined"]["audio_files"]]
    return [n for n in names if not (wav_dir / n).exists()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--keep-archive", action="store_true",
                        help="keep spiral_dataset.tar.gz after extracting it")
    args = parser.parse_args()
    root = Path(os.environ.get("LSR_DATA", "data/datasets")) / "spiral"
    root.mkdir(parents=True, exist_ok=True)

    for name, digest in JSONS.items():
        path = root / name
        if not path.exists():
            fetch(JSON_URL.format(name), path)
        if sha256(path) != digest:
            raise OSError(f"{path}: checksum differs from the pinned commit {COMMIT[:7]}")
        print(f"ok {path}")

    wav_dir = root / "wavs"
    if not missing_wavs(root / "data.jsonl", wav_dir):
        print(f"ok {wav_dir}: every WAV named in data.jsonl is there")
        return
    archive = root / "spiral_dataset.tar.gz"
    if not archive.exists():
        print(f"downloading {archive.name} ({AUDIO_SIZE / 1e9:.1f} GB)")
        fetch(AUDIO_URL, archive, AUDIO_SIZE)
    if sha256(archive) != AUDIO_SHA256:
        raise OSError(f"{archive}: checksum differs from the one recorded on 2026-11-11")
    n = extract_wavs(archive, wav_dir)
    missing = missing_wavs(root / "data.jsonl", wav_dir)
    if missing:
        raise OSError(f"{archive}: {len(missing)} WAVs named in data.jsonl are not in it, "
                      f"e.g. {missing[:3]}")
    print(f"ok {wav_dir}: {n} WAVs extracted")
    if not args.keep_archive:
        archive.unlink()


if __name__ == "__main__":
    main()
