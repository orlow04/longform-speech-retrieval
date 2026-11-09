"""Download CORAAL-QA and the CORAAL components it covers into data/datasets/.

    uv run python scripts/download_coraal.py                 # all five components
    uv run python scripts/download_coraal.py --components LES VLD   # smaller first try

Writes ``coraal_qa/`` (the questions, cloned from GitHub) and
``coraal/<COMPONENT>/`` (audio and time-aligned transcripts, latest release per
component, from lingtools.uoregon.edu). CORAAL is CC BY-NC-SA 4.0. Root:
``$LSR_DATA`` (default ``data/datasets``). Re-running skips what is present.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import tarfile
import urllib.request
from pathlib import Path

BASE = "https://lingtools.uoregon.edu/coraal"
QA_REPO = "https://github.com/christinachance/CORAAL-QA.git"
COMPONENTS = ["DCB", "LES", "PRV", "ROC", "VLD"]  # the ones CORAAL-QA labels


def listing(url: str) -> list[str]:
    html = urllib.request.urlopen(url).read().decode("utf-8", "replace")
    return re.findall(r'href="([^"?/][^"]*)"', html)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--components", nargs="+", default=COMPONENTS, choices=COMPONENTS)
    args = parser.parse_args()
    root = Path(os.environ.get("LSR_DATA", "data/datasets"))

    qa = root / "coraal_qa"
    if not qa.exists():
        subprocess.run(["git", "clone", "--depth", "1", QA_REPO, str(qa)], check=True)

    for comp in args.components:
        dest = root / "coraal" / comp
        dest.mkdir(parents=True, exist_ok=True)
        versions = sorted(v for v in listing(f"{BASE}/{comp.lower()}/") if v[:2] == "20")
        url = f"{BASE}/{comp.lower()}/{versions[-1]}"
        wanted = re.compile(r"_(audio_part\d+|textfiles)_.*\.tar\.gz$")
        files = [f for f in listing(url) if wanted.search(f)]
        for name in files:
            marker = dest / f".{name}.done"
            if marker.exists():
                continue
            print(f"{comp}: {name}", flush=True)
            archive = dest / name
            urllib.request.urlretrieve(url + name, archive)
            with tarfile.open(archive) as tar:
                tar.extractall(dest, filter="data")
            archive.unlink()
            marker.touch()
    print(f"done: {root / 'coraal'}")


if __name__ == "__main__":
    main()
