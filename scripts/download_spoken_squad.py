"""Download Spoken-SQuAD's JSON and one split's audio into data/datasets/spoken_squad/.

    uv run python scripts/download_spoken_squad.py            # dev audio (~4.7 GB)
    uv run python scripts/download_spoken_squad.py --split train   # ~41 GB

The release is one 46 GB zip holding ``dev_wav.zip`` and ``train_wav.zip``. This
reads the outer zip over HTTP range requests and copies out only the inner zip
asked for, then extracts it — so the dev split never costs the train download.
Root directory: ``$LSR_DATA`` (default ``data/datasets``).
"""

from __future__ import annotations

import argparse
import io
import os
import shutil
import urllib.request
import zipfile
from pathlib import Path

AUDIO_URL = "http://speech.ee.ntu.edu.tw/~chiahsuan/Spoken-SQuAD/Spoken-SQuAD_audio.zip"
JSON_URL = "https://raw.githubusercontent.com/Chia-Hsuan-Lee/Spoken-SQuAD/master/{}"
JSONS = {"dev": "spoken_test-v1.1.json", "train": "spoken_train-v1.1.json"}


class HttpRangeFile(io.RawIOBase):
    """A read-only, seekable file over HTTP range requests."""

    def __init__(self, url: str):
        self.url, self.pos = url, 0
        head = urllib.request.urlopen(urllib.request.Request(url, method="HEAD"))
        self.size = int(head.headers["Content-Length"])

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        base = {0: 0, 1: self.pos, 2: self.size}[whence]
        self.pos = base + offset
        return self.pos

    def readinto(self, buf):
        n = min(len(buf), self.size - self.pos)
        if n <= 0:
            return 0
        req = urllib.request.Request(
            self.url, headers={"Range": f"bytes={self.pos}-{self.pos + n - 1}"})
        data = urllib.request.urlopen(req).read()
        buf[: len(data)] = data
        self.pos += len(data)
        return len(data)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--split", choices=sorted(JSONS), default="dev")
    args = parser.parse_args()
    root = Path(os.environ.get("LSR_DATA", "data/datasets")) / "spoken_squad"
    root.mkdir(parents=True, exist_ok=True)

    json_path = root / JSONS[args.split]
    if not json_path.exists():
        urllib.request.urlretrieve(JSON_URL.format(JSONS[args.split]), json_path)
        print(f"wrote {json_path}")

    inner_zip = root / f"{args.split}_wav.zip"
    if not inner_zip.exists():
        outer = zipfile.ZipFile(io.BufferedReader(HttpRangeFile(AUDIO_URL), 1 << 24))
        member = outer.getinfo(f"Spoken-SQuAD_audio/{args.split}_wav.zip")
        print(f"copying {member.filename} ({member.file_size / 2**30:.1f} GB)")
        tmp = inner_zip.with_suffix(".part")
        with outer.open(member) as src, open(tmp, "wb") as dst:
            shutil.copyfileobj(src, dst, 1 << 26)
        tmp.rename(inner_zip)

    with zipfile.ZipFile(inner_zip) as z:
        z.extractall(root)
        top = sorted({n.split("/", 1)[0] for n in z.namelist()})
    print(f"extracted into {root}: {top}")


if __name__ == "__main__":
    main()
