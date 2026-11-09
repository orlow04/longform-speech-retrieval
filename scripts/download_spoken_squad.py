"""Download Spoken-SQuAD's JSON and one split's audio into data/datasets/spoken_squad/.

Also fetches the matching SQuAD v1.1 JSON into data/datasets/squad/: Spoken-SQuAD's
contexts are ASR transcripts, and the builder needs the text that was read aloud.

    uv run python scripts/download_spoken_squad.py            # dev audio (~4.7 GB)
    uv run python scripts/download_spoken_squad.py --split train   # ~41 GB
    uv run python scripts/download_spoken_squad.py --articles 0 10 11   # ~3 articles, ~150 MB

The release is one 46 GB zip holding ``dev_wav.zip`` and ``train_wav.zip``. This
reads the outer zip over HTTP range requests and copies out only the inner zip
asked for, then extracts it — so the dev split never costs the train download.
With ``--articles`` it goes one level further and fetches only those articles'
sentence WAVs out of the inner zip, for quick tries.
Root directory: ``$LSR_DATA`` (default ``data/datasets``).
"""

from __future__ import annotations

import argparse
import io
import os
import shutil
import struct
import threading
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

AUDIO_URL = "http://speech.ee.ntu.edu.tw/~chiahsuan/Spoken-SQuAD/Spoken-SQuAD_audio.zip"
JSON_URL = "https://raw.githubusercontent.com/Chia-Hsuan-Lee/Spoken-SQuAD/master/{}"
JSONS = {"dev": "spoken_test-v1.1.json", "train": "spoken_train-v1.1.json"}
# The text that was read aloud: Spoken-SQuAD's JSON contexts are ASR output.
SQUAD_URL = "https://rajpurkar.github.io/SQuAD-explorer/dataset/{}"
SQUAD_JSONS = {"dev": "dev-v1.1.json", "train": "train-v1.1.json"}


class HttpRangeFile(io.RawIOBase):
    """A read-only, seekable file over HTTP range requests."""

    def __init__(self, url: str):
        self.url, self.pos = url, 0
        head = urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=60)
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
        for attempt in range(5):  # the server drops connections under load
            try:
                data = urllib.request.urlopen(req, timeout=60).read()
                break
            except OSError:
                if attempt == 4:
                    raise
        buf[: len(data)] = data
        self.pos += len(data)
        return len(data)


class Window(io.RawIOBase):
    """Bytes ``[start, start + size)`` of another seekable raw file, as a file."""

    def __init__(self, raw, start: int, size: int):
        self.raw, self.start, self.size, self.pos = raw, start, size, 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        self.pos = {0: 0, 1: self.pos, 2: self.size}[whence] + offset
        return self.pos

    def readinto(self, buf):
        n = min(len(buf), self.size - self.pos)
        if n <= 0:
            return 0
        self.raw.seek(self.start + self.pos)
        data = self.raw.read(n)
        buf[: len(data)] = data
        self.pos += len(data)
        return len(data)


def inner_zip(raw: HttpRangeFile, outer: zipfile.ZipFile, name: str) -> zipfile.ZipFile:
    """Open a stored (uncompressed) member of ``outer`` as a zip, in place."""
    member = outer.getinfo(name)
    if member.compress_type != zipfile.ZIP_STORED:
        raise ValueError(f"{name} is compressed; cannot open it in place")
    raw.seek(member.header_offset)
    name_len, extra_len = struct.unpack("<HH", raw.read(30)[26:30])
    start = member.header_offset + 30 + name_len + extra_len
    return zipfile.ZipFile(io.BufferedReader(Window(raw, start, member.compress_size), 1 << 20))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--split", choices=sorted(JSONS), default="dev")
    parser.add_argument("--articles", type=int, nargs="+",
                        help="fetch only these article indices' WAVs")
    parser.add_argument("--threads", type=int, default=6)
    args = parser.parse_args()
    root = Path(os.environ.get("LSR_DATA", "data/datasets")) / "spoken_squad"
    root.mkdir(parents=True, exist_ok=True)

    json_path = root / JSONS[args.split]
    if not json_path.exists():
        urllib.request.urlretrieve(JSON_URL.format(JSONS[args.split]), json_path)
        print(f"wrote {json_path}")
    squad_path = root.parent / "squad" / SQUAD_JSONS[args.split]
    if not squad_path.exists():
        squad_path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(SQUAD_URL.format(SQUAD_JSONS[args.split]), squad_path)
        print(f"wrote {squad_path}")

    if args.articles:
        member_name = f"Spoken-SQuAD_audio/{args.split}_wav.zip"
        local = threading.local()

        def handle() -> zipfile.ZipFile:
            # One connection and one parsed directory per thread: zipfile
            # objects over a shared seekable stream are not thread-safe.
            if not hasattr(local, "inner"):
                raw = HttpRangeFile(AUDIO_URL)
                outer = zipfile.ZipFile(io.BufferedReader(raw, 1 << 16))
                local.inner = inner_zip(raw, outer, member_name)
            return local.inner

        prefixes = tuple(f"{a}_" for a in args.articles)
        members = [m for m in handle().infolist()
                   if m.filename.rsplit("/", 1)[-1].startswith(prefixes)]
        # Size check, not existence: an interrupted run leaves truncated files.
        todo = [m.filename for m in members
                if not (root / m.filename).exists()
                or (root / m.filename).stat().st_size != m.file_size]
        print(f"fetching {len(todo)} of {len(members)} WAVs for articles {args.articles}")
        with ThreadPoolExecutor(args.threads) as pool:
            list(pool.map(lambda name: handle().extract(name, root), todo))
        print(f"extracted into {root}")
        return

    inner_path = root / f"{args.split}_wav.zip"
    if not inner_path.exists():
        outer = zipfile.ZipFile(io.BufferedReader(HttpRangeFile(AUDIO_URL), 1 << 24))
        member = outer.getinfo(f"Spoken-SQuAD_audio/{args.split}_wav.zip")
        print(f"copying {member.filename} ({member.file_size / 2**30:.1f} GB)")
        tmp = inner_path.with_suffix(".part")
        with outer.open(member) as src, open(tmp, "wb") as dst:
            shutil.copyfileobj(src, dst, 1 << 26)
        tmp.rename(inner_path)

    with zipfile.ZipFile(inner_path) as z:
        z.extractall(root)
        top = sorted({n.split("/", 1)[0] for n in z.namelist()})
    print(f"extracted into {root}: {top}")


if __name__ == "__main__":
    main()
