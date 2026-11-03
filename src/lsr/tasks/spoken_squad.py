"""Build Spoken-SQuAD tasks from the raw release.

Spoken-SQuAD is SQuAD with every context read aloud by TTS, one WAV per
sentence, named ``{article}_{paragraph}_{chunk}.wav`` (indices into the SQuAD
JSON). Seed-VC voice-converted copies append the voice: ``..._{chunk}__{voice}.wav``.
Questions exist only as text.

Two granularities come out of one pass:

- **Paragraph** — one recording per paragraph (its chunks concatenated in chunk
  order), written as CLSR-format manifests (``que.txt`` / ``con.txt`` /
  ``wav.scp``, one row per question) so ``ManifestTask("spoken_squad", ...)``
  loads it. This is CLSR's direction: text question -> spoken context. CLASP's
  task over the same audio is the reverse self-retrieval, a different task.
- **Article** — one recording per article (its paragraphs concatenated in
  order), with every paragraph's start and end time recorded in
  ``articles.jsonl``. A question's gold span is its paragraph's span. This is
  the long-form setting: several minutes of audio per article, built from data
  already on hand, with exact ground truth.

Audio I/O is the standard library's ``wave``, so this runs in the core
environment. It handles PCM WAV only, and refuses to concatenate files whose
format differs rather than resampling silently.
"""

from __future__ import annotations

import json
import wave
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Paragraph:
    article: int
    paragraph: int
    title: str
    context: str
    questions: list[tuple[str, str]]  # (squad qa id, question text)
    wav_paths: list[Path] = field(default_factory=list)

    @property
    def id(self) -> str:
        return f"{self.article}_{self.paragraph}"


def collect_paragraphs(json_path: Path, wav_dir: Path) -> tuple[list[Paragraph], int]:
    """Pair each SQuAD paragraph with its sentence WAVs, in chunk order.

    Returns the paragraphs that have audio and the count of those that do not
    (Spoken-SQuAD drops some; the count belongs in the build log, not in a
    silent filter).
    """
    chunks: dict[tuple[int, int], list[tuple[int, Path]]] = defaultdict(list)
    voices: set[str] = set()
    for wav in sorted(Path(wav_dir).glob("*.wav")):
        stem, _, voice = wav.stem.partition("__")
        try:
            a, p, c = (int(x) for x in stem.split("_"))
        except ValueError:
            continue
        voices.add(voice)
        chunks[(a, p)].append((c, wav))
    if len(voices) > 1:
        raise ValueError(f"{wav_dir} mixes voices {sorted(voices)}; build one voice at a time")

    data = json.loads(Path(json_path).read_text(encoding="utf-8"))["data"]
    out, missing = [], 0
    for a, article in enumerate(data):
        for p, para in enumerate(article["paragraphs"]):
            found = sorted(chunks.get((a, p), []))
            if not found:
                missing += 1
                continue
            idx = [c for c, _ in found]
            if idx != list(range(len(idx))):
                raise ValueError(f"paragraph {a}_{p} has non-contiguous chunks {idx}")
            out.append(Paragraph(
                article=a, paragraph=p, title=article.get("title", ""),
                context=para["context"],
                questions=[(qa["id"], qa["question"]) for qa in para["qas"]],
                wav_paths=[w for _, w in found],
            ))
    return out, missing


def concat_wavs(paths: list[Path], out_path: Path) -> list[tuple[float, float]]:
    """Concatenate PCM WAVs into ``out_path``; return each input's (start_s, end_s)."""
    if not paths:
        raise ValueError("nothing to concatenate")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    spans, n_frames, params = [], 0, None
    with wave.open(str(out_path), "wb") as out:
        for path in paths:
            with wave.open(str(path), "rb") as src:
                fmt = (src.getnchannels(), src.getsampwidth(), src.getframerate())
                if params is None:
                    params = fmt
                    out.setnchannels(fmt[0])
                    out.setsampwidth(fmt[1])
                    out.setframerate(fmt[2])
                elif fmt != params:
                    raise ValueError(f"{path} has format {fmt}, expected {params}")
                frames = src.readframes(src.getnframes())
                count = src.getnframes()
            out.writeframes(frames)
            spans.append((n_frames / params[2], (n_frames + count) / params[2]))
            n_frames += count
    return spans


def build_spoken_squad(json_path: Path, wav_dir: Path, out_dir: Path) -> dict[str, int]:
    """Write paragraph manifests, paragraph and article audio, and ``articles.jsonl``.

    Layout under ``out_dir`` (one split)::

        que.txt  con.txt  wav.scp     CLSR format, one row per question
        articles.jsonl                one line per article, see module docstring
        audio/paragraphs/{a}_{p}.wav
        audio/articles/{a}.wav

    Audio paths in the manifests are relative to ``out_dir``, so the directory
    can be rsynced as a whole; pass ``audio_root=out_dir`` to ``ManifestTask``.
    Returns counts for the build log.
    """
    out_dir = Path(out_dir)
    paragraphs, missing = collect_paragraphs(json_path, wav_dir)

    by_article: dict[int, list[Paragraph]] = defaultdict(list)
    que, con, scp = [], [], []
    for para in paragraphs:
        rel = Path("audio/paragraphs") / f"{para.id}.wav"
        concat_wavs(para.wav_paths, out_dir / rel)
        by_article[para.article].append(para)
        context = " ".join(para.context.split())
        for qid, question in para.questions:
            que.append(f"{qid} {' '.join(question.split())}\n")
            con.append(f"{qid} {context}\n")
            scp.append(f"{qid} {rel}\n")

    with open(out_dir / "articles.jsonl", "w", encoding="utf-8") as f:
        for a, paras in sorted(by_article.items()):
            rel = Path("audio/articles") / f"{a}.wav"
            spans = concat_wavs(
                [out_dir / "audio/paragraphs" / f"{p.id}.wav" for p in paras], out_dir / rel
            )
            f.write(json.dumps({
                "article_id": str(a),
                "title": paras[0].title,
                "audio": str(rel),
                "duration_s": spans[-1][1],
                "paragraphs": [
                    {"paragraph_id": p.id, "start_s": s, "end_s": e,
                     "question_ids": [q for q, _ in p.questions]}
                    for p, (s, e) in zip(paras, spans)
                ],
            }) + "\n")

    for name, lines in (("que.txt", que), ("con.txt", con), ("wav.scp", scp)):
        (out_dir / name).write_text("".join(lines), encoding="utf-8")
    return {
        "articles": len(by_article),
        "paragraphs": len(paragraphs),
        "paragraphs_without_audio": missing,
        "questions": len(que),
    }
