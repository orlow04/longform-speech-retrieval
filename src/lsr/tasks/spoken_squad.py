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


def build_spoken_squad(
    json_path: Path, wav_dir: Path, out_dir: Path, articles: list[int] | None = None
) -> dict[str, int]:
    """Write paragraph manifests, paragraph and article audio, and ``articles.jsonl``.

    Layout under ``out_dir`` (one split)::

        que.txt  con.txt  wav.scp     CLSR format, one row per question
        articles.jsonl                one line per article, see module docstring
        audio/paragraphs/{a}_{p}.wav
        audio/articles/{a}.wav

    Audio paths in the manifests are relative to ``out_dir``, so the directory
    can be rsynced as a whole; pass ``audio_root=out_dir`` to ``ManifestTask``.
    ``articles`` restricts the build to those article indices — for smoke runs,
    and so that conditions built from different voice folders cover the same
    articles. An index with no audio raises rather than shrinking the set.
    Returns counts for the build log.
    """
    out_dir = Path(out_dir)
    paragraphs, missing = collect_paragraphs(json_path, wav_dir)
    if articles is not None:
        absent = set(articles) - {p.article for p in paragraphs}
        if absent:
            raise ValueError(f"{wav_dir} has no audio for articles {sorted(absent)}")
        paragraphs = [p for p in paragraphs if p.article in set(articles)]

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


@dataclass
class LongFormTask:
    """Text question -> window of an article recording, from a build directory.

    Documents are the windows ``segmenter`` cuts from every article recording.
    A window is relevant to a question when it overlaps the question's
    paragraph by at least ``min(min_overlap_s, paragraph duration)`` seconds —
    enough that a window grazing the paragraph's edge does not count, without
    making a short paragraph impossible to hit.

    The pool is the whole archive. For "find it in this recording", rank with
    ``lsr.eval.ranks_within_groups`` using ``query_group`` / ``document_group``.
    """

    build_dir: Path
    segmenter: object  # lsr.longaudio.Segmenter; untyped to keep tasks free of longaudio
    min_overlap_s: float = 5.0
    query_modality: str = "text"
    document_modality: str = "audio"
    name: str = field(init=False)

    def __post_init__(self) -> None:
        from lsr.longaudio import as_documents, overlap_s
        from lsr.tasks.base import Query
        from lsr.tasks.clsr import read_manifest

        self.build_dir = Path(self.build_dir)
        self.name = f"spoken_squad_longform_tq2ac_{self.build_dir.name}_{self.segmenter.name}"
        questions = read_manifest(self.build_dir / "que.txt")
        self.articles = [
            json.loads(line)
            for line in (self.build_dir / "articles.jsonl").read_text().splitlines()
        ]
        self.gold_span: dict[str, tuple[float, float]] = {}
        self.query_group: dict[str, str] = {}
        self.document_group: dict[str, str] = {}
        self._queries, self._documents, self._qrels = [], [], {}
        for art in self.articles:
            audio = str(self.build_dir / art["audio"])
            segments = self.segmenter.segment(audio, art["article_id"],
                                              duration_s=art["duration_s"])
            self._documents += as_documents(segments)
            for s in segments:
                self.document_group[s.id] = art["article_id"]
            for para in art["paragraphs"]:
                span = (para["start_s"], para["end_s"])
                need = min(self.min_overlap_s, span[1] - span[0])
                relevant = {s.id for s in segments
                            if overlap_s(s.start_s, s.end_s, *span) >= need}
                for qid in para["question_ids"]:
                    self._queries.append(Query(id=qid, text=questions[qid]))
                    self._qrels[qid] = relevant
                    self.gold_span[qid] = span
                    self.query_group[qid] = art["article_id"]

    def queries(self):
        yield from self._queries

    def documents(self):
        yield from self._documents

    def qrels(self) -> dict[str, set[str]]:
        return {k: set(v) for k, v in self._qrels.items()}
